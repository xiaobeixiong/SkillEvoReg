from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import openpyxl

from skillopt.envs.spreadsheetbench.agentic_ccv import (
    AgenticCCVRuntime,
    _compact_history,
    _is_retryable_codegen_failure,
    _scout_source_pool,
)
from skillopt.envs.spreadsheetbench import agentic_ccv
from skillopt.envs.spreadsheetbench.ccv import (
    _apply_mutation_operations,
    _cell_changes,
    _normalize_mutation_code,
    _restore_unchanged_formula_caches,
    _validate_attack_golden_consistency,
    _validate_declared_locations,
    _validate_mutation_code,
    run_spreadsheet_ccv,
)


class _Loader:
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def get_split_items(self, split: str) -> list[dict]:
        assert split == "valid_seen"
        return list(self.items)


class _Adapter:
    def __init__(self, data_root: str, items: list[dict]) -> None:
        self.data_root = data_root
        self.dataloader = _Loader(items)
        self._cfg = {"out_root": ""}
        self.calls: list[str] = []

    def rollout(self, items, skill_content: str, out_dir: str):
        self.calls.append(skill_content)
        is_attack = str(items[0]["id"]).startswith("ccv-")
        hard = 1 if skill_content == "old" or not is_attack else 0
        return [{
            "id": items[0]["id"],
            "hard": hard,
            "phase": "eval",
            "fail_reason": (
                ""
                if hard
                else "eval-mismatch: value@Sheet!A1: gt=1 pred=0"
            ),
            "cases": [{"stage": "eval", "ok": bool(hard)}],
        }]


def _write_workbooks(task_dir: Path) -> None:
    task_dir.mkdir(parents=True)
    for name in ("initial.xlsx", "golden.xlsx"):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet["A1"] = 1
        sheet["B1"] = "clean"
        workbook.save(task_dir / name)
        workbook.close()


def test_parallel_scout_pool_respects_exact_source_budget() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        items = []
        rollouts = []
        for index in range(3):
            source_id = f"source-{index}"
            _write_workbooks(root / "spreadsheet" / source_id)
            items.append({
                "id": source_id,
                "instruction": f"Keep A1 equal to {index}.",
                "spreadsheet_path": f"spreadsheet/{source_id}",
                "instruction_type": f"type-{index}",
                "answer_position": "A1",
            })
            rollouts.append({"id": source_id, "hard": 1})
        summaries, source_items, old_passes = _scout_source_pool(
            _Adapter(str(root), items),
            source_split="valid_seen",
            source_size=2,
            trajectory_context={"current_step_rollouts": rollouts},
        )
        assert [row["id"] for row in summaries] == ["source-0", "source-1"]
        assert set(source_items) == {"source-0", "source-1"}
        assert set(old_passes) == {"source-0", "source-1"}


def test_parallel_scout_history_is_limited_to_allowed_sources() -> None:
    adapter = _Adapter("", [])
    history = _compact_history(
        adapter,
        {
            "current_step_rollouts": [
                {"id": "allowed", "hard": 1},
                {"id": "context-only", "hard": 1},
            ],
            "selection_rollouts": [
                {"id": "allowed", "hard": 0},
                {"id": "another-context-only", "hard": 1},
            ],
        },
        history_items=10,
        allowed_source_ids={"allowed"},
    )
    assert history
    assert {row["id"] for row in history} == {"allowed"}


def test_parallel_scout_empty_briefs_are_valid_no_attack(monkeypatch) -> None:
    def fake_structured_agent(**kwargs):
        del kwargs
        return {"briefs": []}, {"status": "success"}, ""

    monkeypatch.setattr(
        agentic_ccv, "_run_harbor_json_agent", fake_structured_agent,
    )
    briefs, summary, error = asyncio.run(
        agentic_ccv._run_scout_once_harbor(
            old_skill="old",
            candidate_skill="candidate",
            history=[{"id": "source-1", "hard": 1}],
            source_summaries=[{"id": "source-1"}],
            verifier_contract={},
            max_briefs=1,
            out_dir="unused",
            model="gpt-5.4-mini",
            timeout=10,
            provider="codex_exec",
            effort="medium",
        )
    )
    assert briefs == []
    assert error == ""
    assert "controller_error" not in summary


def test_spreadsheet_ccv_rejects_first_old_pass_new_fail() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _write_workbooks(root / "spreadsheet" / "source-1")
        item = {
            "id": "source-1",
            "instruction": "Keep A1 equal to 1.",
            "spreadsheet_path": "spreadsheet/source-1",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }
        adapter = _Adapter(str(root), [item])
        responses = iter([
            (
                json.dumps({
                    "decision": "patterns",
                    "patterns": [{
                        "pattern_id": "p1",
                        "source_id": "source-1",
                        "delta_evidence": ["new"],
                        "incidental_assumption": "B1 is fixed",
                    }],
                }),
                {"prompt_tokens": 10},
            ),
            (
                """```json
{"decision":"attack","intervention":"change B1","modified_locations":["Sheet!B1"],"adversarial_instruction":"Keep A1 equal to 1.","preserved_invariants":["A1"],"expected_target_failure":"wrong A1","verifier_plan":"existing"}
```
```python
import openpyxl
for path in (ADV_INPUT_PATH, ADV_GOLDEN_PATH):
    wb = openpyxl.load_workbook(path)
    wb.active["B1"] = "changed"
    wb.save(path)
    wb.close()
```""",
                {"prompt_tokens": 20},
            ),
        ])

        def fake_llm(**kwargs):
            return next(responses)

        result = run_spreadsheet_ccv(
            adapter,
            old_skill="old",
            candidate_skill="new",
            step=1,
            seed=41,
            out_dir=str(root / "ccv"),
            source_split="valid_seen",
            source_size=1,
            max_patterns=1,
            attacks_per_pattern=1,
            max_attacks=1,
            history_items=0,
            attacker_model="",
            reasoning_effort="medium",
            builder_reasoning_effort="low",
            max_completion_tokens=1000,
            timeout=30,
            retries=1,
            llm_call=fake_llm,
        )

        assert result["valid_attacks"] == 1
        assert result["regressions"] == 1
        assert result["early_stopped"]
        assert adapter.calls == ["old", "new"]


def test_mutation_code_normalizes_path_environment_access() -> None:
    code = _normalize_mutation_code(
        'import os\nimport openpyxl\n'
        'wb = openpyxl.load_workbook(os.environ["ADV_INPUT_PATH"])\n'
        'wb.save(os.environ["ADV_INPUT_PATH"])\n'
    )
    assert "import os" not in code
    assert "os.environ" not in code
    _validate_mutation_code(code)


def test_declared_locations_must_match_actual_changes() -> None:
    records = [{
        "input_cell_changes": [{"location": "Sheet!B1"}],
    }]
    try:
        _validate_declared_locations(
            ["Sheet!C1"],
            records,
            instruction_changed=False,
        )
    except ValueError as exc:
        assert "undeclared" in str(exc)
    else:
        raise AssertionError("mismatched modified_locations must be rejected")


def test_empty_code_block_is_retryable_codegen_noise() -> None:
    assert _is_retryable_codegen_failure({
        "phase": "extract",
        "fail_reason": "empty-code-block",
    })
    assert not _is_retryable_codegen_failure({
        "phase": "eval",
        "fail_reason": "eval-mismatch: value@Sheet!A1: gt=1 pred=0",
    })


def test_cell_changes_handles_read_only_empty_cells_and_expanded_ranges() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        clean_path = root / "clean.xlsx"
        changed_path = root / "changed.xlsx"
        clean = openpyxl.Workbook()
        clean.active["A1"] = "header"
        clean.save(clean_path)
        clean.close()
        changed = openpyxl.load_workbook(clean_path)
        changed.active["B10"] = "new"
        changed.save(changed_path)
        changed.close()
        changes = _cell_changes(str(clean_path), str(changed_path))
        assert [row["location"] for row in changes] == ["Sheet!B10"]


def test_structured_mutation_changes_only_copy_and_preserves_source() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "source.xlsx"
        attacked = root / "attacked.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet["A1"] = "unchanged"
        sheet["B1"] = 1
        sheet["C1"] = ""
        workbook.save(source)
        workbook.close()
        shutil.copy2(source, attacked)

        source_hash_before = source.read_bytes()
        _apply_mutation_operations(
            input_path=str(attacked),
            golden_path=str(attacked),
            operations=[{
                "op": "set_cell",
                "workbook": "input",
                "location": "Sheet!B1",
                "value": True,
                "value_type": "boolean",
            }],
        )

        assert source.read_bytes() == source_hash_before
        changes = _cell_changes(str(source), str(attacked))
        assert [row["location"] for row in changes] == ["Sheet!B1"]
        changed = openpyxl.load_workbook(attacked, data_only=False)
        assert changed.active["A1"].value == "unchanged"
        assert changed.active["B1"].value is True
        changed.close()


def test_structured_mutation_rejects_unknown_or_duplicate_targets() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "book.xlsx"
        workbook = openpyxl.Workbook()
        workbook.active["A1"] = 1
        workbook.save(path)
        workbook.close()

        for operations, expected in [
            ([{"op": "delete_file", "location": "Sheet!A1"}], "unsupported"),
            ([
                {"op": "clear_cell", "location": "Sheet!A1"},
                {"op": "set_cell", "location": "Sheet!A1", "value": 2},
            ], "multiple operations"),
            ([{"op": "set_cell", "location": "Missing!A1", "value": 2}], "unknown worksheet"),
        ]:
            copied = Path(temp) / f"copy-{len(expected)}.xlsx"
            shutil.copy2(path, copied)
            try:
                _apply_mutation_operations(
                    input_path=str(copied),
                    golden_path=str(copied),
                    operations=operations,
                )
            except ValueError as exc:
                assert expected in str(exc)
            else:
                raise AssertionError("unsafe structured operation must be rejected")


def test_agentic_source_contract_and_preflight_are_explicit_and_free() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _write_workbooks(root / "spreadsheet" / "source-1")
        item = {
            "id": "source-1",
            "instruction": "Keep A1 equal to 1.",
            "spreadsheet_path": "spreadsheet/source-1",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }
        runtime = AgenticCCVRuntime(
            _Adapter(str(root), [item]),
            old_skill="old",
            candidate_skill="new",
            out_dir=str(root / "agentic"),
            source_split="valid_seen",
            source_size=1,
            max_attacks=1,
            history_items=0,
            trajectory_context={},
        )
        inspected = runtime.inspect_source("source-1")
        contract = inspected["mutation_contract"]
        assert contract["constraints"]["allow_new_cells"] is False
        assert "Sheet!B1" in contract["input"]["editable_locations"]
        assert "Sheet!C15" not in contract["input"]["editable_locations"]

        invalid_before = runtime.budget()["invalid_proposals"]
        try:
            runtime.validate_operations(
                "source-1",
                [{"op": "set_formula", "workbook": "input",
                  "location": "Sheet!C15", "value": '=""'}],
                ["Sheet!C15"],
            )
        except ValueError as exc:
            assert "not editable" in str(exc)
        else:
            raise AssertionError("preflight must reject a non-editable location")
        assert runtime.budget()["invalid_proposals"] == invalid_before

        valid = runtime.validate_operations(
            "source-1",
            [{"op": "set_cell", "workbook": "input",
              "location": "Sheet!B1", "value": "changed"}],
            ["Sheet!B1"],
        )
        assert valid["status"] == "valid"
        assert valid["consumes_invalid_proposal_budget"] is False
        assert runtime.budget()["operation_preflight_calls"] == 2
        runtime.finish_pairs()


def test_restore_unchanged_formula_caches_preserves_changed_oracle_cell() -> None:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        clean_path = root / "clean.xlsx"
        mutated_path = root / "mutated.xlsx"
        workbook = openpyxl.Workbook()
        workbook.active["A1"] = '="old"'
        workbook.active["B1"] = '="target"'
        workbook.save(clean_path)
        workbook.close()

        patched_path = root / "patched.xlsx"
        with zipfile.ZipFile(clean_path, "r") as source, zipfile.ZipFile(
            patched_path, "w"
        ) as destination:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == "xl/worksheets/sheet1.xml":
                    tree = ET.fromstring(data)
                    values = {"A1": "old", "B1": "target"}
                    for cell in tree.iter(f"{{{namespace}}}c"):
                        coordinate = cell.get("r")
                        if coordinate not in values:
                            continue
                        cell.set("t", "str")
                        value = cell.find(f"{{{namespace}}}v")
                        assert value is not None
                        value.text = values[coordinate]
                    data = ET.tostring(
                        tree, encoding="utf-8", xml_declaration=True
                    )
                destination.writestr(info, data)
        os.replace(patched_path, clean_path)
        shutil.copy2(clean_path, mutated_path)

        mutated = openpyxl.load_workbook(mutated_path)
        mutated.active["B1"] = "changed"
        mutated.save(mutated_path)
        mutated.close()

        restored = _restore_unchanged_formula_caches(
            str(clean_path), str(mutated_path)
        )
        result = openpyxl.load_workbook(mutated_path, data_only=True)
        assert result.active["A1"].value == "old"
        assert result.active["B1"].value == "changed"
        result.close()
        assert restored == 1


def test_agentic_ccv_runtime_rejects_first_old_pass_new_fail() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _write_workbooks(root / "spreadsheet" / "source-1")
        item = {
            "id": "source-1",
            "instruction": "Keep A1 equal to 1.",
            "spreadsheet_path": "spreadsheet/source-1",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }
        adapter = _Adapter(str(root), [item])
        runtime = AgenticCCVRuntime(
            adapter,
            old_skill="old",
            candidate_skill="new",
            out_dir=str(root / "agentic"),
            source_split="valid_seen",
            source_size=1,
            max_attacks=1,
            history_items=0,
            trajectory_context={},
        )
        inspected = runtime.inspect_source("source-1")
        assert inspected["id"] == "source-1"
        qualified = runtime.qualify_source("source-1")
        assert qualified["status"] == "qualified"
        attack = runtime.propose_attack(
            json.dumps({
                "source_task_id": "source-1",
                "hypothesis": "The candidate assumes B1 is fixed.",
                "skill_delta_evidence": ["new"],
                "trajectory_evidence": [],
                "true_invariant": "A1 remains equal to 1.",
                "intervention": "Change irrelevant metadata B1.",
                "modified_locations": ["Sheet!B1"],
                "adversarial_instruction": "Keep A1 equal to 1.",
                "preserved_invariants": ["A1"],
                "expected_candidate_failure": "S_candidate writes the wrong A1.",
                "expected_failure_locations": ["Sheet!A1"],
                "why_old_should_still_pass": (
                    "S_old ignores the irrelevant B1 metadata."
                ),
                "difficulty_control": "Only irrelevant metadata changes.",
                "causal_trigger": (
                    "Changing B1 activates the candidate's new fixed-B1 "
                    "assumption, which causes its wrong A1 output."
                ),
                "why_clean_source_does_not_trigger": (
                    "The clean B1 value satisfies the candidate's fixed-B1 "
                    "assumption."
                ),
            }),
            """\
import openpyxl
for path in (ADV_INPUT_PATH, ADV_GOLDEN_PATH):
    wb = openpyxl.load_workbook(path)
    wb.active["B1"] = "changed"
    wb.save(path)
    wb.close()
""",
        )
        paired = runtime.run_pair(attack["attack_id"])
        assert paired["old_pass_candidate_fail"]
        result = runtime.result()
        assert result["valid_attacks"] == 1
        assert result["regressions"] == 1
        assert result["early_stopped"]
        assert adapter.calls == ["old", "old", "new", "new"]


def test_agentic_ccv_retries_transient_malformed_old_codegen() -> None:
    class FlakyOldAdapter(_Adapter):
        def __init__(self, data_root: str, items: list[dict]) -> None:
            super().__init__(data_root, items)
            self.failed_attacked_old_once = False

        def rollout(self, items, skill_content: str, out_dir: str):
            is_attack = str(items[0]["id"]).startswith("ccv-")
            if (
                is_attack
                and skill_content == "old"
                and not self.failed_attacked_old_once
            ):
                self.failed_attacked_old_once = True
                self.calls.append(skill_content)
                return [{
                    "id": items[0]["id"],
                    "hard": 0,
                    "phase": "exec",
                    "fail_reason": (
                        "exec-error: IndentationError: expected an "
                        "indented block"
                    ),
                    "cases": [{"stage": "exec", "ok": False}],
                }]
            return super().rollout(items, skill_content, out_dir)

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _write_workbooks(root / "spreadsheet" / "source-1")
        item = {
            "id": "source-1",
            "instruction": "Keep A1 equal to 1.",
            "spreadsheet_path": "spreadsheet/source-1",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }
        adapter = FlakyOldAdapter(str(root), [item])
        runtime = AgenticCCVRuntime(
            adapter,
            old_skill="old",
            candidate_skill="new",
            out_dir=str(root / "agentic"),
            source_split="valid_seen",
            source_size=1,
            max_attacks=1,
            history_items=0,
            trajectory_context={},
        )
        runtime.inspect_source("source-1")
        assert runtime.qualify_source("source-1")["status"] == "qualified"
        attack = runtime.propose_attack(
            json.dumps({
                "source_task_id": "source-1",
                "hypothesis": "The candidate assumes B1 is fixed.",
                "skill_delta_evidence": ["new"],
                "trajectory_evidence": [],
                "true_invariant": "A1 remains equal to 1.",
                "intervention": "Change irrelevant metadata B1.",
                "modified_locations": ["Sheet!B1"],
                "adversarial_instruction": "Keep A1 equal to 1.",
                "preserved_invariants": ["A1"],
                "expected_candidate_failure": (
                    "S_candidate writes the wrong A1."
                ),
                "expected_failure_locations": ["Sheet!A1"],
                "why_old_should_still_pass": "S_old ignores B1.",
                "difficulty_control": "Only irrelevant metadata changes.",
                "causal_trigger": (
                    "Changing B1 activates the candidate's new fixed-B1 "
                    "assumption, which causes its wrong A1 output."
                ),
                "why_clean_source_does_not_trigger": (
                    "The clean B1 value satisfies the candidate's fixed-B1 "
                    "assumption."
                ),
            }),
            """\
import openpyxl
wb = openpyxl.load_workbook(ADV_INPUT_PATH)
wb.active["B1"] = "changed"
wb.save(ADV_INPUT_PATH)
wb.close()
""",
        )
        paired = runtime.run_pair(attack["attack_id"])
        assert paired["status"] == "update_regression"
        result = runtime.result()
        assert result["regressions"] == 1
        assert len(result["attack_records"][0]["old_attempts"]) == 2
        assert adapter.calls == ["old", "old", "old", "new", "new"]


def test_agentic_ccv_does_not_confirm_failure_outside_declared_region() -> None:
    class WrongLocationAdapter(_Adapter):
        def rollout(self, items, skill_content: str, out_dir: str):
            result = super().rollout(items, skill_content, out_dir)
            if not result[0]["hard"]:
                result[0]["fail_reason"] = (
                    "eval-mismatch: value@Sheet!B1: gt='clean' pred='bad'"
                )
            return result

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _write_workbooks(root / "spreadsheet" / "source-1")
        item = {
            "id": "source-1",
            "instruction": "Keep A1 equal to 1.",
            "spreadsheet_path": "spreadsheet/source-1",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }
        runtime = AgenticCCVRuntime(
            WrongLocationAdapter(str(root), [item]),
            old_skill="old",
            candidate_skill="new",
            out_dir=str(root / "agentic"),
            source_split="valid_seen",
            source_size=1,
            max_attacks=1,
            history_items=0,
            trajectory_context={},
        )
        runtime.inspect_source("source-1")
        assert runtime.qualify_source("source-1")["status"] == "qualified"
        attack = runtime.propose_attack(
            json.dumps({
                "source_task_id": "source-1",
                "hypothesis": "The candidate mishandles A1.",
                "skill_delta_evidence": ["new"],
                "trajectory_evidence": [],
                "true_invariant": "A1 remains equal to 1.",
                "intervention": "Change irrelevant metadata B1.",
                "modified_locations": ["Sheet!B1"],
                "adversarial_instruction": "Keep A1 equal to 1.",
                "preserved_invariants": ["A1"],
                "expected_candidate_failure": (
                    "S_candidate writes the wrong A1."
                ),
                "expected_failure_locations": ["Sheet!A1"],
                "why_old_should_still_pass": "S_old ignores B1.",
                "difficulty_control": "Only irrelevant metadata changes.",
                "causal_trigger": (
                    "Changing B1 activates the candidate's new fixed-B1 "
                    "assumption, which causes its wrong A1 output."
                ),
                "why_clean_source_does_not_trigger": (
                    "The clean B1 value satisfies the candidate's fixed-B1 "
                    "assumption."
                ),
            }),
            """\
import openpyxl
wb = openpyxl.load_workbook(ADV_INPUT_PATH)
wb.active["B1"] = "changed"
wb.save(ADV_INPUT_PATH)
wb.close()
""",
        )
        paired = runtime.run_pair(attack["attack_id"])
        assert paired["status"] == "non_causal_candidate_failure"
        assert not paired["old_pass_candidate_fail"]
        assert not paired["causal_validation"]["failure_location_match"]
        result = runtime.result()
        assert result["regressions"] == 0
        assert result["inconclusive_failures"] == 1


def test_agentic_ccv_runtime_enforces_source_and_pair_budgets() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        items = []
        for source_id in ("source-1", "source-2"):
            _write_workbooks(root / "spreadsheet" / source_id)
            items.append({
                "id": source_id,
                "instruction": "Keep A1 equal to 1.",
                "spreadsheet_path": f"spreadsheet/{source_id}",
                "instruction_type": "Cell-Level Manipulation",
                "answer_position": "A1",
            })
        runtime = AgenticCCVRuntime(
            _Adapter(str(root), items),
            old_skill="old",
            candidate_skill="new",
            out_dir=str(root / "agentic"),
            source_split="valid_seen",
            source_size=1,
            max_attacks=1,
            history_items=0,
            trajectory_context={},
        )
        runtime.inspect_source("source-1")
        try:
            runtime.inspect_source("source-2")
        except ValueError as exc:
            assert "inspection budget exhausted" in str(exc)
        else:
            raise AssertionError("second source inspection must be rejected")


def test_agentic_ccv_rejects_erased_golden_answer() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _write_workbooks(root / "spreadsheet" / "source-1")
        item = {
            "id": "source-1",
            "instruction": "Keep A1 equal to 1.",
            "spreadsheet_path": "spreadsheet/source-1",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }
        runtime = AgenticCCVRuntime(
            _Adapter(str(root), [item]),
            old_skill="old",
            candidate_skill="new",
            out_dir=str(root / "agentic"),
            source_split="valid_seen",
            source_size=1,
            max_attacks=1,
            history_items=0,
            trajectory_context={},
        )
        runtime.inspect_source("source-1")
        assert runtime.qualify_source("source-1")["status"] == "qualified"
        payload = {
            "source_task_id": "source-1",
            "hypothesis": "The candidate assumes B1 is fixed.",
            "skill_delta_evidence": ["new"],
            "trajectory_evidence": [],
            "true_invariant": "A1 remains equal to 1.",
            "intervention": "Change irrelevant metadata B1.",
            "modified_locations": ["Sheet!B1", "Sheet!A1"],
            "adversarial_instruction": "Keep A1 equal to 1.",
            "preserved_invariants": ["A1"],
            "expected_candidate_failure": "S_candidate writes the wrong A1.",
            "expected_failure_locations": ["Sheet!A1"],
            "why_old_should_still_pass": "S_old ignores B1.",
            "difficulty_control": "Only irrelevant metadata changes.",
            "causal_trigger": (
                "Changing B1 activates the candidate's new fixed-B1 "
                "assumption, which causes its wrong A1 output."
            ),
            "why_clean_source_does_not_trigger": (
                "The clean B1 value satisfies the candidate's fixed-B1 "
                "assumption."
            ),
        }
        try:
            runtime.propose_attack(
                json.dumps(payload),
                """\
import openpyxl
wb = openpyxl.load_workbook(ADV_INPUT_PATH)
wb.active["B1"] = "changed"
wb.save(ADV_INPUT_PATH)
wb.close()
wbg = openpyxl.load_workbook(ADV_GOLDEN_PATH)
wbg.active["A1"] = None
wbg.save(ADV_GOLDEN_PATH)
wbg.close()
""",
            )
        except ValueError as exc:
            assert "erased populated verifier answer cells" in str(exc)
        else:
            raise AssertionError("erased golden answers must be rejected")


def test_ccv_rejects_stale_golden_when_input_changes_answer_dependency() -> None:
    with tempfile.TemporaryDirectory() as temp:
        golden_path = Path(temp) / "golden.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Sheet"
        sheet["A1"] = 1
        sheet["A2"] = 2
        sheet["C1"] = "=SUM(A1:A2)"
        workbook.save(golden_path)
        workbook.close()
        records = [{
            "golden_changed": False,
            "clean_golden": str(golden_path),
            "input_cell_changes": [{"location": "Sheet!A2"}],
        }]
        try:
            _validate_attack_golden_consistency(records, "Sheet!C1")
        except ValueError as exc:
            assert "answer-formula dependency" in str(exc)
            assert "Sheet!A2" in str(exc)
        else:
            raise AssertionError("stale golden dependency attack must be rejected")


def test_ccv_allows_unchanged_golden_for_irrelevant_input_change() -> None:
    with tempfile.TemporaryDirectory() as temp:
        golden_path = Path(temp) / "golden.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Sheet"
        sheet["A1"] = 1
        sheet["C1"] = "=SUM(A1:A1)"
        sheet["D9"] = "metadata"
        workbook.save(golden_path)
        workbook.close()
        _validate_attack_golden_consistency([{
            "golden_changed": False,
            "clean_golden": str(golden_path),
            "input_cell_changes": [{"location": "Sheet!D9"}],
        }], "Sheet!C1")


