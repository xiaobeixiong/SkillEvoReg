"""Delta-conditioned CCV provider for SpreadsheetBench.

The provider runs only at a skill commit boundary after the ordinary
validation and complexity pre-gates have passed. An attacker LLM mines open
ended, update-related failure hypotheses from the whole skill delta and recent
execution history, then writes bounded workbook mutation programs. The normal
SpreadsheetBench rollout and verifier are reused for the paired old/new test.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
import glob
import hashlib
from itertools import zip_longest
import json
import os
import posixpath
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
import xml.etree.ElementTree as ET
import zipfile

import openpyxl
from openpyxl.formula import Tokenizer
from openpyxl.utils.cell import range_boundaries

import skillopt.envs.spreadsheetbench.evaluator as spreadsheet_evaluator
from skillopt.envs.spreadsheetbench.codegen_agent import _preview_workbook
from skillopt.envs.spreadsheetbench.rollout import _find_test_cases
from skillopt.model import chat_optimizer
from skillopt.model.azure_openai import chat_with_deployment
from skillopt.utils import extract_json


PATTERN_SYSTEM = """\
You are CCV, an auditor for overfitting in an evolving spreadsheet agent skill.

Analyze the complete behavioral change from S_old to S_new. Do not restrict the
analysis to a single atom. A hypothesis may involve additions, deletions,
rewrites, priority changes, interactions among multiple new rules, interactions
between old and new rules, or an overall execution-policy change visible in the
trajectory history.

Find only assumptions introduced or strengthened by this update that are not
logically required by the task family. Every hypothesis must be falsifiable by
a minimal, valid workbook change that preserves the latent task objective and
the existing deterministic verifier contract.

Do not use a fixed attack taxonomy. Do not propose prompt injection, malformed
files, ambiguity, impossible states, hidden requirements, verifier exploits, or
mere increases in task difficulty. Return no_attack rather than inventing a
weakness.

Return JSON only:
{
  "decision": "patterns" | "no_attack",
  "patterns": [
    {
      "pattern_id": "p1",
      "source_id": "one supplied source task id",
      "delta_evidence": ["exact changed skill span or interaction"],
      "trajectory_evidence": "specific historical behavior or none",
      "true_invariant": "what a correct solver must preserve",
      "incidental_assumption": "falsifiable update-induced assumption",
      "minimal_intervention": "smallest workbook/input change",
      "predicted_failure": "specific observable output failure",
      "difficulty_control": "why the task remains equally solvable",
      "confidence": 0.0
    }
  ]
}
"""


ATTACK_SYSTEM = """\
You generate one minimal SpreadsheetBench counterexample for a specific
update-induced skill weakness.

The task directory has already been copied. Your Python program receives only:

  ADV_INPUT_PATH
  ADV_GOLDEN_PATH

Open and mutate those two workbooks with openpyxl, then save them back to the
same paths. Apply corresponding structural changes to the golden workbook when
required, but never copy answer values from the golden workbook into the input.

The answer position and instruction type must remain unchanged so the existing
SpreadsheetBench verifier can judge the result. You may minimally rewrite the
instruction only when needed to describe the same latent objective after the
workbook intervention.

Constraints:
- Use only openpyxl and the Python standard library modules copy, re, math, and
  datetime.
- Preserve unrelated workbook content.
- No prompt injection, malformed files, macros, external links, randomness,
  network, filesystem discovery, evaluator exploits, or hidden requirements.
- Do not read or write any path except ADV_INPUT_PATH and ADV_GOLDEN_PATH.
- Do not narrate, promise future inspection, or ask for files. Use the supplied
  workbook preview and return the final JSON/Python blocks immediately.
- If the proposed intervention cannot satisfy all constraints, return
  no_attack.

Return exactly one JSON block followed, only for an attack, by one Python block:
```json
{
  "decision": "attack" | "no_attack",
  "intervention": "precise description",
  "modified_locations": ["exact Sheet!Cell or explicit structural component"],
  "adversarial_instruction": "complete instruction, or the unchanged original",
  "preserved_invariants": ["invariant"],
  "expected_target_failure": "observable failure",
  "verifier_plan": "existing SpreadsheetBench verifier at the unchanged range"
}
```
```python
# mutation program
```

Choose exact cells from the supplied workbook manifest when the intervention is
cell-level. Never scan for and mutate the first arbitrary value of a Python
type. A type-change attack must identify a cell that actually participates in
the task relation described by the instruction.
"""

SOURCE_ALIGNMENT_SYSTEM = """\
You align already-discovered CCV hypotheses to the allowed attack-source
tasks. Historical trajectory task IDs are evidence only and may never be used
as source_id. For each hypothesis, choose an allowed source only when its
instruction and workbook visibly support the proposed minimal intervention.
Otherwise drop that hypothesis. Do not invent a new weakness.

Return JSON only:
{
  "alignments": [
    {"pattern_id": "p1", "source_id": "one exact allowed id"}
  ]
}
"""

SOURCE_SELECTION_SYSTEM = """\
Select source tasks for delta-conditioned adversarial probing. You receive one
semantic skill update, recent execution history, and the complete allowed
source pool. Choose tasks whose existing workbook structure and objective are
most suitable for a minimal counterfactual that can falsify an assumption
introduced or strengthened by the update.

Selection is open-ended and may consider interactions or global behavior
changes, but source IDs must come exactly from the allowed pool. Do not claim
that an attack already exists and do not modify tasks in this step.

Return JSON only:
{"source_ids": ["exact allowed id"]}
"""


@dataclass
class GeneratedAttack:
    item: dict
    record: dict


def _merge_usage(total: dict[str, int], usage: dict | None) -> None:
    for key, value in (usage or {}).items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + int(value)


def _call_attacker(
    *,
    system: str,
    user: str,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    retries: int,
    timeout: int,
) -> tuple[str, dict]:
    kwargs = {
        "system": system,
        "user": user,
        "max_completion_tokens": max_completion_tokens,
        "retries": retries,
        "stage": "ccv_attacker",
        "reasoning_effort": reasoning_effort or None,
        "timeout": timeout,
    }
    if model:
        return chat_with_deployment(deployment=model, **kwargs)
    return chat_optimizer(**kwargs)


def _semantic_diff(old_skill: str, new_skill: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            old_skill.splitlines(),
            new_skill.splitlines(),
            fromfile="S_old",
            tofile="S_new",
            lineterm="",
            n=3,
        )
    )


def _compact_result(row: dict, *, max_text: int = 2400) -> dict:
    def clipped(value: Any, limit: int = max_text) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        half = max(1, limit // 2)
        return text[:half] + "\n...[truncated]...\n" + text[-half:]

    return {
        "id": str(row.get("id", "")),
        "hard": row.get("hard"),
        "phase": row.get("phase", ""),
        "fail_reason": clipped(row.get("fail_reason", ""), 800),
        "task_description": clipped(row.get("task_description", ""), 1200),
        "spreadsheet_preview": clipped(row.get("spreadsheet_preview", "")),
        "cases": row.get("cases", [])[:4] if isinstance(row.get("cases"), list) else [],
    }


def _load_recent_trajectories(out_root: str, limit: int) -> list[dict]:
    if not out_root or limit <= 0:
        return []
    paths = sorted(
        glob.glob(os.path.join(out_root, "steps", "step_*", "**", "results.jsonl"), recursive=True),
        reverse=True,
    )
    records: list[dict] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        for row in reversed(rows):
            compact = _compact_result(row)
            prediction_dir = os.path.join(
                os.path.dirname(path), "predictions", str(row.get("id", "")),
            )
            code_path = os.path.join(prediction_dir, "code.py")
            if os.path.exists(code_path):
                try:
                    with open(code_path, encoding="utf-8") as handle:
                        compact["generated_code"] = handle.read()[-2400:]
                except OSError:
                    pass
            records.append(compact)
            if len(records) >= limit:
                return records
    return records


def _answer_position(item: dict) -> str:
    position = str(item.get("answer_position", "") or "")
    sheet = str(item.get("answer_sheet", "") or "")
    if position and sheet and "!" not in position:
        return f"{sheet}!{position}"
    return position


def _source_manifest(adapter, item: dict) -> dict | None:
    spreadsheet_path = str(
        item.get("spreadsheet_path", f"spreadsheet/{item.get('id')}")
    )
    task_dir = (
        spreadsheet_path
        if os.path.isabs(spreadsheet_path)
        else os.path.join(adapter.data_root, spreadsheet_path)
    )
    cases = _find_test_cases(task_dir)
    if not cases:
        return None
    try:
        preview = _preview_workbook(cases[0][1])
    except Exception as exc:  # noqa: BLE001
        preview = f"(preview failed: {type(exc).__name__}: {exc})"
    workbook_manifest = _workbook_manifest(cases[0][1])
    return {
        "id": str(item.get("id")),
        "instruction": str(item.get("instruction", "")),
        "instruction_type": str(item.get("instruction_type", "")),
        "answer_position": _answer_position(item),
        "task_dir": task_dir,
        "case_count": len(cases),
        "preview": preview[:7000],
        "workbook_manifest": workbook_manifest,
        "mutation_contract": _build_mutation_contract(cases),
    }


def _workbook_manifest(path: str, *, max_cells: int = 600) -> dict:
    """Return bounded, coordinate- and type-aware input metadata for attacks."""
    workbook = openpyxl.load_workbook(path, data_only=False, read_only=True)
    try:
        sheets = []
        remaining = max_cells
        for sheet in workbook.worksheets:
            cells = []
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value is None:
                        continue
                    if remaining <= 0:
                        break
                    value = repr(cell.value)
                    cells.append({
                        "cell": cell.coordinate,
                        "type": type(cell.value).__name__,
                        "value": value[:300],
                    })
                    remaining -= 1
                if remaining <= 0:
                    break
            sheets.append({
                "sheet": sheet.title,
                "max_row": sheet.max_row,
                "max_column": sheet.max_column,
                "nonempty_cells": cells,
            })
            if remaining <= 0:
                break
        return {
            "sheets": sheets,
            "truncated": remaining <= 0,
            "max_cells": max_cells,
        }
    finally:
        workbook.close()


def _cell_changes(clean_path: str, changed_path: str, *, limit: int = 1000) -> list[dict]:
    def stable_value(value: Any) -> Any:
        if isinstance(value, openpyxl.worksheet.formula.ArrayFormula):
            return ("ArrayFormula", value.ref, value.text)
        if isinstance(value, openpyxl.worksheet.formula.DataTableFormula):
            return ("DataTableFormula", tuple(sorted(vars(value).items())))
        return (type(value).__qualname__, value)

    clean = openpyxl.load_workbook(clean_path, data_only=False, read_only=True)
    changed = openpyxl.load_workbook(changed_path, data_only=False, read_only=True)
    differences: list[dict] = []
    try:
        for sheet_name in sorted(set(clean.sheetnames) | set(changed.sheetnames)):
            if sheet_name not in clean.sheetnames or sheet_name not in changed.sheetnames:
                differences.append({"location": f"sheet:{sheet_name}", "kind": "sheet_set"})
                continue
            left = clean[sheet_name]
            right = changed[sheet_name]
            max_row = max(left.max_row, right.max_row)
            max_column = max(left.max_column, right.max_column)
            left_rows = left.iter_rows(
                min_row=1,
                max_row=max_row,
                min_col=1,
                max_col=max_column,
                values_only=True,
            )
            right_rows = right.iter_rows(
                min_row=1,
                max_row=max_row,
                min_col=1,
                max_col=max_column,
                values_only=True,
            )
            for row, (left_row, right_row) in enumerate(
                zip_longest(left_rows, right_rows, fillvalue=()), start=1
            ):
                for column in range(1, max_column + 1):
                    before = (
                        left_row[column - 1]
                        if column <= len(left_row)
                        else None
                    )
                    after = (
                        right_row[column - 1]
                        if column <= len(right_row)
                        else None
                    )
                    before_key = stable_value(before)
                    after_key = stable_value(after)
                    if before_key == after_key:
                        continue
                    coordinate = (
                        f"{openpyxl.utils.get_column_letter(column)}{row}"
                    )
                    differences.append({
                        "location": f"{sheet_name}!{coordinate}",
                        "before": repr(before_key)[:300],
                        "before_type": type(before).__name__,
                        "after": repr(after_key)[:300],
                        "after_type": type(after).__name__,
                    })
                    if len(differences) >= limit:
                        return differences
        return differences
    finally:
        clean.close()
        changed.close()


def _validate_declared_locations(
    declared_locations: list,
    case_records: list[dict],
    *,
    instruction_changed: bool,
) -> None:
    declared = {
        str(location).strip().casefold()
        for location in declared_locations
        if str(location).strip()
    }
    actual_cells = {
        str(change.get("location", "")).strip().casefold()
        for case in case_records
        for change in [
            *case.get("input_cell_changes", []),
            *case.get("golden_cell_changes", []),
        ]
        if str(change.get("location", "")).strip()
    }
    undeclared = sorted(actual_cells - declared)
    if undeclared:
        raise ValueError(
            "attack changed undeclared workbook locations: "
            + ", ".join(undeclared[:20])
        )
    if actual_cells and not (actual_cells & declared):
        raise ValueError("declared locations do not match any workbook change")
    if instruction_changed and not any(
        location == "instruction" or location.startswith("instruction:")
        for location in declared
    ):
        raise ValueError("changed instruction must be declared as instruction")
    if not actual_cells and not instruction_changed:
        raise ValueError("attack has no observable declared input change")


def _location_inside_answer_position(
    location: str,
    answer_position: str,
) -> bool:
    """Return whether one exact changed cell lies in a verifier answer range."""
    if "!" not in str(location):
        return False
    changed_sheet, changed_cell = str(location).rsplit("!", 1)
    changed_sheet = changed_sheet.strip().strip("'\"").casefold()
    try:
        changed_col, changed_row, _, _ = range_boundaries(changed_cell)
    except ValueError:
        return False
    for reference in str(answer_position or "").split(","):
        if "!" in reference:
            answer_sheet, answer_cells = reference.rsplit("!", 1)
            answer_sheet = answer_sheet.strip().strip("'\"").casefold()
        else:
            answer_sheet = changed_sheet
            answer_cells = reference
        if answer_sheet != changed_sheet:
            continue
        try:
            min_col, min_row, max_col, max_row = range_boundaries(
                answer_cells.strip().strip("'\"")
            )
        except ValueError:
            continue
        if (
            min_col <= changed_col <= max_col
            and min_row <= changed_row <= max_row
        ):
            return True
    return False


def _validate_nonblank_golden_answers(
    case_records: list[dict],
    answer_position: str,
) -> None:
    """Reject an oracle that erases a previously populated answer cell."""
    degraded = [
        str(change.get("location", ""))
        for case in case_records
        for change in case.get("golden_cell_changes", [])
        if str(change.get("after_type", "")) == "NoneType"
        and _location_inside_answer_position(
            str(change.get("location", "")),
            answer_position,
        )
    ]
    if degraded:
        raise ValueError(
            "attack erased populated verifier answer cells in the golden "
            "workbook: "
            + ", ".join(degraded[:20])
        )


def _split_cell_location(location: str) -> tuple[str, int, int] | None:
    if "!" not in str(location):
        return None
    sheet_name, coordinate = str(location).rsplit("!", 1)
    try:
        column, row, _, _ = range_boundaries(coordinate.replace("$", ""))
    except ValueError:
        return None
    return sheet_name.strip().strip("'\"").casefold(), row, column


def _answer_cells(
    workbook: Any, answer_position: str,
) -> list[tuple[str, int, int]]:
    cells: list[tuple[str, int, int]] = []
    default_sheet = workbook.sheetnames[0] if workbook.sheetnames else ""
    for reference in str(answer_position or "").split(","):
        reference = reference.strip()
        if not reference:
            continue
        if "!" in reference:
            sheet_name, cell_range = reference.rsplit("!", 1)
            sheet_name = sheet_name.strip().strip("'\"")
        else:
            sheet_name, cell_range = default_sheet, reference
        if sheet_name not in workbook.sheetnames:
            continue
        try:
            min_col, min_row, max_col, max_row = range_boundaries(
                cell_range.replace("$", "").strip("'\"")
            )
        except ValueError:
            continue
        for row in range(min_row, max_row + 1):
            for column in range(min_col, max_col + 1):
                cells.append((sheet_name, row, column))
    return cells


def _formula_depends_on_cell(
    workbook: Any,
    *,
    answer_position: str,
    changed_location: str,
    max_formula_cells: int = 5000,
) -> bool:
    """Conservatively trace answer-formula references to one changed input."""
    changed = _split_cell_location(changed_location)
    if changed is None:
        return False
    changed_sheet, changed_row, changed_col = changed
    queue = _answer_cells(workbook, answer_position)
    visited: set[tuple[str, int, int]] = set()
    inspected = 0
    while queue and inspected < max_formula_cells:
        sheet_name, row, column = queue.pop()
        key = (sheet_name.casefold(), row, column)
        if key in visited:
            continue
        visited.add(key)
        inspected += 1
        value = workbook[sheet_name].cell(row=row, column=column).value
        if not isinstance(value, str) or not value.startswith("="):
            continue
        try:
            tokens = Tokenizer(value).items
        except Exception:  # noqa: BLE001
            continue
        for token in tokens:
            if token.type != "OPERAND" or token.subtype != "RANGE":
                continue
            raw = str(token.value).strip()
            # Ignore named/structured/external references that cannot be
            # resolved safely by openpyxl's A1 range parser.
            if "[" in raw or "]" in raw:
                continue
            if "!" in raw:
                ref_sheet, ref_cells = raw.rsplit("!", 1)
                ref_sheet = ref_sheet.strip().strip("'\"")
            else:
                ref_sheet, ref_cells = sheet_name, raw
            if ref_sheet not in workbook.sheetnames:
                continue
            try:
                min_col, min_row, max_col, max_row = range_boundaries(
                    ref_cells.replace("$", "")
                )
            except ValueError:
                continue
            if (
                ref_sheet.casefold() == changed_sheet
                and min_row <= changed_row <= max_row
                and min_col <= changed_col <= max_col
            ):
                return True
            # Follow formulas referenced by the answer. Avoid expanding huge
            # raw-data ranges; direct membership above already catches a
            # changed input inside such a range.
            area = (max_row - min_row + 1) * (max_col - min_col + 1)
            if area > max_formula_cells:
                continue
            for ref_row in range(min_row, max_row + 1):
                for ref_col in range(min_col, max_col + 1):
                    ref_value = workbook[ref_sheet].cell(
                        row=ref_row, column=ref_col,
                    ).value
                    if isinstance(ref_value, str) and ref_value.startswith("="):
                        queue.append((ref_sheet, ref_row, ref_col))
    return False


def _validate_attack_golden_consistency(
    case_records: list[dict], answer_position: str,
) -> None:
    """Reject stale-golden attacks that mutate answer dependencies.

    Semantics-preserving formatting or irrelevant-data perturbations may keep
    the golden workbook unchanged.  A mutation of a direct/transitive answer
    formula dependency may not: that would compare both skills against an
    oracle for a different input.
    """
    inconsistent: list[str] = []
    for case in case_records:
        if case.get("golden_changed"):
            continue
        workbook = openpyxl.load_workbook(
            case["clean_golden"], data_only=False, read_only=False,
        )
        try:
            for change in case.get("input_cell_changes", []):
                location = str(change.get("location", ""))
                if _location_inside_answer_position(location, answer_position) or (
                    _formula_depends_on_cell(
                        workbook,
                        answer_position=answer_position,
                        changed_location=location,
                    )
                ):
                    inconsistent.append(location)
        finally:
            workbook.close()
    if inconsistent:
        raise ValueError(
            "attack changed an answer cell or answer-formula dependency but "
            "left the golden workbook unchanged: "
            + ", ".join(sorted(set(inconsistent))[:20])
        )


def _fallback_sample(items: list[dict], size: int, seed: int) -> list[dict]:
    sampled = list(items)
    random.Random(seed).shuffle(sampled)
    return sampled[:size] if size > 0 else sampled


def _extract_python(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _normalize_mutation_code(code: str) -> str:
    """Normalize a common harmless model mistake before safety validation."""
    normalized = re.sub(r"(?m)^\s*import\s+os\s*$", "", code)
    normalized = re.sub(
        r"os\.environ\[\s*(['\"])ADV_INPUT_PATH\1\s*\]",
        "ADV_INPUT_PATH",
        normalized,
    )
    normalized = re.sub(
        r"os\.environ\[\s*(['\"])ADV_GOLDEN_PATH\1\s*\]",
        "ADV_GOLDEN_PATH",
        normalized,
    )
    return normalized.strip()


def _validate_mutation_code(code: str) -> None:
    if not code.strip():
        raise ValueError("empty mutation program")
    tree = ast.parse(code)
    allowed_imports = {"openpyxl", "copy", "re", "math", "datetime"}
    banned_names = {
        "__import__", "eval", "exec", "compile", "open", "input", "breakpoint",
        "globals", "locals", "vars", "getattr", "setattr", "delattr",
    }
    protected_paths = {"ADV_INPUT_PATH", "ADV_GOLDEN_PATH"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] not in allowed_imports for alias in node.names):
                raise ValueError("mutation program imports a disallowed module")
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".", 1)[0] not in allowed_imports:
                raise ValueError("mutation program imports from a disallowed module")
        elif isinstance(node, ast.Name) and node.id in banned_names:
            raise ValueError(f"mutation program uses disallowed name {node.id!r}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("mutation program uses a private/dunder attribute")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("mutation program may not use global/nonlocal")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in protected_paths:
                    raise ValueError("mutation program may not overwrite path variables")


def _run_mutation_program(
    code: str,
    *,
    input_path: str,
    golden_path: str,
    work_dir: str,
    timeout: int = 30,
) -> tuple[bool, str]:
    _validate_mutation_code(code)
    wrapper = (
        f"ADV_INPUT_PATH = {os.path.abspath(input_path)!r}\n"
        f"ADV_GOLDEN_PATH = {os.path.abspath(golden_path)!r}\n"
        + code
        + "\n"
    )
    script_path = os.path.join(work_dir, "mutation_runtime.py")
    with open(script_path, "w", encoding="utf-8") as handle:
        handle.write(wrapper)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-I", script_path],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "mutation failed").strip()
        return False, detail[-2000:]
    return True, ""


_XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_DOC_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_XLSX_PACKAGE_REL_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)


def _xlsx_sheet_members(path: str) -> dict[str, str]:
    """Map visible worksheet names to their ZIP members without resaving XLSX."""
    with zipfile.ZipFile(path, "r") as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
    targets = {
        str(rel.get("Id")): str(rel.get("Target"))
        for rel in relationships.findall(
            f"{{{_XLSX_PACKAGE_REL_NS}}}Relationship"
        )
    }
    result: dict[str, str] = {}
    for sheet in workbook.findall(f".//{{{_XLSX_MAIN_NS}}}sheet"):
        rel_id = sheet.get(f"{{{_XLSX_DOC_REL_NS}}}id")
        target = targets.get(str(rel_id), "")
        if not target:
            continue
        if target.startswith("/"):
            member = target.lstrip("/")
        else:
            member = posixpath.normpath(posixpath.join("xl", target))
        result[str(sheet.get("name"))] = member
    return result


def _xlsx_existing_cell_locations(path: str) -> set[str]:
    """Return exact worksheet cells editable without creating XML nodes."""
    members = _xlsx_sheet_members(path)
    locations: set[str] = set()
    with zipfile.ZipFile(path, "r") as archive:
        for sheet_name, member in members.items():
            root = ET.fromstring(archive.read(member))
            for cell in root.iter(f"{{{_XLSX_MAIN_NS}}}c"):
                coordinate = str(cell.get("r", "")).replace("$", "").upper()
                if re.fullmatch(r"[A-Z]{1,3}[1-9][0-9]*", coordinate):
                    locations.add(f"{sheet_name}!{coordinate}")
    return locations


def _build_mutation_contract(
    cases: list[tuple[int, str, str]], *, max_reported_locations: int = 5000
) -> dict:
    """Build the machine-readable contract enforced by cell mutations."""
    def common_locations(paths: list[str]) -> set[str]:
        per_case = [_xlsx_existing_cell_locations(path) for path in paths]
        return set.intersection(*per_case) if per_case else set()

    def reported(locations: set[str]) -> dict:
        ordered = sorted(locations, key=str.casefold)
        return {
            "editable_locations": ordered[:max_reported_locations],
            "editable_location_count": len(ordered),
            "editable_locations_truncated": len(ordered) > max_reported_locations,
        }

    input_locations = common_locations([case[1] for case in cases])
    golden_locations = common_locations([case[2] for case in cases])
    return {
        "version": "spreadsheet-cell-operations-v1",
        "operation_schema": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "items": {
                "type": "object",
                "required": ["op", "workbook", "location"],
                "properties": {
                    "op": {"enum": ["set_cell", "clear_cell", "set_formula"]},
                    "workbook": {"enum": ["input", "golden"]},
                    "location": {"pattern": "^.+![A-Z]{1,3}[1-9][0-9]*$"},
                    "value": {"description": "Required by set_cell and set_formula."},
                    "value_type": {"enum": ["auto", "string", "number", "boolean"]},
                },
            },
        },
        "constraints": {
            "allow_new_cells": False,
            "allow_new_sheets": False,
            "duplicate_targets_allowed": False,
            "modified_locations_must_cover_operations": True,
            "operations_must_create_observable_change": True,
            "instruction_change_requires_modified_location": "instruction",
            "preflight_required": True,
        },
        "input": reported(input_locations),
        "golden": reported(golden_locations),
    }


def _validate_operations_against_manifest(
    source_manifest: dict, operations: list[dict]
) -> dict:
    """Cheaply validate structured operations without copying or mutation."""
    if not isinstance(operations, list) or not operations:
        raise ValueError("operations must be a non-empty JSON list")
    if len(operations) > 16:
        raise ValueError("at most 16 structured mutation operations are allowed")
    cases = _find_test_cases(str(source_manifest.get("task_dir", "")))
    if not cases:
        raise ValueError("source task has no runnable cases")
    allowed_ops = {"set_cell", "clear_cell", "set_formula"}
    editable = {
        "input": set.intersection(*[
            _xlsx_existing_cell_locations(case[1]) for case in cases
        ]),
        "golden": set.intersection(*[
            _xlsx_existing_cell_locations(case[2]) for case in cases
        ]),
    }
    editable_folded = {
        workbook: {location.casefold() for location in locations}
        for workbook, locations in editable.items()
    }
    seen: set[tuple[str, str]] = set()
    normalized: list[dict] = []
    for raw_operation in operations:
        if not isinstance(raw_operation, dict):
            raise ValueError("every mutation operation must be a JSON object")
        operation = dict(raw_operation)
        op = str(operation.get("op", "")).strip().lower()
        workbook = str(operation.get("workbook", "input")).strip().lower()
        if op not in allowed_ops:
            raise ValueError(f"unsupported structured mutation operation: {op!r}")
        if workbook not in editable:
            raise ValueError("operation workbook must be 'input' or 'golden'")
        sheet_name, coordinate = _split_operation_location(
            str(operation.get("location", ""))
        )
        location = f"{sheet_name}!{coordinate}"
        key = (workbook, location.casefold())
        if key in seen:
            raise ValueError("multiple operations target the same workbook cell")
        seen.add(key)
        if location.casefold() not in editable_folded[workbook]:
            raise ValueError(
                f"{workbook} operation location is not editable in every case: "
                f"{location}; choose a location listed by mutation_contract"
            )
        if op in {"set_cell", "set_formula"} and "value" not in operation:
            raise ValueError(f"{op} requires a value")
        probe = ET.Element(f"{{{_XLSX_MAIN_NS}}}c", {"r": coordinate})
        _set_xml_cell_value(probe, {**operation, "op": op})
        normalized.append({
            **operation, "op": op, "workbook": workbook, "location": location,
        })
    return {"status": "valid", "operations": normalized}


def _split_operation_location(location: str) -> tuple[str, str]:
    text = str(location or "").strip()
    if "!" not in text:
        raise ValueError("operation location must be an exact Sheet!Cell")
    sheet_name, coordinate = text.rsplit("!", 1)
    sheet_name = sheet_name.strip().strip("'\"")
    coordinate = coordinate.strip().replace("$", "").upper()
    if not re.fullmatch(r"[A-Z]{1,3}[1-9][0-9]*", coordinate):
        raise ValueError("operation location must contain one exact cell")
    if not sheet_name:
        raise ValueError("operation sheet name is empty")
    return sheet_name, coordinate


def _set_xml_cell_value(cell: ET.Element, operation: dict) -> None:
    for child in list(cell):
        if child.tag in {
            f"{{{_XLSX_MAIN_NS}}}f",
            f"{{{_XLSX_MAIN_NS}}}v",
            f"{{{_XLSX_MAIN_NS}}}is",
        }:
            cell.remove(child)
    op = str(operation.get("op", "")).strip().lower()
    if op == "clear_cell":
        cell.attrib.pop("t", None)
        return

    value = operation.get("value")
    value_type = str(operation.get("value_type", "auto")).strip().lower()
    if op == "set_formula":
        value_type = "formula"
    if value is None:
        cell.attrib.pop("t", None)
        return
    if value_type == "auto":
        if isinstance(value, bool):
            value_type = "boolean"
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            value_type = "number"
        elif isinstance(value, str) and value.startswith("="):
            value_type = "formula"
        else:
            value_type = "string"

    if value_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError("boolean cell operation requires a JSON boolean")
        cell.set("t", "b")
        node = ET.SubElement(cell, f"{{{_XLSX_MAIN_NS}}}v")
        node.text = "1" if value else "0"
    elif value_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("number cell operation requires a JSON number")
        cell.attrib.pop("t", None)
        node = ET.SubElement(cell, f"{{{_XLSX_MAIN_NS}}}v")
        node.text = str(value)
    elif value_type == "string":
        if not isinstance(value, str):
            raise ValueError("string cell operation requires a JSON string")
        cell.set("t", "inlineStr")
        inline = ET.SubElement(cell, f"{{{_XLSX_MAIN_NS}}}is")
        text = ET.SubElement(inline, f"{{{_XLSX_MAIN_NS}}}t")
        text.text = value
        if value[:1].isspace() or value[-1:].isspace():
            text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    elif value_type == "formula":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("formula cell operation requires a formula string")
        cell.attrib.pop("t", None)
        formula = ET.SubElement(cell, f"{{{_XLSX_MAIN_NS}}}f")
        formula.text = value.strip().removeprefix("=")
    else:
        raise ValueError(f"unsupported cell value_type: {value_type!r}")


def _rewrite_xlsx_cells(path: str, operations: list[dict]) -> None:
    sheet_members = _xlsx_sheet_members(path)
    grouped: dict[str, list[tuple[str, dict]]] = {}
    for operation in operations:
        sheet_name, coordinate = _split_operation_location(
            str(operation.get("location", ""))
        )
        member = sheet_members.get(sheet_name)
        if not member:
            raise ValueError(f"unknown worksheet in operation: {sheet_name!r}")
        grouped.setdefault(member, []).append((coordinate, operation))

    replacements: dict[str, bytes] = {}
    handle, temporary_path = tempfile.mkstemp(
        suffix=".xlsx", dir=os.path.dirname(os.path.abspath(path))
    )
    os.close(handle)
    try:
        with zipfile.ZipFile(path, "r") as source:
            for member, member_operations in grouped.items():
                root = ET.fromstring(source.read(member))
                cells = {
                    str(cell.get("r", "")).upper(): cell
                    for cell in root.iter(f"{{{_XLSX_MAIN_NS}}}c")
                }
                for coordinate, operation in member_operations:
                    cell = cells.get(coordinate)
                    if cell is None:
                        raise ValueError(
                            "structured cell operations currently require an "
                            f"existing cell: {coordinate}"
                        )
                    _set_xml_cell_value(cell, operation)
                replacements[member] = ET.tostring(
                    root, encoding="utf-8", xml_declaration=True
                )
            with zipfile.ZipFile(temporary_path, "w") as destination:
                for info in source.infolist():
                    destination.writestr(
                        info, replacements.get(info.filename, source.read(info.filename))
                    )
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _apply_mutation_operations(
    *, input_path: str, golden_path: str, operations: list[dict]
) -> None:
    """Apply bounded declarative cell edits to copied workbooks only."""
    if not isinstance(operations, list) or not operations:
        raise ValueError("operations must be a non-empty JSON list")
    if len(operations) > 16:
        raise ValueError("at most 16 structured mutation operations are allowed")
    allowed_ops = {"set_cell", "clear_cell", "set_formula"}
    by_workbook: dict[str, list[dict]] = {"input": [], "golden": []}
    seen: set[tuple[str, str]] = set()
    for raw_operation in operations:
        if not isinstance(raw_operation, dict):
            raise ValueError("every mutation operation must be a JSON object")
        operation = dict(raw_operation)
        op = str(operation.get("op", "")).strip().lower()
        workbook = str(operation.get("workbook", "input")).strip().lower()
        if op not in allowed_ops:
            raise ValueError(f"unsupported structured mutation operation: {op!r}")
        if workbook not in by_workbook:
            raise ValueError("operation workbook must be 'input' or 'golden'")
        sheet_name, coordinate = _split_operation_location(
            str(operation.get("location", ""))
        )
        key = (workbook, f"{sheet_name}!{coordinate}".casefold())
        if key in seen:
            raise ValueError("multiple operations target the same workbook cell")
        seen.add(key)
        operation["op"] = op
        operation["workbook"] = workbook
        by_workbook[workbook].append(operation)
    if by_workbook["input"]:
        _rewrite_xlsx_cells(input_path, by_workbook["input"])
    if by_workbook["golden"]:
        _rewrite_xlsx_cells(golden_path, by_workbook["golden"])


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restore_unchanged_formula_caches(
    clean_path: str,
    mutated_path: str,
) -> int:
    """Restore cached values that openpyxl drops for unchanged formulas.

    Attack code may legitimately replace one golden formula with a new oracle
    value. Re-saving the workbook with openpyxl also clears cached values from
    every untouched formula, which makes the verifier read ``None``. Restore
    only cells whose formula text is unchanged; deliberately replaced answer
    cells remain exactly as authored by the attack agent.
    """
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    cell_tag = f"{{{namespace}}}c"
    formula_tag = f"{{{namespace}}}f"
    value_tag = f"{{{namespace}}}v"
    restored = 0
    temporary_path: str | None = None

    with zipfile.ZipFile(clean_path, "r") as clean_zip, zipfile.ZipFile(
        mutated_path, "r"
    ) as mutated_zip:
        clean_names = set(clean_zip.namelist())
        replacements: dict[str, bytes] = {}
        for name in mutated_zip.namelist():
            if (
                not name.startswith("xl/worksheets/")
                or not name.endswith(".xml")
                or name not in clean_names
            ):
                continue
            clean_root = ET.fromstring(clean_zip.read(name))
            mutated_root = ET.fromstring(mutated_zip.read(name))
            clean_cells = {
                str(cell.get("r", "")): cell
                for cell in clean_root.iter(cell_tag)
                if cell.get("r")
            }
            changed = False
            for cell in mutated_root.iter(cell_tag):
                coordinate = str(cell.get("r", ""))
                clean_cell = clean_cells.get(coordinate)
                if clean_cell is None:
                    continue
                clean_formula = clean_cell.find(formula_tag)
                mutated_formula = cell.find(formula_tag)
                if clean_formula is None or mutated_formula is None:
                    continue
                if (clean_formula.text or "") != (mutated_formula.text or ""):
                    continue
                clean_value = clean_cell.find(value_tag)
                mutated_value = cell.find(value_tag)
                if clean_value is None or clean_value.text is None:
                    continue
                if mutated_value is not None and mutated_value.text is not None:
                    continue
                if mutated_value is None:
                    mutated_value = ET.Element(value_tag)
                    children = list(cell)
                    cell.insert(children.index(mutated_formula) + 1, mutated_value)
                mutated_value.text = clean_value.text
                clean_type = clean_cell.get("t")
                if clean_type is None:
                    cell.attrib.pop("t", None)
                else:
                    cell.set("t", clean_type)
                restored += 1
                changed = True
            if changed:
                ET.register_namespace(
                    "", "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                )
                replacements[name] = ET.tostring(
                    mutated_root,
                    encoding="utf-8",
                    xml_declaration=True,
                )

        if not replacements:
            return 0
        fd, temporary_path = tempfile.mkstemp(
            suffix=".xlsx",
            dir=os.path.dirname(os.path.abspath(mutated_path)),
        )
        os.close(fd)
        with zipfile.ZipFile(temporary_path, "w") as output_zip:
            for info in mutated_zip.infolist():
                output_zip.writestr(
                    info,
                    replacements.get(info.filename, mutated_zip.read(info.filename)),
                )
    try:
        os.replace(temporary_path, mutated_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
    return restored


def _validate_workbook(path: str) -> tuple[bool, str]:
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
        if not workbook.sheetnames:
            return False, "workbook has no sheets"
        workbook.close()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _materialize_attack(
    *,
    adapter,
    source_item: dict,
    source_manifest: dict,
    pattern: dict,
    attack_index: int,
    response: str,
    out_dir: str,
) -> GeneratedAttack | None:
    payload = extract_json(response)
    if not isinstance(payload, dict) or payload.get("decision") != "attack":
        return None
    modified_locations = payload.get("modified_locations")
    if not isinstance(modified_locations, list) or not any(
        str(location).strip() for location in modified_locations
    ):
        raise ValueError("attack must declare at least one exact modified location")
    operations = payload.get("operations")
    code = _normalize_mutation_code(_extract_python(response))
    if operations is None:
        _validate_mutation_code(code)
    elif not isinstance(operations, list) or not operations:
        raise ValueError("attack operations must be a non-empty JSON list")

    source_dir = source_manifest["task_dir"]
    attack_id = f"ccv-s{attack_index:02d}-{re.sub(r'[^A-Za-z0-9_-]', '_', str(source_item['id']))}"
    generated_root = os.path.abspath(os.path.join(out_dir, "generated_data"))
    attack_dir = os.path.abspath(os.path.join(generated_root, attack_id))
    if os.path.commonpath([generated_root, attack_dir]) != generated_root:
        raise ValueError("generated attack path escapes CCV output root")
    if os.path.exists(attack_dir):
        shutil.rmtree(attack_dir)
    shutil.copytree(source_dir, attack_dir)

    original_cases = _find_test_cases(source_dir)
    attack_cases = _find_test_cases(attack_dir)
    if len(original_cases) != len(attack_cases) or not attack_cases:
        raise ValueError("copied attack task has mismatched test cases")

    original_hashes = {
        path: _sha256(path)
        for case in original_cases
        for path in case[1:]
    }
    changed = False
    case_records = []
    for (case_no, clean_input, clean_gold), (_, adv_input, adv_gold) in zip(
        original_cases, attack_cases,
    ):
        before_input = _sha256(adv_input)
        before_gold = _sha256(adv_gold)
        if operations is not None:
            _apply_mutation_operations(
                input_path=adv_input,
                golden_path=adv_gold,
                operations=operations,
            )
        else:
            ok, error = _run_mutation_program(
                code,
                input_path=adv_input,
                golden_path=adv_gold,
                work_dir=attack_dir,
            )
            if not ok:
                raise ValueError(f"mutation failed for case {case_no}: {error}")
        restored_formula_caches = _restore_unchanged_formula_caches(
            clean_gold,
            adv_gold,
        )
        for workbook_path in (adv_input, adv_gold):
            valid, reason = _validate_workbook(workbook_path)
            if not valid:
                raise ValueError(f"invalid workbook {workbook_path}: {reason}")
        input_changed = before_input != _sha256(adv_input)
        gold_changed = before_gold != _sha256(adv_gold)
        input_cell_changes = _cell_changes(clean_input, adv_input)
        golden_cell_changes = _cell_changes(clean_gold, adv_gold)
        changed = changed or input_changed
        case_records.append(
            {
                "case": case_no,
                "input_changed": input_changed,
                "golden_changed": gold_changed,
                "input_cell_changes": input_cell_changes,
                "golden_cell_changes": golden_cell_changes,
                "restored_formula_caches": restored_formula_caches,
                "clean_input": clean_input,
                "clean_golden": clean_gold,
            }
        )

    changed_originals = [
        path for path, digest in original_hashes.items() if _sha256(path) != digest
    ]
    if changed_originals:
        raise RuntimeError(
            "CCV safety violation: original source workbook changed: "
            + ", ".join(changed_originals)
        )

    instruction = str(
        payload.get("adversarial_instruction") or source_item.get("instruction", "")
    ).strip()
    instruction_changed = instruction != str(source_item.get("instruction", "")).strip()
    if not changed and not instruction_changed:
        raise ValueError("attack changes neither workbook input nor instruction")
    answer_position = _answer_position(source_item)
    _validate_nonblank_golden_answers(case_records, answer_position)
    _validate_attack_golden_consistency(case_records, answer_position)
    _validate_declared_locations(
        modified_locations,
        case_records,
        instruction_changed=instruction_changed,
    )

    attack_item = dict(source_item)
    attack_item["id"] = attack_id
    attack_item["spreadsheet_path"] = attack_dir
    attack_item["instruction"] = instruction
    attack_item["ccv_metadata"] = {
        "source_id": str(source_item["id"]),
        "pattern": pattern,
        "intervention": payload.get("intervention", ""),
        "expected_target_failure": payload.get("expected_target_failure", ""),
    }

    answer_position = _answer_position(attack_item)
    instruction_type = str(attack_item.get("instruction_type", ""))
    for (_, _, _), (_, _, adv_gold) in zip(original_cases, attack_cases):
        verifier_result = spreadsheet_evaluator.evaluate(
            adv_gold, adv_gold, instruction_type, answer_position,
        )
        if not verifier_result.get("ok"):
            raise ValueError(
                "existing verifier cannot validate generated golden workbook: "
                f"{verifier_result.get('reason', '')}"
            )

    record = {
        "attack_id": attack_id,
        "source_id": str(source_item["id"]),
        "pattern": pattern,
        "generation": payload,
        "mutation_code": code,
        "mutation_operations": operations or [],
        "cases": case_records,
        "answer_position": answer_position,
        "instruction_type": instruction_type,
        "status": "materialized",
    }
    with open(os.path.join(attack_dir, "attack.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    if operations is not None:
        with open(
            os.path.join(attack_dir, "operations.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(operations, handle, ensure_ascii=False, indent=2)
    else:
        with open(
            os.path.join(attack_dir, "mutation.py"), "w", encoding="utf-8"
        ) as handle:
            handle.write(code + "\n")
    return GeneratedAttack(item=attack_item, record=record)


def _is_transport_failure(result: dict) -> bool:
    reason = str(result.get("fail_reason", "") or "")
    return (
        result.get("phase") == "setup"
        or reason.startswith("llm-call-failed:")
        or reason.startswith("unexpected:")
        or reason == "no-test-cases"
    )


def run_spreadsheet_ccv(
    adapter,
    *,
    old_skill: str,
    candidate_skill: str,
    step: int,
    seed: int,
    out_dir: str,
    source_split: str,
    source_size: int,
    max_patterns: int,
    attacks_per_pattern: int,
    max_attacks: int,
    history_items: int,
    attacker_model: str,
    reasoning_effort: str,
    builder_reasoning_effort: str,
    max_completion_tokens: int,
    timeout: int,
    retries: int,
    trajectory_context: dict | None = None,
    llm_call: Callable[..., tuple[str, dict]] | None = None,
) -> dict:
    """Run bounded, open-ended CCV and return the generic result schema."""
    os.makedirs(out_dir, exist_ok=True)
    call = llm_call or _call_attacker
    usage: dict[str, int] = {}
    records: list[dict] = []

    context = trajectory_context or {}
    compact_context = []
    for key in ("current_step_rollouts", "selection_rollouts"):
        for row in context.get(key, []) or []:
            if isinstance(row, dict):
                compact_context.append(_compact_result(row))
    compact_context.extend(
        _load_recent_trajectories(
            str(getattr(adapter, "_cfg", {}).get("out_root", "") or ""),
            history_items,
        )
    )
    compact_context = compact_context[:history_items]

    source_pool = adapter.dataloader.get_split_items(source_split)
    source_items = _fallback_sample(source_pool, source_size, seed)
    if source_size > 0 and len(source_pool) > source_size:
        selection_user = json.dumps(
            {
                "semantic_delta": _semantic_diff(old_skill, candidate_skill),
                "historical_execution_trajectories": compact_context,
                "allowed_source_pool": [
                    {
                        "id": str(item.get("id", "")),
                        "instruction": str(item.get("instruction", "")),
                        "instruction_type": str(item.get("instruction_type", "")),
                        "answer_position": _answer_position(item),
                    }
                    for item in source_pool
                ],
                "max_sources": source_size,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            selection_path = os.path.join(out_dir, "source_selection_response.txt")
            if os.path.exists(selection_path):
                with open(selection_path, encoding="utf-8") as handle:
                    selection_response = handle.read()
            else:
                selection_response, selection_usage = call(
                    system=SOURCE_SELECTION_SYSTEM,
                    user=selection_user,
                    model=attacker_model,
                    reasoning_effort=reasoning_effort,
                    max_completion_tokens=min(max_completion_tokens, 6000),
                    retries=retries,
                    timeout=timeout,
                )
                _merge_usage(usage, selection_usage)
                with open(selection_path, "w", encoding="utf-8") as handle:
                    handle.write(selection_response)
            selection_payload = extract_json(selection_response)
            selected_ids = (
                selection_payload.get("source_ids", [])
                if isinstance(selection_payload, dict)
                else []
            )
            pool_by_id = {str(item.get("id", "")): item for item in source_pool}
            selected = []
            seen = set()
            for source_id in selected_ids:
                normalized_id = str(source_id)
                if normalized_id in pool_by_id and normalized_id not in seen:
                    selected.append(pool_by_id[normalized_id])
                    seen.add(normalized_id)
                if len(selected) >= source_size:
                    break
            if selected:
                source_items = selected
        except Exception as exc:  # noqa: BLE001
            with open(
                os.path.join(out_dir, "source_selection_error.txt"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")

    manifests = [
        manifest
        for item in source_items
        if (manifest := _source_manifest(adapter, item)) is not None
    ]
    source_by_id = {str(item["id"]): item for item in source_items}
    if not manifests:
        return {
            "status": "no_attack",
            "generated_attacks": 0,
            "valid_attacks": 0,
            "patterns": 0,
            "attack_records": [],
            "usage": usage,
        }

    pattern_user = json.dumps(
        {
            "S_old": old_skill,
            "S_new": candidate_skill,
            "semantic_delta": _semantic_diff(old_skill, candidate_skill),
            "source_tasks": [
                {key: value for key, value in manifest.items() if key != "task_dir"}
                for manifest in manifests
            ],
            "historical_execution_trajectories": compact_context,
            "max_patterns": max_patterns,
        },
        ensure_ascii=False,
        indent=2,
    )
    pattern_path = os.path.join(out_dir, "pattern_response.txt")
    if os.path.exists(pattern_path):
        with open(pattern_path, encoding="utf-8") as handle:
            pattern_response = handle.read()
    else:
        pattern_response, pattern_usage = call(
            system=PATTERN_SYSTEM,
            user=pattern_user,
            model=attacker_model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            timeout=timeout,
        )
        _merge_usage(usage, pattern_usage)
        with open(pattern_path, "w", encoding="utf-8") as handle:
            handle.write(pattern_response)
    pattern_payload = extract_json(pattern_response)
    patterns = (
        pattern_payload.get("patterns", [])
        if isinstance(pattern_payload, dict)
        and pattern_payload.get("decision") == "patterns"
        else []
    )
    patterns = [pattern for pattern in patterns if isinstance(pattern, dict)][:max_patterns]
    if not patterns:
        return {
            "status": "no_attack",
            "generated_attacks": 0,
            "valid_attacks": 0,
            "patterns": 0,
            "attack_records": [],
            "usage": usage,
        }

    manifests_by_id = {str(manifest["id"]): manifest for manifest in manifests}
    unknown_source_patterns = [
        pattern
        for pattern in patterns
        if str(pattern.get("source_id", "")) not in manifests_by_id
    ]
    if unknown_source_patterns:
        alignment_user = json.dumps(
            {
                "allowed_attack_sources": [
                    {key: value for key, value in manifest.items() if key != "task_dir"}
                    for manifest in manifests
                ],
                "hypotheses_to_align": unknown_source_patterns,
                "allowed_source_ids": sorted(manifests_by_id),
            },
            ensure_ascii=False,
            indent=2,
        )
        alignment_path = os.path.join(out_dir, "source_alignment_response.txt")
        if os.path.exists(alignment_path):
            with open(alignment_path, encoding="utf-8") as handle:
                alignment_response = handle.read()
        else:
            alignment_response, alignment_usage = call(
                system=SOURCE_ALIGNMENT_SYSTEM,
                user=alignment_user,
                model=attacker_model,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=min(max_completion_tokens, 6000),
                retries=retries,
                timeout=timeout,
            )
            _merge_usage(usage, alignment_usage)
            with open(alignment_path, "w", encoding="utf-8") as handle:
                handle.write(alignment_response)
        alignment_payload = extract_json(alignment_response)
        alignments = (
            alignment_payload.get("alignments", [])
            if isinstance(alignment_payload, dict)
            else []
        )
        aligned_source = {
            str(row.get("pattern_id", "")): str(row.get("source_id", ""))
            for row in alignments
            if isinstance(row, dict)
            and str(row.get("source_id", "")) in manifests_by_id
        }
        repaired_patterns = []
        for pattern in patterns:
            source_id = str(pattern.get("source_id", ""))
            if source_id in manifests_by_id:
                repaired_patterns.append(pattern)
                continue
            replacement = aligned_source.get(str(pattern.get("pattern_id", "")))
            if replacement:
                repaired = dict(pattern)
                repaired["historical_evidence_source_id"] = source_id
                repaired["source_id"] = replacement
                repaired_patterns.append(repaired)
        patterns = repaired_patterns

    generated = 0
    valid = 0
    invalid = 0
    transport_failures = 0
    old_successes = 0
    candidate_successes = 0
    regressions = 0
    early_stopped = False

    for pattern in patterns:
        source_id = str(pattern.get("source_id", ""))
        source_item = source_by_id.get(source_id)
        source_manifest = manifests_by_id.get(source_id)
        if source_item is None or source_manifest is None:
            records.append(
                {
                    "status": "invalid_pattern",
                    "reason": "unknown source_id",
                    "pattern": pattern,
                }
            )
            continue

        for attempt in range(attacks_per_pattern):
            if generated >= max_attacks or early_stopped:
                break
            generated += 1
            attack_user = json.dumps(
                {
                    "semantic_delta": _semantic_diff(old_skill, candidate_skill),
                    "selected_pattern": pattern,
                    "source_task": {
                        key: value
                        for key, value in source_manifest.items()
                        if key != "task_dir"
                    },
                    "attempt": attempt + 1,
                    "existing_verifier_contract": {
                        "instruction_type": source_manifest["instruction_type"],
                        "answer_position": source_manifest["answer_position"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            try:
                response_path = os.path.join(
                    out_dir, f"attack_{generated:02d}_response.txt",
                )
                if os.path.exists(response_path):
                    with open(response_path, encoding="utf-8") as handle:
                        response = handle.read()
                else:
                    response, attack_usage = call(
                        system=ATTACK_SYSTEM,
                        user=attack_user,
                        model=attacker_model,
                        reasoning_effort=builder_reasoning_effort,
                        max_completion_tokens=max_completion_tokens,
                        retries=retries,
                        timeout=timeout,
                    )
                    _merge_usage(usage, attack_usage)
                    with open(response_path, "w", encoding="utf-8") as handle:
                        handle.write(response)
                attack = _materialize_attack(
                    adapter=adapter,
                    source_item=source_item,
                    source_manifest=source_manifest,
                    pattern=pattern,
                    attack_index=generated,
                    response=response,
                    out_dir=out_dir,
                )
                if attack is None:
                    records.append(
                        {
                            "status": "generator_no_attack",
                            "source_id": source_id,
                            "pattern": pattern,
                            "attempt": attempt + 1,
                        }
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                invalid += 1
                records.append(
                    {
                        "status": "invalid_attack",
                        "source_id": source_id,
                        "pattern": pattern,
                        "attempt": attempt + 1,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            valid += 1
            attack_record = attack.record
            old_dir = os.path.join(out_dir, "paired_runs", attack.item["id"], "old")
            old_result = adapter.rollout([attack.item], old_skill, old_dir)[0]
            attack_record["old_result"] = _compact_result(old_result)
            if _is_transport_failure(old_result):
                valid -= 1
                transport_failures += 1
                attack_record["status"] = "old_transport_failure"
                records.append(attack_record)
                continue
            if not bool(old_result.get("hard")):
                attack_record["status"] = "old_failed_skip_candidate"
                records.append(attack_record)
                continue

            old_successes += 1
            candidate_dir = os.path.join(
                out_dir, "paired_runs", attack.item["id"], "candidate",
            )
            candidate_result = adapter.rollout(
                [attack.item], candidate_skill, candidate_dir,
            )[0]
            attack_record["candidate_result"] = _compact_result(candidate_result)
            if _is_transport_failure(candidate_result):
                valid -= 1
                transport_failures += 1
                attack_record["status"] = "candidate_transport_failure"
                records.append(attack_record)
                continue
            if bool(candidate_result.get("hard")):
                candidate_successes += 1
                attack_record["status"] = "both_pass"
            else:
                regressions += 1
                early_stopped = True
                attack_record["status"] = "update_regression"
            records.append(attack_record)

    result = {
        "status": (
            "error"
            if transport_failures
            else ("complete" if valid else "no_attack")
        ),
        "generated_attacks": generated,
        "valid_attacks": valid,
        "old_successes": old_successes,
        "candidate_successes": candidate_successes,
        "regressions": regressions,
        "improvements": 0,
        "patterns": len(patterns),
        "invalid_attacks": invalid,
        "transport_failures": transport_failures,
        "early_stopped": early_stopped,
        "attack_records": records,
        "usage": usage,
        "error": (
            f"{transport_failures} paired CCV rollout(s) exhausted retries"
            if transport_failures
            else ""
        ),
    }
    with open(os.path.join(out_dir, "ccv_result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result
