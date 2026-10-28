from __future__ import annotations

from dataclasses import replace
import json
import tempfile
from pathlib import Path

import pytest

from skillopt.regularization.atoms import AtomicSegmenter, exact_span_atoms, rule_segment
from skillopt.regularization.complexity import measure_complexity, normalized_growth
from skillopt.regularization.core import (
    SkillEvoReg,
    scheduled_complexity_lambda,
    scheduled_epoch_value,
)
from skillopt.regularization.dropout import (
    PLACEHOLDER_PREFIX,
    apply_atomic_dropout,
    contains_placeholder,
    masked_atom_preservation_errors,
    redact_masked_atom_bodies,
    remove_placeholders,
)
from skillopt.regularization.gate import evaluate_regularized_gate
from skillopt.regularization.local_repair import (
    apply_replacements,
    editable_regions,
    regularization_repair_regions,
    new_exact_duplicate_atom_errors,
    protected_content_hash,
    validate_delta_operations,
    validate_skill_document,
    validate_repaired_document,
)
from skillopt.regularization.epoch_feedback import EpochFeedbackRuntime
from skillopt.regularization.ccv import CCVResult, run_adapter_ccv
from skillopt.regularization.verifier import resolve_verifier_contract
from skillopt.engine.trainer import (
    _build_step_local_preselection_feedback,
    _mean_repeat_scores,
    _run_regularization_local_edit,
)
from skillopt.optimizer.clip import structural_edit_metrics
from skillopt.optimizer.skill import apply_patch_with_report


def test_rule_segmentation_is_exact_cover() -> None:
    skill = "# Procedure\n\n1. Read the workbook.\n2. Verify every output.\n\n```python\nif x:\n    y()\n```\n"
    atoms = rule_segment(skill)
    assert "".join(atom.span for atom in atoms) == skill
    assert any(atom.type == "code" for atom in atoms)


def test_step_local_preselection_routes_dropout_and_excess_growth() -> None:
    class Regularization:
        complexity_free_growth = 0.01
        complexity_enabled = True

        @staticmethod
        def measure(skill):
            return measure_complexity(skill, AtomicSegmenter(mode="rule"))

    old = "# Skill\n\n- Keep one general rule.\n"
    candidate = old + "- Add a distinct supported rule with detail.\n"
    feedback, telemetry = _build_step_local_preselection_feedback(
        Regularization(), old_skill=old, candidate_skill=candidate,
        dropout_report={"masked_count": 1, "mode": "training_rollout"},
    )
    assert feedback is not None
    assert feedback["abnormal_modules"] == ["dropout", "complexity"]
    assert telemetry["dropout_triggered"] is True
    assert telemetry["complexity_triggered"] is True


def test_step_local_preselection_can_ablate_complexity_without_disabling_regularization() -> None:
    class Regularization:
        complexity_free_growth = 0.0
        complexity_enabled = False

        @staticmethod
        def measure(skill):
            return measure_complexity(skill, AtomicSegmenter(mode="rule"))

    old = "# Skill\n\n- Keep one general rule.\n"
    candidate = old + "- Add a distinct supported rule with detail.\n"
    feedback, telemetry = _build_step_local_preselection_feedback(
        Regularization(), old_skill=old, candidate_skill=candidate,
        dropout_report={"masked_count": 0, "mode": "training_rollout"},
    )
    assert feedback is None
    assert telemetry["complexity_enabled"] is False
    assert telemetry["complexity_triggered"] is False


def test_regularization_local_edit_uses_feedback_and_does_not_evaluate(
    monkeypatch, tmp_path,
) -> None:
    anchor = "# Skill\n\n- Apply one general rule to repeated narrow cases.\n"
    candidate = anchor + "- Apply one narrow rule to repeated special cases.\n"
    captured = {}

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            '{"operations":[{"source_region_id":"r1","operation":"REWRITE",'
            '"match_region_id":null}],"replacements":[{"region_id":"r1",'
            '"replacement":"- Apply one general rule across repeated cases.\\n"}],'
            '"change_summary":["merged duplicates"]}',
            {"prompt_tokens": 10, "completion_tokens": 5},
        )

    monkeypatch.setattr(
        "skillopt.engine.trainer.chat_optimizer", fake_chat_optimizer,
    )
    feedback = {
        "source": "SkillEvoReg",
        "reasons": ["negative_regularization_score"],
        "revision_advice": ["merge duplicate rules"],
    }
    result = _run_regularization_local_edit(
        candidate,
        feedback,
        anchor_skill=anchor,
        out_dir=str(tmp_path),
        reasoning_effort="medium",
        max_completion_tokens=4096,
    )

    assert result["status"] == "ok"
    assert result["new_skill"].endswith("\n")
    assert captured["stage"] == "regularization_local_edit"
    assert "merge duplicate rules" in captured["user"]
    assert "never return the complete skill" in captured["system"]
    assert "Operations are authoritative" in captured["system"]
    assert "runtime canonicalizes their edits" in captured["system"]
    assert "Do not return replacement rows for m* atoms" in captured["system"]
    assert "MERGE and DELETE are invalid for an already-deleted delta" in captured["system"]
    assert "unlisted source delta is already a deterministic NOOP" in captured["system"]
    assert "exactly one transferable procedural invariant" not in captured["system"]
    assert "ordered compression policy" not in captured["system"]
    assert (tmp_path / "local_editd_skill.md").exists()


def test_regularization_local_edit_requires_abstraction_for_confirmed_ccv(
    monkeypatch, tmp_path,
) -> None:
    anchor = (
        "# Skill\n\n"
        "- Generalize incidental assumptions into transferable procedural invariants.\n"
    )
    candidate = anchor + (
        "- Generalize one concrete field assumption into a procedural invariant.\n"
    )
    captured = {}

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            '{"operations":[{"source_region_id":"r1","operation":"REWRITE",'
            '"match_region_id":null}],"replacements":[{"region_id":"r1",'
            '"replacement":"- Generalize field assumptions into transferable invariants.\\n"}],'
            '"change_summary":["generalized the responsible rule"]}',
            {"prompt_tokens": 10, "completion_tokens": 5},
        )

    monkeypatch.setattr(
        "skillopt.engine.trainer.chat_optimizer", fake_chat_optimizer,
    )
    feedback = {
        "source": "SkillEvoReg",
        "abnormal_modules": ["ccv"],
        "reasons": ["ccv_update_regression"],
        "ccv": {
            "confirmed_regressions": 1,
            "counterexamples": [{
                "hypothesis": "A concrete field-specific assumption.",
            }],
        },
    }
    result = _run_regularization_local_edit(
        candidate,
        feedback,
        anchor_skill=anchor,
        out_dir=str(tmp_path),
        reasoning_effort="medium",
        max_completion_tokens=4096,
    )

    assert result["status"] == "ok"
    assert "exactly one transferable procedural invariant" in captured["system"]
    assert "Do not append a new checklist branch" in captured["system"]
    assert "counterexample-specific field names" in captured["system"]
    assert "ordered compression policy" not in captured["system"]


def test_regularization_local_edit_retries_destructive_compression(
    monkeypatch, tmp_path,
) -> None:
    anchor = (
        "# Skill\n\n"
        "- Preserve this stable capability across workbook variants.\n"
    )
    candidate = anchor + (
        "- Preserve this stable capability for one workbook variant.\n"
    )
    replies = iter([
        '{"new_skill":"# Skill\\n\\n- Tiny replacement.\\n",'
        '"change_summary":["compressed everything"]}',
        '{"operations":[{"source_region_id":"r1","operation":"REWRITE",'
        '"match_region_id":null}],"replacements":[{"region_id":"r1",'
        '"replacement":"- Preserve stable capability across variants.\\n"}],'
        '"change_summary":["repaired only source update"]}',
    ])
    calls = []

    def fake_chat_optimizer(**kwargs):
        calls.append(kwargs)
        return next(replies), {"prompt_tokens": 10, "completion_tokens": 5}

    monkeypatch.setattr(
        "skillopt.engine.trainer.chat_optimizer", fake_chat_optimizer,
    )
    result = _run_regularization_local_edit(
        candidate,
        {
            "source": "SkillEvoReg",
            "abnormal_modules": ["complexity"],
            "complexity": {"triggered": True},
        },
        anchor_skill=anchor,
        out_dir=str(tmp_path),
        reasoning_effort="medium",
        max_completion_tokens=4096,
    )

    assert result["status"] == "ok"
    assert len(calls) == 2
    assert "Preserve this stable capability" in result["new_skill"]
    assert result["validator_attempts"][0]["validation_errors"]
    assert result["validator_attempts"][1]["validation_errors"] == []
    assert "Structural validator feedback" in calls[1]["user"]


def test_regularization_local_edit_records_but_accepts_complexity_increase(
    monkeypatch, tmp_path,
) -> None:
    anchor = (
        "# Skill\n\n"
        "- Keep one concise capability reminder for repeated cases.\n"
    )
    candidate = anchor + (
        "- Keep a duplicate capability reminder for one repeated case.\n"
    )
    replies = iter([
        '{"operations":[{"source_region_id":"r1","operation":"REWRITE",'
        '"match_region_id":null}],"replacements":[{"region_id":"r1",'
        '"replacement":"- Duplicate capability reminder with a much longer '
        'and more specific explanation that increases complexity.\\n"}],'
        '"change_summary":["expanded"]}',
    ])
    calls = []

    def fake_chat_optimizer(**kwargs):
        calls.append(kwargs)
        return next(replies), {"prompt_tokens": 10, "completion_tokens": 5}

    monkeypatch.setattr(
        "skillopt.engine.trainer.chat_optimizer", fake_chat_optimizer,
    )
    result = _run_regularization_local_edit(
        candidate,
        {
            "source": "SkillEvoReg",
            "abnormal_modules": ["complexity"],
            "complexity": {"triggered": True},
        },
        anchor_skill=anchor,
        out_dir=str(tmp_path),
        reasoning_effort="medium",
        max_completion_tokens=4096,
    )
    assert result["status"] == "ok"
    assert "much longer" in result["new_skill"]
    assert len(calls) == 1
    telemetry = result["validator_attempts"][0]["telemetry"]
    assert telemetry["complexity_direction"] == "increased"
    assert telemetry["complexity_delta"] > 0
    assert result["validator_attempts"][0]["validation_errors"] == []


def test_regularization_local_edit_noop_preserves_candidate(monkeypatch, tmp_path) -> None:
    anchor = (
        "# Skill\n\n"
        "- Keep one stable source update across repeated cases.\n"
    )
    candidate = anchor + (
        "- Keep this stable source update unchanged for repeated cases.\n"
    )

    def fake_chat_optimizer(**_kwargs):
        return (
            '{"operations":[{"source_region_id":"r1","operation":"NOOP",'
            '"match_region_id":null}],"replacements":[],'
            '"change_summary":["no repair needed"]}',
            {"prompt_tokens": 10, "completion_tokens": 5},
        )

    monkeypatch.setattr(
        "skillopt.engine.trainer.chat_optimizer", fake_chat_optimizer,
    )
    result = _run_regularization_local_edit(
        candidate,
        {"source": "SkillEvoReg", "abnormal_modules": ["ccv"]},
        anchor_skill=anchor,
        out_dir=str(tmp_path),
        reasoning_effort="medium",
        max_completion_tokens=4096,
    )
    assert result["status"] == "ok"
    assert result["noop"] is True
    assert result["new_skill"] == candidate


def test_complexity_all_noop_gets_one_soft_retry_then_is_accepted(
    monkeypatch, tmp_path,
) -> None:
    anchor = (
        "# Skill\n\n"
        "- Keep one stable source update across repeated cases.\n"
    )
    candidate = anchor + (
        "- Keep this stable source update unchanged for repeated cases.\n"
    )
    calls = []

    def fake_chat_optimizer(**kwargs):
        calls.append(kwargs)
        return (
            '{"operations":[{"source_region_id":"r1","operation":"NOOP",'
            '"match_region_id":null}],"replacements":[],'
            '"change_summary":["no safe repair"]}',
            {"prompt_tokens": 10, "completion_tokens": 5},
        )

    monkeypatch.setattr(
        "skillopt.engine.trainer.chat_optimizer", fake_chat_optimizer,
    )
    result = _run_regularization_local_edit(
        candidate,
        {
            "source": "SkillEvoReg",
            "abnormal_modules": ["complexity"],
            "complexity": {"triggered": True},
        },
        anchor_skill=anchor,
        out_dir=str(tmp_path),
        reasoning_effort="medium",
        max_completion_tokens=4096,
    )

    assert result["status"] == "ok"
    assert result["noop"] is True
    assert result["new_skill"] == candidate
    assert len(calls) == 2
    assert "Optional complexity soft retry" in calls[1]["user"]
    attempts = result["validator_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["validation_errors"] == []
    assert attempts[0]["telemetry"]["soft_retry_reason"] == (
        "complexity_first_attempt_all_noop"
    )
    for attempt in attempts:
        telemetry = attempt["telemetry"]
        assert telemetry["complexity_delta"] == pytest.approx(0.0)
        assert telemetry["complexity_direction"] == "unchanged"


def test_regularization_local_edit_all_unmatched_is_automatic_noop(
    monkeypatch, tmp_path,
) -> None:
    anchor = "# Skill\n\n- alpha bravo charlie delta echo foxtrot golf.\n"
    candidate = anchor + "- 101 202 303 404 505 606 707.\n"
    assert not any(
        region.role == "existing_match"
        for region in regularization_repair_regions(anchor, candidate)
    )
    calls = []

    def fail_if_called(**kwargs):
        calls.append(kwargs)
        raise AssertionError("unmatched source delta must not call the model")

    monkeypatch.setattr(
        "skillopt.engine.trainer.chat_optimizer", fail_if_called,
    )
    result = _run_regularization_local_edit(
        candidate,
        {"source": "SkillEvoReg", "abnormal_modules": ["complexity"]},
        anchor_skill=anchor,
        out_dir=str(tmp_path),
        reasoning_effort="medium",
        max_completion_tokens=4096,
    )

    assert calls == []
    assert result["status"] == "ok"
    assert result["noop"] is True
    assert result["new_skill"] == candidate
    assert result["automatic_noop_source_region_ids"] == ["r1"]
    telemetry = result["validator_attempts"][0]["telemetry"]
    assert telemetry["model_called"] is False
    assert telemetry["complexity_delta"] == pytest.approx(0.0)
    assert telemetry["complexity_direction"] == "unchanged"
    assert telemetry["delta_operations"] == [{
        "source_region_id": "r1",
        "operation": "NOOP",
        "match_region_id": "",
    }]


def test_regularization_local_edit_mixed_match_only_sends_matched_source(
    monkeypatch, tmp_path,
) -> None:
    anchor = (
        "# Skill\n\n"
        "- Use one stable idempotency key across retry requests.\n"
        "- alpha bravo charlie delta echo foxtrot golf.\n"
        "- Preserve unrelated workbook formulas and formatting during edits.\n"
    )
    candidate = (
        "# Skill\n\n"
        "- Use one stable idempotency key across retry requests.\n"
        "- Keep the stable idempotency key across retry attempts.\n"
        "- alpha bravo charlie delta echo foxtrot golf.\n"
        "- 101 202 303 404 505 606 707.\n"
        "- Preserve unrelated workbook formulas and formatting during edits.\n"
    )
    captured = {}

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            '{"operations":[{"source_region_id":"r1","operation":"REWRITE",'
            '"match_region_id":null}],"replacements":[{"region_id":"r1",'
            '"replacement":"- Use one stable idempotency key for all retry attempts.\\n"}],'
            '"change_summary":["generalized matched delta"]}',
            {"prompt_tokens": 10, "completion_tokens": 5},
        )

    monkeypatch.setattr(
        "skillopt.engine.trainer.chat_optimizer", fake_chat_optimizer,
    )
    result = _run_regularization_local_edit(
        candidate,
        {"source": "SkillEvoReg", "abnormal_modules": ["ccv"]},
        anchor_skill=anchor,
        out_dir=str(tmp_path),
        reasoning_effort="medium",
        max_completion_tokens=4096,
    )

    assert captured["stage"] == "regularization_local_edit"
    assert result["status"] == "ok"
    assert result["noop"] is False
    assert result["automatic_noop_source_region_ids"] == ["r2"]
    assert "- 101 202 303 404 505 606 707.\n" in result["new_skill"]
    request = json.loads((tmp_path / "local_edit_request.json").read_text())
    assert "Cosine similarity only nominates a pair" in request["system"]
    assert "preserve the full union" in request["system"]
    assert "if any supported behavior would be lost, use NOOP" in request["system"]
    assert {region["region_id"] for region in request["editable_regions"]} == {
        "r1", "m1",
    }
    assert request["automatic_noop_source_region_ids"] == ["r2"]
    operations = result["validator_attempts"][0]["telemetry"]["delta_operations"]
    assert operations == [
        {
            "source_region_id": "r1",
            "operation": "REWRITE",
            "match_region_id": "",
        },
        {
            "source_region_id": "r2",
            "operation": "NOOP",
            "match_region_id": "",
        },
    ]


def test_unmatched_source_delta_validator_allows_only_noop() -> None:
    anchor = "# Skill\n\n- alpha bravo charlie delta echo foxtrot golf.\n"
    candidate = anchor + "- 101 202 303 404 505 606 707.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")

    operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "NOOP",
            "match_region_id": None,
        }]},
        regions,
        {},
        abnormal_modules=frozenset({"complexity"}),
    )
    assert operations[0]["operation"] == "NOOP"
    assert errors == []

    replacements_by_operation = {
        "REWRITE": {source.region_id: "- generalized numeric procedure.\n"},
        "DELETE": {source.region_id: ""},
        "MERGE": {source.region_id: "- generalized numeric procedure.\n"},
    }
    for operation, replacements in replacements_by_operation.items():
        _operations, errors = validate_delta_operations(
            {"operations": [{
                "source_region_id": source.region_id,
                "operation": operation,
                "match_region_id": None,
            }]},
            regions,
            replacements,
            abnormal_modules=frozenset({"complexity"}),
        )
        assert any(
            f"{operation} for {source.region_id} is invalid because the source "
            "delta has no similarity-qualified existing atom" == error
            for error in errors
        )


def test_validator_does_not_accept_match_below_similarity_threshold() -> None:
    anchor = (
        "# Skill\n\n"
        "- Use one stable idempotency key across retry requests.\n"
    )
    candidate = anchor + (
        "- Keep the stable idempotency key across retry attempts.\n"
    )
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    regions = [
        replace(region, similarity=0.499999)
        if region.role == "existing_match"
        else region
        for region in regions
    ]
    _operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "REWRITE",
            "match_region_id": None,
        }]},
        regions,
        {source.region_id: "- Use one stable retry idempotency key.\n"},
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == [
        "REWRITE for r1 is invalid because the source delta has no "
        "similarity-qualified existing atom"
    ]


def test_local_repair_preserves_every_byte_outside_source_delta() -> None:
    anchor = "# Skill\n\n- Stable first rule.\n- Stable second rule.\n"
    candidate = anchor + "- Narrow source rule.\n"
    regions = editable_regions(anchor, candidate)
    before_hash = protected_content_hash(candidate, regions)
    repaired = apply_replacements(
        candidate, regions, {"r1": "- Generalized source rule."},
    )
    assert repaired == anchor + "- Generalized source rule.\n"
    assert protected_content_hash(repaired, editable_regions(anchor, repaired)) == before_hash


def test_local_repair_ignores_frontmatter_only_changes() -> None:
    anchor = (
        "---\nname: sheet\ndescription: Old description.\n---\n\n"
        "# Workflow\n\n- Preserve formulas.\n"
    )
    candidate = anchor.replace("Old description.", "New description.")
    assert editable_regions(anchor, candidate) == []

    candidate += "- Validate the saved workbook.\n"
    regions = editable_regions(anchor, candidate)
    assert len(regions) == 1
    assert regions[0].anchor_text == ""
    assert "Validate the saved workbook" in regions[0].candidate_text
    assert "description:" not in regions[0].candidate_text


def test_regularization_repair_opens_one_similar_existing_atom() -> None:
    anchor = (
        "# Skill\n\n- Always write literal values into checked cells.\n"
        "- Preserve unrelated workbook formatting.\n"
    )
    candidate = anchor + "- For checked lookup cells, write literal results.\n"
    regions = regularization_repair_regions(anchor, candidate)
    history = [region for region in regions if region.role == "existing_match"]
    assert len(history) == 1
    assert "literal values" in history[0].candidate_text
    assert not history[0].candidate_text.lstrip().startswith("#")
    assert len([region for region in regions if region.role == "source_delta"]) == 1


def test_delta_delete_removes_and_noop_preserves_source_delta() -> None:
    anchor = "# Skill\n\n- General existing rule for checked values.\n"
    candidate = anchor + "- General existing rule for all checked values.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "DELETE",
            "match_region_id": None,
        }]}, regions, {source.region_id: ""},
        abnormal_modules=frozenset({"complexity"}),
    )
    assert operations[0]["operation"] == "DELETE"
    assert errors == []
    operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "NOOP",
            "match_region_id": None,
        }]}, regions, {},
        abnormal_modules=frozenset({"complexity"}),
    )
    assert operations[0]["operation"] == "NOOP"
    assert errors == []


def test_operation_first_canonicalization_discards_contradictory_edits() -> None:
    anchor = "# Skill\n\n- Use one stable idempotency key for retries.\n"
    candidate = anchor + "- Use one stable idempotency key across retries.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    matched = next(region for region in regions if region.role == "existing_match")

    cases = [
        (
            "NOOP",
            {
                source.region_id: "- Contradictory rewrite.\n",
                matched.region_id: "- Contradictory match rewrite.\n",
            },
            {},
            "",
        ),
        (
            "REWRITE",
            {
                source.region_id: "- Use one stable key for every retry.\n",
                matched.region_id: "- Contradictory match rewrite.\n",
            },
            {source.region_id: "- Use one stable key for every retry.\n"},
            "",
        ),
        (
            "DELETE",
            {
                source.region_id: "- Contradictory non-empty source.\n",
                matched.region_id: "- Contradictory match rewrite.\n",
            },
            {source.region_id: ""},
            "",
        ),
        (
            "MERGE",
            {
                source.region_id: "- Use one stable idempotency key for retries.\n",
                matched.region_id: "- Contradictory non-empty duplicate.\n",
            },
            {
                source.region_id: "- Use one stable idempotency key for retries.\n",
                matched.region_id: "",
            },
            matched.region_id,
        ),
    ]
    for operation, replacements, expected, expected_match in cases:
        operations, errors = validate_delta_operations(
            {"operations": [{
                "source_region_id": source.region_id,
                "operation": operation,
                "match_region_id": "wrong-model-match",
            }]},
            regions,
            replacements,
            abnormal_modules=frozenset({"complexity"}),
        )
        assert errors == []
        assert replacements == expected
        assert operations == [{
            "source_region_id": source.region_id,
            "operation": operation,
            "match_region_id": expected_match,
        }]


def test_operation_first_canonicalization_keeps_structural_rejections() -> None:
    anchor = "# Skill\n\n- Use one stable idempotency key for retries.\n"
    candidate = anchor + "- Keep the idempotency key stable across retry attempts.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")

    _operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": "unknown",
            "operation": "NOOP",
        }]},
        regions,
        {},
        abnormal_modules=frozenset({"complexity"}),
    )
    assert "unknown source-delta region: 'unknown'" in errors
    assert "missing decisions for source deltas: r1" in errors

    _operations, errors = validate_delta_operations(
        {"operations": [
            {"source_region_id": source.region_id, "operation": "NOOP"},
            {"source_region_id": source.region_id, "operation": "NOOP"},
        ]},
        regions,
        {},
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == ["duplicate delta decision for r1"]

    for operation in ("REWRITE", "MERGE"):
        _operations, errors = validate_delta_operations(
            {"operations": [{
                "source_region_id": source.region_id,
                "operation": operation,
            }]},
            regions,
            {},
            abnormal_modules=frozenset({"complexity"}),
        )
        assert any("requires a non-empty" in error for error in errors)


def test_deleted_source_delta_can_noop_or_rewrite_but_not_delete_again() -> None:
    anchor = (
        "# Skill\n\n"
        "- Use one stable idempotency key for retries.\n"
        "- Use one stable idempotency key across retries.\n"
    )
    candidate = "# Skill\n\n- Use one stable idempotency key for retries.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    matched = next(region for region in regions if region.role == "existing_match")
    assert source.candidate_text == ""
    base = {"source_region_id": source.region_id, "match_region_id": None}

    _operations, errors = validate_delta_operations(
        {"operations": [{**base, "operation": "NOOP"}]}, regions, {},
        abnormal_modules=frozenset({"ccv"}),
    )
    assert errors == []
    _operations, errors = validate_delta_operations(
        {"operations": [{**base, "operation": "REWRITE"}]}, regions,
        {source.region_id: "- Restored transferable capability.\n"},
        abnormal_modules=frozenset({"ccv"}),
    )
    assert errors == []
    _operations, errors = validate_delta_operations(
        {"operations": [{**base, "operation": "DELETE"}]}, regions,
        {source.region_id: ""}, abnormal_modules=frozenset({"ccv"}),
    )
    assert errors == [
        "DELETE for r1 is invalid because the source delta is already empty"
    ]
    _operations, errors = validate_delta_operations(
        {"operations": [{
            **base,
            "operation": "MERGE",
            "match_region_id": matched.region_id,
        }]},
        regions,
        {
            source.region_id: "- Use one stable idempotency key across retries.\n",
            matched.region_id: "",
        },
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == [
        "MERGE for r1 is invalid because the source delta is already empty"
    ]


def test_regularization_assigns_one_unique_existing_atom_per_source_delta() -> None:
    anchor = (
        "# Skill\n\n"
        "- Retry requests with one stable idempotency key.\n"
        "- Preserve unrelated workbook formatting.\n"
        "- Validate required schema fields before submission.\n"
    )
    candidate = (
        "# Skill\n\n"
        "- Retry requests with one stable idempotency key.\n"
        "- Retry requests using one stable idempotency key.\n"
        "- Preserve unrelated workbook formatting.\n"
        "- Validate required schema fields before submission.\n"
        "- Check required schema fields before sending the request.\n"
    )
    regions = regularization_repair_regions(anchor, candidate)
    sources = [r for r in regions if r.role == "source_delta"]
    history = [r for r in regions if r.role == "existing_match"]
    assert len(sources) == 2
    assert len(history) == 2
    assert {r.source_region_id for r in history} == {r.region_id for r in sources}
    assert len({(r.start, r.end) for r in history}) == 2

    history_by_source = {r.source_region_id: r for r in history}
    replacements: dict[str, str] = {}
    operations = []
    for source in sources:
        matched = history_by_source[source.region_id]
        operations.append({
            "source_region_id": source.region_id,
            "operation": "MERGE",
            "match_region_id": matched.region_id,
        })
        replacements[source.region_id] = (
            "- Use one stable idempotency key across all retry attempts.\n"
            if "idempotency" in source.candidate_text
            else "- Validate all required schema fields before submission.\n"
        )
        replacements[matched.region_id] = ""
    normalized, errors = validate_delta_operations(
        {"operations": operations}, regions, replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert len(normalized) == 2
    assert errors == []
    repaired = apply_replacements(candidate, regions, replacements)
    assert repaired.count("idempotency key") == 1
    assert repaired.count("required schema fields") == 1

    operations[1]["match_region_id"] = operations[0]["match_region_id"]
    normalized, errors = validate_delta_operations(
        {"operations": operations}, regions, replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == []
    assert {
        item["source_region_id"]: item["match_region_id"]
        for item in normalized
    } == {
        source.region_id: history_by_source[source.region_id].region_id
        for source in sources
    }


def test_regularization_merge_can_consolidate_three_historical_atoms() -> None:
    anchor = (
        "# Skill\n\n"
        "- Retry transient requests with bounded exponential backoff.\n"
        "- Reuse one stable idempotency key across every retry attempt.\n"
        "- Stop retrying permanent validation and authentication failures.\n"
    )
    candidate = anchor + (
        "- Retry safely with bounded backoff, stable identity, and permanent-error classification.\n"
    )
    regions = regularization_repair_regions(
        anchor, candidate, min_similarity=0.0,
    )
    source = next(region for region in regions if region.role == "source_delta")
    matches = [
        region for region in regions
        if region.role == "existing_match"
        and region.source_region_id == source.region_id
    ]
    assert len(matches) == 3
    replacements = {
        source.region_id: (
            "- Retry only transient failures with bounded exponential backoff, "
            "reuse one stable idempotency key, and stop on permanent validation "
            "or authentication failures.\n"
        ),
    }
    operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "MERGE",
            "match_region_id": "model-does-not-own-the-cluster",
        }]},
        regions,
        replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == []
    assert operations[0]["match_region_ids"] == [
        match.region_id for match in matches
    ]
    assert all(replacements[match.region_id] == "" for match in matches)
    repaired = apply_replacements(candidate, regions, replacements)
    assert repaired.count("- Retry") == 1


def test_same_batch_group_is_only_a_composite_history_query(monkeypatch) -> None:
    import numpy as np
    import skillopt.regularization.local_repair as local_repair

    anchor = "# Skill\n\n- Inspect the workbook's actual populated extent.\n"
    candidate = anchor + (
        "- Use actual row and column bounds instead of a truncated preview.\n"
        "- Scan the complete populated region before processing values.\n"
    )
    calls = 0

    def fake_similarity(queries, documents, *, model_id_or_path):
        nonlocal calls
        calls += 1
        if calls == 1:  # same-batch grouping
            return np.asarray([[1.0, 0.90], [0.90, 1.0]])
        assert len(queries) == 1
        assert "truncated preview" in queries[0]
        assert "complete populated region" in queries[0]
        return np.full((len(queries), len(documents)), 0.85)  # group -> history

    monkeypatch.setattr(local_repair, "cosine_similarity_matrix", fake_similarity)
    regions = regularization_repair_regions(anchor, candidate, min_similarity=0.75)
    source = next(region for region in regions if region.role == "source_delta")
    member = next(region for region in regions if region.role == "same_batch_member")
    history = next(region for region in regions if region.role == "existing_match")
    assert member.source_region_id == source.region_id
    assert history.source_region_id == source.region_id
    assert member.match_scope == "same_batch"
    assert history.match_scope == "history"

    replacements = {source.region_id: "- Inspect the complete workbook extent.\n"}
    _operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "MERGE",
            "match_region_id": history.region_id,
        }]},
        regions,
        replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == []
    assert replacements[member.region_id] == ""
    assert replacements[history.region_id] == ""


def test_same_batch_group_without_history_cannot_authorize_merge(monkeypatch) -> None:
    import numpy as np
    import skillopt.regularization.local_repair as local_repair

    anchor = "# Skill\n\n- Preserve unrelated workbook content.\n"
    candidate = anchor + (
        "- Compute displayed results as literal values.\n"
        "- Store final displayed results rather than unevaluated formulas.\n"
    )
    calls = 0

    def fake_similarity(queries, documents, *, model_id_or_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.asarray([[1.0, 0.90], [0.90, 1.0]])
        return np.full((len(queries), len(documents)), 0.40)

    monkeypatch.setattr(local_repair, "cosine_similarity_matrix", fake_similarity)
    regions = regularization_repair_regions(anchor, candidate, min_similarity=0.75)
    source = next(region for region in regions if region.role == "source_delta")
    assert any(region.role == "same_batch_member" for region in regions)
    assert not any(region.role == "existing_match" for region in regions)
    _operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "MERGE",
            "match_region_id": None,
        }]},
        regions,
        {source.region_id: "- Compute and store displayed values.\n"},
        abnormal_modules=frozenset({"complexity"}),
    )
    assert any("no similarity-qualified existing atom" in error for error in errors)

def test_multi_atom_diff_hunk_is_split_and_each_pair_is_isolated() -> None:
    anchor = (
        "# Skill\n\n"
        "- Retry requests with one stable idempotency key.\n"
        "- Validate required schema fields before submission.\n"
        "- Preserve unrelated workbook formatting.\n"
    )
    candidate = anchor + (
        "- Retry requests using one stable idempotency key.\n"
        "- Check required schema fields before sending the request.\n"
    )
    regions = regularization_repair_regions(anchor, candidate)
    sources = [region for region in regions if region.role == "source_delta"]
    matches = [region for region in regions if region.role == "existing_match"]
    assert len(sources) == 2
    assert all(source.candidate_text.count("\n") == 1 for source in sources)
    assert len(matches) == 2
    assert {match.source_region_id for match in matches} == {
        source.region_id for source in sources
    }

    retry_source = next(
        source for source in sources if "idempotency" in source.candidate_text
    )
    schema_source = next(
        source for source in sources if "schema" in source.candidate_text
    )
    retry_match = next(
        match for match in matches
        if match.source_region_id == retry_source.region_id
    )
    replacements = {
        retry_source.region_id: (
            "- Use one stable idempotency key across every retry attempt.\n"
        ),
        # Contradictory edits outside the declared operation are discarded.
        schema_source.region_id: "- Delete this useful schema rule.\n",
        retry_match.region_id: "- Rewrite the linked history incorrectly.\n",
    }
    operations, errors = validate_delta_operations(
        {"operations": [
            {
                "source_region_id": retry_source.region_id,
                "operation": "MERGE",
                "match_region_id": "model-cannot-choose-the-pair",
            },
            {
                "source_region_id": schema_source.region_id,
                "operation": "NOOP",
                "match_region_id": None,
            },
        ]},
        regions,
        replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == []
    assert next(
        row for row in operations
        if row["source_region_id"] == retry_source.region_id
    )["match_region_id"] == retry_match.region_id
    repaired = apply_replacements(candidate, regions, replacements)
    assert repaired.count("idempotency key") == 1
    assert schema_source.candidate_text in repaired
    assert "Delete this useful schema rule" not in repaired
    assert "Preserve unrelated workbook formatting" in repaired


def test_adjacent_deleted_atoms_remain_separate_zero_width_deltas() -> None:
    anchor = (
        "# Skill\n\n"
        "- Preserve the stable capability.\n"
        "- First removable narrow exception.\n"
        "- Second removable narrow exception.\n"
    )
    candidate = "# Skill\n\n- Preserve the stable capability.\n"
    sources = editable_regions(anchor, candidate)
    assert len(sources) == 2
    assert all(source.start == source.end for source in sources)
    assert [source.anchor_text for source in sources] == [
        "- First removable narrow exception.\n",
        "- Second removable narrow exception.\n",
    ]


def test_structural_heading_delta_is_not_paired_with_rule_text() -> None:
    anchor = (
        "# Skill\n\n"
        "- Replicate a displayed value from another cell exactly.\n"
    )
    candidate = anchor + "### Replicating a displayed value from another cell\n"
    regions = regularization_repair_regions(anchor, candidate, min_similarity=0.0)
    source = next(region for region in regions if region.role == "source_delta")
    assert source.atom_type == "other"
    assert not any(region.role == "existing_match" for region in regions)


def test_merge_canonicalizes_linked_atom_and_rejects_missing_source_body() -> None:
    anchor = (
        "# Skill\n\n"
        "- Use a stable idempotency key for retries.\n"
        "- Preserve unrelated workbook formatting.\n"
    )
    candidate = anchor + "- Keep the idempotency key unchanged across retries.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    matched = next(region for region in regions if region.role == "existing_match")
    payload = {"operations": [{
        "source_region_id": source.region_id,
        "operation": "MERGE",
        "match_region_id": matched.region_id,
    }]}

    replacements = {
        source.region_id: "- Use one stable idempotency key across retries.\n",
        matched.region_id: matched.candidate_text,
    }
    operations, errors = validate_delta_operations(
        payload,
        regions,
        replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == []
    assert operations[0]["match_region_id"] == matched.region_id
    assert replacements == {
        source.region_id: "- Use one stable idempotency key across retries.\n",
        matched.region_id: "",
    }

    replacements = {source.region_id: "", matched.region_id: "not empty"}
    _operations, errors = validate_delta_operations(
        payload,
        regions,
        replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == [
        f"MERGE for {source.region_id} requires a non-empty merged source replacement"
    ]


def test_local_repair_uses_whole_atom_boundaries_and_rejects_broken_fence() -> None:
    anchor = "# Skill\n\n- Stable.\n"
    candidate = anchor + "```python\nprint('narrow')\n```\n"
    regions = editable_regions(anchor, candidate)
    assert regions[0].candidate_text.startswith("```python")
    repaired = apply_replacements(candidate, regions, {"r1": "```python\nprint('general')"})
    assert validate_repaired_document(repaired) == [
        "repair produced an unbalanced fenced code block"
    ]


def test_new_exact_duplicate_atom_invariant_is_incremental_and_narrow() -> None:
    rule = "- Preserve formulas and unrelated workbook formatting during every edit.\n"
    heading = "# Skill\n\n"

    # A repair may introduce one new rule, but not a second exact copy.
    assert new_exact_duplicate_atom_errors(heading, heading + rule) == []
    errors = new_exact_duplicate_atom_errors(
        heading + rule,
        heading + rule + rule.upper().replace("- ", "* ", 1),
    )
    assert len(errors) == 1
    assert "before=1, after=2" in errors[0]

    # Pre-existing duplication is grandfathered when the count does not grow.
    assert new_exact_duplicate_atom_errors(
        heading + rule + rule,
        heading + rule + rule,
    ) == []

    # A MERGE-style move keeps the global count at one.
    before_move = heading + rule + "- Validate every saved workbook before delivery.\n"
    after_move = (
        heading
        + "- Validate every saved workbook before delivery.\n"
        + rule
    )
    assert new_exact_duplicate_atom_errors(before_move, after_move) == []

    # Fenced code is deliberately outside this exact-rule invariant.
    code = "```python\nprint('same block')\n```\n"
    assert new_exact_duplicate_atom_errors(heading + code, heading + code + code) == []


def test_regularization_local_edit_retries_a_new_exact_duplicate(
    monkeypatch, tmp_path,
) -> None:
    canonical = (
        "- Preserve formulas and unrelated workbook formatting during every edit.\n"
    )
    anchor = "# Skill\n\n" + canonical
    candidate = anchor + (
        "- Preserve unrelated formulas and workbook formatting during each edit.\n"
    )
    replies = iter([
        '{"operations":[{"source_region_id":"r1","operation":"REWRITE",'
        '"match_region_id":null}],"replacements":[{"region_id":"r1",'
        '"replacement":"- Preserve formulas and unrelated workbook formatting '
        'during every edit.\\n"}],"change_summary":["copied"]}',
        '{"operations":[{"source_region_id":"r1","operation":"MERGE",'
        '"match_region_id":"m1"}],"replacements":[{"region_id":"r1",'
        '"replacement":"- Preserve formulas and unrelated workbook formatting '
        'during every edit.\\n"}],"change_summary":["collapsed"]}',
    ])
    calls = []

    def fake_chat_optimizer(**kwargs):
        calls.append(kwargs)
        return next(replies), {"prompt_tokens": 10, "completion_tokens": 5}

    monkeypatch.setattr(
        "skillopt.engine.trainer.chat_optimizer", fake_chat_optimizer,
    )
    result = _run_regularization_local_edit(
        candidate,
        {
            "source": "SkillEvoReg",
            "abnormal_modules": ["complexity"],
            "complexity": {"triggered": True},
        },
        anchor_skill=anchor,
        out_dir=str(tmp_path),
        reasoning_effort="medium",
        max_completion_tokens=4096,
    )

    assert result["status"] == "ok"
    assert len(calls) == 2
    assert "Do not copy or restate a read-only atom" in calls[0]["system"]
    first_errors = result["validator_attempts"][0]["validation_errors"]
    assert len(first_errors) == 1
    assert "introduced an exact duplicate rule" in first_errors[0]
    assert result["validator_attempts"][1]["validation_errors"] == []
    assert result["new_skill"].count(canonical) == 1


def test_shared_skill_validation_blocks_malformed_ordinary_candidate() -> None:
    assert validate_skill_document("# Skill\n\n```python\nprint('broken')\n") == [
        "skill document has an unbalanced fenced code block"
    ]
    assert validate_skill_document(
        "# Skill\n\n<MASKED_SKILL_ATOM_1>\n"
    ) == ["skill document contains a dropout placeholder"]




def test_llm_spans_fill_omitted_gaps_without_rewriting() -> None:
    skill = "Header\nRule A\nRule B\n"
    atoms = exact_span_atoms(skill, [
        {"type": "constraint", "span": "Rule A"},
        {"type": "constraint", "span": "Rule B"},
    ])
    assert "".join(atom.span for atom in atoms) == skill
    assert atoms[1].span == "Rule A"


def test_llm_segmenter_uses_configured_completion_budget() -> None:
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return (
            '{"segments":[{"type":"constraint","span":"Keep A1 unchanged."}]}',
            {},
        )

    segmenter = AtomicSegmenter(
        "llm",
        llm_call=fake_llm,
        max_completion_tokens=65536,
    )
    atoms, metadata = segmenter.segment("Keep A1 unchanged.")
    assert "".join(atom.span for atom in atoms) == "Keep A1 unchanged."
    assert not metadata["fallback"]
    assert "last_error" not in metadata
    assert captured["max_completion_tokens"] == 65536


def test_segment_cache_survives_a_new_segmenter_instance() -> None:
    skill = "- Rule one.\n- Rule two.\n"
    with tempfile.TemporaryDirectory() as tmp_dir:
        first = AtomicSegmenter("rule", cache_dir=tmp_dir)
        first_atoms, first_meta = first.segment(skill)
        second = AtomicSegmenter("rule", cache_dir=tmp_dir)
        second_atoms, second_meta = second.segment(skill)
    assert first_atoms == second_atoms
    assert not first_meta["cache_hit"]
    assert second_meta["cache_hit"]
    assert second_meta["cache_source"] == "disk"


def test_complexity_penalizes_only_growth_above_old_skill_budget() -> None:
    segmenter = AtomicSegmenter("rule")
    old = measure_complexity("One coherent rule.", segmenter)
    candidate = measure_complexity("- Rule one.\n- Rule two.\n- Rule three.\n", segmenter)
    growth = normalized_growth(old, candidate, free_growth_budget=0.10)
    assert candidate.atom_count > old.atom_count
    assert candidate.total > old.total
    assert growth["normalizer"] == max(old.total, 1.0)
    assert growth["normalizer_source"] == "old_skill"
    assert growth["relative_growth"] > growth["normalized_growth"]
    assert growth["normalized_growth"] == max(
        0.0, growth["relative_growth"] - 0.10,
    )
    assert growth["normalized_growth"] > 0


def test_complexity_free_budget_does_not_penalize_small_growth() -> None:
    segmenter = AtomicSegmenter("rule")
    old = measure_complexity("A" * 1000, segmenter)
    candidate = measure_complexity("A" * 1050, segmenter)
    growth = normalized_growth(old, candidate, free_growth_budget=0.10)
    assert growth["relative_growth"] <= 0.10
    assert growth["normalized_growth"] == 0.0


def test_complexity_lambda_inverse_cosine_schedule() -> None:
    values = [
        scheduled_complexity_lambda(
            step=step,
            total_steps=32,
            minimum=0.01,
            maximum=0.75,
            scheduler="inverse_cosine",
        )
        for step in (0, 16, 32)
    ]
    assert values[0] == 0.01
    assert abs(values[1] - 0.38) < 1e-12
    assert values[2] == 0.75


def test_epoch_linear_regularization_schedule_has_a_true_warmup() -> None:
    dropout = [
        round(scheduled_epoch_value(
            step=step,
            steps_per_epoch=4,
            minimum=0.0,
            maximum=0.05,
            scheduler="epoch_linear",
            warmup_epochs=1,
            ramp_epochs=5,
        ), 4)
        for step in (1, 5, 9, 13, 17, 21, 25)
    ]
    assert dropout == [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.05]
    ccv_sources = [
        round(scheduled_epoch_value(
            step=step,
            steps_per_epoch=4,
            minimum=0,
            maximum=4,
            scheduler="epoch_linear",
            warmup_epochs=1,
            ramp_epochs=4,
        ))
        for step in (1, 5, 9, 13, 17, 21)
    ]
    assert ccv_sources == [0, 1, 2, 3, 4, 4]


def test_atomic_dropout_is_deterministic_and_detectable() -> None:
    skill = "- Rule one.\n- Rule two.\n- Rule three.\n"
    atoms = rule_segment(skill)
    first = apply_atomic_dropout(skill, atoms, probability=1.0, seed=7, max_atoms=1)
    second = apply_atomic_dropout(skill, atoms, probability=1.0, seed=7, max_atoms=1)
    assert first.proposal_skill == second.proposal_skill
    assert len(first.masked_atoms) == 1
    assert contains_placeholder(first.proposal_skill)


def test_atomic_dropout_can_guarantee_at_least_one_without_a_cap() -> None:
    skill = "- Rule one.\n- Rule two.\n- Rule three.\n"
    atoms = rule_segment(skill)
    result = apply_atomic_dropout(
        skill,
        atoms,
        probability=0.0,
        seed=7,
        max_atoms=0,
        ensure_at_least_one=True,
    )
    assert len(result.masked_atoms) == 1
    cleaned, removed = remove_placeholders(result.proposal_skill)
    assert removed == 1
    assert not contains_placeholder(cleaned)


def test_dropout_patch_commits_to_unmasked_skill_without_deleting_atoms() -> None:
    skill = "- Rule one.\n- Rule two.\n- Rule three.\n"
    dropped = apply_atomic_dropout(
        skill, rule_segment(skill), probability=1.0, seed=7, max_atoms=1,
    )
    patch = {
        "reasoning": "add one evidence-supported rule",
        "edits": [{"op": "append", "content": "- Rule four.\n"}],
    }

    # This mirrors trainer training/update: proposal_skill is used by the
    # rollout and optimizer, but the ranked patch is committed against the
    # complete incumbent.
    candidate, report = apply_patch_with_report(skill, patch)

    assert sum(row["status"].startswith("applied") for row in report) == 1
    assert candidate == skill + "\n- Rule four.\n"
    assert masked_atom_preservation_errors(candidate, dropped.to_dict()) == []


def test_dropout_preservation_invariant_detects_old_pruning_behavior() -> None:
    skill = "- Rule one.\n- Rule two.\n- Rule three.\n"
    dropped = apply_atomic_dropout(
        skill, rule_segment(skill), probability=1.0, seed=7, max_atoms=1,
    )
    old_candidate, _ = remove_placeholders(dropped.proposal_skill)

    assert masked_atom_preservation_errors(old_candidate, dropped.to_dict())


def test_dropout_preservation_allows_adjacent_visible_atom_edits() -> None:
    skill = "# Workflow\n\n- Rule one.\n- Rule two.\n- Rule three.\n"
    dropped = apply_atomic_dropout(
        skill, rule_segment(skill), probability=1.0, seed=7, max_atoms=1,
    )
    candidate = skill.replace("- Rule one.", "- Rule one, carefully.").replace(
        "- Rule three.", "- Rule three, with verification."
    )

    assert masked_atom_preservation_errors(candidate, dropped.to_dict()) == []


def test_dropout_preservation_rejects_relocated_masked_atom() -> None:
    skill = "# Workflow\n\n- Rule one.\n- Rule two.\n- Rule three.\n"
    dropped = apply_atomic_dropout(
        skill, rule_segment(skill), probability=1.0, seed=7, max_atoms=1,
    )
    hidden = dropped.masked_atoms[0]["original_span"]
    candidate = skill.replace(hidden, "", 1).rstrip() + "\n" + hidden

    assert masked_atom_preservation_errors(candidate, dropped.to_dict())


def test_dropout_redacts_hidden_body_from_optimizer_history() -> None:
    skill = "- Visible rule.\n- Secret temporary rule.\n"
    dropped = apply_atomic_dropout(
        skill, rule_segment(skill), probability=1.0, seed=5, max_atoms=1,
    )
    row = dropped.masked_atoms[0]
    history = f"Prior patch repeated:\n{row['original_span']}\nend"
    redacted, count = redact_masked_atom_bodies(history, dropped.to_dict())
    assert count >= 1
    assert row["original_span"].strip() not in redacted
    assert PLACEHOLDER_PREFIX in redacted


def test_progressive_dropout_does_not_force_an_epoch_one_mask() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": True,
        "regularization_dropout_probability_min": 0.0,
        "regularization_dropout_probability_max": 0.05,
        "regularization_dropout_scheduler": "epoch_linear",
        "regularization_dropout_warmup_epochs": 1,
        "regularization_dropout_ramp_epochs": 5,
        "regularization_dropout_ensure_at_least_one": False,
        "num_epochs": 8,
        "steps_per_epoch": 4,
    }, initial_skill="- Rule one.\n- Rule two.\n", adapter=Adapter())
    proposal, report, context = regularization.prepare_proposal(
        "- Rule one.\n- Rule two.\n", seed=7, step=1,
    )
    assert proposal == "- Rule one.\n- Rule two.\n"
    assert report["probability"] == 0.0
    assert report["masked_count"] == 0
    assert context == ""


def test_regularization_finalizes_placeholder_without_markdown_rewrite() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": True,
        "regularization_ccv_enabled": False,
    }, initial_skill="Initial rule.", adapter=Adapter())
    candidate, report = regularization.finalize_candidate(
        '- <MASKED_SKILL_ATOM id="m1" type="constraint">\n- Keep this.\n'
    )
    assert candidate == "- \n- Keep this.\n"
    assert report["removed_placeholders"] == 1
    assert not report["placeholder_remaining"]


def test_regularization_feedback_is_answer_free_and_next_step_oriented() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_feedback_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_ccv_enabled": False,
        "regularization_ccv_source_split": "train",
    }, initial_skill="Initial rule.", adapter=Adapter())
    feedback = regularization.build_feedback({
        "decision": {
            "accepted": False,
            "action": "regularization_reject",
            "reasons": [
                "negative_regularization_score",
                "complexity_regularized_rejection",
                "ccv_update_regression",
            ],
            "complexity_triggered": True,
            "delta_iid": 0.05,
            "regularization_score": -0.01,
        },
        "complexity": {
            "growth": {
                "relative_growth": 0.30,
                "normalized_growth": 0.30,
            },
            "lambda_complex": 0.20,
            "update_diagnostics": {
                "whole_skill_net_change": 3.5,
                "top_complexity_contributors": [{
                    "candidate_atom_id": "a0007",
                    "candidate_span_preview": "Do not leak this atom span.",
                    "diagnosis": "overlapping_expansion",
                    "source_edit": {"index": 2},
                }],
            },
        },
        "ccv": {
            "status": "complete",
            "regressions": 1,
            "attack_records": [{
                "status": "update_regression",
                "attack_id": "attack-1",
                "source_id": "train-1",
                "pattern": {"incidental_assumption": "Assumes exact keys."},
                "generation": {"intervention": "Perturb a training key."},
                "candidate_result": {
                    "phase": "eval",
                    "fail_reason": (
                        "eval-mismatch: value@Sheet!C5: "
                        "gt='secret' pred='wrong'"
                    ),
                },
            }],
        },
    }, edit_budget=4)
    assert feedback is not None
    assert feedback["abnormal_modules"] == ["complexity", "ccv"]
    assert feedback["complexity"]["penalty"] == 0.06
    assert feedback["complexity"]["dominant_pattern_counts"] == {
        "overlapping_expansion": 1,
    }
    assert feedback["complexity"]["next_update_policy"]["max_selected_edits"] == 4
    assert feedback["complexity"]["next_update_policy"]["ordered_operations"] == [
        "merge semantically overlapping or duplicate rules",
        "remove narrow, one-off, task-specific, or unsupported rules",
        "replace clusters of similar rules with one transferable general rule",
        "compress repeated explanations, procedures, and examples",
    ]
    assert feedback["ccv"]["repair_policy"][
        "derive_exactly_one_transferable_invariant"
    ] is True
    assert "top_complexity_contributors" not in feedback["complexity"]
    assert "candidate_atom_id" not in str(feedback)
    assert "Do not leak this atom span" not in str(feedback)
    assert "validation" not in feedback
    assert "regularization_score" not in feedback["complexity"]
    assert feedback["provenance"] == {
        "validation_signal": "accept_reject_only",
        "ccv_source_split": "train",
        "test_access": False,
    }
    failure = feedback["ccv"]["counterexamples"][0]["observed_failure"]
    assert failure == {"phase": "eval", "locations": ["Sheet!C5"]}
    assert "secret" not in str(feedback)
    assert "wrong" not in str(feedback)
    assert feedback["revision_advice"]
    assert "answer" not in str(feedback).lower()


def test_regularization_feedback_routes_only_complexity_when_it_causes_rejection() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_feedback_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_ccv_enabled": False,
    }, initial_skill="Initial rule.", adapter=Adapter())
    feedback = regularization.build_feedback({
        "decision": {
            "accepted": False,
            "action": "regularization_reject",
            "reasons": [
                "negative_regularization_score", "complexity_regularized_rejection",
            ],
            "complexity_triggered": True,
        },
        "complexity": {
            "growth": {
                "relative_growth": 0.20,
                "free_growth_budget": 0.10,
                "normalized_growth": 0.10,
            },
            "lambda_complex": 0.15,
            "update_diagnostics": {},
        },
        "ccv": {"status": "complete", "regressions": 0},
    })
    assert feedback is not None
    assert feedback["abnormal_modules"] == ["complexity"]
    assert "complexity" in feedback
    assert "ccv" not in feedback


def test_inconclusive_ccv_does_not_suppress_complexity_feedback() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_feedback_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_ccv_enabled": True,
    }, initial_skill="Initial rule.", adapter=Adapter())
    feedback = regularization.build_feedback({
        "decision": {
            "accepted": False,
            "action": "regularization_defer",
            "reasons": [
                "negative_regularization_score", "complexity_regularized_rejection",
                "ccv_inconclusive",
            ],
            "complexity_triggered": True,
        },
        "complexity": {
            "growth": {
                "relative_growth": 0.20,
                "free_growth_budget": 0.10,
                "normalized_growth": 0.10,
            },
            "lambda_complex": 0.15,
            "update_diagnostics": {},
        },
        "ccv": {"status": "inconclusive", "regressions": 0},
    })
    assert feedback is not None
    assert feedback["abnormal_modules"] == ["complexity"]
    assert "complexity" in feedback
    assert "ccv" not in feedback


def test_regularization_feedback_routes_only_ccv_without_compression_signal() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_feedback_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_ccv_enabled": True,
    }, initial_skill="Initial rule.", adapter=Adapter())
    feedback = regularization.build_feedback({
        "decision": {
            "accepted": False,
            "action": "regularization_reject",
            "reasons": ["ccv_update_regression"],
        },
        "complexity": {
            "growth": {
                "relative_growth": 0.05,
                "free_growth_budget": 0.10,
                "normalized_growth": 0.0,
            },
            "lambda_complex": 0.15,
            "update_diagnostics": {},
        },
        "ccv": {"status": "complete", "regressions": 1},
    })
    assert feedback is not None
    assert feedback["abnormal_modules"] == ["ccv"]
    assert "ccv" in feedback
    assert "complexity" not in feedback
    assert "compress" not in str(feedback).lower()


def test_regularization_feedback_ignores_validation_only_rejection() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_feedback_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_ccv_enabled": False,
    }, initial_skill="Initial rule.", adapter=Adapter())
    feedback = regularization.build_feedback({
        "decision": {
            "accepted": False,
            "action": "regularization_reject",
            "reasons": ["validation_regression", "negative_regularization_score"],
        },
        "complexity": {
            "growth": {
                "relative_growth": 0.05,
                "free_growth_budget": 0.10,
                "normalized_growth": 0.0,
            },
            "lambda_complex": 0.15,
            "update_diagnostics": {},
        },
        "ccv": {"status": "complete", "regressions": 0},
    })
    assert feedback is None


def test_epoch_feedback_is_routed_only_after_complexity_flips_score() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_feedback_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_ccv_enabled": False,
        "regularization_lambda_complex_min": 1.0,
        "regularization_lambda_complex_max": 1.0,
        "regularization_lambda_complex_scheduler": "constant",
        "num_epochs": 2,
        "steps_per_epoch": 1,
    }, initial_skill="- General rule.\n", adapter=Adapter())
    feedback, telemetry = regularization.build_epoch_module_feedback(
        old_skill="- General rule.\n",
        candidate_skill=(
            "- General rule.\n"
            "- Narrow special-case rule with several repeated details.\n"
        ),
        current_score=0.60,
        candidate_score=0.61,
        global_step=1,
        ccv=CCVResult(status="complete"),
        edit_budget=4,
    )
    assert telemetry["scope"] == "epoch_cumulative"
    assert telemetry["decision"]["complexity_triggered"] is True
    assert feedback is not None
    assert feedback["abnormal_modules"] == ["complexity"]


def test_epoch_validation_regression_does_not_trigger_compression_feedback() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_feedback_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_ccv_enabled": False,
        "regularization_lambda_complex_min": 1.0,
        "regularization_lambda_complex_max": 1.0,
        "regularization_lambda_complex_scheduler": "constant",
        "num_epochs": 2,
        "steps_per_epoch": 1,
    }, initial_skill="- General rule.\n", adapter=Adapter())
    feedback, telemetry = regularization.build_epoch_module_feedback(
        old_skill="- General rule.\n",
        candidate_skill=(
            "- General rule.\n"
            "- Narrow special-case rule with several repeated details.\n"
        ),
        current_score=0.60,
        candidate_score=0.59,
        global_step=1,
        ccv=CCVResult(status="complete"),
    )
    assert telemetry["decision"]["complexity_triggered"] is False
    assert "validation_regression" in telemetry["decision"]["reasons"]
    assert feedback is None


def test_epoch_confirmed_ccv_routes_ccv_without_complexity() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_feedback_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_ccv_enabled": True,
        "regularization_lambda_complex_min": 0.0,
        "regularization_lambda_complex_max": 0.0,
        "regularization_lambda_complex_scheduler": "constant",
        "num_epochs": 2,
        "steps_per_epoch": 1,
    }, initial_skill="- General rule.\n", adapter=Adapter())
    feedback, telemetry = regularization.build_epoch_module_feedback(
        old_skill="- General rule.\n",
        candidate_skill="- General rule.\n",
        current_score=0.60,
        candidate_score=0.61,
        global_step=1,
        ccv=CCVResult(status="complete", regressions=1),
    )
    assert telemetry["decision"]["complexity_triggered"] is False
    assert feedback is not None
    assert feedback["abnormal_modules"] == ["ccv"]
    assert "complexity" not in feedback


def test_regularization_teacher_feedback_requires_a_teacher_model() -> None:
    class Adapter:
        pass

    try:
        SkillEvoReg({
            "regularization_enabled": True,
            "regularization_feedback_enabled": True,
            "regularization_feedback_author": "teacher",
            "regularization_segmenter": "rule",
        }, initial_skill="Initial rule.", adapter=Adapter())
    except ValueError as exc:
        assert "feedback_teacher_model" in str(exc)
    else:
        raise AssertionError("teacher feedback must require an explicit model")


def test_regularization_teacher_accepts_an_explicit_model() -> None:
    class Adapter:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_feedback_enabled": True,
        "regularization_feedback_author": "teacher",
        "regularization_feedback_teacher_model": "claude-opus-4-8-thinking",
        "regularization_segmenter": "rule",
    }, initial_skill="Initial rule.", adapter=Adapter())
    assert regularization.feedback_teacher_model == "claude-opus-4-8-thinking"


def test_regularization_accepts_iid_gain_that_pays_complexity_cost() -> None:
    gate, decision = evaluate_regularized_gate(
        candidate_skill="candidate",
        candidate_score=0.70,
        current_skill="old",
        current_score=0.60,
        best_skill="old",
        best_score=0.60,
        best_step=0,
        global_step=1,
        normalized_complexity_growth=0.10,
        lambda_complex=0.20,
        placeholder_free=True,
        ccv=CCVResult(status="complete", valid_attacks=4),
        tau_adv=0.0,
        ccv_required=True,
    )
    assert decision.accepted
    assert decision.complexity_triggered is False
    assert gate.action == "accept_new_best"


def test_regularization_attributes_rejection_to_complexity_only_when_cost_flips_score() -> None:
    gate, decision = evaluate_regularized_gate(
        candidate_skill="candidate",
        candidate_score=0.61,
        current_skill="old",
        current_score=0.60,
        best_skill="old",
        best_score=0.60,
        best_step=0,
        global_step=1,
        normalized_complexity_growth=0.10,
        lambda_complex=0.20,
        placeholder_free=True,
        ccv=CCVResult(status="complete"),
        tau_adv=0.0,
        ccv_required=False,
    )
    assert gate.action == "reject"
    assert abs(decision.complexity_cost - 0.02) < 1e-12
    assert decision.complexity_triggered is True
    assert "complexity_regularized_rejection" in decision.reasons


def test_validation_regression_does_not_emit_complexity_attribution() -> None:
    gate, decision = evaluate_regularized_gate(
        candidate_skill="candidate",
        candidate_score=0.59,
        current_skill="old",
        current_score=0.60,
        best_skill="old",
        best_score=0.60,
        best_step=0,
        global_step=1,
        normalized_complexity_growth=0.50,
        lambda_complex=0.20,
        placeholder_free=True,
        ccv=CCVResult(status="complete"),
        tau_adv=0.0,
        ccv_required=False,
    )
    assert gate.action == "reject"
    assert "validation_regression" in decision.reasons
    assert decision.complexity_triggered is False
    assert "complexity_regularized_rejection" not in decision.reasons


def test_regularization_rejects_exact_zero_score_boundary() -> None:
    gate, decision = evaluate_regularized_gate(
        candidate_skill="candidate",
        candidate_score=0.60,
        current_skill="old",
        current_score=0.60,
        best_skill="old",
        best_score=0.60,
        best_step=0,
        global_step=1,
        normalized_complexity_growth=0.0,
        lambda_complex=0.10,
        placeholder_free=True,
        ccv=CCVResult(status="no_attack"),
        tau_adv=0.0,
        ccv_required=True,
    )
    assert not decision.accepted
    assert decision.regularization_score == 0.0
    assert "negative_regularization_score" in decision.reasons
    assert gate.action == "reject"


def test_structural_edit_metrics_are_factual_and_target_aware() -> None:
    skill = "# Skill\n\n- Preserve formulas.\n\n- Normalize lookup keys.\n"
    valid = structural_edit_metrics(
        skill,
        {
            "op": "replace",
            "target": "- Preserve formulas.",
            "content": "- Preserve formulas and named ranges.",
            "support_count": 7,
        },
        index=2,
    )
    invalid = structural_edit_metrics(
        skill,
        {
            "op": "insert_after",
            "target": "missing target",
            "content": "- Append a disconnected rule.",
            "support_count": 1,
        },
        index=3,
    )
    assert valid["eligible"] is True
    assert valid["target_match"] is True
    assert valid["support_count"] == 7
    assert isinstance(valid["token_delta"], int)
    assert 0.0 <= valid["max_lexical_overlap"] <= 1.0
    assert invalid["eligible"] is False
    assert invalid["target_match"] is False
    assert "invalid_target_or_operation" in invalid["structural_flags"]


def test_regularization_ignores_placeholder_but_rejects_update_induced_regression() -> None:
    gate, decision = evaluate_regularized_gate(
        candidate_skill='<MASKED_SKILL_ATOM id="m1" type="constraint">',
        candidate_score=0.80,
        current_skill="old",
        current_score=0.60,
        best_skill="old",
        best_score=0.60,
        best_step=0,
        global_step=1,
        normalized_complexity_growth=0.0,
        lambda_complex=0.1,
        placeholder_free=False,
        ccv=CCVResult(
            status="complete",
            valid_attacks=4,
            old_successes=4,
            candidate_successes=3,
            regressions=1,
        ),
        tau_adv=0.0,
        ccv_required=True,
    )
    assert not decision.accepted
    assert gate.action == "reject"
    assert "dropout_placeholder_remaining" not in decision.reasons
    assert "ccv_update_regression" in decision.reasons


def test_regularization_rejects_ccv_transport_error() -> None:
    gate, decision = evaluate_regularized_gate(
        candidate_skill="candidate",
        candidate_score=0.70,
        current_skill="old",
        current_score=0.60,
        best_skill="old",
        best_score=0.60,
        best_step=0,
        global_step=1,
        normalized_complexity_growth=0.0,
        lambda_complex=0.1,
        placeholder_free=True,
        ccv=CCVResult(
            status="error",
            transport_failures=1,
            error="paired rollout exhausted retries",
        ),
        tau_adv=0.0,
        ccv_required=True,
    )
    assert not decision.accepted
    assert gate.action == "reject"
    assert "ccv_inconclusive" in decision.reasons


def test_core_records_ccv_abstention_without_fabricating_attacks() -> None:
    class AdapterWithoutCCV:
        pass

    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_lambda_complex": 0.0,
        "regularization_ccv_enabled": True,
        "regularization_ccv_required": False,
    }, initial_skill="Initial rule.", adapter=AdapterWithoutCCV())
    with tempfile.TemporaryDirectory() as tmp_dir:
        gate, telemetry = regularization.evaluate(
            old_skill="Initial rule.",
            candidate_skill="Initial rule. Better rule.",
            candidate_score=0.7,
            current_score=0.6,
            best_skill="Initial rule.",
            best_score=0.6,
            best_step=0,
            global_step=1,
            seed=41,
            out_dir=tmp_dir,
        )
    assert gate.action == "accept_new_best"
    assert telemetry["ccv"]["status"] == "provider_unavailable"
    assert telemetry["ccv"]["valid_attacks"] == 0


def test_core_skips_ccv_when_regularized_val_gate_does_not_clear() -> None:
    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls = []

        def run_ccv(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "no_attack"}

    adapter = RecordingAdapter()
    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_lambda_complex_min": 1.0,
        "regularization_lambda_complex_max": 1.0,
        "regularization_ccv_enabled": True,
        "regularization_ccv_required": True,
    }, initial_skill="Initial rule.", adapter=adapter)
    candidate = "Initial rule.\n" + "\n".join(
        f"- Added rule {i}." for i in range(30)
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        gate, telemetry = regularization.evaluate(
            old_skill="Initial rule.",
            candidate_skill=candidate,
            candidate_score=0.61,
            current_score=0.60,
            best_skill="Initial rule.",
            best_score=0.60,
            best_step=0,
            global_step=1,
            seed=41,
            out_dir=tmp_dir,
        )
    assert gate.action == "reject"
    assert telemetry["decision"]["regularization_score"] < 0.0
    assert telemetry["ccv"]["status"] == "skipped_precheck"
    assert adapter.calls == []


def test_progressive_ccv_is_off_then_uses_scheduled_budgets() -> None:
    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls = []

        def run_ccv(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "no_attack"}

    adapter = RecordingAdapter()
    regularization = SkillEvoReg({
        "regularization_enabled": True,
        "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False,
        "regularization_lambda_complex": 0.0,
        "regularization_ccv_enabled": True,
        "regularization_ccv_required": True,
        "regularization_ccv_source_size_min": 0,
        "regularization_ccv_source_size": 4,
        "regularization_ccv_max_attacks_min": 0,
        "regularization_ccv_max_attacks": 4,
        "regularization_ccv_scheduler": "epoch_linear",
        "regularization_ccv_warmup_epochs": 1,
        "regularization_ccv_ramp_epochs": 4,
        "num_epochs": 8,
        "steps_per_epoch": 4,
    }, initial_skill="Initial rule.", adapter=adapter)
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, first = regularization.evaluate(
            old_skill="Initial rule.", candidate_skill="Initial rule. Better.",
            candidate_score=0.7, current_score=0.6,
            best_skill="Initial rule.", best_score=0.6, best_step=0,
            global_step=1, seed=41, out_dir=tmp_dir,
        )
        _, second = regularization.evaluate(
            old_skill="Initial rule.", candidate_skill="Initial rule. Better.",
            candidate_score=0.7, current_score=0.6,
            best_skill="Initial rule.", best_score=0.6, best_step=0,
            global_step=5, seed=42, out_dir=tmp_dir,
        )
    assert first["ccv"]["status"] == "scheduled_off"
    assert first["ccv_schedule"]["active"] is False
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["source_size"] == 1
    assert adapter.calls[0]["max_attacks"] == 1
    assert second["ccv_schedule"]["active"] is True


def test_adapter_ccv_preserves_provider_name() -> None:
    class RecordingAdapter:
        def __init__(self) -> None:
            self.kwargs = {}

        def run_ccv(self, **kwargs):
            self.kwargs = kwargs
            return {"status": "no_attack"}

        def get_verifier_contract(self):
            return {
                "id": "public-test-verifier",
                "normalization": ["lowercase strings"],
                "equivalences": ["null equals empty"],
            }

    adapter = RecordingAdapter()
    result = run_adapter_ccv(
        adapter,
        old_skill="old",
        candidate_skill="candidate",
        step=1,
        seed=101,
        out_dir="unused",
        source_split="valid_seen",
        source_size=1,
        max_patterns=1,
        attacks_per_pattern=1,
        max_attacks=1,
        history_items=0,
        attacker_model="claude-opus-4-8-thinking",
        reasoning_effort="high",
        builder_reasoning_effort="low",
        max_completion_tokens=12000,
        timeout=300,
        retries=1,
        provider="claude_agent_sdk",
    )
    assert result.status == "no_attack"
    assert adapter.kwargs["provider"] == "claude_agent_sdk"
    assert adapter.kwargs["verifier_contract"]["id"] == "public-test-verifier"
    assert resolve_verifier_contract(adapter)["public"] is True


def test_complexity_reduction_bonus_can_break_a_val_tie() -> None:
    gate, decision = evaluate_regularized_gate(
        candidate_skill="short", candidate_score=0.60,
        current_skill="long", current_score=0.60,
        best_skill="long", best_score=0.60, best_step=0, global_step=8,
        normalized_complexity_growth=0.0, lambda_complex=0.10,
        placeholder_free=True, ccv=CCVResult(status="deferred_epoch"),
        tau_adv=0.0, ccv_required=True,
        normalized_complexity_reduction=0.20, complexity_bonus=0.01,
    )
    assert decision.accepted
    assert decision.regularization_score == 0.002
    assert gate.action == "accept"


def test_complexity_reduction_bonus_never_buys_a_val_regression() -> None:
    gate, decision = evaluate_regularized_gate(
        candidate_skill="short", candidate_score=0.599,
        current_skill="long", current_score=0.60,
        best_skill="long", best_score=0.60, best_step=0, global_step=8,
        normalized_complexity_growth=0.0, lambda_complex=0.10,
        placeholder_free=True, ccv=CCVResult(status="deferred_epoch"),
        tau_adv=0.0, ccv_required=True,
        normalized_complexity_reduction=0.20, complexity_bonus=0.01,
    )
    assert not decision.accepted
    assert "validation_regression" in decision.reasons
    assert gate.action == "reject"


def test_paired_selection_repeat_scores_are_averaged_independently() -> None:
    candidate = [
        {"hard": 0.60, "soft": 0.70},
        {"hard": 0.56, "soft": 0.62},
    ]
    incumbent = [
        {"hard": 0.54, "soft": 0.58},
        {"hard": 0.52, "soft": 0.56},
    ]
    candidate_mean = _mean_repeat_scores(candidate)
    incumbent_mean = _mean_repeat_scores(incumbent)
    assert abs(candidate_mean[0] - 0.58) < 1e-12
    assert abs(candidate_mean[1] - 0.66) < 1e-12
    assert abs(incumbent_mean[0] - 0.53) < 1e-12
    assert abs(incumbent_mean[1] - 0.57) < 1e-12


def test_epoch_feedback_requires_all_train_trajectory_pages() -> None:
    runtime = EpochFeedbackRuntime(
        {"train_trajectories": [{"id": str(i)} for i in range(5)]},
        out_dir=".", page_size=2,
    )
    runtime.page(runtime.trajectories, 1, trajectories=True)
    rejected = runtime.submit({
        "failure_modes": [], "successful_patterns": [],
        "next_epoch_guidance": {"strategy": "compress"},
    })
    assert rejected["status"] == "rejected_incomplete_coverage"
    runtime.page(runtime.trajectories, 2, trajectories=True)
    runtime.page(runtime.trajectories, 3, trajectories=True)
    accepted = runtime.submit({
        "failure_modes": [], "successful_patterns": [],
        "next_epoch_guidance": {"strategy": "compress"},
    })
    assert accepted["status"] == "accepted"
    assert runtime.submitted["evidence_coverage"]["train_trajectories_read"] == 5


def test_epoch_feedback_exposes_only_verifier_confirmed_ccv_regressions() -> None:
    runtime = EpochFeedbackRuntime(
        {
            "train_trajectories": [],
            "transaction": {"ccv": {"attack_records": [{
                "status": "old_failed_skip_candidate",
                "pattern": {"true_invariant": "misleading StyleProxy story"},
            }]}},
            "ccv": {
                "status": "complete",
                "generated_attacks": 4,
                "paired_evaluated_attacks": 2,
                "attack_records": [
                    {
                        "attack_id": "style", "source_id": "51249",
                        "status": "old_failed_skip_candidate",
                        "pattern": {
                            "delta_evidence": ["removed safe style copying"],
                            "true_invariant": "misleading StyleProxy story",
                        },
                        "old_result": {"hard": 0},
                    },
                    {"attack_id": "invalid", "status": "invalid_attack"},
                    {
                        "attack_id": "both", "source_id": "32902",
                        "status": "both_pass", "old_result": {"hard": 1},
                        "candidate_result": {"hard": 1},
                    },
                    {
                        "attack_id": "scientific", "source_id": "585-41",
                        "status": "update_regression",
                        "old_result": {"hard": 1},
                        "candidate_result": {
                            "hard": 0, "fail_reason": "gt=1e-07 pred=107",
                        },
                        "candidate_clean_result": {"hard": 1},
                        "pattern": {
                            "delta_evidence": ["removed safe type conversion"],
                            "true_invariant": "preserve numeric values",
                        },
                        "generation": {"modified_locations": ["Sheet1!E1"]},
                    },
                ],
            },
        }, out_dir=".", page_size=10,
    )
    overview = runtime.overview()
    encoded = json.dumps(overview)
    assert "removed safe type conversion" in encoded
    assert "misleading StyleProxy story" not in encoded
    assert "removed safe style copying" not in encoded
    facts = overview["scap_verifier_facts"]
    assert facts["confirmed_regression_count"] == 1
    assert facts["confirmed_regressions"][0]["source_id"] == "585-41"
    assert facts["record_status_counts"] == {
        "old_failed_skip_candidate": 1,
        "invalid_attack": 1,
        "both_pass": 1,
        "update_regression": 1,
    }

    accepted = runtime.submit({
        "failure_modes": [], "successful_patterns": [],
        "scap_response": "StyleProxy was verifier-confirmed",
        "next_epoch_guidance": {"strategy": "preserve safe conversion"},
    })
    assert accepted["status"] == "accepted"
    assert "StyleProxy" not in runtime.submitted["scap_response"]
    assert "585-41" in runtime.submitted["scap_response"]
    assert "1e-07 pred=107" in runtime.submitted["scap_response"]


def test_epoch_ccv_granularity_defers_step_attack() -> None:
    class Adapter:
        def run_ccv(self, **kwargs):
            raise AssertionError("per-step CCV must not run")

    regularization = SkillEvoReg({
        "regularization_enabled": True, "regularization_segmenter": "rule",
        "regularization_dropout_enabled": False, "regularization_ccv_enabled": True,
        "regularization_ccv_granularity": "epoch",
    }, initial_skill="Initial rule.", adapter=Adapter())
    with tempfile.TemporaryDirectory() as out_dir:
        gate, telemetry = regularization.evaluate(
            old_skill="Initial rule.", candidate_skill="Initial rule. Better rule.",
            candidate_score=0.8, current_score=0.6, best_skill="Initial rule.",
            best_score=0.6, best_step=0, global_step=1, seed=7, out_dir=out_dir,
        )
    assert telemetry["ccv"]["status"] == "deferred_epoch"
    assert gate.action in {"accept", "accept_new_best"}
