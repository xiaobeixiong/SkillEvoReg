from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import uuid
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skillevolbench.components.online_regularization import (
    DropoutRoundTripError,
    PLACEHOLDER_PREFIX,
    clean_patch_placeholders,
    complexity_budget,
    complexity_instruction,
    complexity_within_budget,
    hard_complexity_instruction,
    mask_skill_texts,
    materialize_training_skill_view,
    measure_family_complexity,
    measure_patch_complexity,
    project_patch_to_existing_family,
    local_edit_complexity_telemetry,
    local_edit_trust_region_errors,
    redact_masked_atom_bodies,
    restore_patch_placeholders,
    rule_segment,
)
from skillevolbench.components.local_repair import (
    apply_replacements,
    editable_regions,
    regularization_repair_regions,
    new_exact_duplicate_atom_errors,
    protected_content_hash,
    validate_delta_operations,
    validate_repaired_document,
)
from skillevolbench.components.ccv import SkillEvolBenchCCVAdapter
from skillevolbench.components.skill_author import (
    LiteLLMClient,
    PatchGenerationFailure,
    SkillAuthor,
)
from skillevolbench.components.skill_document import validate_skill_document
from skillevolbench.components.regularization_consumption import (
    build_regularization_signal_bundle,
    regularization_update_instruction,
)
from skillevolbench.components.freeze_controller import LibraryFreezeController
from skillevolbench.schemas import (
    BaselineConfig,
    OperationType,
    RunConfig,
    SkillPatch,
    StrategyConfig,
)
from skillevolbench.baselines.runtime import BaselineRuntime
from skillevolbench.harbor_ext.hooks import SkillEvolBenchHooks, _ccv_side
from skillevolbench.harbor_ext._patches import _refresh_instruction
from skillevolbench.harbor_ext.retry import (
    RETRYABLE_INFRASTRUCTURE_EXCEPTIONS,
    is_retryable_infrastructure_exception,
)
from skillevolbench.metrics.revision_safety import compute_revision_safety
from skillevolbench.metrics.cost import compute_cost
from skillevolbench.metrics.task_success import compute_task_success
from skillevolbench.discovery import (
    TaskRegistry,
    default_skills_root,
    default_tasks_root,
)
from skillevolbench.scheduler import compute_task_order, assert_order_invariants
from skillevolbench.schemas import EnvOrders
from skillevolbench.stores import EventStore, LibraryStore
from skillevolbench.strategies.base import ApplyPatch, EvolutionContext, NoOp
from skillevolbench.strategies.chain import ChainEvolution
from scripts.run import _build_run_config


SKILL = """---
name: diagnose
description: Diagnose failures systematically.
---

# Workflow

1. Read the error log before editing code.
2. Reproduce the failure under realistic pressure.
3. Fix the root cause; never swallow the exception.
4. Run both outcome and process checks.
"""


def test_local_edit_trust_region_rejects_historical_anchor_undershoot() -> None:
    errors = local_edit_trust_region_errors(
        abnormal_modules=["complexity"],
        measured={"total": 17.93, "atom_count": 41},
        anchor={"total": 23.92, "atom_count": 52},
        current_texts={"E4-LS4.skill": SKILL},
        repaired_texts={"E4-LS4.skill": SKILL},
        repair_scope_skill_ids=["E4-LS4.skill"],
    )
    assert errors == [
        "repair undershot the pre-update complexity anchor; remove only the "
        "source patch excess"
    ]


def test_local_edit_complexity_direction_is_telemetry_not_a_gate() -> None:
    errors = local_edit_trust_region_errors(
        abnormal_modules=["complexity"],
        measured={"total": 25.0, "atom_count": 50},
        anchor=None,
        current_texts={"E1-LS1.skill": SKILL},
        repaired_texts={"E1-LS1.skill": SKILL},
        repair_scope_skill_ids=["E1-LS1.skill"],
        source_candidate_measurement={"total": 24.0, "atom_count": 49},
    )
    assert errors == []
    telemetry = local_edit_complexity_telemetry(
        abnormal_modules=["complexity"],
        measured={"total": 25.0, "atom_count": 50, "token_count": 900},
        source_candidate_measurement={
            "total": 24.0, "atom_count": 49, "token_count": 880,
        },
    )
    assert telemetry == {
        "source_candidate_total": 24.0,
        "repair_total": 25.0,
        "delta_total": 1.0,
        "source_candidate_atom_count": 49,
        "repair_atom_count": 50,
        "delta_atom_count": 1,
        "source_candidate_token_count": 880,
        "repair_token_count": 900,
        "delta_token_count": 20,
        "direction": "increased",
        "acceptance_gate": False,
    }


def test_ccv_only_trust_region_allows_one_atom_not_whole_skill_compression() -> None:
    current = "# Skill\n\n- A.\n- B.\n- C.\n- D.\n"
    repaired = "# Skill\n\n- One replacement.\n"
    errors = local_edit_trust_region_errors(
        abnormal_modules=["ccv"],
        measured={},
        anchor=None,
        current_texts={"E1-LS1.skill": current},
        repaired_texts={"E1-LS1.skill": repaired},
        repair_scope_skill_ids=["E1-LS1.skill"],
    )
    assert errors and "CCV-only repair deleted more than" in errors[0]


def test_model_preset_persists_effective_same_model_routing(
    tmp_path, monkeypatch,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("YIBU_API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("YIBU_API_KEY", "test-only")
    args = argparse.Namespace(
        baseline_name=None,
        baseline=repo / "configs/baselines/skillevoreg_local_edit.yaml",
        strategy=repo / "configs/strategies/chain_skillevoreg_local_edit.yaml",
        model_yaml=repo / "configs/models/gpt-5.4-mini-yibu.yaml",
        run_id="routing_audit_test",
        order_seed="A",
        workspace_root=str(tmp_path),
        api_base=None,
        api_key_env_var="ANTHROPIC_API_KEY",
        dry_run=True,
        max_tasks=1,
    )
    config = _build_run_config(args)
    expected = "openai/gpt-5.4-mini"
    assert config.strategy.author_model == expected
    assert config.effective_model_routing == {
        "task_agent": expected,
        "skill_author": expected,
        "retriever": expected,
        "ccv_probe_attacker": expected,
        "ccv_old_candidate_executor": expected,
        "guard_feedback": "deterministic_rules",
        "regularization_local_edit": expected,
    }
    persisted = config.model_dump(mode="json")
    assert persisted["strategy"]["author_model"] == expected
    assert persisted["effective_model_routing"] == config.effective_model_routing
    assert config.baseline.agent_kwargs["reasoning_effort"] == "high"
    assert config.effective_reasoning_effort == {
        "task_agent": "high",
        "skill_author": "high",
        "retriever": "high",
        "ccv_probe_attacker": "high",
        "ccv_old_candidate_executor": "high",
        "regularization_local_edit": "high",
    }
    assert persisted["effective_reasoning_effort"] == (
        config.effective_reasoning_effort
    )


def test_medium_model_preset_applies_to_every_active_role(
    tmp_path, monkeypatch,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("YIBU_API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("YIBU_API_KEY", "test-only")
    args = argparse.Namespace(
        baseline_name=None,
        baseline=repo / "configs/baselines/skillevoreg_local_edit.yaml",
        strategy=repo / "configs/strategies/chain_skillevoreg_local_edit.yaml",
        model_yaml=repo / "configs/models/gpt-5.4-mini-yibu-medium.yaml",
        run_id="medium_routing_audit_test",
        order_seed="A",
        workspace_root=str(tmp_path),
        api_base=None,
        api_key_env_var="ANTHROPIC_API_KEY",
        dry_run=True,
        max_tasks=1,
    )
    config = _build_run_config(args)
    assert config.baseline.agent_kwargs["reasoning_effort"] == "medium"
    assert {
        value for value in config.effective_reasoning_effort.values()
        if value not in {"disabled", "not_applicable"}
    } == {"medium"}


def test_litellm_client_sends_high_reasoning_effort_sync_and_async(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
        usage=_Usage(),
    )

    def completion(**kwargs):
        calls.append(kwargs)
        return response

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return response

    fake_litellm = SimpleNamespace(
        completion=completion,
        acompletion=acompletion,
        drop_params=None,
        num_retries=None,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    client = LiteLLMClient(
        model="openai/gpt-5.4-mini",
        reasoning_effort="high",
    )
    client("sync")
    asyncio.run(client.acall("async"))
    assert len(calls) == 2
    assert all(call["reasoning_effort"] == "high" for call in calls)


def test_rule_segment_is_exact_cover() -> None:
    atoms = rule_segment(SKILL)
    assert len(atoms) >= 4
    assert "".join(atom.span for atom in atoms) == SKILL


def test_dropout_masks_prompt_only_and_roundtrip_restores_original_atom() -> None:
    masked, report = mask_skill_texts(
        {"E1-LS1.diagnose": SKILL},
        probability=0.01,
        seed=7,
        max_atoms=4,
        min_eligible_atoms=4,
        ensure_at_least_one=True,
    )
    assert report["masked_count"] == 1
    assert PLACEHOLDER_PREFIX in masked["E1-LS1.diagnose"]
    assert PLACEHOLDER_PREFIX not in SKILL

    patch = SkillPatch(
        patch_id=str(uuid.uuid4()),
        summary="regularization cleanup",
        upsert_files={"diagnose/SKILL.md": masked["E1-LS1.diagnose"]},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    restored, metadata = restore_patch_placeholders(patch, report)
    assert metadata["restored_count"] == 1
    assert metadata["preservation_ok"] is True
    assert restored.upsert_files["diagnose/SKILL.md"] == SKILL
    assert PLACEHOLDER_PREFIX not in restored.upsert_files["diagnose/SKILL.md"]


def test_dropout_roundtrip_removes_synthetic_closing_marker() -> None:
    masked, report = mask_skill_texts(
        {"E1-LS1.diagnose": SKILL},
        probability=0.01,
        seed=7,
        max_atoms=4,
        min_eligible_atoms=4,
        ensure_at_least_one=True,
    )
    proposal = masked["E1-LS1.diagnose"] + "\n</MASKED_SKILL_ATOM>\n"
    patch = SkillPatch(
        patch_id=str(uuid.uuid4()),
        summary="synthetic closing marker",
        upsert_files={"diagnose/SKILL.md": proposal},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )

    restored, metadata = restore_patch_placeholders(patch, report)

    assert metadata["preservation_ok"] is True
    assert "MASKED_SKILL_ATOM" not in restored.upsert_files["diagnose/SKILL.md"]
    assert validate_skill_document(restored.upsert_files["diagnose/SKILL.md"]) == []


def test_closing_dropout_marker_is_blocked_from_persistence() -> None:
    text = SKILL + "\n</MASKED_SKILL_ATOM>\n"
    assert any(
        "dropout placeholder" in error
        for error in validate_skill_document(text)
    )
    patch = SkillPatch(
        patch_id=str(uuid.uuid4()),
        summary="closing marker must not persist",
        upsert_files={"diagnose/SKILL.md": text},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    with pytest.raises(DropoutRoundTripError, match="dropout marker"):
        clean_patch_placeholders(patch)


def test_dropout_roundtrip_rejects_missing_or_duplicate_marker() -> None:
    masked, report = mask_skill_texts(
        {"E1-LS1.diagnose": SKILL},
        probability=0.01,
        seed=7,
        max_atoms=4,
        min_eligible_atoms=4,
        ensure_at_least_one=True,
    )
    marker = report["masked_atoms"][0]["marker"]
    base = dict(
        patch_id=str(uuid.uuid4()),
        summary="bad roundtrip",
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    missing = SkillPatch(
        **base,
        upsert_files={"diagnose/SKILL.md": masked["E1-LS1.diagnose"].replace(marker, "")},
    )
    with pytest.raises(DropoutRoundTripError, match="exactly once"):
        restore_patch_placeholders(missing, report)

    duplicate = SkillPatch(
        **{**base, "patch_id": str(uuid.uuid4())},
        upsert_files={"diagnose/SKILL.md": masked["E1-LS1.diagnose"] + marker},
    )
    with pytest.raises(DropoutRoundTripError, match="found 2"):
        restore_patch_placeholders(duplicate, report)

    moved_text = masked["E1-LS1.diagnose"].replace(marker, "") + marker
    moved = SkillPatch(
        **{**base, "patch_id": str(uuid.uuid4())},
        upsert_files={"diagnose/SKILL.md": moved_text},
    )
    with pytest.raises(DropoutRoundTripError, match="moved or changed"):
        restore_patch_placeholders(moved, report)


def test_dropout_roundtrip_allows_edits_to_adjacent_visible_atoms() -> None:
    masked, report = mask_skill_texts(
        {"E1-LS1.diagnose": SKILL},
        probability=0.01,
        seed=7,
        max_atoms=4,
        min_eligible_atoms=4,
        ensure_at_least_one=True,
    )
    marker = report["masked_atoms"][0]["marker"]
    proposal = masked["E1-LS1.diagnose"]
    proposal = proposal.replace(
        "# Workflow", "# Reliable workflow", 1,
    ).replace(
        "4. Run both outcome and process checks.",
        "4. Run focused checks, then the complete relevant suite.",
        1,
    )
    assert proposal.count(marker) == 1
    patch = SkillPatch(
        patch_id=str(uuid.uuid4()),
        summary="edit visible neighbours",
        upsert_files={"diagnose/SKILL.md": proposal},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )

    restored, metadata = restore_patch_placeholders(patch, report)

    assert metadata["preservation_ok"] is True
    assert metadata["structural_anchor_mode"] == "atom_order"
    assert report["masked_atoms"][0]["original_span"] in (
        restored.upsert_files["diagnose/SKILL.md"]
    )
    assert "# Reliable workflow" in restored.upsert_files["diagnose/SKILL.md"]


def test_dropout_roundtrip_rejects_marker_embedded_in_visible_atom() -> None:
    masked, report = mask_skill_texts(
        {"E1-LS1.diagnose": SKILL},
        probability=0.01,
        seed=7,
        max_atoms=4,
        min_eligible_atoms=4,
        ensure_at_least_one=True,
    )
    marker = report["masked_atoms"][0]["marker"]
    proposal = masked["E1-LS1.diagnose"].replace(marker, "prefix " + marker)
    patch = SkillPatch(
        patch_id=str(uuid.uuid4()),
        summary="embed marker",
        upsert_files={"diagnose/SKILL.md": proposal},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )

    with pytest.raises(DropoutRoundTripError, match="standalone atom slot"):
        restore_patch_placeholders(patch, report)


def test_manifest_versions_only_materially_changed_skills(tmp_path) -> None:
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.relevant", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    sibling = SKILL.replace("name: diagnose", "name: sibling")
    library.create_skill(
        skill_id="E1-LS1.sibling", content=sibling, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    patch = SkillPatch(
        patch_id=str(uuid.uuid4()), summary="change only relevant",
        upsert_files={
            "relevant/SKILL.md": SKILL + "\n- New transferable check.\n",
            "sibling/SKILL.md": sibling,
        },
        target_skill_ids=["E1-LS1.relevant", "E1-LS1.sibling"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    result = library.apply_patch(patch, "E1-LS1-T2", "chain")
    manifest = library._load_manifest()
    assert result.affected_skill_ids == ["E1-LS1.relevant"]
    assert manifest.skills["E1-LS1.relevant"].current_version == 2
    assert manifest.skills["E1-LS1.sibling"].current_version == 1


def test_skill_author_repairs_unbalanced_fence_in_ordinary_revision() -> None:
    broken = SKILL + "\n```text\norphan fence\n"
    raw = __import__("json").dumps({
        "summary": "broken",
        "operation_type": "revise",
        "upsert_files": {"diagnose/SKILL.md": broken},
        "delete_paths": [],
    })
    patch = SkillAuthor._parse_revision_patch(
        raw=raw,
        target_skill_ids=["E1-LS1.diagnose"],
        triggered_by_task="E1-LS1-T2",
        failure_summary="",
        proposing_mode="minimal_edit",
        attempt_count=1,
    )
    repaired = patch.upsert_files["diagnose/SKILL.md"]
    assert repaired.startswith(broken)
    assert repaired.endswith("```\n")
    assert validate_skill_document(repaired) == []


def test_skill_author_repairs_fence_before_dropout_roundtrip() -> None:
    masked_texts, report = mask_skill_texts(
        {"E1-LS1.diagnose": SKILL},
        probability=1e-12,
        seed=97,
        max_atoms=1,
        min_eligible_atoms=1,
        ensure_at_least_one=True,
    )
    masked = masked_texts["E1-LS1.diagnose"]
    broken = masked + "\n```text\nnew visible diagnostic note\n"
    raw = __import__("json").dumps({
        "summary": "repair both structures",
        "operation_type": "revise",
        "upsert_files": {"diagnose/SKILL.md": broken},
        "delete_paths": [],
    })
    patch = SkillAuthor._parse_revision_patch(
        raw=raw,
        target_skill_ids=["E1-LS1.diagnose"],
        triggered_by_task="E1-LS1-T2",
        failure_summary="",
        proposing_mode="minimal_edit",
        attempt_count=1,
    )
    restored, metadata = restore_patch_placeholders(patch, report)
    text = restored.upsert_files["diagnose/SKILL.md"]
    assert metadata["preservation_ok"] is True
    assert PLACEHOLDER_PREFIX not in text
    assert "new visible diagnostic note" in text
    assert validate_skill_document(text) == []


def test_skill_author_allows_temporary_dropout_marker_but_store_refuses_it(
    tmp_path,
) -> None:
    masked_texts, report = mask_skill_texts(
        {"E1-LS1.diagnose": SKILL},
        probability=1e-12,
        seed=91,
        max_atoms=1,
        min_eligible_atoms=1,
        ensure_at_least_one=True,
    )
    marker = report["masked_atoms"][0]["marker"]
    masked = masked_texts["E1-LS1.diagnose"]
    raw = __import__("json").dumps({
        "summary": "masked training proposal",
        "operation_type": "revise",
        "upsert_files": {"diagnose/SKILL.md": masked},
        "delete_paths": [],
    })
    patch = SkillAuthor._parse_revision_patch(
        raw=raw,
        target_skill_ids=["E1-LS1.diagnose"],
        triggered_by_task="E1-LS1-T2",
        failure_summary="",
        proposing_mode="minimal_edit",
        attempt_count=1,
    )
    assert marker in patch.upsert_files["diagnose/SKILL.md"]

    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    with pytest.raises(ValueError, match="dropout placeholder"):
        library.apply_patch(patch, "E1-LS1-T2", "chain")

    restored, roundtrip = restore_patch_placeholders(patch, report)
    assert roundtrip["preservation_ok"] is True
    assert marker not in restored.upsert_files["diagnose/SKILL.md"]
    result = library.apply_patch(restored, "E1-LS1-T2", "chain")
    assert result is not None
    assert marker not in library.get_skill(
        "E1-LS1.diagnose"
    ).files["SKILL.md"]


def test_library_store_atomically_refuses_malformed_skill_patch(tmp_path) -> None:
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    before_hash = library.compute_hash()
    before_text = library.get_skill("E1-LS1.diagnose").files["SKILL.md"]
    patch = SkillPatch(
        patch_id=str(uuid.uuid4()), summary="broken ordinary revision",
        upsert_files={"diagnose/SKILL.md": SKILL + "\n```text\norphan\n"},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    with pytest.raises(ValueError, match="unbalanced Markdown"):
        library.apply_patch(patch, "E1-LS1-T2", "chain")
    assert library.compute_hash() == before_hash
    assert library.get_skill("E1-LS1.diagnose").files["SKILL.md"] == before_text


def test_complexity_regularizer_is_prompt_guidance_not_gate() -> None:
    guidance = complexity_instruction(
        {"total": 10.0, "token_count": 800, "atom_count": 20},
        complexity_lambda=0.15,
        free_growth_budget=0.10,
    )
    assert "proposal guidance" in guidance
    assert "10.0%" in guidance
    assert "weight 0.1500" in guidance


def test_hard_complexity_projection_restores_anchor_tree(tmp_path) -> None:
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    anchor = measure_family_complexity(library, "E1-LS1")
    oversized = SkillPatch(
        patch_id=str(uuid.uuid4()), summary="append many local rules",
        upsert_files={
            "diagnose/SKILL.md": SKILL + "\n" + "\n".join(
                f"- Special-case rule {index}: inspect local value {index}."
                for index in range(30)
            ) + "\n"
        },
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    budget = complexity_budget(
        anchor, max_relative_growth=0.15, max_atom_growth=4
    )
    oversized_measurement = measure_patch_complexity(
        library, "E1-LS1", oversized
    )
    assert not complexity_within_budget(oversized_measurement, budget)

    projected = project_patch_to_existing_family(
        library, "E1-LS1", oversized
    )
    projected_measurement = measure_patch_complexity(
        library, "E1-LS1", projected
    )
    assert projected_measurement == anchor
    assert complexity_within_budget(projected_measurement, budget)
    assert projected.delete_paths == []


def test_hard_complexity_prompt_blocks_fragmented_fake_compression() -> None:
    prompt = hard_complexity_instruction(
        {"max_total": 12.5, "max_atoms": 24},
        {"total": 18.0, "atom_count": 41},
        attempt=1,
    )
    assert "one general decision rule" in prompt
    assert "Merge adjacent bullets" in prompt
    assert "individual filenames, constants, test cases" in prompt
    assert "splitting prose into more headings, bullets, or short atoms" in prompt
    assert "atom count is a hard constraint" in prompt
    assert "smallest complete workflow" in prompt


def test_chain_reports_complexity_anomaly_without_inline_repair(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[1]
    baseline = BaselineConfig.from_yaml(
        repo / "configs/baselines/skillevoreg_online.yaml"
    )
    config = StrategyConfig.from_yaml(
        repo / "configs/strategies/chain_skillevoreg_online.yaml"
    ).model_copy(update={"regularization_dropout_probability": 0.0})
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    anchor = measure_family_complexity(library, "E1-LS1")
    events = EventStore(tmp_path / "events")
    events.record("proposal_regularization_applied", {
        "task_id": "E1-LS1-T1", "candidate_complexity": anchor,
    })
    oversized = SkillPatch(
        patch_id=str(uuid.uuid4()), summary="oversized",
        upsert_files={
            "diagnose/SKILL.md": SKILL + "\n" + "\n".join(
                f"- Local exception {index}: retain detail {index}."
                for index in range(30)
            ) + "\n"
        },
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )

    class OversizedEvolver:
        def __init__(self):
            self.calls = 0

        def propose(self, **_kwargs):
            self.calls += 1
            return oversized.model_copy(update={"patch_id": str(uuid.uuid4())})

    evolver = OversizedEvolver()
    chain = ChainEvolution(
        evolver=evolver, retriever=None, library=library,
        replay_store=None, event_store=events, config=config,
    )
    outcome = SimpleNamespace(
        task_id="E1-LS1-T2", verifier_passed=False,
        reward=0.0, failure_summary="failed",
    )
    ctx = EvolutionContext(
        task=SimpleNamespace(
            task_id="E1-LS1-T2", family_id="E1-LS1", role="enriched",
        ),
        outcome=outcome,
        compacted=SimpleNamespace(text="trace"),
        pre_retrieval=None,
        skills_actually_used=["E1-LS1.diagnose"],
        baseline=baseline,
        replay_store=None,
        library=library,
    )
    decision = chain.decide(ctx)
    assert isinstance(decision, ApplyPatch)
    projection = decision.regularization_metadata["complexity_projection"]
    assert projection["triggered"] is True
    assert projection["consumed_during_proposal"] is False
    assert evolver.calls == 1
    assert not complexity_within_budget(
        measure_patch_complexity(library, "E1-LS1", decision.patch),
        projection["budget"],
    )


def test_chain_consumes_exact_training_mask_and_keeps_other_context_visible(
    tmp_path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    baseline = BaselineConfig.from_yaml(
        repo / "configs/baselines/skillevoreg_online.yaml"
    )
    config = StrategyConfig.from_yaml(
        repo / "configs/strategies/chain_skillevoreg_online.yaml"
    ).model_copy(update={
        "regularization_dropout_probability": 1e-12,
        "regularization_dropout_ensure_at_least_one": True,
    })
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.relevant", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    sibling = SKILL.replace("name: diagnose", "name: sibling")
    library.create_skill(
        skill_id="E1-LS1.sibling", content=sibling, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    calls = []

    class CapturingEvolver:
        def propose(self, **kwargs):
            calls.append(kwargs)
            return SkillPatch(
                patch_id=str(uuid.uuid4()), summary="keep relevant",
                upsert_files={
                    "relevant/SKILL.md":
                    kwargs["skill_text_overrides"]["E1-LS1.relevant"]
                },
                target_skill_ids=["E1-LS1.relevant", "E1-LS1.sibling"],
                operation_type=OperationType.REVISE,
                triggered_by_task="E1-LS1-T2",
            )

    chain = ChainEvolution(
        evolver=CapturingEvolver(), retriever=None, library=library,
        replay_store=None, event_store=EventStore(tmp_path / "events"),
        config=config,
    )
    masked, dropout_report = mask_skill_texts(
        {"E1-LS1.relevant": SKILL}, probability=1.0, seed=17,
        max_atoms=1, min_eligible_atoms=1, ensure_at_least_one=True,
    )
    dropout_report["scope_skill_ids"] = ["E1-LS1.relevant"]
    dropout_report["mode"] = "training_rollout"
    execution_overrides = {"E1-LS1.sibling": sibling, **masked}
    ctx = EvolutionContext(
        task=SimpleNamespace(
            task_id="E1-LS1-T2", family_id="E1-LS1", role="enriched",
        ),
        outcome=SimpleNamespace(
            task_id="E1-LS1-T2", verifier_passed=True,
            reward=1.0, failure_summary="",
        ),
        compacted=SimpleNamespace(
            text="trace\n" + dropout_report["masked_atoms"][0]["original_span"]
        ),
        pre_retrieval=None,
        skills_actually_used=["E1-LS1.relevant"],
        baseline=baseline,
        replay_store=None,
        library=library,
        learning_pass=1,
        execution_dropout_report=dropout_report,
        execution_skill_text_overrides=execution_overrides,
    )

    decision = chain.decide(ctx)
    overrides = calls[0]["skill_text_overrides"]
    assert PLACEHOLDER_PREFIX in overrides["E1-LS1.relevant"]
    assert overrides["E1-LS1.sibling"] == sibling
    assert PLACEHOLDER_PREFIX not in overrides["E1-LS1.sibling"]
    assert decision.regularization_metadata["dropout"]["masked_count"] == 1
    assert decision.regularization_metadata["dropout"]["scope_skill_ids"] == [
        "E1-LS1.relevant"
    ]
    assert decision.regularization_metadata["dropout_roundtrip"]["restored_count"] == 1
    assert decision.patch.upsert_files["relevant/SKILL.md"] == SKILL
    assert dropout_report["masked_atoms"][0]["original_span"].strip() not in (
        calls[0]["compacted"].text
    )
    assert len(calls) == 1


def test_chain_retries_one_bad_dropout_roundtrip_then_restores(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[1]
    baseline = BaselineConfig.from_yaml(
        repo / "configs/baselines/skillevoreg_online.yaml"
    )
    config = StrategyConfig.from_yaml(
        repo / "configs/strategies/chain_skillevoreg_online.yaml"
    ).model_copy(update={
        "regularization_dropout_probability": 1e-12,
        "regularization_dropout_ensure_at_least_one": True,
    })
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.relevant", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )

    class RetryEvolver:
        def __init__(self):
            self.calls = 0

        def propose(self, **kwargs):
            self.calls += 1
            content = kwargs["skill_text_overrides"]["E1-LS1.relevant"]
            if self.calls == 1:
                content = content.replace(report_marker(content), "")
            return SkillPatch(
                patch_id=str(uuid.uuid4()), summary="retry marker",
                upsert_files={"relevant/SKILL.md": content},
                target_skill_ids=["E1-LS1.relevant"],
                operation_type=OperationType.REVISE,
                triggered_by_task="E1-LS1-T2",
            )

    def report_marker(text: str) -> str:
        start = text.index("<MASKED_SKILL_ATOM")
        return text[start:text.index(">", start) + 1]

    evolver = RetryEvolver()
    chain = ChainEvolution(
        evolver=evolver, retriever=None, library=library,
        replay_store=None, event_store=EventStore(tmp_path / "events"),
        config=config,
    )
    masked, dropout_report = mask_skill_texts(
        {"E1-LS1.relevant": SKILL}, probability=1.0, seed=19,
        max_atoms=1, min_eligible_atoms=1, ensure_at_least_one=True,
    )
    dropout_report["scope_skill_ids"] = ["E1-LS1.relevant"]
    dropout_report["mode"] = "training_rollout"
    ctx = EvolutionContext(
        task=SimpleNamespace(
            task_id="E1-LS1-T2", family_id="E1-LS1", role="enriched",
        ),
        outcome=SimpleNamespace(
            task_id="E1-LS1-T2", verifier_passed=True,
            reward=1.0, failure_summary="",
        ),
        compacted=SimpleNamespace(text="trace"),
        pre_retrieval=None,
        skills_actually_used=["E1-LS1.relevant"],
        baseline=baseline,
        replay_store=None,
        library=library,
        learning_pass=1,
        execution_dropout_report=dropout_report,
        execution_skill_text_overrides=masked,
    )
    decision = chain.decide(ctx)
    assert isinstance(decision, ApplyPatch)
    assert evolver.calls == 2
    assert decision.patch.upsert_files["relevant/SKILL.md"] == SKILL

    class AlwaysBadEvolver:
        def __init__(self):
            self.calls = 0

        def propose(self, **kwargs):
            self.calls += 1
            content = kwargs["skill_text_overrides"]["E1-LS1.relevant"]
            content = content.replace(report_marker(content), "")
            return SkillPatch(
                patch_id=str(uuid.uuid4()), summary="drop marker",
                upsert_files={"relevant/SKILL.md": content},
                target_skill_ids=["E1-LS1.relevant"],
                operation_type=OperationType.REVISE,
                triggered_by_task="E1-LS1-T2",
            )

    bad = AlwaysBadEvolver()
    before_hash = library.compute_hash()
    failed = ChainEvolution(
        evolver=bad, retriever=None, library=library,
        replay_store=None, event_store=EventStore(tmp_path / "events-bad"),
        config=config,
    ).decide(ctx)
    assert isinstance(failed, NoOp)
    assert failed.reason == "dropout_roundtrip_failed"
    assert bad.calls == 2
    assert library.compute_hash() == before_hash


def test_training_skill_view_hides_body_without_mutating_live_library(
    tmp_path,
) -> None:
    live = tmp_path / "live"
    skill_dir = live / "diagnose"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL)
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "check.py").write_text("print('ok')\n")
    masked, report = mask_skill_texts(
        {"E1-LS1.diagnose": SKILL}, probability=1.0, seed=23,
        max_atoms=1, min_eligible_atoms=1, ensure_at_least_one=True,
    )
    view = tmp_path / "training-view"
    materialize_training_skill_view(live, view, masked)

    row = report["masked_atoms"][0]
    view_text = (view / "diagnose" / "SKILL.md").read_text()
    assert row["marker"] in view_text
    assert row["original_span"] not in view_text
    assert (skill_dir / "SKILL.md").read_text() == SKILL
    assert (view / "diagnose" / "scripts" / "check.py").is_file()


def test_training_dropout_redacts_hidden_body_from_old_trajectory() -> None:
    masked, report = mask_skill_texts(
        {"E1-LS1.diagnose": SKILL}, probability=1.0, seed=29,
        max_atoms=1, min_eligible_atoms=1, ensure_at_least_one=True,
    )
    del masked
    row = report["masked_atoms"][0]
    trace = f"PASS-1 TRACE\n{row['original_span']}\nEND"
    redacted, count = redact_masked_atom_bodies(trace, report)
    assert count >= 1
    assert row["original_span"].strip() not in redacted
    assert row["marker"] in redacted


def test_hook_prepares_exact_training_mask_without_body_leak(tmp_path) -> None:
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    before_hash = library.compute_hash()
    events = EventStore(tmp_path / "events")
    runtime = SimpleNamespace(
        strategy_config=SimpleNamespace(
            regularization_proposal_regularization=True,
            regularization_dropout_enabled=True,
            regularization_start_learning_pass=2,
            regularization_dropout_probability=1e-12,
            regularization_dropout_max_atoms=4,
            regularization_dropout_min_eligible_atoms=1,
            regularization_dropout_ensure_at_least_one=True,
        ),
        library=library,
        run_root=tmp_path,
        event_store=events,
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    report, overrides = hooks._prepare_training_dropout_view(
        SimpleNamespace(
            task_id="E1-LS1-T2", family_id="E1-LS1", role="enriched",
        ),
        "E1-LS1-T2__pass2",
        2,
    )

    assert report is not None and overrides is not None
    assert report["masked_count"] == 1
    row = report["masked_atoms"][0]
    masked = overrides["E1-LS1.diagnose"]
    assert row["marker"] in masked
    assert row["original_span"] not in masked
    view_text = (
        tmp_path / "runtime" / "E1-LS1-T2__pass2"
        / "training-skill-view" / "diagnose" / "SKILL.md"
    ).read_text()
    assert view_text == masked
    assert library.compute_hash() == before_hash
    event = events.events_of_type("training_dropout_view_prepared")[-1]
    serialized = str(event)
    assert row["original_span"] not in serialized
    assert "original_span" not in serialized


def test_hook_never_prepares_training_dropout_for_replay(tmp_path) -> None:
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    events = EventStore(tmp_path / "events")
    runtime = SimpleNamespace(
        strategy_config=SimpleNamespace(
            regularization_proposal_regularization=True,
            regularization_dropout_enabled=True,
            regularization_start_learning_pass=2,
            regularization_dropout_probability=1.0,
            regularization_dropout_max_atoms=4,
            regularization_dropout_min_eligible_atoms=1,
            regularization_dropout_ensure_at_least_one=True,
        ),
        library=library,
        run_root=tmp_path,
        event_store=events,
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)

    report, overrides = hooks._prepare_training_dropout_view(
        SimpleNamespace(
            task_id="E1-LS1-T2", family_id="E1-LS1", role="enriched",
        ),
        "E1-LS1-T2__pass2__replay",
        2,
        is_replay=True,
    )

    assert report is None
    assert overrides is None
    assert not (
        tmp_path / "runtime" / "E1-LS1-T2__pass2__replay"
        / "training-skill-view"
    ).exists()
    assert events.events_of_type("training_dropout_view_prepared") == []


def test_environment_switch_rebinds_strategy_library(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[1]
    baseline = BaselineConfig.from_yaml(
        repo / "configs/baselines/skillevoreg_online.yaml"
    )
    strategy = StrategyConfig.from_yaml(
        repo / "configs/strategies/chain_skillevoreg_online.yaml"
    )
    assert baseline.allow_rollback is False
    assert strategy.regularization_proposal_regularization is True
    assert strategy.regularization_dropout_probability == 0.05
    assert strategy.regularization_complexity_lambda == 0.15
    assert strategy.regularization_free_growth == 0.10
    assert strategy.regularization_hard_complexity_enabled is True
    assert strategy.regularization_hard_max_relative_growth == 0.15
    assert strategy.regularization_hard_max_atom_growth == 4
    assert strategy.regularization_ccv_min_old_score == 0.80
    assert strategy.regularization_ccv_min_regression_drop == 0.05
    assert strategy.regularization_consumption_mode == "deferred"
    assert not hasattr(strategy, "online_episode_gate")
    config = RunConfig(
        run_id="switch-test",
        baseline=baseline,
        strategy=strategy,
        workspace_root=tmp_path,
    )
    runtime = BaselineRuntime.build(
        config,
        llm_author_call=lambda _prompt: "{}",
    )
    runtime.switch_env("E1")
    assert runtime.strategy.library is runtime.library


def test_local_edit_has_distinct_valid_run_config(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[1]
    baseline = BaselineConfig.from_yaml(
        repo / "configs/baselines/skillevoreg_local_edit.yaml"
    )
    strategy = StrategyConfig.from_yaml(
        repo / "configs/strategies/chain_skillevoreg_local_edit.yaml"
    )
    config = RunConfig(
        run_id="local-edit-config-test",
        baseline=baseline,
        strategy=strategy,
        workspace_root=tmp_path,
    )
    assert config.strategy.regularization_consumption_mode == "local_edit"
    assert config.baseline.default_strategy == config.strategy.name
    assert config.strategy.shared_t1_anchor_required is False
    assert config.strategy.shared_t1_skip_execution is False




def test_shared_t1_schedule_skips_only_original_t1() -> None:
    repo = Path(__file__).resolve().parents[1]
    registry = TaskRegistry.from_disk(
        default_skills_root(), default_tasks_root()
    )
    env_orders = EnvOrders.from_yaml(repo / "configs/env_orders.yaml")
    order = compute_task_order(
        registry,
        env_orders,
        "A",
        within_env_replay=True,
        replay_eval=False,
        skip_original_t1=True,
    )
    assert len(order) == 240
    assert not any(
        not record.is_replay and record.spec.role.value == "canonical"
        for record in order
    )
    assert sum(
        record.is_replay and record.spec.role.value == "canonical"
        for record in order
    ) == 30
    assert_order_invariants(order, skip_original_t1=True)


def test_two_pass_schedule_repeats_learning_before_eval() -> None:
    repo = Path(__file__).resolve().parents[1]
    registry = TaskRegistry.from_disk(
        default_skills_root(), default_tasks_root()
    )
    env_orders = EnvOrders.from_yaml(repo / "configs/env_orders.yaml")
    order = compute_task_order(
        registry,
        env_orders,
        "A",
        within_env_replay=True,
        replay_eval=False,
        learning_passes=2,
    )
    assert len(order) == 360
    e1_originals = [
        row for row in order
        if row.spec.environment_id == "E1" and not row.is_replay
    ]
    assert [row.learning_pass for row in e1_originals[:15]] == [1] * 15
    assert [row.learning_pass for row in e1_originals[15:30]] == [2] * 15
    assert all(
        row.spec.role.value in {"context-shift", "adversarial", "composition"}
        for row in e1_originals[30:]
    )
    assert_order_invariants(order, learning_passes=2)




def test_ccv_adapter_preserves_original_and_adds_bounded_appendix() -> None:
    response = {
        "decision": "probe",
        "attack_type": "ordering_dependency",
        "targeted_assumption": "Candidate overweights one log ordering.",
        "predicted_failure_region": "diagnosis before reproduction",
        "goal_preservation": "The requested fix and verifier stay unchanged.",
        "appendix": "A timestamp in the supplied log may be delayed; treat it as contextual evidence only.",
    }
    adapter = SkillEvolBenchCCVAdapter(
        lambda _prompt, **_kwargs: __import__("json").dumps(response),
        max_appendix_chars=300,
    )
    original = "Fix the race without serializing all requests."
    probe = adapter.generate(
        original_instruction=original, old_skill="old", candidate_skill="new",
        task_id="E1-LS1-T2",
    )
    assert probe.status == "ready"
    assert probe.attack_type == "ordering_dependency"
    assert probe.generation_mode == "llm"
    assert probe.generation_attempts == 1
    materialized = adapter.materialize(original, probe)
    assert materialized.startswith(original)
    assert "timestamp" in materialized


def test_ccv_adapter_limits_responsibility_to_changed_skills() -> None:
    response = {
        "decision": "probe",
        "attack_type": "missing_branch",
        "targeted_assumption": "Candidate narrows one branch.",
        "predicted_failure_region": "alternate branch",
        "responsible_skill_ids": ["E1-LS1.changed", "E1-LS1.untouched"],
        "goal_preservation": "The original goal remains unchanged.",
        "appendix": "Inputs may exercise another documented branch.",
    }
    adapter = SkillEvolBenchCCVAdapter(
        lambda _prompt, **_kwargs: __import__("json").dumps(response)
    )
    probe = adapter.generate(
        original_instruction="Fix it.", old_skill="old", candidate_skill="new",
        task_id="E1-LS1-T3",
        candidate_changed_skill_ids=["E1-LS1.changed"],
    )
    assert probe.status == "ready"
    assert probe.responsible_skill_ids == ("E1-LS1.changed",)


def test_ccv_adapter_rejects_answer_or_hidden_test_language() -> None:
    response = {
        "decision": "probe", "attack_type": "over_specificity",
        "targeted_assumption": "x",
        "predicted_failure_region": "x", "goal_preservation": "x",
        "appendix": "Read the hidden test and copy the expected patch.",
    }
    adapter = SkillEvolBenchCCVAdapter(
        lambda _prompt, **_kwargs: __import__("json").dumps(response)
    )
    probe = adapter.generate(
        original_instruction="Fix it.", old_skill="old", candidate_skill="new",
        task_id="E1-LS1-T2",
    )
    assert probe.status == "inconclusive_generation_failed"
    assert probe.generation_mode == "llm_failed"
    assert probe.generation_attempts == 3
    assert probe.appendix == ""
    assert "forbidden_goal_or_answer_language" in probe.error


def test_ccv_adapter_rejects_rephrased_verifier_feedback() -> None:
    response = {
        "decision": "probe", "attack_type": "tool_dependency",
        "targeted_assumption": "x",
        "predicted_failure_region": "x", "goal_preservation": "x",
        "appendix": "Recent verifier output suggests editing the auth catch path.",
    }
    adapter = SkillEvolBenchCCVAdapter(
        lambda _prompt, **_kwargs: __import__("json").dumps(response)
    )
    probe = adapter.generate(
        original_instruction="Fix it.", old_skill="old", candidate_skill="new",
        task_id="E1-LS1-T2",
    )
    assert probe.status == "inconclusive_generation_failed"
    assert probe.generation_mode == "llm_failed"
    assert probe.generation_attempts == 3
    assert probe.appendix == ""


def test_ccv_model_no_attack_is_inconclusive_when_skills_differ() -> None:
    adapter = SkillEvolBenchCCVAdapter(
        lambda _prompt, **_kwargs: '{"decision":"no_probe"}'
    )
    probe = adapter.generate(
        original_instruction="Fix it.",
        old_skill="Use the repository evidence.",
        candidate_skill="First run helper.py, then edit output.json.",
        task_id="E1-LS1-T2",
    )
    assert probe.status == "inconclusive_generation_failed"
    assert probe.generation_mode == "llm_failed"
    assert probe.generation_attempts == 3
    assert probe.attack_type == ""


def test_ccv_adapter_retries_invalid_output_with_same_model() -> None:
    responses = iter([
        '{"decision":"no_probe"}',
        __import__("json").dumps({
            "decision": "probe",
            "attack_type": "missing_branch",
            "targeted_assumption": "Candidate omits an alternate branch.",
            "predicted_failure_region": "alternate valid branch",
            "goal_preservation": "Original task and verifier stay unchanged.",
            "appendix": "Inputs may exercise another documented branch; preserve all required behavior.",
        }),
    ])
    calls = []

    def call(prompt, **_kwargs):
        calls.append(prompt)
        return next(responses)

    adapter = SkillEvolBenchCCVAdapter(call, generation_retries=2)
    probe = adapter.generate(
        original_instruction="Fix it.", old_skill="old", candidate_skill="new",
        task_id="E1-LS1-T2",
    )
    assert probe.status == "ready"
    assert probe.generation_mode == "llm"
    assert probe.generation_attempts == 2
    assert len(calls) == 2
    assert "previous_invalid_response_reason" in calls[1]


def test_ccv_deterministic_fallback_requires_explicit_legacy_opt_in() -> None:
    adapter = SkillEvolBenchCCVAdapter(
        lambda _prompt, **_kwargs: '{"decision":"no_probe"}',
        generation_retries=0,
        allow_deterministic_fallback=True,
    )
    probe = adapter.generate(
        original_instruction="Fix it.", old_skill="old", candidate_skill="new",
        task_id="E1-LS1-T2",
    )
    assert probe.status == "ready"
    assert probe.generation_mode == "deterministic_fallback"
    assert probe.generation_attempts == 1


def test_regularization_bundle_targets_only_abnormal_modules() -> None:
    bundle = build_regularization_signal_bundle(
        task_id="E1-LS1-T2",
        complexity_projection={
            "triggered": True,
            "initial_candidate": {"total": 18.0, "atom_count": 30},
            "budget": {"max_total": 12.5, "max_atoms": 24},
        },
        ccv_decision={
            "confirmed_regression": True,
            "score_drop": 0.125,
            "probe": {
                "attack_type": "missing_branch",
                "targeted_assumption": "Candidate assumes one branch.",
                "predicted_failure_region": "alternate valid branch",
            }
        },
    )
    instruction = regularization_update_instruction(bundle)
    assert bundle.abnormal_modules == ("complexity", "ccv")
    assert "0.1250" in instruction
    assert "missing_branch" in instruction
    assert "total <= 12.5000" in instruction
    assert "atoms <= 24" in instruction
    assert "total >= 5.5000, atoms >= 6" in instruction
    assert "merge semantically overlapping or duplicate atoms" in instruction
    assert "remove narrow, one-off, task-specific" in instruction
    assert "replace clusters of similar rules with one general rule" in instruction
    assert "exactly one transferable procedural invariant" in instruction
    assert "one-off exception for the probe" in instruction
    assert "must not copy or restate probe-specific field names" in instruction
    assert "probe answers" in instruction
    assert "dropout" not in instruction.lower()


def test_step_local_bundle_routes_actual_dropout_without_forced_compression() -> None:
    bundle = build_regularization_signal_bundle(
        task_id="E1-LS1-T2",
        dropout_report={"masked_count": 1, "mode": "training_rollout"},
    )
    instruction = regularization_update_instruction(bundle)
    assert bundle.abnormal_modules == ("dropout",)
    assert "partial training-time skill view" in instruction
    assert "does not require compression" in instruction


def test_step_local_ccv_suffixes_are_four_distinct_phases() -> None:
    assert _ccv_side("E1-LS1-T2__ccv_old_clean") == "old_clean"
    assert _ccv_side("E1-LS1-T2__ccv_candidate_clean") == "candidate_clean"
    assert _ccv_side("E1-LS1-T2__ccv_old_attack") == "old_attack"
    assert _ccv_side("E1-LS1-T2__ccv_candidate_attack") == "candidate_attack"


def test_step_local_ccv_reuses_passed_training_as_old_clean() -> None:
    from skillevolbench.harbor_ext.hooks import _step_local_ccv_gate

    gate = _step_local_ccv_gate(
        old_library_hash="old-tree",
        candidate_library_hash="candidate-tree",
        training_passed=True,
        training_score=1.0,
    )
    assert gate["active"] is True
    assert gate["old_clean_reused"] is True
    assert gate["old_clean_passed"] is True
    assert gate["old_clean_score"] == 1.0


def test_step_local_ccv_skips_failed_training_source() -> None:
    from skillevolbench.harbor_ext.hooks import _step_local_ccv_gate

    gate = _step_local_ccv_gate(
        old_library_hash="old-tree",
        candidate_library_hash="candidate-tree",
        training_passed=False,
        training_score=0.85,
    )
    assert gate["active"] is False
    assert gate["probe"]["status"] == "inconclusive_unqualified_source"
    assert gate["skip_reason"] == "training_source_not_verifier_passed"


def test_step_local_ccv_runtime_name_resolves_to_original_task(tmp_path) -> None:
    spec = SimpleNamespace(task_id="E1-LS1-T2")

    class Registry:
        def task(self, task_id):
            assert task_id == "E1-LS1-T2"
            return SimpleNamespace(spec=spec)

    hooks = SkillEvolBenchHooks(SimpleNamespace(), Registry(), None, None)
    event = SimpleNamespace(
        task_name="E1-LS1-T2__pass2__ccv_candidate_attack"
    )
    assert hooks._resolve_task(event) is spec


def test_step_local_clean_slot_executes_real_instruction(tmp_path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("ORIGINAL SOURCE\n")
    task = SimpleNamespace(
        task_id="E1-LS1-T2", family_id="E1-LS1", environment_id="E1",
        task_slug="unused", harbor=SimpleNamespace(instruction="instruction.md"),
    )

    class Registry:
        def task(self, task_id):
            assert task_id == task.task_id
            return SimpleNamespace(folder=task_dir, spec=task)

    class Library:
        def rollback_to(self, commit, reason=""):
            assert commit == "old"

        def compute_hash(self):
            return "old-tree"

    runtime_dir = tmp_path / "runtime" / "old-clean"

    class Builder:
        def build(self, **kwargs):
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "instruction.md").write_text(
                "REAL CLEAN INSTRUCTION\n"
            )
            return runtime_dir

    runtime = SimpleNamespace(
        switch_env=lambda _env: None,
        library=Library(),
        retriever=SimpleNamespace(
            retrieve=lambda **_kwargs: SimpleNamespace(skills=[])
        ),
        baseline=SimpleNamespace(skill_retrieval_k=5),
        run_root=tmp_path / "run",
        event_store=EventStore(tmp_path / "events"),
    )
    hooks = SkillEvolBenchHooks(runtime, Registry(), Builder(), None)
    hooks._pending_ccv_candidates[task.task_id] = {
        "active": True, "old_commit": "old", "candidate_commit": "new",
    }
    asyncio.run(hooks._on_ccv_trial_started(
        None, task, "E1-LS1-T2__pass2__ccv_old_clean", "old_clean",
    ))
    assert (runtime_dir / "instruction.md").read_text() == (
        "REAL CLEAN INSTRUCTION\n"
    )


def test_identical_candidate_drains_prescheduled_ccv_slot_without_execution(
    tmp_path,
) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("ORIGINAL SOURCE\n")
    task = SimpleNamespace(
        task_id="E1-LS1-T2", family_id="E1-LS1", environment_id="E1",
        task_slug="unused", harbor=SimpleNamespace(instruction="instruction.md"),
    )

    class Registry:
        def task(self, task_id):
            assert task_id == task.task_id
            return SimpleNamespace(folder=task_dir, spec=task)

    class Builder:
        def build(self, **kwargs):
            runtime_dir = tmp_path / "runtime" / kwargs["runtime_basename"]
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "instruction.md").write_text("REAL INSTRUCTION\n")
            return runtime_dir

    runtime = SimpleNamespace(
        switch_env=lambda _env: None,
        library=SimpleNamespace(compute_hash=lambda: "same-tree"),
        retriever=SimpleNamespace(
            retrieve=lambda **_kwargs: SimpleNamespace(skills=[])
        ),
        baseline=SimpleNamespace(skill_retrieval_k=5),
        run_root=tmp_path / "run",
        event_store=EventStore(tmp_path / "events"),
    )
    hooks = SkillEvolBenchHooks(runtime, Registry(), Builder(), None)
    hooks._pending_ccv_candidates[task.task_id] = {
        "active": False,
        "old_library_hash": "same-tree",
        "candidate_library_hash": "same-tree",
        "probe": {
            "status": "skipped_no_material_candidate",
            "error": "old and candidate library hashes are identical",
        },
    }
    runtime_name = "E1-LS1-T2__pass2__ccv_old_clean"
    asyncio.run(hooks._on_ccv_trial_started(
        None, task, runtime_name, "old_clean",
    ))
    instruction = (
        tmp_path / "runtime" / runtime_name / "instruction.md"
    ).read_text()
    assert "inactive" in instruction
    started = runtime.event_store.events_of_type("ccv_trial_started")
    assert started[0]["active"] is False
    assert started[0]["probe"]["status"] == (
        "skipped_no_material_candidate"
    )


def test_step_local_noop_candidate_finalization_is_a_clean_skip(tmp_path) -> None:
    events = EventStore(tmp_path / "events")
    runtime = SimpleNamespace(
        event_store=events,
        strategy_config=SimpleNamespace(regularization_step_local_pipeline=True),
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    hooks._finalize_step_local_candidate(
        SimpleNamespace(task_id="E1-LS1-T2"),
        {"active": False, "probe": {"status": "skipped_no_patch"}},
    )
    finalized = events.events_of_type("step_local_candidate_finalized")
    assert finalized[0]["status"] == "skipped_no_staged_candidate"


def test_inconclusive_ccv_does_not_suppress_complexity_local_edit(
    tmp_path,
) -> None:
    events = EventStore(tmp_path / "events")
    runtime = SimpleNamespace(
        event_store=events,
        strategy_config=SimpleNamespace(regularization_consumption_mode="local_edit"),
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    hooks._pending_ccv_candidates["E1-LS1-T2"] = {
        "active": False,
        "probe": {
            "status": "inconclusive_generation_failed",
            "generation_mode": "llm_failed",
        },
        "complexity_projection": {
            "triggered": True,
            "initial_candidate": {"total": 18.0, "atom_count": 30},
            "budget": {"max_total": 12.5, "max_atoms": 24},
        },
    }
    consumed = []
    hooks._apply_regularization_local_edit = (
        lambda task, pending, bundle: consumed.append(bundle)
    )
    hooks._finish_ccv_trial(
        SimpleNamespace(task_id="E1-LS1-T2"),
        SimpleNamespace(normalized_score=0.0, verifier_passed=False),
        "candidate",
    )
    assert len(consumed) == 1
    assert consumed[0].abnormal_modules == ("complexity",)
    decision = events.events_of_type("ccv_decision")[0]
    assert decision["status"] == "inconclusive_generation_failed"


def test_confirmed_ccv_regression_retains_candidate_for_feedback(tmp_path) -> None:
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    old_commit = library.head_commit()
    candidate = SkillPatch(
        patch_id=str(uuid.uuid4()), summary="candidate",
        upsert_files={"diagnose/SKILL.md": SKILL + "\nCandidate rule.\n"},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    applied = library.apply_patch(candidate, "E1-LS1-T2", "chain")
    events = EventStore(tmp_path / "events")
    runtime = SimpleNamespace(
        library=library, event_store=events,
        strategy_config=SimpleNamespace(
            regularization_ccv_min_old_score=0.80,
            regularization_ccv_min_regression_drop=0.05,
        ),
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    hooks._pending_ccv_candidates["E1-LS1-T2"] = {
        "active": True, "old_commit": old_commit,
        "candidate_commit": applied.commit_hash,
        "patch_id": candidate.patch_id, "old_attack_score": 1.0,
        "old_attack_passed": True,
        "probe": {"status": "ready", "targeted_assumption": "ordering"},
    }
    hooks._finish_ccv_trial(
        SimpleNamespace(task_id="E1-LS1-T2"),
        SimpleNamespace(normalized_score=0.8, verifier_passed=False),
        "candidate",
    )
    assert library.head_commit() == applied.commit_hash
    decision = events.events_of_type("ccv_decision")[0]
    assert decision["confirmed_regression"] is True
    assert decision["feedback_only"] is True
    assert decision["candidate_retained"] is True
    assert events.events_of_type("patch_rejected") == []
    assert events.events_of_type("rollback_decision") == []


def test_confirmed_ccv_regression_local_edits_once_and_marks_consumed(
    tmp_path,
) -> None:
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    sibling = SKILL.replace("name: diagnose", "name: sibling")
    library.create_skill(
        skill_id="E1-LS1.sibling", content=sibling, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    old_commit = library.head_commit()
    candidate = SkillPatch(
        patch_id=str(uuid.uuid4()), summary="candidate",
        upsert_files={"diagnose/SKILL.md": SKILL + "\nCandidate rule.\n"},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    applied = library.apply_patch(candidate, "E1-LS1-T2", "chain")
    class RepairEvolver:
        def __init__(self):
            self.calls = 0
            self.regularization_instruction = ""

        def propose_local_repair(self, **kwargs):
            self.calls += 1
            self.regularization_instruction = kwargs["regularization_instruction"]
            assert set(kwargs["family_skill_texts"]) == {
                "E1-LS1.diagnose", "E1-LS1.sibling",
            }
            assert set(kwargs["editable_regions_by_skill"]) == {
                "E1-LS1.diagnose",
            }
            if self.calls < 3:
                raise PatchGenerationFailure("synthetic malformed JSON")
            return {
                "operations": [{
                    "skill_id": "E1-LS1.diagnose",
                    "source_region_id": "r1",
                    "operation": "REWRITE",
                    "match_region_id": "",
                }],
                "replacements": {
                    "E1-LS1.diagnose": {
                        "r1": "\nGeneralized repaired rule.\n",
                    },
                },
                "change_summary": ["generalized only source delta"],
            }

    events = EventStore(tmp_path / "events")
    evolver = RepairEvolver()
    freeze_ctrl = LibraryFreezeController(library, events)
    runtime = SimpleNamespace(
        library=library,
        event_store=events,
        freeze_ctrl=freeze_ctrl,
        strategy=SimpleNamespace(
            name="chain_skillevoreg_online", evolver=evolver,
        ),
        strategy_config=SimpleNamespace(
            regularization_ccv_min_old_score=0.80,
            regularization_ccv_min_regression_drop=0.05,
            regularization_consumption_mode="local_edit",
        ),
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    hooks._pending_ccv_candidates["E1-LS1-T2"] = {
        "active": True,
        "old_commit": old_commit,
        "candidate_commit": applied.commit_hash,
        "patch_id": candidate.patch_id,
        "old_attack_score": 1.0,
        "old_attack_passed": True,
        "probe": {
            "status": "ready",
            "attack_type": "missing_branch",
            "targeted_assumption": "Candidate assumes one branch.",
            "predicted_failure_region": "alternate valid branch",
            "responsible_skill_ids": ["E1-LS1.diagnose"],
        },
        "source_affected_skill_ids": ["E1-LS1.diagnose"],
        "ccv_candidate_changed_skill_ids": ["E1-LS1.diagnose"],
        "complexity_projection": {
            "triggered": True,
            "initial_candidate": {"total": 2000.0, "atom_count": 2000},
            # Deliberately impossible for the generated repair: a regularizer
            # budget is telemetry and must never become an acceptance gate.
            "budget": {"max_total": 0.1, "max_atoms": 0},
        },
    }
    task = SimpleNamespace(task_id="E1-LS1-T2", family_id="E1-LS1")
    hooks._finish_ccv_trial(
        task,
        SimpleNamespace(normalized_score=0.8, verifier_passed=False),
        "candidate",
    )

    assert evolver.calls == 3
    assert "GUARD UPDATE REQUIRED" in evolver.regularization_instruction
    assert "COMPLEXITY ANOMALY" in evolver.regularization_instruction
    assert "SCAP REGRESSION" in evolver.regularization_instruction
    assert "Modify only these causally implicated skills" in evolver.regularization_instruction
    assert "LOCAL REPAIR JSON RETRY" in evolver.regularization_instruction
    assert "NOOP|REWRITE|MERGE|DELETE" in evolver.regularization_instruction
    assert "Generalized repaired rule" in hooks._family_skill_text("E1-LS1")
    assert "Unrelated destructive rewrite" not in hooks._family_skill_text("E1-LS1")
    assert library.get_skill("E1-LS1.sibling").files["SKILL.md"] == sibling
    quick = events.events_of_type("regularization_local_edit_applied")
    assert len(quick) == 1
    assert quick[0]["abnormal_modules"] == ["complexity", "ccv"]
    assert quick[0]["repair_scope_skill_ids"] == ["E1-LS1.diagnose"]
    assert quick[0]["second_evaluation"] is False
    assert quick[0]["budget_exceeded_after_update"] is True
    assert quick[0]["complexity_budget"] == {
        "max_total": 0.1, "max_atoms": 0,
    }
    assert quick[0]["complexity_direction"]["direction"] == "decreased"
    assert quick[0]["complexity_direction"]["acceptance_gate"] is False
    assert len(quick[0]["validator_attempts"]) == 3
    assert quick[0]["validator_attempts"][2]["errors"] == []
    consumed = events.events_of_type("regularization_feedback_consumed")
    assert len(consumed) == 1
    assert consumed[0]["consumed_by"] == "immediate_local_edit"
    assert events.events_of_type("patch_rejected") == []
    assert events.events_of_type("rollback_decision") == []


def _matched_delta_local_edit_fixture(tmp_path, evolver):
    library = LibraryStore.init(tmp_path / "library")
    original = (
        SKILL
        + "\n- Reuse one stable idempotency key for every retry attempt.\n"
    )
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=original, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    old_commit = library.head_commit()
    candidate_text = (
        original
        + "- Reuse one stable idempotency key for every retry.\n"
    )
    source_patch = SkillPatch(
        patch_id=str(uuid.uuid4()), summary="candidate",
        upsert_files={"diagnose/SKILL.md": candidate_text},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    library.apply_patch(source_patch, "E1-LS1-T2", "chain")
    events = EventStore(tmp_path / "events")
    runtime = SimpleNamespace(
        library=library,
        event_store=events,
        freeze_ctrl=LibraryFreezeController(library, events),
        strategy=SimpleNamespace(
            name="chain_skillevoreg_online", evolver=evolver,
        ),
        strategy_config=SimpleNamespace(regularization_consumption_mode="local_edit"),
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    pending = {
        "old_commit": old_commit,
        "patch_id": source_patch.patch_id,
        "source_affected_skill_ids": ["E1-LS1.diagnose"],
        "ccv_candidate_changed_skill_ids": ["E1-LS1.diagnose"],
        "complexity_projection": {
            "triggered": True,
            "initial_candidate": {"total": 20.0, "atom_count": 8},
            "budget": {"max_total": 18.0, "max_atoms": 7},
        },
    }
    task = SimpleNamespace(task_id="E1-LS1-T2", family_id="E1-LS1")
    return hooks, library, events, pending, task, candidate_text


class _MatchedNoopEvolver:
    def __init__(self, second_mode: str = "noop") -> None:
        self.calls = 0
        self.instructions: list[str] = []
        self.second_mode = second_mode

    def propose_local_repair(self, **kwargs):
        self.calls += 1
        self.instructions.append(kwargs["regularization_instruction"])
        skill_id, regions = next(iter(kwargs["editable_regions_by_skill"].items()))
        source = next(region for region in regions if region.role == "source_delta")
        matched = next(region for region in regions if region.role == "existing_match")
        if self.calls == 1 or self.second_mode == "noop":
            return {
                "operations": [{
                    "skill_id": skill_id,
                    "source_region_id": source.region_id,
                    "operation": "NOOP",
                    "match_region_id": "",
                }],
                "replacements": {},
                "change_summary": ["keep the source delta"],
            }
        return {
            "operations": [{
                "skill_id": skill_id,
                "source_region_id": source.region_id,
                "operation": "REWRITE",
                "match_region_id": "",
            }],
            # This deliberately creates a second exact copy of the matched
            # read-only rule, so the structural retry must fall back to NOOP.
            "replacements": {skill_id: {
                source.region_id: matched.candidate_text,
            }},
            "change_summary": ["duplicated the matched rule"],
        }


def test_complexity_all_noop_gets_one_soft_subsumption_retry(tmp_path) -> None:
    evolver = _MatchedNoopEvolver()
    hooks, _library, events, pending, task, _candidate = (
        _matched_delta_local_edit_fixture(tmp_path, evolver)
    )
    bundle = build_regularization_signal_bundle(
        task_id=task.task_id,
        complexity_projection=pending["complexity_projection"],
        ccv_decision={},
    )
    hooks._apply_regularization_local_edit(task, pending, bundle)

    assert evolver.calls == 2
    assert "SOFT SUBSUMPTION RECHECK" not in evolver.instructions[0]
    assert "SOFT SUBSUMPTION RECHECK" in evolver.instructions[1]
    noop = events.events_of_type("regularization_local_edit_noop")
    assert len(noop) == 1
    assert noop[0]["reason"] == "soft_retry_confirmed_noop"
    assert noop[0]["noop_soft_retry_triggered"] is True
    assert noop[0]["noop_fallback_used"] is False


def test_ccv_only_all_noop_is_accepted_without_soft_retry(tmp_path) -> None:
    evolver = _MatchedNoopEvolver()
    hooks, _library, events, pending, task, _candidate = (
        _matched_delta_local_edit_fixture(tmp_path, evolver)
    )
    bundle = build_regularization_signal_bundle(
        task_id=task.task_id,
        complexity_projection={"triggered": False},
        ccv_decision={
            "confirmed_regression": True,
            "score_drop": 0.2,
            "probe": {
                "responsible_skill_ids": ["E1-LS1.diagnose"],
                "targeted_assumption": "retry identity",
            },
        },
    )
    hooks._apply_regularization_local_edit(task, pending, bundle)

    assert evolver.calls == 1
    noop = events.events_of_type("regularization_local_edit_noop")
    assert len(noop) == 1
    assert noop[0]["reason"] == "all_noop"
    assert noop[0]["noop_soft_retry_triggered"] is False


def test_complexity_soft_retry_structural_failure_uses_first_noop(tmp_path) -> None:
    evolver = _MatchedNoopEvolver(second_mode="duplicate")
    hooks, library, events, pending, task, candidate = (
        _matched_delta_local_edit_fixture(tmp_path, evolver)
    )
    bundle = build_regularization_signal_bundle(
        task_id=task.task_id,
        complexity_projection=pending["complexity_projection"],
        ccv_decision={},
    )
    hooks._apply_regularization_local_edit(task, pending, bundle)

    assert evolver.calls == 2
    assert library.get_skill("E1-LS1.diagnose").files["SKILL.md"] == candidate
    noop = events.events_of_type("regularization_local_edit_noop")
    assert len(noop) == 1
    assert noop[0]["reason"] == "soft_retry_structural_validation_failed"
    assert noop[0]["noop_fallback_used"] is True
    assert any(
        "new exact duplicate rule atom" in error
        for error in noop[0]["validator_attempts"][-1]["errors"]
    )
    assert events.events_of_type("regularization_local_edit_applied") == []


def test_online_local_repair_preserves_sibling_and_historical_bytes() -> None:
    anchor = SKILL
    candidate = SKILL + "\nCandidate-specific rule.\n"
    regions = editable_regions(anchor, candidate)
    before_hash = protected_content_hash(candidate, regions)
    repaired = apply_replacements(
        candidate, regions, {"r1": "Transferable replacement rule.\n"},
    )
    assert repaired == SKILL + "\nTransferable replacement rule.\n"
    assert protected_content_hash(
        repaired, editable_regions(anchor, repaired)
    ) == before_hash
    assert validate_repaired_document(repaired) == []


def test_exact_duplicate_invariant_rejects_new_second_copy() -> None:
    rule = "- Validate every required schema field before sending the remote request.\n"
    before = SKILL + "\n" + rule
    after = before + rule
    errors = new_exact_duplicate_atom_errors(before, after)
    assert len(errors) == 1
    assert "count 1 -> 2" in errors[0]


def test_exact_duplicate_invariant_allows_first_copy() -> None:
    rule = "- Validate every required schema field before sending the remote request.\n"
    assert new_exact_duplicate_atom_errors(SKILL, SKILL + "\n" + rule) == []


def test_exact_duplicate_invariant_tolerates_existing_duplicate_count() -> None:
    rule = "- Validate every required schema field before sending the remote request.\n"
    duplicated = SKILL + "\n" + rule + rule
    assert new_exact_duplicate_atom_errors(duplicated, duplicated) == []


def test_exact_duplicate_invariant_allows_merge_move_without_multiplication() -> None:
    rule = "- Validate every required schema field before sending the remote request.\n"
    other = "- Preserve the original authentication headers during every retry attempt.\n"
    before = SKILL + "\n" + rule + other
    after = SKILL + "\n" + other + rule
    assert new_exact_duplicate_atom_errors(before, after) == []


def test_exact_duplicate_invariant_ignores_fenced_code() -> None:
    code = "```python\nvalidate_required_schema_fields_before_remote_request()\n```\n"
    after = SKILL + "\n" + code + "\n" + code
    assert new_exact_duplicate_atom_errors(SKILL, after) == []


def test_online_local_repair_ignores_frontmatter_only_changes() -> None:
    candidate = SKILL.replace(
        "Diagnose failures systematically.",
        "Diagnose failures using a revised description.",
    )
    assert editable_regions(SKILL, candidate) == []

    candidate += "\nValidate the final diagnosis.\n"
    regions = editable_regions(SKILL, candidate)
    assert len(regions) == 1
    assert regions[0].anchor_text == ""
    assert "Validate the final diagnosis" in regions[0].candidate_text
    assert "description:" not in regions[0].candidate_text


def test_online_regularization_repair_opens_only_one_similar_existing_atom() -> None:
    anchor = (
        SKILL
        + "\n- Reuse one idempotency key for every retry attempt.\n"
        + "- Preserve unrelated authentication behavior.\n"
    )
    candidate = anchor + "- Reuse one idempotency key for every retry.\n"
    regions = regularization_repair_regions(anchor, candidate)
    history = [region for region in regions if region.role == "existing_match"]
    assert len(history) == 1
    assert "idempotency key" in history[0].candidate_text
    assert not history[0].candidate_text.lstrip().startswith("#")
    source = next(region for region in regions if region.role == "source_delta")
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


def test_online_regularization_decides_independently_for_each_source_delta() -> None:
    anchor = (
        SKILL
        + "\n- Retry requests with one stable idempotency key.\n"
        + "- Preserve unrelated authentication behavior.\n"
        + "- Validate required schema fields before submission.\n"
    )
    candidate = (
        SKILL
        + "\n- Retry requests with one stable idempotency key.\n"
        + "- Retry requests using one stable idempotency key.\n"
        + "- Preserve unrelated authentication behavior.\n"
        + "- Validate required schema fields before submission.\n"
        + "- Check required schema fields before sending the request.\n"
    )
    regions = regularization_repair_regions(anchor, candidate)
    sources = [r for r in regions if r.role == "source_delta"]
    history = [r for r in regions if r.role == "existing_match"]
    assert len(sources) == 2
    assert len(history) == 2
    assert {r.source_region_id for r in history} == {r.region_id for r in sources}
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
        replacements[source.region_id] = source.candidate_text
        replacements[matched.region_id] = ""
    normalized, errors = validate_delta_operations(
        {"operations": operations}, regions, replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert len(normalized) == 2
    assert errors == []


def test_online_regularization_merge_collapses_source_and_match_into_one_rule() -> None:
    anchor = SKILL + "\n- Retry requests with one stable idempotency key.\n"
    candidate = anchor + "- Retry requests using one stable idempotency key.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    matched = next(region for region in regions if region.role == "existing_match")
    merged_rule = "- Use one stable idempotency key across all retries.\n"
    replacements = {
        source.region_id: merged_rule,
        matched.region_id: "",
    }
    operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "MERGE",
            "match_region_id": matched.region_id,
        }]},
        regions,
        replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert operations[0]["operation"] == "MERGE"
    assert errors == []

    repaired = apply_replacements(candidate, regions, replacements)
    assert repaired.count("stable idempotency key") == 1
    assert "Retry requests using one stable idempotency key" not in repaired
    assert validate_repaired_document(repaired) == []


def test_online_regularization_merge_canonicalizes_linked_existing_atom_deletion() -> None:
    anchor = SKILL + "\n- Retry requests with one stable idempotency key.\n"
    candidate = anchor + "- Retry requests using one stable idempotency key.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    matched = next(region for region in regions if region.role == "existing_match")
    operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "MERGE",
            "match_region_id": matched.region_id,
        }]},
        regions,
        {
            source.region_id: "- Use one stable idempotency key across all retries.\n",
            matched.region_id: "- Retain a stable idempotency key for retries.\n",
        },
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == []
    assert operations[0]["match_region_id"] == matched.region_id


def test_online_regularization_merge_requires_nonempty_source_replacement() -> None:
    anchor = SKILL + "\n- Retry requests with one stable idempotency key.\n"
    candidate = anchor + "- Retry requests using one stable idempotency key.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    matched = next(region for region in regions if region.role == "existing_match")
    replacements = {source.region_id: "", matched.region_id: "not authoritative"}
    _operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "MERGE",
            "match_region_id": matched.region_id,
        }]},
        regions,
        replacements,
        abnormal_modules=frozenset({"complexity"}),
    )
    assert any("requires a non-empty merged source replacement" in error for error in errors)


def test_online_deleted_delta_cannot_be_deleted_twice() -> None:
    candidate = SKILL + "\n- Preserve one stable capability.\n"
    anchor = candidate + "- Preserve one stable capability across tasks.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    assert source.candidate_text == ""
    _operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "DELETE",
            "match_region_id": None,
        }]}, regions, {source.region_id: ""},
        abnormal_modules=frozenset({"ccv"}),
    )
    assert any(
        error == "DELETE for r1 is invalid because the source delta is already empty"
        for error in errors
    )


def test_online_deleted_delta_cannot_merge_with_existing_atom() -> None:
    anchor = (
        SKILL
        + "\n- Retry requests with one stable idempotency key.\n"
        + "- Retry requests using one stable idempotency key.\n"
    )
    candidate = SKILL + "\n- Retry requests with one stable idempotency key.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    matched = next(region for region in regions if region.role == "existing_match")
    assert source.candidate_text == ""
    _operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": "MERGE",
            "match_region_id": matched.region_id,
        }]},
        regions,
        {
            source.region_id: "- Use one stable idempotency key across all retries.\n",
            matched.region_id: "",
        },
        abnormal_modules=frozenset({"complexity"}),
    )
    assert errors == [
        "MERGE for r1 is invalid because the source delta is already empty"
    ]


def test_skill_author_accepts_per_delta_repair_contract() -> None:
    anchor = SKILL + "\n- Retry requests with one stable idempotency key.\n"
    candidate = anchor + "- Retry requests using one stable idempotency key.\n"
    regions = regularization_repair_regions(anchor, candidate)

    def good_call(_prompt, **kwargs):
        assert "Never copy or restate an adjacent read-only atom" in kwargs[
            "system_prompt"
        ]
        assert "Cosine similarity only nominates a pair" in kwargs["system_prompt"]
        assert "preserve the full union" in kwargs["system_prompt"]
        assert "if any supported behavior would be lost, use NOOP" in kwargs[
            "system_prompt"
        ]
        return (
            '{"operations":[{"skill_id":"E1-LS1.diagnose",'
            '"source_region_id":"r1","operation":"REWRITE",'
            '"match_region_id":null}],"replacements":['
            '{"skill_id":"E1-LS1.diagnose","region_id":"r1",'
            '"replacement":"\\nTransferable rule.\\n"}],'
            '"change_summary":["rewrote the source delta"]}'
        )

    author = SkillAuthor(sync_call=good_call)
    result = author.propose_local_repair(
        task_id="E1-LS1-T2",
        regularization_instruction="repair complexity",
        family_skill_texts={"E1-LS1.diagnose": candidate},
        editable_regions_by_skill={"E1-LS1.diagnose": regions},
        abnormal_modules=("complexity",),
    )
    assert result["operations"] == [{
        "skill_id": "E1-LS1.diagnose",
        "source_region_id": "r1",
        "operation": "REWRITE",
        "match_region_id": "",
    }]
    assert (
        result["replacements"]["E1-LS1.diagnose"]["r1"].strip()
        == "Transferable rule."
    )


@pytest.mark.parametrize("operation", ["NOOP", "REWRITE", "DELETE"])
def test_skill_author_operation_first_normalizes_non_merge_contract(
    operation: str,
) -> None:
    anchor = SKILL + "\n- Retry requests with one stable idempotency key.\n"
    candidate = anchor + "- Retry requests using one stable idempotency key.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    matched = next(region for region in regions if region.role == "existing_match")
    payload = {
        "operations": [{
            "skill_id": "E1-LS1.diagnose",
            "source_region_id": source.region_id,
            "operation": operation,
            # Deliberately noisy: non-MERGE operations never own a match.
            "match_region_id": matched.region_id,
        }],
        "replacements": [
            {
                "skill_id": "E1-LS1.diagnose",
                "region_id": source.region_id,
                "replacement": "\nTransferable normalized rule.\n",
            },
            {
                "skill_id": "E1-LS1.diagnose",
                "region_id": matched.region_id,
                "replacement": "\nMust be ignored outside MERGE.\n",
            },
        ],
        "change_summary": ["normalize by declared operation"],
    }
    result = SkillAuthor(
        sync_call=lambda _prompt, **_kwargs: json.dumps(payload)
    ).propose_local_repair(
        task_id="E1-LS1-T2",
        regularization_instruction="repair the source delta",
        family_skill_texts={"E1-LS1.diagnose": candidate},
        editable_regions_by_skill={"E1-LS1.diagnose": regions},
        abnormal_modules=("complexity",),
    )
    assert result["operations"][0]["match_region_id"] == ""
    if operation == "NOOP":
        assert result["replacements"] == {}
    elif operation == "REWRITE":
        assert result["replacements"] == {
            "E1-LS1.diagnose": {
                source.region_id: "\nTransferable normalized rule.\n",
            },
        }
    else:
        assert result["replacements"] == {
            "E1-LS1.diagnose": {source.region_id: ""},
        }


def test_skill_author_merge_uses_deterministic_link_and_deletes_match() -> None:
    anchor = (
        SKILL
        + "\n- Retry requests with one stable idempotency key.\n"
        + "- Preserve unrelated authentication behavior.\n"
        + "- Validate required schema fields before submission.\n"
    )
    candidate = (
        SKILL
        + "\n- Retry requests with one stable idempotency key.\n"
        + "- Retry requests using one stable idempotency key.\n"
        + "- Preserve unrelated authentication behavior.\n"
        + "- Validate required schema fields before submission.\n"
        + "- Check required schema fields before sending the request.\n"
    )
    regions = regularization_repair_regions(anchor, candidate)
    sources = [region for region in regions if region.role == "source_delta"]
    match_by_source = {
        region.source_region_id: region
        for region in regions if region.role == "existing_match"
    }
    merge_source, noop_source = sources
    correct_match = match_by_source[merge_source.region_id]
    wrong_match = match_by_source[noop_source.region_id]
    merged_text = "- Use one stable canonical value through validation.\n"
    payload = {
        "operations": [
            {
                "skill_id": "E1-LS1.diagnose",
                "source_region_id": merge_source.region_id,
                "operation": "MERGE",
                "match_region_id": wrong_match.region_id,
            },
            {
                "skill_id": "E1-LS1.diagnose",
                "source_region_id": noop_source.region_id,
                "operation": "NOOP",
                "match_region_id": wrong_match.region_id,
            },
        ],
        "replacements": [
            {
                "skill_id": "E1-LS1.diagnose",
                "region_id": merge_source.region_id,
                "replacement": merged_text,
            },
            {
                "skill_id": "E1-LS1.diagnose",
                "region_id": correct_match.region_id,
                "replacement": "wrong non-empty match text",
            },
            {
                "skill_id": "E1-LS1.diagnose",
                "region_id": wrong_match.region_id,
                "replacement": "wrong linked match text",
            },
        ],
        "change_summary": ["merge one source deterministically"],
    }
    result = SkillAuthor(
        sync_call=lambda _prompt, **_kwargs: json.dumps(payload)
    ).propose_local_repair(
        task_id="E1-LS1-T2",
        regularization_instruction="repair redundant rules",
        family_skill_texts={"E1-LS1.diagnose": candidate},
        editable_regions_by_skill={"E1-LS1.diagnose": regions},
        abnormal_modules=("complexity",),
    )
    by_source = {
        row["source_region_id"]: row for row in result["operations"]
    }
    assert by_source[merge_source.region_id]["match_region_id"] == (
        correct_match.region_id
    )
    assert by_source[noop_source.region_id]["match_region_id"] == ""
    assert result["replacements"] == {
        "E1-LS1.diagnose": {
            merge_source.region_id: merged_text,
            correct_match.region_id: "",
        },
    }


def test_local_repair_merge_can_consolidate_three_historical_atoms() -> None:
    anchor = (
        SKILL
        + "\n- Retry transient requests with bounded exponential backoff.\n"
        + "- Reuse one stable idempotency key across every retry attempt.\n"
        + "- Stop retrying permanent validation and authentication failures.\n"
    )
    candidate = anchor + (
        "- Retry safely with bounded backoff, stable identity, and permanent-error classification.\n"
    )
    regions = regularization_repair_regions(anchor, candidate, min_similarity=0.0)
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


def test_online_same_batch_group_queries_history_without_editing(monkeypatch) -> None:
    import numpy as np
    import skillevolbench.components.local_repair as local_repair

    anchor = SKILL + "\n- Inspect the complete task state before editing.\n"
    candidate = anchor + (
        "- Read all relevant project files before making a change.\n"
        "- Inspect the full affected project region before patching.\n"
    )
    calls = 0

    def fake_similarity(queries, documents, *, model_id_or_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.asarray([[1.0, 0.90], [0.90, 1.0]])
        assert len(queries) == 1
        assert "project files" in queries[0]
        assert "project region" in queries[0]
        return np.full((len(queries), len(documents)), 0.85)

    monkeypatch.setattr(local_repair, "cosine_similarity_matrix", fake_similarity)
    regions = regularization_repair_regions(anchor, candidate, min_similarity=0.75)
    source = next(region for region in regions if region.role == "source_delta")
    member = next(region for region in regions if region.role == "same_batch_member")
    history = next(region for region in regions if region.role == "existing_match")
    assert member.source_region_id == history.source_region_id == source.region_id
    replacements = {source.region_id: "- Inspect all relevant task state.\n"}
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


@pytest.mark.parametrize(
    ("failure_kind", "expected_error"),
    [
        ("missing", "missing decisions for source deltas"),
        ("duplicate", "duplicate delta decision"),
        ("blank_rewrite", "requires a non-empty source replacement"),
    ],
)
def test_skill_author_canonicalization_keeps_strict_decision_errors(
    failure_kind: str,
    expected_error: str,
) -> None:
    anchor = SKILL + "\n- Retry requests with one stable idempotency key.\n"
    candidate = anchor + "- Retry requests using one stable idempotency key.\n"
    regions = regularization_repair_regions(anchor, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    base = {
        "skill_id": "E1-LS1.diagnose",
        "source_region_id": source.region_id,
        "operation": "NOOP",
        "match_region_id": None,
    }
    if failure_kind == "missing":
        operations = []
    elif failure_kind == "duplicate":
        operations = [base, dict(base)]
    else:
        operations = [{**base, "operation": "REWRITE"}]
    payload = {
        "operations": operations,
        "replacements": [],
        "change_summary": [],
    }
    author = SkillAuthor(
        sync_call=lambda _prompt, **_kwargs: json.dumps(payload)
    )
    with pytest.raises(PatchGenerationFailure, match=expected_error):
        author.propose_local_repair(
            task_id="E1-LS1-T2",
            regularization_instruction="repair the source delta",
            family_skill_texts={"E1-LS1.diagnose": candidate},
            editable_regions_by_skill={"E1-LS1.diagnose": regions},
            abnormal_modules=("complexity",),
        )


def test_skill_author_noop_keeps_delta_without_replacement() -> None:
    candidate = SKILL + "\nCandidate-specific rule.\n"
    regions = editable_regions(SKILL, candidate)
    calls = 0

    def noop_call(_prompt, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("unmatched source delta must not be sent to the model")

    author = SkillAuthor(sync_call=noop_call)
    result = author.propose_local_repair(
        task_id="E1-LS1-T2",
        regularization_instruction="inspect the source delta",
        family_skill_texts={"E1-LS1.diagnose": candidate},
        editable_regions_by_skill={"E1-LS1.diagnose": regions},
        abnormal_modules=("ccv",),
    )
    assert result["operations"][0]["operation"] == "NOOP"
    assert result["operations"][0]["source_region_id"] == "r1"
    assert result["replacements"] == {}
    assert calls == 0


def test_skill_author_match_below_point50_is_unmatched_and_skips_model() -> None:
    anchor = SKILL + "\n- Retry requests with one stable idempotency key.\n"
    candidate = anchor + "- Retry requests using one stable idempotency key.\n"
    regions = regularization_repair_regions(anchor, candidate)
    matched = next(region for region in regions if region.role == "existing_match")
    below_threshold = type(matched)(
        **{**matched.to_dict(), "similarity": 0.499999}
    )
    regions = [
        below_threshold if region.region_id == matched.region_id else region
        for region in regions
    ]
    calls = 0

    def forbidden_call(_prompt, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("below-threshold match must not reach the model")

    result = SkillAuthor(sync_call=forbidden_call).propose_local_repair(
        task_id="E1-LS1-T2",
        regularization_instruction="inspect the source delta",
        family_skill_texts={"E1-LS1.diagnose": candidate},
        editable_regions_by_skill={"E1-LS1.diagnose": regions},
        abnormal_modules=("complexity",),
    )
    assert calls == 0
    assert result["operations"] == [{
        "skill_id": "E1-LS1.diagnose",
        "source_region_id": "r1",
        "operation": "NOOP",
        "match_region_id": "",
    }]
    assert result["replacements"] == {}


def test_skill_author_mixed_matches_sends_only_matched_source_to_model() -> None:
    anchor = (
        SKILL
        + "\n- Retry requests with one stable idempotency key.\n"
        + "- Preserve unrelated authentication behavior.\n"
        + "- Validate required schema fields before submission.\n"
    )
    candidate = (
        SKILL
        + "\n- Retry requests with one stable idempotency key.\n"
        + "- Retry requests using one stable idempotency key.\n"
        + "- Preserve unrelated authentication behavior.\n"
        + "- Validate required schema fields before submission.\n"
        + "- Check required schema fields before sending the request.\n"
    )
    regions = regularization_repair_regions(anchor, candidate)
    sources = [region for region in regions if region.role == "source_delta"]
    matches = [region for region in regions if region.role == "existing_match"]
    assert len(sources) == 2
    assert len(matches) == 2
    retained_match = matches[0]
    matched_source_id = retained_match.source_region_id
    unmatched_source_id = next(
        source.region_id for source in sources
        if source.region_id != matched_source_id
    )
    mixed_regions = [
        region for region in regions
        if region.region_id != matches[1].region_id
    ]
    assert {
        region.source_region_id for region in mixed_regions
        if region.role == "existing_match"
    } == {matched_source_id}
    calls = 0

    def mixed_call(prompt, **_kwargs):
        nonlocal calls
        calls += 1
        assert f'"region_id": "{matched_source_id}"' in prompt
        assert f'"region_id": "{retained_match.region_id}"' in prompt
        assert f'"region_id": "{unmatched_source_id}"' not in prompt
        return (
            '{"operations":[{"skill_id":"E1-LS1.diagnose",'
            f'"source_region_id":"{matched_source_id}","operation":"NOOP",'
            '"match_region_id":null}],"replacements":[],'
            '"change_summary":["keep matched source unchanged"]}'
        )

    author = SkillAuthor(sync_call=mixed_call)
    result = author.propose_local_repair(
        task_id="E1-LS1-T2",
        regularization_instruction="inspect the source deltas",
        family_skill_texts={"E1-LS1.diagnose": candidate},
        editable_regions_by_skill={"E1-LS1.diagnose": mixed_regions},
        abnormal_modules=("complexity",),
    )
    assert calls == 1
    assert {
        (item["source_region_id"], item["operation"])
        for item in result["operations"]
    } == {
        (matched_source_id, "NOOP"),
        (unmatched_source_id, "NOOP"),
    }
    assert result["replacements"] == {}


def test_online_multi_atom_diff_hunk_is_split_into_isolated_pairs() -> None:
    anchor = (
        SKILL
        + "\n- Retry requests with one stable idempotency key.\n"
        + "- Validate required schema fields before submission.\n"
        + "- Preserve unrelated authentication behavior.\n"
    )
    candidate = (
        anchor
        + "- Retry requests using one stable idempotency key.\n"
        + "- Check required schema fields before sending the request.\n"
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
    assert "Preserve unrelated authentication behavior" in repaired


def test_online_adjacent_deleted_atoms_are_independent_deltas() -> None:
    anchor = (
        SKILL
        + "\n- First removable narrow exception.\n"
        + "- Second removable narrow exception.\n"
    )
    sources = editable_regions(anchor, SKILL)
    assert len(sources) == 2
    assert all(source.start == source.end for source in sources)
    assert [source.anchor_text for source in sources] == [
        "- First removable narrow exception.\n",
        "- Second removable narrow exception.\n",
    ]


def test_online_structural_heading_delta_is_not_paired_with_rule_text() -> None:
    anchor = (
        SKILL
        + "\n- Replicate a displayed value from another cell exactly.\n"
    )
    candidate = anchor + "### Replicating a displayed value from another cell\n"
    regions = regularization_repair_regions(anchor, candidate, min_similarity=0.0)
    source = next(region for region in regions if region.role == "source_delta")
    assert source.atom_type == "other"
    assert not any(region.role == "existing_match" for region in regions)


@pytest.mark.parametrize("operation", ["REWRITE", "DELETE", "MERGE"])
def test_online_unmatched_source_rejects_every_non_noop_operation(
    operation: str,
) -> None:
    candidate = SKILL + "\nCandidate-specific rule.\n"
    regions = editable_regions(SKILL, candidate)
    source = next(region for region in regions if region.role == "source_delta")
    replacement = "" if operation == "DELETE" else "\nTransferable rule.\n"
    _operations, errors = validate_delta_operations(
        {"operations": [{
            "source_region_id": source.region_id,
            "operation": operation,
            "match_region_id": None,
        }]},
        regions,
        {source.region_id: replacement},
        abnormal_modules=frozenset({"complexity"}),
    )
    assert any(
        "source delta has no similarity-qualified existing atom" in error
        for error in errors
    )


def test_skill_author_local_contract_rejects_cross_scope_output() -> None:
    anchor = SKILL + "\n- Retry requests with one stable idempotency key.\n"
    candidate = anchor + "- Retry requests using one stable idempotency key.\n"
    regions = regularization_repair_regions(anchor, candidate)

    def bad_call(_prompt, **_kwargs):
        return (
            '{"replacements":[{"skill_id":"E1-LS1.sibling",'
            '"region_id":"r1","replacement":"rewrite"}]}'
        )

    author = SkillAuthor(sync_call=bad_call)
    with pytest.raises(PatchGenerationFailure, match="unknown editable region"):
        author.propose_local_repair(
            task_id="E1-LS1-T2",
            regularization_instruction="repair complexity",
            family_skill_texts={
                "E1-LS1.diagnose": candidate,
                "E1-LS1.sibling": SKILL.replace("diagnose", "sibling"),
            },
            editable_regions_by_skill={"E1-LS1.diagnose": regions},
        )


def test_library_reads_source_anchor_without_checkout(tmp_path) -> None:
    library = LibraryStore.init(tmp_path / "library")
    library.create_skill(
        skill_id="E1-LS1.diagnose", content=SKILL, source="induction",
        family_id="E1-LS1", created_from_task="E1-LS1-T1",
    )
    old_commit = library.head_commit()
    patch = SkillPatch(
        patch_id=str(uuid.uuid4()), summary="source delta",
        upsert_files={"diagnose/SKILL.md": SKILL + "\nNew rule.\n"},
        target_skill_ids=["E1-LS1.diagnose"],
        operation_type=OperationType.REVISE,
        triggered_by_task="E1-LS1-T2",
    )
    new_commit = library.apply_patch(patch, "E1-LS1-T2", "chain").commit_hash
    assert library.has_commit(old_commit) is True
    assert library.has_commit("not-a-real-commit") is False
    assert library.get_skill_text_at_commit("E1-LS1.diagnose", old_commit) == SKILL
    assert library.get_skill_text_at_commit(
        "E1-LS1.diagnose", new_commit,
    ) == SKILL + "\nNew rule.\n"
    assert library.head_commit() == new_commit


@pytest.mark.parametrize(
    ("env_id", "candidate_commit", "skill_id", "max_editable_fraction"),
    [
        (
            "E4", "386fd913e573e03cd51e54cbddead9bf72909abb",
            "E4-LS4.document-diff-comparison", 0.60,
        ),
        (
            "E5", "9efa23a038411aaa6344f6735b8a4848d0bf75bc",
            "E5-LS2.misleading-claim-review", 0.50,
        ),
        (
            "E5", "19d13af184a49e8a9b917e4083bdea624c75a476",
            "E5-LS2.evidence-grounded-comparison", 0.40,
        ),
    ],
)
def test_historical_regularization_trigger_has_replayable_local_regions(
    env_id: str,
    candidate_commit: str,
    skill_id: str,
    max_editable_fraction: float,
) -> None:
    run_root = (
        Path(__file__).resolve().parents[1]
        / "workspace/runs/"
        "main_skillevolbench_secondpass_regularization_repaired_sharedanchor_medium_seedA_01"
    )
    library_root = run_root / "library" / env_id
    if not (library_root / ".git").exists():
        pytest.skip("historical SkillEvolBench replay library is unavailable")
    old_commit = subprocess.run(
        ["git", "rev-parse", f"{candidate_commit}~2"],
        cwd=library_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    library = LibraryStore(library_root)
    anchor = library.get_skill_text_at_commit(skill_id, old_commit) or ""
    candidate = library.get_skill_text_at_commit(skill_id, candidate_commit) or ""
    regions = editable_regions(anchor, candidate)
    assert regions
    assert (
        sum(len(region.candidate_text) for region in regions) / len(candidate)
        < max_editable_fraction
    )
    target = max(regions, key=lambda region: len(region.candidate_text))
    replacement = target.candidate_text + "\n"
    repaired = apply_replacements(
        candidate, regions, {target.region_id: replacement},
    )
    assert repaired == candidate[:target.start] + replacement + candidate[target.end:]
    assert validate_repaired_document(repaired) == []


def test_ccv_source_qualification_requires_exact_old_tree_not_clean_pass() -> None:
    source = SimpleNamespace(task_id="E1-LS1-T2", family_id="E1-LS1")
    records = [
        SimpleNamespace(
            task_id="E1-LS1-T2",
            library_hash_pre="tree-a", replay_mode="primary",
            outcome=SimpleNamespace(verifier_passed=True, normalized_score=1.0),
        )
    ]
    runtime = SimpleNamespace(
        replay_store=SimpleNamespace(tasks_by_family=lambda _family: records)
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    assert hooks._qualify_ccv_source(source, "tree-a")["qualified"] is True
    assert hooks._qualify_ccv_source(source, "tree-b")["qualified"] is False
    records[0].outcome.verifier_passed = False
    records[0].outcome.normalized_score = 0.6
    qualification = hooks._qualify_ccv_source(source, "tree-a")
    assert qualification["qualified"] is True
    assert qualification["source_passed"] is False
    assert qualification["source_score"] == 0.6


def test_ccv_continuous_reward_thresholds_define_valid_regression(tmp_path) -> None:
    events = EventStore(tmp_path / "events")
    runtime = SimpleNamespace(
        library=SimpleNamespace(compute_hash=lambda: "candidate-tree"),
        event_store=events,
        strategy_config=SimpleNamespace(
            regularization_ccv_min_old_score=0.80,
            regularization_ccv_min_regression_drop=0.05,
        ),
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    hooks._pending_ccv_candidates["E1-LS1-T2"] = {
        "active": True,
        "old_attack_score": 0.86,
        "old_attack_passed": False,
        "probe": {"status": "ready", "attack_type": "missing_branch"},
    }
    hooks._finish_ccv_trial(
        SimpleNamespace(task_id="E1-LS1-T2"),
        SimpleNamespace(normalized_score=0.79, verifier_passed=False),
        "candidate",
    )
    decision = events.events_of_type("ccv_decision")[0]
    assert decision["valid_pair"] is True
    assert decision["confirmed_regression"] is True
    assert decision["score_drop"] > 0.05


def test_step_local_ccv_requires_both_clean_results(tmp_path) -> None:
    events = EventStore(tmp_path / "events")
    runtime = SimpleNamespace(
        library=SimpleNamespace(compute_hash=lambda: "candidate-tree"),
        event_store=events,
        strategy_config=SimpleNamespace(
            regularization_step_local_pipeline=True,
            regularization_ccv_min_old_score=0.80,
            regularization_ccv_min_regression_drop=0.05,
        ),
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    hooks._pending_ccv_candidates["E1-LS1-T2"] = {
        "active": True,
        "old_attack_score": 1.0,
        "old_attack_passed": True,
        "probe": {"status": "ready"},
    }
    hooks._finalize_step_local_candidate = lambda *_args: None
    hooks._finish_ccv_trial(
        SimpleNamespace(task_id="E1-LS1-T2"),
        SimpleNamespace(normalized_score=0.0, verifier_passed=False),
        "candidate_attack",
    )
    decision = events.events_of_type("ccv_decision")[0]
    assert decision["status"] == "inconclusive_clean_failed"
    assert decision["valid_pair"] is False
    assert decision["confirmed_regression"] is False


def test_ccv_metrics_exclude_inconclusive_old_failure() -> None:
    report = compute_revision_safety(
        patch_events=[], replay_records=[],
        ccv_events=[
            {"event_type": "ccv_decision", "status": "inconclusive_old_failed",
             "valid_pair": False, "confirmed_regression": False},
            {"event_type": "ccv_decision", "status": "complete",
             "valid_pair": True, "confirmed_regression": False},
        ],
    )
    assert report.ccv_attempts == 2
    assert report.ccv_valid_pairs == 1


def test_cost_counter_includes_ccv_host_calls() -> None:
    client = SimpleNamespace(
        _n_calls=4, _tag="skill_author", model="openai/gpt-5.4-mini",
        _n_input_tokens=100, _n_output_tokens=20, _n_cache_tokens=0,
    )
    report = compute_cost(
        event_counts={"patch_proposed": 3, "trial_ended_learning": 3},
        n_tasks_attempted=3, n_tasks_passed=2, evaluation_sr=0.5,
        host_clients=[client],
    )
    assert report.n_llm_calls_estimate == 4
    assert report.llm_calls_per_task == 4 / 3


def test_harbor_instruction_refresh_supports_current_task_attribute(tmp_path) -> None:
    instruction = tmp_path / "instruction.md"
    instruction.write_text("fresh CCV instruction")
    task = SimpleNamespace(
        instruction="stale",
        paths=SimpleNamespace(instruction_path=instruction),
    )
    trial = SimpleNamespace(task=task)
    _refresh_instruction(trial, lambda text: text)
    assert task.instruction == "fresh CCV instruction"


def test_harbor_instruction_refresh_fails_closed_when_path_is_missing(tmp_path) -> None:
    task = SimpleNamespace(
        instruction="stale",
        paths=SimpleNamespace(instruction_path=tmp_path / "missing.md"),
    )
    with pytest.raises(RuntimeError, match="instruction integrity failure"):
        _refresh_instruction(SimpleNamespace(task=task), lambda text: text)


def test_retry_policy_is_infrastructure_only() -> None:
    assert is_retryable_infrastructure_exception("InvalidEncryptedContentError")
    assert is_retryable_infrastructure_exception("NetworkConnectionError")
    assert not is_retryable_infrastructure_exception("AgentTimeoutError")
    assert not is_retryable_infrastructure_exception("VerifierTimeoutError")
    assert not is_retryable_infrastructure_exception("NonZeroAgentExitCodeError")
    assert "AgentTimeoutError" not in RETRYABLE_INFRASTRUCTURE_EXCEPTIONS


def test_formal_run_default_uses_five_infrastructure_retries() -> None:
    from skillevolbench.schemas import RunConfig

    field = RunConfig.model_fields["harbor_infrastructure_max_retries"]
    assert field.default == 5


def test_formal_run_default_uses_uniform_600_second_agent_timeout() -> None:
    from skillevolbench.schemas import RunConfig

    field = RunConfig.model_fields["harbor_agent_timeout_sec"]
    assert field.default == 600.0


def test_lifelong_run_rejects_parallel_harbor_trials(tmp_path) -> None:
    with pytest.raises(ValueError):
        RunConfig(
            run_id="parallel-is-invalid",
            baseline=BaselineConfig(),
            strategy=StrategyConfig(),
            workspace_root=tmp_path,
            harbor_n_concurrent_trials=2,
        )


def test_exhausted_infrastructure_retries_abort_lifelong_chain() -> None:
    recorded = []
    runtime = SimpleNamespace(
        run_config=SimpleNamespace(harbor_infrastructure_max_retries=5),
        event_store=SimpleNamespace(
            record=lambda event_type, payload: recorded.append(
                (event_type, payload)
            )
        ),
        library=SimpleNamespace(compute_hash=lambda: "unchanged"),
    )
    hooks = SkillEvolBenchHooks(runtime, None, None, None)
    hooks._resolve_task = lambda _event: SimpleNamespace(task_id="E1-LS1-T2")
    hooks._runtime_basename = lambda _event: "E1-LS1-T2"
    event = SimpleNamespace(result=SimpleNamespace(
        exception_info=SimpleNamespace(
            exception_type="InvalidEncryptedContentError"
        )
    ))

    for _attempt in range(5):
        hooks._pre_state_cache["E1-LS1-T2"] = {}
        hooks._on_trial_ended_sync(event)

    hooks._pre_state_cache["E1-LS1-T2"] = {}
    with pytest.raises(RuntimeError, match="retry budget exhausted"):
        hooks._on_trial_ended_sync(event)

    assert recorded[-1][0] == "trial_infrastructure_retries_exhausted"
    assert recorded[-1][1]["total_attempts"] == 6


def test_failed_local_edit_falls_back_to_exactly_once_deferred_feedback(
    tmp_path,
) -> None:
    events = EventStore(tmp_path / "events")
    events.record("regularization_signal_bundle", {
        "task_id": "E1-LS1-T2",
        "abnormal_modules": ["complexity"],
        "complexity": {
            "enabled": True,
            "triggered": True,
            "initial_candidate": {"total": 15.29, "atom_count": 36},
            "budget": {"max_total": 13.5542, "max_atoms": 41},
        },
        "ccv": {},
        "actionable": True,
        "consumption_mode": "local_edit",
        "consumed": False,
    })
    events.record("regularization_local_edit_failed", {
        "task_id": "E1-LS1-T2",
        "reason": "complexity_budget_violation",
    })
    chain = ChainEvolution.__new__(ChainEvolution)
    chain.event_store = events

    first = chain._latest_regularization_feedback("E1-LS1")
    second = chain._latest_regularization_feedback("E1-LS1")

    assert "COMPLEXITY ANOMALY" in first
    assert second == ""
    consumed = events.events_of_type("regularization_feedback_consumed")
    assert len(consumed) == 1
    assert consumed[0]["source_task_id"] == "E1-LS1-T2"
    assert consumed[0]["consumed_by"] == "next_step_feedback"


def test_codex_proxy_encrypted_content_error_is_classified_for_retry() -> None:
    from agents_port.preinstalled import (
        CodexPreinstalled,
        InvalidEncryptedContentError,
    )

    patterns = [
        (pattern.pattern, pattern.exception)
        for pattern in CodexPreinstalled.ERROR_PATTERNS
    ]
    assert any(
        exception is InvalidEncryptedContentError
        and "invalid_encrypted_content" in pattern
        for pattern, exception in patterns
    )
