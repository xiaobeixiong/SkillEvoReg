#!/usr/bin/env python3
"""One-call live smoke for SkillEvolBench atom-local Local editing."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from skillevolbench.components.regularization_consumption import (
    RegularizationSignalBundle,
    regularization_update_instruction,
)
from skillevolbench.components.local_repair import (
    apply_replacements,
    regularization_repair_regions,
    protected_content_hash,
    validate_repaired_document,
)
from skillevolbench.components.skill_author import LiteLLMClient, SkillAuthor


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[2]


def main() -> int:
    key = (
        PROJECT_ROOT / "benchmark/.secrets/evaluation_key.txt"
    ).read_text().strip()
    skill_id = "E0-LS0.generic-review"
    sibling_id = "E0-LS0.generic-evidence"
    anchor = (
        "---\nname: generic-review\ndescription: Review claims consistently.\n---\n\n"
        "# Workflow\n\n"
        "- Read the source before classifying a claim.\n"
        "- Preserve uncertainty when evidence is incomplete.\n"
    )
    candidate = anchor + (
        "- Add a separate exception for every observed wording pattern and "
        "repeat the same evidence checks in each exception.\n"
    )
    regions = regularization_repair_regions(anchor, candidate)
    family_texts = {
        skill_id: candidate,
        sibling_id: (
            "---\nname: generic-evidence\ndescription: Gather evidence.\n---\n\n"
            "# Workflow\n\n- Record source provenance.\n"
        ),
    }
    bundle = RegularizationSignalBundle(
        task_id="E0-LS0-T2",
        abnormal_modules=("complexity",),
        complexity={
            "initial_candidate": {"total": 42.62, "atom_count": 103},
            "budget": {"max_total": 45.54, "max_atoms": 102},
            "anchor": {"total": 39.60, "atom_count": 98},
        },
    )
    instruction = regularization_update_instruction(
        bundle, repair_scope_skill_ids=[skill_id],
    )
    client = LiteLLMClient(
        model="openai/gpt-5.4-mini",
        api_base="https://yibuapi.com/v1",
        api_key=key,
        max_tokens=8192,
        reasoning_effort="medium",
        tag="regularization_local_repair_smoke",
    )
    author = SkillAuthor(sync_call=client.__call__, async_call=client.acall)
    generated = author.propose_local_repair(
        task_id="E0-LS0-T2",
        regularization_instruction=instruction,
        family_skill_texts=family_texts,
        editable_regions_by_skill={skill_id: regions},
        abnormal_modules=("complexity",),
    )
    repaired = apply_replacements(
        candidate, regions, generated["replacements"][skill_id],
    )
    errors = validate_repaired_document(repaired)
    if errors:
        raise RuntimeError(f"SkillEvolBench local repair smoke invalid: {errors}")
    after_regions = regularization_repair_regions(anchor, repaired)
    if protected_content_hash(candidate, regions) != protected_content_hash(
        repaired, after_regions,
    ):
        raise RuntimeError("SkillEvolBench smoke changed protected source content")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "workspace/smokes" / f"regularization_local_repair_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "ok",
        "source": "synthetic_complexity_delta",
        "editable_region_count": len(regions),
        "replacement_region_ids": sorted(generated["replacements"][skill_id]),
        "delta_operations": generated.get("operations", []),
        "protected_content_unchanged": True,
        "change_summary": generated.get("change_summary", []),
    }
    (out_dir / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    (out_dir / "repaired_skill.md").write_text(repaired)
    (out_dir / "generated_replacements.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2)
    )
    print(json.dumps({**report, "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
