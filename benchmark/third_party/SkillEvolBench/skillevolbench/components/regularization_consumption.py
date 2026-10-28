"""Framework-agnostic consumption policies for SkillEvoReg signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RegularizationConsumptionMode = Literal["deferred", "local_edit"]


@dataclass(frozen=True)
class RegularizationSignalBundle:
    """One candidate's actionable Regularization findings.

    In the step-local protocol, an actually masked training proposal is also
    actionable: its source delta is consolidated before CCV. Complexity and
    CCV contribute only when their respective checks fire.
    """

    task_id: str
    abnormal_modules: tuple[str, ...] = ()
    complexity: dict[str, Any] = field(default_factory=dict)
    ccv: dict[str, Any] = field(default_factory=dict)
    dropout: dict[str, Any] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return bool(self.abnormal_modules)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_regularization_signal_bundle(
    *,
    task_id: str,
    complexity_projection: dict[str, Any] | None = None,
    ccv_decision: dict[str, Any] | None = None,
    dropout_report: dict[str, Any] | None = None,
) -> RegularizationSignalBundle:
    complexity = complexity_projection or {}
    ccv = ccv_decision or {}
    dropout = dropout_report or {}
    abnormal: list[str] = []
    if int(dropout.get("masked_count", 0) or 0) > 0:
        abnormal.append("dropout")
    if bool(complexity.get("triggered")):
        abnormal.append("complexity")
    if bool(ccv.get("confirmed_regression")):
        abnormal.append("ccv")
        if bool(ccv.get("allow_unmatched_source_rewrite")):
            abnormal.append("ccv_unmatched_rewrite")
    return RegularizationSignalBundle(
        task_id=task_id,
        abnormal_modules=tuple(abnormal),
        complexity=complexity if "complexity" in abnormal else {},
        ccv=ccv if "ccv" in abnormal else {},
        dropout=dropout if "dropout" in abnormal else {},
    )


def regularization_update_instruction(
    bundle: RegularizationSignalBundle,
    *, repair_scope_skill_ids: tuple[str, ...] | list[str] = (),
) -> str:
    """Render only the abnormal modules into one answer-free correction."""
    parts = [
        "GUARD UPDATE REQUIRED. Correct the current candidate once using only "
        "the abnormal Guard signals below. Preserve unrelated behavior. Do not "
        "encode task-specific literals, verifier output, hidden tests, probe "
        "answers, or one-off exceptions. Prefer revising/generalizing existing "
        "rules over appending narrow rules."
    ]
    if repair_scope_skill_ids:
        scope = ", ".join(str(item) for item in repair_scope_skill_ids)
        parts.append(
            "MODIFICATION SCOPE: the complete same-family library is context "
            f"only. Modify only these causally implicated skills: {scope}. "
            "Do not rewrite, delete, or create any sibling skill outside this "
            "scope."
        )
    if "dropout" in bundle.abnormal_modules:
        parts.append(
            "DROPOUT UPDATE: the source patch was learned from a partial "
            "training-time skill view and its hidden atoms have now been "
            "restored. Consolidate only the source delta against retained "
            "historical atoms. Preserve novel train-supported semantics; "
            "merge compensatory duplicates, rewrite conflicts, or choose "
            "NOOP. Dropout alone does not require compression."
        )
    if "complexity" in bundle.abnormal_modules:
        measured = bundle.complexity.get("initial_candidate") or {}
        budget = bundle.complexity.get("budget") or {}
        anchor = bundle.complexity.get("anchor") or {}
        measured_total = float(measured.get("total", 0.0))
        measured_atoms = int(measured.get("atom_count", 0))
        max_total = float(budget.get("max_total", 0.0))
        max_atoms = int(budget.get("max_atoms", 0))
        excess_total = max(0.0, measured_total - max_total)
        excess_atoms = max(0, measured_atoms - max_atoms)
        anchor_total = float(anchor.get("total", 0.0))
        anchor_atoms = int(anchor.get("atom_count", 0))
        parts.append(
            "COMPLEXITY ANOMALY: candidate exceeds the configured capacity "
            f"budget (total={measured_total:.4f}, atoms={measured_atoms}; "
            f"required total <= {max_total:.4f}, atoms <= {max_atoms}; "
            f"net reduction needed: total >= {excess_total:.4f}, atoms >= "
            f"{excess_atoms}). Refactor the current skill in this order: "
            "(1) merge semantically overlapping or duplicate atoms; "
            "(2) remove narrow, one-off, task-specific, or low-transfer rules; "
            "(3) replace clusters of similar rules with one general rule; "
            "(4) compress repeated explanations and examples. Do not solve "
            "this anomaly by appending new rules. Return coherent replacement "
            "blocks for only the source-delta regions while preserving unrelated "
            "capabilities. This is a trust-region repair of only the source "
            "patch's excess: do not compress the family below its pre-update "
            f"anchor (total >= {anchor_total:.4f}, atoms >= {anchor_atoms})."
        )
    if "ccv" in bundle.abnormal_modules:
        probe = bundle.ccv.get("probe") or {}
        parts.append(
            "SCAP REGRESSION: paired counterfactual evaluation confirmed a "
            f"normalized score drop of "
            f"{float(bundle.ccv.get('score_drop', 0.0) or 0.0):.4f}; "
            f"attack_type={str(probe.get('attack_type', 'unspecified'))[:120]}. "
            "Revise, generalize, or remove the incidental assumption implicated "
            "by this regression, and restore the transferable capability it "
            "damaged. Derive exactly one transferable procedural invariant "
            "from the counterexample and use it to replace, merge, or "
            "generalize the responsible existing rule. Do not append a new "
            "checklist branch or one-off exception for the probe. The repaired "
            "skill must not copy or restate probe-specific field names, file "
            "names, schema keys, literal values, concrete formats, or wording; "
            "describe the invariant at the capability level instead. "
            f"Suspected incidental assumption: "
            f"{str(probe.get('targeted_assumption', 'unspecified'))[:1200]}. "
            f"Predicted failure region: "
            f"{str(probe.get('predicted_failure_region', 'unspecified'))[:1200]}."
        )
    parts.append(
        "Return only the requested local region replacements; do not return a "
        "complete skill or discuss the diagnosis."
    )
    return "\n" + "\n".join(parts) + "\n"


__all__ = [
    "RegularizationConsumptionMode",
    "RegularizationSignalBundle",
    "build_regularization_signal_bundle",
    "regularization_update_instruction",
]
