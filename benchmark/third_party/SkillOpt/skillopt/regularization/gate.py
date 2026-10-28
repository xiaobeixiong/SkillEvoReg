"""Pure SkillEvoReg accept/reject decision."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from skillopt.evaluation.gate import GateResult
from skillopt.regularization.ccv import CCVResult


@dataclass(frozen=True)
class RegularizationDecision:
    accepted: bool
    action: str
    reasons: list[str]
    delta_iid: float
    complexity_growth: float
    complexity_reduction: float
    lambda_complex: float
    complexity_cost: float
    complexity_triggered: bool
    complexity_bonus: float
    regularization_score: float
    placeholder_free: bool
    ccv_status: str
    valid_attacks: int
    oar: float
    tau_adv: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_regularized_gate(
    *,
    candidate_skill: str,
    candidate_score: float,
    current_skill: str,
    current_score: float,
    best_skill: str,
    best_score: float,
    best_step: int,
    global_step: int,
    normalized_complexity_growth: float,
    lambda_complex: float,
    placeholder_free: bool,
    ccv: CCVResult,
    tau_adv: float,
    ccv_required: bool,
    normalized_complexity_reduction: float = 0.0,
    complexity_bonus: float = 0.0,
) -> tuple[GateResult, RegularizationDecision]:
    delta_iid = float(candidate_score) - float(current_score)
    regularization_score = (
        delta_iid
        - float(lambda_complex) * float(normalized_complexity_growth)
        + float(complexity_bonus) * float(normalized_complexity_reduction)
    )
    complexity_cost = (
        float(lambda_complex) * float(normalized_complexity_growth)
    )
    score_without_complexity = (
        delta_iid
        + float(complexity_bonus) * float(normalized_complexity_reduction)
    )
    # Attribute a repair signal only when complexity independently changes a
    # non-rejecting regularized update into a rejection.  Validation-negative
    # candidates remain validation failures even if they also grew.
    complexity_triggered = bool(
        score_without_complexity >= 0.0
        and regularization_score <= 0.0
        and complexity_cost > 0.0
    )
    reasons: list[str] = []
    # Complexity reduction may break a quantized Val tie, but cannot buy its
    # way out of an observed Val regression.
    if delta_iid < 0.0:
        reasons.append("validation_regression")
    if regularization_score <= 0.0:
        reasons.append("negative_regularization_score")
    if complexity_triggered:
        reasons.append("complexity_regularized_rejection")
    # An optional probe may fail to construct a valid attack; that is not
    # evidence that the candidate regressed. Fail closed only when CCV is
    # explicitly required. A verifier-confirmed regression remains a veto.
    if ccv_required and ccv.status in {"error", "inconclusive"}:
        reasons.append("ccv_inconclusive")
    if ccv_required and ccv.status == "provider_unavailable":
        reasons.append("ccv_provider_unavailable")
    # CCV is an update-level regression veto, not a cross-step robustness
    # score. One verifier-confirmed old-pass/new-fail pair rejects the update.
    if ccv.regressions > 0:
        reasons.append("ccv_update_regression")

    accepted = not reasons
    if accepted:
        if candidate_score > best_score:
            action = "accept_new_best"
            gate = GateResult(
                action=action,
                current_skill=candidate_skill,
                current_score=candidate_score,
                best_skill=candidate_skill,
                best_score=candidate_score,
                best_step=global_step,
            )
        else:
            action = "accept"
            gate = GateResult(
                action=action,
                current_skill=candidate_skill,
                current_score=candidate_score,
                best_skill=best_skill,
                best_score=best_score,
                best_step=best_step,
            )
    else:
        action = (
            "regularization_defer"
            if "ccv_inconclusive" in reasons
            else "regularization_reject"
        )
        gate = GateResult(
            action="reject",
            current_skill=current_skill,
            current_score=current_score,
            best_skill=best_skill,
            best_score=best_score,
            best_step=best_step,
        )

    return gate, RegularizationDecision(
        accepted=accepted,
        action=action,
        reasons=reasons,
        delta_iid=delta_iid,
        complexity_growth=normalized_complexity_growth,
        complexity_reduction=normalized_complexity_reduction,
        lambda_complex=lambda_complex,
        complexity_cost=complexity_cost,
        complexity_triggered=complexity_triggered,
        complexity_bonus=complexity_bonus,
        regularization_score=regularization_score,
        placeholder_free=placeholder_free,
        ccv_status=ccv.status,
        valid_attacks=ccv.valid_attacks,
        oar=ccv.oar,
        tau_adv=tau_adv,
    )
