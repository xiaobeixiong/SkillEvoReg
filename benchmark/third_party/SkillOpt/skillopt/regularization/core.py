"""Runtime orchestration for SkillEvoReg."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import replace
from difflib import SequenceMatcher, unified_diff

from skillopt.regularization.atoms import AtomicSegmenter
from skillopt.regularization.complexity import (
    ComplexityReport,
    analyze_update_complexity,
    measure_complexity,
    normalized_growth,
)
from skillopt.regularization.epoch_feedback import run_epoch_feedback_agent
from skillopt.regularization.dropout import (
    apply_atomic_dropout,
    contains_placeholder,
    dropout_instruction,
    remove_placeholders,
)
from skillopt.regularization.gate import evaluate_regularized_gate
from skillopt.regularization.ccv import CCVResult, run_adapter_ccv
from skillopt.model import chat_optimizer, chat_target, chat_with_deployment
from skillopt.utils.json_utils import extract_json


_FAILURE_LOCATION_RE = re.compile(
    r"@(?P<location>(?:'[^']+'|[^!:@]+)!\$?[A-Z]{1,3}\$?\d+)",
    flags=re.IGNORECASE,
)


def _answer_free_failure_summary(result: dict | None) -> dict:
    """Keep causal locations and phase without exposing oracle cell values."""
    compact = result or {}
    reason = str(compact.get("fail_reason", "") or "")
    locations = list(dict.fromkeys(
        match.group("location")
        for match in _FAILURE_LOCATION_RE.finditer(reason)
    ))
    return {
        "phase": str(compact.get("phase", "") or ""),
        "locations": locations,
    }


def scheduled_complexity_lambda(
    *,
    step: int,
    total_steps: int,
    minimum: float,
    maximum: float,
    scheduler: str,
) -> float:
    """Return the complexity weight for a 1-indexed optimization step."""
    mode = str(scheduler or "constant").lower()
    if mode == "constant":
        return float(maximum)
    if mode != "inverse_cosine":
        raise ValueError(
            "regularization.lambda_complex_scheduler must be "
            f"'constant' or 'inverse_cosine', got {scheduler!r}"
        )
    horizon = max(1, int(total_steps))
    progress = min(max(float(step) / horizon, 0.0), 1.0)
    return float(minimum) + 0.5 * (float(maximum) - float(minimum)) * (
        1.0 - math.cos(math.pi * progress)
    )


def scheduled_epoch_value(
    *,
    step: int,
    steps_per_epoch: int,
    minimum: float,
    maximum: float,
    scheduler: str,
    warmup_epochs: int = 0,
    ramp_epochs: int = 1,
) -> float:
    """Schedule any regularization strength without coupling it to an environment."""
    mode = str(scheduler or "constant").lower()
    if mode == "constant":
        return float(maximum)
    if mode != "epoch_linear":
        raise ValueError(
            "regularization progressive scheduler must be 'constant' or "
            f"'epoch_linear', got {scheduler!r}"
        )
    epoch = (max(1, int(step)) - 1) // max(1, int(steps_per_epoch)) + 1
    active_epochs = max(0, epoch - max(0, int(warmup_epochs)))
    progress = min(active_epochs / max(1, int(ramp_epochs)), 1.0)
    return float(minimum) + (float(maximum) - float(minimum)) * progress


class SkillEvoReg:
    """Composable regularization installed at SkillOpt's candidate update boundary."""

    def __init__(self, cfg: dict, *, initial_skill: str, adapter) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.enabled = bool(cfg.get("regularization_enabled", False))
        self.steps_per_epoch = max(1, int(cfg.get("steps_per_epoch", 1)))
        self.num_epochs = max(1, int(cfg.get("num_epochs", 1)))
        self.total_steps = self.num_epochs * self.steps_per_epoch
        self.complexity_enabled = bool(
            cfg.get("regularization_complexity_enabled", True)
        )
        self.dropout_enabled = bool(cfg.get("regularization_dropout_enabled", True))
        self.dropout_probability = float(cfg.get("regularization_dropout_probability", 0.1))
        self.dropout_probability_min = float(
            cfg.get("regularization_dropout_probability_min", self.dropout_probability)
        )
        self.dropout_probability_max = float(
            cfg.get("regularization_dropout_probability_max", self.dropout_probability)
        )
        self.dropout_scheduler = str(
            cfg.get("regularization_dropout_scheduler", "constant") or "constant"
        )
        self.dropout_warmup_epochs = max(
            0, int(cfg.get("regularization_dropout_warmup_epochs", 0) or 0)
        )
        self.dropout_ramp_epochs = max(
            1, int(cfg.get("regularization_dropout_ramp_epochs", 1) or 1)
        )
        self.dropout_max_atoms = int(cfg.get("regularization_dropout_max_atoms", 0) or 0)
        self.dropout_ensure_at_least_one = bool(
            cfg.get("regularization_dropout_ensure_at_least_one", False)
        )
        self.feedback_enabled = bool(cfg.get("regularization_feedback_enabled", False))
        self.feedback_author = str(
            cfg.get("regularization_feedback_author", "heuristic") or "heuristic"
        ).strip().lower()
        if self.feedback_author not in {"heuristic", "target", "teacher"}:
            raise ValueError(
                "regularization.feedback_author must be 'heuristic', 'target', or "
                f"'teacher', got {self.feedback_author!r}"
            )
        self.feedback_teacher_model = str(
            cfg.get("regularization_feedback_teacher_model", "") or ""
        ).strip()
        if self.feedback_author == "teacher" and not self.feedback_teacher_model:
            raise ValueError(
                "regularization.feedback_teacher_model is required when "
                "regularization.feedback_author='teacher'"
            )
        self.feedback_max_completion_tokens = max(
            256, int(cfg.get("regularization_feedback_max_completion_tokens", 2000) or 2000)
        )
        self.feedback_retries = max(
            1, int(cfg.get("regularization_feedback_retries", 2) or 2)
        )
        self.feedback_timeout = max(
            1, int(cfg.get("regularization_feedback_timeout", 120) or 120)
        )
        self.epoch_agents_enabled = bool(
            cfg.get("regularization_epoch_agents_enabled", False)
        )
        self.ccv_granularity = str(
            cfg.get(
                "regularization_ccv_granularity",
                "epoch" if self.epoch_agents_enabled else "step",
            ) or "step"
        ).strip().lower()
        if self.ccv_granularity not in {"step", "epoch"}:
            raise ValueError(
                "regularization.ccv_granularity must be 'step' or 'epoch', got "
                f"{self.ccv_granularity!r}"
            )
        legacy_lambda = float(cfg.get("regularization_lambda_complex", 0.1))
        self.lambda_complex_min = float(
            cfg.get("regularization_lambda_complex_min", legacy_lambda)
        )
        self.lambda_complex_max = float(
            cfg.get("regularization_lambda_complex_max", legacy_lambda)
        )
        self.lambda_complex_scheduler = str(
            cfg.get("regularization_lambda_complex_scheduler", "constant")
        )
        self.lambda_complex = self.lambda_complex_max
        self.complexity_bonus_min = float(
            cfg.get("regularization_complexity_bonus_min", 0.0)
        )
        self.complexity_bonus_max = float(
            cfg.get("regularization_complexity_bonus_max", 0.0)
        )
        self.complexity_bonus_scheduler = str(
            cfg.get("regularization_complexity_bonus_scheduler", "inverse_cosine")
        )
        self.complexity_reduction_cap = max(
            0.0, float(cfg.get("regularization_complexity_reduction_cap", 0.20))
        )
        self.complexity_free_growth = float(
            cfg.get("regularization_complexity_free_growth", 0.1)
        )
        self.tau_adv = float(cfg.get("regularization_tau_adv", 0.0))
        self.ccv_enabled = bool(cfg.get("regularization_ccv_enabled", True))
        self.ccv_required = bool(cfg.get("regularization_ccv_required", False))
        self.ccv_provider = str(
            cfg.get("regularization_ccv_provider", "legacy") or "legacy"
        )
        self.ccv_source_split = str(cfg.get("regularization_ccv_source_split", "valid_seen"))
        self.ccv_source_size = int(cfg.get("regularization_ccv_source_size", 0) or 0)
        self.ccv_source_size_min = int(
            cfg.get("regularization_ccv_source_size_min", self.ccv_source_size) or 0
        )
        self.ccv_scheduler = str(
            cfg.get("regularization_ccv_scheduler", "constant") or "constant"
        )
        self.ccv_warmup_epochs = max(
            0, int(cfg.get("regularization_ccv_warmup_epochs", 0) or 0)
        )
        self.ccv_ramp_epochs = max(
            1, int(cfg.get("regularization_ccv_ramp_epochs", 1) or 1)
        )
        self.ccv_min_qualified_sources = int(
            cfg.get("regularization_ccv_min_qualified_sources", 1) or 1
        )
        self.ccv_qualification_passes = int(
            cfg.get("regularization_ccv_qualification_passes", 1) or 1
        )
        self.ccv_inconclusive_retries = max(
            0, int(cfg.get("regularization_ccv_inconclusive_retries", 0) or 0)
        )
        self.ccv_max_patterns = int(cfg.get("regularization_ccv_max_patterns", 3) or 3)
        self.ccv_attacks_per_pattern = int(
            cfg.get("regularization_ccv_attacks_per_pattern", 2) or 2
        )
        self.ccv_max_attacks = int(cfg.get("regularization_ccv_max_attacks", 6) or 6)
        self.ccv_max_attacks_min = int(
            cfg.get("regularization_ccv_max_attacks_min", self.ccv_max_attacks) or 0
        )
        self.ccv_history_items = int(cfg.get("regularization_ccv_history_items", 12) or 12)
        self.ccv_attacker_model = str(cfg.get("regularization_ccv_attacker_model", "") or "")
        self.ccv_reasoning_effort = str(
            cfg.get("regularization_ccv_reasoning_effort", "medium") or ""
        )
        self.ccv_builder_reasoning_effort = str(
            cfg.get("regularization_ccv_builder_reasoning_effort", "low") or ""
        )
        self.ccv_max_completion_tokens = int(
            cfg.get("regularization_ccv_max_completion_tokens", 12000) or 12000
        )
        self.ccv_timeout = int(cfg.get("regularization_ccv_timeout", 300) or 300)
        self.ccv_retries = int(cfg.get("regularization_ccv_retries", 3) or 3)
        self.ccv_agent_max_turns = int(
            cfg.get("regularization_ccv_agent_max_turns", 96) or 96
        )
        self.ccv_agent_timeout = int(
            cfg.get("regularization_ccv_agent_timeout", 3600) or 3600
        )
        self.ccv_agent_max_thinking_tokens = int(
            cfg.get("regularization_ccv_agent_max_thinking_tokens", 8192) or 8192
        )
        self.ccv_agent_base_url = str(
            cfg.get("regularization_ccv_agent_base_url", "https://yibuapi.com")
            or "https://yibuapi.com"
        )
        self.ccv_agent_api_key_env = str(
            cfg.get("regularization_ccv_agent_api_key_env", "ANTHROPIC_API_KEY")
            or "ANTHROPIC_API_KEY"
        )
        self.epoch_feedback_agent_enabled = bool(
            cfg.get("regularization_epoch_feedback_agent_enabled", self.epoch_agents_enabled)
        )
        self.epoch_feedback_agent_provider = str(
            cfg.get("regularization_epoch_feedback_agent_provider", "claude_agent_sdk")
            or "claude_agent_sdk"
        ).strip().lower()
        self.epoch_feedback_agent_model = str(
            cfg.get("regularization_epoch_feedback_agent_model", self.ccv_attacker_model) or self.ccv_attacker_model
        ).strip()
        self.epoch_feedback_agent_effort = str(
            cfg.get("regularization_epoch_feedback_agent_effort", "medium") or "medium"
        )
        self.epoch_feedback_agent_max_turns = max(
            4, int(cfg.get("regularization_epoch_feedback_agent_max_turns", 24) or 24)
        )
        self.epoch_feedback_agent_timeout = max(
            60, int(cfg.get("regularization_epoch_feedback_agent_timeout", 1200) or 1200)
        )
        self.epoch_feedback_agent_max_thinking_tokens = max(
            1024, int(cfg.get("regularization_epoch_feedback_agent_max_thinking_tokens", 8192) or 8192)
        )
        self.epoch_feedback_page_size = max(
            1, int(cfg.get("regularization_epoch_feedback_page_size", 10) or 10)
        )
        self.epoch_feedback_retries = max(
            0, int(cfg.get("regularization_epoch_feedback_retries", 1) or 0)
        )
        self.epoch_feedback_required = bool(
            cfg.get("regularization_epoch_feedback_required", False)
        )
        if self.epoch_feedback_agent_enabled and not self.epoch_feedback_agent_model:
            raise ValueError("regularization.epoch_feedback_agent_model is required when the epoch Feedback Agent is enabled")
        self.segmenter = AtomicSegmenter(
            str(cfg.get("regularization_segmenter", "llm")),
            llm_call=chat_optimizer,
            retries=int(cfg.get("regularization_segmenter_retries", 2) or 2),
            max_completion_tokens=int(
                cfg.get("regularization_segmenter_max_completion_tokens", 65536)
                or 65536
            ),
            cache_dir=os.path.join(
                str(cfg.get("out_root", "") or "."),
                "regularization",
                "atom_cache",
            ),
        )
        self.token_scale = float(cfg.get("regularization_token_scale", 100.0))
        self.atom_cost = float(cfg.get("regularization_atom_cost", 0.2))
        self.enable_code_complexity = bool(
            cfg.get("regularization_enable_code_complexity", True)
        )
        self.initial_complexity = self.measure(initial_skill)

    def measure(self, skill: str) -> ComplexityReport:
        return measure_complexity(
            skill,
            self.segmenter,
            token_scale=self.token_scale,
            atom_cost=self.atom_cost,
            enable_code_complexity=self.enable_code_complexity,
        )

    def prepare_proposal(
        self, skill: str, *, seed: int, step: int = 1
    ) -> tuple[str, dict, str]:
        old_complexity = self.measure(skill)
        if not self.enabled or not self.dropout_enabled:
            report = {
                "enabled": False,
                "masked_count": 0,
                "old_complexity": old_complexity.to_dict(),
            }
            return skill, report, ""
        atoms, _ = self.segmenter.segment(skill)
        active_probability = scheduled_epoch_value(
            step=step,
            steps_per_epoch=self.steps_per_epoch,
            minimum=self.dropout_probability_min,
            maximum=self.dropout_probability_max,
            scheduler=self.dropout_scheduler,
            warmup_epochs=self.dropout_warmup_epochs,
            ramp_epochs=self.dropout_ramp_epochs,
        )
        result = apply_atomic_dropout(
            skill,
            atoms,
            probability=active_probability,
            seed=seed,
            max_atoms=self.dropout_max_atoms,
            ensure_at_least_one=self.dropout_ensure_at_least_one,
        )
        report = {
            "enabled": True,
            **result.to_dict(),
            "old_complexity": old_complexity.to_dict(),
            "scheduler": self.dropout_scheduler,
            "warmup_epochs": self.dropout_warmup_epochs,
            "ramp_epochs": self.dropout_ramp_epochs,
        }
        return (
            result.proposal_skill,
            report,
            dropout_instruction(len(result.masked_atoms)),
        )

    def finalize_candidate(
        self, skill: str, *, dropout_report: dict | None = None
    ) -> tuple[str, dict]:
        """Remove synthetic dropout markers without rewriting user content."""
        cleaned, removed_count = remove_placeholders(skill)
        candidate_atoms, _ = self.segmenter.segment(cleaned)
        candidate_spans = [
            " ".join(atom.span.lower().split()) for atom in candidate_atoms
            if atom.span.strip()
        ]
        outcomes = []
        for masked in (dropout_report or {}).get("masked_atoms", []) or []:
            original = " ".join(str(masked.get("original_span", "")).lower().split())
            best = max(
                (SequenceMatcher(None, original, span).ratio() for span in candidate_spans),
                default=0.0,
            )
            if original and original in " ".join(cleaned.lower().split()):
                outcome = "reconstructed_exact"
            elif best >= 0.82:
                outcome = "reconstructed_or_compressed"
            elif best >= 0.55:
                outcome = "modified"
            else:
                outcome = "deleted_or_absorbed"
            outcomes.append({
                "mask_id": masked.get("mask_id"),
                "atom_id": masked.get("atom_id"),
                "outcome": outcome,
                "closest_candidate_similarity": round(best, 4),
            })
        return cleaned, {
            "placeholder_fallback": "delete_marker_only",
            "removed_placeholders": removed_count,
            "placeholder_remaining": contains_placeholder(cleaned),
            "semantic_outcomes": outcomes,
            "semantic_outcome_counts": {
                label: sum(1 for row in outcomes if row["outcome"] == label)
                for label in sorted({row["outcome"] for row in outcomes})
            },
        }

    def evaluate(
        self,
        *,
        old_skill: str,
        candidate_skill: str,
        candidate_score: float,
        current_score: float,
        best_skill: str,
        best_score: float,
        best_step: int,
        global_step: int,
        seed: int,
        out_dir: str,
        trajectory_context: dict | None = None,
        run_ccv: bool = True,
    ):
        old_report = self.measure(old_skill)
        candidate_report = self.measure(candidate_skill)
        growth = normalized_growth(
            old_report,
            candidate_report,
            free_growth_budget=self.complexity_free_growth,
        )
        context = trajectory_context or {}
        update_diagnostics = analyze_update_complexity(
            old_report,
            candidate_report,
            edits=context.get("ranked_edits", []),
        )

        placeholder_free = not contains_placeholder(candidate_skill)
        delta_iid = float(candidate_score) - float(current_score)
        lambda_complex = scheduled_complexity_lambda(
            step=global_step,
            total_steps=self.total_steps,
            minimum=self.lambda_complex_min,
            maximum=self.lambda_complex_max,
            scheduler=self.lambda_complex_scheduler,
        )
        relative_reduction = max(
            0.0,
            (old_report.total - candidate_report.total)
            / max(old_report.total, 1.0),
        )
        capped_reduction = min(relative_reduction, self.complexity_reduction_cap)
        complexity_bonus = scheduled_complexity_lambda(
            step=global_step,
            total_steps=self.total_steps,
            minimum=self.complexity_bonus_min,
            maximum=self.complexity_bonus_max,
            scheduler=self.complexity_bonus_scheduler,
        )
        pre_regularization_score = (
            delta_iid - lambda_complex * growth["normalized_growth"]
            + complexity_bonus * capped_reduction
        )

        active_ccv_source_size = max(0, int(round(scheduled_epoch_value(
            step=global_step,
            steps_per_epoch=self.steps_per_epoch,
            minimum=self.ccv_source_size_min,
            maximum=self.ccv_source_size,
            scheduler=self.ccv_scheduler,
            warmup_epochs=self.ccv_warmup_epochs,
            ramp_epochs=self.ccv_ramp_epochs,
        ))))
        active_ccv_max_attacks = max(0, int(round(scheduled_epoch_value(
            step=global_step,
            steps_per_epoch=self.steps_per_epoch,
            minimum=self.ccv_max_attacks_min,
            maximum=self.ccv_max_attacks,
            scheduler=self.ccv_scheduler,
            warmup_epochs=self.ccv_warmup_epochs,
            ramp_epochs=self.ccv_ramp_epochs,
        ))))
        ccv_active = self.ccv_enabled and (
            self.ccv_scheduler.lower() == "constant"
            or (
                active_ccv_source_size > 0
                and active_ccv_max_attacks > 0
            )
        )

        ccv = CCVResult(status="disabled")
        # CCV is the final veto, so only attack candidates that already clear
        # the strict regularized Val gate.
        if not run_ccv:
            ccv = CCVResult(status="preselection_pipeline_complete")
        elif self.ccv_granularity == "epoch":
            ccv = CCVResult(status="deferred_epoch")
        elif ccv_active and pre_regularization_score > 0.0:
            ccv_dir = os.path.join(out_dir, "ccv")
            os.makedirs(ccv_dir, exist_ok=True)
            attempt_rows = []
            for attempt in range(self.ccv_inconclusive_retries + 1):
                attempt_dir = (
                    ccv_dir if attempt == 0
                    else os.path.join(ccv_dir, f"retry_{attempt:02d}")
                )
                ccv = run_adapter_ccv(
                    self.adapter,
                    old_skill=old_skill,
                    candidate_skill=candidate_skill,
                    step=global_step,
                    seed=seed + attempt * 7919,
                    out_dir=attempt_dir,
                    source_split=self.ccv_source_split,
                    source_size=active_ccv_source_size,
                    min_qualified_sources=(
                        min(
                            self.ccv_min_qualified_sources,
                            active_ccv_source_size,
                        )
                        if active_ccv_source_size > 0
                        else self.ccv_min_qualified_sources
                    ),
                    qualification_passes=self.ccv_qualification_passes,
                    max_patterns=self.ccv_max_patterns,
                    attacks_per_pattern=self.ccv_attacks_per_pattern,
                    max_attacks=active_ccv_max_attacks,
                    history_items=self.ccv_history_items,
                    attacker_model=self.ccv_attacker_model,
                    reasoning_effort=self.ccv_reasoning_effort,
                    builder_reasoning_effort=self.ccv_builder_reasoning_effort,
                    max_completion_tokens=self.ccv_max_completion_tokens,
                    timeout=self.ccv_timeout,
                    retries=self.ccv_retries,
                    trajectory_context=trajectory_context,
                    provider=self.ccv_provider,
                    agent_max_turns=self.ccv_agent_max_turns,
                    agent_timeout=self.ccv_agent_timeout,
                    agent_max_thinking_tokens=self.ccv_agent_max_thinking_tokens,
                    agent_base_url=self.ccv_agent_base_url,
                    agent_api_key_env=self.ccv_agent_api_key_env,
                )
                attempt_rows.append({
                    "attempt": attempt + 1,
                    "status": ccv.status,
                    "error": ccv.error,
                    "valid_attacks": ccv.valid_attacks,
                    "regressions": ccv.regressions,
                })
                if ccv.status not in {"error", "inconclusive"}:
                    break
            ccv = replace(
                ccv,
                usage={**ccv.usage, "regularization_attempts": attempt_rows},
            )
        elif self.ccv_enabled and not ccv_active:
            ccv = CCVResult(status="scheduled_off")
        elif self.ccv_enabled:
            ccv = CCVResult(status="skipped_precheck")

        gate, decision = evaluate_regularized_gate(
            candidate_skill=candidate_skill,
            candidate_score=candidate_score,
            current_skill=old_skill,
            current_score=current_score,
            best_skill=best_skill,
            best_score=best_score,
            best_step=best_step,
            global_step=global_step,
            normalized_complexity_growth=growth["normalized_growth"],
            normalized_complexity_reduction=capped_reduction,
            lambda_complex=lambda_complex,
            complexity_bonus=complexity_bonus,
            placeholder_free=placeholder_free,
            ccv=ccv,
            tau_adv=self.tau_adv,
            ccv_required=self.ccv_required and ccv_active,
        )
        telemetry = {
            "decision": decision.to_dict(),
            "complexity": {
                "initial": self.initial_complexity.to_dict(),
                "old": old_report.to_dict(),
                "candidate": candidate_report.to_dict(),
                "growth": growth,
                "lambda_complex": lambda_complex,
                "lambda_complex_scheduler": self.lambda_complex_scheduler,
                "lambda_complex_min": self.lambda_complex_min,
                "lambda_complex_max": self.lambda_complex_max,
                "lambda_complex_total_steps": self.total_steps,
                "relative_reduction": relative_reduction,
                "capped_reduction": capped_reduction,
                "complexity_reduction_cap": self.complexity_reduction_cap,
                "complexity_bonus": complexity_bonus,
                "complexity_bonus_scheduler": self.complexity_bonus_scheduler,
                "complexity_bonus_min": self.complexity_bonus_min,
                "complexity_bonus_max": self.complexity_bonus_max,
                "update_diagnostics": update_diagnostics,
            },
            "ccv": ccv.to_dict(),
            "ccv_schedule": {
                "scheduler": self.ccv_scheduler,
                "warmup_epochs": self.ccv_warmup_epochs,
                "ramp_epochs": self.ccv_ramp_epochs,
                "source_size": active_ccv_source_size,
                "max_attacks": active_ccv_max_attacks,
                "active": ccv_active,
            },
        }
        with open(os.path.join(out_dir, "skillevoreg.json"), "w", encoding="utf-8") as f:
            json.dump(telemetry, f, ensure_ascii=False, indent=2)
        return gate, telemetry

    def run_epoch_ccv(
        self,
        *,
        old_skill: str,
        candidate_skill: str,
        global_step: int,
        seed: int,
        out_dir: str,
        trajectory_context: dict | None = None,
    ) -> tuple[CCVResult, dict]:
        """Run the cumulative CCV veto once for an epoch transaction."""
        if old_skill == candidate_skill:
            return CCVResult(status="no_epoch_delta"), {
                "active": False, "reason": "no_epoch_delta",
                "source_size": 0, "max_attacks": 0,
            }
        source_size = max(0, int(round(scheduled_epoch_value(
            step=global_step, steps_per_epoch=self.steps_per_epoch,
            minimum=self.ccv_source_size_min, maximum=self.ccv_source_size,
            scheduler=self.ccv_scheduler, warmup_epochs=self.ccv_warmup_epochs,
            ramp_epochs=self.ccv_ramp_epochs,
        ))))
        max_attacks = max(0, int(round(scheduled_epoch_value(
            step=global_step, steps_per_epoch=self.steps_per_epoch,
            minimum=self.ccv_max_attacks_min, maximum=self.ccv_max_attacks,
            scheduler=self.ccv_scheduler, warmup_epochs=self.ccv_warmup_epochs,
            ramp_epochs=self.ccv_ramp_epochs,
        ))))
        active = self.ccv_enabled and (
            self.ccv_scheduler.lower() == "constant"
            or (source_size > 0 and max_attacks > 0)
        )
        schedule = {
            "scheduler": self.ccv_scheduler,
            "warmup_epochs": self.ccv_warmup_epochs,
            "ramp_epochs": self.ccv_ramp_epochs,
            "source_size": source_size,
            "max_attacks": max_attacks,
            "active": active,
        }
        if not self.ccv_enabled:
            return CCVResult(status="disabled"), schedule
        if not active:
            return CCVResult(status="scheduled_off"), schedule
        os.makedirs(out_dir, exist_ok=True)
        result = CCVResult(status="error", error="CCV did not run")
        attempts = []
        for attempt in range(self.ccv_inconclusive_retries + 1):
            attempt_dir = out_dir if attempt == 0 else os.path.join(out_dir, f"retry_{attempt:02d}")
            result = run_adapter_ccv(
                self.adapter, old_skill=old_skill, candidate_skill=candidate_skill,
                step=global_step, seed=seed + attempt * 7919, out_dir=attempt_dir,
                source_split=self.ccv_source_split, source_size=source_size,
                min_qualified_sources=(
                    min(self.ccv_min_qualified_sources, source_size)
                    if source_size > 0 else self.ccv_min_qualified_sources
                ),
                qualification_passes=self.ccv_qualification_passes,
                max_patterns=self.ccv_max_patterns,
                attacks_per_pattern=self.ccv_attacks_per_pattern,
                max_attacks=max_attacks, history_items=self.ccv_history_items,
                attacker_model=self.ccv_attacker_model,
                reasoning_effort=self.ccv_reasoning_effort,
                builder_reasoning_effort=self.ccv_builder_reasoning_effort,
                max_completion_tokens=self.ccv_max_completion_tokens,
                timeout=self.ccv_timeout, retries=self.ccv_retries,
                trajectory_context=trajectory_context or {}, provider=self.ccv_provider,
                agent_max_turns=self.ccv_agent_max_turns,
                agent_timeout=self.ccv_agent_timeout,
                agent_max_thinking_tokens=self.ccv_agent_max_thinking_tokens,
                agent_base_url=self.ccv_agent_base_url,
                agent_api_key_env=self.ccv_agent_api_key_env,
            )
            attempts.append({
                "attempt": attempt + 1, "status": result.status,
                "error": result.error, "valid_attacks": result.valid_attacks,
                "regressions": result.regressions,
            })
            if result.status not in {"error", "inconclusive"}:
                break
        result = replace(result, usage={**result.usage, "regularization_attempts": attempts})
        with open(os.path.join(out_dir, "epoch_ccv_result.json"), "w", encoding="utf-8") as handle:
            json.dump({"schedule": schedule, "result": result.to_dict()}, handle, ensure_ascii=False, indent=2)
        return result, schedule

    def run_step_ccv(
        self,
        *,
        old_skill: str,
        candidate_skill: str,
        global_step: int,
        seed: int,
        out_dir: str,
        trajectory_context: dict | None = None,
    ) -> tuple[CCVResult, dict]:
        """Run one bounded Train-source CCV arm before final Val.

        The provider performs clean old/new qualification followed by one
        attacked old/new pair.  This method deliberately caps both source and
        attack counts at one for the step-local protocol.
        """
        schedule = {
            "scheduler": "step_local_fixed",
            "source_size": 1,
            "max_attacks": 1,
            "qualification_passes": 1,
            "active": bool(self.ccv_enabled and old_skill != candidate_skill),
        }
        if not self.ccv_enabled:
            return CCVResult(status="disabled"), schedule
        if old_skill == candidate_skill:
            return CCVResult(status="no_step_delta"), {
                **schedule, "active": False, "reason": "no_step_delta",
            }
        os.makedirs(out_dir, exist_ok=True)
        result = CCVResult(status="error", error="step-local CCV did not run")
        attempts = []
        for attempt in range(self.ccv_inconclusive_retries + 1):
            attempt_dir = (
                out_dir if attempt == 0
                else os.path.join(out_dir, f"retry_{attempt:02d}")
            )
            result = run_adapter_ccv(
                self.adapter,
                old_skill=old_skill,
                candidate_skill=candidate_skill,
                step=global_step,
                seed=seed + attempt * 7919,
                out_dir=attempt_dir,
                source_split="train",
                source_size=1,
                min_qualified_sources=1,
                qualification_passes=1,
                max_patterns=1,
                attacks_per_pattern=1,
                max_attacks=1,
                history_items=self.ccv_history_items,
                attacker_model=self.ccv_attacker_model,
                reasoning_effort=self.ccv_reasoning_effort,
                builder_reasoning_effort=self.ccv_builder_reasoning_effort,
                max_completion_tokens=self.ccv_max_completion_tokens,
                timeout=self.ccv_timeout,
                retries=self.ccv_retries,
                trajectory_context={
                    **(trajectory_context or {}),
                    "force_old_clean_evaluation": True,
                },
                provider=self.ccv_provider,
                agent_max_turns=self.ccv_agent_max_turns,
                agent_timeout=self.ccv_agent_timeout,
                agent_max_thinking_tokens=self.ccv_agent_max_thinking_tokens,
                agent_base_url=self.ccv_agent_base_url,
                agent_api_key_env=self.ccv_agent_api_key_env,
            )
            attempts.append({
                "attempt": attempt + 1,
                "status": result.status,
                "error": result.error,
                "valid_attacks": result.valid_attacks,
                "regressions": result.regressions,
            })
            if result.status not in {"error", "inconclusive"}:
                break
        result = replace(
            result, usage={**result.usage, "regularization_attempts": attempts},
        )
        with open(
            os.path.join(out_dir, "step_ccv_result.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(
                {"schedule": schedule, "result": result.to_dict()},
                handle, ensure_ascii=False, indent=2,
            )
        return result, schedule

    def run_epoch_feedback(self, *, evidence: dict, out_dir: str) -> dict:
        """Run one generic, bounded Feedback Analyst Agent for the next epoch."""
        if not self.epoch_feedback_agent_enabled:
            return {"status": "disabled", "next_epoch_guidance": {}}
        return run_epoch_feedback_agent(
            evidence=evidence, out_dir=out_dir, model=self.epoch_feedback_agent_model,
            effort=self.epoch_feedback_agent_effort,
            max_turns=self.epoch_feedback_agent_max_turns,
            timeout=self.epoch_feedback_agent_timeout,
            max_thinking_tokens=self.epoch_feedback_agent_max_thinking_tokens,
            page_size=self.epoch_feedback_page_size,
            retries=self.epoch_feedback_retries,
            provider=self.epoch_feedback_agent_provider,
            base_url=self.ccv_agent_base_url,
            api_key_env=self.ccv_agent_api_key_env,
        )

    def build_epoch_module_feedback(
        self,
        *,
        old_skill: str,
        candidate_skill: str,
        current_score: float,
        candidate_score: float,
        global_step: int,
        ccv: CCVResult,
        edit_budget: int | None = None,
    ) -> tuple[dict | None, dict]:
        """Attribute cumulative epoch anomalies with the normal Regularization score.

        The epoch analyst is downstream of this controller decision.  It may
        explain an already-confirmed module anomaly, but it cannot invent an
        anomaly or trigger an unconditional rewrite.
        """
        old_report = self.measure(old_skill)
        candidate_report = self.measure(candidate_skill)
        growth = normalized_growth(
            old_report,
            candidate_report,
            free_growth_budget=self.complexity_free_growth,
        )
        relative_reduction = max(
            0.0,
            (old_report.total - candidate_report.total)
            / max(old_report.total, 1.0),
        )
        capped_reduction = min(
            relative_reduction, self.complexity_reduction_cap,
        )
        lambda_complex = scheduled_complexity_lambda(
            step=global_step,
            total_steps=self.total_steps,
            minimum=self.lambda_complex_min,
            maximum=self.lambda_complex_max,
            scheduler=self.lambda_complex_scheduler,
        )
        complexity_bonus = scheduled_complexity_lambda(
            step=global_step,
            total_steps=self.total_steps,
            minimum=self.complexity_bonus_min,
            maximum=self.complexity_bonus_max,
            scheduler=self.complexity_bonus_scheduler,
        )
        _, decision = evaluate_regularized_gate(
            candidate_skill=candidate_skill,
            candidate_score=candidate_score,
            current_skill=old_skill,
            current_score=current_score,
            best_skill=old_skill,
            best_score=current_score,
            best_step=max(0, global_step - self.steps_per_epoch),
            global_step=global_step,
            normalized_complexity_growth=growth["normalized_growth"],
            normalized_complexity_reduction=capped_reduction,
            lambda_complex=lambda_complex,
            complexity_bonus=complexity_bonus,
            placeholder_free=not contains_placeholder(candidate_skill),
            ccv=ccv,
            tau_adv=self.tau_adv,
            # Provider/inconclusive status is telemetry, not an actionable
            # module. Only confirmed CCV regressions enter feedback.
            ccv_required=False,
        )
        telemetry = {
            "decision": decision.to_dict(),
            "complexity": {
                "initial": self.initial_complexity.to_dict(),
                "old": old_report.to_dict(),
                "candidate": candidate_report.to_dict(),
                "growth": growth,
                "lambda_complex": lambda_complex,
                "relative_reduction": relative_reduction,
                "capped_reduction": capped_reduction,
                "complexity_bonus": complexity_bonus,
                "update_diagnostics": analyze_update_complexity(
                    old_report, candidate_report, edits=[],
                ),
            },
            "ccv": ccv.to_dict(),
            "scope": "epoch_cumulative",
        }
        feedback = self.build_feedback(
            telemetry,
            edit_budget=edit_budget,
            current_skill=old_skill,
            candidate_skill=candidate_skill,
        )
        return feedback, telemetry

    def build_feedback(
        self,
        telemetry: dict,
        *,
        dropout_report: dict | None = None,
        edit_budget: int | None = None,
        current_skill: str = "",
        candidate_skill: str = "",
    ) -> dict | None:
        """Build module-routed, Train-derived, answer-free diagnostics.

        A rejected update is not itself a Regularization anomaly.  Complexity guidance
        is emitted only for growth beyond the configured free-growth budget,
        only when the scheduled regularization cost independently causes a
        rejection, and CCV guidance only for a verifier-confirmed regression.
        Deferred and local-edit modes consume this exact same payload.
        """
        if not self.feedback_enabled:
            return None
        decision = telemetry.get("decision", {})
        if decision.get("accepted"):
            return None
        complexity = telemetry.get("complexity", {})
        growth = complexity.get("growth", {})
        diagnostics = complexity.get("update_diagnostics", {}) or {}
        ccv = telemetry.get("ccv", {})
        # Complexity is actionable only when its regularization term is an
        # independent cause of rejection.  Mere positive growth on a candidate
        # already rejected by validation or CCV must not compress anything.
        complexity_triggered = bool(
            decision.get("complexity_triggered")
            or "complexity_regularized_rejection" in (
                decision.get("reasons", []) or []
            )
        )
        ccv_regressions = int(ccv.get("regressions", 0) or 0)
        ccv_triggered = ccv_regressions > 0
        abnormal_modules = [
            name for name, triggered in (
                ("complexity", complexity_triggered),
                ("ccv", ccv_triggered),
            ) if triggered
        ]
        # Validation regression is selection evidence, not sufficiently
        # specific repair evidence.  Do not invent a compression or CCV
        # correction when neither Regularization module fired.
        if not abnormal_modules:
            return None
        counterexamples = []
        contributors = diagnostics.get("top_complexity_contributors", []) or []
        pattern_counts: dict[str, int] = {}
        for contributor in contributors:
            diagnosis = str(contributor.get("diagnosis", "") or "").strip()
            if diagnosis:
                pattern_counts[diagnosis] = pattern_counts.get(diagnosis, 0) + 1
        for record in (ccv.get("attack_records", []) or []) if ccv_triggered else []:
            if record.get("status") != "update_regression":
                continue
            counterexamples.append({
                "attack_id": record.get("attack_id"),
                "source_id": record.get("source_id"),
                "hypothesis": (record.get("pattern") or {}).get(
                    "incidental_assumption", ""
                ),
                "intervention": (record.get("generation") or {}).get(
                    "intervention", ""
                ),
                "observed_failure": _answer_free_failure_summary(
                    record.get("candidate_result")
                ),
            })
        advice = []
        reasons = list(decision.get("reasons", []) or [])
        if complexity_triggered:
            advice.append(
                "Refactor the candidate so its validation gain is not erased "
                "by scheduled complexity cost, while preserving supported capabilities."
            )
        if ccv_triggered:
            advice.append(
                "Abstract the confirmed CCV regression into one transferable "
                "procedural invariant and revise the responsible rule."
            )
        active_edit_budget = max(
            1,
            int(edit_budget) if edit_budget is not None
            else max(1, len(contributors) // 2),
        )
        complexity_policy = {
            "objective": (
                "reduce the source update's net contribution to whole-skill "
                "complexity without removing train-supported capabilities"
            ),
            "max_selected_edits": active_edit_budget,
            "prefer_replacement_over_append": True,
            "reduce_total_complexity": True,
            "ordered_operations": [
                "merge semantically overlapping or duplicate rules",
                "remove narrow, one-off, task-specific, or unsupported rules",
                "replace clusters of similar rules with one transferable general rule",
                "compress repeated explanations, procedures, and examples",
            ],
            "avoid": ["large_append", "duplicate_rule", "unsupported_example"],
        }
        if complexity_triggered and contributors:
            advice.append(
                "Use the full current skill as read-only context, but compress "
                "only the material introduced or changed by the immediately "
                "preceding source update."
            )
        feedback = {
            "source": "SkillEvoReg",
            "abnormal_modules": abnormal_modules,
            "author": {
                "mode": self.feedback_author,
                "model": (
                    self.feedback_teacher_model
                    if self.feedback_author == "teacher" else None
                ),
            },
            "decision": decision.get("action"),
            "reasons": reasons,
            "provenance": {
                "validation_signal": "accept_reject_only",
                "ccv_source_split": self.ccv_source_split,
                "test_access": False,
            },
            "revision_advice": advice,
        }
        if complexity_triggered:
            feedback["complexity"] = {
                "triggered": True,
                "relative_growth": growth.get("relative_growth"),
                "free_growth_budget": growth.get("free_growth_budget"),
                "penalized_growth": growth.get("normalized_growth"),
                "lambda": complexity.get("lambda_complex"),
                "penalty": (
                    float(complexity.get("lambda_complex", 0.0))
                    * float(growth.get("normalized_growth", 0.0))
                ),
                "whole_skill_net_change": diagnostics.get("whole_skill_net_change"),
                "dominant_pattern_counts": pattern_counts,
                "next_update_policy": complexity_policy,
            }
        if ccv_triggered:
            feedback["ccv"] = {
                "triggered": True,
                "status": ccv.get("status"),
                "confirmed_regressions": ccv_regressions,
                "counterexamples": counterexamples,
                "repair_policy": {
                    "objective": (
                        "restore the regressed transferable capability by "
                        "revising the implicated incidental assumption"
                    ),
                    "derive_exactly_one_transferable_invariant": True,
                    "prefer": [
                        "revise the responsible existing rule",
                        "generalize or merge the responsible rule",
                        "remove the incidental assumption",
                    ],
                    "avoid": [
                        "one_off_exception",
                        "new_checklist_branch",
                        "probe_specific_literals",
                        "field_or_file_names_from_counterexample",
                    ],
                },
            }
        semantic_guidance = self._build_semantic_guidance(
            feedback,
            current_skill=current_skill,
            candidate_skill=candidate_skill,
        )
        if semantic_guidance is not None:
            feedback["semantic_repair_guidance"] = semantic_guidance
        return feedback

    def _build_semantic_guidance(
        self,
        feedback: dict,
        *,
        current_skill: str,
        candidate_skill: str,
    ) -> dict | None:
        """Optionally add an answer-free semantic critique to rule feedback."""
        if self.feedback_author == "heuristic":
            return None
        system = (
            "You are an answer-free critic for iterative skill optimisation. "
            "Turn the supplied aggregate guard diagnostics into concise, "
            "actionable repair guidance. Do not invent validation examples, "
            "answers, task outcomes, or evidence that is not in the input. "
            "Return exactly one JSON object with string-list fields `keep`, "
            "`revise`, `avoid`, plus string fields `next_edit_strategy` and "
            "`evidence_scope`. Each list must contain at most three items."
        )
        skill_delta = "\n".join(unified_diff(
            current_skill.splitlines(),
            candidate_skill.splitlines(),
            fromfile="current_skill",
            tofile="rejected_candidate",
            lineterm="",
            n=3,
        ))
        # Regularization against a pathological full rewrite consuming the critic's
        # context budget. The complete skills remain in normal checkpoints;
        # the critic needs only enough local context to diagnose this update.
        if len(skill_delta) > 24000:
            skill_delta = skill_delta[:12000] + "\n...[truncated]...\n" + skill_delta[-12000:]
        user = (
            "Guard diagnostics (aggregate only):\n```json\n"
            + json.dumps(feedback, ensure_ascii=False)
            + "\n```\n\nRejected skill delta (not validation data):\n```diff\n"
            + (skill_delta or "(no textual diff available)")
            + "\n```"
        )
        try:
            kwargs = {
                "system": system,
                "user": user,
                "max_completion_tokens": self.feedback_max_completion_tokens,
                "retries": self.feedback_retries,
                "stage": "guard_feedback",
                "timeout": self.feedback_timeout,
            }
            if self.feedback_author == "target":
                response, usage = chat_target(**kwargs)
            else:
                response, usage = chat_with_deployment(
                    deployment=self.feedback_teacher_model, **kwargs
                )
            parsed = extract_json(response)
            if not isinstance(parsed, dict):
                return {
                    "status": "unavailable_invalid_json",
                    "author": self.feedback_author,
                    "usage": usage,
                }
            allowed = {
                key: parsed[key]
                for key in ("keep", "revise", "avoid", "next_edit_strategy", "evidence_scope")
                if key in parsed
            }
            if not allowed:
                return {
                    "status": "unavailable_empty_guidance",
                    "author": self.feedback_author,
                    "usage": usage,
                }
            allowed.update({
                "status": "ok",
                "author": self.feedback_author,
                "usage": usage,
            })
            return allowed
        except Exception as exc:  # Feedback must never turn a reject into a crash.
            return {
                "status": "unavailable_error",
                "author": self.feedback_author,
                "error": f"{type(exc).__name__}: {exc}",
            }
