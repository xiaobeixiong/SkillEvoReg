"""Strategy configuration schema.

Models yaml files in ``configs/strategies/`` as strict Pydantic v2
``StrategyConfig`` objects.

A strategy decides how skill edits are proposed during T1-T3 learning roles.
Public strategies:

* ``chain``       -- one candidate, free-form revision
* ``chain_tier3`` -- one candidate that must include a Tier-3 support artifact
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


StrategyName = Literal[
    "chain",
    "chain_tier3",
    "chain_skillevoreg_online",
    "chain_skillevoreg_local_edit",
]
CandidateMode = Literal[
    "minimal_edit",
    "scope_narrow",
    "refactor",
    "retire_replace",
    "free_form",
    "tier3_required",
]
RegularizationConsumptionMode = Literal["deferred", "local_edit"]
RegularizationCCVSchedule = Literal["per_step", "family_end"]


class StrategyConfig(BaseModel):
    """One revision strategy loaded from ``configs/strategies/<name>.yaml``."""

    model_config = ConfigDict(use_enum_values=True)

    name: StrategyName = Field(..., description="chain | chain_tier3")

    n_candidates: int = Field(default=1, ge=1, le=16)
    candidate_modes: list[CandidateMode] = Field(default_factory=lambda: ["free_form"])

    author_model: str = "anthropic/claude-opus-4-5"
    author_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    author_max_tokens: int = Field(default=16384, ge=1)
    # Opt-in updater ablation; existing configs retain the original author.
    author_update_style: Literal["full_rewrite", "local_patch"] = "full_rewrite"

    # Optional paired-experiment control. The manifest stores an exact post-T1
    # skill and may also store the source T1 replay state. With
    # shared_t1_skip_execution enabled, the scheduler omits original T1 and
    # the runtime installs both immediately before T2; otherwise only the
    # stochastic induction result is replaced after a normally executed T1.
    shared_t1_anchor_manifest: Path | None = None
    shared_t1_anchor_required: bool = False
    shared_t1_skip_execution: bool = False

    # Public strategies do not use a separate judge, but the fields remain so
    # older config files can still parse cleanly.
    judge_model: str = "anthropic/claude-haiku-4-5"
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    judge_max_tokens: int = Field(default=1024, ge=1)

    # Proposal-time SkillEvoReg regularisation preserves the paper baseline's
    # one-candidate, always-apply update semantics. Complexity and dropout shape
    # SkillAuthor's proposal; CCV supplies feedback to a later proposal.
    regularization_proposal_regularization: bool = False
    # In a two-pass late-regularization run, pass 1 behaves as plain Chain and
    # all Regularization modules become active together at pass 2.
    regularization_start_learning_pass: int = Field(default=1, ge=1, le=2)

    regularization_complexity_lambda: float = Field(default=0.15, ge=0.0)
    regularization_free_growth: float = Field(default=0.10, ge=0.0)
    # Complexity anomaly detector, anchored to the family's post-T1 skill.
    # The cap is cumulative over T2/T3 (not compounded once per update); its
    # finding is consumed according to regularization_consumption_mode.
    regularization_hard_complexity_enabled: bool = False
    regularization_hard_max_relative_growth: float = Field(default=0.15, ge=0.0)
    regularization_hard_max_atom_growth: int = Field(default=4, ge=0)

    # Atomic skill dropout is training-time regularisation. The task agent and
    # SkillAuthor consume the same per-trial masked view; Val/eval/CCV remain
    # unmasked and the patch is restored/applied to the complete live library.
    regularization_dropout_enabled: bool = False
    regularization_dropout_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    regularization_dropout_max_atoms: int = Field(default=4, ge=0)
    regularization_dropout_min_eligible_atoms: int = Field(default=4, ge=1)
    regularization_dropout_ensure_at_least_one: bool = True

    # Candidate-conditioned CCV for the validation-free online protocol.
    # SkillEvolBench uses an instruction-appendix adapter: the immutable task
    # instruction remains verbatim and the attacker may add only a bounded,
    # answer-free robustness context.  The original verifier is therefore
    # reused unchanged for paired old/candidate executions.
    regularization_ccv_enabled: bool = False
    regularization_ccv_roles: list[str] = Field(
        default_factory=lambda: ["enriched", "variant"]
    )
    regularization_ccv_max_appendix_chars: int = Field(default=1600, ge=128, le=8000)
    # Invalid/unsafe generations are retried with the same model. Exhaustion is
    # recorded as inconclusive; formal runs never replace model output with an
    # unreported rule-based probe.
    regularization_ccv_generation_retries: int = Field(default=2, ge=0, le=8)
    regularization_ccv_deterministic_fallback: bool = False
    regularization_ccv_min_old_score: float = Field(default=0.80, ge=0.0, le=1.0)
    regularization_ccv_min_regression_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    # family_end runs one differential probe after pass-2 T3, comparing the
    # immutable pre-pass family skill with the final post-pass family skill.
    regularization_ccv_schedule: RegularizationCCVSchedule = "per_step"
    # Top-level consumption policy for actionable Regularization findings. Dropout stays
    # stochastic training regularisation; complexity/CCV anomalies are either fed to
    # the next proposal or consumed by one immediate unevaluated correction.
    regularization_consumption_mode: RegularizationConsumptionMode = "deferred"
    # Candidate formation is fully task-local: Dropout/complexity repair,
    # four-slot clean/attack CCV, then final always-apply commit.
    regularization_step_local_pipeline: bool = False
    # Symmetric source-atom/history-atom retrieval for Local editing.
    regularization_embedding_model: str = "BAAI/bge-base-en-v1.5"
    regularization_embedding_match_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0,
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_patch_fields(cls, data):
        """Accept pre-rename configs that used patch_* for author settings."""
        if not isinstance(data, dict):
            return data
        upgraded = dict(data)
        legacy_map = {
            "patch_model": "author_model",
            "patch_temperature": "author_temperature",
            "patch_max_tokens": "author_max_tokens",
        }
        for old_key, new_key in legacy_map.items():
            if old_key in upgraded and new_key not in upgraded:
                upgraded[new_key] = upgraded[old_key]
        return upgraded

    @field_validator("candidate_modes")
    @classmethod
    def _modes_unique_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("candidate_modes must not be empty")
        if len(set(v)) != len(v):
            raise ValueError(f"candidate_modes contains duplicates: {v}")
        return v

    @model_validator(mode="after")
    def _check_strategy_invariants(self) -> "StrategyConfig":
        if self.n_candidates != 1:
            raise ValueError(
                f"{self.name} strategy requires n_candidates=1, "
                f"got {self.n_candidates}"
            )
        if len(self.candidate_modes) != 1:
            raise ValueError(
                f"{self.name} strategy requires exactly one candidate_mode, "
                f"got {self.candidate_modes}"
            )
        learning_roles = {"canonical", "enriched", "variant"}
        if len(set(self.regularization_ccv_roles)) != len(self.regularization_ccv_roles):
            raise ValueError("regularization_ccv_roles contains duplicates")
        if not set(self.regularization_ccv_roles).issubset(learning_roles):
            raise ValueError(
                "regularization_ccv_roles must be a subset of canonical/enriched/variant"
            )
        if self.shared_t1_anchor_required and self.shared_t1_anchor_manifest is None:
            raise ValueError(
                "shared_t1_anchor_required=True requires "
                "shared_t1_anchor_manifest"
            )
        if self.shared_t1_skip_execution and self.shared_t1_anchor_manifest is None:
            raise ValueError(
                "shared_t1_skip_execution=True requires "
                "shared_t1_anchor_manifest"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> "StrategyConfig":
        path = Path(path)
        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
        anchor_path = data.get("shared_t1_anchor_manifest")
        if anchor_path:
            resolved = Path(anchor_path).expanduser()
            if not resolved.is_absolute():
                resolved = (path.parent / resolved).resolve()
            data["shared_t1_anchor_manifest"] = resolved
        return cls.model_validate(data)


__all__ = [
    "StrategyName",
    "CandidateMode",
    "RegularizationConsumptionMode",
    "RegularizationCCVSchedule",
    "StrategyConfig",
]
