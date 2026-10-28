"""Run-level configuration schema.

Bundles BaselineConfig + StrategyConfig + per-run knobs (run_id, order_seed,
workspace, Harbor orchestrator type, LLM endpoint) into a single ``RunConfig``
that is the unique input to ``LifelongRunner.run()``.

Two auxiliary loaders also live here:

* ``EnvOrders.from_yaml(...)`` -> reads ``configs/env_orders.yaml``
                                    and validates the three seeds A/B/C.
* ``LLMDefaults.from_yaml(...)`` -> reads ``configs/llm.yaml`` and supplies
                                    fallback model names + endpoint.

Hard invariants enforced here:

* ``harbor_n_concurrent_trials == 1`` (lifelong protocol -- §4.3 of design).
* ``order_seed`` must be one of A / B / C.
* ``baseline.default_strategy`` must agree with ``strategy.name`` (or be
  ``"none"`` when the baseline doesn't use a strategy).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skillevolbench.schemas.baseline import BaselineConfig
from skillevolbench.schemas.strategy import StrategyConfig


HarborOrchestratorType = Literal["local", "daytona", "modal", "e2b"]
OrderSeed = Literal["A", "B", "C"]


# ---------------------------------------------------------------------------
# Auxiliary configs
# ---------------------------------------------------------------------------


class EnvOrders(BaseModel):
    """Three environment orderings used by the Order-Sensitivity ablation."""

    seed_a: list[str] = Field(..., min_length=6, max_length=6)
    seed_b: list[str] = Field(..., min_length=6, max_length=6)
    seed_c: list[str] = Field(..., min_length=6, max_length=6)

    @field_validator("seed_a", "seed_b", "seed_c")
    @classmethod
    def _check_envs(cls, v: list[str]) -> list[str]:
        expected = {f"E{i}" for i in range(1, 7)}
        if set(v) != expected:
            raise ValueError(
                f"each seed must be a permutation of E1..E6; got {v}"
            )
        if len(v) != len(set(v)):
            raise ValueError(f"seed has duplicates: {v}")
        return v

    @classmethod
    def from_yaml(cls, path: Path | str) -> "EnvOrders":
        path = Path(path)
        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    def for_seed(self, seed: OrderSeed) -> list[str]:
        return {"A": self.seed_a, "B": self.seed_b, "C": self.seed_c}[seed]


class LLMDefaults(BaseModel):
    """Default LLM endpoint config from ``configs/llm.yaml``."""

    api_base: Optional[str] = None
    api_key_env_var: str = "ANTHROPIC_API_KEY"

    default_agent_model: str = "anthropic/claude-opus-4-5"
    default_author_model: str = "anthropic/claude-opus-4-5"
    default_judge_model: str = "anthropic/claude-haiku-4-5"
    default_embed_model: str = "Qwen/Qwen3-Embedding-4B"

    author_temperature: float = 0.2
    author_max_tokens: int = 16384
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_patch_fields(cls, data):
        """Accept pre-rename configs that used patch_* for author settings."""
        if not isinstance(data, dict):
            return data
        upgraded = dict(data)
        legacy_map = {
            "default_patch_model": "default_author_model",
            "patch_temperature": "author_temperature",
            "patch_max_tokens": "author_max_tokens",
        }
        for old_key, new_key in legacy_map.items():
            if old_key in upgraded and new_key not in upgraded:
                upgraded[new_key] = upgraded[old_key]
        return upgraded

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LLMDefaults":
        path = Path(path)
        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    """Top-level config of one ``LifelongRunner.run()`` invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ===== Identification =====
    run_id: str = Field(..., description="Unique within workspace_root")

    # ===== Asset paths (allow override for fixtures / mini benchmarks) =====
    benchmark_skills_root: Path = Path("benchmark/skills")
    benchmark_tasks_root: Path = Path("benchmark/tasks")

    # ===== Configs =====
    baseline: BaselineConfig
    strategy: StrategyConfig
    # Audit record of the effective (post-preset) model routing. Values are
    # deliberately model identifiers or non-LLM markers, never credentials.
    # Keeping this in config.json prevents a fallback model in a source YAML
    # from being mistaken for the model that actually served the run.
    effective_model_routing: dict[str, str] = Field(default_factory=dict)
    # Same audit principle as model routing: reasoning effort must be explicit
    # for every LLM-backed role instead of relying on adapter/provider defaults.
    effective_reasoning_effort: dict[str, str] = Field(default_factory=dict)

    # ===== Run params =====
    order_seed: OrderSeed = "A"
    workspace_root: Path = Path("workspace/runs")

    # ===== Execution =====
    harbor_orchestrator_type: HarborOrchestratorType = "local"
    # Lifelong protocol hard requirement: tasks must run sequentially so the
    # global skill library is consistent across trials. Any value other than 1
    # breaks the score-before-maintain rule.
    harbor_n_concurrent_trials: int = Field(default=1, ge=1, le=1)
    # Formal runs use one identical agent wall-clock budget for every task.
    # Without an explicit override Harbor inherits each task.toml value, which
    # mixes 600- and 900-second budgets and makes paired results incomparable.
    harbor_agent_timeout_sec: float = Field(default=600.0, ge=60.0, le=600.0)
    # Retry only explicitly classified transient infrastructure failures.
    # Agent/verifier timeouts remain terminal benchmark outcomes.
    # Five retries (six total attempts) are deliberate for long formal runs:
    # the OpenAI-compatible proxy occasionally emits clustered transient
    # encrypted-content/network failures. Only the narrow allow-list in
    # harbor_ext.retry is eligible; semantic failures and the 600-second agent
    # timeout remain terminal outcomes.
    harbor_infrastructure_max_retries: int = Field(default=5, ge=0, le=5)

    # ===== LLM endpoints =====
    api_base: Optional[str] = None
    api_key_env_var: str = "ANTHROPIC_API_KEY"

    # ===== Execution mode =====
    dry_run: bool = False
    # Optional fixture knob: when set, truncate the scheduled task list to the
    # first N tasks and skip the strict full-benchmark order invariant. Public
    # launchers leave this as None.
    max_tasks: Optional[int] = Field(default=None, ge=1)
    # Dedicated shared-anchor source mode. Execute the 90 original learning
    # tasks (T1-T3) and no evaluation/replay trials. Keeping T1-T3 preserves
    # the canonical cross-family library context seen by later-family T1s;
    # the exporter then extracts each family's exact post-T1 commit.
    anchor_source_only: bool = False
    # Repeat the complete T1-T3 learning block inside each environment before
    # freezing for T4-T6. Two passes model early acquisition followed by late
    # consolidation without inventing a validation gate.
    learning_passes: int = Field(default=1, ge=1, le=2)
    # Optional immutable source run whose complete T1-T3 state seeds a shared
    # second learning pass. Both baseline and Regularization can therefore start from
    # the exact same early-learning trajectories and libraries.
    shared_learning_anchor_source: Path | None = None

    # -----------------------------------------------------------------
    # Field validators
    # -----------------------------------------------------------------

    @field_validator("run_id")
    @classmethod
    def _check_run_id(cls, v: str) -> str:
        if not v or "/" in v or "\\" in v or v.startswith("."):
            raise ValueError(f"run_id must not contain path separators: {v!r}")
        if any(c.isspace() for c in v):
            raise ValueError(f"run_id must not contain whitespace: {v!r}")
        return v

    @field_validator("workspace_root")
    @classmethod
    def _resolve_workspace_root(cls, v: Path) -> Path:
        # Force absolute. Downstream `build_job_config` writes
        # `library_active_path` into Harbor's docker-compose mount source,
        # and Docker/Podman's bind-mount semantics require absolute paths
        # -- a relative source silently produces nested host paths
        # (e.g. `runtime/<task_id>/environment/workspace/runs/.../library/active`).
        return v.expanduser().resolve()

    @field_validator("shared_learning_anchor_source")
    @classmethod
    def _resolve_shared_learning_anchor_source(
        cls, v: Path | None,
    ) -> Path | None:
        return v.expanduser().resolve() if v is not None else None

    # -----------------------------------------------------------------
    # Cross-field invariants
    # -----------------------------------------------------------------

    @model_validator(mode="after")
    def _check_run_invariants(self) -> "RunConfig":
        # ---- 1. n_concurrent_trials must be 1 ----
        # Engineering Design §4.3: hard requirement of the lifelong protocol.
        if self.harbor_n_concurrent_trials != 1:
            raise ValueError(
                f"harbor_n_concurrent_trials must be 1 for the lifelong "
                f"protocol; got {self.harbor_n_concurrent_trials}"
            )

        if self.anchor_source_only and self.max_tasks is not None:
            raise ValueError(
                "anchor_source_only cannot be combined with max_tasks"
            )
        if self.strategy.regularization_start_learning_pass > self.learning_passes:
            raise ValueError(
                "regularization_start_learning_pass cannot exceed learning_passes"
            )
        if self.anchor_source_only and self.learning_passes != 1:
            raise ValueError(
                "anchor_source_only requires learning_passes=1"
            )
        if self.shared_learning_anchor_source is not None:
            if self.learning_passes != 2:
                raise ValueError(
                    "shared_learning_anchor_source requires learning_passes=2"
                )
            if self.strategy.shared_t1_skip_execution:
                raise ValueError(
                    "shared_learning_anchor_source cannot combine with "
                    "shared_t1_skip_execution"
                )
        if self.anchor_source_only and self.strategy.shared_t1_skip_execution:
            raise ValueError(
                "anchor_source_only must execute T1 and cannot consume a shared "
                "T1 anchor"
            )

        # ---- 2. baseline.default_strategy <-> strategy.name ----
        # The baseline's "default_strategy" lives in its yaml as a hint about
        # which strategy yaml is expected to be paired with it. RunConfig's
        # `strategy` overrides that, but if the baseline expects something
        # specific, we shouldn't silently mix.
        bl_default = self.baseline.default_strategy
        if bl_default != "none" and self.strategy.name != bl_default:
            raise ValueError(
                f"baseline {self.baseline.name!r} expects strategy "
                f"{bl_default!r} but RunConfig pairs it with strategy "
                f"{self.strategy.name!r}; explicit override is allowed but "
                "must be configured by changing baseline.default_strategy "
                "first (otherwise result attribution becomes ambiguous)."
            )
        if bl_default == "none" and self.strategy.name != "chain":
            # When the baseline declares it doesn't use a strategy (No-Skill,
            # Curated-Static, etc.) we still need to load *something* to have
            # a complete RunConfig. By convention the harness will pair these
            # baselines with a "no-op" strategy at the runtime layer (Part 8).
            # Here we only allow chain (whose name == 'chain') as the trivial
            # placeholder to keep the schema simple.
            raise ValueError(
                f"baseline {self.baseline.name!r} has default_strategy='none' "
                f"so it must be paired with the 'chain' placeholder strategy; "
                f"got {self.strategy.name!r}"
            )

        # ---- 3. composed model-routing audit must agree with configs ----
        routing = self.effective_model_routing
        if routing:
            required = {
                "task_agent", "skill_author", "retriever",
                "ccv_probe_attacker", "ccv_old_candidate_executor",
                "guard_feedback", "regularization_local_edit",
            }
            missing = required - routing.keys()
            if missing:
                raise ValueError(
                    "effective_model_routing missing keys: "
                    + ", ".join(sorted(missing))
                )
            if routing["task_agent"] != self.baseline.model_name:
                raise ValueError(
                    "effective task_agent model disagrees with baseline.model_name"
                )
            if (
                routing["skill_author"] != "disabled"
                and routing["skill_author"] != self.strategy.author_model
            ):
                raise ValueError(
                    "effective skill_author model disagrees with "
                    "strategy.author_model"
                )
            if self.strategy.regularization_ccv_enabled:
                if routing["ccv_probe_attacker"] != self.strategy.author_model:
                    raise ValueError(
                        "CCV attacker must use the effective SkillAuthor model"
                    )
                if routing["ccv_old_candidate_executor"] != self.baseline.model_name:
                    raise ValueError(
                        "CCV paired execution must use the task-agent model"
                    )

        efforts = self.effective_reasoning_effort
        if efforts:
            required_effort_roles = {
                "task_agent", "skill_author", "retriever",
                "ccv_probe_attacker", "ccv_old_candidate_executor",
                "regularization_local_edit",
            }
            missing_efforts = required_effort_roles - efforts.keys()
            if missing_efforts:
                raise ValueError(
                    "effective_reasoning_effort missing keys: "
                    + ", ".join(sorted(missing_efforts))
                )
            active_efforts = {
                value for value in efforts.values()
                if value not in {"disabled", "not_applicable"}
            }
            supported_efforts = {"medium", "high"}
            if (
                len(active_efforts) != 1
                or not active_efforts.issubset(supported_efforts)
            ):
                raise ValueError(
                    "all active LLM roles must use the same supported "
                    "reasoning_effort ('medium' or 'high'); "
                    f"got {sorted(active_efforts)!r}"
                )

        return self

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        """Path under which all run artifacts live."""
        return self.workspace_root / self.run_id

    @property
    def api_key(self) -> Optional[str]:
        """Resolve the API key from the configured environment variable.

        Returns None if the env var is unset; downstream code should raise
        when it actually tries to call the LLM and the key is missing.
        """
        return os.environ.get(self.api_key_env_var)

    @classmethod
    def from_paths(
        cls,
        *,
        run_id: str,
        baseline_yaml: Path | str,
        strategy_yaml: Path | str,
        order_seed: OrderSeed = "A",
        workspace_root: Path | str = "workspace/runs",
        api_base: Optional[str] = None,
        api_key_env_var: str = "ANTHROPIC_API_KEY",
        dry_run: bool = False,
    ) -> "RunConfig":
        """Compose a RunConfig from yaml file paths."""
        baseline = BaselineConfig.from_yaml(baseline_yaml)
        strategy = StrategyConfig.from_yaml(strategy_yaml)
        return cls(
            run_id=run_id,
            baseline=baseline,
            strategy=strategy,
            order_seed=order_seed,
            workspace_root=Path(workspace_root),
            api_base=api_base,
            api_key_env_var=api_key_env_var,
            dry_run=dry_run,
        )

    @staticmethod
    def make_run_id(
        baseline_name: str, strategy_name: str, order_seed: str = "A",
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Canonical run_id format used by every script in scripts/."""
        ts = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M")
        return f"{baseline_name}__{strategy_name}__seed{order_seed}__{ts}"


__all__ = [
    "HarborOrchestratorType",
    "OrderSeed",
    "EnvOrders",
    "LLMDefaults",
    "RunConfig",
]
