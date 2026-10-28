"""CCV result schema and adapter hook."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from skillopt.regularization.verifier import resolve_verifier_contract


@dataclass(frozen=True)
class CCVResult:
    status: str = "disabled"
    generated_attacks: int = 0
    valid_attacks: int = 0
    materialized_attacks: int = 0
    validator_passed_attacks: int = 0
    old_qualified_attacks: int = 0
    paired_evaluated_attacks: int = 0
    confirmed_regressions: int = 0
    old_successes: int = 0
    candidate_successes: int = 0
    regressions: int = 0
    improvements: int = 0
    patterns: int = 0
    invalid_attacks: int = 0
    transport_failures: int = 0
    inconclusive_failures: int = 0
    qualified_sources: int = 0
    source_qualification_failures: int = 0
    early_stopped: bool = False
    attack_records: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str = ""

    @property
    def oar(self) -> float:
        return self.regressions / self.valid_attacks if self.valid_attacks else 0.0

    @property
    def old_score(self) -> float:
        return self.old_successes / self.valid_attacks if self.valid_attacks else 0.0

    @property
    def candidate_score(self) -> float:
        return self.candidate_successes / self.valid_attacks if self.valid_attacks else 0.0

    @property
    def positive_regression(self) -> float:
        return max(0.0, self.old_score - self.candidate_score)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "oar": self.oar,
            "old_score": self.old_score,
            "candidate_score": self.candidate_score,
            "positive_regression": self.positive_regression,
        }


def coerce_ccv_result(value) -> CCVResult:
    if isinstance(value, CCVResult):
        return value
    if not isinstance(value, dict):
        raise TypeError("CCV provider must return CCVResult or dict")
    allowed = {
        "status", "generated_attacks", "valid_attacks", "old_successes",
        "materialized_attacks", "validator_passed_attacks",
        "old_qualified_attacks", "paired_evaluated_attacks",
        "confirmed_regressions",
        "candidate_successes", "regressions", "improvements",
        "patterns", "invalid_attacks", "transport_failures",
        "inconclusive_failures", "qualified_sources",
        "source_qualification_failures",
        "early_stopped", "attack_records",
        "usage", "error",
    }
    return CCVResult(**{key: value[key] for key in allowed if key in value})


def run_adapter_ccv(
    adapter,
    *,
    old_skill: str,
    candidate_skill: str,
    step: int,
    seed: int,
    out_dir: str,
    source_split: str,
    source_size: int,
    min_qualified_sources: int = 1,
    qualification_passes: int = 1,
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
    provider: str = "legacy",
    agent_max_turns: int = 96,
    agent_timeout: int = 3600,
    agent_max_thinking_tokens: int = 8192,
    agent_base_url: str = "https://yibuapi.com",
    agent_api_key_env: str = "ANTHROPIC_API_KEY",
) -> CCVResult:
    adapter_provider = getattr(adapter, "run_ccv", None)
    if not callable(adapter_provider):
        return CCVResult(status="provider_unavailable")
    try:
        verifier_contract = resolve_verifier_contract(adapter)
        raw = adapter_provider(
            old_skill=old_skill,
            candidate_skill=candidate_skill,
            step=step,
            seed=seed,
            out_dir=out_dir,
            source_split=source_split,
            source_size=source_size,
            min_qualified_sources=min_qualified_sources,
            qualification_passes=qualification_passes,
            max_patterns=max_patterns,
            attacks_per_pattern=attacks_per_pattern,
            max_attacks=max_attacks,
            history_items=history_items,
            attacker_model=attacker_model,
            reasoning_effort=reasoning_effort,
            builder_reasoning_effort=builder_reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
            retries=retries,
            trajectory_context=trajectory_context or {},
            provider=provider,
            agent_max_turns=agent_max_turns,
            agent_timeout=agent_timeout,
            agent_max_thinking_tokens=agent_max_thinking_tokens,
            agent_base_url=agent_base_url,
            agent_api_key_env=agent_api_key_env,
            verifier_contract=verifier_contract,
        )
        result = coerce_ccv_result(raw)
        if result.valid_attacks < 0 or result.regressions > result.valid_attacks:
            raise ValueError("invalid CCV counts")
        return result
    except Exception as exc:  # noqa: BLE001
        return CCVResult(status="error", error=f"{type(exc).__name__}: {exc}")
