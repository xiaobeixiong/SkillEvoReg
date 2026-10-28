"""CLI: launch one ``LifelongRunner.run()`` from yaml configs.

Usage::

    python -m scripts.run \\
        --baseline configs/baselines/selfgen_experience_always.yaml \\
        --strategy configs/strategies/chain.yaml \\
        --order-seed A

    # With a model preset (e.g. swap Claude Opus 4-5 for Sonnet 4-6):
    python -m scripts.run \\
        --baseline-name selfgen_experience_always \\
        --model-yaml configs/models/claude-sonnet-4.6.yaml \\
        --order-seed A

    python -m scripts.run --baseline ... --dry-run
        # Run preflight + scheduler only; do NOT touch Harbor.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from skillevolbench.baselines import load_baseline
from skillevolbench.orchestration import LifelongRunner
from skillevolbench.schemas import (
    BaselineConfig,
    RunConfig,
    StrategyConfig,
)


def _apply_model_preset(
    baseline: BaselineConfig, preset_path: Path,
) -> tuple[BaselineConfig, str]:
    """Load a model preset yaml, override baseline.harbor_agent_name +
    baseline.model_name, and export agent_env vars to ``os.environ`` so
    the preinstalled adapter passes them through to the container.

    Returns (mutated_baseline, short_id). short_id is used in run_id so
    that runs from different model presets don't collide.
    """
    with open(preset_path) as fh:
        preset = yaml.safe_load(fh) or {}

    required = {"name", "harbor_agent_name", "agent_model_name"}
    missing = required - preset.keys()
    if missing:
        raise SystemExit(f"model preset {preset_path} missing fields: {missing}")

    # Provider-specific presets may declare environment variables that must be
    # present before any run directory or Harbor trial is created. Fail closed
    # instead of resolving a missing ${VAR} reference to an empty credential.
    required_env = [str(name) for name in (preset.get("required_env") or [])]
    missing_env = [name for name in required_env if not os.environ.get(name)]
    if missing_env:
        raise SystemExit(
            f"model preset {preset_path} requires non-empty environment "
            f"variables: {', '.join(missing_env)}"
        )

    # Export agent_env to os.environ. Resolve ${VAR} references against
    # the current environment (so .harbor-agents.env values flow through).
    resolved_agent_env: dict[str, str] = {}
    for k, v in (preset.get("agent_env") or {}).items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            v = os.environ.get(v[2:-1], "")
        resolved_agent_env[k] = str(v)
        os.environ[k] = str(v)

    # Mutate baseline (re-validate to re-run protocol invariants).
    baseline_dict = baseline.model_dump()
    baseline_dict["harbor_agent_name"] = preset["harbor_agent_name"]
    baseline_dict["model_name"] = preset["agent_model_name"]

    # Persist non-secret Codex routing knobs into AgentConfig.kwargs so Harbor
    # does not depend on process-env timing when it instantiates the adapter.
    if preset["harbor_agent_name"] == "codex":
        agent_kwargs = dict(baseline_dict.get("agent_kwargs") or {})
        codex_kwargs = {
            "base_url": resolved_agent_env.get("OPENAI_BASE_URL"),
            "provider": resolved_agent_env.get("CODEX_MODEL_PROVIDER"),
            "env_key": resolved_agent_env.get("CODEX_PROVIDER_ENV_KEY"),
            "wire_api": resolved_agent_env.get("CODEX_WIRE_API"),
            "api_version": (
                resolved_agent_env.get("AZURE_OPENAI_API_VERSION")
                or os.environ.get("AZURE_OPENAI_API_VERSION")
            ),
            "verbosity": resolved_agent_env.get("CODEX_MODEL_VERBOSITY"),
            "reasoning_effort": preset.get("agent_reasoning_effort"),
        }
        agent_kwargs.update(
            {key: value for key, value in codex_kwargs.items() if value}
        )
        baseline_dict["agent_kwargs"] = agent_kwargs

    baseline = BaselineConfig.model_validate(baseline_dict)

    # Host-side LiteLLM routing for SkillAuthor / Judge / LLMSelfRetriever.
    # We can't smuggle a full config dict through env vars cleanly, so we
    # publish 3 well-known sentinel envvars that BaselineRuntime reads.
    host = preset.get("host_litellm") or {}
    if host:
        if "model" in host:
            os.environ["SEVB_HOST_LITELLM_MODEL"] = host["model"]
        if "reasoning_effort" in host:
            os.environ["SEVB_HOST_LITELLM_REASONING_EFFORT"] = str(
                host["reasoning_effort"]
            )
        if "api_base_env" in host:
            api_base = os.environ.get(host["api_base_env"], "")
            if api_base:
                os.environ["SEVB_HOST_LITELLM_API_BASE"] = api_base
        if "api_key_env" in host:
            api_key = os.environ.get(host["api_key_env"], "")
            if api_key:
                os.environ["SEVB_HOST_LITELLM_API_KEY"] = api_key

    short_id = preset.get("short_id") or preset["name"]
    return baseline, short_id


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    if args.baseline_name is not None:
        baseline = load_baseline(args.baseline_name)
    elif args.baseline is not None:
        baseline = BaselineConfig.from_yaml(args.baseline)
    else:
        raise SystemExit("--baseline or --baseline-name is required")

    # Optional: apply a model preset on top of the baseline.
    model_short_id: str | None = None
    if args.model_yaml is not None:
        baseline, model_short_id = _apply_model_preset(baseline, args.model_yaml)

    strategy_path = args.strategy
    if strategy_path is None:
        # Use the baseline's default_strategy (or "chain" placeholder).
        sname = baseline.default_strategy
        if sname == "none":
            sname = "chain"
        strategy_path = REPO_ROOT / "configs" / "strategies" / f"{sname}.yaml"
    strategy = StrategyConfig.from_yaml(strategy_path)

    # A model preset is authoritative for every host-side LLM call. Persist
    # that resolved model back into the composed strategy so config.json is a
    # truthful execution record rather than a mix of effective and fallback
    # settings.
    host_model = os.environ.get("SEVB_HOST_LITELLM_MODEL")
    host_reasoning_effort = os.environ.get(
        "SEVB_HOST_LITELLM_REASONING_EFFORT"
    )
    if host_model:
        strategy = strategy.model_copy(update={"author_model": host_model})

    uses_skill_author = (
        baseline.allow_self_gen_induction
        or baseline.allow_zero_shot_creation
        or baseline.allow_revision
    )
    effective_model_routing = {
        "task_agent": baseline.model_name,
        "skill_author": (
            strategy.author_model if uses_skill_author else "disabled"
        ),
        "retriever": (
            (host_model or baseline.model_name)
            if baseline.retriever_type == "llm_self"
            else f"non_llm:{baseline.retriever_type}"
        ),
        "ccv_probe_attacker": (
            strategy.author_model if strategy.regularization_ccv_enabled else "disabled"
        ),
        "ccv_old_candidate_executor": (
            baseline.model_name if strategy.regularization_ccv_enabled else "disabled"
        ),
        "guard_feedback": (
            "deterministic_rules"
            if strategy.regularization_proposal_regularization else "disabled"
        ),
        "regularization_local_edit": (
            strategy.author_model
            if strategy.regularization_consumption_mode == "local_edit" else "disabled"
        ),
    }
    agent_reasoning_effort = str(
        baseline.agent_kwargs.get("reasoning_effort") or ""
    )
    effective_reasoning_effort = {}
    if agent_reasoning_effort or host_reasoning_effort:
        effective_reasoning_effort = {
            "task_agent": agent_reasoning_effort,
            "skill_author": (
                host_reasoning_effort if uses_skill_author else "disabled"
            ),
            "retriever": (
                host_reasoning_effort
                if baseline.retriever_type == "llm_self" else "not_applicable"
            ),
            "ccv_probe_attacker": (
                host_reasoning_effort
                if strategy.regularization_ccv_enabled else "disabled"
            ),
            "ccv_old_candidate_executor": (
                agent_reasoning_effort
                if strategy.regularization_ccv_enabled else "disabled"
            ),
            "regularization_local_edit": (
                host_reasoning_effort
                if strategy.regularization_consumption_mode == "local_edit"
                else "disabled"
            ),
        }

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = RunConfig.make_run_id(
            baseline.name, strategy.name, args.order_seed,
        )
        if model_short_id:
            # Insert model token before the timestamp:
            #   selfgen__chain__seedA__20260501_2030
            #   -> selfgen__chain__sonnet-4-6__seedA__20260501_2030
            parts = run_id.split("__")
            # parts = [baseline, strategy, "seedX", timestamp]
            run_id = "__".join(parts[:2] + [model_short_id] + parts[2:])
    return RunConfig(
        run_id=run_id,
        baseline=baseline,
        strategy=strategy,
        effective_model_routing=effective_model_routing,
        effective_reasoning_effort=effective_reasoning_effort,
        order_seed=args.order_seed,
        workspace_root=Path(args.workspace_root),
        api_base=args.api_base,
        api_key_env_var=args.api_key_env_var,
        dry_run=args.dry_run,
        max_tasks=args.max_tasks,
        anchor_source_only=bool(getattr(args, "anchor_source_only", False)),
        learning_passes=int(getattr(args, "learning_passes", 1)),
        shared_learning_anchor_source=getattr(
            args, "shared_learning_anchor_source", None
        ),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path,
                   help="path to a baseline yaml (mutually exclusive with --baseline-name)")
    p.add_argument("--baseline-name", help="canonical baseline name (e.g. 'no_skill')")
    p.add_argument("--strategy", type=Path,
                   help="path to a strategy yaml (defaults to baseline's default_strategy)")
    p.add_argument("--model-yaml", type=Path, default=None,
                   help="path to a model preset yaml (configs/models/*.yaml). "
                        "Overrides baseline.harbor_agent_name + baseline.model_name "
                        "and exports agent_env vars.")
    p.add_argument("--order-seed", default="A", choices=["A", "B", "C"])
    p.add_argument("--run-id", help="override the auto-generated run_id")
    p.add_argument("--workspace-root", default="workspace/runs",
                   help="parent directory for run artifacts (default: workspace/runs)")
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-key-env-var", default="ANTHROPIC_API_KEY")
    p.add_argument("--dry-run", action="store_true",
                   help="run preflight + scheduler only; no Harbor")
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help=(
            "truncate the canonical schedule to its first N tasks; "
            "intended only for smoke tests, never reported benchmark runs"
        ),
    )
    p.add_argument(
        "--anchor-source-only",
        action="store_true",
        help=(
            "execute the 90 original T1-T3 learning tasks for protocol-faithful "
            "shared-anchor generation; skips T4-T6 and all replay trials"
        ),
    )
    p.add_argument(
        "--learning-passes", type=int, choices=[1, 2], default=1,
        help="number of complete T1-T3 passes before T4-T6 (default: 1)",
    )
    p.add_argument(
        "--shared-learning-anchor-source", type=Path, default=None,
        help=(
            "completed source run whose T1-T3 libraries and trajectories seed "
            "a shared pass 2; requires --learning-passes 2"
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _setup_logging(args.verbose)

    config = _build_run_config(args)
    logging.getLogger(__name__).info(
        "Effective model routing: %s", config.effective_model_routing,
    )
    logging.getLogger(__name__).info(
        "Effective reasoning effort: %s",
        config.effective_reasoning_effort,
    )
    runner = LifelongRunner(config=config)

    if args.dry_run:
        runtime, ordered_tasks, _ = runner.prepare()
        print(f"Dry-run OK: run_id={config.run_id}")
        print(f"  baseline={config.baseline.name} strategy={config.strategy.name} "
              f"seed={config.order_seed}")
        print(f"  effective_models={config.effective_model_routing}")
        print(f"  effective_reasoning={config.effective_reasoning_effort}")
        print(f"  run_root={runtime.run_root}")
        print(f"  scheduled {len(ordered_tasks)} tasks")
        print(f"  first 3: {[r.spec.task_id for r in ordered_tasks[:3]]}")
        print(f"  last 3:  {[r.spec.task_id for r in ordered_tasks[-3:]]}")
        return 0

    report = asyncio.run(runner.run())
    print(f"Run complete: {config.run_id}")
    print(f"  overall_sr           = {report.task_success.get('overall_sr', 0):.3f}")
    print(f"  evaluation_sr        = {report.task_success.get('evaluation_sr', 0):.3f}")
    print(f"  t6_composition_rate  = {report.task_success.get('t6_composition_rate', 0):.3f}")
    print(f"  active_skills        = {report.library_health.get('active_skill_count', 0)}")
    print(f"  retired_skills       = {report.library_health.get('retired_skill_count', 0)}")
    print(f"  patches_applied      = {report.library_health.get('n_patches_applied', 0)}")
    print(f"  report               = {(Path(config.workspace_root) / config.run_id) / 'reports' / 'full_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
