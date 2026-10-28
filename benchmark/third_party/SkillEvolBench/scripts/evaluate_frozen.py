"""Evaluate a frozen library from an explicitly supplied source run.

The six cells are the Cartesian product of:

* first-pass baseline, second-pass unregularized, second-pass Regularization; and
* final-library T1-T3 replay, or frozen T4-T6 evaluation.

Every cell clones its source run's final per-environment libraries, freezes
the private copies before the first task, and never invokes skill induction,
revision, CCV, feedback, or local edit.  Source runs remain read-only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run import _apply_model_preset
from skillevolbench.baselines import BaselineRuntime
from skillevolbench.discovery import (
    TaskRecord,
    TaskRegistry,
    default_skills_root,
    default_tasks_root,
)
from skillevolbench.harbor_ext.job_builder import build_job_config
from skillevolbench.metrics.reporter import ReportGenerator
from skillevolbench.orchestration import LifelongRunner
from skillevolbench.scheduler import compute_task_order
from skillevolbench.schemas import EnvOrders, RunConfig, StrategyConfig
from skillevolbench.stores.library_store import LibraryStore


Method = Literal["baseline", "noreg", "regularization"]
Block = Literal["t1-t3", "t4-t6"]

OUTPUT_ROOT = REPO_ROOT / "workspace" / "frozen_eval_repeat2_3way"
MODEL_PRESET = (
    REPO_ROOT / "configs" / "models" / "gpt-5.4-mini-yibu-medium.yaml"
)
ENV_ORDERS = REPO_ROOT / "configs" / "env_orders.yaml"
CHAIN_STRATEGY = REPO_ROOT / "configs" / "strategies" / "chain.yaml"

def _source_library_path(source_root: Path, env_id: str) -> Path:
    """Read only the explicitly supplied run; never substitute historical shards."""
    return source_root / "library" / env_id

LEARNING_ROLES = frozenset({"canonical", "enriched", "variant"})
EVALUATION_ROLES = frozenset(
    {"context-shift", "adversarial", "composition"}
)


@dataclass(frozen=True)
class EvalTarget:
    method: Method
    block: Block
    source_root: Path
    only_task: str | None = None
    repeat_index: int = 2
    output_root: Path = OUTPUT_ROOT

    @property
    def run_id(self) -> str:
        block_id = self.block.replace("-", "_")
        suffix = f"_repair_{self.only_task}" if self.only_task else ""
        return f"{self.method}_frozen_{block_id}_repeat{self.repeat_index}_medium_seedA{suffix}"

    @property
    def run_root(self) -> Path:
        return self.output_root / self.run_id


def _target(method: Method, block: Block, source_root: Path) -> EvalTarget:
    return EvalTarget(method=method, block=block, source_root=source_root)


def _target_schedule(config: RunConfig, target: EvalTarget):
    schedule, registry = _schedule(config, target.block)
    if target.only_task:
        schedule = [r for r in schedule if r.spec.task_id == target.only_task]
        if len(schedule) != 1:
            raise ValueError(f"task {target.only_task} does not belong to {target.block}")
    return schedule, registry


def _benchmark_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _library_hashes(run_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for index in range(1, 7):
        env_id = f"E{index}"
        env_root = _source_library_path(run_root, env_id)
        if not (env_root / "manifest.yaml").is_file():
            raise FileNotFoundError(f"missing source library: {env_root}")
        if not (env_root / ".git").exists():
            raise FileNotFoundError(f"missing source library git state: {env_root}")
        hashes[env_id] = LibraryStore(env_root).compute_hash()
    return hashes


def _load_eval_config(target: EvalTarget) -> RunConfig:
    config_path = target.source_root / "config.json"
    report_path = target.source_root / "reports" / "full_report.json"
    if not config_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            f"source run is not complete: {target.source_root}"
        )
    source = RunConfig.model_validate_json(config_path.read_text())

    baseline, _ = _apply_model_preset(source.baseline, MODEL_PRESET)
    baseline_data = baseline.model_dump()
    baseline_data.update(
        name="frozen_final_skill_eval",
        description=(
            "Read-only evaluation of a private clone of a completed run's "
            "final skill library."
        ),
        skill_init="empty",
        allow_self_gen_induction=False,
        allow_zero_shot_creation=False,
        allow_curated_inject=False,
        allow_revision=False,
        allow_retirement=False,
        allow_quarantine=False,
        allow_applicability_narrow=False,
        allow_rollback=False,
        allow_post_eval_maintenance=False,
        within_env_replay=False,
        replay_eval=False,
        dual_t6_retrieval=False,
        feedback_level="none",
        feedback_to_skill_text=False,
        feedback_to_memory=False,
        default_strategy="none",
        revision_trigger="never",
    )

    strategy = StrategyConfig.from_yaml(CHAIN_STRATEGY)
    effort = "medium"
    routing = {
        "task_agent": baseline.model_name,
        "skill_author": "disabled",
        "retriever": baseline.model_name,
        "ccv_probe_attacker": "disabled",
        "ccv_old_candidate_executor": "disabled",
        "guard_feedback": "disabled",
        "regularization_local_edit": "disabled",
    }
    reasoning = {
        "task_agent": effort,
        "skill_author": "disabled",
        "retriever": effort,
        "ccv_probe_attacker": "disabled",
        "ccv_old_candidate_executor": "disabled",
        "regularization_local_edit": "disabled",
    }

    return RunConfig.model_validate(
        {
            **source.model_dump(),
            "run_id": target.run_id,
            "workspace_root": target.output_root,
            "baseline": baseline_data,
            "strategy": strategy.model_dump(),
            "effective_model_routing": routing,
            "effective_reasoning_effort": reasoning,
            "dry_run": False,
            "max_tasks": None,
            "anchor_source_only": False,
            "learning_passes": 1,
            "shared_learning_anchor_source": None,
        }
    )


def _schedule(
    config: RunConfig, block: Block
) -> tuple[list[TaskRecord], TaskRegistry]:
    registry = TaskRegistry.from_disk(default_skills_root(), default_tasks_root())
    orders = EnvOrders.from_yaml(ENV_ORDERS)
    full = compute_task_order(
        registry,
        orders,
        config.order_seed,
        within_env_replay=False,
        replay_eval=False,
        learning_passes=1,
    )
    roles = LEARNING_ROLES if block == "t1-t3" else EVALUATION_ROLES
    selected = [record for record in full if record.spec.role.value in roles]
    if block == "t1-t3":
        # Replay mode is the harness-level guarantee that learning tasks never
        # dispatch to the strategy.  The wrapper below also freezes every
        # environment before its first replay.
        selected = [replace(record, is_replay=True) for record in selected]
    if len(selected) != 90:
        raise AssertionError(
            f"{block}: expected 90 selected tasks, got {len(selected)}"
        )
    return selected, registry


def _copy_final_libraries(
    source_root: Path, target_root: Path
) -> dict[str, str]:
    source_hashes = _library_hashes(source_root)
    target_library = target_root / "library"
    target_library.mkdir(parents=True, exist_ok=True)
    for env_id, expected_hash in source_hashes.items():
        source = _source_library_path(source_root, env_id)
        destination = target_library / env_id
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite target library: {destination}"
            )
        shutil.copytree(source, destination, symlinks=True)
        (destination / ".frozen").unlink(missing_ok=True)
        actual_hash = LibraryStore(destination).compute_hash()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"library clone hash mismatch for {env_id}: "
                f"{actual_hash} != {expected_hash}"
            )
    return source_hashes


def _expected_record_path(target: EvalTarget, task_id: str) -> Path:
    suffix = "__replay" if target.block == "t1-t3" else ""
    return (
        target.run_root
        / "stores"
        / "replay"
        / "records"
        / f"{task_id}{suffix}.json"
    )


def _target_complete(target: EvalTarget, schedule: list[TaskRecord]) -> bool:
    report = target.run_root / "reports" / "full_report.json"
    summary = target.run_root / "reports" / "frozen_eval_summary.json"
    return report.is_file() and summary.is_file() and all(
        _expected_record_path(target, record.spec.task_id).is_file()
        for record in schedule
    )


def _write_frozen_summary(target: EvalTarget, runtime: Any) -> Path:
    """Write metrics that include replay-mode T1-T3 records explicitly.

    The benchmark's generic headline reporter intentionally excludes replay
    records from primary task-success rates.  That is correct for main runs,
    but a T1-T3-only supplementary job consists entirely of such records, so
    this protocol-specific summary is the authoritative repeat-2 artifact.
    """
    records = runtime.replay_store.all_records()
    expected_roles = LEARNING_ROLES if target.block == "t1-t3" else EVALUATION_ROLES
    selected = [record for record in records if record.task_role in expected_roles]
    expected = 1 if target.only_task else 90
    if target.only_task:
        selected = [r for r in selected if r.task_id.removesuffix("__replay") == target.only_task]
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} frozen evaluation records, got {len(selected)}"
        )

    def summarize(rows: list[Any]) -> dict[str, Any]:
        count = len(rows)
        return {
            "n_tasks": count,
            "strict_pass_rate": (
                sum(bool(row.outcome.verifier_passed) for row in rows) / count
            ),
            "continuous_mean": (
                sum(float(row.outcome.normalized_score) for row in rows) / count
            ),
            "outcome_pass_rate": (
                sum(bool(row.outcome.outcome_passed) for row in rows) / count
            ),
            "process_pass_rate": (
                sum(bool(row.outcome.process_passed) for row in rows) / count
            ),
        }

    by_role = {
        role: summarize([row for row in selected if row.task_role == role])
        for role in sorted(expected_roles)
        if any(row.task_role == role for row in selected)
    }
    payload = {
        "schema_version": 1,
        "protocol": f"frozen_final_library_repeat{target.repeat_index}",
        "method": target.method,
        "block": target.block,
        "source_run": str(target.source_root),
        "overall": summarize(selected),
        "by_role": by_role,
        "tasks": [
            {
                "task_id": row.task_id,
                "family_id": row.family_id,
                "env_id": row.env_id,
                "role": row.task_role,
                "strict_pass": bool(row.outcome.verifier_passed),
                "continuous_score": float(row.outcome.normalized_score),
                "outcome_pass": bool(row.outcome.outcome_passed),
                "process_pass": bool(row.outcome.process_passed),
            }
            for row in sorted(selected, key=lambda item: item.task_id)
        ],
    }
    path = runtime.run_root / "reports" / "frozen_eval_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


async def _run_target(target: EvalTarget) -> None:
    config = _load_eval_config(target)
    schedule, registry = _target_schedule(config, target)
    if _target_complete(target, schedule):
        print(f"SKIP complete: {target.run_id}", flush=True)
        return
    if target.run_root.exists() and any(target.run_root.iterdir()):
        raise FileExistsError(
            f"incomplete non-empty target exists: {target.run_root}. "
            "Inspect it before removal; this runner never overwrites results."
        )

    runner = LifelongRunner(config=config)
    runner._preflight()
    runtime = BaselineRuntime.build(config)
    source_hashes = _copy_final_libraries(
        target.source_root, runtime.run_root
    )
    runner._persist_run_config(runtime.run_root)
    benchmark_hash = runner._snapshot_benchmark_hash(runtime.run_root)

    audit = {
        "schema_version": 1,
        "protocol": f"frozen_final_library_repeat{target.repeat_index}",
        "method": target.method,
        "block": target.block,
        "source_run": str(target.source_root),
        "source_library_hashes": source_hashes,
        "source_library_paths": {
            env_id: str(_source_library_path(target.source_root, env_id))
            for env_id in source_hashes
        },
        "selected_task_count": len(schedule),
        "selected_task_ids": [record.spec.task_id for record in schedule],
        "skill_mutation_enabled": False,
        "ccv_enabled": False,
        "local_edit_enabled": False,
        "model": "openai/gpt-5.4-mini",
        "reasoning_effort": "medium",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (runtime.run_root / "evaluation_protocol.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    from harbor.job import Job
    from skillevolbench.harbor_ext import SkillEvolBenchHooks
    from skillevolbench.harbor_ext._patches import apply_harbor_patches

    apply_harbor_patches()
    job = await Job.create(build_job_config(config, schedule))
    hooks = SkillEvolBenchHooks(
        runtime=runtime,
        task_registry=registry,
        runtime_builder=runtime.runtime_builder,
        prompt_builder=runtime.prompt_builder,
    )

    async def frozen_on_trial_started(event: Any) -> None:
        task = hooks._resolve_task(event)
        if hooks._current_env != task.environment_id:
            if hooks._current_env is not None:
                await hooks._handle_env_transition(
                    hooks._current_env, task.environment_id
                )
            runtime.switch_env(task.environment_id)
            runtime.freeze_ctrl.freeze(task.environment_id)
            hooks._current_env = task.environment_id
        elif not runtime.freeze_ctrl.frozen:
            runtime.freeze_ctrl.freeze(task.environment_id)
        await hooks.on_trial_started(event)

    job.on_trial_started(frozen_on_trial_started)
    job.on_trial_ended(hooks.on_trial_ended)
    runtime.event_store.record(
        "run_started",
        {
            "run_id": config.run_id,
            "evaluation_only": True,
            "source_run": str(target.source_root),
            "benchmark_hash": benchmark_hash,
            "n_tasks": len(schedule),
            "block": target.block,
        },
    )

    try:
        await job.run()
    finally:
        if hooks._current_env is not None:
            await hooks._handle_env_transition(hooks._current_env, "END")
            hooks._current_env = None
        if _library_hashes(target.source_root) != source_hashes:
            raise RuntimeError("source library changed during frozen evaluation")
        if _benchmark_digest(REPO_ROOT / "benchmark") != benchmark_hash:
            raise RuntimeError("benchmark assets changed during frozen evaluation")

    report_generator = ReportGenerator(
        runtime.run_root,
        config,
        task_registry=registry,
        host_llm_clients=runtime.host_llm_clients,
    )
    report_path = report_generator.write(report_generator.generate())
    frozen_summary_path = _write_frozen_summary(target, runtime)
    runtime.event_store.record(
        "run_finished",
        {
            "run_id": config.run_id,
            "evaluation_only": True,
            "report_path": str(report_path),
            "frozen_summary_path": str(frozen_summary_path),
        },
    )
    print(
        f"DONE {target.method} {target.block}: {frozen_summary_path}",
        flush=True,
    )


def _check_target(target: EvalTarget) -> None:
    config = _load_eval_config(target)
    LifelongRunner(config=config)._preflight()
    schedule, _ = _target_schedule(config, target)
    hashes = _library_hashes(target.source_root)
    source_config = RunConfig.model_validate_json(
        (target.source_root / "config.json").read_text()
    )
    if source_config.baseline.model_name != "openai/gpt-5.4-mini":
        raise RuntimeError("source task-agent model is not GPT-5.4-mini")
    if source_config.baseline.agent_kwargs.get("reasoning_effort") != "medium":
        raise RuntimeError("source task-agent reasoning effort is not medium")
    print(
        json.dumps(
            {
                "run_id": target.run_id,
                "source_run": str(target.source_root),
                "block": target.block,
                "selected_tasks": len(schedule),
                "first_task": schedule[0].spec.task_id,
                "last_task": schedule[-1].spec.task_id,
                "source_library_hashes": hashes,
                "source_library_paths": {
                    env_id: str(_source_library_path(target.source_root, env_id))
                    for env_id in hashes
                },
                "output_exists": target.run_root.exists(),
                "model": config.baseline.model_name,
                "reasoning_effort": config.baseline.agent_kwargs.get(
                    "reasoning_effort"
                ),
                "skill_mutation_enabled": False,
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("baseline", "noreg", "regularization"))
    parser.add_argument("block", choices=("t1-t3", "t4-t6"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--repeat", type=int, choices=(1, 2), default=2)
    parser.add_argument("--only-task", help="Explicit task ID; validated against the selected block")
    parser.add_argument(
        "--check", action="store_true", help="read-only protocol validation"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    target = replace(_target(args.method, args.block, args.source_run.resolve()),
                     output_root=args.output_root.resolve(), repeat_index=args.repeat)
    if args.only_task:
        target = replace(target, only_task=args.only_task)
    if args.check:
        _check_target(target)
    else:
        asyncio.run(_run_target(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
