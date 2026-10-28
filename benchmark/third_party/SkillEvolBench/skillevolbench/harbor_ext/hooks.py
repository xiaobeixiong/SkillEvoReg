"""``SkillEvolBenchHooks`` -- the protocol's nervous system (Part 4 §4.2).

Two async hooks register with ``harbor.Job``:

* ``on_trial_started`` -- before container start. Performs curated injection
                          (Path B), zero-shot induction (Self-Gen-Zero-Shot),
                          env transition + freeze, retrieval, runtime build.
* ``on_trial_ended``   -- after verifier completes. Parses outcome, persists
                          ReplayRecord, and (in learning blocks) dispatches
                          to ``strategy.decide()``.

This module **does not** import Harbor at module load. Harbor's
``TrialHookEvent`` type is imported under ``TYPE_CHECKING`` so the hook
class is unit-testable on machines without the Harbor SDK. The hook
forwards calls to a ``RuntimeProtocol`` (Part 4 ``types.py``) -- Part 8's
``BaselineRuntime`` is the concrete implementer.

The flow tracks Engineering Design §4.2 verbatim. Methods ending in ``_`` are
internal hooks for unit-test seams.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevolbench.schemas import OperationType, SkillPatch, TaskRole, TaskPhase

if TYPE_CHECKING:
    # These types only need to exist at type-check time. They are NOT
    # imported at runtime to keep this module Harbor-free at import.
    from harbor.trial.hooks import TrialHookEvent  # type: ignore

    from skillevolbench.harbor_ext.types import (
        RuntimeProtocol,
        TaskRegistryProtocol,
        RuntimeBuilderProtocol,
        PromptBuilderProtocol,
    )


_LOG = logging.getLogger(__name__)


_LEARNING_ROLES = {TaskRole.CANONICAL, TaskRole.ENRICHED, TaskRole.VARIANT}
_EVAL_ROLES = {TaskRole.CONTEXT_SHIFT, TaskRole.ADVERSARIAL, TaskRole.COMPOSITION}


# Keep in sync with skillevolbench.harbor_ext.job_builder.SHADOW_T6_SUFFIX
# and REPLAY_SUFFIX. Duplicated here to avoid an import cycle.
_SHADOW_T6_SUFFIX = "__oracle_shadow"
_REPLAY_SUFFIX = "__replay"
_LEARNING_PASS2_SUFFIX = "__pass2"
_CCV_OLD_SUFFIX = "__ccv_old"
_CCV_CANDIDATE_SUFFIX = "__ccv_candidate"
_CCV_OLD_CLEAN_SUFFIX = "__ccv_old_clean"
_CCV_CANDIDATE_CLEAN_SUFFIX = "__ccv_candidate_clean"
_CCV_OLD_ATTACK_SUFFIX = "__ccv_old_attack"
_CCV_CANDIDATE_ATTACK_SUFFIX = "__ccv_candidate_attack"


def _is_shadow_trial_name(raw: str) -> bool:
    """A dual-T6 shadow trial's runtime dir basename ends with the suffix.

    Use ``endswith`` not ``in``: a substring check would misclassify any
    future task whose name happens to contain the literal "__oracle_shadow"
    as a shadow trial. The suffix is appended at the end by job_builder
    when building the shadow runtime dir, so endswith is the precise
    invariant.
    """
    return raw.endswith(_SHADOW_T6_SUFFIX)


def _is_replay_trial_name(raw: str) -> bool:
    """A within-env replay trial's runtime dir basename ends with the suffix."""
    return raw.endswith(_REPLAY_SUFFIX)


def _ccv_side(raw: str) -> str:
    if raw.endswith(_CCV_OLD_CLEAN_SUFFIX):
        return "old_clean"
    if raw.endswith(_CCV_CANDIDATE_CLEAN_SUFFIX):
        return "candidate_clean"
    if raw.endswith(_CCV_OLD_ATTACK_SUFFIX):
        return "old_attack"
    if raw.endswith(_CCV_CANDIDATE_ATTACK_SUFFIX):
        return "candidate_attack"
    if raw.endswith(_CCV_OLD_SUFFIX):
        return "old"
    if raw.endswith(_CCV_CANDIDATE_SUFFIX):
        return "candidate"
    return ""


def _learning_pass_from_name(raw: str) -> int:
    return 2 if _LEARNING_PASS2_SUFFIX in raw else 1


def _step_local_ccv_gate(
    *,
    old_library_hash: str,
    candidate_library_hash: str,
    training_passed: bool,
    training_score: float,
) -> dict[str, Any]:
    """Choose whether a per-step CCV diagnostic needs real execution.

    The primary training rollout is the source qualification for S_old, as in
    SpreadsheetBench.  It used the pre-update skill and the exact same source
    task.  A failed source is not attackable; an unchanged candidate has no
    update to attack.
    """
    if (
        old_library_hash
        and candidate_library_hash
        and old_library_hash == candidate_library_hash
    ):
        return {
            "active": False,
            "old_clean_reused": False,
            "probe": {
                "status": "skipped_no_material_candidate",
                "error": "old and candidate library hashes are identical",
            },
            "skip_reason": "identical_old_candidate_library_hash",
        }
    if not training_passed:
        return {
            "active": False,
            "old_clean_reused": False,
            "probe": {
                "status": "inconclusive_unqualified_source",
                "error": "the pre-update training rollout did not pass",
            },
            "skip_reason": "training_source_not_verifier_passed",
        }
    return {
        "active": True,
        "old_clean_reused": True,
        "old_clean_passed": True,
        "old_clean_score": float(training_score),
        "probe": {},
        "skip_reason": "",
    }


class SkillEvolBenchHooks:
    """Trial-lifecycle hooks driving the lifelong protocol.

    The hook is owned by Part 9's ``LifelongRunner`` -- it constructs the
    runtime + registry + builders, then registers the hook on the
    ``harbor.Job``. After registration, all hook activity is driven by
    Harbor's per-trial dispatch.
    """

    def __init__(
        self,
        runtime: "RuntimeProtocol",
        task_registry: "TaskRegistryProtocol",
        runtime_builder: "RuntimeBuilderProtocol",
        prompt_builder: "PromptBuilderProtocol",
    ) -> None:
        self.runtime = runtime
        self.task_registry = task_registry
        self.runtime_builder = runtime_builder
        self.prompt_builder = prompt_builder

        # Bookkeeping across the lifetime of one Harbor Job:
        self._current_env: str | None = None
        self._eval_outcomes_buffer: list = []
        self._pre_state_cache: dict[str, dict] = {}
        # Count of LEARNING-role trials completed per env. Used to detect
        # the last learning trial of an env (which is the right moment to
        # freeze, BEFORE the first eval trial's container is constructed).
        # See _on_trial_ended_sync's freeze trigger.
        self._learning_completed_per_env: dict[str, int] = {}
        # CCV may temporarily switch old/new snapshots for paired execution.
        # Its consumption policy is independent of regression detection:
        # deferred mode supplies the next proposal, while local-edit mode
        # performs one immediate answer-free corrective rewrite.
        self._pending_ccv_candidates: dict[str, dict[str, Any]] = {}
        # Harbor emits END for every failed attempt. Track transient failures
        # by semantic runtime name so the first N attempts remain side-effect
        # free, but an exhausted retry budget aborts the lifelong chain rather
        # than silently skipping one state-defining task.
        self._infrastructure_failures: dict[str, int] = {}
        # Exact per-family state immediately before the second learning pass.
        # Family-end CCV uses this immutable anchor as its old side.
        self._late_regularization_family_anchors: dict[str, dict[str, Any]] = {}

    # ==================================================================
    # HOOK 1: on_trial_started
    # ==================================================================

    async def on_trial_started(self, event: "TrialHookEvent") -> None:
        """Pre-trial: curated/zero-shot seed, freeze trigger, retrieval, build."""
        task = self._resolve_task(event)
        runtime_basename = self._runtime_basename(event)
        learning_pass = _learning_pass_from_name(runtime_basename)
        is_shadow = _is_shadow_trial_name(runtime_basename)

        # Shadow trials are dual-T6 second runs. They short-circuit ALL
        # mutating logic (curated inject / zero-shot create / env
        # transition / freeze) -- the primary trial that ran microseconds
        # earlier already did all of that, the library is frozen, and the
        # only thing the shadow needs is a fresh OracleRetriever-based
        # instruction.md targeted at its own runtime dir.
        if is_shadow:
            await self._on_shadow_trial_started(event, task, runtime_basename)
            return

        ccv_side = _ccv_side(runtime_basename)
        if ccv_side:
            await self._on_ccv_trial_started(
                event, task, runtime_basename, ccv_side
            )
            return

        # Detect within-env replay trial. Replays share a TaskSpec with
        # an original (same family, same role, same task content) but
        # run AFTER the env's 30 originals have completed -- library is
        # at its post-evolution state and must stay frozen for the
        # replay's duration. We branch BEFORE env-transition logic so
        # replays don't accidentally trigger unfreeze/maintain.
        is_replay = _is_replay_trial_name(runtime_basename)

        # ---- 1. Env transition: unfreeze previous env + maintain ----
        # MUST run before steps 2-3 (Path-A/B writes to library): on the
        # first task of a new env, the previous env's eval block left the
        # library frozen, so any inject_curated / create_skill call here
        # would raise PermissionError("... called while frozen") and crash
        # the run. Unfreezing first lets the new env's canonical task
        # mutate the library legally.
        # Replay trials never trigger env transition: by construction
        # they're scheduled AFTER the env's 30 originals, so the env_id
        # hasn't changed (unfreeze would let mutation back in, the very
        # thing replay is meant to forbid).
        if (
            not is_replay
            and self._current_env is not None
            and task.environment_id != self._current_env
        ):
            await self._handle_env_transition(self._current_env, task.environment_id)
        self._current_env = task.environment_id

        # ---- 1b. Per-env library swap (library_scope=="environment") ----
        # Lazy-init this env's LibraryStore (own git repo + manifest) on
        # first encounter; subsequent calls within the same env are cheap
        # dict lookups. Also rebuilds freeze_ctrl + snapshot_store so they
        # point at the current env's library. No-op for library_scope=
        # "global" (single library wired at runtime build time).
        # MUST run BEFORE inject_curated / zero_shot_create / freeze
        # below -- they access ``runtime.library`` and need the right
        # env's LibraryStore.
        self.runtime.switch_env(task.environment_id)

        if (
            learning_pass == 2
            and task.role == TaskRole.CANONICAL.value
            and task.family_id not in self._late_regularization_family_anchors
        ):
            from skillevolbench.components.online_regularization import (
                measure_family_complexity,
            )
            self._late_regularization_family_anchors[task.family_id] = {
                "commit": self.runtime.library.head_commit(),
                "library_hash": self.runtime.library.compute_hash(),
                "skill": self._family_skill_text(task.family_id),
                "complexity": measure_family_complexity(
                    self.runtime.library, task.family_id
                ),
            }
            self.runtime.event_store.record("late_regularization_anchor_captured", {
                "task_id": task.task_id,
                "family_id": task.family_id,
                "learning_pass": learning_pass,
                **self._late_regularization_family_anchors[task.family_id],
            })

        # Shared-T1 paired runs omit the stochastic canonical execution from
        # the schedule. Immediately before the family's first real task (T2),
        # install the exact baseline-derived post-T1 skill and replay state.
        # This runs before retrieval so T2 observes the same initial library
        # and cumulative T1 trajectory as the source baseline.
        if (
            not is_replay
            and task.role == TaskRole.ENRICHED.value
            and bool(getattr(
                self.runtime.strategy_config,
                "shared_t1_skip_execution",
                False,
            ))
        ):
            self._ensure_shared_t1_state(task)

        # ---- 2. Path-B curated injection (on first arrival of family) ----
        # SKIPPED for replays: curated v0 was already injected during the
        # original T1 of this family; replay must observe the post-
        # evolution library, not re-inject v0.
        if (
            not is_replay
            and self.runtime.baseline.allow_curated_inject
            and task.role == TaskRole.CANONICAL.value
            and not self.runtime.library.has_curated_for(task.family_id)
        ):
            self._inject_curated_v0(task)

        # ---- 3. Self-Gen-Zero-Shot creation (before any execution) ----
        # SKIPPED for replays (same rationale as step 2).
        if (
            not is_replay
            and self.runtime.baseline.allow_zero_shot_creation
            and task.role == TaskRole.CANONICAL.value
            and not self.runtime.library.has_seed_for(task.family_id)
        ):
            await self._create_zero_shot_skill(task)

        # ---- 4. Freeze trigger fallback: idempotent if T3 trial_ended
        # already wrote the marker (the canonical place; see
        # _on_trial_ended_sync). We keep this T4-START fallback so a
        # missed T3 ended (e.g., orphan_trial_ended) still flips into
        # the eval block.
        # SKIPPED for replays: library is already frozen (replays only
        # ever run after the eval block, so freeze marker was set by T4
        # of this env's original phase). Re-asserting freeze here is
        # idempotent but the explicit skip makes intent clear.
        if (
            not is_replay
            and task.role == TaskRole.CONTEXT_SHIFT.value
            and not self.runtime.freeze_ctrl.frozen
        ):
            self.runtime.freeze_ctrl.freeze(task.environment_id)

        # ---- 5. Skill retrieval (if baseline uses skill library) ----
        retrieval = None
        if self.runtime.baseline.use_skill_library:
            instruction_text = self._read_instruction(task)
            retrieval = self.runtime.retriever.retrieve(
                query=instruction_text,
                library=self.runtime.library,
                k=self.runtime.baseline.skill_retrieval_k,
            )
            if hasattr(self.runtime, "retrieval_store"):
                self.runtime.retrieval_store.record(
                    task_id=(
                        runtime_basename if learning_pass > 1 else task.task_id
                    ),
                    retrieval=retrieval,
                    required_skills=task.required_skills,
                )

        # ---- 5b. Training-time atomic Dropout view -------------------
        # The task agent must experience the same masked skill that later
        # conditions SkillAuthor.  GlobalLibraryEnvironment already bound a
        # per-trial directory; populate it now, before the container starts,
        # without changing the live library or its hash.
        execution_dropout_report, execution_skill_text_overrides = (
            self._prepare_training_dropout_view(
                task, runtime_basename, learning_pass,
                is_replay=is_replay,
            )
        )

        # ---- 6. Trajectory / history retrieval (control baselines) ----
        trajectories: list = []
        if self.runtime.baseline.use_trajectory_rag and self.runtime.trajectory_retriever:
            trajectories = self.runtime.trajectory_retriever.retrieve(
                task=task, k=self.runtime.baseline.trajectory_retrieval_k,
            )
        history_context: str | None = None
        if self.runtime.baseline.use_history_context and self.runtime.history_retriever:
            history_context = self.runtime.history_retriever.build_context(
                task=task,
                max_tokens=self.runtime.baseline.history_context_max_tokens,
            )

        # ---- 7. Build runtime task copy (overwrites instruction.md) ----
        # Pass runtime_basename so replay/shadow trials write to their own
        # __replay / __oracle_shadow runtime dir (NOT the bare task_id dir,
        # which would clobber the original trial's instruction.md and leave
        # the replay agent reading an un-injected bare task instruction).
        runtime_task_dir = self.runtime_builder.build(
            task=task,
            run_root=self.runtime.run_root,
            baseline=self.runtime.baseline,
            retrieved_skills=(retrieval.skills if retrieval else []),
            retrieved_trajectories=trajectories,
            history_context=history_context,
            library_frozen=self.runtime.freeze_ctrl.frozen,
            runtime_basename=runtime_basename,
        )
        if execution_dropout_report is not None:
            import json
            audit_path = runtime_task_dir / "injection-context.json"
            audit = json.loads(audit_path.read_text())
            audit["training_dropout"] = {
                "enabled": bool(execution_dropout_report.get("enabled")),
                "probability": execution_dropout_report.get("probability", 0.0),
                "eligible_atoms": execution_dropout_report.get("eligible_atoms", 0),
                "masked_count": execution_dropout_report.get("masked_count", 0),
                "masked_atoms": [
                    {
                        "mask_id": row.get("mask_id"),
                        "skill_id": row.get("skill_id"),
                        "atom_id": row.get("atom_id"),
                        "atom_type": row.get("atom_type"),
                    }
                    for row in execution_dropout_report.get("masked_atoms", [])
                ],
                "body_text_exposed": False,
            }
            audit_path.write_text(json.dumps(audit, indent=2))

        # ---- 8. Cache pre-state for on_trial_ended ----
        cache_key = (
            runtime_basename
            if (is_shadow or is_replay or learning_pass > 1)
            else task.task_id
        )
        self._pre_state_cache[cache_key] = {
            "library_hash": self.runtime.library.compute_hash(),
            "retrieval": retrieval,
            "runtime_dir": runtime_task_dir,
            "started_at": datetime.now(timezone.utc),
            "execution_dropout_report": execution_dropout_report,
            "execution_skill_text_overrides": execution_skill_text_overrides,
        }

        self.runtime.event_store.record(
            "trial_started",
            {
                "task_id": task.task_id,
                "learning_pass": learning_pass,
                "role": task.role,
                "phase": task.phase,
                "library_hash": self.runtime.library.compute_hash(),
                "retrieved_skill_ids": (
                    [s.skill_id for s in retrieval.skills] if retrieval else []
                ),
                "training_dropout_masked_count": int(
                    (execution_dropout_report or {}).get("masked_count", 0)
                ),
            },
        )

    def _prepare_training_dropout_view(
        self, task: Any, runtime_basename: str, learning_pass: int, *,
        is_replay: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        """Build the per-trial masked library seen by the training agent."""
        cfg = self.runtime.strategy_config
        role = task.role.value if hasattr(task.role, "value") else str(task.role)
        active = bool(
            not is_replay
            and role in {r.value for r in _LEARNING_ROLES}
            and getattr(cfg, "regularization_proposal_regularization", False)
            and getattr(cfg, "regularization_dropout_enabled", False)
            and learning_pass >= int(getattr(
                cfg, "regularization_start_learning_pass", 1
            ))
            and float(getattr(cfg, "regularization_dropout_probability", 0.0)) > 0.0
        )
        if not active:
            return None, None

        from skillevolbench.components.online_regularization import (
            mask_skill_texts,
            materialize_training_skill_view,
            stable_seed,
        )

        original_texts: dict[str, str] = {}
        for entry in self.runtime.library.skills_in_family(task.family_id):
            original_texts[entry.skill_id] = self.runtime.library.get_skill(
                entry.skill_id
            ).files.get("SKILL.md", "")
        masked_texts, report = mask_skill_texts(
            original_texts,
            probability=float(getattr(
                cfg, "regularization_dropout_probability", 0.0
            )),
            seed=stable_seed(
                task.task_id, f"learning-pass-{learning_pass}",
                "training-rollout-dropout",
            ),
            max_atoms=int(getattr(cfg, "regularization_dropout_max_atoms", 4)),
            min_eligible_atoms=int(getattr(
                cfg, "regularization_dropout_min_eligible_atoms", 4
            )),
            ensure_at_least_one=bool(getattr(
                cfg, "regularization_dropout_ensure_at_least_one", True
            )),
        )
        report["scope_skill_ids"] = list(original_texts)
        report["mode"] = "training_rollout"
        report["body_text_exposed"] = False
        view_path = (
            self.runtime.run_root / "runtime" / runtime_basename
            / "training-skill-view"
        )
        materialize_training_skill_view(
            self.runtime.library.active_dir, view_path, masked_texts
        )
        self.runtime.event_store.record("training_dropout_view_prepared", {
            "task_id": task.task_id,
            "learning_pass": learning_pass,
            "probability": report.get("probability", 0.0),
            "eligible_atoms": report.get("eligible_atoms", 0),
            "masked_count": report.get("masked_count", 0),
            "scope_skill_ids": report.get("scope_skill_ids", []),
            "masked_atom_refs": [
                {
                    "mask_id": row.get("mask_id"),
                    "skill_id": row.get("skill_id"),
                    "atom_id": row.get("atom_id"),
                    "atom_type": row.get("atom_type"),
                }
                for row in report.get("masked_atoms", [])
            ],
            "live_library_hash": self.runtime.library.compute_hash(),
            "body_text_exposed": False,
        })
        return report, masked_texts

    async def _on_ccv_trial_started(
        self, event: "TrialHookEvent", task: Any, runtime_basename: str,
        side: str,
    ) -> None:
        """Prepare a feedback-only paired CCV diagnostic.

        Snapshot switches are temporary execution setup, not update decisions:
        the candidate snapshot is restored for the second side and remains the
        final library regardless of the diagnostic outcome.
        """
        del event
        self.runtime.switch_env(task.environment_id)
        pending = self._pending_ccv_candidates.get(task.task_id)
        active = bool(pending and pending.get("active"))
        step_local = side in {
            "old_clean", "candidate_clean", "old_attack", "candidate_attack",
        }
        materialize_attack = side in {
            "old_attack", "candidate_attack", "old", "candidate",
        }
        # Step-local mode executes two real clean qualification slots followed
        # by the paired attacked slots.  Legacy mode keeps the historical
        # two-slot diagnostic behavior.
        source_task = task
        if active and step_local:
            if side == "old_clean":
                if pending.get("old_clean_reused"):
                    # The pre-update training rollout already passed this
                    # exact source. Drain the immutable Harbor slot without a
                    # second target-model execution.
                    active = False
                else:
                    self.runtime.library.rollback_to(
                        pending["old_commit"], reason="ccv_clean_old"
                    )
            elif side == "candidate_clean":
                self.runtime.library.rollback_to(
                    pending["candidate_commit"], reason="ccv_clean_candidate"
                )
            elif side == "old_attack":
                clean_qualified = bool(
                    pending.get("old_clean_passed")
                    and pending.get("candidate_clean_passed")
                )
                if clean_qualified:
                    probe = self.runtime.ccv_adapter.generate(
                        original_instruction=self._read_instruction(source_task),
                        old_skill=pending["old_skill"],
                        candidate_skill=pending["candidate_skill"],
                        task_id=task.task_id,
                        trajectory_summary=pending.get(
                            "trajectory_summary", ""
                        ),
                        candidate_changed_skill_ids=list(
                            pending.get("ccv_candidate_changed_skill_ids")
                            or ()
                        ),
                    )
                    pending["probe"] = probe.to_dict()
                    active = probe.status == "ready"
                else:
                    from skillevolbench.components.ccv import CCVProbe
                    probe = CCVProbe(
                        status="inconclusive_unqualified_source",
                        error=(
                            "old and candidate must both pass the clean "
                            "source before attack generation"
                        ),
                    )
                    pending["probe"] = probe.to_dict()
                    active = False
                pending["active"] = active
                if active:
                    self.runtime.library.rollback_to(
                        pending["old_commit"], reason="ccv_attack_old"
                    )
            elif side == "candidate_attack":
                active = bool(
                    pending.get("active")
                    and (pending.get("probe") or {}).get("status") == "ready"
                )
                if active:
                    self.runtime.library.rollback_to(
                        pending["candidate_commit"],
                        reason="ccv_attack_candidate",
                    )
        elif active and side == "old":
            if pending.get("late_family_end"):
                qualification = {
                    "qualified": True,
                    "source_task_id": source_task.task_id,
                    "old_library_hash": pending["old_library_hash"],
                    "reason": "late_pass_anchor_paired_old_trial",
                }
            else:
                qualification = self._qualify_ccv_source(
                    source_task,
                    pending["old_library_hash"],
                    record_task_id=pending.get("source_record_task_id"),
                )
            pending["source_qualification"] = qualification
            if qualification["qualified"]:
                probe = self.runtime.ccv_adapter.generate(
                    original_instruction=self._read_instruction(source_task),
                    old_skill=pending["old_skill"],
                    candidate_skill=pending["candidate_skill"],
                    task_id=task.task_id,
                    trajectory_summary=pending.get("trajectory_summary", ""),
                    candidate_changed_skill_ids=list(
                        pending.get("ccv_candidate_changed_skill_ids") or ()
                    ),
                )
            else:
                from skillevolbench.components.ccv import CCVProbe
                probe = CCVProbe(
                    status="inconclusive_unqualified_source",
                    error=qualification["reason"],
                )
            pending["probe"] = probe.to_dict()
            active = probe.status == "ready"
            pending["active"] = active
            if active:
                self.runtime.library.rollback_to(
                    pending["old_commit"], reason="ccv_paired_old"
                )
        elif active and side == "candidate":
            # Restore the exact staged candidate before its paired run.
            self.runtime.library.rollback_to(
                pending["candidate_commit"], reason="ccv_paired_candidate"
            )

        retrieval = self.runtime.retriever.retrieve(
            query=self._read_instruction(source_task), library=self.runtime.library,
            k=self.runtime.baseline.skill_retrieval_k,
        )
        runtime_dir = self.runtime_builder.build(
            task=source_task, run_root=self.runtime.run_root,
            baseline=self.runtime.baseline,
            retrieved_skills=retrieval.skills,
            retrieved_trajectories=[], history_context=None,
            library_frozen=False, runtime_basename=runtime_basename,
        )
        if active and materialize_attack:
            from skillevolbench.components.ccv import CCVProbe
            probe = CCVProbe(**pending["probe"])
            attacked = self.runtime.ccv_adapter.materialize(
                (self.task_registry.task(source_task.task_id).folder /
                 source_task.harbor.instruction).read_text(), probe,
            )
            # Rebuild the injected prompt with the attacked public instruction.
            attacked_injected = self.prompt_builder.build(
                original_instruction=attacked, baseline=self.runtime.baseline,
                task=source_task, retrieved_skills=retrieval.skills,
                retrieved_trajectories=[], history_context=None,
                library_frozen=False,
            )
            (runtime_dir / source_task.harbor.instruction).write_text(attacked_injected)
        elif not active:
            # Harbor's task graph is immutable once the job starts. If no valid
            # patch/probe exists, drain the pre-scheduled slot cheaply.
            (runtime_dir / source_task.harbor.instruction).write_text(
                "This pre-scheduled CCV slot is inactive because no valid "
                "diagnostic pair is available. Do not inspect or modify "
                "the repository. Finish immediately.\n"
            )
        self._pre_state_cache[runtime_basename] = {
            "library_hash": self.runtime.library.compute_hash(),
            "retrieval": retrieval, "runtime_dir": runtime_dir,
            "started_at": datetime.now(timezone.utc), "is_ccv": True,
        }
        self.runtime.event_store.record("ccv_trial_started", {
            "task_id": task.task_id, "runtime": runtime_basename,
            "side": side, "active": active,
            "source_task_id": source_task.task_id,
            "source_qualification": (
                pending.get("source_qualification", {}) if pending else {}
            ),
            "probe": pending.get("probe", {}) if pending else {},
        })

    async def _on_shadow_trial_started(
        self,
        event: "TrialHookEvent",
        task: Any,
        runtime_basename: str,
    ) -> None:
        """Dual-T6 shadow: build instruction.md at the shadow dir using
        OracleRetriever (required_skills) and cache pre-state under a
        shadow-suffixed key so the ended hook can recover it.

        Library is frozen at this point (we're inside the eval block of
        the env that just produced the primary T6 trial), so the second
        retrieval/agent-run sees the identical library state -- the
        comparison is paired-by-trial.
        """
        # Build a fresh OracleRetriever; falls back to the runtime's
        # primary retriever for non-T6 (defensive -- shadows are only
        # scheduled for T6 anyway).
        from skillevolbench.components.retriever import OracleRetriever
        oracle = OracleRetriever(
            task_registry=self.task_registry,
            inner=self.runtime.retriever,
        )
        instruction_text = self._read_instruction(task)
        retrieval = oracle.retrieve(
            query=instruction_text,
            library=self.runtime.library,
            k=self.runtime.baseline.skill_retrieval_k,
            task_id=task.task_id,
        )

        # Build instruction.md + injection-context.json at the shadow's
        # own runtime dir (RuntimeBuilder uses task.task_id by default;
        # for the shadow we need the suffixed dir, so we patch
        # ``task.task_id`` momentarily via a small wrapper object).
        from types import SimpleNamespace
        shadow_task = SimpleNamespace(
            task_id=runtime_basename,                # <task_id>__oracle_shadow
            task_slug=task.task_slug,
            family_id=task.family_id,
            environment_id=task.environment_id,
            role=task.role,
            phase=task.phase,
            harbor=task.harbor,
            required_skills=task.required_skills,
            latent_skill_id=task.latent_skill_id,
        )
        runtime_task_dir = self.runtime_builder.build(
            task=shadow_task,
            run_root=self.runtime.run_root,
            baseline=self.runtime.baseline,
            retrieved_skills=(retrieval.skills if retrieval else []),
            retrieved_trajectories=[],   # shadow does not see traj RAG
            history_context=None,        # shadow does not see history
            library_frozen=True,         # always frozen (we're at T6 eval)
            runtime_basename=runtime_basename,
        )

        # Cache pre-state under the shadow-suffixed key.
        self._pre_state_cache[runtime_basename] = {
            "library_hash": self.runtime.library.compute_hash(),
            "retrieval": retrieval,
            "runtime_dir": runtime_task_dir,
            "started_at": datetime.now(timezone.utc),
            "is_shadow": True,
        }

        self.runtime.event_store.record(
            "shadow_trial_started",
            {
                "task_id": task.task_id,
                "shadow_runtime": runtime_basename,
                "library_hash": self.runtime.library.compute_hash(),
                "retrieved_skill_ids": (
                    [s.skill_id for s in retrieval.skills] if retrieval else []
                ),
            },
        )

    # ==================================================================
    # HOOK 2: on_trial_ended
    # ==================================================================

    async def on_trial_ended(self, event: "TrialHookEvent") -> None:
        """Post-trial: parse outcome, persist replay, dispatch to strategy."""
        if event.result is None:
            self.runtime.event_store.record(
                "trial_ended_no_result", {"task_name": getattr(event, "task_name", "?")}
            )
            return
        # Run the substantive logic in a worker thread so we don't block
        # Harbor's event loop while doing sqlite + git work.
        await asyncio.to_thread(self._on_trial_ended_sync, event)

    def _on_trial_ended_sync(self, event: "TrialHookEvent") -> None:
        from skillevolbench.harbor_ext.retry import (
            is_retryable_infrastructure_exception,
        )

        task = self._resolve_task(event)
        runtime_basename = self._runtime_basename(event)
        learning_pass = _learning_pass_from_name(runtime_basename)
        is_shadow = _is_shadow_trial_name(runtime_basename)
        is_replay = _is_replay_trial_name(runtime_basename)
        ccv_side = _ccv_side(runtime_basename)
        # Pre-state was cached under the runtime-dir basename so primary
        # and shadow / replay records of the same task_id don't alias.
        cache_key = (
            runtime_basename
            if (is_shadow or is_replay or ccv_side or learning_pass > 1)
            else task.task_id
        )
        pre_state = self._pre_state_cache.pop(cache_key, None)
        if pre_state is None:
            self.runtime.event_store.record(
                "orphan_trial_ended",
                {"task_id": task.task_id, "is_shadow": is_shadow},
            )
            return

        # Harbor retries these failures in a newly-created Trial/container.
        # Do not let a disposable failed attempt enter replay memory, evolve a
        # skill, advance freeze counters, or affect CCV state.  Harbor's own
        # Job hook temporarily records the attempt and replaces it when the
        # retry starts/completes.
        exception_info = getattr(event.result, "exception_info", None)
        exception_type = getattr(exception_info, "exception_type", None)
        if is_retryable_infrastructure_exception(exception_type):
            failures = self._infrastructure_failures.get(runtime_basename, 0) + 1
            self._infrastructure_failures[runtime_basename] = failures
            max_retries = int(getattr(
                self.runtime.run_config,
                "harbor_infrastructure_max_retries",
                0,
            ))
            self.runtime.event_store.record(
                "trial_infrastructure_error_ignored",
                {
                    "task_id": task.task_id,
                    "runtime_basename": runtime_basename,
                    "exception_type": exception_type,
                    "failed_attempt": failures,
                    "max_retries": max_retries,
                    "library_hash_unchanged": self.runtime.library.compute_hash(),
                },
            )
            if failures > max_retries:
                self.runtime.event_store.record(
                    "trial_infrastructure_retries_exhausted",
                    {
                        "task_id": task.task_id,
                        "runtime_basename": runtime_basename,
                        "exception_type": exception_type,
                        "total_attempts": failures,
                        "max_retries": max_retries,
                    },
                )
                raise RuntimeError(
                    "SkillEvolBench infrastructure retry budget exhausted for "
                    f"{runtime_basename}: {exception_type} persisted for "
                    f"{failures} attempts"
                )
            return
        self._infrastructure_failures.pop(runtime_basename, None)

        # 1. Parse verifier output (Part 6 VerifierAdapter).
        outcome = self.runtime.verifier_adapter.parse(event.result)

        # 2. Extract actually-used skills from the real agent trajectory.
        # Missing or metadata-only trajectories invalidate experience-based
        # evolution and its usage metrics, so fail immediately instead of
        # silently persisting a trajectory-blind ReplayRecord.
        trajectory_path = outcome.trajectory_path
        if (
            trajectory_path is None
            or trajectory_path.name == "manifest.json"
            or not trajectory_path.exists()
        ):
            self.runtime.event_store.record(
                "trajectory_integrity_error",
                {
                    "task_id": task.task_id,
                    "runtime_basename": runtime_basename,
                    "trajectory_path": (
                        str(trajectory_path) if trajectory_path else ""
                    ),
                    "reason": "missing_or_metadata_trajectory",
                },
            )
            raise RuntimeError(
                "SkillEvolBench trajectory integrity failure for "
                f"{runtime_basename}: expected an agent trajectory, got "
                f"{trajectory_path!s}"
            )

        raw_skill_slugs = self.runtime.trajectory_extractor.extract_skills_used(
            trajectory_path,
        )
        skills_used = self.runtime.trajectory_extractor.extract_skills_used(
            trajectory_path,
            library=self.runtime.library,
        )
        if raw_skill_slugs and not skills_used:
            self.runtime.event_store.record(
                "trajectory_integrity_error",
                {
                    "task_id": task.task_id,
                    "runtime_basename": runtime_basename,
                    "trajectory_path": str(trajectory_path),
                    "reason": "opened_skills_not_in_manifest",
                    "raw_skill_slugs": raw_skill_slugs,
                },
            )
            raise RuntimeError(
                "SkillEvolBench trajectory integrity failure for "
                f"{runtime_basename}: agent opened {raw_skill_slugs!r}, but "
                "none map to the active library manifest"
            )

        # 3. Compact trajectory in TWO modes for replay store:
        #   (a) RICH -- preserves reasoning + full obs, fed into SkillAuthor's
        #       cumulative-trace input for revision/induction.
        #   (b) ROUGH -- drops reasoning, brief obs, smaller budget. Fed into
        #       TrajectoryRetriever (raw_trajectory_rag baseline). Keeping
        #       these separate ensures the "raw episodic experience" baseline
        #       doesn't silently inherit abstraction-friendly preprocessing.
        compacted = self.runtime.compactor.compact(
            outcome.trajectory_path, outcome=outcome,
        )
        compacted_rough = self.runtime.compactor_rough.compact(
            outcome.trajectory_path, outcome=outcome,
        )

        # 4. Persist ReplayRecord (Part 5 ReplayStore).
        from skillevolbench.schemas.replay import ReplayRecord  # lazy: stub-safe

        # Choose record_task_id so original/shadow/replay records don't
        # alias under the same key. Shadow records use runtime_basename
        # (suffixed with __oracle_shadow); replay records similarly use
        # the __replay-suffixed runtime_basename.
        if is_shadow or is_replay or ccv_side or learning_pass > 1:
            record_task_id = runtime_basename
        else:
            record_task_id = task.task_id
        # Mode tag distinguishes record types at metric-aggregation time.
        if is_shadow:
            mode = "shadow_oracle"
        elif is_replay:
            mode = "within_env_replay"
        elif ccv_side:
            mode = f"ccv_{ccv_side}"
        elif learning_pass > 1:
            mode = f"learning_pass_{learning_pass}"
        else:
            mode = "primary"
        record = ReplayRecord(
            task_id=record_task_id,
            family_id=task.family_id,
            env_id=task.environment_id,
            task_role=task.role,
            timestamp=datetime.now(timezone.utc),
            library_hash_pre=pre_state["library_hash"],
            library_hash_post=self.runtime.library.compute_hash(),
            outcome=outcome,
            retrieval=pre_state["retrieval"],
            skills_actually_used=skills_used,
            trajectory_compact=compacted.to_dict(),
            trajectory_compact_rough=compacted_rough.to_dict(),
            replay_mode=mode,
        )
        self.runtime.replay_store.persist(record)

        if ccv_side:
            self._finish_ccv_trial(task, outcome, ccv_side)
            self.runtime.event_store.record("ccv_trial_ended", {
                "task_id": task.task_id, "runtime": runtime_basename,
                "side": ccv_side, "reward": outcome.reward,
                "normalized_score": outcome.normalized_score,
            })
            if ccv_side in {"candidate", "candidate_attack"}:
                self._freeze_if_learning_complete(task.environment_id)
            return

        # Shadow trials short-circuit: no eval-buffer push (would alias
        # the primary's metrics), no strategy.decide, no maintenance.
        if is_shadow:
            self._assert_freeze_invariant(task)
            self.runtime.event_store.record(
                "shadow_trial_ended",
                {
                    "task_id": task.task_id,
                    "shadow_runtime": runtime_basename,
                    "verifier_passed": outcome.verifier_passed,
                    "reward": outcome.reward,
                },
            )
            return

        # Replay trials short-circuit similarly: persist for metric
        # computation (replay_pass_rate vs original_pass_rate -> evolution
        # lift) but skip strategy.decide entirely. The library is frozen
        # at the env's post-evolution state, and replays are designed to
        # observe it without mutating it. ``_assert_freeze_invariant``
        # confirms the library actually stayed frozen across the replay.
        if is_replay:
            self._assert_freeze_invariant(task)
            self.runtime.event_store.record(
                "replay_trial_ended",
                {
                    "task_id": task.task_id,
                    "replay_runtime": runtime_basename,
                    "env_id": task.environment_id,
                    "verifier_passed": outcome.verifier_passed,
                    "reward": outcome.reward,
                },
            )
            return

        # 5. Eval block: skip strategy entirely + assert freeze invariant.
        if task.role in {r.value for r in _EVAL_ROLES}:
            self._eval_outcomes_buffer.append(record)
            self._assert_freeze_invariant(task)
            self.runtime.event_store.record(
                "trial_ended_eval",
                {
                    "task_id": task.task_id,
                    "verifier_passed": outcome.verifier_passed,
                    "reward": outcome.reward,
                },
            )
            return

        # 6. Learning block: build context + dispatch to strategy.
        from skillevolbench.strategies.base import (  # lazy: stub-safe at runtime
            ApplyPatch, NoOp, EvolutionContext,
        )

        ctx = EvolutionContext(
            task=task,
            outcome=outcome,
            compacted=compacted,
            pre_retrieval=pre_state["retrieval"],
            skills_actually_used=skills_used,
            baseline=self.runtime.baseline,
            replay_store=self.runtime.replay_store,
            library=self.runtime.library,
            learning_pass=learning_pass,
            record_task_id=record_task_id,
            execution_dropout_report=pre_state.get(
                "execution_dropout_report"
            ),
            execution_skill_text_overrides=pre_state.get(
                "execution_skill_text_overrides"
            ),
        )
        decision = self.runtime.strategy.decide(ctx)

        # 7. Apply / NoOp -- all writes go through freeze_ctrl.
        # (Rollback decision was an RGPE-only path; removed.)
        if isinstance(decision, ApplyPatch):
            cfg = self.runtime.strategy_config
            proposal_regularization = bool(getattr(
                cfg, "regularization_proposal_regularization", False
            )) and learning_pass >= int(getattr(
                cfg, "regularization_start_learning_pass", 1
            ))
            old_commit = (
                self.runtime.library.head_commit()
                if proposal_regularization
                and hasattr(self.runtime.library, "head_commit")
                else ""
            )
            if proposal_regularization:
                from skillevolbench.components.online_regularization import (
                    measure_family_complexity,
                )
                old_library_hash = self.runtime.library.compute_hash()
                old_complexity = measure_family_complexity(
                    self.runtime.library, task.family_id
                )
                old_skill = self._family_skill_text(task.family_id)
            else:
                old_library_hash = ""
                old_complexity = {}
                old_skill = ""
            apply_result = self.runtime.freeze_ctrl.submit_patch(
                patch=decision.patch,
                strategy_name=self.runtime.strategy.name,
                current_task=(
                    runtime_basename if learning_pass > 1 else task.task_id
                ),
            )
            shared_t1_anchor = decision.regularization_metadata.get("shared_t1_anchor")
            if shared_t1_anchor and apply_result is not None:
                self.runtime.event_store.record(
                    "shared_t1_anchor_applied",
                    {
                        **shared_t1_anchor,
                        "library_commit": apply_result.commit_hash,
                        "library_hash_after": self.runtime.library.compute_hash(),
                    },
                )
            if proposal_regularization and apply_result is not None:
                candidate_complexity = measure_family_complexity(
                    self.runtime.library, task.family_id
                )
                role = task.role.value if hasattr(task.role, "value") else str(task.role)
                common = {
                    "patch_id": decision.patch.patch_id,
                    "old_commit": old_commit,
                    "old_complexity": old_complexity,
                    "candidate_complexity": candidate_complexity,
                    "lambda_complex": float(cfg.regularization_complexity_lambda),
                    "free_growth_budget": float(cfg.regularization_free_growth),
                    "dropout": decision.regularization_metadata.get("dropout", {}),
                    "dropout_context_redactions": int(
                        decision.regularization_metadata.get(
                            "dropout_context_redactions", 0
                        ) or 0
                    ),
                    "candidate_commit": apply_result.commit_hash,
                    "old_library_hash": old_library_hash,
                    "old_skill": old_skill,
                    "candidate_skill": self._family_skill_text(task.family_id),
                    "trajectory_summary": compacted.text,
                    "role": role,
                    "learning_pass": learning_pass,
                    "source_record_task_id": record_task_id,
                    "source_training_passed": bool(
                        outcome.verifier_passed
                    ),
                    "source_training_score": float(
                        outcome.normalized_score
                    ),
                    "source_affected_skill_ids": list(
                        apply_result.affected_skill_ids
                    ),
                    "source_delete_paths": list(decision.patch.delete_paths),
                    "complexity_projection": decision.regularization_metadata.get(
                        "complexity_projection", {}
                    ),
                }
                # The feasible patch is final immediately, matching the
                # paper's always-apply chain. Complexity constrains proposal
                # formation; task performance never selects or rejects it.
                projection = common["complexity_projection"]
                self.runtime.event_store.record(
                    "proposal_regularization_applied",
                    {
                        "task_id": task.task_id,
                        "patch_id": decision.patch.patch_id,
                        "dropout": common["dropout"],
                        "dropout_context_redactions": common[
                            "dropout_context_redactions"
                        ],
                        "old_complexity": old_complexity,
                        "candidate_complexity": candidate_complexity,
                        "delta_complexity": (
                            float(candidate_complexity.get("total", 0.0))
                            - float(old_complexity.get("total", 0.0))
                        ),
                        "complexity_projection": common[
                            "complexity_projection"
                        ],
                        "complexity_anchor": (
                            projection.get("anchor", {})
                            if isinstance(projection, dict) else {}
                        ),
                        "learning_pass": learning_pass,
                        "affected_skill_ids": list(
                            apply_result.affected_skill_ids
                        ),
                        "acceptance_semantics": "always_apply",
                    },
                )
                step_local_pipeline = bool(getattr(
                    cfg, "regularization_step_local_pipeline", False
                ))
                if step_local_pipeline:
                    # Candidate formation precedes CCV: consolidate a
                    # Dropout-produced source delta and/or an over-budget
                    # complexity delta immediately, then make that repaired
                    # commit the candidate side of all four CCV slots.
                    self._consume_regularization_signals(
                        task, common, ccv_decision=None
                    )
                    common.update({
                        "candidate_commit": self.runtime.library.head_commit(),
                        "candidate_library_hash": (
                            self.runtime.library.compute_hash()
                        ),
                        "candidate_skill": self._family_skill_text(
                            task.family_id
                        ),
                        "candidate_complexity": measure_family_complexity(
                            self.runtime.library, task.family_id
                        ),
                    })
                if self._ccv_scheduled_for_task(task, learning_pass):
                    ccv_changed_skill_ids = list(
                        apply_result.affected_skill_ids
                    )
                    if (
                        getattr(cfg, "regularization_ccv_schedule", "per_step")
                        == "family_end"
                    ):
                        accumulated: list[str] = []
                        for row in self.runtime.event_store.events_of_type(
                            "proposal_regularization_applied"
                        ):
                            if not str(row.get("task_id", "")).startswith(
                                f"{task.family_id}-"
                            ):
                                continue
                            if int(row.get("learning_pass", 0) or 0) != int(
                                learning_pass
                            ):
                                continue
                            accumulated.extend(
                                row.get("affected_skill_ids") or ()
                            )
                        ccv_changed_skill_ids = list(dict.fromkeys(
                            accumulated or ccv_changed_skill_ids
                        ))
                        anchor = self._late_regularization_family_anchors.get(
                            task.family_id
                        )
                        if anchor:
                            common.update({
                                "old_commit": anchor["commit"],
                                "old_library_hash": anchor["library_hash"],
                                "old_skill": anchor["skill"],
                                "old_complexity": anchor["complexity"],
                                "late_family_end": True,
                            })
                    common["ccv_candidate_changed_skill_ids"] = (
                        ccv_changed_skill_ids
                    )
                    old_hash = str(common.get("old_library_hash", ""))
                    candidate_hash = str(
                        common.get("candidate_library_hash")
                        or self.runtime.library.compute_hash()
                    )
                    per_step_reuse = bool(
                        step_local_pipeline
                        and getattr(cfg, "regularization_ccv_schedule", "per_step")
                        == "per_step"
                    )
                    gate = (
                        _step_local_ccv_gate(
                            old_library_hash=old_hash,
                            candidate_library_hash=candidate_hash,
                            training_passed=bool(
                                common.get("source_training_passed")
                            ),
                            training_score=float(
                                common.get("source_training_score", 0.0)
                            ),
                        )
                        if per_step_reuse else {
                            "active": True,
                            "old_clean_reused": False,
                            "probe": {},
                            "skip_reason": "",
                        }
                    )
                    self._pending_ccv_candidates[task.task_id] = {
                        **common,
                        "candidate_library_hash": candidate_hash,
                        **gate,
                        "feedback_only": True,
                    }
                    if gate.get("skip_reason"):
                        self.runtime.event_store.record("ccv_skipped", {
                            "task_id": task.task_id,
                            "status": gate["probe"]["status"],
                            "reason": gate["skip_reason"],
                            "old_library_hash": old_hash,
                            "candidate_library_hash": candidate_hash,
                        })
                    elif gate.get("old_clean_reused"):
                        self.runtime.event_store.record(
                            "ccv_old_clean_reused", {
                                "task_id": task.task_id,
                                "source": "pre_update_training_rollout",
                                "verifier_passed": True,
                                "score": gate["old_clean_score"],
                                "old_library_hash": old_hash,
                            },
                        )
                elif not step_local_pipeline:
                    self._consume_regularization_signals(
                        task, common, ccv_decision=None
                    )
            elif proposal_regularization and self._ccv_scheduled_for_task(
                task, learning_pass
            ):
                self._pending_ccv_candidates[task.task_id] = {
                    "active": False,
                    "feedback_only": True,
                    "probe": {"status": "skipped_no_applied_patch"},
                }
        else:  # NoOp
            self.runtime.event_store.record(
                "noop_decision",
                {"task_id": task.task_id, "reason": decision.reason},
            )
            if self._ccv_scheduled_for_task(task, learning_pass):
                self._pending_ccv_candidates[task.task_id] = {
                    "active": False,
                    "feedback_only": True,
                    "probe": {"status": "skipped_no_patch"},
                }

        self.runtime.event_store.record(
            "trial_ended_learning",
            {
                "task_id": task.task_id,
                "learning_pass": learning_pass,
                "verifier_passed": outcome.verifier_passed,
                "decision_type": type(decision).__name__,
                "library_hash_after": self.runtime.library.compute_hash(),
            },
        )

        # 8. Freeze BEFORE the first T4's container is constructed.
        # Harbor builds the trial environment in ``Trial.__init__``
        # (harbor/trial/trial.py), which runs BEFORE TrialEvent.START
        # fires. ``GlobalLibraryEnvironment.__init__`` reads the
        # ``.frozen`` marker once at that moment to decide whether the 5
        # skill mounts should be ``read_only=True``. If we wait until
        # T4's START hook to write the marker, T4's mounts are already
        # baked as read-write.
        #
        # The schedule (scheduler.py) is::
        #
        #     normal:     LS1[T1,T2,T3], ..., LS5[T1,T2,T3]
        #     shared T1:  LS1[T2,T3],    ..., LS5[T2,T3]
        #     env eval block:      LS1[T4,T5,T6], ..., LS5[T4,T5,T6]
        #
        # So we must NOT freeze at every T3 ended (LS1-T3 ended is
        # followed by LS2-T1 which still needs to inject_curated). The
        # right moment is "the last LEARNING task of this env just
        # finished". We count completed learning trials per env and
        # freeze when the count reaches the configured number of executed
        # learning roles per family (3 normally, 2 with shared T1).
        #
        # Idempotent with the T4-START fallback in ``on_trial_started``
        # (covers orphan-trial-ended edge cases).
        if task.role in {r.value for r in _LEARNING_ROLES}:
            env_id = task.environment_id
            self._learning_completed_per_env[env_id] = (
                self._learning_completed_per_env.get(env_id, 0) + 1
            )
            n_families_in_env = len(self.task_registry.families_in_env(env_id))
            first_pass_roles = (
                2 if bool(getattr(
                    self.runtime.strategy_config,
                    "shared_t1_skip_execution",
                    False,
                )) else 3
            )
            learning_passes = int(getattr(
                self.runtime.run_config, "learning_passes", 1
            ))
            first_pass_count = 0 if getattr(
                self.runtime.run_config, "shared_learning_anchor_source", None
            ) is not None else first_pass_roles
            roles_per_family = first_pass_count + 3 * (learning_passes - 1)
            target = n_families_in_env * roles_per_family
            delay_for_diagnostic = self._ccv_scheduled_for_task(
                task, learning_pass
            )
            if (
                self._learning_completed_per_env[env_id] >= target
                and not self.runtime.freeze_ctrl.frozen
                and not delay_for_diagnostic
            ):
                self.runtime.freeze_ctrl.freeze(env_id)

    def _finish_ccv_trial(self, task: Any, outcome: Any, side: str) -> None:
        pending = self._pending_ccv_candidates.get(task.task_id)
        if not pending:
            return
        if side == "old_clean":
            if pending.get("old_clean_reused"):
                # The inactive placeholder must not overwrite the real,
                # verifier-passed training result cached above.
                return
            pending["old_clean_score"] = float(outcome.normalized_score)
            pending["old_clean_passed"] = bool(outcome.verifier_passed)
            return
        if side == "candidate_clean":
            pending["candidate_clean_score"] = float(outcome.normalized_score)
            pending["candidate_clean_passed"] = bool(outcome.verifier_passed)
            return
        if side in {"old", "old_attack"}:
            pending["old_attack_score"] = float(outcome.normalized_score)
            pending["old_attack_passed"] = bool(outcome.verifier_passed)
            return
        self._pending_ccv_candidates.pop(task.task_id, None)
        if not pending.get("active"):
            probe_status = (pending.get("probe") or {}).get("status", "skipped")
            decision_event = {
                "task_id": task.task_id, "status": probe_status,
                "probe": pending.get("probe", {}), "confirmed_regression": False,
                "valid_pair": False,
            }
            self.runtime.event_store.record("ccv_decision", decision_event)
            consume_pending = pending
            if bool(getattr(
                self.runtime.strategy_config,
                "regularization_step_local_pipeline",
                False,
            )):
                consume_pending = {
                    **pending, "complexity_projection": {}, "dropout": {},
                }
            self._consume_regularization_signals(task, consume_pending, decision_event)
            self._finalize_step_local_candidate(task, pending)
            return
        old_score = float(pending.get("old_attack_score", 0.0))
        candidate_score = float(outcome.normalized_score)
        min_old_score = float(getattr(
            self.runtime.strategy_config, "regularization_ccv_min_old_score", 0.80
        ))
        min_regression_drop = float(getattr(
            self.runtime.strategy_config,
            "regularization_ccv_min_regression_drop", 0.05,
        ))
        score_drop = old_score - candidate_score
        step_local_pipeline = bool(getattr(
            self.runtime.strategy_config, "regularization_step_local_pipeline", False
        ))
        clean_pair = bool(
            pending.get("old_clean_passed", not step_local_pipeline)
            and pending.get("candidate_clean_passed", not step_local_pipeline)
        )
        valid_pair = clean_pair and old_score >= min_old_score
        regression = valid_pair and score_drop >= min_regression_drop
        status = (
            "complete" if valid_pair
            else "inconclusive_clean_failed" if not clean_pair
            else "inconclusive_old_failed"
        )
        decision_event = {
            "task_id": task.task_id, "status": status,
            "probe": pending.get("probe", {}),
            "old_score": old_score, "candidate_score": candidate_score,
            "score_drop": score_drop,
            "min_old_score": min_old_score,
            "min_regression_drop": min_regression_drop,
            "old_passed": bool(pending.get("old_attack_passed")),
            "candidate_passed": bool(outcome.verifier_passed),
            "old_clean_score": pending.get("old_clean_score"),
            "candidate_clean_score": pending.get("candidate_clean_score"),
            "old_clean_passed": pending.get("old_clean_passed"),
            "candidate_clean_passed": pending.get("candidate_clean_passed"),
            "clean_pair_qualified": clean_pair,
            "confirmed_regression": regression,
            "allow_unmatched_source_rewrite": bool(
                regression and step_local_pipeline
            ),
            "valid_pair": valid_pair,
            "feedback_only": True,
            "candidate_retained": True,
            "library_hash_after": self.runtime.library.compute_hash(),
        }
        self.runtime.event_store.record("ccv_decision", decision_event)

        consume_pending = pending
        if bool(getattr(
            self.runtime.strategy_config, "regularization_step_local_pipeline", False
        )):
            consume_pending = {
                **pending, "complexity_projection": {}, "dropout": {},
            }
        self._consume_regularization_signals(task, consume_pending, decision_event)
        self._finalize_step_local_candidate(task, pending)

    def _finalize_step_local_candidate(
        self, task: Any, pending: dict[str, Any],
    ) -> None:
        """Squash raw/repair staging commits into one active revision."""
        if not bool(getattr(
            self.runtime.strategy_config, "regularization_step_local_pipeline", False
        )):
            return
        from skillevolbench.schemas import OperationType, SkillPatch
        from skillevolbench.stores.library_store import skill_id_to_slug

        old_commit = str(pending.get("old_commit", ""))
        if not old_commit and not pending.get("patch_id"):
            self.runtime.event_store.record("step_local_candidate_finalized", {
                "task_id": task.task_id,
                "status": "skipped_no_staged_candidate",
            })
            return
        if not old_commit or not self.runtime.library.has_commit(old_commit):
            raise RuntimeError(
                "step-local finalization lost the exact pre-task commit"
            )
        affected = list(dict.fromkeys(
            pending.get("source_affected_skill_ids") or ()
        ))
        upserts: dict[str, str] = {}
        for skill_id in affected:
            try:
                skill = self.runtime.library.get_skill(skill_id)
            except Exception:
                continue
            slug = skill_id_to_slug(skill_id)
            for path, content in skill.files.items():
                upserts[f"{slug}/{path}"] = content
        delete_paths = list(pending.get("source_delete_paths") or ())
        self.runtime.library.rollback_to(
            old_commit, reason="step_local_squash_to_pre_task"
        )
        if not upserts and not delete_paths:
            self.runtime.event_store.record("step_local_candidate_finalized", {
                "task_id": task.task_id,
                "status": "no_material_candidate",
                "old_commit": old_commit,
            })
            return
        patch = SkillPatch(
            patch_id=str(uuid.uuid4()),
            summary="step-local final candidate",
            upsert_files=upserts,
            delete_paths=delete_paths,
            target_skill_ids=affected,
            operation_type=OperationType.REVISE,
            triggered_by_task=task.task_id,
            proposing_mode="step_local_final_candidate",
            attempt_count=1,
        )
        strategy_name = f"{self.runtime.strategy.name}:step_local_final"
        self.runtime.event_store.record_patch_proposed(patch, strategy_name)
        result = self.runtime.freeze_ctrl.submit_patch(
            patch=patch,
            strategy_name=strategy_name,
            current_task=task.task_id,
        )
        if result is None:
            raise RuntimeError("step-local final candidate could not be committed")
        self.runtime.event_store.record("step_local_candidate_finalized", {
            "task_id": task.task_id,
            "status": "committed",
            "old_commit": old_commit,
            "staged_candidate_commit": pending.get("candidate_commit", ""),
            "final_commit": result.commit_hash,
            "affected_skill_ids": affected,
        })

    def _consume_regularization_signals(
        self, task: Any, pending: dict[str, Any],
        ccv_decision: dict[str, Any] | None,
    ) -> None:
        """Bundle actionable module findings and route them exactly once."""
        from skillevolbench.components.regularization_consumption import (
            build_regularization_signal_bundle,
        )

        bundle = build_regularization_signal_bundle(
            task_id=task.task_id,
            complexity_projection=pending.get("complexity_projection") or {},
            ccv_decision=ccv_decision or {},
            dropout_report=(
                pending.get("dropout") or {}
                if bool(getattr(
                    self.runtime.strategy_config,
                    "regularization_step_local_pipeline",
                    False,
                ))
                else {}
            ),
        )
        consumption_mode = str(getattr(
            self.runtime.strategy_config,
            "regularization_consumption_mode",
            "deferred",
        ))
        self.runtime.event_store.record("regularization_signal_bundle", {
            **bundle.to_dict(),
            "actionable": bundle.actionable,
            "consumption_mode": consumption_mode,
            "consumed": False,
        })
        if bundle.actionable and consumption_mode == "local_edit":
            self._apply_regularization_local_edit(task, pending, bundle)

    def _apply_regularization_local_edit(
        self, task: Any, pending: dict[str, Any], bundle: Any,
    ) -> None:
        """Consume one Regularization bundle by an immediate corrective patch."""
        from skillevolbench.components.online_regularization import (
            clean_patch_placeholders,
            complexity_within_budget,
            measure_patch_complexity,
            patch_skill_text_overrides,
            local_edit_complexity_telemetry,
            local_edit_trust_region_errors,
        )
        from skillevolbench.components.local_repair import (
            DEFAULT_MAX_EXISTING_MATCHES_PER_SOURCE,
            apply_replacements,
            regularization_repair_regions,
            is_similarity_qualified,
            new_exact_duplicate_atom_errors,
            protected_content_hash,
            validate_repaired_document,
        )
        from skillevolbench.components.regularization_consumption import (
            regularization_update_instruction,
        )
        from skillevolbench.components.skill_author import (
            PatchGenerationFailure,
            local_repair_json_retry_instruction,
        )
        from skillevolbench.components.semantic_retrieval import (
            DEFAULT_EMBEDDING_MATCH_THRESHOLD,
            DEFAULT_EMBEDDING_MODEL,
        )
        from skillevolbench.stores.library_store import skill_id_to_slug

        family_id = task.family_id
        entries = self.runtime.library.skills_in_family(family_id)
        target_skill_ids = [entry.skill_id for entry in entries]
        repair_scope_skill_ids: list[str] = []
        if "dropout" in bundle.abnormal_modules:
            repair_scope_skill_ids.extend(
                pending.get("source_affected_skill_ids") or ()
            )
        if "complexity" in bundle.abnormal_modules:
            repair_scope_skill_ids.extend(
                pending.get("source_affected_skill_ids") or ()
            )
        if "ccv" in bundle.abnormal_modules:
            probe = bundle.ccv.get("probe") or {}
            repair_scope_skill_ids.extend(
                probe.get("responsible_skill_ids")
                or pending.get("ccv_candidate_changed_skill_ids")
                or ()
            )
        valid_targets = set(target_skill_ids)
        source_changed = set(pending.get("source_affected_skill_ids") or ())
        source_changed.update(
            pending.get("ccv_candidate_changed_skill_ids") or ()
        )
        repair_scope_skill_ids = [
            skill_id for skill_id in dict.fromkeys(repair_scope_skill_ids)
            if skill_id in valid_targets and skill_id in source_changed
        ]
        # The capacity budget is a regularization target, not an acceptance
        # gate. Record whether one repair reached it, but apply every
        # structurally valid material repair without a second evaluation.
        budget = (
            (pending.get("complexity_projection") or {}).get("budget") or None
        )
        base_event = {
            "task_id": task.task_id,
            "source_patch_id": pending.get("patch_id", ""),
            "consumption_mode": "local_edit",
            "abnormal_modules": list(bundle.abnormal_modules),
            "repair_scope_skill_ids": repair_scope_skill_ids,
            "second_evaluation": False,
        }
        if not target_skill_ids:
            self.runtime.event_store.record("regularization_local_edit_failed", {
                **base_event, "reason": "no_family_skill",
            })
            return
        if not repair_scope_skill_ids:
            self.runtime.event_store.record("regularization_local_edit_failed", {
                **base_event, "reason": "no_causally_implicated_skill",
            })
            return

        current_texts: dict[str, str] = {}
        for entry in entries:
            skill = self.runtime.library.get_skill(entry.skill_id)
            for path, content in skill.files.items():
                if path.endswith("SKILL.md"):
                    current_texts[entry.skill_id] = content
        old_commit = str(pending.get("old_commit", ""))
        if not self.runtime.library.has_commit(old_commit):
            self.runtime.event_store.record("regularization_local_edit_failed", {
                **base_event, "reason": "missing_or_invalid_source_anchor_commit",
            })
            return
        regions_by_skill: dict[str, list[Any]] = {}
        anchor_texts: dict[str, str] = {}
        embedding_model = getattr(
            self.runtime.strategy_config,
            "regularization_embedding_model",
            DEFAULT_EMBEDDING_MODEL,
        )
        embedding_match_threshold = float(getattr(
            self.runtime.strategy_config,
            "regularization_embedding_match_threshold",
            DEFAULT_EMBEDDING_MATCH_THRESHOLD,
        ))
        for skill_id in repair_scope_skill_ids:
            anchor_text = self.runtime.library.get_skill_text_at_commit(
                skill_id, old_commit,
            )
            # ``None`` is valid for a skill created by the source patch: its
            # pre-update body is empty.  The immutable YAML frontmatter in the
            # candidate is still excluded by regularization_repair_regions().
            anchor_texts[skill_id] = anchor_text or ""
            regions = regularization_repair_regions(
                anchor_texts[skill_id], current_texts.get(skill_id, ""),
                min_similarity=embedding_match_threshold,
                embedding_model=embedding_model,
            )
            if regions:
                regions_by_skill[skill_id] = regions
        base_event["editable_regions"] = {
            skill_id: [
                {
                    "region_id": region.region_id,
                    "start": region.start,
                    "end": region.end,
                    "anchor_chars": len(region.anchor_text),
                    "candidate_chars": len(region.candidate_text),
                    "role": region.role,
                    "source_region_id": region.source_region_id,
                    "similarity": region.similarity,
                    "similarity_metric": region.similarity_metric,
                    "embedding_model": region.embedding_model,
                    "match_threshold": region.match_threshold,
                    "match_scope": region.match_scope,
                }
                for region in regions
            ]
            for skill_id, regions in regions_by_skill.items()
        }
        base_event["protected_content_hashes"] = {
            skill_id: protected_content_hash(
                current_texts[skill_id], regions_by_skill[skill_id]
            )
            for skill_id in regions_by_skill
        }
        base_event["retrieval"] = {
            "method": "dense_embedding",
            "model": embedding_model,
            "metric": "cosine",
            "threshold": embedding_match_threshold,
            "max_matches_per_source": (
                DEFAULT_MAX_EXISTING_MATCHES_PER_SOURCE
            ),
            "query_instruction": None,
            "order": ["same_batch_grouping", "history_retrieval"],
        }
        if not regions_by_skill:
            self.runtime.event_store.record("regularization_local_edit_failed", {
                **base_event, "reason": "no_source_delta_semantic_region",
            })
            return
        instruction = regularization_update_instruction(
            bundle,
            repair_scope_skill_ids=repair_scope_skill_ids,
        )
        validator_attempts: list[dict[str, Any]] = []
        patch = None
        repaired_texts: dict[str, str] = {}
        measured: dict[str, Any] = {}
        complexity_direction: dict[str, Any] = {}
        removed_placeholders = 0
        source_candidate_measurement = (
            (pending.get("complexity_projection") or {}).get(
                "initial_candidate"
            ) or None
        )
        has_model_visible_matches = any(
            is_similarity_qualified(region)
            for regions in regions_by_skill.values()
            for region in regions
        )
        noop_fallback_operations: list[dict[str, Any]] | None = None
        noop_soft_retry_triggered = False

        def accept_noop(
            delta_operations: list[dict[str, Any]], *,
            reason: str,
            fallback_used: bool,
        ) -> None:
            noop_complexity_direction = local_edit_complexity_telemetry(
                abnormal_modules=bundle.abnormal_modules,
                measured=source_candidate_measurement or {},
                source_candidate_measurement=source_candidate_measurement,
            )
            self.runtime.event_store.record("regularization_local_edit_noop", {
                **base_event,
                "delta_operations": delta_operations,
                "candidate_preserved": True,
                "complexity_direction": noop_complexity_direction,
                "consumed": True,
                "reason": reason,
                "noop_soft_retry_triggered": noop_soft_retry_triggered,
                "noop_fallback_used": fallback_used,
                "validator_attempts": validator_attempts,
            })
            self.runtime.event_store.record("regularization_feedback_consumed", {
                "family_id": family_id,
                "source_task_id": task.task_id,
                "consumed_by": "immediate_local_edit_noop",
                "repair_patch_id": "",
            })

        for attempt_index in range(3):
            try:
                generated = self.runtime.strategy.evolver.propose_local_repair(
                    task_id=task.task_id,
                    regularization_instruction=instruction,
                    family_skill_texts=current_texts,
                    editable_regions_by_skill=regions_by_skill,
                    abnormal_modules=tuple(bundle.abnormal_modules),
                )
            except PatchGenerationFailure as exc:
                validator_attempts.append({
                    "attempt": attempt_index + 1,
                    "errors": [str(exc)[:500]],
                })
                if noop_fallback_operations is not None:
                    accept_noop(
                        noop_fallback_operations,
                        reason="soft_retry_generation_failed",
                        fallback_used=True,
                    )
                    return
                instruction += local_repair_json_retry_instruction()
                continue

            delta_operations = generated.get("operations") or []
            all_noop = bool(delta_operations) and all(
                item.get("operation") == "NOOP"
                for item in delta_operations
                if isinstance(item, dict)
            ) and all(isinstance(item, dict) for item in delta_operations)
            if all_noop:
                validator_attempts.append({
                    "attempt": attempt_index + 1,
                    "errors": [],
                    "decision": "all_noop",
                })
                if (
                    "complexity" in bundle.abnormal_modules
                    and has_model_visible_matches
                    and noop_fallback_operations is None
                    and attempt_index < 2
                ):
                    noop_fallback_operations = [
                        dict(item) for item in delta_operations
                        if isinstance(item, dict)
                    ]
                    noop_soft_retry_triggered = True
                    instruction += (
                        "\nSOFT SUBSUMPTION RECHECK: The previous decision was "
                        "all NOOP. Reconsider every threshold-qualified source "
                        "r*/linked m* pair independently: decide whether either "
                        "rule semantically subsumes the other. MERGE only when one "
                        "coherent general rule can replace both; use REWRITE or "
                        "DELETE only when the Regularization evidence directly warrants it. "
                        "NOOP remains fully valid. Do not force any change."
                    )
                    continue
                accept_noop(
                    [dict(item) for item in delta_operations],
                    reason=(
                        "soft_retry_confirmed_noop"
                        if noop_fallback_operations is not None
                        else "all_noop"
                    ),
                    fallback_used=False,
                )
                return

            upserts: dict[str, str] = {}
            local_errors: list[str] = []
            generated_replacements = generated.get("replacements") or {}
            for skill_id, replacements in generated_replacements.items():
                try:
                    repaired = apply_replacements(
                        current_texts[skill_id],
                        regions_by_skill[skill_id],
                        replacements,
                    )
                    local_errors.extend(
                        f"{skill_id}: {error}"
                        for error in validate_repaired_document(repaired)
                    )
                    local_errors.extend(
                        f"{skill_id}: {error}"
                        for error in new_exact_duplicate_atom_errors(
                            current_texts[skill_id], repaired,
                        )
                    )
                    if repaired != current_texts[skill_id]:
                        upserts[
                            f"{skill_id_to_slug(skill_id)}/SKILL.md"
                        ] = repaired
                except Exception as exc:  # noqa: BLE001
                    local_errors.append(
                        f"{skill_id}: {type(exc).__name__}: {exc}"
                    )
            if not upserts:
                local_errors.append("local repair made no material skill change")
            proposed = SkillPatch(
                patch_id=str(uuid.uuid4()),
                summary=str(
                    "; ".join(generated.get("change_summary") or ())
                    or "SkillEvoReg atom-local local edit"
                )[:200],
                upsert_files=upserts,
                delete_paths=[],
                target_skill_ids=list(upserts and generated_replacements or ()),
                operation_type=OperationType.REVISE,
                triggered_by_task=task.task_id,
                proposing_mode="regularization_atom_local_repair",
                attempt_count=attempt_index + 1,
            )
            proposed, removed = clean_patch_placeholders(proposed)
            proposed_measured = measure_patch_complexity(
                self.runtime.library, family_id, proposed
            )
            proposed_texts = patch_skill_text_overrides(
                self.runtime.library, family_id, proposed
            )
            errors = local_errors + local_edit_trust_region_errors(
                abnormal_modules=bundle.abnormal_modules,
                measured=proposed_measured,
                anchor=None,
                current_texts=current_texts,
                repaired_texts=proposed_texts,
                repair_scope_skill_ids=list(generated_replacements),
                source_candidate_measurement=source_candidate_measurement,
            )
            attempt_complexity_direction = local_edit_complexity_telemetry(
                abnormal_modules=bundle.abnormal_modules,
                measured=proposed_measured,
                source_candidate_measurement=source_candidate_measurement,
            )
            validator_attempts.append({
                "attempt": attempt_index + 1,
                "errors": errors,
                "complexity": proposed_measured,
                "complexity_direction": attempt_complexity_direction,
            })
            if not errors:
                patch = proposed
                repaired_texts = proposed_texts
                measured = proposed_measured
                complexity_direction = attempt_complexity_direction
                removed_placeholders = removed
                break
            if noop_fallback_operations is not None:
                accept_noop(
                    noop_fallback_operations,
                    reason="soft_retry_structural_validation_failed",
                    fallback_used=True,
                )
                return
            instruction += (
                "\nSTRUCTURAL VALIDATOR RETRY: " + " ".join(errors)
                + local_repair_json_retry_instruction()
            )

        if patch is None:
            if noop_fallback_operations is not None:
                accept_noop(
                    noop_fallback_operations,
                    reason="soft_retry_exhausted",
                    fallback_used=True,
                )
                return
            self.runtime.event_store.record("regularization_local_edit_failed", {
                **base_event,
                "reason": "structural_validator_rejected",
                "validator_attempts": validator_attempts,
                "candidate_preserved": True,
            })
            return

        budget_exceeded_after_update = bool(
            budget and not complexity_within_budget(measured, budget)
        )
        if repaired_texts == current_texts:
            self.runtime.event_store.record("regularization_local_edit_failed", {
                **base_event, "reason": "no_material_skill_change",
            })
            return

        strategy_name = f"{self.runtime.strategy.name}:regularization_local_edit"
        self.runtime.event_store.record_patch_proposed(patch, strategy_name)
        apply_result = self.runtime.freeze_ctrl.submit_patch(
            patch=patch,
            strategy_name=strategy_name,
            current_task=task.task_id,
        )
        if apply_result is None:
            self.runtime.event_store.record("regularization_local_edit_failed", {
                **base_event, "reason": "library_frozen",
            })
            return

        self.runtime.event_store.record("regularization_local_edit_applied", {
            **base_event,
            "delta_operations": generated.get("operations", []),
            "repair_patch_id": patch.patch_id,
            "repair_commit": apply_result.commit_hash,
            "removed_placeholders": removed_placeholders,
            "complexity": measured,
            "complexity_direction": complexity_direction,
            "complexity_budget": budget or {},
            "budget_exceeded_after_update": budget_exceeded_after_update,
            "validator_attempts": validator_attempts,
            "consumed": True,
        })
        self.runtime.event_store.record("regularization_feedback_consumed", {
            "family_id": family_id,
            "source_task_id": task.task_id,
            "consumed_by": "immediate_local_edit",
            "repair_patch_id": patch.patch_id,
        })

    def _ccv_scheduled_for_task(
        self, task: Any, learning_pass: int = 1,
    ) -> bool:
        cfg = getattr(self.runtime, "strategy_config", None)
        raw_role = getattr(task, "role", "")
        role = raw_role.value if hasattr(raw_role, "value") else str(raw_role)
        if learning_pass < int(getattr(cfg, "regularization_start_learning_pass", 1)):
            return False
        if (
            getattr(cfg, "regularization_ccv_schedule", "per_step") == "family_end"
            and role != TaskRole.VARIANT.value
        ):
            return False
        return bool(getattr(cfg, "regularization_ccv_enabled", False)) and role in set(
            getattr(cfg, "regularization_ccv_roles", [])
        )

    def _qualify_ccv_source(
        self, source_task: Any, old_library_hash: str,
        *, record_task_id: str | None = None,
    ) -> dict:
        records = self.runtime.replay_store.tasks_by_family(source_task.family_id)
        matches = [
            r for r in records
            if r.task_id == (record_task_id or source_task.task_id)
            and r.library_hash_pre == old_library_hash
            and r.replay_mode == "primary"
        ]
        if not matches:
            return {
                "qualified": False, "source_task_id": source_task.task_id,
                "old_library_hash": old_library_hash,
                "reason": "no_primary_record_for_exact_old_library",
            }
        record = matches[-1]
        return {
            "qualified": True, "source_task_id": source_task.task_id,
            "qualification_record": record.task_id,
            "old_library_hash": old_library_hash,
            "source_score": float(record.outcome.normalized_score),
            "source_passed": bool(record.outcome.verifier_passed),
        }

    def _family_skill_text(self, family_id: str) -> str:
        blocks = []
        for entry in self.runtime.library.skills_in_family(family_id):
            skill = self.runtime.library.get_skill(entry.skill_id)
            for path, content in sorted(skill.files.items()):
                blocks.append(f"## {skill.skill_id}/{path}\n{content}")
        return "\n\n".join(blocks)

    def _freeze_if_learning_complete(self, env_id: str) -> None:
        n_families = len(self.task_registry.families_in_env(env_id))
        first_pass_roles = (
            2 if bool(getattr(
                self.runtime.strategy_config,
                "shared_t1_skip_execution",
                False,
            )) else 3
        )
        learning_passes = int(getattr(
            self.runtime.run_config, "learning_passes", 1
        ))
        first_pass_count = 0 if getattr(
            self.runtime.run_config, "shared_learning_anchor_source", None
        ) is not None else first_pass_roles
        roles_per_family = first_pass_count + 3 * (learning_passes - 1)
        target = n_families * roles_per_family
        if (
            self._learning_completed_per_env.get(env_id, 0) >= target
            and not self.runtime.freeze_ctrl.frozen
        ):
            self.runtime.freeze_ctrl.freeze(env_id)

    def _ensure_shared_t1_state(self, task: Any) -> None:
        """Install one verified shared T1 skill + replay before T2."""
        from skillevolbench.components.online_regularization import (
            measure_family_complexity,
        )

        store = getattr(self.runtime.strategy, "shared_t1_anchors", None)
        if store is None:
            raise RuntimeError(
                "shared_t1_skip_execution is enabled but no anchor store exists"
            )
        t1_task_id = f"{task.family_id}-T1"
        anchor = store.anchor_for(
            family_id=task.family_id,
            task_id=t1_task_id,
            latent_skill_id=task.latent_skill_id,
        )
        if anchor.replay_record is None:
            raise RuntimeError(
                f"shared T1 anchor {task.family_id} has no replay state"
            )

        library = self.runtime.library
        had_seed = library.has_seed_for(task.family_id)
        if had_seed:
            installed = library.get_skill(anchor.patch.target_skill_ids[0])
            slug = anchor.patch.target_skill_ids[0].split(".", 1)[1]
            expected = {
                path.removeprefix(f"{slug}/"): content
                for path, content in anchor.patch.upsert_files.items()
                if path.startswith(f"{slug}/")
            }
            if installed.files != expected:
                raise RuntimeError(
                    f"existing T1 state for {task.family_id} does not match "
                    "the configured shared anchor"
                )
            apply_result = None
        else:
            old_hash = library.compute_hash()
            old_complexity = measure_family_complexity(library, task.family_id)
            self.runtime.event_store.record_patch_proposed(
                anchor.patch, "shared_t1_anchor"
            )
            apply_result = self.runtime.freeze_ctrl.submit_patch(
                patch=anchor.patch,
                strategy_name="shared_t1_anchor",
                current_task=t1_task_id,
            )
            if apply_result is None:
                raise RuntimeError(
                    f"failed to apply shared T1 anchor for {task.family_id}"
                )
            candidate_complexity = measure_family_complexity(
                library, task.family_id
            )
            self.runtime.event_store.record(
                "proposal_regularization_applied",
                {
                    "task_id": t1_task_id,
                    "patch_id": anchor.patch.patch_id,
                    "dropout": {},
                    "old_complexity": old_complexity,
                    "candidate_complexity": candidate_complexity,
                    "delta_complexity": (
                        float(candidate_complexity.get("total", 0.0))
                        - float(old_complexity.get("total", 0.0))
                    ),
                    "complexity_projection": {},
                    "acceptance_semantics": "shared_t1_initialization",
                },
            )
            self.runtime.event_store.record(
                "shared_t1_anchor_applied",
                {
                    **anchor.audit,
                    "library_hash_before": old_hash,
                    "library_commit": apply_result.commit_hash,
                    "library_hash_after": library.compute_hash(),
                },
            )

        replay = self.runtime.replay_store
        if replay.has(t1_task_id):
            existing_record = replay.get(t1_task_id)
            if (
                existing_record.model_dump(mode="json")
                != anchor.replay_record.model_dump(mode="json")
            ):
                raise RuntimeError(
                    f"existing replay state for {task.family_id} does not "
                    "match the configured shared T1 anchor"
                )
        else:
            replay.persist(anchor.replay_record)

        self.runtime.event_store.record(
            "shared_t1_state_ready",
            {
                **anchor.audit,
                "loaded_before_task": task.task_id,
                "library_hash": library.compute_hash(),
                "replay_record_mode": anchor.replay_record.replay_mode,
                "resumed_existing_skill": had_seed,
            },
        )

    # ==================================================================
    # Env transition handler
    # ==================================================================

    async def _handle_env_transition(self, prev_env: str, new_env: str) -> None:
        """Called after the last task of ``prev_env`` -- runs unfreeze +
        post-eval maintenance + tags a snapshot. Optionally clears the
        active library when ``baseline.library_scope == "environment"``.
        """
        self.runtime.freeze_ctrl.unfreeze_and_maintain(
            env_id=prev_env,
            baseline=self.runtime.baseline,
            eval_records=self._eval_outcomes_buffer,
        )
        self._eval_outcomes_buffer.clear()

        # Tag a per-env snapshot before clearing the active library.
        # IMPORTANT: this MUST happen BEFORE clear_active below -- once the
        # working tree is wiped, the snapshot tag would point at the post-
        # wipe (empty) state instead of the env's final state.
        if self.runtime.snapshot_store is not None:
            self.runtime.snapshot_store.tag(f"after-{prev_env}")

        # Per-env physical isolation (library_scope=="environment"):
        # each env has its own LibraryStore at ``library/<env_id>/`` with
        # its own git repo + manifest. No "reset" / "wipe" is needed --
        # the next env's first task calls ``switch_env(new_env)`` which
        # lazy-inits ``library/<new_env>/`` (empty by construction).
        # ``runtime.library`` is then swapped to that LibraryStore and
        # inject_curated / induce_skill / zero_shot_create see the new
        # env's empty library and re-seed naturally. We just record the
        # event for audit.
        scope = getattr(self.runtime.baseline, "library_scope", "global")
        if scope == "environment":
            self.runtime.event_store.record(
                "library_swap_on_env_transition",
                {
                    "from_env": prev_env,
                    "to_env": new_env,
                    "from_env_library_path": str(
                        getattr(self.runtime, "library_root",
                                self.runtime.run_root / "library") / prev_env
                    ),
                },
            )

        self.runtime.event_store.record(
            "env_transition",
            {
                "from_env": prev_env,
                "to_env": new_env,
                "library_hash": self.runtime.library.compute_hash(),
            },
        )

    # ==================================================================
    # Internal helpers (also unit-test seams)
    # ==================================================================

    def _resolve_task(self, event: "TrialHookEvent") -> Any:
        """Recover the ``TaskSpec`` for the trial that fired this event.

        Harbor 0.6+ sets ``TrialHookEvent.task_name`` to ``TaskConfig.name``
        (auto-derived from the basename of ``TaskConfig.path``). We arrange
        the runtime layout as ``runtime/<task_id>/`` (no extra subdir), so
        the basename IS ``task_id`` and we look it up directly.

        Dual-T6 shadow trials use ``runtime/<task_id>__oracle_shadow/``;
        we strip that suffix before registry lookup so the same TaskSpec
        is returned for both primary and shadow.

        Defensive fallback: if a fixture (or older Harbor version) passes
        a path-shaped value containing ``runtime/<task_id>/...``, extract
        the ``<task_id>`` segment.
        """
        raw = str(getattr(event, "task_name", "") or "")
        # Path-shaped fallback (legacy / fixture inputs).
        if "/" in raw or "\\" in raw:
            parts = Path(raw).parts
            # Old layout: .../runtime/<task_id>/harbor-task-copy
            if "harbor-task-copy" in parts:
                idx = parts.index("harbor-task-copy")
                if idx > 0:
                    raw = parts[idx - 1]
            # New layout: .../runtime/<task_id>
            elif "runtime" in parts:
                idx = parts.index("runtime")
                if idx + 1 < len(parts):
                    raw = parts[idx + 1]
            else:
                raw = Path(raw).name
        if not raw:
            raise KeyError(
                f"Could not resolve task_id from TrialHookEvent.task_name="
                f"{getattr(event, 'task_name', None)!r}"
            )
        # Strip dual-T6 shadow / within-env replay suffixes before
        # registry lookup -- both share a TaskSpec with their original.
        canonical = raw
        if canonical.endswith(_SHADOW_T6_SUFFIX):
            canonical = canonical[: -len(_SHADOW_T6_SUFFIX)]
        if canonical.endswith(_REPLAY_SUFFIX):
            canonical = canonical[: -len(_REPLAY_SUFFIX)]
        for suffix in (
            _CCV_OLD_CLEAN_SUFFIX,
            _CCV_CANDIDATE_CLEAN_SUFFIX,
            _CCV_OLD_ATTACK_SUFFIX,
            _CCV_CANDIDATE_ATTACK_SUFFIX,
            _CCV_OLD_SUFFIX,
            _CCV_CANDIDATE_SUFFIX,
        ):
            if canonical.endswith(suffix):
                canonical = canonical[: -len(suffix)]
                break
        if canonical.endswith(_LEARNING_PASS2_SUFFIX):
            canonical = canonical[: -len(_LEARNING_PASS2_SUFFIX)]
        return self.task_registry.task(canonical).spec

    def _runtime_basename(self, event: "TrialHookEvent") -> str:
        """Return the runtime-dir basename for this trial (with any
        ``__oracle_shadow`` suffix preserved). Used to detect dual-T6
        shadow trials."""
        raw = str(getattr(event, "task_name", "") or "")
        if "/" in raw or "\\" in raw:
            parts = Path(raw).parts
            if "harbor-task-copy" in parts:
                idx = parts.index("harbor-task-copy")
                if idx > 0:
                    raw = parts[idx - 1]
            elif "runtime" in parts:
                idx = parts.index("runtime")
                if idx + 1 < len(parts):
                    raw = parts[idx + 1]
            else:
                raw = Path(raw).name
        return raw

    def _read_instruction(self, task: Any) -> str:
        """Read the original (non-injected) instruction text from disk.

        We pass the *original* text to the retriever so the retrieval query
        is independent of any baseline-specific injection. The injected
        version goes into the *runtime* copy that the agent reads.
        """
        path = Path(self.runtime.run_root) / ".." / ".." / ".." / (
            f"benchmark/tasks/{task.task_slug}/{task.harbor.instruction}"
        )
        # Falling back: ask the registry record for the resolved folder.
        record = self.task_registry.task(task.task_id)
        instr_path = record.folder / record.spec.harbor.instruction
        return instr_path.read_text()

    def _inject_curated_v0(self, task: Any) -> None:
        """Path-B: copy the curated SKILL.md into the live library."""
        family = self.task_registry.family(task.family_id)
        curated_path = family.folder / family.meta.curated_skill_path
        if not curated_path.exists():
            raise FileNotFoundError(
                f"Path-B baseline {self.runtime.baseline.name!r} expects curated "
                f"SKILL.md at {curated_path} but it is missing"
            )
        self.runtime.library.inject_curated(
            skill_md_path=curated_path,
            family_id=task.family_id,
            latent_skill_id=task.latent_skill_id,
            created_from_task=task.task_id,
        )
        self.runtime.event_store.record(
            "curated_seeded",
            {"family_id": task.family_id, "task_id": task.task_id},
        )

    async def _create_zero_shot_skill(self, task: Any) -> None:
        """Self-Gen-Zero-Shot: ask the evolver to write a skill from family
        label *before* any execution trace exists."""
        family_meta = self.task_registry.family(task.family_id).meta
        content = await self.runtime.evolver.zero_shot_create(
            family_id=task.family_id,
            latent_skill_id=task.latent_skill_id,
            family_meta=family_meta,
        )
        self.runtime.library.create_skill(
            skill_id=task.latent_skill_id,
            content=content,
            source="zero_shot",
            family_id=task.family_id,
            created_from_task=task.task_id,
        )
        self.runtime.event_store.record(
            "zero_shot_created",
            {
                "family_id": task.family_id,
                "task_id": task.task_id,
                "latent_skill_id": task.latent_skill_id,
            },
        )

    def _assert_freeze_invariant(self, task: Any) -> None:
        """Hard assert: library hash never changes during T4-T6.

        Engineering Design §0.2 invariant 4. If this ever raises in
        production, treat the run's data as invalid -- something modified
        the library inside the eval block.
        """
        if self.runtime.freeze_ctrl.frozen:
            current = self.runtime.library.compute_hash()
            expected = self.runtime.freeze_ctrl.frozen_hash
            if current != expected:
                raise AssertionError(
                    f"Library hash changed during eval block at task "
                    f"{task.task_id!r}: got {current!r} expected {expected!r}"
                )


__all__ = ["SkillEvolBenchHooks"]
