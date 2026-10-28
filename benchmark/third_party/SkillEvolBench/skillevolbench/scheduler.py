"""Global 180-task order scheduler (Part 9).

Produces the canonical global execution order::

    for env in env_order(seed):                    # 6 envs
        # Learning block (family-major within env)
        for family in families_in_env(env):        # 5 families
            for role in [canonical, enriched, variant]:
                yield task                          # 3 tasks
        # Evaluation block (family-major within env)
        for family in families_in_env(env):
            for role in [context-shift, adversarial, composition]:
                yield task                          # 3 tasks
        # OPTIONAL within-env replay block (when ``within_env_replay``):
        for family in families_in_env(env):
            for role in all 6 roles:
                yield task as REPLAY (TaskRecord.is_replay=True)

= 180 tasks total per run (270 with the default learning-only replay).
When ``skip_original_t1=True``, the 30 original T1 executions are replaced by
preloaded shared state, giving 150 tasks (240 with learning-only replay).
Deterministic given ``(registry, env_orders, seed, within_env_replay)``.

The "learning block runs before eval block" structure is the
score-before-maintain rule's enabling assumption -- T6 composition tasks
(eval) need *all* same-env families' skills already in the library, which
means T1-T3 of every same-env family must have run first.

Replay block (when enabled) runs LAST within each env: library is at its
post-evolution state, frozen since T4 began; replays observe that state
and never mutate it. Replays cover all 6 roles (T1-T6) so we can compare
T1 replay (against the evolved library) to T1 original (against the
empty / v0 library) -- the evolution-lift signal.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator, Optional

from skillevolbench.discovery import TaskRegistry, TaskRecord
from skillevolbench.schemas import EnvOrders, OrderSeed, TaskRole


# Roles in the order they execute within a family during the learning block.
_LEARNING_ROLES: tuple[str, ...] = (
    TaskRole.CANONICAL.value,
    TaskRole.ENRICHED.value,
    TaskRole.VARIANT.value,
)
# Eval-block roles in fixed order. T4/T5/T6 share the freeze; the order
# inside the block is determined by family then role (T4-T5-T6 of fam1, then
# fam2, etc.) -- this is what the Engineering Design §2.4 mermaid shows.
_EVAL_ROLES: tuple[str, ...] = (
    TaskRole.CONTEXT_SHIFT.value,
    TaskRole.ADVERSARIAL.value,
    TaskRole.COMPOSITION.value,
)


def compute_task_order(
    registry: TaskRegistry,
    env_orders: EnvOrders,
    order_seed: OrderSeed,
    within_env_replay: bool = False,
    replay_eval: bool = False,
    skip_original_t1: bool = False,
    learning_passes: int = 1,
    skip_first_learning_pass: bool = False,
) -> list[TaskRecord]:
    """Build the ordered list of TaskRecords for a run.

    Parameters
    ----------
    registry
        Loaded :class:`TaskRegistry` (Part 1).
    env_orders
        Loaded :class:`EnvOrders` from ``configs/env_orders.yaml``.
    order_seed
        ``"A" | "B" | "C"``.
    within_env_replay
        When True, after each env's 30 tasks the same 30 are emitted again
        as replay records (``is_replay=True``). Total = 360 instead of 180.

    Returns
    -------
    A list of TaskRecords in canonical execution order. Shared-T1 runs omit
    only original T1 records; final-library T1 replays remain scheduled.
    """
    env_sequence = env_orders.for_seed(order_seed)
    return list(_iter_tasks(
        registry,
        env_sequence,
        within_env_replay,
        replay_eval,
        skip_original_t1,
        learning_passes,
        skip_first_learning_pass,
    ))


def _iter_tasks(
    registry: TaskRegistry,
    env_sequence: list[str],
    within_env_replay: bool = False,
    replay_eval: bool = False,
    skip_original_t1: bool = False,
    learning_passes: int = 1,
    skip_first_learning_pass: bool = False,
) -> Iterator[TaskRecord]:
    for env_id in env_sequence:
        families = registry.families_in_env(env_id)
        env_originals: list[TaskRecord] = []
        # One or two complete learning passes. A complete family-major pass is
        # finished before the next starts, so pass 2 sees a stable pass-1
        # library for every family.
        for learning_pass in range(1, learning_passes + 1):
            if learning_pass == 1 and skip_first_learning_pass:
                continue
            for family in families:
                for role in _LEARNING_ROLES:
                    rec = replace(
                        _resolve_task(registry, family.meta.family_id, role),
                        learning_pass=learning_pass,
                    )
                    if learning_pass == 1 or (
                        learning_pass == 2 and skip_first_learning_pass
                    ):
                        env_originals.append(rec)
                    if (
                        skip_original_t1
                        and learning_pass == 1
                        and role == TaskRole.CANONICAL.value
                    ):
                        continue
                    yield rec
        # eval block (T4, T5, T6 of every family in this env)
        for family in families:
            for role in _EVAL_ROLES:
                rec = _resolve_task(registry, family.meta.family_id, role)
                env_originals.append(rec)
                yield rec
        # OPTIONAL replay block. By default replays only the 15 LEARNING
        # trials (T1-T3) -- those test "evolved library helps on training
        # tasks". Eval trials (T4-T6) ran under the same frozen library
        # state, so replaying them would just measure LLM variance (and
        # cost 2x replay budget). Set ``replay_eval=True`` for ablations
        # that explicitly want eval-trial variance + maintenance studies.
        if within_env_replay:
            for original in env_originals:
                if not replay_eval and original.spec.role.value in _EVAL_ROLES:
                    continue
                yield replace(original, is_replay=True)


def _resolve_task(
    registry: TaskRegistry,
    family_id: str,
    role: str,
) -> TaskRecord:
    """Pick the unique TaskRecord for ``(family_id, role)``."""
    for r in registry.tasks_in_family(family_id):
        if r.spec.role.value == role:
            return r
    raise KeyError(f"No task found for family {family_id!r} role {role!r}")


# ---------------------------------------------------------------------------
# Sanity checks (used by LifelongRunner preflight)
# ---------------------------------------------------------------------------


def assert_order_invariants(
    records: list[TaskRecord], *, skip_original_t1: bool = False,
    learning_passes: int = 1,
    skip_first_learning_pass: bool = False,
) -> None:
    """Re-check schedule invariants on a computed order.

    Engineering Design §2.4 -- the learning block of an env precedes its
    eval block. We verify by walking the list and checking that within each
    env, all learning-role records come before all eval-role records.

    Total length depends on within_env_replay + replay_eval flags:
      - replay off:                       180 (30/env x 6 envs)
      - replay on, replay_eval=False:     270 (45/env -- 30 orig + 15 learning replays)
      - replay on, replay_eval=True:      360 (60/env -- 30 orig + 30 replays)
    With ``skip_original_t1=True``, subtract 30 originals from each total:
    150 / 240 / 330 respectively.
    Original tasks must precede their replays within each env.
    """
    # Defensive ``getattr``: legacy fixtures and ad-hoc test stubs may
    # pass record-shaped objects without the ``is_replay`` field. Default
    # to False so existing 180-task invariant checks still work for them.
    replays = [r for r in records if getattr(r, "is_replay", False)]
    has_replay = bool(replays)
    # Replay set is either learning-only (15/env) or full (30/env).
    replays_include_eval = any(
        r.spec.role.value in _EVAL_ROLES for r in replays
    )
    skipped_per_env = 5 if skip_original_t1 else 0
    executed_learning_passes = learning_passes - int(skip_first_learning_pass)
    extra_learning_per_env = 15 * (executed_learning_passes - 1)
    if not has_replay:
        expected_total = 180 - (skipped_per_env * 6) + extra_learning_per_env * 6
        expected_per_env = 30 - skipped_per_env + extra_learning_per_env
        expected_replays_per_env = 0
    elif replays_include_eval:
        expected_total = 360 - (skipped_per_env * 6) + extra_learning_per_env * 6
        expected_per_env = 60 - skipped_per_env + extra_learning_per_env
        expected_replays_per_env = 30
    else:
        # Learning-only replay (the recommended/cheaper default).
        expected_total = 270 - (skipped_per_env * 6) + extra_learning_per_env * 6
        expected_per_env = 45 - skipped_per_env + extra_learning_per_env
        expected_replays_per_env = 15
    if len(records) != expected_total:
        raise AssertionError(
            f"Expected {expected_total} tasks, got {len(records)}"
        )

    # Group by env in *iteration* order; verify learning-then-eval-then-replay.
    seen_envs: list[str] = []
    env_groups: dict[str, list[TaskRecord]] = {}
    for r in records:
        env_id = r.spec.environment_id
        if env_id not in env_groups:
            env_groups[env_id] = []
            seen_envs.append(env_id)
        env_groups[env_id].append(r)

    if len(env_groups) != 6:
        raise AssertionError(
            f"Expected 6 environments, got {sorted(env_groups)}"
        )

    for env_id, group in env_groups.items():
        if len(group) != expected_per_env:
            raise AssertionError(
                f"env {env_id!r}: expected {expected_per_env} tasks, got {len(group)}"
            )
        # Within the env, learning roles must all precede eval roles
        # (originals only); replays go LAST.
        originals = [r for r in group if not getattr(r, "is_replay", False)]
        replays = [r for r in group if getattr(r, "is_replay", False)]
        # Replays must follow originals positionally
        n_orig = len(originals)
        for i, r in enumerate(group[:n_orig]):
            if getattr(r, "is_replay", False):
                raise AssertionError(
                    f"env {env_id!r}: original task {r.spec.task_id} "
                    f"out of order -- replay appeared before all originals"
                )
        if has_replay and len(replays) != expected_replays_per_env:
            raise AssertionError(
                f"env {env_id!r}: expected {expected_replays_per_env} replays, "
                f"got {len(replays)}"
            )
        for learning_pass in range(1, learning_passes + 1):
            if learning_pass == 1 and skip_first_learning_pass:
                continue
            pass_rows = [
                r for r in originals
                if r.spec.role.value in _LEARNING_ROLES
                and getattr(r, "learning_pass", 1) == learning_pass
            ]
            expected_learning = 15 - (
                skipped_per_env if learning_pass == 1 else 0
            )
            if len(pass_rows) != expected_learning:
                raise AssertionError(
                    f"env {env_id!r}: pass {learning_pass} expected "
                    f"{expected_learning} learning tasks, got {len(pass_rows)}"
                )
        seen_eval = False
        for rec in originals:
            role = rec.spec.role.value
            if role in _EVAL_ROLES:
                seen_eval = True
            elif role in _LEARNING_ROLES and seen_eval:
                raise AssertionError(
                    f"env {env_id!r}: learning-role {rec.spec.task_id} found "
                    f"after eval-role tasks. Learning block must precede eval."
                )

    # Each env must contain all 5 families × 6 roles in originals
    # (and again in replays if enabled).
    for env_id, group in env_groups.items():
        originals = [r for r in group if not getattr(r, "is_replay", False)]
        seen_orig = {(r.spec.family_id, r.spec.role.value) for r in originals}
        expected_originals = 25 if skip_original_t1 else 30
        if len(seen_orig) != expected_originals:
            raise AssertionError(
                f"env {env_id!r}: expected {expected_originals} unique "
                f"(family, role) originals, "
                f"got {len(seen_orig)}"
            )
        if skip_original_t1:
            leaked_t1 = {
                pair for pair in seen_orig if pair[1] == TaskRole.CANONICAL.value
            }
            if leaked_t1:
                raise AssertionError(
                    f"env {env_id!r}: shared-T1 schedule contains original T1: "
                    f"{sorted(leaked_t1)}"
                )
        if has_replay:
            replays = [r for r in group if getattr(r, "is_replay", False)]
            seen_replay = {(r.spec.family_id, r.spec.role.value) for r in replays}
            if replays_include_eval:
                # Full replay still covers T1-T6. Shared T1 removes only the
                # original execution, not the final-library T1 replay.
                families = {family for family, _role in seen_orig}
                expected_full = {
                    (family, role)
                    for family in families
                    for role in (_LEARNING_ROLES + _EVAL_ROLES)
                }
                if seen_replay != expected_full:
                    raise AssertionError(
                        f"env {env_id!r}: full replay set mismatch; "
                        f"missing={expected_full - seen_replay}, "
                        f"extra={seen_replay - expected_full}"
                    )
            else:
                # Learning-only mirror: replays = exactly the 15 learning roles.
                families = {family for family, _role in seen_orig}
                expected_learning = {
                    (family, role)
                    for family in families
                    for role in _LEARNING_ROLES
                }
                if seen_replay != expected_learning:
                    raise AssertionError(
                        f"env {env_id!r}: learning-only replay set mismatch; "
                        f"missing={expected_learning - seen_replay}, "
                        f"extra={seen_replay - expected_learning}"
                    )


__all__ = [
    "compute_task_order",
    "assert_order_invariants",
]
