"""``ChainEvolution`` -- single-candidate linear revision (Part 7 §7.2).

The default strategy. Three branches:

* T1 + ``allow_self_gen_induction`` and no seed yet -> ``induce_skill`` and
  return :class:`ApplyPatch`.
* T2/T3 + ``_should_revise`` is True -> ``evolver.propose(mode=<chain mode>)``
  -> :class:`ApplyPatch`. Default mode is ``free_form`` (LLM may add
  scripts/references/assets when the trace shows they help). Override via
  ``StrategyConfig.candidate_modes`` in yaml.
* Otherwise -> :class:`NoOp` with a documented reason.

Failures from :class:`SkillAuthor` (already with internal fallback chain)
become :class:`NoOp` with the failure type encoded in the reason -- the
hook records them as ``noop_decision`` events.
"""

from __future__ import annotations

import logging

from skillevolbench.components.skill_author import PatchGenerationFailure
from skillevolbench.components.online_regularization import (
    DropoutRoundTripError,
    clean_patch_placeholders,
    complexity_budget,
    complexity_instruction,
    complexity_within_budget,
    dropout_instruction,
    measure_family_complexity,
    measure_patch_complexity,
    redact_masked_atom_bodies,
    restore_patch_placeholders,
)
from skillevolbench.strategies.base import (
    ApplyPatch,
    EvolutionContext,
    EvolutionDecision,
    EvolutionStrategy,
    NoOp,
)


_LOG = logging.getLogger(__name__)


class ChainEvolution(EvolutionStrategy):
    name: str = "chain"

    def _decide_impl(self, ctx: EvolutionContext) -> EvolutionDecision:
        role = self._role_value(ctx)

        # T1: induce iff Self-Gen baseline AND no seed yet. For curated /
        # zero-shot baselines (which already have a v0 in the library at T1)
        # this short-circuits and we fall through to the revision path so a
        # canonical-task failure can still trigger a revise on v0.
        if role == "canonical" and self._should_induce(ctx):
            return self._induce_or_noop(ctx)

        # T1 (with seed) / T2 / T3: revision on failure
        if not self._should_revise(ctx):
            return NoOp(reason=self._why_not_revising(ctx))

        # The "suggested target" is what the 3-tier priority points to --
        # we still surface this in the prompt as a hint (the retriever's
        # best guess at what to revise).
        suggested_target_skill_ids = self._select_target_skills(ctx)
        if not suggested_target_skill_ids:
            return NoOp(reason="no_target_skills_to_revise")

        # But pass ALL family skills to propose() -- the LLM then sees
        # the whole family's library and can decide to revise the
        # suggested target, revise a different family skill, or create
        # a new sibling. Without this, the LLM was hard-constrained to
        # only the retriever's pick and could not consolidate across
        # sibling skills or fix a wrong target.
        all_family_skill_ids = self._all_family_skill_ids(ctx)
        # Defensive: ensure the suggested target is in the list (it may
        # already be there from skills_in_family; this dedupe + ordering
        # keeps the suggested first for the prompt's "← suggested" marker).
        target_skill_ids = list(dict.fromkeys(
            suggested_target_skill_ids + all_family_skill_ids
        ))

        # Cumulative trace: T_t sees T_1...T_t together so revisions can
        # reflect across the family's full episodic history (not just the
        # most recent trial). Helper returns a single compacted-shaped
        # object whose .text concatenates all prior learning-block
        # trajectories with per-trial headers.
        cumulative = self._build_cumulative_trace(ctx)

        # Chain config guarantees exactly one candidate_mode (validated in
        # StrategyConfig._check_strategy_invariants). Honour it; the
        # in-code fallback is ``free_form`` to match StrategyConfig's
        # field default and stay consistent with the comprehensive +
        # adversarial mindset baked into the system prompt.
        mode = (self.config.candidate_modes or ["free_form"])[0]
        regularization_metadata: dict = {}
        skill_text_overrides = None
        regularization_text = ""
        regularize = bool(getattr(
            self.config, "regularization_proposal_regularization", False
        )) and ctx.learning_pass >= int(getattr(
            self.config, "regularization_start_learning_pass", 1
        ))
        if regularize:
            original_texts = {}
            for skill_id in target_skill_ids:
                try:
                    original_texts[skill_id] = self.library.get_skill(
                        skill_id
                    ).files.get("SKILL.md", "")
                except Exception:
                    continue
            dropout_enabled = bool(getattr(
                self.config, "regularization_dropout_enabled", False
            )) and float(getattr(
                self.config, "regularization_dropout_probability", 0.0
            )) > 0.0
            if dropout_enabled:
                # Fail closed: a formal training-time Dropout proposal must
                # consume the exact mask used by the task agent. Generating a
                # fresh proposal-only mask here would silently revert to the
                # old, weaker semantics.
                if (
                    ctx.execution_dropout_report is None
                    or ctx.execution_skill_text_overrides is None
                ):
                    self.event_store.record("candidate_failed", {
                        "task_id": ctx.task.task_id,
                        "strategy": self.name,
                        "error": "training_dropout_context_missing",
                    })
                    return NoOp(reason="training_dropout_context_missing")
                dropout_report = dict(ctx.execution_dropout_report)
                skill_text_overrides = dict(original_texts)
                skill_text_overrides.update(
                    ctx.execution_skill_text_overrides
                )
            else:
                dropout_report = {
                    "enabled": False,
                    "masked_count": 0,
                    "masked_atoms": [],
                    "scope_skill_ids": [],
                    "mode": "disabled",
                }
                skill_text_overrides = dict(original_texts)
            regularization_metadata["dropout"] = dropout_report
            regularization_text = dropout_instruction(dropout_report["masked_count"])
            redacted_trace, trace_redactions = redact_masked_atom_bodies(
                getattr(cumulative, "text", ""), dropout_report
            )
            if trace_redactions:
                from types import SimpleNamespace
                cumulative = SimpleNamespace(text=redacted_trace)
            regularization_metadata["dropout_context_redactions"] = trace_redactions
            current_complexity = measure_family_complexity(
                self.library, ctx.task.family_id
            )
            regularization_metadata["complexity_before"] = current_complexity
            regularization_text += complexity_instruction(
                current_complexity,
                complexity_lambda=float(getattr(
                    self.config, "regularization_complexity_lambda", 0.0
                )),
                free_growth_budget=float(getattr(
                    self.config, "regularization_free_growth", 0.0
                )),
            )
            deferred_regularization_feedback = self._latest_regularization_feedback(
                ctx.task.family_id
            )
            if deferred_regularization_feedback:
                regularization_text += deferred_regularization_feedback
        patch = None
        roundtrip_error = ""
        for proposal_attempt in range(2):
            retry_instruction = ""
            if proposal_attempt:
                retry_instruction = (
                    "\nDROPOUT MARKER ROUND-TRIP RETRY: the previous proposal "
                    "dropped, duplicated, embedded, or structurally moved a "
                    "temporary mask "
                    "marker. For every existing masked SKILL.md you upsert, "
                    "copy each <MASKED_SKILL_ATOM ...> marker exactly once, "
                    "unchanged, as a standalone atom in its original order. "
                    "Adjacent visible atoms may be edited. If you do not "
                    "revise that skill, omit its upsert entirely.\n"
                )
            try:
                patch = self.evolver.propose(
                    library=self.library,
                    compacted=cumulative,
                    outcome=ctx.outcome,
                    target_skill_ids=target_skill_ids,
                    mode=mode,
                    suggested_target_skill_ids=suggested_target_skill_ids,
                    skill_text_overrides=skill_text_overrides,
                    regularization_instruction=regularization_text + retry_instruction,
                )
                if regularize and dropout_report.get("masked_count", 0):
                    patch, roundtrip = restore_patch_placeholders(
                        patch, dropout_report,
                    )
                    regularization_metadata["dropout_roundtrip"] = roundtrip
                else:
                    patch, removed_placeholders = clean_patch_placeholders(patch)
                    if regularization_metadata:
                        regularization_metadata["removed_placeholders"] = removed_placeholders
                roundtrip_error = ""
                break
            except DropoutRoundTripError as exc:
                roundtrip_error = str(exc)
                self.event_store.record(
                    "dropout_roundtrip_retry",
                    {
                        "task_id": ctx.task.task_id,
                        "strategy": self.name,
                        "attempt": proposal_attempt + 1,
                        "error": roundtrip_error[:500],
                    },
                )
                patch = None
            except PatchGenerationFailure as exc:
                self.event_store.record(
                    "candidate_failed",
                    {
                        "task_id": ctx.task.task_id,
                        "strategy": self.name,
                        "error": str(exc)[:200],
                    },
                )
                return NoOp(reason=f"patch_generation_failed:{type(exc).__name__}")

        if patch is None:
            self.event_store.record(
                "candidate_failed",
                {
                    "task_id": ctx.task.task_id,
                    "strategy": self.name,
                    "error": f"dropout_roundtrip_failed:{roundtrip_error}"[:500],
                },
            )
            return NoOp(reason="dropout_roundtrip_failed")

        if regularize and bool(getattr(
            self.config, "regularization_hard_complexity_enabled", False
        )):
            anchor = self._complexity_anchor(
                ctx.task.family_id,
                fallback=regularization_metadata["complexity_before"],
            )
            budget = complexity_budget(
                anchor,
                max_relative_growth=float(getattr(
                    self.config, "regularization_hard_max_relative_growth", 0.15
                )),
                max_atom_growth=int(getattr(
                    self.config, "regularization_hard_max_atom_growth", 4
                )),
            )
            initial_measurement = measure_patch_complexity(
                self.library, ctx.task.family_id, patch
            )
            projection = {
                "enabled": True,
                "anchor": anchor,
                "budget": budget,
                "initial_candidate": initial_measurement,
                "triggered": not complexity_within_budget(
                    initial_measurement, budget
                ),
                # Consumption is intentionally outside proposal formation.
                # Deferred mode gives this anomaly to the next step; quick
                # mode performs one combined Regularization update after CCV finishes.
                "consumed_during_proposal": False,
            }
            regularization_metadata["complexity_projection"] = projection

        self.event_store.record_patch_proposed(patch, self.name)
        return ApplyPatch(patch=patch, regularization_metadata=regularization_metadata)

    def _complexity_anchor(
        self, family_id: str, *, fallback: dict,
    ) -> dict:
        """Return the immutable post-T1 anchor for this family."""
        rows = self.event_store.events_of_type("proposal_regularization_applied")
        for row in rows:
            if str(row.get("task_id", "")).startswith(f"{family_id}-"):
                anchor = row.get("complexity_anchor") or row.get("old_complexity")
                if isinstance(anchor, dict) and anchor:
                    return anchor
        return fallback

    def _latest_regularization_feedback(self, family_id: str) -> str:
        """Consume the newest deferred Regularization signal bundle exactly once."""
        from skillevolbench.components.regularization_consumption import (
            RegularizationSignalBundle,
            regularization_update_instruction,
        )

        rows = self.event_store.events_of_type("regularization_signal_bundle")
        consumed_sources = {
            str(row.get("source_task_id", ""))
            for row in self.event_store.events_of_type("regularization_feedback_consumed")
        }
        for row in reversed(rows):
            if not row.get("actionable"):
                continue
            if not str(row.get("task_id", "")).startswith(f"{family_id}-"):
                continue
            if str(row.get("task_id", "")) in consumed_sources:
                continue
            bundle = RegularizationSignalBundle(
                task_id=str(row.get("task_id", "")),
                abnormal_modules=tuple(row.get("abnormal_modules") or ()),
                complexity=row.get("complexity") or {},
                ccv=row.get("ccv") or {},
                dropout=row.get("dropout") or {},
            )
            self.event_store.record("regularization_feedback_consumed", {
                "family_id": family_id,
                "source_task_id": row.get("task_id"),
                "consumed_by": "next_step_feedback",
            })
            return regularization_update_instruction(bundle)
        return ""

    # Compatibility seam for older tests/tools that named the CCV-only
    # deferred consumer directly.
    def _latest_ccv_feedback(self, family_id: str) -> str:
        return self._latest_regularization_feedback(family_id)

    # ------------------------------------------------------------------

    def _why_not_revising(self, ctx: EvolutionContext) -> str:
        """Render a human-readable reason for the NoOp event."""
        if not ctx.baseline.allow_revision:
            return "baseline_does_not_allow_revision"
        role = self._role_value(ctx)
        if role not in {"canonical", "enriched", "variant"}:
            return f"role_not_revisable:{role}"
        if role == "canonical" and not ctx.library.has_seed_for(ctx.task.family_id):
            return "canonical_revision_needs_existing_seed"
        trigger = getattr(ctx.baseline, "revision_trigger", "never")
        if trigger == "never":
            return "revision_trigger_never"
        if trigger == "fail_only" and ctx.outcome.verifier_passed:
            return "trigger_fail_only_but_passed"
        return "no_revise_for_unknown_reason"


__all__ = ["ChainEvolution"]
