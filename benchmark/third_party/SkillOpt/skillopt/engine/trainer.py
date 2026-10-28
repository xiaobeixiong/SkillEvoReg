"""ReflACT Trainer — the main training loop.

Orchestrates the 6-stage ReflACT pipeline:
  1. Rollout   — execute episodes with current skill
  2. Reflect   — analyze trajectories, generate patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — rank and select top edits
  5. Update    — apply edits to skill document
  6. Evaluate  — validate candidate skill, accept/reject

The trainer is environment-agnostic; all environment-specific logic is
delegated to an :class:`~skillopt.envs.base.EnvAdapter` instance.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from difflib import unified_diff

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.evaluation.gate import GateResult, evaluate_gate, select_gate_score
from skillopt.gradient.aggregate import merge_patches
from skillopt.regularization import SkillEvoReg
from skillopt.regularization.atoms import AtomicSegmenter
from skillopt.regularization.complexity import measure_complexity, normalized_growth
from skillopt.regularization.dropout import (
    masked_atom_preservation_errors,
    redact_masked_atom_bodies,
)
from skillopt.regularization.local_repair import (
    DEFAULT_MAX_EXISTING_MATCHES_PER_SOURCE,
    apply_replacements,
    regularization_repair_regions,
    is_similarity_qualified,
    new_exact_duplicate_atom_errors,
    parse_replacements,
    protected_content_hash,
    render_regions,
    validate_delta_operations,
    validate_repaired_document,
    validate_skill_document,
)
from skillopt.regularization.ccv import coerce_ccv_result
from skillopt.regularization.semantic_retrieval import (
    DEFAULT_EMBEDDING_MATCH_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
)
from skillopt.optimizer.meta_skill import run_meta_skill
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.lr_autonomous import decide_autonomous_learning_rate
from skillopt.optimizer.rewrite import rewrite_skill_from_suggestions
from skillopt.optimizer.scheduler import build_scheduler
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.optimizer.appendix import (
    append_to_appendix_field,
    extract_appendix_notes as extract_appendix_notes_from_skill,
    inject_empty_appendix_field,
    _strip_all_appendix_fields,
)
from skillopt.optimizer.skill_aware import (
    configure_skill_aware_reflection,
    consolidate_appendix_notes,
    extract_appendix_notes as extract_appendix_notes_from_result,
)
from skillopt.optimizer.slow_update import (
    build_comparison_pairs,
    extract_slow_update_field,
    inject_empty_slow_update_field,
    replace_slow_update_field,
    run_slow_update,
    save_comparison_pairs,
)
from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_label,
    short_item_summary,
)
from skillopt.model import (
    chat_optimizer,
    configure_azure_openai,
    configure_claude_code_exec,
    configure_codex_exec,
    configure_harbor_exec,
    configure_minimax_chat,
    configure_qwen_chat,
    get_token_summary,
    reset_token_tracker,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
    set_optimizer_backend,
    set_optimizer_deployment,
)
from skillopt.utils import compute_score, extract_json, skill_hash


def _mean_repeat_scores(rows: list[dict]) -> tuple[float, float]:
    """Return hard/soft means for independently evaluated repeats."""
    if not rows:
        raise ValueError("repeat score rows must not be empty")
    return (
        sum(float(row["hard"]) for row in rows) / len(rows),
        sum(float(row["soft"]) for row in rows) / len(rows),
    )


# ── Skill-aware reflection: appendix flush ───────────────────────────────────

def _flush_skill_aware_appendix(
    current_skill: str,
    all_raw_patches: list,
    step_rec: dict,
    step_dir: str,
    cfg: dict,
) -> str:
    """Append this step's EXECUTION_LAPSE notes into the protected appendix.

    Returns the (possibly) updated skill. Must be called on BOTH the normal
    update path and the skip branches: a lapse-only step yields no body
    patches by design (analysts return ``edits: []`` carriers), so the skip
    paths would otherwise silently drop every note of the step.
    """
    step_appendix_notes: list[str] = []
    for rp in all_raw_patches:
        if isinstance(rp, dict):
            step_appendix_notes.extend(extract_appendix_notes_from_result(rp))
    if not step_appendix_notes:
        return current_skill

    before_notes = extract_appendix_notes_from_skill(current_skill)
    current_skill = append_to_appendix_field(
        current_skill, step_appendix_notes,
    )
    after_notes = extract_appendix_notes_from_skill(current_skill)
    n_added = len(after_notes) - len(before_notes)
    step_rec["n_execution_lapse_notes"] = len(step_appendix_notes)
    step_rec["n_appendix_notes_added"] = n_added
    step_rec["n_appendix_notes_total"] = len(after_notes)
    with open(os.path.join(step_dir, "appendix_notes.json"), "w") as f:
        json.dump(
            {
                "step_notes": step_appendix_notes,
                "appendix_after": after_notes,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(
        f"    [skill-aware] +{n_added} appendix note(s) "
        f"(total {len(after_notes)}) from {len(step_appendix_notes)} lapse signal(s)"
    )
    # Threshold-gated LLM consolidation (paper Eq.11): when the
    # appendix grows past N notes, compact it with one optimizer
    # call (dedupe / merge / shorten). 0 disables it. Any failure
    # leaves the appendix unchanged.
    consolidate_threshold = int(
        cfg.get("skill_aware_consolidate_threshold", 0) or 0
    )
    if consolidate_threshold > 0 and len(after_notes) > consolidate_threshold:
        compacted = consolidate_appendix_notes(
            after_notes, chat_fn=chat_optimizer,
        )
        if compacted and len(compacted) < len(after_notes):
            current_skill = append_to_appendix_field(
                _strip_all_appendix_fields(current_skill), compacted,
            )
            step_rec["n_appendix_notes_consolidated"] = len(compacted)
            step_rec["n_appendix_notes_total"] = len(compacted)
            print(
                f"    [skill-aware] consolidated appendix "
                f"{len(after_notes)} -> {len(compacted)} notes"
            )
    return current_skill


# ── Patch normalization ───────────────────────────────────────────────────────

def _normalise_patches(
    raw_patches: list[dict | None],
    update_mode: str = "patch",
) -> tuple[list[dict], list[dict]]:
    """Extract inner 'patch' sub-dict, split into failure/success lists.

    Each element is expected to conform to :class:`~skillopt.types.RawPatch`.
    """
    mode = normalize_update_mode(update_mode)
    failure: list[dict] = []
    success: list[dict] = []
    for p in raw_patches:
        if not isinstance(p, dict):
            continue
        inner = p.get("patch", p)
        if not isinstance(inner, dict):
            continue
        items = get_payload_items(inner, mode)
        if not items:
            continue
        support = max(int(p.get("batch_size", 0) or 0), 1)
        for item in items:
            if isinstance(item, dict):
                item.setdefault("source_type", p.get("source_type", "failure"))
                item.setdefault("support_count", support)
        if p.get("source_type", "failure") == "success":
            success.append(inner)
        else:
            failure.append(inner)
    return failure, success


def _normalise_longitudinal_pair_policy(policy: str | None) -> str:
    raw = str(policy or "mixed").strip().lower()
    aliases = {
        "mixed": "mixed",
        "default": "mixed",
        "random": "mixed",
        "all": "mixed",
        "changed": "changed",
        "change": "changed",
        "delta": "changed",
        "10_01": "changed",
        "01_10": "changed",
        "unchanged": "unchanged",
        "stable": "unchanged",
        "same": "unchanged",
        "00_11": "unchanged",
    }
    if raw not in aliases:
        raise ValueError(
            "optimizer.longitudinal_pair_policy must be one of "
            "mixed, changed, unchanged"
        )
    return aliases[raw]


def _normalise_lr_control_mode(mode: str | None) -> str:
    raw = str(mode or "fixed").strip().lower()
    aliases = {
        "fixed": "fixed",
        "manual": "fixed",
        "scheduler": "fixed",
        "scheduled": "fixed",
        "autonomous": "autonomous",
        "auto": "autonomous",
        "optimizer": "autonomous",
        "none": "none",
        "off": "none",
        "no_lr": "none",
    }
    if raw not in aliases:
        raise ValueError("optimizer.lr_control_mode must be one of fixed, autonomous, none")
    return aliases[raw]


def _filter_longitudinal_pairs(pairs: list[dict], policy: str) -> list[dict]:
    if policy == "mixed":
        return pairs
    if policy == "changed":
        keep = {"improved", "regressed"}
    elif policy == "unchanged":
        keep = {"persistent_fail", "stable_success"}
    else:
        raise ValueError(f"Unknown longitudinal pair policy: {policy}")
    return [p for p in pairs if p.get("category") in keep]


def _pair_category_counts(pairs: list[dict]) -> dict[str, int]:
    counts = {
        "improved": 0,
        "regressed": 0,
        "persistent_fail": 0,
        "stable_success": 0,
    }
    for pair in pairs:
        cat = str(pair.get("category", ""))
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _safe_pair_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe[:80] or "item"


def _build_longitudinal_pairs(
    *,
    adapter: EnvAdapter,
    dataloader,
    prev_skill: str,
    curr_skill: str,
    initial_items: list[dict],
    initial_prev_results: list[dict],
    initial_curr_results: list[dict],
    prev_rollout_dir: str,
    curr_rollout_dir: str,
    policy: str,
    target_n: int,
    seed: int,
    out_root: str,
) -> tuple[list[dict], list[dict]]:
    """Build longitudinal pairs, optionally filtering by change category.

    ``mixed`` preserves the legacy behavior exactly. ``changed`` keeps only
    10/01 pairs and attempts to top up to ``target_n`` by scanning the train
    split once. ``unchanged`` keeps only 00/11 pairs and does not top up.
    """
    all_pairs = build_comparison_pairs(
        initial_prev_results,
        initial_curr_results,
        initial_items,
        prev_rollout_dir=prev_rollout_dir,
        curr_rollout_dir=curr_rollout_dir,
    )
    selected_pairs = _filter_longitudinal_pairs(all_pairs, policy)
    if policy != "changed" or len(selected_pairs) >= target_n or dataloader is None:
        return selected_pairs, all_pairs

    train_items = list(getattr(dataloader, "train_items", []) or [])
    if not train_items:
        return selected_pairs, all_pairs

    seen_ids = {str(p.get("id", "")) for p in all_pairs}
    rng = random.Random(seed)
    candidates = list(train_items)
    rng.shuffle(candidates)
    candidates = [item for item in candidates if str(item.get("id", "")) not in seen_ids]

    for idx, item in enumerate(candidates):
        if len(selected_pairs) >= target_n:
            break
        item_id = _safe_pair_id(str(item.get("id", f"item_{idx}")))
        batch = BatchSpec(
            phase="train",
            split="train",
            seed=seed + idx + 1,
            batch_size=1,
            payload=[item],
        )
        env = adapter.build_env_from_batch(batch, out_root=out_root)
        prev_dir = os.path.join(prev_rollout_dir, "topup", item_id)
        curr_dir = os.path.join(curr_rollout_dir, "topup", item_id)
        prev_results = adapter.rollout(env, prev_skill, prev_dir)
        curr_results = adapter.rollout(env, curr_skill, curr_dir)
        pair = build_comparison_pairs(
            prev_results,
            curr_results,
            [item],
            prev_rollout_dir=prev_dir,
            curr_rollout_dir=curr_dir,
        )
        all_pairs.extend(pair)
        selected_pairs.extend(_filter_longitudinal_pairs(pair, policy))

    return selected_pairs[:target_n], all_pairs


# ── History / persistence helpers ─────────────────────────────────────────────

_SECRET_KEYS = {
    "azure_api_key",
    "api_key",
    "openai_api_key",
}


def _redact_value(val: str) -> str:
    if len(val) <= 8:
        return "*" * len(val)
    return f"{val[:4]}...{val[-4:]}"


def _redact_cfg(cfg: dict) -> dict:
    redacted = dict(cfg)
    for key in list(redacted):
        if key.lower() in _SECRET_KEYS and redacted.get(key):
            redacted[key] = _redact_value(str(redacted[key]))
    return redacted

def _load_history(out_root: str) -> list[dict]:
    path = os.path.join(out_root, "history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_history(out_root: str, history: list[dict]) -> None:
    path = os.path.join(out_root, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _save_skill(out_root: str, step: int, content: str) -> None:
    skills_dir = os.path.join(out_root, "skills")
    os.makedirs(skills_dir, exist_ok=True)
    with open(os.path.join(skills_dir, f"skill_v{step:04d}.md"), "w") as f:
        f.write(content)


def _load_skill(out_root: str, step: int) -> str:
    path = os.path.join(out_root, "skills", f"skill_v{step:04d}.md")
    with open(path) as f:
        return f.read()


def _load_meta_skill_content(out_root: str, epoch: int) -> str:
    if epoch <= 0:
        return ""
    path = os.path.join(
        out_root, "meta_skill", f"epoch_{epoch:02d}", "meta_skill_result.json",
    )
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            result = json.load(f)
        return str(result.get("meta_skill_content", "")).strip()
    except Exception:
        return ""


def _load_runtime_state(out_root: str) -> dict | None:
    path = os.path.join(out_root, "runtime_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else None
    except Exception:
        return None


def _save_runtime_state(out_root: str, state: dict) -> None:
    path = os.path.join(out_root, "runtime_state.json")
    with open(path, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_epoch_agent_evidence(out_root: str, first_step: int, last_step: int) -> tuple[list[dict], list[dict]]:
    """Load every persisted Train trajectory and compact step record in an epoch."""
    trajectories: list[dict] = []
    step_records: list[dict] = []
    for step in range(first_step, last_step + 1):
        step_dir = os.path.join(out_root, "steps", f"step_{step:04d}")
        results_path = os.path.join(step_dir, "rollout", "results.jsonl")
        if os.path.exists(results_path):
            with open(results_path, encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        row = dict(row)
                        row["epoch_evidence_id"] = f"step_{step:04d}:trajectory_{index:04d}:{row.get('id', '')}"
                        trajectories.append(row)
        record_path = os.path.join(step_dir, "step_record.json")
        if os.path.exists(record_path):
            try:
                with open(record_path, encoding="utf-8") as handle:
                    record = json.load(handle)
                if isinstance(record, dict):
                    regularization_row = record.get("skillevoreg") or {}
                    decision = regularization_row.get("decision") or {}
                    compact = {
                        "step": record.get("step"),
                        "epoch": record.get("epoch"),
                        "step_in_epoch": record.get("step_in_epoch"),
                        "action": record.get("action"),
                        "candidate_hash": record.get("candidate_hash"),
                        "candidate_skill_len": record.get("candidate_skill_len"),
                        "committed_skill_len": record.get("skill_len"),
                        "regularization": {
                            "action": decision.get("action"),
                            "reasons": decision.get("reasons", []),
                            "complexity": regularization_row.get("complexity", {}),
                            "ccv": regularization_row.get("ccv", {}),
                        },
                        "dropout": record.get("regularization_dropout", {}),
                    }
                    for name in ("ranked_edits.json", "edit_apply_report.json", "trajectory_digest.json"):
                        evidence_path = os.path.join(step_dir, name)
                        if os.path.exists(evidence_path):
                            try:
                                with open(evidence_path, encoding="utf-8") as evidence_handle:
                                    compact[name.removesuffix(".json")] = json.load(evidence_handle)
                            except Exception:
                                pass
                    old_skill_path = os.path.join(out_root, "skills", f"skill_v{step - 1:04d}.md")
                    candidate_path = os.path.join(step_dir, "candidate_skill.md")
                    if os.path.exists(old_skill_path) and os.path.exists(candidate_path):
                        with open(old_skill_path, encoding="utf-8") as handle:
                            old_skill = handle.read()
                        with open(candidate_path, encoding="utf-8") as handle:
                            candidate_skill = handle.read()
                        compact["candidate_delta"] = _skill_delta(old_skill, candidate_skill, limit=12000)
                    step_records.append(compact)
            except Exception:
                pass
    return trajectories, step_records


def _load_epoch_edit_budget(
    out_root: str,
    first_step: int,
    last_step: int,
) -> int | None:
    """Restore the latest persisted edit budget for an epoch-tail resume."""
    for step in range(last_step, first_step - 1, -1):
        record_path = os.path.join(
            out_root, "steps", f"step_{step:04d}", "step_record.json",
        )
        if not os.path.exists(record_path):
            continue
        try:
            with open(record_path, encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            continue
        if not isinstance(record, dict) or "edit_budget" not in record:
            continue
        value = record.get("edit_budget")
        return None if value is None else int(value)
    return None


def _skill_delta(old_skill: str, new_skill: str, *, limit: int = 30000) -> str:
    delta = "\n".join(unified_diff(
        old_skill.splitlines(), new_skill.splitlines(),
        fromfile="epoch_start_skill", tofile="epoch_end_skill", lineterm="", n=3,
    ))
    if len(delta) > limit:
        half = limit // 2
        return delta[:half] + "\n...[delta clipped]...\n" + delta[-half:]
    return delta


def _resolve_train_size(cfg: dict, dataloader) -> int:
    configured = int(cfg.get("train_size", 0) or 0)
    inferred: int | None = None

    if dataloader is not None:
        getter = getattr(dataloader, "get_train_size", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                value = None
            if value is not None:
                inferred = int(value)
        elif hasattr(dataloader, "train_items"):
            try:
                inferred = len(getattr(dataloader, "train_items"))
            except Exception:
                inferred = None

    if inferred is not None and inferred <= 0:
        inferred = None

    if configured > 0 and inferred is not None and configured != inferred:
        raise ValueError(
            f"Configured train_size={configured} does not match loaded train split "
            f"size={inferred}. Fix the config or the dataset split."
        )

    train_size = configured if configured > 0 else inferred
    if train_size is None or train_size <= 0:
        raise ValueError(
            "Unable to determine train_size automatically. "
            "Provide train.train_size in the config for this environment."
        )
    return int(train_size)


def _compute_task_type_buckets(results: list[dict], task_types: list[str]) -> dict[str, dict]:
    """Compute per-task-type success rates."""
    buckets: dict[str, dict] = {}
    for task in task_types + ["overall"]:
        buckets[task] = {"total": 0, "hard": 0, "soft": 0.0}
    for r in results:
        tt = r.get("task_type", "other")
        for key in [tt, "overall"]:
            if key not in buckets:
                buckets[key] = {"total": 0, "hard": 0, "soft": 0.0}
            buckets[key]["total"] += 1
            buckets[key]["hard"] += float(r.get("hard", 0))
            buckets[key]["soft"] += float(r.get("soft", 0.0))
    return buckets


def _format_rejection_buffer(buffer: list[dict]) -> str:
    """**DEPRECATED** — kept for backward compat; use _format_step_buffer."""
    return _format_step_buffer(buffer)


def _extract_failure_patterns(
    rollout_results: list[dict],
    step_dir: str,
) -> list[dict]:
    """Extract compact failure patterns from rollout results.

    Uses analyst ``failure_summary`` from minibatch patches when available,
    otherwise falls back to ``fail_reason`` prefix grouping.
    """
    failures = [r for r in rollout_results if not r.get("hard") or float(r.get("hard", 0)) < 1e-9]
    if not failures:
        return []

    # Group by fail_reason prefix
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in failures:
        reason = r.get("fail_reason", "unknown")
        prefix = reason.split(":")[0].strip() if ":" in reason else reason
        groups[prefix].append(r)

    # Try richer descriptions from analyst patches
    analyst_descs: list[str] = []
    patch_globs = [
        os.path.join(step_dir, "patches", "minibatch_fail_*.json"),
        os.path.join(step_dir, "batch_*", "patches", "minibatch_fail_*.json"),
    ]
    seen_patch_files: set[str] = set()
    for pattern in patch_globs:
        for fname in sorted(glob.glob(pattern)):
            if fname in seen_patch_files:
                continue
            seen_patch_files.add(fname)
            try:
                with open(fname) as f:
                    patch = json.load(f)
                for fs in patch.get("failure_summary", []):
                    ft = fs.get("failure_type", "")
                    sd = fs.get("description", "")
                    analyst_descs.append(f"{ft}: {sd}" if sd else ft)
            except Exception:
                pass

    patterns = []
    desc_iter = iter(analyst_descs)
    for prefix, items in groups.items():
        desc = next(desc_iter, None) or prefix
        patterns.append({
            "pattern": desc,
            "count": len(items),
            "task_ids": [str(r.get("id", "?")) for r in items],
        })
    return patterns


def _format_step_buffer(buffer: list[dict]) -> str:
    """Format the unified step buffer into a single context block.

    Each entry captures what happened at a previous step: failure patterns
    observed during rollout, and — when the step was rejected — the specific
    edits that were tried and the resulting score drop.

    Returns empty string when *buffer* is empty.
    """
    if not buffer:
        return ""

    parts = [
        "Below is a summary of previous steps in this epoch. "
        "Use it to avoid repeating ineffective edits and to prioritise "
        "failure patterns that remain unsolved.\n"
    ]

    for entry in buffer:
        step = entry["step"]
        action = entry["action"]
        n_fail = entry.get("n_fail", 0)
        n_total = entry.get("n_total", "?")

        parts.append(f"### Step {step} — {action.upper()} ({n_fail}/{n_total} failed)")

        # Failure patterns
        for p in entry.get("failure_patterns", []):
            ids = ", ".join(p["task_ids"])
            parts.append(f'  - "{p["pattern"]}" (×{p["count"]}, tasks: {ids})')

        # Rejected edits (only present on reject)
        rejected = entry.get("rejected_edits", [])
        if rejected:
            score_before = entry.get("score_before", "?")
            score_after = entry.get("score_after", "?")
            parts.append(
                f"  Rejected edits (score {score_before} → {score_after}):"
            )
            for i, e in enumerate(rejected, 1):
                if e.get("op") is not None:
                    op = e.get("op", "?")
                    content = e.get("content", "")
                    target = e.get("target", "")
                    if target:
                        parts.append(f'    {i}. [{op}] target="{target}" → "{content}"')
                    else:
                        parts.append(f'    {i}. [{op}] "{content}"')
                else:
                    kind = e.get("type", "?")
                    title = e.get("title", "")
                    instruction = e.get("instruction", "")
                    parts.append(f'    {i}. [{kind}] "{title}" → "{instruction}"')

    return "\n".join(parts)


def _format_regularization_feedback(feedback: dict | None) -> str:
    """Format answer-free Regularization diagnostics for the optimizer."""
    if not feedback:
        return ""
    if feedback.get("author") == "epoch_feedback_agent":
        lead = (
            "Below is Train-only semantic guidance produced after the preceding "
            "epoch. Apply it throughout this epoch as a general update direction. "
            "It contains no validation or test examples."
        )
    else:
        lead = (
            "Below is structured feedback from EvoSkillGuard about the immediately "
            "preceding rejected candidate. Use it to revise the update direction. "
            "Do not infer validation examples or answers from aggregate scores."
        )
    return lead + "\n\n```json\n" + json.dumps(feedback, ensure_ascii=False, indent=2) + "\n```"


def _record_local_edit_complexity(
    candidate_skill: str,
    repaired_skill: str,
    telemetry: dict,
) -> None:
    """Record repair direction without turning it into an acceptance gate."""
    segmenter = AtomicSegmenter(mode="rule")
    before = measure_complexity(candidate_skill, segmenter).total
    after = measure_complexity(repaired_skill, segmenter).total
    delta = after - before
    if abs(delta) <= 1e-12:
        direction = "unchanged"
    elif delta < 0:
        direction = "decreased"
    else:
        direction = "increased"
    telemetry.update({
        "candidate_complexity": before,
        "repaired_complexity": after,
        "complexity_delta": delta,
        "complexity_direction": direction,
    })


def _run_regularization_local_edit(
    current_skill: str,
    feedback: dict,
    *,
    anchor_skill: str | None = None,
    out_dir: str,
    reasoning_effort: str | None,
    max_completion_tokens: int,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_match_threshold: float = DEFAULT_EMBEDDING_MATCH_THRESHOLD,
) -> dict:
    """Repair only source-delta semantic blocks, without another evaluation."""
    os.makedirs(out_dir, exist_ok=True)
    feedback_context = _format_regularization_feedback(feedback)
    abnormal_modules = set(feedback.get("abnormal_modules") or ())
    # Backward-compatible inference for older persisted feedback payloads.
    has_complexity_issue = bool(
        "complexity" in abnormal_modules or feedback.get("complexity")
    )
    has_dropout_issue = bool(
        "dropout" in abnormal_modules or feedback.get("dropout")
    )
    ccv = feedback.get("ccv") or {}
    has_confirmed_ccv_regression = bool(
        int(ccv.get("confirmed_regressions", 0) or 0) > 0
        or ccv.get("counterexamples")
    )
    allow_unmatched_ccv_rewrite = bool(
        has_confirmed_ccv_regression
        and ccv.get("allow_unmatched_source_rewrite", False)
    )
    if allow_unmatched_ccv_rewrite:
        abnormal_modules.add("ccv_unmatched_rewrite")
    ccv_abstraction_rule = ""
    if has_confirmed_ccv_regression:
        ccv_abstraction_rule = (
            " Derive exactly one transferable procedural invariant from the "
            "confirmed SCAP counterexample and use it to replace, merge, or "
            "generalize the responsible existing rule. Do not append a new "
            "checklist branch. Do not copy or restate counterexample-specific "
            "field names, file names, schema keys, literal values, concrete "
            "formats, or wording; express the invariant at the capability "
            "level instead."
        )
    module_instructions = []
    if has_dropout_issue:
        module_instructions.append(
            "For the training-time Dropout update, consolidate only the "
            "source delta learned from the partial skill view. Preserve every "
            "novel train-supported capability. Remove or merge compensatory "
            "duplicates and resolve contradictions with retained historical "
            "atoms, but do not force compression merely because Dropout ran."
        )
    if has_complexity_issue:
        module_instructions.append(
            "For the confirmed complexity-regularized rejection, follow the "
            "supplied ordered compression policy: merge duplicates and "
            "semantic overlaps, delete narrow/task-specific/unsupported "
            "rules, replace similar rule clusters with transferable general "
            "rules, and compress repeated explanations and examples. Do not "
            "append rules as a response to complexity."
        )
    if has_confirmed_ccv_regression:
        module_instructions.append(
            "For the confirmed SCAP regression, revise, generalize, or remove "
            "the incidental assumption and restore the exposed transferable "
            "capability; never add a one-off exception for a single example."
        )
    if allow_unmatched_ccv_rewrite:
        module_instructions.append(
            "For this confirmed step-local SCAP regression only, a source "
            "delta with no threshold-qualified historical atom may choose "
            "REWRITE to incorporate the transferable invariant locally; it "
            "may otherwise choose NOOP. MERGE and DELETE still require their "
            "normal evidence."
        )
    if anchor_skill is None:
        result = {
            "status": "error",
            "error": "missing pre-update anchor for atom-local repair",
        }
        with open(os.path.join(out_dir, "local_edit_result.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        return result
    all_regions = regularization_repair_regions(
        anchor_skill,
        current_skill,
        min_similarity=embedding_match_threshold,
        embedding_model=embedding_model,
    )
    source_regions = [
        region for region in all_regions if region.role == "source_delta"
    ]
    matched_source_ids = {
        region.source_region_id
        for region in all_regions
        if is_similarity_qualified(region)
    }
    automatic_noop_sources = [
        region
        for region in source_regions
        if region.region_id not in matched_source_ids
        and not allow_unmatched_ccv_rewrite
    ]
    automatic_noop_operations = [{
        "source_region_id": region.region_id,
        "operation": "NOOP",
        "match_region_id": "",
    } for region in automatic_noop_sources]
    regions = [
        region
        for region in all_regions
        if is_similarity_qualified(region)
        or region.region_id in matched_source_ids
        or (
            region.role == "same_batch_member"
            and region.source_region_id in matched_source_ids
        )
        or (
            allow_unmatched_ccv_rewrite
            and region.role in {"source_delta", "same_batch_member"}
        )
    ]
    # Unmatched source deltas are outside the editable region set, so they are
    # protected byte-for-byte and resolved deterministically without an LLM.
    protected_hash = protected_content_hash(current_skill, regions)

    system = (
        "You are the EvoSkillGuard quick-update repair operator. Treat the "
        "listed source groups as independent update regions. A multi-atom source "
        "diff is first split into atoms; mutually similar same-batch atoms are "
        "then concatenated only for historical retrieval and remain unchanged "
        "unless a history-backed operation is chosen. Only source groups "
        "with a similarity-qualified linked existing atom are listed; every "
        "unlisted source delta is already a deterministic NOOP and is not yours "
        "to modify or decide. Cosine similarity only nominates a pair or a small "
        "cluster of up to three historical atoms for semantic inspection; it is never evidence that the "
        "rules are interchangeable or "
        "must be merged. For every listed r* source delta, choose one operation in the "
        "same response using these semantics: "
        "NOOP is the default when the source adds a distinct or complementary capability, "
        "when both atoms remain independently useful, or whenever semantic preservation "
        "is uncertain; it keeps the source delta byte-for-byte. "
        "REWRITE replaces only the source delta when that atom is overspecific, unclear, "
        "or internally redundant but its unique supported capability should remain; the "
        "rewrite must preserve every supported condition, action, exception, ordering "
        "constraint, boundary convention, and output requirement from the source, and it "
        "must leave all linked m* atoms unchanged. "
        "MERGE is allowed only when one coherent atomic rule can preserve the full union "
        "of the source r* and every linked m* atom. Topic similarity alone is insufficient. "
        "Before choosing MERGE, compare the entire cluster bidirectionally and ensure the merged "
        "rule explicitly retains every distinct supported condition, action, exception, "
        "ordering constraint, boundary convention, and output requirement from every atom; if "
        "any atom contributes a unique behavior that cannot be stated coherently in one "
        "atom, choose NOOP. "
        "DELETE removes only a non-empty source delta and is allowed only when the source "
        "is wholly unsupported, harmful, or fully subsumed by retained content; complexity "
        "pressure or topical similarity alone never justifies deleting a unique capability. "
        "For an already-deleted delta, use NOOP to keep the deletion "
        "or REWRITE to restore/generalize content at that position; MERGE and DELETE "
        "are invalid for an already-deleted delta. "
        "Each m* explicitly names its source r* and records whether it is a "
        "same-batch member or a historical candidate. Same-batch membership alone "
        "never authorizes a repair; it only forms the composite history query. Preserve "
        "train-supported capabilities. "
        + " ".join(module_instructions)
        + ccv_abstraction_rule
        + " The full skill and neighbouring atoms are read-only context. "
        "Do not copy or restate a read-only atom into an editable source delta; "
        "the runtime rejects newly introduced exact duplicate rules. "
        "For REWRITE, MERGE, and DELETE, perform a final semantic-loss check against "
        "the pre-operation source and all linked atoms; if any supported behavior would be "
        "lost, use NOOP instead. Return only complete replacement Markdown blocks for one or more "
        "listed region IDs; never return the complete skill. Each replacement "
        "must be a coherent semantic unit (whole paragraph, list item, or code "
        "block), preserve local transitions and numbering, and must not contain "
        "region markers. A similar existing atom may be used by only its linked delta "
        "and at most once globally. The runtime, not the model, owns every r*/m* "
        "cluster. Operations are authoritative and the runtime canonicalizes "
        "their edits: NOOP preserves the source and all candidates and ignores replacement rows; REWRITE "
        "uses only a non-empty source replacement and leaves all linked atoms unchanged; "
        "DELETE needs no replacement body because the runtime clears only the non-empty "
        "source; MERGE requires a non-empty merged source replacement and the runtime "
        "deterministically clears all of that source's similarity-qualified linked atoms. "
        "Do not return replacement rows for m* atoms. Return one decision for every "
        "listed source delta. The runtime owns the linked match set; use null for "
        "match_region_id with NOOP, REWRITE, or DELETE. "
        "Return JSON exactly as {`operations`:[{`source_region_id`:`r1`,"
        "`operation`:`NOOP|REWRITE|MERGE|DELETE`,`match_region_id`:`m1`}],"
        "`replacements`:[{`region_id`:`r1`,`replacement`:`...`}],"
        "`change_summary`:[`...`]}."
    )
    user = (
        f"## Full Current Skill (read-only context)\n{current_skill}\n\n"
        f"## Editable Source-Delta Regions\n{render_regions(regions)}\n\n"
        f"## Guard Feedback\n{feedback_context}\n\n"
        "Perform the independent local repairs in one response. Include a decision "
        "for every listed r* source delta; omit only replacement regions that the "
        "declared operation does not modify. Do not propose another evaluation or "
        "accept/reject decision."
    )
    with open(os.path.join(out_dir, "local_edit_request.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "system": system,
            "feedback": feedback,
            "retrieval": {
            "method": "dense_embedding",
                "model": embedding_model,
                "metric": "cosine",
                "threshold": embedding_match_threshold,
                "max_matches_per_source": (
                    DEFAULT_MAX_EXISTING_MATCHES_PER_SOURCE
                ),
            "query_instruction": None,
            "order": ["same_batch_grouping", "history_retrieval"],
            },
            "editable_regions": [region.to_dict() for region in regions],
            "source_delta_count": len(source_regions),
            "automatic_noop_source_region_ids": [
                region.region_id for region in automatic_noop_sources
            ],
            "protected_content_hash": protected_hash,
        }, handle, ensure_ascii=False, indent=2)
    result: dict = {"status": "error", "error": "local edit did not run"}
    attempts: list[dict] = []
    if not source_regions:
        result = {
            "status": "error",
            "error": "no source-delta semantic region available for repair",
            "editable_region_count": 0,
            "protected_content_hash": protected_hash,
        }
        with open(os.path.join(out_dir, "local_edit_result.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        return result
    if not regions:
        operations, operation_errors = validate_delta_operations(
            {"operations": automatic_noop_operations},
            all_regions,
            {},
            abnormal_modules=frozenset(abnormal_modules),
        )
        telemetry = {
            "editable_region_count": 0,
            "source_delta_count": len(source_regions),
            "automatic_noop_source_region_ids": [
                region.region_id for region in automatic_noop_sources
            ],
            "protected_content_hash": protected_hash,
            "replacement_region_ids": [],
            "delta_operations": operations,
            "existing_match_region_ids": [],
            "candidate_tokens": len(re.findall(r"\w+|[^\w\s]", current_skill)),
            "repaired_tokens": len(re.findall(r"\w+|[^\w\s]", current_skill)),
            "model_called": False,
        }
        if has_complexity_issue:
            _record_local_edit_complexity(
                current_skill, current_skill, telemetry,
            )
        result = {
            "status": "ok" if not operation_errors else "error",
            "new_skill": current_skill,
            "change_summary": [
                "kept source deltas without a similarity-qualified existing atom"
            ],
            "usage": {},
            "validator_attempts": [{
                "attempt": 0,
                "validation_errors": operation_errors,
                "telemetry": telemetry,
                "usage": {},
            }],
            "editable_region_count": 0,
            "source_delta_count": len(source_regions),
            "automatic_noop_source_region_ids": [
                region.region_id for region in automatic_noop_sources
            ],
            "protected_content_hash": protected_hash,
            "noop": True,
        }
        if operation_errors:
            result["error"] = "automatic NOOP contract validation failed"
        with open(
            os.path.join(out_dir, "local_edit_result.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        if result["status"] == "ok":
            with open(
                os.path.join(out_dir, "local_editd_skill.md"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(current_skill)
        return result
    retry_note = ""
    soft_noop_fallback: dict | None = None
    for attempt_index in range(2):
        try:
            response, usage = chat_optimizer(
                system=system,
                user=user + retry_note,
                max_completion_tokens=max_completion_tokens,
                retries=3,
                stage="regularization_local_edit",
                reasoning_effort=reasoning_effort,
            )
            parsed = extract_json(response)
            errors: list[str] = []
            telemetry: dict = {
                "editable_region_count": len(regions),
                "source_delta_count": len(source_regions),
                "automatic_noop_source_region_ids": [
                    region.region_id for region in automatic_noop_sources
                ],
                "protected_content_hash": protected_hash,
                "model_called": True,
            }
            new_skill = ""
            replacements: dict[str, str] = {}
            operations: list[dict[str, str]] = []
            try:
                replacements = parse_replacements(parsed or {}, regions)
                combined_payload = dict(parsed or {})
                model_operations = combined_payload.get("operations")
                if isinstance(model_operations, list):
                    combined_payload["operations"] = [
                        *model_operations,
                        *automatic_noop_operations,
                    ]
                operations, operation_errors = validate_delta_operations(
                    combined_payload, all_regions, replacements,
                    abnormal_modules=frozenset(abnormal_modules),
                )
                source_order = {
                    region.region_id: index
                    for index, region in enumerate(source_regions)
                }
                operations.sort(
                    key=lambda item: source_order[item["source_region_id"]]
                )
                errors.extend(operation_errors)
                new_skill = apply_replacements(
                    current_skill, all_regions, replacements,
                )
                errors.extend(validate_repaired_document(new_skill))
                errors.extend(new_exact_duplicate_atom_errors(
                    current_skill, new_skill,
                ))
                all_noop = bool(operations) and all(
                    item["operation"] == "NOOP" for item in operations
                )
                if skill_hash(new_skill) == skill_hash(current_skill) and not all_noop:
                    errors.append("local repair made no material change")
                if has_complexity_issue and new_skill:
                    _record_local_edit_complexity(
                        current_skill, new_skill, telemetry,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"local replacement contract failed: {type(exc).__name__}: {exc}")
            telemetry["replacement_region_ids"] = sorted(replacements)
            telemetry["delta_operations"] = operations
            telemetry["existing_match_region_ids"] = sorted(
                region.region_id for region in regions
                if region.role == "existing_match"
            )
            telemetry["candidate_tokens"] = len(
                re.findall(r"\w+|[^\w\s]", current_skill)
            )
            telemetry["repaired_tokens"] = len(
                re.findall(r"\w+|[^\w\s]", new_skill)
            ) if new_skill else 0
            attempts.append({
                "attempt": attempt_index + 1,
                "validation_errors": errors,
                "telemetry": telemetry,
                "usage": usage,
            })
            if errors:
                retry_note = (
                    "\n\n## Structural validator feedback from the previous "
                    "attempt\n- " + "\n- ".join(errors)
                    + "\nRetry once using only the listed region IDs and return "
                    "coherent replacement blocks, not a complete skill."
                )
                continue
            if has_complexity_issue and all_noop and attempt_index == 0:
                telemetry["soft_retry_reason"] = "complexity_first_attempt_all_noop"
                soft_noop_fallback = {
                    "new_skill": new_skill,
                    "change_summary": (parsed or {}).get("change_summary", []),
                    "usage": usage,
                }
                retry_note = (
                    "\n\n## Optional complexity soft retry\n"
                    "The first response selected NOOP for every editable source delta. "
                    "That is valid, but make one final attempt to find a safe local "
                    "REWRITE, DELETE, or MERGE. Do not force a change: return NOOP again "
                    "when no safe repair is supported."
                )
                continue
            result = {
                "status": "ok",
                "new_skill": new_skill,
                "change_summary": (parsed or {}).get("change_summary", []),
                "usage": usage,
                "validator_attempts": attempts,
                "editable_region_count": len(regions),
                "source_delta_count": len(source_regions),
                "automatic_noop_source_region_ids": [
                    region.region_id for region in automatic_noop_sources
                ],
                "protected_content_hash": protected_hash,
                "noop": all_noop,
            }
            break
        except Exception as exc:  # noqa: BLE001
            attempts.append({
                "attempt": attempt_index + 1,
                "validation_errors": [f"{type(exc).__name__}: {exc}"],
            })
            retry_note = (
                "\n\nThe previous response could not be parsed. Retry once "
                "with the exact local replacement JSON contract."
            )
    else:
        if soft_noop_fallback is not None:
            result = {
                "status": "ok",
                **soft_noop_fallback,
                "validator_attempts": attempts,
                "editable_region_count": len(regions),
                "source_delta_count": len(source_regions),
                "automatic_noop_source_region_ids": [
                    region.region_id for region in automatic_noop_sources
                ],
                "protected_content_hash": protected_hash,
                "noop": True,
                "soft_retry_fallback_used": True,
            }
        else:
            result = {
                "status": "error",
                "error": "local structural validator rejected both repair attempts",
                "validator_attempts": attempts,
                "editable_region_count": len(regions),
                "protected_content_hash": protected_hash,
            }
    with open(os.path.join(out_dir, "local_edit_result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    if result.get("status") == "ok":
        with open(os.path.join(out_dir, "local_editd_skill.md"), "w", encoding="utf-8") as handle:
            handle.write(result["new_skill"])
    return result


def _build_step_local_preselection_feedback(
    regularization: SkillEvoReg,
    *,
    old_skill: str,
    candidate_skill: str,
    dropout_report: dict | None,
) -> tuple[dict | None, dict]:
    """Build pre-Val Dropout/complexity signals from the source update only."""
    old_report = regularization.measure(old_skill)
    candidate_report = regularization.measure(candidate_skill)
    growth = normalized_growth(
        old_report,
        candidate_report,
        free_growth_budget=regularization.complexity_free_growth,
    )
    dropout = dict(dropout_report or {})
    dropout_triggered = int(dropout.get("masked_count", 0) or 0) > 0
    complexity_enabled = bool(getattr(regularization, "complexity_enabled", True))
    complexity_triggered = (
        complexity_enabled
        and float(growth.get("normalized_growth", 0.0)) > 0.0
    )
    abnormal_modules = [
        name for name, triggered in (
            ("dropout", dropout_triggered),
            ("complexity", complexity_triggered),
        ) if triggered
    ]
    telemetry = {
        "abnormal_modules": abnormal_modules,
        "dropout": dropout,
        "dropout_triggered": dropout_triggered,
        "complexity_triggered": complexity_triggered,
        "complexity_enabled": complexity_enabled,
        "old_complexity": old_report.to_dict(),
        "candidate_complexity": candidate_report.to_dict(),
        "complexity_growth": growth,
    }
    if not abnormal_modules:
        return None, telemetry
    feedback = {
        "status": "ok",
        "source": "SkillEvoReg",
        "author": "step_local_deterministic_router",
        "abnormal_modules": abnormal_modules,
        "provenance": {
            "train_only": True,
            "validation_access": False,
            "test_access": False,
        },
    }
    if dropout_triggered:
        feedback["dropout"] = {
            "triggered": True,
            "masked_count": int(dropout.get("masked_count", 0) or 0),
            "mode": dropout.get("mode", "training_rollout"),
            "repair_policy": {
                "objective": (
                    "consolidate the source delta produced from a partial "
                    "training-time skill view without forcing compression"
                ),
                "preserve_novel_supported_semantics": True,
                "prefer": ["merge_duplicate", "rewrite_compensation", "noop"],
            },
        }
    if complexity_triggered:
        feedback["complexity"] = {
            "triggered": True,
            "relative_growth": growth.get("relative_growth"),
            "free_growth_budget": growth.get("free_growth_budget"),
            "penalized_growth": growth.get("normalized_growth"),
            "next_update_policy": {
                "objective": (
                    "reduce the source delta's net complexity contribution "
                    "while preserving train-supported capabilities"
                ),
                "prefer_replacement_over_append": True,
                "reduce_total_complexity": True,
            },
        }
    return feedback, telemetry


def _build_step_local_ccv_feedback(ccv: dict) -> dict | None:
    """Route only verifier-confirmed CCV regressions into local repair."""
    regressions = int(
        ccv.get("confirmed_regressions", ccv.get("regressions", 0)) or 0
    )
    if regressions <= 0:
        return None
    counterexamples = []
    for record in ccv.get("attack_records", []) or []:
        if record.get("status") != "update_regression":
            continue
        pattern = record.get("pattern") or {}
        generation = record.get("generation") or {}
        counterexamples.append({
            "attack_id": record.get("attack_id"),
            "source_id": record.get("source_id"),
            "hypothesis": pattern.get("incidental_assumption", ""),
            "intervention": generation.get("intervention", ""),
        })
    return {
        "status": "ok",
        "source": "SkillEvoReg",
        "author": "step_local_ccv_router",
        "abnormal_modules": ["ccv"],
        "provenance": {
            "ccv_source_split": "train",
            "validation_access": False,
            "test_access": False,
        },
        "ccv": {
            "triggered": True,
            "confirmed_regressions": regressions,
            "allow_unmatched_source_rewrite": True,
            "counterexamples": counterexamples,
            "repair_policy": {
                "derive_exactly_one_transferable_invariant": True,
                "prefer": ["rewrite", "merge", "generalize"],
                "append_only_if_no_responsible_atom_exists": True,
            },
        },
    }


# ── Trainer ──────────────────────────────────────────────────────────────────

class ReflACTTrainer:
    """Main ReflACT training loop.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. See ``configs/alfworld_default.yaml``
        for the full list of keys.
    adapter : EnvAdapter
        Environment adapter instance.
    """

    def __init__(self, cfg: dict, adapter: EnvAdapter) -> None:
        self.cfg = cfg
        self.adapter = adapter

    def train(self) -> dict:
        """Execute the full ReflACT training loop. Returns summary dict."""
        cfg = self.cfg
        adapter = self.adapter
        out_root = cfg["out_root"]
        os.makedirs(out_root, exist_ok=True)
        os.environ["SKILLOPT_API_EVENTS_PATH"] = os.path.abspath(
            os.path.join(out_root, "api_events.jsonl")
        )

        # ── Adapter setup (one-time init) ────────────────────────────
        adapter.setup(cfg)
        dataloader = adapter.get_dataloader()

        def _build_train_env(batch: BatchSpec):
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size, batch.seed

        def _build_eval_env(split: str, env_num: int, seed: int):
            if dataloader is None:
                env_manager = adapter.build_eval_env(
                    env_num=env_num,
                    split=split,
                    seed=seed,
                    out_root=out_root,
                )
                actual_n = len(env_manager) if hasattr(env_manager, "__len__") else env_num
                return env_manager, actual_n

            batch = dataloader.build_eval_batch(
                env_num=env_num,
                split=split,
                seed=seed,
                out_root=out_root,
            )
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size

        # ── Configure models ─────────────────────────────────────────────
        backend = cfg.get("model_backend", "azure_openai")
        configure_azure_openai(
            endpoint=(
                cfg.get("azure_openai_endpoint")
                or cfg.get("azure_endpoint")
                or None
            ),
            api_version=(
                cfg.get("azure_openai_api_version")
                or cfg.get("azure_api_version")
                or None
            ),
            api_key=(
                cfg.get("azure_openai_api_key")
                or cfg.get("azure_api_key")
                or None
            ),
            auth_mode=cfg.get("azure_openai_auth_mode") or None,
            ad_scope=cfg.get("azure_openai_ad_scope") or None,
            managed_identity_client_id=cfg.get("azure_openai_managed_identity_client_id") or None,
            optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or None,
            optimizer_api_version=cfg.get("optimizer_azure_openai_api_version") or None,
            optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or None,
            optimizer_auth_mode=cfg.get("optimizer_azure_openai_auth_mode") or None,
            optimizer_ad_scope=cfg.get("optimizer_azure_openai_ad_scope") or None,
            optimizer_managed_identity_client_id=(
                cfg.get("optimizer_azure_openai_managed_identity_client_id") or None
            ),
            target_endpoint=cfg.get("target_azure_openai_endpoint") or None,
            target_api_version=cfg.get("target_azure_openai_api_version") or None,
            target_api_key=cfg.get("target_azure_openai_api_key") or None,
            target_auth_mode=cfg.get("target_azure_openai_auth_mode") or None,
            target_ad_scope=cfg.get("target_azure_openai_ad_scope") or None,
            target_managed_identity_client_id=(
                cfg.get("target_azure_openai_managed_identity_client_id") or None
            ),
        )
        optimizer_backend = cfg.get("optimizer_backend")
        target_backend = cfg.get("target_backend")
        if not optimizer_backend or not target_backend:
            if backend in {"claude", "claude_chat"}:
                optimizer_backend = optimizer_backend or "claude_chat"
                target_backend = target_backend or "claude_chat"
            elif backend in {"codex", "codex_exec"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "codex_exec"
            elif backend in {"harbor", "harbor_codex_exec"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "harbor_codex_exec"
            elif backend == "claude_code_exec":
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "claude_code_exec"
            elif backend in {"qwen", "qwen_chat"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "qwen_chat"
            else:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "openai_chat"
            cfg["optimizer_backend"] = optimizer_backend
            cfg["target_backend"] = target_backend
        set_optimizer_backend(optimizer_backend)
        set_target_backend(target_backend)
        set_optimizer_deployment(cfg["optimizer_model"])
        set_target_deployment(cfg["target_model"])
        configure_codex_exec(
            path=cfg.get("codex_exec_path", "codex"),
            sandbox=cfg.get("codex_exec_sandbox", "workspace-write"),
            profile=cfg.get("codex_exec_profile", ""),
            full_auto=cfg.get("codex_exec_full_auto", False),
            reasoning_effort=cfg.get("codex_exec_reasoning_effort", "none"),
            use_sdk=cfg.get("codex_exec_use_sdk", None),
            network_access=cfg.get("codex_exec_network_access", False),
            web_search=cfg.get("codex_exec_web_search", False),
            approval_policy=cfg.get("codex_exec_approval_policy", "never"),
        )
        configure_harbor_exec(
            python=cfg.get("harbor_exec_python", ""),
            root=cfg.get("harbor_exec_root", ""),
            jobs_root=cfg.get("harbor_exec_jobs_root", ""),
            infrastructure_retries=cfg.get("harbor_exec_infrastructure_retries", 2),
            workers=cfg.get("workers", 1),
            image=cfg.get(
                "harbor_exec_image", "agent-overfit-skillopt-runtime:latest"
            ),
        )
        configure_claude_code_exec(
            path=cfg.get("claude_code_exec_path", "claude"),
            profile=cfg.get("claude_code_exec_profile", ""),
            use_sdk=cfg.get("claude_code_exec_use_sdk", None),
            effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
            max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
        )
        configure_qwen_chat(
            base_url=cfg.get("qwen_chat_base_url") or None,
            api_key=cfg.get("qwen_chat_api_key") or None,
            temperature=cfg.get("qwen_chat_temperature"),
            timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
            max_tokens=cfg.get("qwen_chat_max_tokens"),
            enable_thinking=cfg.get("qwen_chat_enable_thinking"),
            optimizer_base_url=cfg.get("optimizer_qwen_chat_base_url") or None,
            optimizer_api_key=cfg.get("optimizer_qwen_chat_api_key") or None,
            optimizer_temperature=cfg.get("optimizer_qwen_chat_temperature"),
            optimizer_timeout_seconds=cfg.get("optimizer_qwen_chat_timeout_seconds"),
            optimizer_max_tokens=cfg.get("optimizer_qwen_chat_max_tokens"),
            optimizer_enable_thinking=cfg.get("optimizer_qwen_chat_enable_thinking"),
            target_base_url=cfg.get("target_qwen_chat_base_url") or None,
            target_api_key=cfg.get("target_qwen_chat_api_key") or None,
            target_temperature=cfg.get("target_qwen_chat_temperature"),
            target_timeout_seconds=cfg.get("target_qwen_chat_timeout_seconds"),
            target_max_tokens=cfg.get("target_qwen_chat_max_tokens"),
            target_enable_thinking=cfg.get("target_qwen_chat_enable_thinking"),
        )
        configure_minimax_chat(
            base_url=cfg.get("minimax_base_url") or None,
            api_key=cfg.get("minimax_api_key") or None,
            temperature=cfg.get("minimax_temperature"),
            max_tokens=cfg.get("minimax_max_tokens"),
            enable_thinking=cfg.get("minimax_enable_thinking"),
        )
        minimax_model_cfg = cfg.get("minimax_model")
        if minimax_model_cfg and cfg.get("target_backend") == "minimax_chat":
            set_target_deployment(str(minimax_model_cfg))
        os.environ["REFLACT_CODEX_TRACE_TO_OPTIMIZER"] = (
            "1"
            if target_backend == "codex_exec" and cfg.get("codex_trace_to_optimizer", False)
            else "0"
        )
        reasoning = cfg.get("reasoning_effort", "") or None
        set_reasoning_effort(reasoning)
        print(
            f"  [model config] backend={backend}  "
            f"optimizer={cfg['optimizer_model']} ({optimizer_backend})  "
            f"target={cfg['target_model']} ({target_backend})  "
            f"reasoning={reasoning or 'off'}"
        )

        # ── Initialize Ray ───────────────────────────────────────────────
        if adapter.requires_ray():
            try:
                import ray
            except ImportError as e:
                raise ImportError(
                    "This environment requires ray, but ray is not installed."
                ) from e

            if not ray.is_initialized():
                ray.init(num_gpus=0)

        # ── Load initial skill ───────────────────────────────────────────
        skill_init_path = os.path.abspath(cfg["skill_init"])
        if os.path.exists(skill_init_path):
            with open(skill_init_path) as f:
                skill_init = f.read()
            print(f"  [initial skill] {skill_init_path} ({len(skill_init)} chars)")
        else:
            skill_init = ""
            print("  [initial skill] no initial skill file — starting from blank")

        regularization = (
            SkillEvoReg(cfg, initial_skill=skill_init, adapter=adapter)
            if bool(cfg.get("regularization_enabled", False))
            else None
        )

        # ── Training parameters ──────────────────────────────────────────
        batch_size = cfg["batch_size"]
        num_epochs = cfg["num_epochs"]
        accumulation = cfg["accumulation"]
        seed = cfg["seed"]
        merge_bs = cfg["merge_batch_size"]
        max_analyst_rounds = int(cfg.get("max_analyst_rounds", 3) or 3)
        update_mode = normalize_update_mode(cfg.get("skill_update_mode", "patch"))
        lr_control_mode = _normalise_lr_control_mode(cfg.get("lr_control_mode", "fixed"))
        if is_full_rewrite_minibatch_mode(update_mode):
            lr_control_mode = "none"
        longitudinal_pair_policy = _normalise_longitudinal_pair_policy(
            cfg.get("longitudinal_pair_policy", "mixed")
        )
        rewrite_reasoning_effort = cfg.get("rewrite_reasoning_effort", "high")
        if rewrite_reasoning_effort == "":
            rewrite_reasoning_effort = None
        rewrite_max_completion_tokens = int(cfg.get("rewrite_max_completion_tokens", 64000))
        regularization_feedback_consumption_mode = str(
            cfg.get("regularization_feedback_consumption_mode", "deferred")
        ).strip().lower()
        regularization_step_local_pipeline = bool(
            cfg.get("regularization_step_local_pipeline", False)
        )
        regularization_embedding_model = str(
            cfg.get("regularization_embedding_model", DEFAULT_EMBEDDING_MODEL)
        ).strip()
        regularization_embedding_match_threshold = float(cfg.get(
            "regularization_embedding_match_threshold",
            DEFAULT_EMBEDDING_MATCH_THRESHOLD,
        ))
        if regularization_feedback_consumption_mode not in {"deferred", "local_edit"}:
            raise ValueError(
                "regularization.feedback_consumption_mode must be 'deferred' or "
                f"'local_edit', got {regularization_feedback_consumption_mode!r}"
            )
        if not regularization_embedding_model:
            raise ValueError("regularization.embedding_model must be non-empty")
        if not 0.0 <= regularization_embedding_match_threshold <= 1.0:
            raise ValueError(
                "regularization.embedding_match_threshold must be in [0, 1], got "
                f"{regularization_embedding_match_threshold}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if accumulation <= 0:
            raise ValueError(f"accumulation must be positive, got {accumulation}")

        train_size = _resolve_train_size(cfg, dataloader)
        configured_steps_per_epoch = int(cfg.get("steps_per_epoch", 0) or 0)
        if configured_steps_per_epoch > 0:
            steps_per_epoch = configured_steps_per_epoch
            steps_source = "explicit"
        else:
            steps_per_epoch = math.ceil(train_size / (batch_size * accumulation))
            steps_source = "auto"
        batches_per_epoch = steps_per_epoch * accumulation
        total_steps = num_epochs * steps_per_epoch
        samples_per_epoch = min(train_size, batches_per_epoch * batch_size)

        # Persist resolved derived fields so config.json / summary.json match
        # the actual runtime recipe.
        cfg["train_size"] = train_size
        cfg["steps_per_epoch"] = steps_per_epoch
        cfg["batches_per_epoch"] = batches_per_epoch
        cfg["samples_per_epoch"] = samples_per_epoch
        cfg["skill_update_mode"] = update_mode
        cfg["lr_control_mode"] = lr_control_mode

        # Save config after deriving runtime values.
        with open(os.path.join(out_root, "config.json"), "w") as f:
            json.dump(_redact_cfg(cfg), f, indent=2, ensure_ascii=False)

        train_pool_size = samples_per_epoch

        scheduler = build_scheduler(
            mode=cfg.get("lr_scheduler", "constant"),
            max_lr=cfg["edit_budget"],
            min_lr=cfg.get("min_edit_budget", 2),
            total_steps=total_steps,
        )

        # Fixed training pool: base seeds (each seed = one deterministic batch)
        if dataloader is not None:
            base_seeds = dataloader.make_base_seeds(
                steps_per_epoch=steps_per_epoch,
                accumulation=accumulation,
                seed=seed,
            )
        else:
            base_seeds = [seed + i + 1 for i in range(batches_per_epoch)]

        print(f"\n  [config] epochs={num_epochs} steps/epoch={steps_per_epoch} "
              f"({steps_source}) accum={accumulation} batch_size={batch_size}")
        print(f"  [config] train_size={train_size}")
        print(f"  [config] batches/epoch={batches_per_epoch} "
              f"total_steps={total_steps} "
              f"games/epoch={train_pool_size}")
        print(f"  [config] lr_scheduler={cfg.get('lr_scheduler', 'constant')} "
              f"edit_budget={cfg['edit_budget']} "
              f"min_edit_budget={cfg.get('min_edit_budget', 2)}")
        print(f"  [config] skill_update_mode={update_mode} "
              f"lr_control_mode={lr_control_mode} "
              f"rewrite_reasoning_effort={rewrite_reasoning_effort or 'off'} "
              f"rewrite_max_completion_tokens={rewrite_max_completion_tokens} "
              f"max_analyst_rounds={max_analyst_rounds}")
        print(f"  [config] longitudinal_pair_policy={longitudinal_pair_policy}")
        print(f"  [config] base_seeds={base_seeds}")

        # ── Resume check ─────────────────────────────────────────────────
        history = _load_history(out_root)
        runtime_state = _load_runtime_state(out_root)
        if runtime_state:
            last_step = int(runtime_state.get("last_completed_step", 0) or 0)
            current_skill_path = runtime_state.get("current_skill_path") or os.path.join(
                out_root, "skills", f"skill_v{last_step:04d}.md",
            )
            with open(current_skill_path) as f:
                current_skill = f.read()
            best_skill_path = runtime_state.get("best_skill_path") or os.path.join(
                out_root, "best_skill.md",
            )
            if os.path.exists(best_skill_path):
                with open(best_skill_path) as f:
                    best_skill = f.read()
            else:
                best_skill = current_skill
            current_score = float(runtime_state.get("current_score", -1.0) or -1.0)
            best_score = float(runtime_state.get("best_score", current_score) or current_score)
            best_step = runtime_state.get("best_step", last_step)
            current_origin = str(
                runtime_state.get("current_origin")
                or (f"step_{last_step:04d}" if last_step > 0 else "initial_skill")
            )
            best_origin = str(runtime_state.get("best_origin") or current_origin)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f} "
                f"(origin={current_origin})"
            )
        elif history:
            last_step = history[-1]["step"]
            current_skill = _load_skill(out_root, last_step)
            best_rec = max(history, key=lambda h: h.get("best_score", 0.0))
            best_score = best_rec["best_score"]
            best_step = best_rec["best_step"]
            best_skill_path = os.path.join(out_root, "best_skill.md")
            if os.path.exists(best_skill_path):
                with open(best_skill_path) as f:
                    best_skill = f.read()
            else:
                best_skill = _load_skill(out_root, best_step)
            current_score = history[-1].get("current_score", best_score)
            current_origin = f"step_{last_step:04d}"
            best_origin = f"step_{int(best_step):04d}" if isinstance(best_step, int) else str(best_step)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f}"
            )
        else:
            current_skill = skill_init
            best_skill = skill_init
            best_score = -1.0
            current_score = -1.0
            best_step = 0
            current_origin = "initial_skill"
            best_origin = "initial_skill"
            resume_from = 1

        _save_skill(out_root, 0, skill_init)

        use_skill_aware = cfg.get("use_skill_aware_reflection", False)
        # Publish the toggle process-wide so run_minibatch_reflect resolves it
        # from config for EVERY env adapter — no per-benchmark wiring needed.
        configure_skill_aware_reflection(
            use_skill_aware,
            cfg.get("skill_aware_appendix_source", "both"),
        )
        if use_skill_aware:
            current_skill = inject_empty_appendix_field(current_skill)

        epoch_transaction = dict((runtime_state or {}).get("epoch_transaction") or {})
        epoch_feedback_for_steps = (runtime_state or {}).get("epoch_feedback_for_steps")
        local_edit_fallback_feedback = (runtime_state or {}).get(
            "local_edit_fallback_feedback"
        )

        def _persist_runtime_state(last_completed_step: int) -> None:
            _save_runtime_state(
                out_root,
                {
                    "last_completed_step": last_completed_step,
                    "current_skill_path": os.path.join(
                        out_root, "skills", f"skill_v{last_completed_step:04d}.md",
                    ),
                    "current_score": current_score,
                    "current_origin": current_origin,
                    "best_skill_path": os.path.join(out_root, "best_skill.md"),
                    "best_score": best_score,
                    "best_step": best_step,
                    "best_origin": best_origin,
                    "epoch_transaction": epoch_transaction,
                    "epoch_feedback_for_steps": epoch_feedback_for_steps,
                    "local_edit_fallback_feedback": local_edit_fallback_feedback,
                },
            )

        # ── Selection cache ──────────────────────────────────────────────
        sel_cache: dict[str, tuple[float, float]] = {}
        for rec in history:
            sh = rec.get("candidate_hash", "")
            if sh and rec.get("selection_hard") is not None:
                sel_cache[sh] = (rec["selection_hard"], rec["selection_soft"])

        # ── Baseline evaluation on selection set ─────────────────────────
        # `use_gate=False` keeps validation running (selection rollout +
        # scoring are unconditional below) but force-accepts every candidate
        # instead of gating it; final skill is chosen manually afterwards.
        use_gate = cfg.get("use_gate", True) is not False
        paired_selection_repeats = max(
            0, int(cfg.get("paired_selection_repeats", 0) or 0)
        )
        gate_metric = str(cfg.get("gate_metric", "hard")).strip().lower()
        if gate_metric not in {"hard", "soft", "mixed"}:
            raise ValueError(
                f"evaluation.gate_metric must be 'hard' | 'soft' | 'mixed', "
                f"got {gate_metric!r}"
            )
        gate_mixed_weight = float(cfg.get("gate_mixed_weight", 0.5))
        use_semantic_density = bool(cfg.get("use_semantic_density", False))
        if regularization is not None and not use_gate:
            raise ValueError(
                "SkillEvoReg requires evaluation.use_gate=true because it "
                "replaces the native accept/reject decision."
            )
        if regularization is not None and use_semantic_density:
            raise ValueError(
                "SkillEvoReg requires evaluation.use_semantic_density=false "
                "so DeltaIID is not contaminated by a skill-text bonus."
            )
        if regularization is not None and bool(cfg.get("use_slow_update", False)):
            raise ValueError(
                "SkillEvoReg v1 requires optimizer.use_slow_update=false. "
                "The epoch-level slow-update path is a second candidate-update "
                "boundary and must not bypass the regularization."
            )
        semantic_density_weight = float(cfg.get("semantic_density_weight", 0.05))
        leading_words_raw = cfg.get("leading_words", None)
        leading_words = None
        if leading_words_raw is not None:
            if isinstance(leading_words_raw, str):
                leading_words = [w.strip() for w in leading_words_raw.split(",") if w.strip()]
            else:
                leading_words = list(leading_words_raw)
        if not 0.0 <= gate_mixed_weight <= 1.0:
            raise ValueError(
                f"evaluation.gate_mixed_weight must be in [0, 1], "
                f"got {gate_mixed_weight}"
            )
        print(
            f"  [gate] metric={gate_metric}"
            + (
                f" mixed_weight={gate_mixed_weight}"
                if gate_metric == "mixed"
                else ""
            )
            + ("" if use_gate
               else "  (DISABLED → validation runs, candidates force-accepted)")
        )
        if regularization is not None:
            print(
                "  [SkillEvoReg] enabled "
                f"lambda_complex={regularization.lambda_complex_min}"
                f"->{regularization.lambda_complex_max} "
                f"lambda_scheduler={regularization.lambda_complex_scheduler} "
                f"complexity_free_growth={regularization.complexity_free_growth} "
                f"dropout={regularization.dropout_enabled} "
                f"p_drop={regularization.dropout_probability} "
                f"dropout_min_one={regularization.dropout_ensure_at_least_one} "
                f"feedback={regularization.feedback_enabled} "
                f"ccv={regularization.ccv_enabled} "
                f"ccv_required={regularization.ccv_required} "
                f"tau_adv={regularization.tau_adv}"
            )
            print(
                "  [Regularization feedback consumption] "
                f"mode={regularization_feedback_consumption_mode}"
            )
        slow_gate_with_selection = bool(
            cfg.get("slow_update_gate_with_selection", False)
        )
        if paired_selection_repeats >= 2:
            print(
                "  [paired selection] candidate+incumbent repeats="
                f"{paired_selection_repeats}; cache disabled for gate decisions"
            )
        print(
            "  [slow update] acceptance="
            + ("gated (selection-set validation)"
               if slow_gate_with_selection
               else "force-accept (unconditional)")
        )
        if current_score < 0:
            print(f"\n{'='*60}")
            print("  BASELINE — evaluate initial skill on Selection set (valid_seen)")
            print(f"{'='*60}")
            sel_env, sel_n = _build_eval_env(
                split="valid_seen",
                env_num=cfg["sel_env_num"],
                seed=seed,
            )
            print(f"  Selection items: {sel_n}")
            baseline_dir = os.path.join(out_root, "selection_eval_baseline")
            baseline_results = adapter.rollout(sel_env, skill_init, baseline_dir)
            baseline_hard, baseline_soft = compute_score(baseline_results)
            current_score = select_gate_score(
                baseline_hard, baseline_soft, gate_metric, gate_mixed_weight,
                skill_content=skill_init,
                use_semantic_density=use_semantic_density,
                semantic_density_weight=semantic_density_weight,
                leading_words=leading_words,
            )
            best_score = current_score
            sh = skill_hash(skill_init)
            sel_cache[sh] = (baseline_hard, baseline_soft)
            current_origin = "initial_skill"
            best_origin = "initial_skill"
            _persist_runtime_state(0)
            print(
                f"  [baseline result] selection hard={baseline_hard:.4f} "
                f"soft={baseline_soft:.4f} "
                f"gate[{gate_metric}]={current_score:.4f}"
            )

        # ── Training loop ────────────────────────────────────────────────
        pending_regularization_feedback = None
        if (
            regularization_feedback_consumption_mode == "deferred"
            and regularization is not None and regularization.feedback_enabled and history
        ):
            pending_regularization_feedback = history[-1].get("regularization_feedback_next_step")
        if (
            regularization_feedback_consumption_mode == "deferred"
            and regularization is not None and regularization.epoch_agents_enabled
        ):
            pending_regularization_feedback = epoch_feedback_for_steps

        t_loop_start = time.time()

        if resume_from > total_steps:
            print(f"\n  [skip] all {total_steps} steps complete — jumping to evaluation")

        global_step = 0
        for epoch in range(1, num_epochs + 1):
            if dataloader is not None:
                epoch_batches = dataloader.plan_train_epoch(
                    epoch=epoch,
                    steps_per_epoch=steps_per_epoch,
                    accumulation=accumulation,
                    batch_size=batch_size,
                    seed=seed,
                    out_root=out_root,
                )
                shuffled_seeds = [batch.seed for batch in epoch_batches]
            else:
                epoch_batches = []
                epoch_rng = random.Random(seed + epoch * 1000)
                shuffled_seeds = base_seeds.copy()
                epoch_rng.shuffle(shuffled_seeds)

            # Step buffer: accumulates per-step context (failure patterns +
            # rejected edits) within this epoch so optimizers see full history.
            step_buffer: list[dict] = []
            active_meta_skill = (
                _load_meta_skill_content(out_root, epoch - 1)
                if cfg.get("use_meta_skill", False)
                else ""
            )

            print(
                f"\n  [EPOCH {epoch}/{num_epochs}] "
                f"shuffled_seeds={shuffled_seeds}"
            )
            if active_meta_skill:
                print(
                    f"  [meta skill] loaded from epoch {epoch - 1} "
                    f"({len(active_meta_skill)} chars)"
                )

            epoch_first_step = (epoch - 1) * steps_per_epoch + 1
            epoch_last_step = epoch * steps_per_epoch
            epoch_edit_budget = (
                _load_epoch_edit_budget(
                    out_root,
                    epoch_first_step,
                    min(epoch_last_step, resume_from - 1),
                )
                if resume_from > epoch_first_step
                else None
            )
            if epoch_first_step < resume_from <= epoch_last_step:
                for prior_step in range(epoch_first_step, resume_from):
                    digest_path = os.path.join(
                        out_root, "steps", f"step_{prior_step:04d}",
                        "trajectory_digest.json",
                    )
                    if not os.path.exists(digest_path):
                        continue
                    try:
                        with open(digest_path, encoding="utf-8") as handle:
                            digest = json.load(handle)
                        if isinstance(digest, dict):
                            step_buffer.append(digest)
                    except Exception:
                        pass
                if step_buffer:
                    print(
                        f"  [resume] restored {len(step_buffer)} prior "
                        "step-buffer entries for this epoch"
                    )
            transaction_pending_here = (
                int(epoch_transaction.get("epoch", 0) or 0) == epoch
                and epoch_transaction.get("status") != "completed"
            )
            epoch_already_complete = (
                epoch_last_step < resume_from and not transaction_pending_here
            )
            epoch_agent_dir = os.path.join(out_root, "epoch_agents", f"epoch_{epoch:02d}")
            epoch_start_snapshot: dict = {}
            if regularization is not None and regularization.epoch_agents_enabled and not epoch_already_complete:
                if int(epoch_transaction.get("epoch", 0) or 0) == epoch:
                    epoch_start_snapshot = dict(epoch_transaction.get("start") or {})
                else:
                    os.makedirs(epoch_agent_dir, exist_ok=True)
                    start_skill_path = os.path.join(epoch_agent_dir, "epoch_start_skill.md")
                    start_best_path = os.path.join(epoch_agent_dir, "epoch_start_best_skill.md")
                    with open(start_skill_path, "w", encoding="utf-8") as handle:
                        handle.write(current_skill)
                    with open(start_best_path, "w", encoding="utf-8") as handle:
                        handle.write(best_skill)
                    epoch_start_snapshot = {
                        "current_skill_path": start_skill_path,
                        "current_score": current_score,
                        "current_origin": current_origin,
                        "best_skill_path": start_best_path,
                        "best_score": best_score,
                        "best_step": best_step,
                        "best_origin": best_origin,
                    }
                    epoch_transaction = {
                        "epoch": epoch, "status": "training",
                        "first_step": epoch_first_step, "last_step": epoch_last_step,
                        "start": epoch_start_snapshot,
                    }
                    _persist_runtime_state(epoch_first_step - 1)

            for step_in_epoch in range(steps_per_epoch):
                global_step += 1
                if global_step < resume_from:
                    continue

                step_t0 = time.time()
                step_dir = os.path.join(out_root, "steps", f"step_{global_step:04d}")
                os.makedirs(step_dir, exist_ok=True)

                tokens_before = get_token_summary()

                print(
                    f"\n  [STEP {global_step}/{total_steps}] "
                    f"epoch={epoch} step_in_epoch={step_in_epoch} "
                    f"{'='*30}"
                )

                step_rec: dict = {
                    "step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "timing": {},
                    "tokens": {},
                }

                if regularization_feedback_consumption_mode == "local_edit":
                    active_regularization_feedback = local_edit_fallback_feedback
                    local_edit_fallback_feedback = None
                else:
                    active_regularization_feedback = pending_regularization_feedback
                    if regularization is None or not regularization.epoch_agents_enabled:
                        pending_regularization_feedback = None
                regularization_feedback_context = _format_regularization_feedback(
                    active_regularization_feedback
                )
                if active_regularization_feedback:
                    step_rec["regularization_feedback_consumed"] = active_regularization_feedback

                proposal_skill = current_skill
                dropout_context = ""
                dropout_report: dict = {}
                if regularization is not None:
                    dropout_seed = seed + global_step * 104729
                    proposal_skill, dropout_report, dropout_context = (
                        regularization.prepare_proposal(
                            current_skill,
                            seed=dropout_seed,
                            step=global_step,
                        )
                    )
                    with open(
                        os.path.join(step_dir, "atomic_dropout.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(dropout_report, f, ensure_ascii=False, indent=2)
                    with open(
                        os.path.join(step_dir, "proposal_skill.md"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(proposal_skill)
                    step_rec["regularization_dropout"] = {
                        "enabled": dropout_report.get("enabled", False),
                        "masked_count": dropout_report.get("masked_count", 0),
                        "eligible_atoms": dropout_report.get("eligible_atoms", 0),
                        "probability": dropout_report.get("probability", 0.0),
                        "seed": dropout_seed,
                    }

                # ── Accumulation: Rollout + Reflect ──────────────────────
                all_failure_patches: list[dict] = []
                all_success_patches: list[dict] = []
                all_raw_patches: list[dict | None] = []
                all_rollout_results: list[dict] = []
                accum_rollout_stats: list[dict] = []
                total_rollout_time = 0.0
                total_reflect_time = 0.0

                for a in range(accumulation):
                    batch_idx = step_in_epoch * accumulation + a
                    if dataloader is not None:
                        batch_spec = epoch_batches[batch_idx]
                        train_env, train_n, batch_seed = _build_train_env(batch_spec)
                    else:
                        batch_seed = shuffled_seeds[batch_idx]
                        train_env = adapter.build_train_env(
                            batch_size=batch_size,
                            seed=batch_seed,
                            out_root=out_root,
                        )
                        train_n = len(train_env) if hasattr(train_env, "__len__") else batch_size

                    # Directory routing
                    if accumulation > 1:
                        batch_dir = os.path.join(step_dir, f"batch_{a}")
                    else:
                        batch_dir = step_dir

                    rollout_dir = os.path.join(batch_dir, "rollout")
                    patches_dir = os.path.join(batch_dir, "patches")

                    # ① ROLLOUT ────────────────────────────────────────────
                    t_phase = time.time()
                    print(f"    [1/6 ROLLOUT] train items={train_n} (from pool, batch_seed={batch_seed})")
                    # Training-time atomic dropout: the same masked skill is
                    # used for the forward/rollout pass and for reflection.
                    # Val/Test/CCV remain unmasked, and the resulting patch is
                    # committed to the complete incumbent below.
                    rollout_results = adapter.rollout(
                        train_env, proposal_skill, rollout_dir,
                        use_eval_feedback=True,
                    )
                    r_hard, r_soft = compute_score(rollout_results)
                    total_rollout_time += time.time() - t_phase
                    all_rollout_results.extend(rollout_results)
                    print(f"    [1/6 done] hard={r_hard:.4f} soft={r_soft:.4f}")

                    # ② REFLECT ────────────────────────────────────────────
                    t_phase = time.time()
                    pred_dir = os.path.join(rollout_dir, "predictions")

                    # Build step context from buffer
                    step_buffer_context = _format_step_buffer(step_buffer)
                    if regularization_feedback_context:
                        step_buffer_context = (
                            f"{regularization_feedback_context}\n\n{step_buffer_context}"
                            if step_buffer_context
                            else regularization_feedback_context
                        )
                    if dropout_context:
                        step_buffer_context = (
                            f"{dropout_context}\n\n{step_buffer_context}"
                            if step_buffer_context
                            else dropout_context
                        )
                    step_buffer_context, history_redactions = (
                        redact_masked_atom_bodies(
                            step_buffer_context, dropout_report
                        )
                    )
                    proposal_regularization_feedback_context, feedback_redactions = (
                        redact_masked_atom_bodies(
                            regularization_feedback_context, dropout_report
                        )
                    )
                    if history_redactions or feedback_redactions:
                        step_rec["regularization_dropout"]["context_redactions"] = (
                            int(step_rec["regularization_dropout"].get(
                                "context_redactions", 0
                            ))
                            + history_redactions + feedback_redactions
                        )

                    raw_patches = adapter.reflect(
                        rollout_results, proposal_skill, batch_dir,
                        prediction_dir=pred_dir, patches_dir=patches_dir,
                        random_seed=batch_seed,
                        step_buffer_context=step_buffer_context,
                        meta_skill_context=active_meta_skill,
                    )
                    failure_patches, success_patches = _normalise_patches(
                        raw_patches,
                        update_mode=update_mode,
                    )
                    all_failure_patches.extend(failure_patches)
                    all_success_patches.extend(success_patches)
                    all_raw_patches.extend(raw_patches)
                    total_reflect_time += time.time() - t_phase

                    print(
                        f"    [2/6 done] failure_patches={len(failure_patches)} "
                        f"success_patches={len(success_patches)}"
                    )

                    # Track per-batch stats
                    accum_rollout_stats.append({
                        "batch_idx": a,
                        "batch_seed": batch_seed,
                        "n_envs": len(rollout_results),
                        "hard": r_hard,
                        "soft": r_soft,
                        "n_failure_patches": len(failure_patches),
                        "n_success_patches": len(success_patches),
                    })

                # ── End of accumulation loop ─────────────────────────────

                # Aggregate rollout stats across batches
                total_n = sum(b["n_envs"] for b in accum_rollout_stats)
                agg_hard = sum(b["hard"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)
                agg_soft = sum(b["soft"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)

                step_rec["rollout_hard"] = round(agg_hard, 6)
                step_rec["rollout_soft"] = round(agg_soft, 6)
                step_rec["rollout_n"] = total_n
                step_rec["accumulation_batches"] = accum_rollout_stats
                step_rec["timing"]["rollout_s"] = round(total_rollout_time, 1)
                step_rec["timing"]["reflect_s"] = round(total_reflect_time, 1)

                n_total_patches = len(all_failure_patches) + len(all_success_patches)
                step_rec["n_patches"] = n_total_patches
                step_rec["n_failure_patches"] = len(all_failure_patches)
                step_rec["n_success_patches"] = len(all_success_patches)

                if accumulation > 1:
                    print(
                        f"    [accum done] total: failure={len(all_failure_patches)} "
                        f"success={len(all_success_patches)} "
                        f"from {accumulation} batches"
                    )

                # ── No patches? Skip ─────────────────────────────────────
                if not all_failure_patches and not all_success_patches:
                    # Skill-aware: a lapse-only step has no body patches but
                    # may still carry appendix notes — flush them BEFORE
                    # skipping, or they would be silently dropped.
                    if use_skill_aware:
                        current_skill = _flush_skill_aware_appendix(
                            current_skill, all_raw_patches, step_rec, step_dir, cfg,
                        )
                    step_rec["action"] = "skip_no_patches"
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["skill_len"] = len(current_skill)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_skill(out_root, global_step, current_skill)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [skip] no usable patches — skill unchanged")
                    continue

                # ③ AGGREGATE ──────────────────────────────────────────────
                t_phase = time.time()
                merged_patch = merge_patches(
                    proposal_skill, all_failure_patches, all_success_patches,
                    batch_size=merge_bs, verbose=True,
                    workers=cfg["analyst_workers"],
                    update_mode=update_mode,
                    meta_skill_context=active_meta_skill,
                )
                with open(os.path.join(step_dir, "merged_patch.json"), "w") as f:
                    json.dump(merged_patch, f, ensure_ascii=False, indent=2)

                merged_items = get_payload_items(merged_patch, update_mode)
                n_edits_merged = len(merged_items)
                step_rec["n_edits_merged"] = n_edits_merged
                step_rec["timing"]["aggregate_s"] = round(time.time() - t_phase, 1)
                print(f"    [3/6 done] merged {n_edits_merged} {payload_label(update_mode)}")

                # ④ SELECT ─────────────────────────────────────────────────
                t_phase = time.time()
                lr_decision = None
                if is_full_rewrite_minibatch_mode(update_mode):
                    edit_budget = None
                    ranked_patch = merged_patch
                    ranked_items = merged_items
                    n_edits_ranked = len(ranked_items)
                    step_rec["n_edits_ranked"] = n_edits_ranked
                    step_rec["edit_budget"] = None
                    step_rec["lr_control_mode"] = "none"
                    with open(os.path.join(step_dir, "ranked_edits.json"), "w") as f:
                        json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                else:
                    if lr_control_mode == "autonomous":
                        lr_decision = decide_autonomous_learning_rate(
                            skill_content=proposal_skill,
                            merged_patch=merged_patch,
                            update_mode=update_mode,
                            rollout_hard=agg_hard,
                            rollout_soft=agg_soft,
                            rollout_n=total_n,
                            step_buffer_context=step_buffer_context,
                            meta_skill_context=active_meta_skill,
                        )
                        edit_budget = int(lr_decision["learning_rate"])
                        with open(os.path.join(step_dir, "lr_decision.json"), "w") as f:
                            json.dump(lr_decision, f, ensure_ascii=False, indent=2)
                        with open(os.path.join(out_root, "lr_history.jsonl"), "a") as f:
                            f.write(json.dumps({
                                "step": global_step,
                                "epoch": epoch,
                                **lr_decision,
                            }, ensure_ascii=False) + "\n")
                    else:
                        edit_budget = scheduler.step()
                    epoch_edit_budget = edit_budget
                    ranked_patch = rank_and_select(
                        proposal_skill, merged_patch,
                        max_edits=edit_budget,
                        update_mode=update_mode,
                        meta_skill_context=active_meta_skill,
                        feedback_context=proposal_regularization_feedback_context,
                    )
                    with open(os.path.join(step_dir, "ranked_edits.json"), "w") as f:
                        json.dump(ranked_patch, f, ensure_ascii=False, indent=2)

                    ranked_items = get_payload_items(ranked_patch, update_mode)
                    n_edits_ranked = len(ranked_items)
                    step_rec["n_edits_ranked"] = n_edits_ranked
                    step_rec["edit_budget"] = edit_budget
                    step_rec["lr_control_mode"] = lr_control_mode
                    if lr_decision is not None:
                        step_rec["lr_decision"] = lr_decision
                step_rec["timing"]["select_s"] = round(time.time() - t_phase, 1)

                support_counts = [
                    item.get("support_count", 0) for item in ranked_items if isinstance(item, dict)
                ]
                step_rec["support_counts"] = support_counts
                if is_full_rewrite_minibatch_mode(update_mode):
                    print(
                        f"    [4/6 SELECT] skipped LR/select; "
                        f"using {n_edits_ranked} merged {payload_label(update_mode)}"
                    )
                else:
                    print(
                        f"    [4/6 SELECT] "
                        f"{n_edits_merged} -> {n_edits_ranked} {payload_label(update_mode)} "
                        f"(budget={edit_budget}, lr_control={lr_control_mode})"
                    )

                # ⑤ UPDATE ─────────────────────────────────────────────────
                t_phase = time.time()
                rewrite_result = None
                if update_mode == "rewrite_from_suggestions":
                    rewrite_result = rewrite_skill_from_suggestions(
                        proposal_skill,
                        ranked_patch,
                        step_buffer_context=step_buffer_context,
                        env=cfg.get("env"),
                        reasoning_effort=rewrite_reasoning_effort,
                        max_completion_tokens=rewrite_max_completion_tokens,
                    )
                    if rewrite_result and rewrite_result.get("new_skill"):
                        candidate_skill = rewrite_result["new_skill"]
                        apply_report = []
                        with open(os.path.join(step_dir, "rewrite_result.json"), "w") as f:
                            json.dump(rewrite_result, f, ensure_ascii=False, indent=2)
                    else:
                        candidate_skill = current_skill
                        apply_report = []
                elif is_full_rewrite_minibatch_mode(update_mode):
                    skill_candidates = get_payload_items(ranked_patch, update_mode)
                    selected_candidate = next(
                        (
                            item for item in skill_candidates
                            if isinstance(item, dict) and str(item.get("new_skill", "")).strip()
                        ),
                        None,
                    )
                    if selected_candidate:
                        candidate_skill = str(selected_candidate["new_skill"]).rstrip() + "\n"
                        apply_report = []
                        rewrite_result = {
                            "reasoning": ranked_patch.get("reasoning", ""),
                            "change_summary": selected_candidate.get("change_summary", []),
                            "title": selected_candidate.get("title", ""),
                            "source_type": selected_candidate.get("source_type", ""),
                        }
                        with open(os.path.join(step_dir, "full_rewrite_result.json"), "w") as f:
                            json.dump(
                                {
                                    "selected_candidate": selected_candidate,
                                    "merged_patch": ranked_patch,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                    else:
                        candidate_skill = current_skill
                        apply_report = []
                else:
                    # Reflection/ranking and the training rollout see
                    # ``proposal_skill`` with selected atoms hidden, while the
                    # resulting patch is committed to the complete incumbent.
                    # Applying it to proposal_skill would silently turn every
                    # mask into a persistent deletion.
                    candidate_skill, apply_report = apply_patch_with_report(
                        current_skill, ranked_patch,
                    )
                if regularization is not None:
                    candidate_skill, finalize_report = regularization.finalize_candidate(
                        candidate_skill, dropout_report=dropout_report
                    )
                    preservation_errors = masked_atom_preservation_errors(
                        candidate_skill, dropout_report,
                    )
                    finalize_report["preservation_errors"] = preservation_errors
                    finalize_report["preservation_ok"] = not preservation_errors
                    if preservation_errors:
                        # Fail safely before Val/CCV. This is a defensive
                        # invariant for future update-mode refactors.
                        candidate_skill = current_skill
                        step_rec["dropout_invariant_failure"] = preservation_errors
                    dropout_report.update(finalize_report)
                    step_rec["regularization_dropout"].update(finalize_report)
                    with open(
                        os.path.join(
                            step_dir, "dropout_candidate_finalize.json"
                        ),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(
                            finalize_report,
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                document_errors = validate_skill_document(candidate_skill)
                step_rec["candidate_document_validation"] = {
                    "ok": not document_errors,
                    "errors": document_errors,
                }
                if document_errors:
                    # Ordinary improve/rewrite proposals must obey the same
                    # persistence invariants as Regularization repairs.  Reject the
                    # malformed candidate before Val/CCV and retain the
                    # incumbent byte-for-byte.
                    candidate_skill = current_skill
                    step_rec["candidate_document_invalid"] = document_errors
                with open(os.path.join(step_dir, "candidate_skill.md"), "w") as f:
                    f.write(candidate_skill)
                if apply_report:
                    with open(os.path.join(step_dir, "edit_apply_report.json"), "w") as f:
                        json.dump(apply_report, f, indent=2, ensure_ascii=False)

                cand_hash = skill_hash(candidate_skill)
                step_rec["candidate_hash"] = cand_hash
                step_rec["candidate_skill_len"] = len(candidate_skill)
                if rewrite_result:
                    step_rec["rewrite_change_summary"] = rewrite_result.get("change_summary", [])
                if update_mode == "patch":
                    step_rec["edit_apply_summary"] = {
                        "total": len(apply_report),
                        "applied": sum(
                            1 for row in apply_report if str(row.get("status", "")).startswith("applied")
                        ),
                        "skipped": sum(
                            1 for row in apply_report if str(row.get("status", "")).startswith("skipped")
                        ),
                        "errors": sum(
                            1 for row in apply_report if row.get("status") == "error"
                        ),
                    }
                step_rec["timing"]["update_s"] = round(time.time() - t_phase, 1)
                applied_edit_count = int(
                    (step_rec.get("edit_apply_summary") or {}).get("applied", 0)
                )
                no_effective_patch = (
                    update_mode == "patch"
                    and (n_edits_ranked == 0 or applied_edit_count == 0)
                )
                if no_effective_patch or cand_hash == skill_hash(current_skill):
                    # Do not let an empty/failed patch turn Dropout masking into
                    # a standalone skill deletion, and do not spend Val/CCV
                    # budget on a candidate identical to the incumbent.
                    candidate_skill = current_skill
                    cand_hash = skill_hash(current_skill)
                    step_rec["candidate_hash"] = cand_hash
                    step_rec["candidate_skill_len"] = len(current_skill)
                    step_rec["action"] = "no_op"
                    step_rec["no_op_reason"] = (
                        "zero_applied_edits"
                        if no_effective_patch
                        else "candidate_unchanged"
                    )
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["skill_len"] = len(current_skill)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    if active_regularization_feedback:
                        # A structural no-op did not test the repair direction;
                        # keep the same answer-free feedback alive for one more
                        # attempt, including across process resume.
                        step_rec["regularization_feedback_next_step"] = active_regularization_feedback
                        if regularization_feedback_consumption_mode == "local_edit":
                            local_edit_fallback_feedback = active_regularization_feedback
                        else:
                            pending_regularization_feedback = active_regularization_feedback
                    with open(os.path.join(step_dir, "candidate_skill.md"), "w") as f:
                        f.write(current_skill)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_skill(out_root, global_step, current_skill)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [no-op] zero effective edits; skipped Val and CCV")
                    continue
                if (
                    update_mode == "rewrite_from_suggestions"
                    and rewrite_result is None
                ) or (
                    is_full_rewrite_minibatch_mode(update_mode)
                    and rewrite_result is None
                ):
                    # Skill-aware: flush appendix notes before skipping (see
                    # the skip_no_patches branch above).
                    if use_skill_aware:
                        current_skill = _flush_skill_aware_appendix(
                            current_skill, all_raw_patches, step_rec, step_dir, cfg,
                        )
                    step_rec["action"] = "skip_no_rewrite"
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["skill_len"] = len(current_skill)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_skill(out_root, global_step, current_skill)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [skip] no usable rewrite generated — skill unchanged")
                    continue
                print(
                    f"    [5/6 UPDATE] "
                    f"skill_len {len(current_skill)} -> {len(candidate_skill)}"
                )

                # New Regularization protocol: form the complete final candidate before
                # spending any full-Val budget. Dropout/complexity consume the
                # current->candidate delta first; CCV then clean-qualifies one
                # Train source and evaluates one paired attack. Only the final
                # repaired candidate reaches paired validation below.
                step_local_ccv = None
                step_local_ccv_schedule = None
                if regularization_step_local_pipeline and regularization is not None:
                    pre_feedback, pre_telemetry = (
                        _build_step_local_preselection_feedback(
                            regularization,
                            old_skill=current_skill,
                            candidate_skill=candidate_skill,
                            dropout_report=dropout_report,
                        )
                    )
                    step_rec["regularization_step_local_preselection"] = pre_telemetry
                    if pre_feedback is not None:
                        pre_result = _run_regularization_local_edit(
                            candidate_skill,
                            pre_feedback,
                            anchor_skill=current_skill,
                            out_dir=os.path.join(
                                step_dir, "regularization_preselection_local_edit"
                            ),
                            reasoning_effort=rewrite_reasoning_effort,
                            max_completion_tokens=rewrite_max_completion_tokens,
                            embedding_model=regularization_embedding_model,
                            embedding_match_threshold=(
                                regularization_embedding_match_threshold
                            ),
                        )
                        step_rec["regularization_preselection_local_edit"] = {
                            key: value for key, value in pre_result.items()
                            if key != "new_skill"
                        }
                        if pre_result.get("status") == "ok":
                            candidate_skill = pre_result["new_skill"]
                    step_local_ccv, step_local_ccv_schedule = (
                        regularization.run_step_ccv(
                            old_skill=current_skill,
                            candidate_skill=candidate_skill,
                            global_step=global_step,
                            seed=seed + global_step * 130363,
                            out_dir=os.path.join(step_dir, "regularization_step_ccv"),
                            trajectory_context={
                                "current_step_rollouts": all_rollout_results,
                                "prior_step_records": history[
                                    -regularization.ccv_history_items:
                                ],
                                "ranked_edits": ranked_items,
                                "ranked_patch": ranked_patch,
                                "apply_report": apply_report,
                                "dropout_report": dropout_report,
                            },
                        )
                    )
                    step_rec["regularization_step_ccv"] = {
                        "schedule": step_local_ccv_schedule,
                        "result": step_local_ccv.to_dict(),
                    }
                    ccv_feedback = _build_step_local_ccv_feedback(
                        step_local_ccv.to_dict()
                    )
                    if ccv_feedback is not None:
                        ccv_result = _run_regularization_local_edit(
                            candidate_skill,
                            ccv_feedback,
                            anchor_skill=current_skill,
                            out_dir=os.path.join(
                                step_dir, "regularization_ccv_local_edit"
                            ),
                            reasoning_effort=rewrite_reasoning_effort,
                            max_completion_tokens=rewrite_max_completion_tokens,
                            embedding_model=regularization_embedding_model,
                            embedding_match_threshold=(
                                regularization_embedding_match_threshold
                            ),
                        )
                        step_rec["regularization_ccv_local_edit"] = {
                            key: value for key, value in ccv_result.items()
                            if key != "new_skill"
                        }
                        if ccv_result.get("status") == "ok":
                            candidate_skill = ccv_result["new_skill"]
                    cand_hash = skill_hash(candidate_skill)
                    step_rec["candidate_hash"] = cand_hash
                    step_rec["candidate_skill_len"] = len(candidate_skill)
                    step_rec["final_candidate_formed_before_validation"] = True
                    with open(
                        os.path.join(step_dir, "candidate_skill.md"), "w"
                    ) as handle:
                        handle.write(candidate_skill)

                # ⑥ EVALUATE ───────────────────────────────────────────────
                t_phase = time.time()
                selection_trajectories: list[dict] = []
                paired_current_gate_score = current_score
                if paired_selection_repeats >= 2:
                    candidate_repeat_scores: list[dict] = []
                    incumbent_repeat_scores: list[dict] = []
                    print(
                        f"    [6/6 EVALUATE] paired full-selection repeats="
                        f"{paired_selection_repeats} (candidate + incumbent)"
                    )
                    for repeat_index in range(1, paired_selection_repeats + 1):
                        # Use the same item seed within each pair. Model calls
                        # remain independent, while candidate and incumbent see
                        # exactly the same validation examples.
                        pair_seed = seed + global_step * 1009 + repeat_index
                        candidate_env, sel_n = _build_eval_env(
                            split="valid_seen",
                            env_num=cfg["sel_env_num"],
                            seed=pair_seed,
                        )
                        candidate_dir = os.path.join(
                            step_dir, "selection_eval_paired",
                            f"repeat_{repeat_index:02d}", "candidate",
                        )
                        candidate_results = adapter.rollout(
                            candidate_env, candidate_skill, candidate_dir
                        )
                        selection_trajectories.extend(candidate_results)
                        repeat_hard, repeat_soft = compute_score(candidate_results)
                        candidate_repeat_scores.append({
                            "repeat": repeat_index, "seed": pair_seed,
                            "hard": repeat_hard, "soft": repeat_soft,
                            "n_items": sel_n,
                        })

                        incumbent_env, incumbent_n = _build_eval_env(
                            split="valid_seen",
                            env_num=cfg["sel_env_num"],
                            seed=pair_seed,
                        )
                        incumbent_dir = os.path.join(
                            step_dir, "selection_eval_paired",
                            f"repeat_{repeat_index:02d}", "incumbent",
                        )
                        incumbent_results = adapter.rollout(
                            incumbent_env, current_skill, incumbent_dir
                        )
                        incumbent_hard, incumbent_soft = compute_score(
                            incumbent_results
                        )
                        incumbent_repeat_scores.append({
                            "repeat": repeat_index, "seed": pair_seed,
                            "hard": incumbent_hard, "soft": incumbent_soft,
                            "n_items": incumbent_n,
                        })
                        print(
                            f"      pair {repeat_index}/{paired_selection_repeats}: "
                            f"candidate={repeat_hard:.4f} "
                            f"incumbent={incumbent_hard:.4f}"
                        )

                    cand_hard, cand_soft = _mean_repeat_scores(
                        candidate_repeat_scores
                    )
                    incumbent_hard, incumbent_soft = _mean_repeat_scores(
                        incumbent_repeat_scores
                    )
                    paired_current_gate_score = select_gate_score(
                        incumbent_hard, incumbent_soft, gate_metric,
                        gate_mixed_weight, skill_content=current_skill,
                        use_semantic_density=use_semantic_density,
                        semantic_density_weight=semantic_density_weight,
                        leading_words=leading_words,
                    )
                    step_rec["paired_selection"] = {
                        "repeats": paired_selection_repeats,
                        "candidate": candidate_repeat_scores,
                        "incumbent": incumbent_repeat_scores,
                        "candidate_mean": {
                            "hard": cand_hard, "soft": cand_soft,
                        },
                        "incumbent_mean": {
                            "hard": incumbent_hard, "soft": incumbent_soft,
                            "gate_score": paired_current_gate_score,
                        },
                    }
                elif cand_hash in sel_cache:
                    cand_hard, cand_soft = sel_cache[cand_hash]
                    print(
                        f"    [6/6 EVALUATE] "
                        f"cache hit {cand_hash}: hard={cand_hard:.4f}"
                    )
                else:
                    sel_env, sel_n = _build_eval_env(
                        split="valid_seen",
                        env_num=cfg["sel_env_num"],
                        seed=seed,
                    )
                    print(f"    [6/6 EVALUATE] selection items={sel_n}")
                    sel_eval_dir = os.path.join(step_dir, "selection_eval")
                    sel_results = adapter.rollout(sel_env, candidate_skill, sel_eval_dir)
                    selection_trajectories = sel_results
                    cand_hard, cand_soft = compute_score(sel_results)
                    sel_cache[cand_hash] = (cand_hard, cand_soft)

                step_rec["selection_hard"] = cand_hard
                step_rec["selection_soft"] = cand_soft
                if paired_selection_repeats >= 2:
                    step_rec["incumbent_selection_hard"] = incumbent_hard
                    step_rec["incumbent_selection_soft"] = incumbent_soft

                cand_gate_score = select_gate_score(
                    cand_hard, cand_soft, gate_metric, gate_mixed_weight,
                    skill_content=candidate_skill,
                    use_semantic_density=use_semantic_density,
                    semantic_density_weight=semantic_density_weight,
                    leading_words=leading_words,
                )
                regularization_telemetry = None
                local_edit_feedback = None
                if regularization is not None:
                    gate, regularization_telemetry = regularization.evaluate(
                        old_skill=current_skill,
                        candidate_skill=candidate_skill,
                        candidate_score=cand_gate_score,
                        current_score=paired_current_gate_score,
                        best_skill=best_skill,
                        best_score=best_score,
                        best_step=best_step,
                        global_step=global_step,
                        seed=seed + global_step * 130363,
                        out_dir=step_dir,
                        trajectory_context={
                            "current_step_rollouts": all_rollout_results,
                            "selection_rollouts": selection_trajectories,
                            "prior_step_records": history[-regularization.ccv_history_items:],
                            "ranked_edits": ranked_items,
                            "ranked_patch": ranked_patch,
                            "apply_report": apply_report,
                            "dropout_report": dropout_report,
                        },
                        run_ccv=not regularization_step_local_pipeline,
                    )
                elif use_gate:
                    gate = evaluate_gate(
                        candidate_skill=candidate_skill,
                        cand_hard=cand_hard,
                        current_skill=current_skill,
                        current_score=paired_current_gate_score,
                        best_skill=best_skill,
                        best_score=best_score,
                        best_step=best_step,
                        global_step=global_step,
                        cand_soft=cand_soft,
                        metric=gate_metric,
                        mixed_weight=gate_mixed_weight,
                        use_semantic_density=use_semantic_density,
                        semantic_density_weight=semantic_density_weight,
                        leading_words=leading_words,
                    )
                else:
                    # Validation ran (scores recorded above) but the gate is
                    # disabled: force-accept the candidate as the new current
                    # skill. Best-so-far is still tracked for convenience; the
                    # final skill is selected manually from the trajectory.
                    if cand_gate_score > best_score:
                        fa_best_skill = candidate_skill
                        fa_best_score = cand_gate_score
                        fa_best_step = global_step
                    else:
                        fa_best_skill = best_skill
                        fa_best_score = best_score
                        fa_best_step = best_step
                    gate = GateResult(
                        action="force_accept",
                        current_skill=candidate_skill,
                        current_score=cand_gate_score,
                        best_skill=fa_best_skill,
                        best_score=fa_best_score,
                        best_step=fa_best_step,
                    )
                step_rec["gate_metric"] = gate_metric
                step_rec["candidate_gate_score"] = cand_gate_score
                action_label = gate.action
                if regularization_telemetry is not None:
                    decision = regularization_telemetry["decision"]
                    action_label = decision["action"]
                    growth = regularization_telemetry["complexity"]["growth"]
                    reported_ccv = (
                        step_local_ccv.to_dict()
                        if regularization_step_local_pipeline
                        and step_local_ccv is not None
                        else regularization_telemetry["ccv"]
                    )
                    step_rec["skillevoreg"] = {
                        "decision": decision,
                        "complexity": {
                            "initial": regularization_telemetry["complexity"]["initial"]["total"],
                            "old": regularization_telemetry["complexity"]["old"]["total"],
                            "candidate": regularization_telemetry["complexity"]["candidate"]["total"],
                            **growth,
                        },
                        "ccv": {
                            key: reported_ccv.get(key, 0)
                            for key in (
                                "status", "generated_attacks", "valid_attacks",
                                "materialized_attacks", "validator_passed_attacks",
                                "old_qualified_attacks", "paired_evaluated_attacks",
                                "confirmed_regressions",
                                "regressions", "improvements",
                                "transport_failures",
                                "inconclusive_failures",
                                "qualified_sources",
                                "source_qualification_failures",
                                "oar",
                            )
                        },
                        "ccv_schedule": regularization_telemetry.get(
                            "ccv_schedule", {}
                        ),
                    }
                    feedback = None if regularization_step_local_pipeline else regularization.build_feedback(
                        regularization_telemetry,
                        dropout_report=dropout_report,
                        edit_budget=edit_budget,
                        current_skill=current_skill,
                        candidate_skill=candidate_skill,
                    )
                    if feedback is not None:
                        if regularization_feedback_consumption_mode == "local_edit":
                            step_rec["regularization_feedback_local_edit"] = feedback
                            local_edit_feedback = feedback
                        elif regularization.epoch_agents_enabled:
                            step_rec["regularization_feedback_diagnostic"] = feedback
                        else:
                            step_rec["regularization_feedback_next_step"] = feedback
                            pending_regularization_feedback = feedback
                step_rec["action"] = action_label
                prev_current = current_score
                prev_best = best_score
                pre_update_anchor_skill = current_skill
                current_skill = gate.current_skill
                current_score = gate.current_score
                best_skill = gate.best_skill
                best_score = gate.best_score
                best_step = gate.best_step
                if gate.action in {"accept", "accept_new_best", "force_accept"}:
                    current_origin = f"step_{global_step:04d}"
                if gate.action == "accept_new_best" or (
                    gate.action == "force_accept" and best_step == global_step
                ):
                    best_origin = current_origin

                if local_edit_feedback is not None:
                    # Regularization findings describe this just-evaluated candidate.
                    # Repair that candidate directly; a rejection must not
                    # redirect candidate-specific compression onto the old
                    # incumbent.
                    quick_result = _run_regularization_local_edit(
                        candidate_skill,
                        local_edit_feedback,
                        anchor_skill=pre_update_anchor_skill,
                        out_dir=os.path.join(step_dir, "regularization_local_edit"),
                        reasoning_effort=rewrite_reasoning_effort,
                        max_completion_tokens=rewrite_max_completion_tokens,
                        embedding_model=regularization_embedding_model,
                        embedding_match_threshold=regularization_embedding_match_threshold,
                    )
                    step_rec["regularization_local_edit"] = {
                        key: value for key, value in quick_result.items()
                        if key != "new_skill"
                    }
                    step_rec["regularization_local_edit"]["source"] = "candidate_skill"
                    step_rec["regularization_local_edit"]["source_hash"] = cand_hash
                    if quick_result.get("status") == "ok":
                        current_skill = quick_result["new_skill"]
                        # No second evaluation by design.  Carry the repaired
                        # candidate's pre-repair score only as a provisional
                        # runtime value; paired evaluation remeasures it at the
                        # next step.  Best-so-far remains evaluation-backed.
                        current_score = cand_gate_score
                        if quick_result.get("noop"):
                            current_origin = f"step_{global_step:04d}_quick_noop"
                            step_rec["action"] = f"{action_label}_quick_noop"
                            print(
                                "    [Regularization local edit] NOOP kept the source "
                                "delta without a second evaluation"
                            )
                        else:
                            current_origin = f"step_{global_step:04d}_local_edit"
                            step_rec["action"] = f"{action_label}_local_editd"
                            print(
                                "    [Regularization local edit] applied without a second "
                                "evaluation"
                            )
                    else:
                        local_edit_fallback_feedback = local_edit_feedback
                        step_rec["regularization_feedback_next_step"] = local_edit_feedback
                        print(
                            "    [Regularization local edit] failed; preserved the same "
                            "feedback for the next normal step"
                        )

                if use_skill_aware:
                    current_skill = _flush_skill_aware_appendix(
                        current_skill, all_raw_patches, step_rec, step_dir, cfg,
                    )

                if gate_metric == "hard":
                    score_label = f"hard={cand_hard:.4f}"
                elif gate_metric == "soft":
                    score_label = f"soft={cand_soft:.4f}"
                else:
                    score_label = (
                        f"mixed[w={gate_mixed_weight}]={cand_gate_score:.4f} "
                        f"(hard={cand_hard:.4f} soft={cand_soft:.4f})"
                    )
                if paired_selection_repeats >= 2:
                    comparison_label = (
                        f"paired incumbent={paired_current_gate_score:.4f}"
                    )
                else:
                    comparison_label = f"current={prev_current:.4f}"
                if gate.action == "accept_new_best":
                    print(
                        f"    [6/6 EVALUATE] ACCEPT (new best) "
                        f"{score_label} > prev best {prev_best:.4f}"
                    )
                elif gate.action == "accept":
                    print(
                        f"    [6/6 EVALUATE] ACCEPT "
                        f"{score_label} > {comparison_label}"
                    )
                elif gate.action == "force_accept":
                    print(
                        f"    [6/6 EVALUATE] FORCE-ACCEPT (gate disabled) "
                        f"{score_label}"
                    )
                else:
                    regularization_reason = ""
                    if regularization_telemetry is not None:
                        regularization_reason = (
                            " reasons="
                            + ",".join(regularization_telemetry["decision"]["reasons"])
                        )
                    print(
                        f"    [6/6 EVALUATE] REJECT "
                        f"{score_label} {comparison_label}"
                        f"{regularization_reason}"
                    )

                step_rec["timing"]["evaluate_s"] = round(time.time() - t_phase, 1)

                # ── Step buffer: unified failure patterns + rejected edits ─
                action = step_rec.get("action", "unknown")
                n_total = len(all_rollout_results) or 1
                n_fail = sum(1 for r in all_rollout_results if not r.get("hard") or float(r.get("hard", 0)) < 1e-9)
                failure_patterns = _extract_failure_patterns(
                    all_rollout_results, step_dir,
                )

                buf_entry: dict = {
                    "step": global_step,
                    "action": action,
                    "n_total": n_total,
                    "n_fail": n_fail,
                    "failure_patterns": failure_patterns,
                }

                # Attach rejected edits when the step was rejected
                if "reject" in action and ranked_patch:
                    rejected_edits = [
                        short_item_summary(item, update_mode)
                        for item in ranked_items
                        if isinstance(item, dict)
                    ]
                    buf_entry["score_before"] = current_score
                    buf_entry["score_after"] = cand_gate_score
                    buf_entry["rejected_edits"] = rejected_edits

                step_buffer.append(buf_entry)

                # Persist step digest for step buffer context
                digest_path = os.path.join(step_dir, "trajectory_digest.json")
                with open(digest_path, "w") as f:
                    json.dump(buf_entry, f, indent=2, ensure_ascii=False)

                # ── Token snapshot ───────────────────────────────────────
                tokens_after = get_token_summary()
                step_tokens: dict = {}
                for stage in tokens_after:
                    if stage == "_total":
                        continue
                    after = tokens_after[stage]
                    before = tokens_before.get(stage, {})
                    step_tokens[stage] = {
                        "calls": after.get("calls", 0) - before.get("calls", 0),
                        "prompt_tokens": after.get("prompt_tokens", 0)
                        - before.get("prompt_tokens", 0),
                        "completion_tokens": after.get("completion_tokens", 0)
                        - before.get("completion_tokens", 0),
                        "cached_tokens": after.get("cached_tokens", 0)
                        - before.get("cached_tokens", 0),
                        "reasoning_tokens": after.get("reasoning_tokens", 0)
                        - before.get("reasoning_tokens", 0),
                    }
                step_rec["tokens"] = step_tokens

                # ── Save state ───────────────────────────────────────────
                step_rec["current_score"] = current_score
                step_rec["best_score"] = best_score
                step_rec["best_step"] = best_step
                step_rec["current_origin"] = current_origin
                step_rec["best_origin"] = best_origin
                step_rec["skill_len"] = len(current_skill)
                step_rec["wall_time_s"] = round(time.time() - step_t0, 1)

                _save_skill(out_root, global_step, current_skill)
                with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                    f.write(best_skill)
                history.append(step_rec)
                _save_history(out_root, history)
                _persist_runtime_state(global_step)
                with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                    json.dump(step_rec, f, indent=2, ensure_ascii=False)

                timing = step_rec["timing"]
                print(
                    f"\n  [STEP {global_step} done] "
                    f"epoch={epoch} action={step_rec['action']} "
                    f"current={current_score:.4f} best={best_score:.4f} "
                    f"dt={step_rec['wall_time_s']}s\n"
                    f"    timing: rollout={timing.get('rollout_s',0)}s "
                    f"reflect={timing.get('reflect_s',0)}s "
                    f"aggregate={timing.get('aggregate_s',0)}s "
                    f"select={timing.get('select_s',0)}s "
                    f"evaluate={timing.get('evaluate_s',0)}s"
                )

            # ── EPOCH TRANSACTION: cumulative CCV then semantic feedback ──
            if (
                regularization is not None and regularization.epoch_agents_enabled
                and not regularization_step_local_pipeline
                and not epoch_already_complete
                and int(epoch_transaction.get("epoch", 0) or 0) == epoch
            ):
                os.makedirs(epoch_agent_dir, exist_ok=True)
                trajectories, epoch_step_records = _load_epoch_agent_evidence(
                    out_root, epoch_first_step, epoch_last_step
                )
                if epoch_transaction.get("status") == "training":
                    with open(epoch_start_snapshot["current_skill_path"], encoding="utf-8") as handle:
                        start_skill = handle.read()
                    tentative_skill = current_skill
                    tentative_score = current_score
                    print(f"\n  [EPOCH {epoch} CCV AGENT] cumulative delta over steps {epoch_first_step}-{epoch_last_step}")
                    ccv_result, ccv_schedule = regularization.run_epoch_ccv(
                        old_skill=start_skill, candidate_skill=tentative_skill,
                        global_step=epoch_last_step, seed=seed + epoch * 32452843,
                        out_dir=os.path.join(epoch_agent_dir, "ccv"),
                        trajectory_context={
                            "current_step_rollouts": trajectories,
                            "prior_step_records": epoch_step_records,
                            "epoch": epoch,
                        },
                    )
                    rollback_reasons = []
                    if ccv_result.regressions > 0:
                        rollback_reasons.append("ccv_update_regression")
                    if (
                        regularization.ccv_required
                        and ccv_result.status in {"error", "inconclusive"}
                    ):
                        rollback_reasons.append("ccv_inconclusive")
                    if regularization.ccv_required and ccv_result.status == "provider_unavailable":
                        rollback_reasons.append("ccv_provider_unavailable")
                    committed = (
                        not rollback_reasons
                        or regularization_feedback_consumption_mode == "local_edit"
                    )
                    if not committed:
                        with open(epoch_start_snapshot["current_skill_path"], encoding="utf-8") as handle:
                            current_skill = handle.read()
                        with open(epoch_start_snapshot["best_skill_path"], encoding="utf-8") as handle:
                            best_skill = handle.read()
                        current_score = float(epoch_start_snapshot["current_score"])
                        current_origin = str(epoch_start_snapshot["current_origin"])
                        best_score = float(epoch_start_snapshot["best_score"])
                        best_step = epoch_start_snapshot["best_step"]
                        best_origin = str(epoch_start_snapshot["best_origin"])
                    _save_skill(out_root, epoch_last_step, current_skill)
                    with open(os.path.join(out_root, "best_skill.md"), "w", encoding="utf-8") as handle:
                        handle.write(best_skill)
                    decision = {
                        "epoch": epoch, "committed": committed,
                        "action": (
                            "local_edit_pending"
                            if rollback_reasons
                            and regularization_feedback_consumption_mode == "local_edit"
                            else ("commit" if committed else "rollback")
                        ),
                        "reasons": rollback_reasons,
                        "start_origin": epoch_start_snapshot.get("current_origin"),
                        "tentative_end_origin": f"step_{epoch_last_step:04d}",
                        "committed_origin": current_origin,
                        "start_score": float(epoch_start_snapshot["current_score"]),
                        "tentative_score": float(tentative_score),
                        "ccv_schedule": ccv_schedule,
                        "ccv": ccv_result.to_dict(),
                    }
                    with open(os.path.join(epoch_agent_dir, "decision.json"), "w", encoding="utf-8") as handle:
                        json.dump(decision, handle, ensure_ascii=False, indent=2)
                    epoch_transaction.update({
                        "status": "ccv_decided", "decision": decision,
                        "tentative_skill_path": os.path.join(epoch_agent_dir, "epoch_tentative_skill.md"),
                    })
                    with open(epoch_transaction["tentative_skill_path"], "w", encoding="utf-8") as handle:
                        handle.write(tentative_skill)
                    _persist_runtime_state(epoch_last_step)
                else:
                    decision = dict(epoch_transaction.get("decision") or {})
                    tentative_path = epoch_transaction.get("tentative_skill_path")
                    tentative_skill = current_skill
                    if tentative_path and os.path.exists(tentative_path):
                        with open(tentative_path, encoding="utf-8") as handle:
                            tentative_skill = handle.read()

                if epoch_transaction.get("status") == "ccv_decided":
                    start_path = epoch_start_snapshot.get("current_skill_path")
                    with open(start_path, encoding="utf-8") as handle:
                        start_skill = handle.read()
                    epoch_module_feedback, epoch_module_telemetry = (
                        regularization.build_epoch_module_feedback(
                            old_skill=start_skill,
                            candidate_skill=tentative_skill,
                            current_score=float(decision.get(
                                "start_score",
                                epoch_start_snapshot.get("current_score", 0.0),
                            )),
                            candidate_score=float(decision.get(
                                "tentative_score", current_score,
                            )),
                            global_step=epoch_last_step,
                            ccv=coerce_ccv_result(decision.get("ccv", {})),
                            edit_budget=epoch_edit_budget,
                        )
                    )
                    decision["regularization_module_telemetry"] = epoch_module_telemetry
                    decision["abnormal_modules"] = list(
                        (epoch_module_feedback or {}).get(
                            "abnormal_modules", []
                        )
                    )
                    if epoch_module_feedback:
                        decision["action"] = (
                            "local_edit_pending"
                            if regularization_feedback_consumption_mode == "local_edit"
                            else "deferred_feedback_pending"
                        )
                    epoch_transaction["decision"] = decision
                    with open(
                        os.path.join(epoch_agent_dir, "decision.json"),
                        "w", encoding="utf-8",
                    ) as handle:
                        json.dump(decision, handle, ensure_ascii=False, indent=2)
                    module_feedback_path = os.path.join(
                        epoch_agent_dir, "module_feedback.json",
                    )
                    with open(module_feedback_path, "w", encoding="utf-8") as handle:
                        json.dump({
                            "status": (
                                "actionable" if epoch_module_feedback
                                else "no_regularization_anomaly"
                            ),
                            "feedback": epoch_module_feedback,
                            "telemetry": epoch_module_telemetry,
                        }, handle, ensure_ascii=False, indent=2)

                    feedback = None
                    should_consume_this_epoch = bool(
                        epoch_module_feedback
                        and (
                            epoch < num_epochs
                            or regularization_feedback_consumption_mode == "local_edit"
                        )
                    )
                    if should_consume_this_epoch:
                        feedback_dir = os.path.join(epoch_agent_dir, "feedback")
                        completed_feedback_path = os.path.join(feedback_dir, "feedback.json")
                        analyst_feedback = None
                        if os.path.exists(completed_feedback_path):
                            try:
                                with open(completed_feedback_path, encoding="utf-8") as handle:
                                    analyst_feedback = json.load(handle)
                            except Exception:
                                analyst_feedback = None
                        if isinstance(analyst_feedback, dict) and analyst_feedback.get("status") == "ok":
                            print(f"  [EPOCH {epoch} FEEDBACK AGENT] resumed validated result")
                        else:
                            start_complexity = regularization.measure(start_skill).to_dict()
                            tentative_complexity = regularization.measure(tentative_skill).to_dict()
                            dropout_summary = {
                                "steps": len(epoch_step_records),
                                "masked_atoms": sum(int((r.get("dropout") or {}).get("masked_count", 0) or 0) for r in epoch_step_records),
                            }
                            print(f"  [EPOCH {epoch} FEEDBACK AGENT] trajectories={len(trajectories)}")
                            analyst_feedback = regularization.run_epoch_feedback(
                                evidence={
                                    "epoch": epoch, "train_trajectories": trajectories,
                                    "step_records": epoch_step_records,
                                    "transaction": decision,
                                    "actionable_guard_feedback": epoch_module_feedback,
                                    "skill_delta": _skill_delta(start_skill, tentative_skill),
                                    "complexity": {"start": start_complexity, "tentative_end": tentative_complexity},
                                    "ccv": decision.get("ccv", {}),
                                    "dropout_summary": dropout_summary,
                                },
                                out_dir=feedback_dir,
                            )
                        if regularization.epoch_feedback_required and analyst_feedback.get("status") != "ok":
                            epoch_transaction["feedback_failure"] = {
                                "status": analyst_feedback.get("status"),
                                "error": analyst_feedback.get("error"),
                            }
                            _persist_runtime_state(epoch_last_step)
                            raise RuntimeError(
                                f"Epoch {epoch} required Feedback Agent did not return "
                                f"validated guidance: {analyst_feedback.get('error')}"
                            )
                        feedback = dict(epoch_module_feedback)
                        feedback["status"] = "ok"
                        feedback["author"] = "epoch_feedback_agent"
                        feedback["epoch_feedback_analysis"] = {
                            key: value for key, value in analyst_feedback.items()
                            if key not in {"agent_attempts"}
                        }
                        feedback["semantic_repair_guidance"] = (
                            analyst_feedback.get("next_epoch_guidance", {})
                        )
                        consumable_feedback_path = os.path.join(
                            epoch_agent_dir, "consumable_feedback.json",
                        )
                        with open(consumable_feedback_path, "w", encoding="utf-8") as handle:
                            json.dump(feedback, handle, ensure_ascii=False, indent=2)
                    elif epoch_module_feedback:
                        # Final deferred feedback has no consumer; retain the
                        # diagnosis as telemetry without running an agent.
                        feedback = dict(epoch_module_feedback)
                        feedback["status"] = "skipped_no_future_consumer"
                    else:
                        print(
                            f"  [EPOCH {epoch} FEEDBACK] skipped: no causally "
                            "attributed complexity or confirmed CCV anomaly"
                        )
                    if regularization_feedback_consumption_mode == "local_edit":
                        epoch_feedback_for_steps = None
                        if feedback is not None and feedback.get("status") == "ok":
                            quick_result = _run_regularization_local_edit(
                                tentative_skill,
                                feedback,
                                anchor_skill=start_skill,
                                out_dir=os.path.join(epoch_agent_dir, "local_edit"),
                                reasoning_effort=rewrite_reasoning_effort,
                                max_completion_tokens=rewrite_max_completion_tokens,
                                embedding_model=regularization_embedding_model,
                                embedding_match_threshold=regularization_embedding_match_threshold,
                            )
                            epoch_transaction["local_edit"] = {
                                key: value for key, value in quick_result.items()
                                if key != "new_skill"
                            }
                            epoch_transaction["local_edit"]["source"] = (
                                "epoch_tentative_skill"
                            )
                            if quick_result.get("status") == "ok":
                                current_skill = quick_result["new_skill"]
                                suffix = (
                                    "quick_noop" if quick_result.get("noop")
                                    else "local_edit"
                                )
                                current_origin = f"epoch_{epoch:02d}_{suffix}"
                                _save_skill(out_root, epoch_last_step, current_skill)
                                if quick_result.get("noop"):
                                    decision["action"] = "quick_noop"
                                    print(
                                        f"  [EPOCH {epoch} QUICK UPDATE] NOOP "
                                        "kept the source delta without a second "
                                        "evaluation"
                                    )
                                else:
                                    decision["action"] = "local_editd"
                                    print(
                                        f"  [EPOCH {epoch} QUICK UPDATE] applied "
                                        "without a second evaluation"
                                    )
                            else:
                                local_edit_fallback_feedback = feedback
                                decision["action"] = "local_edit_failed_deferred"
                                print(
                                    f"  [EPOCH {epoch} QUICK UPDATE] failed; "
                                    "feedback preserved for the next normal step"
                                )
                            decision["local_edit"] = dict(
                                epoch_transaction["local_edit"]
                            )
                    else:
                        epoch_feedback_for_steps = (
                            feedback
                            if feedback is not None
                            and feedback.get("status") == "ok"
                            else None
                        )
                        pending_regularization_feedback = epoch_feedback_for_steps
                    epoch_transaction["status"] = "completed"
                    epoch_transaction["decision"] = decision
                    epoch_transaction["module_feedback_path"] = module_feedback_path
                    epoch_transaction["feedback_path"] = (
                        os.path.join(epoch_agent_dir, "consumable_feedback.json")
                        if feedback is not None and feedback.get("status") == "ok"
                        else None
                    )
                    if history and history[-1].get("epoch") == epoch:
                        history[-1].update({
                            "current_score": current_score, "best_score": best_score,
                            "best_step": best_step, "current_origin": current_origin,
                            "best_origin": best_origin, "skill_len": len(current_skill),
                            "epoch_transaction": decision,
                            "epoch_feedback_next_epoch": (
                                epoch_feedback_for_steps
                                if regularization_feedback_consumption_mode != "local_edit"
                                else None
                            ),
                            "epoch_feedback_local_edit": (
                                feedback
                                if regularization_feedback_consumption_mode == "local_edit"
                                and feedback is not None
                                and feedback.get("status") == "ok"
                                else None
                            ),
                        })
                        _save_history(out_root, history)
                        record_path = os.path.join(out_root, "steps", f"step_{epoch_last_step:04d}", "step_record.json")
                        if os.path.exists(record_path):
                            with open(record_path, "w", encoding="utf-8") as handle:
                                json.dump(history[-1], handle, ensure_ascii=False, indent=2)
                    _persist_runtime_state(epoch_last_step)

            epoch_last_step_skill = current_skill
            epoch_comparison_pairs: list[dict] | None = None

            # ── SLOW UPDATE (end of epoch) ──────────────────────────────
            use_slow = cfg.get("use_slow_update", False)
            if use_slow:
                slow_dir = os.path.join(out_root, "slow_update", f"epoch_{epoch:02d}")
                slow_done_path = os.path.join(slow_dir, "slow_result.json")

                if os.path.exists(slow_done_path):
                    # Resume support
                    print(
                        f"\n  [SLOW UPDATE epoch {epoch}] "
                        f"resumed — already done"
                    )
                    with open(slow_done_path) as f:
                        slow_saved = json.load(f)
                    comparison_path = os.path.join(slow_dir, "comparison_pairs.json")
                    if os.path.exists(comparison_path):
                        try:
                            with open(comparison_path) as f:
                                epoch_comparison_pairs = json.load(f)
                        except Exception:
                            epoch_comparison_pairs = None
                    if (
                        slow_saved.get("slow_update_content")
                        and epoch >= 2
                    ):
                        action = slow_saved.get("action")
                        if slow_gate_with_selection:
                            # Gated mode (follow SkillReflection): re-apply the
                            # guidance to current_skill only when it was accepted.
                            if action in {"accept", "accept_new_best"}:
                                current_skill = replace_slow_update_field(
                                    current_skill,
                                    slow_saved["slow_update_content"],
                                )
                        elif action in {
                            "accept", "accept_new_best", "force_accept",
                        }:
                            # Force-accept mode: re-apply guidance to
                            # current_skill only. best_skill must remain a
                            # faithful snapshot of the val-best step and must
                            # NOT receive force-injected slow-update content.
                            current_skill = replace_slow_update_field(
                                current_skill, slow_saved["slow_update_content"],
                            )
                elif epoch == 1:
                    # Epoch 1: inject empty placeholder
                    os.makedirs(slow_dir, exist_ok=True)
                    current_skill = inject_empty_slow_update_field(current_skill)
                    current_origin = f"slow_update_placeholder_epoch_{epoch:02d}"
                    _save_skill(out_root, global_step, current_skill)
                    with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                        f.write(best_skill)
                    with open(slow_done_path, "w") as f:
                        json.dump({"action": "inject_placeholder", "epoch": epoch}, f, indent=2)
                    _persist_runtime_state(global_step)
                    print(
                        f"\n  [SLOW UPDATE epoch {epoch}] "
                        f"injected empty placeholder"
                    )
                else:
                    # Epoch 2+: longitudinal comparison
                    os.makedirs(slow_dir, exist_ok=True)
                    print(
                        f"\n  {'='*60}\n"
                        f"  SLOW UPDATE — Epoch {epoch} "
                        f"(comparing epoch {epoch-1} vs {epoch})\n"
                        f"  {'='*60}"
                    )

                    # 1. Get skill from last step of previous epoch
                    prev_epoch_records = [
                        h for h in history if h.get("epoch") == epoch - 1
                    ]
                    prev_epoch_last_step = prev_epoch_records[-1]["step"]
                    prev_skill = _load_skill(out_root, prev_epoch_last_step)

                    # 2. Sample items from train set
                    slow_n = cfg.get("slow_update_samples", 20)
                    slow_seed = seed + epoch * 2000
                    if dataloader is not None:
                        slow_batch = dataloader.build_train_batch(
                            batch_size=slow_n,
                            seed=slow_seed,
                            out_root=out_root,
                        )
                        slow_env = adapter.build_env_from_batch(
                            slow_batch, out_root=out_root,
                        )
                    else:
                        slow_env = adapter.build_train_env(
                            batch_size=slow_n,
                            seed=slow_seed,
                            out_root=out_root,
                        )
                    slow_items = list(slow_env) if hasattr(slow_env, "__iter__") else slow_env
                    print(f"    [slow update] sampled {len(slow_items)} train items (seed={slow_seed})")

                    # 3. Rollout with both skills
                    t_slow = time.time()
                    prev_rollout_dir = os.path.join(slow_dir, "rollout_prev")
                    curr_rollout_dir = os.path.join(slow_dir, "rollout_curr")
                    results_prev = adapter.rollout(slow_env, prev_skill, prev_rollout_dir)
                    results_curr = adapter.rollout(slow_env, current_skill, curr_rollout_dir)

                    prev_hard, _ = compute_score(results_prev)
                    curr_hard, _ = compute_score(results_curr)
                    print(
                        f"    [slow update] prev epoch hard={prev_hard:.4f}  "
                        f"curr epoch hard={curr_hard:.4f}"
                    )

                    # 4. Build and save structured comparison pairs
                    comparison_pairs, all_comparison_pairs = _build_longitudinal_pairs(
                        adapter=adapter,
                        dataloader=dataloader,
                        prev_skill=prev_skill,
                        curr_skill=current_skill,
                        initial_items=slow_items,
                        initial_prev_results=results_prev,
                        initial_curr_results=results_curr,
                        prev_rollout_dir=prev_rollout_dir,
                        curr_rollout_dir=curr_rollout_dir,
                        policy=longitudinal_pair_policy,
                        target_n=slow_n,
                        seed=slow_seed,
                        out_root=out_root,
                    )
                    epoch_comparison_pairs = comparison_pairs
                    if all_comparison_pairs is not comparison_pairs:
                        save_comparison_pairs(
                            all_comparison_pairs,
                            os.path.join(slow_dir, "comparison_pairs_all.json"),
                        )
                    save_comparison_pairs(
                        comparison_pairs,
                        os.path.join(slow_dir, "comparison_pairs.json"),
                    )
                    n_regressed = sum(1 for p in comparison_pairs if p["category"] == "regressed")
                    n_improved = sum(1 for p in comparison_pairs if p["category"] == "improved")
                    n_persist = sum(1 for p in comparison_pairs if p["category"] == "persistent_fail")
                    n_stable = sum(1 for p in comparison_pairs if p["category"] == "stable_success")
                    print(
                        f"    [slow update] comparison: "
                        f"regressed={n_regressed} improved={n_improved} "
                        f"persistent_fail={n_persist} stable_success={n_stable} "
                        f"policy={longitudinal_pair_policy} "
                        f"kept={len(comparison_pairs)}/{len(all_comparison_pairs)}"
                    )

                    # 5. Extract previous slow update guidance for reflection
                    existing_guidance = extract_slow_update_field(current_skill)

                    # 6. Optimizer analysis (with reflection on previous guidance)
                    slow_result = run_slow_update(
                        current_skill,
                        results_prev,
                        results_curr,
                        slow_items,
                        prev_skill=prev_skill,
                        prev_slow_update_content=existing_guidance,
                        prev_rollout_dir=prev_rollout_dir,
                        curr_rollout_dir=curr_rollout_dir,
                        comparison_pairs=comparison_pairs,
                    )
                    slow_time = round(time.time() - t_slow, 1)

                    if slow_result and slow_result.get("slow_update_content"):
                        slow_candidate = replace_slow_update_field(
                            current_skill, slow_result["slow_update_content"],
                        )
                        slow_candidate_hash = skill_hash(slow_candidate)
                        with open(os.path.join(slow_dir, "candidate_skill.md"), "w") as f:
                            f.write(slow_candidate)
                        slow_result["time_s"] = slow_time
                        slow_result["prev_hard"] = prev_hard
                        slow_result["curr_hard"] = curr_hard
                        slow_result["candidate_hash"] = slow_candidate_hash
                        slow_result["update_origin"] = "slow_update_momentum"
                        slow_result["update_target"] = (
                            "Address longitudinal regressions and persistent failures "
                            "observed across adjacent epochs."
                        )

                        # Slow update acceptance — two modes selected via
                        # `optimizer.slow_update_gate_with_selection`.
                        if slow_gate_with_selection:
                            # ── Gated mode (follow SkillReflection) ──────────
                            # Evaluate the slow-update candidate on the
                            # selection set and accept/reject via the same
                            # validation gate used for step-level updates.
                            if slow_candidate_hash in sel_cache:
                                slow_sel_hard, slow_sel_soft = sel_cache[
                                    slow_candidate_hash
                                ]
                                print(
                                    f"    [slow gate] cache hit: "
                                    f"hard={slow_sel_hard:.4f}"
                                )
                            else:
                                sel_env, sel_n = _build_eval_env(
                                    split="valid_seen",
                                    env_num=cfg["sel_env_num"],
                                    seed=seed,
                                )
                                print(f"    [slow gate] selection items={sel_n}")
                                slow_eval_dir = os.path.join(
                                    slow_dir, "selection_eval",
                                )
                                slow_eval_results = adapter.rollout(
                                    sel_env, slow_candidate, slow_eval_dir,
                                )
                                slow_sel_hard, slow_sel_soft = compute_score(
                                    slow_eval_results
                                )
                                sel_cache[slow_candidate_hash] = (
                                    slow_sel_hard, slow_sel_soft,
                                )

                            slow_gate = evaluate_gate(
                                candidate_skill=slow_candidate,
                                cand_hard=slow_sel_hard,
                                current_skill=current_skill,
                                current_score=current_score,
                                best_skill=best_skill,
                                best_score=best_score,
                                best_step=best_step,
                                global_step=global_step,
                                cand_soft=slow_sel_soft,
                                metric=gate_metric,
                                mixed_weight=gate_mixed_weight,
                                use_semantic_density=use_semantic_density,
                                semantic_density_weight=semantic_density_weight,
                                leading_words=leading_words,
                            )
                            slow_result["selection_hard"] = slow_sel_hard
                            slow_result["selection_soft"] = slow_sel_soft
                            slow_result["action"] = slow_gate.action
                            prev_current = current_score
                            prev_best = best_score
                            current_skill = slow_gate.current_skill
                            current_score = slow_gate.current_score
                            best_skill = slow_gate.best_skill
                            best_score = slow_gate.best_score
                            best_step = slow_gate.best_step
                            if slow_gate.action in {"accept", "accept_new_best"}:
                                current_origin = (
                                    f"slow_update_epoch_{epoch:02d}"
                                )
                            if slow_gate.action == "accept_new_best":
                                best_origin = current_origin
                                print(
                                    f"    [slow gate] ACCEPT (new best) "
                                    f"hard={slow_sel_hard:.4f} > "
                                    f"prev best {prev_best:.4f}"
                                )
                            elif slow_gate.action == "accept":
                                print(
                                    f"    [slow gate] ACCEPT "
                                    f"hard={slow_sel_hard:.4f} > "
                                    f"current={prev_current:.4f}"
                                )
                            else:
                                print(
                                    f"    [slow gate] REJECT "
                                    f"hard={slow_sel_hard:.4f} <= "
                                    f"current={current_score:.4f}"
                                )
                            print(
                                f"    [slow update] guidance written "
                                f"({len(slow_result['slow_update_content'])} "
                                f"chars), {slow_time}s"
                            )
                        else:
                            # ── Force-accept mode (default) ──────────────────
                            # The epoch-level longitudinal guidance is injected
                            # into current_skill ONLY, so training continues
                            # with the accumulated slow memory. best_skill is
                            # left untouched: it must remain a faithful snapshot
                            # of the val-best step (which may be a pre-slow step
                            # such as S_0 carrying no slow_update field at all).
                            slow_content = slow_result["slow_update_content"]
                            current_skill = replace_slow_update_field(
                                current_skill, slow_content,
                            )
                            # Update caches so downstream steps use the
                            # slow-update-injected skill for hashing.
                            slow_candidate_hash = skill_hash(current_skill)
                            sel_cache[slow_candidate_hash] = (current_score, 0.0)

                            slow_result["action"] = "force_accept"
                            current_origin = f"slow_update_epoch_{epoch:02d}"

                            print(
                                f"    [slow update] force-injected into "
                                f"current only "
                                f"({len(slow_content)} chars), "
                                f"{slow_time}s"
                            )
                    else:
                        slow_result = slow_result or {}
                        slow_result["action"] = "no_content"
                        slow_result["time_s"] = slow_time
                        print(
                            f"    [slow update] no guidance produced, "
                            f"{slow_time}s"
                        )

                    # 5. Save
                    with open(slow_done_path, "w") as f:
                        json.dump(slow_result, f, indent=2, ensure_ascii=False)
                    _save_skill(out_root, global_step, current_skill)
                    with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                        f.write(best_skill)
                    _persist_runtime_state(global_step)

                    print(
                        f"\n  [SLOW UPDATE epoch {epoch} done] "
                        f"current={current_score:.4f} best={best_score:.4f}"
                    )

            # ── META SKILL (end of epoch, optimizer-side memory) ─────────
            use_meta_skill = cfg.get("use_meta_skill", False)
            if use_meta_skill:
                meta_skill_dir = os.path.join(out_root, "meta_skill", f"epoch_{epoch:02d}")
                meta_skill_done_path = os.path.join(meta_skill_dir, "meta_skill_result.json")
                os.makedirs(meta_skill_dir, exist_ok=True)

                if os.path.exists(meta_skill_done_path):
                    print(f"\n  [META SKILL epoch {epoch}] resumed — already done")
                elif epoch == 1:
                    with open(meta_skill_done_path, "w") as f:
                        json.dump(
                            {"action": "skip_first_epoch", "epoch": epoch},
                            f, indent=2, ensure_ascii=False,
                        )
                    print(f"\n  [META SKILL epoch {epoch}] skipped — first epoch")
                else:
                    print(
                        f"\n  {'='*60}\n"
                        f"  META SKILL — Epoch {epoch} "
                        f"(optimizer memory from epoch {epoch-1} vs {epoch})\n"
                        f"  {'='*60}"
                    )

                    prev_epoch_records = [h for h in history if h.get("epoch") == epoch - 1]
                    prev_epoch_last_step = prev_epoch_records[-1]["step"]
                    prev_skill = _load_skill(out_root, prev_epoch_last_step)
                    prev_meta_skill = _load_meta_skill_content(out_root, epoch - 1)

                    if epoch_comparison_pairs is None:
                        meta_n = cfg.get("slow_update_samples", 20)
                        meta_seed = seed + epoch * 2000
                        if dataloader is not None:
                            meta_batch = dataloader.build_train_batch(
                                batch_size=meta_n,
                                seed=meta_seed,
                                out_root=out_root,
                            )
                            meta_env = adapter.build_env_from_batch(
                                meta_batch, out_root=out_root,
                            )
                        else:
                            meta_env = adapter.build_train_env(
                                batch_size=meta_n,
                                seed=meta_seed,
                                out_root=out_root,
                            )
                        meta_items = list(meta_env) if hasattr(meta_env, "__iter__") else meta_env
                        prev_rollout_dir = os.path.join(meta_skill_dir, "rollout_prev")
                        curr_rollout_dir = os.path.join(meta_skill_dir, "rollout_curr")
                        results_prev = adapter.rollout(meta_env, prev_skill, prev_rollout_dir)
                        results_curr = adapter.rollout(meta_env, epoch_last_step_skill, curr_rollout_dir)
                        epoch_comparison_pairs, all_meta_comparison_pairs = _build_longitudinal_pairs(
                            adapter=adapter,
                            dataloader=dataloader,
                            prev_skill=prev_skill,
                            curr_skill=epoch_last_step_skill,
                            initial_items=meta_items,
                            initial_prev_results=results_prev,
                            initial_curr_results=results_curr,
                            prev_rollout_dir=prev_rollout_dir,
                            curr_rollout_dir=curr_rollout_dir,
                            policy=longitudinal_pair_policy,
                            target_n=meta_n,
                            seed=meta_seed,
                            out_root=out_root,
                        )
                        if all_meta_comparison_pairs is not epoch_comparison_pairs:
                            save_comparison_pairs(
                                all_meta_comparison_pairs,
                                os.path.join(meta_skill_dir, "comparison_pairs_all.json"),
                            )
                        save_comparison_pairs(
                            epoch_comparison_pairs,
                            os.path.join(meta_skill_dir, "comparison_pairs.json"),
                        )
                        meta_counts = _pair_category_counts(epoch_comparison_pairs)
                        print(
                            f"    [meta skill] comparison: "
                            f"regressed={meta_counts.get('regressed', 0)} "
                            f"improved={meta_counts.get('improved', 0)} "
                            f"persistent_fail={meta_counts.get('persistent_fail', 0)} "
                            f"stable_success={meta_counts.get('stable_success', 0)} "
                            f"policy={longitudinal_pair_policy} "
                            f"kept={len(epoch_comparison_pairs)}/{len(all_meta_comparison_pairs)}"
                        )

                    t_meta_skill = time.time()
                    meta_skill_result = run_meta_skill(
                        prev_skill=prev_skill,
                        curr_skill=epoch_last_step_skill,
                        comparison_pairs=epoch_comparison_pairs or [],
                        prev_meta_skill_content=prev_meta_skill,
                    )
                    meta_skill_time = round(time.time() - t_meta_skill, 1)

                    if meta_skill_result and meta_skill_result.get("meta_skill_content"):
                        meta_skill_result["time_s"] = meta_skill_time
                        meta_skill_result["action"] = "write_meta_skill"
                        print(
                            f"    [meta skill] memory written "
                            f"({len(meta_skill_result['meta_skill_content'])} chars), "
                            f"{meta_skill_time}s"
                        )
                    else:
                        meta_skill_result = meta_skill_result or {}
                        meta_skill_result["time_s"] = meta_skill_time
                        meta_skill_result["action"] = "no_content"
                        print(f"    [meta skill] no memory produced, {meta_skill_time}s")

                    with open(meta_skill_done_path, "w") as f:
                        json.dump(meta_skill_result, f, indent=2, ensure_ascii=False)

            # ── OPTIONAL BLOCKING EPOCH-END EVALUATION ─────────────────────
            # This hook is opt-in and is used by longitudinal experiments that
            # require Train/Val/IID/OOD to be evaluated before the next epoch
            # begins. The external evaluator is responsible for idempotency.
            epoch_eval_script = str(
                cfg.get("epoch_end_eval_script", "") or ""
            ).strip()
            if epoch_eval_script and not epoch_already_complete:
                epoch_skill_path = os.path.join(
                    out_root, "skills", f"skill_v{global_step:04d}.md"
                )
                epoch_eval_command = [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-u",
                    epoch_eval_script,
                    "--train-root",
                    out_root,
                    "--seed",
                    str(seed),
                    "--condition",
                    f"epoch_{epoch:02d}",
                    "--skill",
                    epoch_skill_path,
                ]
                config_path = str(cfg.get("_config_path", "") or "").strip()
                if config_path:
                    epoch_eval_command.extend(["--config", config_path])
                retry_limit = int(cfg.get("epoch_end_eval_retries", 0) or 0)
                attempt = 0
                while True:
                    attempt += 1
                    print(
                        f"\n  [EPOCH {epoch} EVALUATION] "
                        f"blocking attempt={attempt}"
                    )
                    completed = subprocess.run(
                        epoch_eval_command,
                        env=os.environ.copy(),
                        check=False,
                    )
                    if completed.returncode == 0:
                        print(
                            f"  [EPOCH {epoch} EVALUATION] completed; "
                            f"continuing training"
                        )
                        break
                    if retry_limit > 0 and attempt >= retry_limit:
                        raise RuntimeError(
                            f"Epoch {epoch} evaluation failed after "
                            f"{attempt} attempts"
                        )
                    delay = 30 if attempt <= 3 else 900
                    print(
                        f"  [EPOCH {epoch} EVALUATION] failed with "
                        f"code={completed.returncode}; retry in {delay}s"
                    )
                    time.sleep(delay)

        # ── Save best skill ──────────────────────────────────────────────
        with open(os.path.join(out_root, "best_skill.md"), "w") as f:
            f.write(best_skill)
        _persist_runtime_state(global_step)
        print(
            f"\n  [done] best skill from step {best_step}, "
            f"score={best_score:.4f}"
        )

        # ── Final test evaluation (valid_unseen) ─────────────────────────
        baseline_test_hard = None
        baseline_test_soft = None
        test_hard = None
        test_soft = None
        final_test_hard = None
        final_test_soft = None
        final_selection_hard = None
        final_selection_soft = None

        if cfg["eval_test"]:
            task_types = adapter.get_task_types()

            # ── Final skill validation (valid_seen) + best promotion ─────
            # The final (last) skill may carry an epoch-end slow_update that
            # was force-injected WITHOUT a val pass (use_gate=false or
            # slow_update_gate_with_selection=false), so it never competed for
            # best. Run one real val on the final skill; if its gate score
            # beats the incumbent best, PROMOTE it to best so that best is the
            # true val-argmax over all skills (including the final slow_update).
            # When final == best, reuse the existing val score (no rollout).
            try:
                if skill_hash(current_skill) == skill_hash(best_skill):
                    final_selection_hard, final_selection_soft = best_score, None
                    print(
                        "\n  [final skill == best skill] "
                        f"final_selection_hard={best_score:.4f} (reused)"
                    )
                else:
                    fval_env, fval_n = _build_eval_env(
                        split="valid_seen",
                        env_num=cfg["sel_env_num"],
                        seed=seed,
                    )
                    fval_dir = os.path.join(out_root, "final_selection_eval")
                    fval_results = adapter.rollout(fval_env, current_skill, fval_dir)
                    final_selection_hard, final_selection_soft = compute_score(fval_results)
                    final_gate_score = select_gate_score(
                        final_selection_hard, final_selection_soft,
                        gate_metric, gate_mixed_weight,
                        skill_content=current_skill,
                        use_semantic_density=use_semantic_density,
                        semantic_density_weight=semantic_density_weight,
                        leading_words=leading_words,
                    )
                    print(
                        f"\n  [final skill val] items={fval_n} "
                        f"final_selection_hard={final_selection_hard:.4f} "
                        f"gate={final_gate_score:.4f} "
                        f"(best={best_score:.4f})"
                    )
                    if final_gate_score > best_score:
                        # Promote: the final (slow-updated) skill is val-better
                        # than the incumbent best. Make it the new best so the
                        # subsequent BEST-skill test rollout evaluates it and
                        # best/final test scores coincide.
                        print(
                            f"  [promote] final {final_gate_score:.4f} > "
                            f"best {best_score:.4f} → final becomes new best "
                            f"(step {global_step}, origin {current_origin})"
                        )
                        best_skill = current_skill
                        best_score = final_gate_score
                        best_step = global_step
                        best_origin = current_origin
                        with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                            f.write(best_skill)
                        _persist_runtime_state(global_step)
            except Exception as _e:  # noqa: BLE001
                final_selection_hard = None
                final_selection_soft = None
                print(f"\n  [final skill val FAILED: {_e!r}]")

            # Baseline: S_0 on test set (valid_unseen)
            print(f"\n{'='*60}")
            print("  BASELINE TEST — evaluate initial skill on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env, test_n = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n}")
            baseline_test_dir = os.path.join(out_root, "test_eval_baseline")
            os.makedirs(baseline_test_dir, exist_ok=True)
            baseline_test_results = adapter.rollout(test_env, skill_init, baseline_test_dir)
            baseline_test_hard, baseline_test_soft = compute_score(baseline_test_results)
            baseline_buckets = _compute_task_type_buckets(baseline_test_results, task_types)
            print("\n  === Baseline Test Results (S_0) ===")
            for task_type in task_types + ["overall"]:
                b = baseline_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(baseline_test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in baseline_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # Best skill on test set
            print(f"\n{'='*60}")
            print("  BEST SKILL TEST — evaluate best skill on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env2, test_n2 = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n2}")
            test_dir = os.path.join(out_root, "test_eval")
            os.makedirs(test_dir, exist_ok=True)
            test_results = adapter.rollout(test_env2, best_skill, test_dir)
            test_hard, test_soft = compute_score(test_results)
            best_buckets = _compute_task_type_buckets(test_results, task_types)
            print("\n  === Best Skill Test Results ===")
            for task_type in task_types + ["overall"]:
                b = best_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in best_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # Final skill (last skill in trajectory) on test set.
            # Distinct from best_skill: with use_gate=False every candidate is
            # force-accepted so the final skill is whatever the last step
            # produced; with use_gate=True it is the last accepted skill, which
            # may differ from the best-on-val skill. We always evaluate it so
            # every run reports baseline / best-on-val / final on test.
            # Regularized so a failure here never prevents summary.json from being
            # written (the orchestrator's post-hoc safety net fills it in).
            try:
                if skill_hash(current_skill) == skill_hash(best_skill):
                    # Final == best: reuse results, skip a redundant rollout.
                    final_test_hard, final_test_soft = test_hard, test_soft
                    final_test_dir = os.path.join(out_root, "test_eval_final")
                    os.makedirs(final_test_dir, exist_ok=True)
                    with open(os.path.join(final_test_dir, "summary.json"), "w") as f:
                        json.dump(
                            {
                                k: {
                                    "total": b["total"],
                                    "hard_acc": b["hard"] / max(b["total"], 1),
                                }
                                for k, b in best_buckets.items()
                            },
                            f, indent=2, ensure_ascii=False,
                        )
                    print(
                        "\n  [final skill == best skill] "
                        f"final_test_hard={final_test_hard:.4f} (reused)"
                    )
                else:
                    print(f"\n{'='*60}")
                    print("  FINAL SKILL TEST — evaluate last skill on Test set (valid_unseen)")
                    print(f"{'='*60}")
                    test_env3, test_n3 = _build_eval_env(
                        split="valid_unseen",
                        env_num=cfg["test_env_num"],
                        seed=seed,
                    )
                    print(f"  Test items: {test_n3}")
                    final_test_dir = os.path.join(out_root, "test_eval_final")
                    os.makedirs(final_test_dir, exist_ok=True)
                    final_test_results = adapter.rollout(test_env3, current_skill, final_test_dir)
                    final_test_hard, final_test_soft = compute_score(final_test_results)
                    final_buckets = _compute_task_type_buckets(final_test_results, task_types)
                    print("\n  === Final Skill Test Results ===")
                    for task_type in task_types + ["overall"]:
                        b = final_buckets.get(task_type, {"total": 0, "hard": 0})
                        t = max(b["total"], 1)
                        print(
                            f"    {task_type:<40s}: "
                            f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                        )
                    with open(os.path.join(final_test_dir, "summary.json"), "w") as f:
                        json.dump(
                            {
                                k: {
                                    "total": b["total"],
                                    "hard_acc": b["hard"] / max(b["total"], 1),
                                }
                                for k, b in final_buckets.items()
                            },
                            f, indent=2, ensure_ascii=False,
                        )
            except Exception as _e:  # noqa: BLE001
                final_test_hard = None
                final_test_soft = None
                print(f"\n  [final skill test FAILED: {_e!r}] "
                      "— will be filled by post-hoc eval")

            # Comparison
            delta_hard = (test_hard or 0) - (baseline_test_hard or 0)
            print(f"\n  === Improvement vs baseline (init S_0) ===")
            print(
                f"    [2] best-on-val hard: {baseline_test_hard:.4f} -> {test_hard:.4f}  "
                f"(delta={delta_hard:+.4f})"
            )
            if final_test_hard is not None:
                final_delta_hard = (final_test_hard or 0) - (baseline_test_hard or 0)
                print(
                    f"    [3] final/last  hard: {baseline_test_hard:.4f} -> {final_test_hard:.4f}  "
                    f"(delta={final_delta_hard:+.4f})"
                )

        # ── Global summary ───────────────────────────────────────────────
        total_wall = time.time() - t_loop_start
        n_accept = sum(1 for h in history if "accept" in h.get("action", ""))
        n_reject = sum(
            1 for h in history
            if str(h.get("action", "")).endswith("reject")
        )
        n_skip = sum(1 for h in history if h.get("action") == "skip_no_patches")

        token_summary = get_token_summary()

        # Epoch-level statistics
        epoch_stats = []
        for e in range(1, num_epochs + 1):
            epoch_records = [h for h in history if h.get("epoch") == e]
            if epoch_records:
                epoch_stats.append({
                    "epoch": e,
                    "steps": [h["step"] for h in epoch_records],
                    "accepts": sum(1 for h in epoch_records if "accept" in h.get("action", "")),
                    "rejects": sum(
                        1 for h in epoch_records
                        if str(h.get("action", "")).endswith("reject")
                    ),
                    "skips": sum(1 for h in epoch_records if h.get("action") == "skip_no_patches"),
                    "best_score_at_epoch_end": epoch_records[-1].get("best_score", 0.0),
                    "current_score_at_epoch_end": epoch_records[-1].get("current_score", 0.0),
                })

        summary = {
            "version": "skillopt-0.1.0",
            "config": _redact_cfg(cfg),
            "baseline_selection_hard": sel_cache.get(
                skill_hash(skill_init), (None, None),
            )[0],
            "best_selection_hard": best_score,
            "final_selection_hard": final_selection_hard,
            "final_selection_soft": final_selection_soft,
            "best_step": best_step,
            "current_origin": current_origin,
            "best_origin": best_origin,
            "total_steps": len(history),
            "total_accepts": n_accept,
            "total_rejects": n_reject,
            "total_skips": n_skip,
            "epoch_stats": epoch_stats,
            "baseline_test_hard": baseline_test_hard,
            "baseline_test_soft": baseline_test_soft,
            "test_hard": test_hard,
            "test_soft": test_soft,
            "final_test_hard": final_test_hard,
            "final_test_soft": final_test_soft,
            "test_delta_hard": (
                (test_hard or 0) - (baseline_test_hard or 0)
                if test_hard is not None
                else None
            ),
            "final_test_delta_hard": (
                (final_test_hard or 0) - (baseline_test_hard or 0)
                if final_test_hard is not None
                else None
            ),
            "total_wall_time_s": round(total_wall, 1),
            "token_summary": token_summary,
        }
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("  Final Summary")
        print(f"{'='*60}")
        print(
            f"  steps={len(history)} accept={n_accept} "
            f"reject={n_reject} skip={n_skip}"
        )
        print(f"  best_score={best_score:.4f} (step {best_step})  wall={total_wall:.0f}s")
        if epoch_stats:
            for es in epoch_stats:
                print(
                    f"    epoch {es['epoch']}: accept={es['accepts']} reject={es['rejects']} "
                    f"best={es['best_score_at_epoch_end']:.4f}"
                )
        if baseline_test_hard is not None:
            print("\n  === TEST scores (3 skills, split=valid_unseen) ===")
            print(
                f"    [1] init/baseline (S_0)          : "
                f"test_hard={baseline_test_hard:.4f}"
            )
        if test_hard is not None:
            print(
                f"    [2] best-on-val (step {best_step})".ljust(37)
                + f": test_hard={test_hard:.4f} test_soft={test_soft:.4f}"
            )
        if final_test_hard is not None:
            print(
                f"    [3] final/last skill             : "
                f"test_hard={final_test_hard:.4f} test_soft={final_test_soft:.4f}"
            )
        if token_summary.get("_total"):
            t = token_summary["_total"]
            print(
                f"  total tokens: {t['total_tokens']:,} "
                f"(prompt={t['prompt_tokens']:,} "
                f"completion={t['completion_tokens']:,} "
                f"calls={t['calls']})"
            )

        return summary
