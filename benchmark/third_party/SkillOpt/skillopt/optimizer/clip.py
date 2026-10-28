"""ReflACT gradient clipping — LLM-driven edit ranking and selection.

Analogous to gradient clipping in neural network training: ranks candidate
edits by importance and selects the top-L to apply, controlling the
effective step size. Previously core/select.py.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re

from skillopt.model import chat_optimizer
from skillopt.optimizer.meta_skill import format_meta_skill_context
from skillopt.optimizer.update_modes import (
    describe_item,
    get_payload_items,
    is_rewrite_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
)
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


_TARGETED_OPS = {"insert_after", "replace", "delete"}
_KNOWN_OPS = _TARGETED_OPS | {"append"}


def _field(edit: object, name: str) -> str:
    value = getattr(edit, name, None)
    if value is None and isinstance(edit, dict):
        value = edit.get(name, "")
    return str(value or "")


def _token_count(text: str) -> int:
    """Cheap deterministic size proxy; formal complexity is computed later."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def structural_edit_metrics(
    skill_content: str,
    edit: object,
    *,
    index: int,
) -> dict:
    """Return factual, non-semantic metadata for one current-step edit."""
    op = _field(edit, "op").strip().lower()
    target = _field(edit, "target")
    content = _field(edit, "content").strip()
    target_match = op == "append" or bool(target and target in skill_content)
    eligible = (
        op in _KNOWN_OPS
        and target_match
        and (op == "delete" or bool(content))
    )

    content_tokens = _token_count(content)
    target_tokens = _token_count(target)
    if op in {"append", "insert_after"}:
        token_delta = content_tokens
    elif op == "replace":
        token_delta = content_tokens - target_tokens
    elif op == "delete":
        token_delta = -target_tokens
    else:
        token_delta = content_tokens

    comparison_skill = skill_content
    if target_match and target and op in _TARGETED_OPS:
        comparison_skill = comparison_skill.replace(target, "", 1)
    normalized_content = " ".join(content.lower().split())
    blocks = [
        " ".join(block.lower().split())
        for block in re.split(r"\n\s*\n", comparison_skill)
        if block.strip()
    ]
    overlap = max(
        (
            SequenceMatcher(None, normalized_content, block).ratio()
            for block in blocks
            if normalized_content
        ),
        default=0.0,
    )
    exact_duplicate = bool(
        normalized_content
        and normalized_content in " ".join(comparison_skill.lower().split())
    )
    flags = []
    if not eligible:
        flags.append("invalid_target_or_operation")
    if op in {"append", "insert_after"} and token_delta > 100:
        flags.append("large_append")
    if exact_duplicate or overlap >= 0.95:
        flags.append("lexical_duplicate")
    if token_delta < 0:
        flags.append("net_compression")

    support = getattr(edit, "support_count", None)
    if support is None and isinstance(edit, dict):
        support = edit.get("support_count", 0)
    return {
        "edit_index": index,
        "operation": op,
        "support_count": int(support or 0),
        "eligible": eligible,
        "target_match": target_match,
        "token_delta": token_delta,
        "max_lexical_overlap": round(overlap, 4),
        "structural_flags": flags,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def rank_and_select(
    skill_content: str,
    patch: dict,
    max_edits: int,
    meta_skill_context: str = "",
    update_mode: str = "patch",
    feedback_context: str = "",
) -> dict:
    """Use a optimizer LLM to rank edits by importance, then keep top-L.

    If the edit pool is within budget, returns the patch unchanged.
    Otherwise, calls the optimizer to rank and select the most impactful edits.

    Parameters
    ----------
    skill_content : str
        Current skill document.
    patch : dict
        Merged :class:`~skillopt.types.Patch` dict with ``edits`` list.
    max_edits : int
        Maximum number of edits to keep (the "edit budget").

    Returns
    -------
    dict
        :class:`~skillopt.types.Patch` dict with selected edits and
        optional ``ranking_details``.
    """
    update_mode = normalize_update_mode(update_mode)
    edits = get_payload_items(patch, update_mode)
    if len(edits) <= max_edits and not feedback_context:
        return patch

    # Structural metadata is enabled only for Regularization feedback runs so the
    # Original SkillOpt baseline keeps byte-for-byte selection behaviour.
    structural_metrics = []
    eligible_indices = set(range(len(edits)))
    if feedback_context and update_mode == "patch":
        structural_metrics = [
            structural_edit_metrics(skill_content, edit, index=i)
            for i, edit in enumerate(edits)
        ]
        eligible_indices = {
            row["edit_index"] for row in structural_metrics if row["eligible"]
        }

    # Build the edit pool description for the optimizer.
    edits_desc = []
    for i, edit in enumerate(edits):
        metrics_suffix = ""
        if structural_metrics:
            metrics_suffix = f"\n    Structural metadata: {structural_metrics[i]}"
        edits_desc.append(
            f"[{i}] {describe_item(edit, update_mode)}{metrics_suffix}"
        )

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## {payload_label(update_mode, title=True)} Pool ({len(edits)} {payload_label(update_mode)}, budget={max_edits})\n"
        + "\n".join(edits_desc)
        + f"\n\nSelect the {max_edits} most important {payload_label(update_mode)}. "
        f"Return their 0-based indices in priority order."
    )
    if feedback_context:
        user += (
            "\n\n## Previous-step global Regularization feedback\n"
            + feedback_context
            + "\nUse the complete Current Skill when choosing the update direction. "
            "Prefer replacing or merging overlapping rules over appending a "
            "shorter but disconnected fragment. Preserve capabilities supported "
            "by the training evidence. Structural metadata is a factual cost and "
            "applicability aid, not a semantic score. Never select an ineligible "
            "edit. You may select fewer than the budget."
        )
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    prompt_name = "ranking_rewrite" if is_rewrite_mode(update_mode) else "ranking"

    try:
        response, _ = chat_optimizer(
            system=load_prompt(prompt_name), user=user,
            max_completion_tokens=65536, retries=3, stage="ranking",
        )
        result = extract_json(response)
        if result and "selected_indices" in result:
            indices = result["selected_indices"]
            selected = []
            seen: set[int] = set()
            for idx in indices:
                if (
                    isinstance(idx, int)
                    and 0 <= idx < len(edits)
                    and idx not in seen
                    and idx in eligible_indices
                ):
                    selected.append(edits[idx])
                    seen.add(idx)
                if len(selected) >= max_edits:
                    break
            if selected or (feedback_context and indices == []):
                return {
                    "reasoning": patch.get("reasoning", "")
                    + f" [optimizer-ranked: selected {len(selected)}/{len(edits)} {payload_label(update_mode)}]",
                    payload_key(update_mode): selected,
                    "ranking_details": result,
                    "structural_metrics": structural_metrics,
                }
    except Exception:  # noqa: BLE001
        pass

    # Fallback: deterministic truncation after applicability filtering.
    fallback_edits = [
        edit for idx, edit in enumerate(edits) if idx in eligible_indices
    ][:max_edits]
    return {
        "reasoning": patch.get("reasoning", "")
        + f" [fallback truncated {len(edits)}->{len(fallback_edits)} {payload_label(update_mode)}]",
        payload_key(update_mode): fallback_edits,
        "structural_metrics": structural_metrics,
    }
