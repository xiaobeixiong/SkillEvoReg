"""Atom-local repair regions for online SkillEvoReg.

This module deliberately contains no model calls.  It computes source-delta
regions and deterministically stitches model-provided replacement blocks into
the candidate skill, leaving all other bytes untouched.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher

from skillevolbench.components.online_regularization import rule_segment
from skillevolbench.components.semantic_retrieval import (
    DEFAULT_EMBEDDING_MATCH_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_METRIC,
    cosine_similarity_matrix,
)
from skillevolbench.components.skill_document import validate_skill_document


@dataclass(frozen=True)
class EditableRegion:
    region_id: str
    start: int
    end: int
    anchor_text: str
    candidate_text: str
    before_context: str
    after_context: str
    role: str = "source_delta"
    similarity: float = 1.0
    source_region_id: str = ""
    atom_type: str = ""
    similarity_metric: str = ""
    embedding_model: str = ""
    match_threshold: float | None = None
    match_scope: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _frontmatter_end(text: str) -> int:
    if not text.startswith("---\n"):
        return 0
    match = re.search(r"(?m)^---\s*$", text[4:])
    if match is None:
        return 0
    end = 4 + match.end()
    return end + (1 if text[end:].startswith("\n") else 0)


def _body_atoms(text: str) -> list[object]:
    """Segment only editable Markdown body, preserving absolute offsets."""
    offset = _frontmatter_end(text)
    return [
        replace(atom, start=atom.start + offset, end=atom.end + offset)
        for atom in rule_segment(text[offset:])
    ]


def _atom_type(atom: object) -> str:
    return str(getattr(atom, "atom_type", getattr(atom, "type", "")))


def _pair_kind(text: str, atom_type: str) -> str:
    """Use structural compatibility without trusting heuristic rule labels."""
    stripped = text.lstrip()
    if stripped.startswith("#"):
        return "heading"
    if atom_type == "code" or stripped.startswith("```"):
        return "code"
    return "rule"


def editable_regions(anchor: str, candidate: str) -> list[EditableRegion]:
    """Locate individual atomic changes from ``anchor`` to ``candidate``.

    SequenceMatcher is used only to discover changed runs.  Each changed new
    atom becomes its own source-delta region; surplus old atoms become separate
    zero-width deletion regions.  Adjacent atoms in one diff hunk are never
    exposed as one model-editable block.
    """
    old_atoms = _body_atoms(anchor)
    new_atoms = _body_atoms(candidate)
    matcher = SequenceMatcher(
        a=[atom.span for atom in old_atoms],
        b=[atom.span for atom in new_atoms],
        autojunk=False,
    )
    regions: list[EditableRegion] = []

    def append_region(
        *, start: int, end: int, anchor_text: str, candidate_text: str,
        atom_type: str,
    ) -> None:
        if not candidate_text.strip() and not anchor_text.strip():
            return
        regions.append(EditableRegion(
            region_id=f"r{len(regions) + 1}",
            start=start,
            end=end,
            anchor_text=anchor_text,
            candidate_text=candidate_text,
            before_context=candidate[max(0, start - 600):start],
            after_context=candidate[end:min(len(candidate), end + 600)],
            atom_type=atom_type,
        ))

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_old = [atom for atom in old_atoms[i1:i2] if atom.span.strip()]
        changed_new = [atom for atom in new_atoms[j1:j2] if atom.span.strip()]
        for index, atom in enumerate(changed_new):
            append_region(
                start=atom.start,
                end=atom.end,
                anchor_text=(
                    changed_old[index].span if index < len(changed_old) else ""
                ),
                candidate_text=atom.span,
                atom_type=_atom_type(atom),
            )
        insertion_point = (
            changed_new[-1].end
            if changed_new
            else (new_atoms[j1].start if j1 < len(new_atoms) else len(candidate))
        )
        for atom in changed_old[len(changed_new):]:
            append_region(
                start=insertion_point,
                end=insertion_point,
                anchor_text=atom.span,
                candidate_text="",
                atom_type=_atom_type(atom),
            )
    return regions


_SIMILARITY_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
MIN_EXISTING_MATCH_SIMILARITY = DEFAULT_EMBEDDING_MATCH_THRESHOLD
DEFAULT_MAX_EXISTING_MATCHES_PER_SOURCE = 3


def is_similarity_qualified(region: EditableRegion) -> bool:
    threshold = (
        MIN_EXISTING_MATCH_SIMILARITY
        if region.match_threshold is None
        else region.match_threshold
    )
    return region.role == "existing_match" and region.similarity >= threshold


def regularization_repair_regions(
    anchor: str,
    candidate: str,
    *,
    max_existing: int | None = None,
    max_existing_per_source: int = DEFAULT_MAX_EXISTING_MATCHES_PER_SOURCE,
    min_similarity: float = MIN_EXISTING_MATCH_SIMILARITY,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[EditableRegion]:
    """Open source clusters using same-batch-first retrieval.

    Each historical atom is assigned to at most one source delta, while one
    source may own a small candidate cluster.  Consequently a repair decision
    can jointly consolidate its atomic source delta and that deterministic
    cluster; the remaining skill text stays outside the editable set.
    """
    source = editable_regions(anchor, candidate)
    if (
        not source
        or (max_existing is not None and max_existing <= 0)
        or max_existing_per_source <= 0
    ):
        return source
    source_texts = [region.candidate_text for region in source]
    source_similarities = cosine_similarity_matrix(
        source_texts, source_texts, model_id_or_path=embedding_model,
    )
    batch_ranked: list[tuple[float, int, int]] = []
    for left in range(len(source)):
        left_text = source_texts[left]
        if not left_text.strip() or left_text.lstrip().startswith("#"):
            continue
        for right in range(left + 1, len(source)):
            right_text = source_texts[right]
            if not right_text.strip() or right_text.lstrip().startswith("#"):
                continue
            if _pair_kind(left_text, source[left].atom_type) != _pair_kind(
                right_text, source[right].atom_type,
            ):
                continue
            score = float(source_similarities[left, right])
            if score >= min_similarity:
                batch_ranked.append((score, left, right))
    batch_ranked.sort(key=lambda item: (-item[0], source[item[1]].start))
    batch_owner: dict[int, int] = {}
    batch_member_counts: dict[int, int] = {}
    batch_leaders: set[int] = set()
    batch_matches: list[EditableRegion] = []
    for score, left, right in batch_ranked:
        if left in batch_owner or right in batch_owner:
            continue
        if left in batch_leaders and right in batch_leaders:
            continue
        if left in batch_leaders:
            leader, follower = left, right
        elif right in batch_leaders:
            leader, follower = right, left
        else:
            leader, follower = left, right
        if batch_member_counts.get(leader, 0) >= max_existing_per_source:
            continue
        batch_leaders.add(leader)
        batch_owner[follower] = leader
        batch_member_counts[leader] = batch_member_counts.get(leader, 0) + 1
        follower_region = source[follower]
        leader_region = source[leader]
        batch_matches.append(EditableRegion(
            region_id=f"m{len(batch_matches) + 1}",
            start=follower_region.start,
            end=follower_region.end,
            anchor_text=follower_region.anchor_text,
            candidate_text=follower_region.candidate_text,
            before_context=follower_region.before_context,
            after_context=follower_region.after_context,
            role="same_batch_member",
            similarity=round(score, 6),
            source_region_id=leader_region.region_id,
            atom_type=follower_region.atom_type,
            similarity_metric=EMBEDDING_METRIC,
            embedding_model=embedding_model,
            match_threshold=min_similarity,
            match_scope="same_batch",
        ))
    canonical_source = [
        region for index, region in enumerate(source) if index not in batch_owner
    ]
    batch_members_by_source: dict[str, list[EditableRegion]] = {}
    for member in batch_matches:
        batch_members_by_source.setdefault(member.source_region_id, []).append(member)

    old_atoms = _body_atoms(anchor)
    new_atoms = _body_atoms(candidate)
    matcher = SequenceMatcher(
        a=[atom.span for atom in old_atoms],
        b=[atom.span for atom in new_atoms],
        autojunk=False,
    )
    existing_atoms: list[object] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for atom in new_atoms[j1:j2]:
            text = atom.span
            stripped = text.strip()
            if (
                stripped.startswith("#")
                or stripped == "---"
                or len(_SIMILARITY_TOKEN_RE.findall(text)) < 6
            ):
                continue
            existing_atoms.append(atom)
    history_source = canonical_source
    queries = []
    for region in history_source:
        group = [region, *sorted(
            batch_members_by_source.get(region.region_id, []),
            key=lambda item: item.start,
        )]
        queries.append("\n".join(
            item.candidate_text or item.anchor_text for item in group
        ))
    similarities = cosine_similarity_matrix(
        queries,
        [atom.span for atom in existing_atoms],
        model_id_or_path=embedding_model,
    )
    ranked: list[tuple[float, int, object]] = []
    for source_index, (source_region, query) in enumerate(zip(history_source, queries)):
        if query.lstrip().startswith("#"):
            continue
        for atom_index, atom in enumerate(existing_atoms):
            atom_type = _atom_type(atom)
            if _pair_kind(query, source_region.atom_type) != _pair_kind(
                atom.span, atom_type,
            ):
                continue
            score = float(similarities[source_index, atom_index])
            if score >= min_similarity:
                ranked.append((score, source_index, atom))
    ranked.sort(key=lambda item: (-item[0], history_source[item[1]].start, item[2].start))
    limit = (
        len(history_source) * max_existing_per_source
        if max_existing is None
        else max_existing
    )
    assigned_source_counts: dict[int, int] = {}
    assigned_atoms: set[tuple[int, int]] = set()
    history: list[EditableRegion] = []
    for score, source_index, atom in ranked:
        atom_key = (atom.start, atom.end)
        if (
            assigned_source_counts.get(source_index, 0)
            >= max_existing_per_source
            or atom_key in assigned_atoms
        ):
            continue
        source_region = history_source[source_index]
        history.append(EditableRegion(
            region_id=f"m{len(batch_matches) + len(history) + 1}",
            start=atom.start,
            end=atom.end,
            anchor_text=atom.span,
            candidate_text=atom.span,
            before_context=candidate[max(0, atom.start - 600):atom.start],
            after_context=candidate[atom.end:min(len(candidate), atom.end + 600)],
            role="existing_match",
            similarity=round(score, 6),
            source_region_id=source_region.region_id,
            atom_type=_atom_type(atom),
            similarity_metric=EMBEDDING_METRIC,
            embedding_model=embedding_model,
            match_threshold=min_similarity,
            match_scope="history",
        ))
        assigned_source_counts[source_index] = (
            assigned_source_counts.get(source_index, 0) + 1
        )
        assigned_atoms.add(atom_key)
        if len(history) >= limit:
            break
    return sorted(canonical_source + batch_matches + history, key=lambda region: region.start)


def render_regions(regions: list[EditableRegion]) -> str:
    return "\n\n".join(
        f"### {region.region_id} ({region.role}, similarity={region.similarity:.4f}, "
        f"source={region.source_region_id or region.region_id})\n"
        f"Pre-update source block:\n```text\n{region.anchor_text}\n```\n"
        f"Current editable block:\n```text\n{region.candidate_text}\n```\n"
        f"Read-only preceding context:\n```text\n{region.before_context}\n```\n"
        f"Read-only following context:\n```text\n{region.after_context}\n```"
        for region in regions
    )


def parse_replacements(payload: dict, regions: list[EditableRegion]) -> dict[str, str]:
    rows = payload.get("replacements") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("model returned no local replacements list")
    allowed = {region.region_id for region in regions}
    replacements: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("replacement must be an object")
        region_id = str(row.get("region_id", ""))
        if region_id not in allowed:
            raise ValueError(f"unknown editable region: {region_id!r}")
        if region_id in replacements:
            raise ValueError(f"duplicate editable region: {region_id}")
        replacement = row.get("replacement")
        if not isinstance(replacement, str):
            raise ValueError(f"replacement for {region_id} must be text")
        replacements[region_id] = replacement
    return replacements


def validate_delta_operations(
    payload: dict,
    regions: list[EditableRegion],
    replacements: dict[str, str],
    *,
    abnormal_modules: set[str] | frozenset[str],
) -> tuple[list[dict[str, str]], list[str]]:
    """Canonicalize one operation for every atomic source delta.

    Pairing is computed by :func:`regularization_repair_regions`, not chosen by the
    model.  Replacement rows are therefore treated as proposed source bodies;
    this function deterministically discards edits that are not owned by the
    declared operation and clears the linked historical atom only for MERGE.
    """
    rows = payload.get("operations") if isinstance(payload, dict) else None
    errors: list[str] = []
    by_id = {region.region_id: region for region in regions}
    sources = [region for region in regions if region.role == "source_delta"]
    linked_matches: dict[str, list[EditableRegion]] = {}
    history_source_ids = {
        region.source_region_id for region in regions
        if is_similarity_qualified(region)
    }
    for region in regions:
        if is_similarity_qualified(region) or (
            region.role == "same_batch_member"
            and region.source_region_id in history_source_ids
        ):
            linked_matches.setdefault(region.source_region_id, []).append(region)
    matched_source_ids = history_source_ids
    if not isinstance(rows, list):
        replacements.clear()
        return [], ["missing delta operations list"]
    normalized: list[dict[str, str]] = []
    canonical_replacements: dict[str, str] = {}
    seen_sources: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            errors.append("delta operation must be an object")
            continue
        source_id = str(row.get("source_region_id", ""))
        operation = str(row.get("operation", "")).strip().upper()
        source = by_id.get(source_id)
        if source is None or source.role != "source_delta":
            errors.append(f"unknown source-delta region: {source_id!r}")
            continue
        if source_id in seen_sources:
            errors.append(f"duplicate delta decision for {source_id}")
            continue
        seen_sources.add(source_id)
        if operation not in {"NOOP", "REWRITE", "MERGE", "DELETE"}:
            errors.append(f"missing or invalid delta operation for {source_id}")
            continue
        if (
            operation != "NOOP"
            and source_id not in matched_source_ids
            and not (
                "ccv_unmatched_rewrite" in abnormal_modules
                and operation == "REWRITE"
            )
        ):
            errors.append(
                f"{operation} for {source_id} is invalid because the source "
                "delta has no similarity-qualified existing atom"
            )
        linked = linked_matches.get(source_id, [])
        source_replacement = replacements.get(source_id)
        canonical_match_id = ""
        if operation == "NOOP":
            # Candidate source atom and linked history atom are both preserved.
            pass
        elif operation == "REWRITE":
            if source_replacement is None or not source_replacement.strip():
                errors.append(
                    f"REWRITE for {source_id} requires a non-empty source replacement"
                )
            else:
                canonical_replacements[source_id] = source_replacement
        elif operation == "MERGE":
            if not source.candidate_text.strip():
                errors.append(
                    f"MERGE for {source_id} is invalid because the source delta "
                    "is already empty"
                )
            if not linked:
                errors.append(
                    f"MERGE for {source_id} requires at least one "
                    "similarity-qualified existing atom"
                )
            if source_replacement is None or not source_replacement.strip():
                errors.append(
                    f"MERGE for {source_id} requires a non-empty merged source "
                    "replacement"
                )
            else:
                canonical_replacements[source_id] = source_replacement
            for match in linked:
                canonical_replacements[match.region_id] = ""
        elif operation == "DELETE":
            if not source.candidate_text.strip():
                errors.append(
                    f"DELETE for {source_id} is invalid because the source delta "
                    "is already empty"
                )
            else:
                canonical_replacements[source_id] = ""
        canonical_match_ids = [match.region_id for match in linked] if operation == "MERGE" else []
        canonical_match_id = canonical_match_ids[0] if len(canonical_match_ids) == 1 else ""
        normalized_row = {
            "source_region_id": source_id,
            "operation": operation,
            "match_region_id": canonical_match_id,
        }
        if len(canonical_match_ids) > 1:
            normalized_row["match_region_ids"] = canonical_match_ids
        normalized.append(normalized_row)
    missing = sorted(region.region_id for region in sources if region.region_id not in seen_sources)
    if missing:
        errors.append("missing decisions for source deltas: " + ", ".join(missing))
    replacements.clear()
    replacements.update(canonical_replacements)
    return normalized, errors


def apply_replacements(
    candidate: str,
    regions: list[EditableRegion],
    replacements: dict[str, str],
) -> str:
    region_by_id = {region.region_id: region for region in regions}
    region_order = {
        region.region_id: index for index, region in enumerate(regions)
    }
    selected = sorted(
        [
            (region_by_id[region_id], replacement)
            for region_id, replacement in replacements.items()
        ],
        key=lambda item: (item[0].start, region_order[item[0].region_id]),
    )
    cursor = 0
    parts: list[str] = []
    for region, replacement in selected:
        if region.start < cursor:
            raise ValueError("overlapping editable regions")
        if region.start and candidate[region.start - 1] != "\n":
            raise ValueError(f"{region.region_id} does not start at a line boundary")
        if region.end < len(candidate) and region.end and candidate[region.end - 1] != "\n":
            raise ValueError(f"{region.region_id} does not end at a line boundary")
        if (
            replacement
            and (region.candidate_text.endswith("\n") or region.end < len(candidate))
            and not replacement.endswith("\n")
        ):
            replacement += "\n"
        parts.extend((candidate[cursor:region.start], replacement))
        cursor = region.end
    parts.append(candidate[cursor:])
    return "".join(parts)


def validate_repaired_document(text: str) -> list[str]:
    errors = validate_skill_document(text)
    try:
        if "".join(atom.span for atom in rule_segment(text)) != text:
            errors.append("repair is not an exact atom cover")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"repair cannot be segmented: {type(exc).__name__}: {exc}")
    return errors


_DUPLICATE_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])\s+",
)
_DUPLICATE_LEXICAL_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _exact_duplicate_atom_key(atom: object) -> str | None:
    """Return a deliberately conservative exact-rule comparison key."""
    text = str(getattr(atom, "span", ""))
    stripped = text.strip()
    if not stripped:
        return None
    if getattr(atom, "atom_type", "") == "code" or stripped.startswith("```"):
        return None
    if stripped == "---" or re.match(r"^#{1,6}\s+", stripped):
        return None
    without_prefix = _DUPLICATE_LIST_PREFIX_RE.sub("", stripped, count=1)
    if len(_DUPLICATE_LEXICAL_TOKEN_RE.findall(without_prefix)) < 6:
        return None
    # Exact means exact after only presentation-level normalization: trim,
    # collapse whitespace, case-fold, and ignore one Markdown list marker.
    return re.sub(r"\s+", " ", without_prefix).casefold()


def new_exact_duplicate_atom_errors(before: str, after: str) -> list[str]:
    """Reject only newly multiplied exact long-rule atoms in one skill.

    A pre-existing duplicate is tolerated.  One newly introduced copy is also
    valid when the rule did not exist before; the invariant fires only when a
    repair increases the count beyond both one and the pre-repair count.
    """
    before_counts = Counter(
        key for atom in rule_segment(before)
        if (key := _exact_duplicate_atom_key(atom)) is not None
    )
    after_counts = Counter(
        key for atom in rule_segment(after)
        if (key := _exact_duplicate_atom_key(atom)) is not None
    )
    errors: list[str] = []
    for key, after_count in sorted(after_counts.items()):
        before_count = before_counts.get(key, 0)
        if after_count <= max(1, before_count):
            continue
        preview = key if len(key) <= 140 else key[:137] + "..."
        errors.append(
            "repair introduced a new exact duplicate rule atom "
            f"(count {before_count} -> {after_count}): {preview!r}"
        )
    return errors


def protected_content_hash(candidate: str, regions: list[EditableRegion]) -> str:
    cursor = 0
    parts: list[str] = []
    for region in sorted(regions, key=lambda item: item.start):
        parts.append(candidate[cursor:region.start])
        cursor = region.end
    parts.append(candidate[cursor:])
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_MAX_EXISTING_MATCHES_PER_SOURCE",
    "EditableRegion",
    "MIN_EXISTING_MATCH_SIMILARITY",
    "apply_replacements",
    "editable_regions",
    "regularization_repair_regions",
    "is_similarity_qualified",
    "new_exact_duplicate_atom_errors",
    "parse_replacements",
    "validate_delta_operations",
    "protected_content_hash",
    "render_regions",
    "validate_repaired_document",
]
