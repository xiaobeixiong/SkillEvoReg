"""Skill complexity metrics used by SkillEvoReg."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from skillopt.regularization.atoms import AtomicSegmenter, SkillAtom


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Stable tokenizer-independent proxy used consistently across conditions."""
    return len(_TOKEN_RE.findall(text))


def code_complexity(
    atom: SkillAtom,
    *,
    line_scale: float = 20.0,
    branch_scale: float = 5.0,
    dependency_scale: float = 3.0,
) -> tuple[float, dict]:
    if atom.type != "code":
        return 0.0, {"code_lines": 0, "branches": 0, "dependencies": 0}
    lines = [
        line for line in atom.span.splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    branches = len(re.findall(
        r"(?m)^\s*(?:if|elif|for|while|try|except|case|match)\b",
        atom.span,
    ))
    dependencies = len(re.findall(
        r"(?m)^\s*(?:from\s+\S+\s+import|import\s+\S+)\b|"
        r"\b(?:requests|httpx)\.(?:get|post|put|delete|patch)\s*\(",
        atom.span,
    ))
    value = (
        len(lines) / max(line_scale, 1.0)
        + 2.0 * branches / max(branch_scale, 1.0)
        + 2.0 * dependencies / max(dependency_scale, 1.0)
    )
    return value, {
        "code_lines": len(lines),
        "branches": branches,
        "dependencies": dependencies,
    }


@dataclass(frozen=True)
class ComplexityReport:
    total: float
    l2: float
    token_count: int
    atom_count: int
    code_complexity: float
    atoms: list[dict]
    segmenter: dict

    def to_dict(self) -> dict:
        return asdict(self)


def measure_complexity(
    skill: str,
    segmenter: AtomicSegmenter,
    *,
    token_scale: float = 100.0,
    atom_cost: float = 0.2,
    enable_code_complexity: bool = True,
) -> ComplexityReport:
    atoms, segmenter_meta = segmenter.segment(skill)
    atom_rows: list[dict] = []
    total = 0.0
    l2 = 0.0
    total_tokens = 0
    total_code = 0.0
    for atom in atoms:
        tokens = count_tokens(atom.span)
        code_value, code_stats = code_complexity(atom)
        if not enable_code_complexity:
            code_value = 0.0
        phi = tokens / max(token_scale, 1.0) + atom_cost + code_value
        total += phi
        l2 += phi * phi
        total_tokens += tokens
        total_code += code_value
        atom_rows.append({
            **atom.to_dict(),
            "token_count": tokens,
            "base_atom_cost": atom_cost,
            "code_complexity": code_value,
            "phi": phi,
            **code_stats,
        })
    return ComplexityReport(
        total=total,
        l2=l2,
        token_count=total_tokens,
        atom_count=len(atoms),
        code_complexity=total_code,
        atoms=atom_rows,
        segmenter=segmenter_meta,
    )


def normalized_growth(
    old: ComplexityReport,
    candidate: ComplexityReport,
    *,
    free_growth_budget: float = 0.0,
) -> dict:
    """Normalize positive growth by the current skill and charge only excess.

    A relative free-growth budget prevents small, routine skill additions from
    turning a quantized validation tie into an automatic rejection.
    """
    delta = max(0.0, candidate.total - old.total)
    normalizer = max(old.total, 1.0)
    relative_growth = delta / normalizer
    free_budget = max(0.0, float(free_growth_budget))
    penalized_growth = max(0.0, relative_growth - free_budget)
    return {
        "delta_positive": delta,
        "normalizer": normalizer,
        "normalizer_source": "old_skill",
        "relative_growth": relative_growth,
        "free_growth_budget": free_budget,
        "penalized_growth": penalized_growth,
        # Backward-compatible key consumed by the gate. It now deliberately
        # means the chargeable growth after the free budget.
        "normalized_growth": penalized_growth,
    }


def _normalized_span(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _span_similarity(left: str, right: str) -> float:
    """Semantic-ish similarity that cannot be dominated by punctuation."""
    left_words = {token.casefold() for token in re.findall(r"\w+", left)}
    right_words = {token.casefold() for token in re.findall(r"\w+", right)}
    if len(left_words) < 3 or len(right_words) < 3:
        return 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    overlap = 2.0 * len(left_words & right_words) / (
        len(left_words) + len(right_words)
    )
    return max(sequence, overlap)


def _compatible_atom_types(left: str, right: str) -> bool:
    if left == right:
        return True
    procedural = {"procedure_step", "constraint", "failure_check", "exception"}
    documentary = {"example", "reference"}
    return (left in procedural and right in procedural) or (
        left in documentary and right in documentary
    )


def analyze_update_complexity(
    old: ComplexityReport,
    candidate: ComplexityReport,
    *,
    edits: list[dict] | None = None,
    max_contributors: int = 3,
) -> dict:
    """Attribute net whole-skill growth to candidate atoms and source edits.

    The comparison is deliberately against the complete current skill.  This
    prevents feedback from blindly shortening a useful insertion when the real
    repair is to merge it into an existing, semantically overlapping rule.
    """
    old_rows = list(old.atoms)
    candidate_rows = list(candidate.atoms)
    edit_rows = [row for row in (edits or []) if isinstance(row, dict)]
    unmatched_old = set(range(len(old_rows)))
    contributors: list[dict] = []

    for candidate_index, row in enumerate(candidate_rows):
        candidate_text = _normalized_span(row.get("span", ""))
        best_index = None
        best_similarity = 0.0
        for old_index in unmatched_old:
            old_text = _normalized_span(old_rows[old_index].get("span", ""))
            similarity = SequenceMatcher(None, old_text, candidate_text).ratio()
            if similarity > best_similarity:
                best_index, best_similarity = old_index, similarity
        if best_index is not None and best_similarity >= 0.98:
            unmatched_old.discard(best_index)
            continue

        closest_index = None
        closest_similarity = 0.0
        for old_index, old_row in enumerate(old_rows):
            if not _compatible_atom_types(
                str(old_row.get("type", "other")),
                str(row.get("type", "other")),
            ):
                continue
            old_text = _normalized_span(old_row.get("span", ""))
            similarity = _span_similarity(old_text, candidate_text)
            if similarity > closest_similarity:
                closest_index, closest_similarity = old_index, similarity
        closest = old_rows[closest_index] if closest_index is not None else None
        kind = "modified" if closest is not None and closest_similarity >= 0.55 else "added"
        replaced_cost = float(closest.get("phi", 0.0)) if kind == "modified" else 0.0
        net_cost = float(row.get("phi", 0.0)) - replaced_cost

        source_edit = None
        source_edit_similarity = 0.0
        for edit_index, edit in enumerate(edit_rows):
            edit_text = _normalized_span(edit.get("content", ""))
            if not edit_text:
                continue
            similarity = (
                1.0 if edit_text in candidate_text or candidate_text in edit_text
                else _span_similarity(edit_text, candidate_text)
            )
            if similarity > source_edit_similarity:
                source_edit_similarity = similarity
                source_edit = {
                    "index": edit_index,
                    "op": edit.get("op"),
                    "target": edit.get("target"),
                    "support_count": edit.get("support_count", 0),
                    "source_type": edit.get("source_type", ""),
                    "content_preview": str(edit.get("content", ""))[:240],
                    "match_similarity": round(similarity, 4),
                }

        if closest_similarity >= 0.60:
            diagnosis, recommendation = "overlapping_expansion", "merge_or_replace_existing_atom"
        elif row.get("type") in {"code", "example"}:
            diagnosis, recommendation = "high_cost_specialization", "compress_to_general_rule"
        elif source_edit and int(source_edit.get("support_count", 0) or 0) <= 1:
            diagnosis, recommendation = "weakly_supported_addition", "remove_or_supply_train_evidence"
        else:
            diagnosis, recommendation = "new_capability", "compress_while_preserving_supported_behavior"
        contributors.append({
            "candidate_atom_id": row.get("atom_id"),
            "candidate_atom_type": row.get("type"),
            "candidate_span_preview": str(row.get("span", ""))[:320],
            "candidate_cost": float(row.get("phi", 0.0)),
            "net_complexity_contribution": net_cost,
            "closest_current_atom": ({
                "atom_id": closest.get("atom_id"),
                "type": closest.get("type"),
                "span_preview": str(closest.get("span", ""))[:240],
                "cost": float(closest.get("phi", 0.0)),
                "similarity": round(closest_similarity, 4),
            } if closest is not None else None),
            "source_edit": source_edit,
            "diagnosis": diagnosis,
            "recommendation": recommendation,
        })

    # Aggregate a multi-atom append/replace back to the optimizer edit that
    # caused it.  Feedback should repair one coherent update component rather
    # than independently shortening fragments without understanding the edit.
    grouped: dict[str, dict] = {}
    for contributor in contributors:
        source_edit = contributor.get("source_edit") or {}
        edit_index = source_edit.get("index")
        group_key = (
            f"edit:{edit_index}" if edit_index is not None
            else f"atom:{contributor.get('candidate_atom_id')}"
        )
        group = grouped.setdefault(group_key, {
            "component_kind": "source_edit" if edit_index is not None else "unmapped_atom",
            "source_edit": source_edit or None,
            "candidate_atom_ids": [],
            "atom_contributors": [],
            "net_complexity_contribution": 0.0,
        })
        group["candidate_atom_ids"].append(contributor.get("candidate_atom_id"))
        group["atom_contributors"].append(contributor)
        group["net_complexity_contribution"] += max(
            0.0, float(contributor.get("net_complexity_contribution", 0.0))
        )
    aggregated = list(grouped.values())
    for group in aggregated:
        atoms = sorted(
            group["atom_contributors"],
            key=lambda row: row["net_complexity_contribution"],
            reverse=True,
        )
        representative = atoms[0]
        group.update({
            "candidate_atom_id": representative.get("candidate_atom_id"),
            "candidate_atom_type": representative.get("candidate_atom_type"),
            "candidate_span_preview": representative.get("candidate_span_preview"),
            "closest_current_atom": representative.get("closest_current_atom"),
            "diagnosis": representative.get("diagnosis"),
            "recommendation": representative.get("recommendation"),
        })
    aggregated.sort(
        key=lambda row: row["net_complexity_contribution"], reverse=True
    )
    return {
        "whole_skill_net_change": float(candidate.total - old.total),
        "old_atom_count": old.atom_count,
        "candidate_atom_count": candidate.atom_count,
        "top_complexity_contributors": aggregated[:max(0, max_contributors)],
    }
