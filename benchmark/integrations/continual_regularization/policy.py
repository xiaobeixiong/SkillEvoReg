"""Pure policy layer. No model calls, environment execution or library writes.

Similarity matrices must come from the existing embedding backend. Thresholds
are explicit experiment inputs, not tuned defaults hidden in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence


META_SKILLS = frozenset({"create-skill", "modify-skill"})


def snapshot(root: Path) -> dict[str, str]:
    """Read learned SKILL.md files only, rejecting symlink escapes."""
    root = root.resolve()
    result = {}
    for path in sorted(root.rglob("SKILL.md")):
        relative = path.relative_to(root)
        if relative.parts[0] in META_SKILLS:
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Skill path escapes library: {relative}")
        result[relative.as_posix()] = path.read_text(encoding="utf-8")
    return result


def classify_changes(old: Mapping[str, str], candidate: Mapping[str, str]):
    """Classify actual writes, not the updater's claimed operation name."""
    return {
        "create": sorted(candidate.keys() - old.keys()),
        "modify": sorted(k for k in old.keys() & candidate.keys()
                         if old[k] != candidate[k]),
        "delete": sorted(old.keys() - candidate.keys()),
    }


@dataclass(frozen=True)
class Match:
    path: str
    coverage: float
    atom_maxima: tuple[float, ...]


def rank_matches(matrices: Mapping[str, Sequence[Sequence[float]]], *,
                 atom_count: int, tau: float, rho: float,
                 top_k: int = 3, audit: list | None = None) -> list[Match]:
    """Each matrix compares the same new atoms with ONE old skill's atoms."""
    if not 0 <= tau <= 1 or not 0 <= rho <= 1 or top_k < 1 or atom_count < 0:
        raise ValueError("Invalid matching policy")
    matches = []
    for path, rows in matrices.items():
        if len(rows) != atom_count:
            raise ValueError("Matrix rows must equal new skill atom count")
        widths = {len(row) for row in rows}
        if len(widths) > 1:
            raise ValueError("Ragged similarity matrix")
        maxima = []
        for row in rows:
            values = [float(v) for v in row]
            if any(not math.isfinite(v) or not -1.000001 <= v <= 1.000001
                   for v in values):
                raise ValueError("Invalid cosine similarity")
            maxima.append(max(values, default=-1.0))
        coverage = sum(v >= tau for v in maxima) / atom_count if atom_count else 0.0
        if audit is not None:
            audit.append(dict(path=path, coverage=coverage, atom_maxima=maxima,
                              tau=tau, rho=rho, selected=False))
        if atom_count and any(len(row) for row in rows) and coverage >= rho:
            matches.append(Match(path, coverage, tuple(maxima)))
    selected = sorted(matches, key=lambda m: (-m.coverage, m.path))[:top_k]
    if audit is not None:
        paths = {m.path for m in selected}
        for row in audit:
            row['selected'] = row['path'] in paths
    return selected


def feedback_modules(*, dropout_consumed: bool, complexity_exceeded: bool):
    return tuple(name for name, active in (
        ("dropout", dropout_consumed), ("complexity", complexity_exceeded)
    ) if active)


def render_instruction(mode: str, modules: Sequence[str]) -> str:
    if mode not in {"create", "modify"} or set(modules) - {"dropout", "complexity"}:
        raise ValueError("Unknown mode or regularization module")
    common = (
        "Perform one Quick Update only. Preserve unique capabilities, preconditions, "
        "exceptions and valid behavior. NOOP preserves the candidate unchanged. "
        "Similarity selects comparison candidates; it does not prove equivalence. "
        "Do not edit skills outside the supplied scope. Do not introduce task IDs, "
        "reference answers or task-specific constants as general rules. "
    )
    if mode == "create":
        common += (
            "Compare the new skill with all supplied old skills jointly. MERGE only "
            "when unique content supplements an existing capability within its scope; "
            "assign each contribution once. Delete the new skill after MERGE only if "
            "all useful content is preserved. REWRITE changes only the new skill when "
            "its unique part is an independently useful capability; old skills remain "
            "unchanged. DELETE removes the new skill only if it has no unique capability "
            "and all useful content is already covered. When uncertain choose NOOP. "
            "With no historical match, do not MERGE; REWRITE of the new skill is allowed "
            "when requested by regularization feedback. "
            "Use receiving_skill_budgets to consider each receiving old skill's growth in this same update. "
            "These are soft targets, not rejection gates; do not move unnecessary verbosity into old skills. "
        )
    else:
        common += (
            "Operate on supplied source-delta atoms and qualified matched atoms only. "
            "MERGE preserves useful semantics from both sides. REWRITE changes only "
            "the source delta, leaving history unchanged. DELETE removes redundant "
            "delta content, never unique capabilities. "
            "For MERGE, use receiving_skill_budgets to consider each receiving old skill's growth in this same update; these are soft targets, not rejection gates or a request for a second compression phase. "
        )
    if "dropout" in modules:
        common += (
            "Dropout feedback: temporary atoms have been restored. Check newly "
            "introduced duplication or conflicts; masking alone is not a compression request. "
        )
    if "complexity" in modules:
        common += (
            "Complexity feedback: the supplied responsibility scope exceeds its soft "
            "budget. In THIS update consolidate redundancy and shorten verbose wording "
            "while preserving semantics. Budgets are soft targets, not acceptance gates. "
            "Do not transfer unnecessary growth into another skill. "
        )
    return common + "Record the outcome; do not run a second compression phase."
