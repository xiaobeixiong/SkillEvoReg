"""Exact-span atomic segmentation for SkillEvoReg."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Callable

from skillopt.utils import extract_json


ATOM_TYPES = frozenset({
    "procedure_step",
    "constraint",
    "assumption",
    "failure_check",
    "exception",
    "tool_usage_rule",
    "example",
    "reference",
    "code",
    "other",
})

SEGMENTER_SYSTEM_PROMPT = """You segment an agent skill into atomic units.
You may only insert semantic boundaries. Never rewrite, summarize, delete, or
add text. Every span must be an exact substring of the supplied skill and the
ordered spans must preserve the original order.

Allowed types: procedure_step, constraint, assumption, failure_check,
exception, tool_usage_rule, example, reference, code, other.

Return JSON only:
{"segments": [{"type": "...", "span": "exact substring"}]}
"""


@dataclass(frozen=True)
class SkillAtom:
    atom_id: str
    type: str
    span: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


def _infer_type(span: str) -> str:
    stripped = span.strip()
    lower = stripped.lower()
    if stripped.startswith("```"):
        return "code"
    if re.search(r"\b(example|for example|e\.g\.)\b", lower):
        return "example"
    if re.search(r"\b(reference|documentation|see also|source)\b", lower):
        return "reference"
    if re.search(r"\b(if .* fails?|failure|verify|check|validate)\b", lower):
        return "failure_check"
    if re.search(r"\b(except|unless|edge case|special case)\b", lower):
        return "exception"
    if re.search(r"\b(must|never|only|ensure|required|do not)\b", lower):
        return "constraint"
    if re.search(r"\b(tool|api|python|shell|browser|workbook|spreadsheet)\b", lower):
        return "tool_usage_rule"
    if re.search(r"\b(assume|assumption|presume)\b", lower):
        return "assumption"
    if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", span):
        return "procedure_step"
    return "other"


def _atoms_from_ranges(skill: str, ranges: list[tuple[int, int, str]]) -> list[SkillAtom]:
    atoms: list[SkillAtom] = []
    for index, (start, end, atom_type) in enumerate(ranges):
        if end <= start:
            continue
        span = skill[start:end]
        atoms.append(SkillAtom(
            atom_id=f"a{index:04d}",
            type=atom_type if atom_type in ATOM_TYPES else "other",
            span=span,
            start=start,
            end=end,
        ))
    if "".join(atom.span for atom in atoms) != skill:
        raise ValueError("atomic segmentation did not preserve the original skill")
    return atoms


def rule_segment(skill: str) -> list[SkillAtom]:
    """Deterministic exact-cover fallback used when LLM segmentation fails."""
    if not skill:
        return []

    boundaries = {0, len(skill)}
    in_fence = False
    offset = 0
    for line in skill.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if not in_fence:
                boundaries.add(offset)
            in_fence = not in_fence
            if not in_fence:
                boundaries.add(offset + len(line))
        elif not in_fence and (
            re.match(r"^#{1,6}\s+", stripped)
            or re.match(r"^(?:[-*+]|\d+[.)])\s+", stripped)
            or not stripped.strip()
        ):
            boundaries.add(offset)
            boundaries.add(offset + len(line))
        offset += len(line)

    ordered = sorted(boundaries)
    ranges: list[tuple[int, int, str]] = []
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        span = skill[start:end]
        atom_type = "code" if span.lstrip().startswith("```") else _infer_type(span)
        ranges.append((start, end, atom_type))
    return _atoms_from_ranges(skill, ranges)


def exact_span_atoms(skill: str, segments: list[dict]) -> list[SkillAtom]:
    """Validate ordered LLM spans and fill omitted gaps without rewriting."""
    if not skill:
        return []
    cursor = 0
    ranges: list[tuple[int, int, str]] = []
    for row in segments:
        if not isinstance(row, dict):
            raise ValueError("segment must be an object")
        span = str(row.get("span", ""))
        if not span:
            raise ValueError("segment span is empty")
        start = skill.find(span, cursor)
        if start < 0:
            raise ValueError("segment is not an ordered exact substring")
        if start > cursor:
            ranges.append((cursor, start, _infer_type(skill[cursor:start])))
        end = start + len(span)
        ranges.append((start, end, str(row.get("type", "other"))))
        cursor = end
    if cursor < len(skill):
        ranges.append((cursor, len(skill), _infer_type(skill[cursor:])))
    return _atoms_from_ranges(skill, ranges)


class AtomicSegmenter:
    """Cached exact-span segmenter with an optional optimizer-LLM backend."""

    def __init__(
        self,
        mode: str = "llm",
        *,
        llm_call: Callable[..., tuple[str, dict]] | None = None,
        retries: int = 2,
        cache_dir: str = "",
        max_completion_tokens: int = 65536,
    ) -> None:
        self.mode = str(mode or "llm").strip().lower()
        self.llm_call = llm_call
        self.retries = max(1, int(retries))
        self.max_completion_tokens = max(1, int(max_completion_tokens))
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else ""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self._cache: dict[str, list[SkillAtom]] = {}

    def segment(self, skill: str) -> tuple[list[SkillAtom], dict]:
        digest = hashlib.sha256(skill.encode("utf-8")).hexdigest()
        if digest in self._cache:
            return self._cache[digest], {"mode": self.mode, "cache_hit": True}
        cache_path = (
            os.path.join(self.cache_dir, f"{digest}.json")
            if self.cache_dir
            else ""
        )
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                payload = json.load(f)
            atoms = [SkillAtom(**row) for row in payload.get("atoms", [])]
            if "".join(atom.span for atom in atoms) != skill:
                raise ValueError(f"invalid persisted atom cache: {cache_path}")
            self._cache[digest] = atoms
            metadata = dict(payload.get("metadata", {}))
            metadata["cache_hit"] = True
            metadata["cache_source"] = "disk"
            return atoms, metadata

        metadata = {"mode": self.mode, "cache_hit": False, "fallback": False}
        atoms: list[SkillAtom] | None = None
        if self.mode == "llm" and self.llm_call is not None and skill.strip():
            user = (
                "## Skill document\n"
                "<skill>\n"
                f"{skill}"
                "\n</skill>\n\n"
                "Insert boundaries only and return the required JSON."
            )
            for attempt in range(1, self.retries + 1):
                try:
                    response, _ = self.llm_call(
                        system=SEGMENTER_SYSTEM_PROMPT,
                        user=user,
                        max_completion_tokens=self.max_completion_tokens,
                        retries=1,
                        stage="regularization_segment",
                    )
                    payload = extract_json(response)
                    segments = payload.get("segments", []) if isinstance(payload, dict) else []
                    atoms = exact_span_atoms(skill, segments)
                    metadata["attempts"] = attempt
                    metadata.pop("last_error", None)
                    break
                except Exception as exc:  # noqa: BLE001
                    metadata["last_error"] = f"{type(exc).__name__}: {exc}"

        if atoms is None:
            atoms = rule_segment(skill)
            metadata["fallback"] = self.mode == "llm"
            metadata["effective_mode"] = "rule"

        self._cache[digest] = atoms
        metadata["atom_count"] = len(atoms)
        metadata["skill_sha256"] = digest
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": metadata,
                        "atoms": [atom.to_dict() for atom in atoms],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        return atoms, metadata


def atoms_to_json(atoms: list[SkillAtom]) -> str:
    return json.dumps([atom.to_dict() for atom in atoms], ensure_ascii=False, indent=2)
