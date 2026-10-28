"""Atomic skill dropout for training rollouts and candidate proposal."""
from __future__ import annotations

import random
import re
import hashlib
from dataclasses import dataclass

from skillopt.regularization.atoms import SkillAtom


PLACEHOLDER_PREFIX = "<MASKED_SKILL_ATOM"
_PLACEHOLDER_RE = re.compile(
    r'<MASKED_SKILL_ATOM\b[^>]*>'
)


@dataclass(frozen=True)
class DropoutResult:
    proposal_skill: str
    masked_atoms: list[dict]
    eligible_atoms: int
    probability: float
    seed: int

    def to_dict(self) -> dict:
        return {
            "masked_atoms": self.masked_atoms,
            "masked_count": len(self.masked_atoms),
            "eligible_atoms": self.eligible_atoms,
            "probability": self.probability,
            "seed": self.seed,
        }


def contains_placeholder(skill: str) -> bool:
    return PLACEHOLDER_PREFIX in skill or bool(_PLACEHOLDER_RE.search(skill))


def remove_placeholders(skill: str) -> tuple[str, int]:
    """Deterministically remove residual dropout markers.

    Deliberately do not rewrite surrounding Markdown. A leftover empty list
    item or heading is less intrusive and more auditable than an LLM cleanup
    pass, while the synthetic marker itself must never reach evaluation.
    """
    cleaned, count = _PLACEHOLDER_RE.subn("", skill)
    return cleaned, count


def apply_atomic_dropout(
    skill: str,
    atoms: list[SkillAtom],
    *,
    probability: float,
    seed: int,
    max_atoms: int = 0,
    ensure_at_least_one: bool = False,
) -> DropoutResult:
    probability = max(0.0, min(1.0, float(probability)))
    eligible = [
        atom for atom in atoms
        if atom.span.strip()
        and "<!-- SLOW_UPDATE_" not in atom.span
        and "<!-- APPENDIX_" not in atom.span
    ]
    rng = random.Random(seed)
    chosen = [atom for atom in eligible if rng.random() < probability]
    if ensure_at_least_one and eligible and not chosen:
        chosen = [rng.choice(eligible)]
    if max_atoms > 0 and len(chosen) > max_atoms:
        chosen = rng.sample(chosen, max_atoms)
        chosen.sort(key=lambda atom: atom.start)

    output: list[str] = []
    cursor = 0
    masked_rows: list[dict] = []
    selected_ids = {atom.atom_id for atom in chosen}
    for mask_index, atom in enumerate(chosen, 1):
        output.append(skill[cursor:atom.start])
        placeholder = (
            f'<MASKED_SKILL_ATOM id="m{mask_index}" type="{atom.type}">'
        )
        leading = atom.span[:len(atom.span) - len(atom.span.lstrip())]
        trailing = atom.span[len(atom.span.rstrip()):]
        output.append(f"{leading}{placeholder}{trailing}")
        masked_rows.append({
            "mask_id": f"m{mask_index}",
            "atom_id": atom.atom_id,
            "type": atom.type,
            "start": atom.start,
            "end": atom.end,
            "original_chars": len(atom.span),
            "original_span": atom.span,
            "span_preview": atom.span[:240],
            "span_sha256": hashlib.sha256(atom.span.encode("utf-8")).hexdigest(),
            "original_occurrences": skill.count(atom.span),
            "left_anchor_atoms": [
                {
                    "atom_id": anchor.atom_id,
                    "span": anchor.span,
                    "sha256": hashlib.sha256(
                        anchor.span.encode("utf-8")
                    ).hexdigest(),
                }
                for anchor in reversed(eligible)
                if anchor.atom_id not in selected_ids and anchor.end <= atom.start
            ],
            "right_anchor_atoms": [
                {
                    "atom_id": anchor.atom_id,
                    "span": anchor.span,
                    "sha256": hashlib.sha256(
                        anchor.span.encode("utf-8")
                    ).hexdigest(),
                }
                for anchor in eligible
                if anchor.atom_id not in selected_ids and anchor.start >= atom.end
            ],
        })
        cursor = atom.end
    output.append(skill[cursor:])
    return DropoutResult(
        proposal_skill="".join(output),
        masked_atoms=masked_rows,
        eligible_atoms=len(eligible),
        probability=probability,
        seed=seed,
    )


def dropout_instruction(masked_count: int) -> str:
    if masked_count <= 0:
        return ""
    return (
        "SkillEvoReg atomic dropout is active. The Current Skill contains "
        f"{masked_count} <MASKED_SKILL_ATOM ...> placeholder(s). Treat each as "
        "temporarily hidden procedural content that was also unavailable during "
        "this training rollout. You may know each hidden atom's id, type, and "
        "location, but its text is intentionally unavailable. The hidden atoms "
        "are restored automatically before the proposed patch is committed. Do "
        "not infer, recreate, replace, or compensate for an atom merely because "
        "it is masked. Do not target a marker or edit across its boundary. "
        "Propose only trajectory-supported edits to visible content. The markers "
        "are training-time regularisation and never authorize persistent deletion."
    )


def masked_atom_preservation_errors(
    candidate_skill: str,
    dropout_report: dict | None,
) -> list[str]:
    """Validate that training-time dropout did not become persistent pruning."""
    errors: list[str] = []
    for row in (dropout_report or {}).get("masked_atoms", []) or []:
        original = str(row.get("original_span", ""))
        expected_count = int(row.get("original_occurrences", 1) or 1)
        positions = _all_occurrences(candidate_skill, original)
        if original and len(positions) != expected_count:
            errors.append(
                "masked atom was not restored exactly once: "
                f"{row.get('mask_id', '?')}/{row.get('atom_id', '?')}"
            )
            continue
        if original and not any(
            _position_preserves_atom_order(
                candidate_skill,
                start=position,
                end=position + len(original),
                row=row,
            )
            for position in positions
        ):
            errors.append(
                "masked atom moved or changed its structural atom order: "
                f"{row.get('mask_id', '?')}/{row.get('atom_id', '?')}"
            )
    return errors


def _all_occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    offsets: list[int] = []
    cursor = 0
    while True:
        position = text.find(needle, cursor)
        if position < 0:
            return offsets
        offsets.append(position)
        cursor = position + len(needle)


def _position_preserves_atom_order(
    text: str,
    *,
    start: int,
    end: int,
    row: dict,
) -> bool:
    """Verify a protected atom against the nearest surviving atom anchors."""
    left_anchors = list(row.get("left_anchor_atoms", []) or [])
    right_anchors = list(row.get("right_anchor_atoms", []) or [])
    if not left_anchors and not right_anchors:
        # Backward-compatible reports only proved body preservation.
        return True
    structural_evidence = False
    for anchor in left_anchors:
        span = str(anchor.get("span", ""))
        occurrences = _all_occurrences(text, span)
        if not occurrences:
            continue
        structural_evidence = True
        if not any(position + len(span) <= start for position in occurrences):
            return False
        break
    for anchor in right_anchors:
        span = str(anchor.get("span", ""))
        occurrences = _all_occurrences(text, span)
        if not occurrences:
            continue
        structural_evidence = True
        if not any(position >= end for position in occurrences):
            return False
        break
    return structural_evidence


def redact_masked_atom_bodies(
    text: str,
    dropout_report: dict | None,
) -> tuple[str, int]:
    """Remove exact hidden atom bodies from optimizer-only history context."""
    redacted = str(text or "")
    count = 0
    for row in (dropout_report or {}).get("masked_atoms", []) or []:
        marker = (
            f'<MASKED_SKILL_ATOM id="{row.get("mask_id", "?")}" '
            f'type="{row.get("type", "other")}">'
        )
        original = str(row.get("original_span", ""))
        variants = [original, original.strip()]
        for variant in dict.fromkeys(v for v in variants if v):
            occurrences = redacted.count(variant)
            if occurrences:
                redacted = redacted.replace(variant, marker)
                count += occurrences
    return redacted, count
