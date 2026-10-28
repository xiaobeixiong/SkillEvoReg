"""SkillEvoReg training-time regularisation primitives for skill evolution.

The benchmark has no independent validation split.  These helpers therefore
operate at the proposal boundary only:

* split SKILL.md into deterministic exact-cover Markdown atoms;
* mask a small seeded subset in the task agent's training-time skill view;
* reuse that exact masked view when SkillAuthor proposes the update;
* measure whole-family complexity before and after a proposed patch;
* render complexity guidance and project proposals into a hard budget.

The live library always keeps complete text.  A per-trial read-only view hides
selected atom bodies from both the task agent and SkillAuthor.  Placeholders
carry only id/type/location information and are replaced with the exact host-
side originals before a returned patch may mutate the live library.
"""

from __future__ import annotations

import hashlib
import random
import re
import shutil
from dataclasses import dataclass
from typing import Any

from skillevolbench.schemas import OperationType, SkillPatch
from skillevolbench.stores.library_store import skill_id_to_slug


PLACEHOLDER_PREFIX = "<MASKED_SKILL_ATOM"
_PLACEHOLDER_RE = re.compile(r"<MASKED_SKILL_ATOM\b[^>]*>")
_PLACEHOLDER_CLOSE_RE = re.compile(r"</MASKED_SKILL_ATOM\s*>")
_ANY_PLACEHOLDER_RE = re.compile(r"</?MASKED_SKILL_ATOM\b[^>]*>")


class DropoutRoundTripError(ValueError):
    """A full-file proposal failed to preserve temporary dropout markers."""


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Atom:
    atom_id: str
    atom_type: str
    span: str
    start: int
    end: int


def _infer_type(span: str) -> str:
    stripped = span.strip()
    lower = stripped.lower()
    if stripped.startswith("```"):
        return "code"
    if re.search(r"\b(example|for example|e\.g\.)\b", lower):
        return "example"
    if re.search(r"\b(if .* fails?|failure|verify|check|validate)\b", lower):
        return "failure_check"
    if re.search(r"\b(except|unless|edge case|special case)\b", lower):
        return "exception"
    if re.search(r"\b(must|never|only|ensure|required|do not)\b", lower):
        return "constraint"
    if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", span):
        return "procedure_step"
    return "other"


def rule_segment(text: str) -> list[Atom]:
    """Return deterministic, exact-cover Markdown atoms."""
    if not text:
        return []
    boundaries = {0, len(text)}
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
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

    atoms: list[Atom] = []
    ordered = sorted(boundaries)
    for index, (start, end) in enumerate(zip(ordered, ordered[1:])):
        if end <= start:
            continue
        span = text[start:end]
        atoms.append(Atom(
            atom_id=f"a{index:04d}",
            atom_type=_infer_type(span),
            span=span,
            start=start,
            end=end,
        ))
    if "".join(atom.span for atom in atoms) != text:
        raise ValueError("rule segmentation did not preserve SKILL.md")
    return atoms


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def mask_skill_texts(
    texts: dict[str, str],
    *,
    probability: float,
    seed: int,
    max_atoms: int,
    min_eligible_atoms: int,
    ensure_at_least_one: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Mask atoms across all same-family SKILL.md documents."""
    probability = min(max(float(probability), 0.0), 1.0)
    rows: list[tuple[str, Atom]] = []
    by_skill: dict[str, list[Atom]] = {}
    for skill_id, text in texts.items():
        atoms = [a for a in rule_segment(text) if a.span.strip()]
        by_skill[skill_id] = atoms
        rows.extend((skill_id, atom) for atom in atoms)

    if probability <= 0.0 or len(rows) < int(min_eligible_atoms):
        return dict(texts), {
            "enabled": probability > 0.0,
            "probability": probability,
            "eligible_atoms": len(rows),
            "masked_count": 0,
            "masked_atoms": [],
            "seed": seed,
        }

    rng = random.Random(seed)
    chosen = [row for row in rows if rng.random() < probability]
    if ensure_at_least_one and not chosen:
        chosen = [rng.choice(rows)]
    if max_atoms > 0 and len(chosen) > max_atoms:
        chosen = rng.sample(chosen, max_atoms)

    chosen_by_skill: dict[str, list[Atom]] = {}
    for skill_id, atom in chosen:
        chosen_by_skill.setdefault(skill_id, []).append(atom)
    output = dict(texts)
    report_rows: list[dict[str, Any]] = []
    mask_index = 0
    for skill_id, atoms in chosen_by_skill.items():
        pieces: list[str] = []
        cursor = 0
        for atom in sorted(atoms, key=lambda item: item.start):
            mask_index += 1
            pieces.append(texts[skill_id][cursor:atom.start])
            leading = atom.span[: len(atom.span) - len(atom.span.lstrip())]
            trailing = atom.span[len(atom.span.rstrip()):]
            marker = (
                f'<MASKED_SKILL_ATOM id="m{mask_index}" '
                f'type="{atom.atom_type}">'
            )
            pieces.append(f"{leading}{marker}{trailing}")
            cursor = atom.end
            report_rows.append({
                "mask_id": f"m{mask_index}",
                "skill_id": skill_id,
                "atom_id": atom.atom_id,
                "atom_type": atom.atom_type,
                "original_chars": len(atom.span),
                "original_sha256": hashlib.sha256(
                    atom.span.encode("utf-8")
                ).hexdigest(),
                "original_span": atom.span,
                # Leading/trailing whitespace stays around the marker in the
                # masked prompt. Replacing the marker with this core restores
                # the original atom byte-for-byte.
                "restoration_text": atom.span[
                    len(leading):len(atom.span) - len(trailing)
                    if trailing else len(atom.span)
                ],
                "marker": marker,
                "preview": atom.span[:240],
            })
        pieces.append(texts[skill_id][cursor:])
        output[skill_id] = "".join(pieces)
        selected_ids = {atom.atom_id for atom in atoms}
        visible_atoms = [
            atom for atom in by_skill[skill_id]
            if atom.atom_id not in selected_ids
        ]
        # Structural anchors are complete visible atoms rather than fragile
        # character windows.  A neighbouring atom may be freely rewritten;
        # the nearest farther atom that survives still proves that the marker
        # has not crossed an existing rule.  The anchor text remains host-only
        # restoration metadata and is never placed in the training view.
        for row in report_rows:
            if row["skill_id"] != skill_id:
                continue
            source_atom = next(
                atom for atom in atoms if atom.atom_id == row["atom_id"]
            )
            row["left_anchor_atoms"] = [
                {
                    "atom_id": atom.atom_id,
                    "span": atom.span,
                    "sha256": hashlib.sha256(
                        atom.span.encode("utf-8")
                    ).hexdigest(),
                }
                for atom in reversed(visible_atoms)
                if atom.end <= source_atom.start
            ]
            row["right_anchor_atoms"] = [
                {
                    "atom_id": atom.atom_id,
                    "span": atom.span,
                    "sha256": hashlib.sha256(
                        atom.span.encode("utf-8")
                    ).hexdigest(),
                }
                for atom in visible_atoms
                if atom.start >= source_atom.end
            ]

    return output, {
        "enabled": True,
        "probability": probability,
        "eligible_atoms": len(rows),
        "masked_count": len(report_rows),
        "masked_atoms": report_rows,
        "seed": seed,
    }


def dropout_instruction(masked_count: int) -> str:
    if masked_count <= 0:
        return ""
    return (
        "SkillEvoReg training-time atomic dropout is active. The Current "
        f"Skill contains {masked_count} <MASKED_SKILL_ATOM ...> marker(s). "
        "The same hidden atom bodies were unavailable to the task agent during "
        "the training execution that produced this trajectory. You may use a "
        "marker's id, type, and location, but the hidden text is intentionally "
        "unavailable. It will be restored automatically after proposal "
        "generation. If you upsert an "
        "existing masked SKILL.md, copy every marker from that file exactly "
        "once, unchanged, as a standalone atom in its original structural "
        "slot. You may freely edit adjacent visible atoms. Do not reconstruct, "
        "infer, compensate for, generalize, omit, duplicate, or move hidden "
        "content. Do not edit across a marker boundary. Propose only episode-"
        "supported changes to visible content."
    )


def materialize_training_skill_view(
    live_active_path: str | Any,
    view_path: str | Any,
    masked_texts: dict[str, str],
) -> None:
    """Create an isolated task-agent view without mutating the live library.

    The destination is cleared on every trial attempt, then receives the full
    active library (including Tier-3 assets).  Only selected ``SKILL.md`` files
    are replaced by their masked versions.  The view contains no restoration
    metadata or original atom bodies.
    """
    from pathlib import Path

    source = Path(live_active_path)
    destination = Path(view_path)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    for skill_id, text in masked_texts.items():
        target = destination / skill_id_to_slug(skill_id) / "SKILL.md"
        if not target.is_file():
            raise FileNotFoundError(
                f"masked training skill is absent from active view: {skill_id}"
            )
        target.write_text(text)


def redact_masked_atom_bodies(
    text: str,
    dropout_report: dict[str, Any] | None,
) -> tuple[str, int]:
    """Remove exact hidden bodies from trajectory/history optimizer input."""
    redacted = str(text or "")
    count = 0
    for row in (dropout_report or {}).get("masked_atoms", []) or []:
        marker = str(row.get("marker", "")) or (
            f'<MASKED_SKILL_ATOM id="{row.get("mask_id", "?")}" '
            f'type="{row.get("atom_type", "other")}">'
        )
        original = str(row.get("original_span", ""))
        restoration = str(row.get("restoration_text", ""))
        variants = [original, original.strip(), restoration]
        for variant in dict.fromkeys(v for v in variants if v):
            occurrences = redacted.count(variant)
            if occurrences:
                redacted = redacted.replace(variant, marker)
                count += occurrences
    return redacted, count


def complexity_instruction(
    current: dict[str, Any], *, complexity_lambda: float,
    free_growth_budget: float,
) -> str:
    """Render a proposal-time soft regulariser without adding a selection gate."""
    return (
        "\nSkillEvoReg complexity regularisation is active. The current "
        f"family skill complexity is {float(current.get('total', 0.0)):.4f} "
        f"({int(current.get('token_count', 0))} tokens, "
        f"{int(current.get('atom_count', 0))} atoms). Prefer consolidating, "
        "generalising, or deleting redundant rules over appending task-local "
        "special cases. Treat positive relative growth beyond "
        f"{float(free_growth_budget):.1%} as a soft cost with weight "
        f"{float(complexity_lambda):.4f}. Add complexity only when the "
        "cumulative episode evidence requires transferable procedure. This is "
        "proposal guidance, not a request to encode scores or thresholds in "
        "the skill.\n"
    )


def _all_occurrences(text: str, needle: str) -> list[int]:
    """Return non-overlapping exact occurrence offsets."""
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


def restore_patch_placeholders(
    patch: SkillPatch,
    dropout_report: dict[str, Any] | None,
) -> tuple[SkillPatch, dict[str, Any]]:
    """Restore masked atoms in a full-file proposal without semantic guessing.

    SkillEvolBench revisions upsert complete SKILL.md files. The model must
    round-trip each marker for every masked skill it rewrites; otherwise the
    patch is rejected and retried. A masked skill omitted from ``upsert_files``
    remains unchanged in the library and therefore needs no marker.
    """
    rows = list((dropout_report or {}).get("masked_atoms", []) or [])
    upserts = dict(patch.upsert_files)
    restored = 0
    untouched_skill_ids: list[str] = []
    errors: list[str] = []
    replacements: dict[str, list[dict[str, Any]]] = {}

    marker_order_by_path: dict[str, list[str]] = {}
    for row in rows:
        skill_id = str(row.get("skill_id", ""))
        slug = skill_id_to_slug(skill_id)
        expected = f"{slug}/SKILL.md"
        paths = [
            path for path in upserts
            if path == expected or path.endswith(f"/{expected}")
        ]
        deleted = any(
            path == slug or path == expected
            or path.endswith(f"/{expected}")
            for path in patch.delete_paths
        )
        if deleted:
            errors.append(f"masked skill cannot be deleted: {skill_id}")
            continue
        if not paths:
            untouched_skill_ids.append(skill_id)
            continue
        if len(paths) != 1:
            errors.append(f"masked skill has ambiguous upsert paths: {skill_id}")
            continue

        path = paths[0]
        marker = str(row.get("marker", ""))
        restoration = str(row.get("restoration_text", ""))
        original = str(row.get("original_span", ""))
        count = upserts[path].count(marker) if marker else 0
        if count != 1:
            errors.append(
                f"marker {row.get('mask_id', '?')} for {skill_id} "
                f"must appear exactly once; found {count}"
            )
            continue
        position = upserts[path].index(marker)
        line_start = upserts[path].rfind("\n", 0, position) + 1
        line_end = upserts[path].find("\n", position + len(marker))
        if line_end < 0:
            line_end = len(upserts[path])
        if upserts[path][line_start:line_end].strip() != marker:
            errors.append(
                f"marker {row.get('mask_id', '?')} for {skill_id} must "
                "remain in a standalone atom slot"
            )
            continue

        left_anchors = list(row.get("left_anchor_atoms", []) or [])
        right_anchors = list(row.get("right_anchor_atoms", []) or [])
        if left_anchors or right_anchors:
            structural_evidence = False
            for anchor in left_anchors:
                span = str(anchor.get("span", ""))
                if not span or span not in upserts[path]:
                    continue
                structural_evidence = True
                if any(
                    start + len(span) <= position
                    for start in _all_occurrences(upserts[path], span)
                ):
                    break
                errors.append(
                    f"marker {row.get('mask_id', '?')} for {skill_id} moved "
                    "or changed its structural atom order"
                )
                break
            if errors:
                continue
            for anchor in right_anchors:
                span = str(anchor.get("span", ""))
                if not span or span not in upserts[path]:
                    continue
                structural_evidence = True
                if any(
                    start >= position + len(marker)
                    for start in _all_occurrences(upserts[path], span)
                ):
                    break
                errors.append(
                    f"marker {row.get('mask_id', '?')} for {skill_id} moved "
                    "or changed its structural atom order"
                )
                break
            if errors:
                continue
            if not structural_evidence:
                errors.append(
                    f"marker {row.get('mask_id', '?')} for {skill_id} moved "
                    "or changed: no surrounding structural atom anchor survived"
                )
                continue
        else:
            # Backward-compatible validation for reports produced before atom
            # anchors were added. New proposals never take this branch.
            left_context = str(row.get("left_context", ""))
            right_context = str(row.get("right_context", ""))
            if left_context and not upserts[path][:position].endswith(left_context):
                errors.append(
                    f"marker {row.get('mask_id', '?')} for {skill_id} "
                    "moved or changed its left context"
                )
                continue
            if right_context and not upserts[path][
                position + len(marker):
            ].startswith(right_context):
                errors.append(
                    f"marker {row.get('mask_id', '?')} for {skill_id} "
                    "moved or changed its right context"
                )
                continue
        marker_order_by_path.setdefault(path, []).append(marker)
        replacements.setdefault(path, []).append(row)

    for path, expected_markers in marker_order_by_path.items():
        actual_markers = _PLACEHOLDER_RE.findall(upserts[path])
        expected_set = set(expected_markers)
        relevant_actual = [m for m in actual_markers if m in expected_set]
        if relevant_actual != expected_markers:
            errors.append(
                f"masked atoms moved or changed their relative order in {path}"
            )

    if errors:
        raise DropoutRoundTripError("; ".join(errors))

    for path, path_rows in replacements.items():
        for row in path_rows:
            marker = str(row.get("marker", ""))
            restoration = str(row.get("restoration_text", ""))
            upserts[path] = upserts[path].replace(marker, restoration, 1)
            original = str(row.get("original_span", ""))
            if original and original not in upserts[path]:
                errors.append(
                    f"marker restoration was not byte-exact for "
                    f"{row.get('mask_id', '?')}/{row.get('skill_id', '')}"
                )
                continue
            restored += 1

        # The training marker is a standalone sentinel, not an XML element.
        # Some full-file authors nevertheless emit a synthetic closing tag.
        # It carries no masked content and must not survive restoration.
        upserts[path] = _PLACEHOLDER_CLOSE_RE.sub("", upserts[path])

    if errors:
        raise DropoutRoundTripError("; ".join(errors))

    unknown = {
        path: _ANY_PLACEHOLDER_RE.findall(content)
        for path, content in upserts.items()
        if _ANY_PLACEHOLDER_RE.search(content)
    }
    if unknown:
        errors.append(f"unrestored or unknown dropout markers remain: {unknown}")
    if errors:
        raise DropoutRoundTripError("; ".join(errors))
    return patch.model_copy(update={"upsert_files": upserts}), {
        "masked_count": len(rows),
        "restored_count": restored,
        "untouched_skill_ids": sorted(set(untouched_skill_ids)),
        "preservation_ok": True,
        "structural_anchor_mode": "atom_order",
    }


def clean_patch_placeholders(patch: SkillPatch) -> tuple[SkillPatch, int]:
    """Legacy safety seam: never silently delete a dropout marker."""
    remaining = sum(
        len(_ANY_PLACEHOLDER_RE.findall(content))
        for path, content in patch.upsert_files.items()
        if path.endswith("SKILL.md")
    )
    if remaining:
        raise DropoutRoundTripError(
            "refusing to delete unresolved dropout markers; use "
            "restore_patch_placeholders with the matching dropout report"
        )
    return patch, 0


def _code_cost(text: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    branches = len(re.findall(
        r"(?m)^\s*(?:if|elif|for|while|try|except|case|match)\b", text,
    ))
    dependencies = len(re.findall(
        r"(?m)^\s*(?:from\s+\S+\s+import|import\s+\S+)\b", text,
    ))
    return len(lines) / 20.0 + 2.0 * branches / 5.0 + 2.0 * dependencies / 3.0


def _measure_files(
    files: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    """Measure ``{path: (skill_id, text)}`` with the public metric."""
    token_count = 0
    atom_count = 0
    code_complexity = 0.0
    rows: list[dict[str, Any]] = []
    for path, (skill_id, text) in sorted(files.items()):
        rel_path = path.split("/", 1)[-1]
        tokens = len(_TOKEN_RE.findall(text))
        if rel_path.endswith("SKILL.md"):
            atoms = rule_segment(text)
            n_atoms = len(atoms)
            code = sum(
                _code_cost(atom.span)
                for atom in atoms if atom.atom_type == "code"
            )
        else:
            n_atoms = 1 if text.strip() else 0
            code = _code_cost(text)
        token_count += tokens
        atom_count += n_atoms
        code_complexity += code
        rows.append({
            "skill_id": skill_id,
            "path": rel_path,
            "tokens": tokens,
            "atoms": n_atoms,
            "code_complexity": round(code, 6),
        })
    total = token_count / 100.0 + 0.2 * atom_count + code_complexity
    return {
        "total": round(total, 8),
        "token_count": token_count,
        "atom_count": atom_count,
        "code_complexity": round(code_complexity, 8),
        "files": rows,
    }


def _family_files(library: Any, family_id: str) -> dict[str, tuple[str, str]]:
    files: dict[str, tuple[str, str]] = {}
    for entry in library.skills_in_family(family_id):
        version = library.get_skill(entry.skill_id)
        slug = skill_id_to_slug(entry.skill_id)
        for rel_path, text in version.files.items():
            files[f"{slug}/{rel_path}"] = (entry.skill_id, text)
    return files


def measure_family_complexity(library: Any, family_id: str) -> dict[str, Any]:
    """Measure all active files belonging to one latent-skill family."""
    return _measure_files(_family_files(library, family_id))


def measure_patch_complexity(
    library: Any, family_id: str, patch: SkillPatch,
) -> dict[str, Any]:
    """Measure the family tree that would result from ``patch`` without IO."""
    files = _family_files(library, family_id)
    slug_to_sid = {
        skill_id_to_slug(entry.skill_id): entry.skill_id
        for entry in library.skills_in_family(family_id)
    }
    slug_to_sid.update({
        skill_id_to_slug(skill_id): skill_id
        for skill_id in patch.target_skill_ids
    })
    for rel_path in patch.delete_paths:
        slug = rel_path.split("/", 1)[0]
        for path in [path for path in files if path.split("/", 1)[0] == slug]:
            files.pop(path)
    for rel_path, text in patch.upsert_files.items():
        slug = rel_path.split("/", 1)[0]
        skill_id = slug_to_sid.get(slug)
        if skill_id is not None:
            files[rel_path] = (skill_id, text)
    return _measure_files(files)


def complexity_budget(
    anchor: dict[str, Any], *, max_relative_growth: float, max_atom_growth: int,
) -> dict[str, float | int]:
    return {
        "max_total": round(
            float(anchor.get("total", 0.0)) * (1.0 + max_relative_growth), 8
        ),
        "max_atoms": int(anchor.get("atom_count", 0)) + max_atom_growth,
    }


def complexity_within_budget(
    measured: dict[str, Any], budget: dict[str, float | int],
) -> bool:
    return (
        float(measured.get("total", 0.0)) <= float(budget["max_total"])
        and int(measured.get("atom_count", 0)) <= int(budget["max_atoms"])
    )


def local_edit_trust_region_errors(
    *,
    abnormal_modules: set[str] | tuple[str, ...] | list[str],
    measured: dict[str, Any],
    anchor: dict[str, Any] | None,
    current_texts: dict[str, str],
    repaired_texts: dict[str, str],
    repair_scope_skill_ids: list[str] | tuple[str, ...],
    source_candidate_measurement: dict[str, Any] | None = None,
) -> list[str]:
    """Validate causal repair bounds without running another task evaluation.

    Complexity direction relative to the source candidate is telemetry, not an
    acceptance gate. A repair may not retroactively compress below an explicit
    pre-update family anchor. A CCV-only signal licenses replacement/removal
    of its one responsible atom, not a compression pass.
    """
    modules = set(abnormal_modules)
    errors: list[str] = []
    anchor = anchor or {}
    if "complexity" in modules and anchor:
        if (
            float(measured.get("total", 0.0))
            < float(anchor.get("total", 0.0))
            or int(measured.get("atom_count", 0))
            < int(anchor.get("atom_count", 0))
        ):
            errors.append(
                "repair undershot the pre-update complexity anchor; "
                "remove only the source patch excess"
            )
    if "ccv" in modules and "complexity" not in modules:
        for skill_id in repair_scope_skill_ids:
            before_atoms = sum(
                1 for atom in rule_segment(current_texts.get(skill_id, ""))
                if atom.span.strip()
            )
            after_atoms = sum(
                1 for atom in rule_segment(repaired_texts.get(skill_id, ""))
                if atom.span.strip()
            )
            if after_atoms < max(0, before_atoms - 1):
                errors.append(
                    f"CCV-only repair deleted more than its one responsible "
                    f"atom in {skill_id}"
                )
    return errors


def local_edit_complexity_telemetry(
    *,
    abnormal_modules: set[str] | tuple[str, ...] | list[str],
    measured: dict[str, Any],
    source_candidate_measurement: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe repair direction without turning it into an acceptance gate."""
    source = source_candidate_measurement or {}
    if "complexity" not in set(abnormal_modules) or not source:
        return {}
    before_total = float(source.get("total", 0.0))
    after_total = float(measured.get("total", 0.0))
    delta_total = after_total - before_total
    if delta_total < 0:
        direction = "decreased"
    elif delta_total > 0:
        direction = "increased"
    else:
        direction = "unchanged"
    before_atoms = int(source.get("atom_count", 0))
    after_atoms = int(measured.get("atom_count", 0))
    before_tokens = int(source.get("token_count", 0))
    after_tokens = int(measured.get("token_count", 0))
    return {
        "source_candidate_total": before_total,
        "repair_total": after_total,
        "delta_total": round(delta_total, 8),
        "source_candidate_atom_count": before_atoms,
        "repair_atom_count": after_atoms,
        "delta_atom_count": after_atoms - before_atoms,
        "source_candidate_token_count": before_tokens,
        "repair_token_count": after_tokens,
        "delta_token_count": after_tokens - before_tokens,
        "direction": direction,
        "acceptance_gate": False,
    }


def patch_skill_text_overrides(
    library: Any, family_id: str, patch: SkillPatch,
) -> dict[str, str]:
    """Return candidate SKILL.md texts for a repair-generation prompt."""
    files = _family_files(library, family_id)
    slug_to_sid = {
        skill_id_to_slug(entry.skill_id): entry.skill_id
        for entry in library.skills_in_family(family_id)
    }
    slug_to_sid.update({
        skill_id_to_slug(skill_id): skill_id
        for skill_id in patch.target_skill_ids
    })
    for rel_path, text in patch.upsert_files.items():
        slug = rel_path.split("/", 1)[0]
        if slug in slug_to_sid:
            files[rel_path] = (slug_to_sid[slug], text)
    output: dict[str, str] = {}
    for path, (skill_id, text) in files.items():
        if path.endswith("/SKILL.md"):
            output[skill_id] = text
    return output


def project_patch_to_existing_family(
    library: Any, family_id: str, patch: SkillPatch,
) -> SkillPatch:
    """Safe hard-budget fallback: project onto the unchanged family tree.

    This is not a performance rejection or rollback: the generated candidate
    is projected into the configured complexity feasible set before it ever
    reaches the library.  Preserving the old tree is safer than truncating a
    procedure at an arbitrary Markdown boundary.
    """
    current = _family_files(library, family_id)
    skill_ids = [entry.skill_id for entry in library.skills_in_family(family_id)]
    if not skill_ids:
        return patch
    return patch.model_copy(update={
        "summary": f"Hard-complexity projection of: {patch.summary}"[:500],
        "upsert_files": {path: text for path, (_, text) in current.items()},
        "delete_paths": [],
        "target_skill_ids": skill_ids,
        "operation_type": OperationType.REVISE,
    })


def hard_complexity_instruction(
    budget: dict[str, float | int], measured: dict[str, Any], attempt: int,
) -> str:
    return (
        "\nHARD COMPLEXITY REPAIR REQUIRED. Rewrite the complete candidate "
        "skill family while preserving only transferable procedure. The "
        f"current candidate measures total={float(measured['total']):.4f}, "
        f"atoms={int(measured['atom_count'])}. The returned patch MUST measure "
        f"total <= {float(budget['max_total']):.4f} and atoms <= "
        f"{int(budget['max_atoms'])}. Prefer one general decision rule over "
        "several narrow exceptions. Merge adjacent bullets that share the "
        "same trigger, action, or verification step. Do not represent "
        "examples, individual filenames, constants, test cases, or one-off "
        "observations as separate rules unless they define a genuinely "
        "transferable boundary condition. Do not satisfy the token budget by "
        "splitting prose into more headings, bullets, or short atoms: atom "
        "count is a hard constraint, not a formatting preference. Consolidate "
        "duplicate or task-specific rules, remove unsupported detail, and "
        "preserve the smallest complete workflow that generalizes across the "
        "observed episodes. Do not merely describe the budget. "
        f"This is repair attempt {attempt}.\n"
    )


__all__ = [
    "DropoutRoundTripError",
    "PLACEHOLDER_PREFIX",
    "clean_patch_placeholders",
    "complexity_instruction",
    "dropout_instruction",
    "complexity_budget",
    "complexity_within_budget",
    "hard_complexity_instruction",
    "mask_skill_texts",
    "measure_family_complexity",
    "measure_patch_complexity",
    "patch_skill_text_overrides",
    "project_patch_to_existing_family",
    "restore_patch_placeholders",
    "rule_segment",
    "local_edit_complexity_telemetry",
    "local_edit_trust_region_errors",
    "stable_seed",
]
