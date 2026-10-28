"""Validated in-memory repair transactions; never writes a live skill library.

All paths are library-relative. Auxiliary files are preserved, not silently
discarded when a SKILL.md is merged. Callers commit only the returned snapshot.
"""
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

from .policy import META_SKILLS


def validate_path(path):
    parts = PurePosixPath(path).parts
    if (not parts or path != PurePosixPath(path).as_posix() or
            PurePosixPath(path).is_absolute() or ".." in parts or
            parts[0] in META_SKILLS or len(parts) < 2):
        raise ValueError(f"Invalid repair path: {path}")


@dataclass(frozen=True)
class Window:
    path: str
    start: int
    end: int
    text: str


def apply_repair(candidate: Mapping[str, str], response: dict, *,
                 source: str, history: set[str], mode: str,
                 windows: Mapping[str, Window] | None = None):
    """Apply one model decision atomically. Invalid output never partly applies.

    Create uses whole-skill replacements. Modify uses exact indexed windows;
    positions refer to the immutable input candidate, not intermediate edits.
    """
    operation = response.get("operation")
    if operation not in {"NOOP", "MERGE", "REWRITE", "DELETE"}:
        raise ValueError("Unknown repair operation")
    if mode not in {"create", "modify"}:
        raise ValueError("Unknown repair mode")
    changes = response.get("changes", {})
    if not isinstance(changes, dict):
        raise ValueError("changes must be an object")
    if operation == "NOOP":
        if changes:
            raise ValueError("NOOP cannot change candidate")
        return dict(candidate)
    if not changes:
        raise ValueError("Non-NOOP requires changes")
    allowed = {source} | (history if operation == "MERGE" else set())
    result = dict(candidate)
    if mode == "create":
        for path, text in changes.items():
            validate_path(path)
            if path not in allowed or path not in candidate:
                raise ValueError("Out-of-scope skill change")
            if text is None:
                if path != source or operation not in {"DELETE", "MERGE"}:
                    raise ValueError("Only redundant new skill may be deleted")
                slug = source.split('/')[0] + '/'
                if any(p.startswith(slug) and p != source for p in candidate):
                    raise ValueError("Cannot delete a skill owning auxiliary files")
                result.pop(path)
            elif not isinstance(text, str) or not text.strip():
                raise ValueError("Invalid skill text")
            else:
                if operation == "DELETE":
                    raise ValueError("DELETE cannot rewrite")
                result[path] = text
        if operation == "DELETE" and changes != {source: None}:
            raise ValueError("DELETE removes only the new skill")
    else:
        edits = {}
        for window_id, text in changes.items():
            if windows is None or window_id not in windows:
                raise ValueError(f"Unknown atom window {window_id!r}; allowed IDs: {sorted(windows or {})}")
            w = windows[window_id]
            validate_path(w.path)
            if operation != 'MERGE' and window_id.startswith('history_'):
                raise ValueError('Only MERGE may edit a historical atom')
            if w.start < 0 or w.end <= w.start or w.end > len(candidate.get(w.path, '')):
                raise ValueError('Invalid atom window coordinates')
            if w.path not in allowed or candidate.get(w.path, '')[w.start:w.end] != w.text:
                raise ValueError("Stale or out-of-scope atom window")
            if not isinstance(text, str):
                raise ValueError("Replacement must be a string")
            if operation == "DELETE" and text:
                raise ValueError("DELETE requires empty replacement")
            edits.setdefault(w.path, []).append((w.start, w.end, text))
        for path, items in edits.items():
            items.sort()
            if any(a[1] > b[0] for a, b in zip(items, items[1:])):
                raise ValueError("Overlapping atom windows")
            for start, end, text in reversed(items):
                result[path] = result[path][:start] + text + result[path][end:]
    return result


def paired_regression(old, new, *, minimum_drop):
    """Probe observations must be validated by the environment adapter.

    API/parse/verifier errors are inconclusive, never regression evidence.
    The two slots must have identical probe identity and grading identity.
    """
    if minimum_drop < 0:
        raise ValueError("minimum_drop must be nonnegative")
    if any(x.get("status") != "ok" for x in (old, new)):
        return "inconclusive"
    for key in ("probe_hash", "grading_hash"):
        if not old.get(key) or old[key] != new.get(key):
            raise ValueError("Unpaired CCV observations")
    import math
    scores = [float(x["score"]) for x in (old, new)]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise ValueError("Invalid normalized probe score")
    if old.get("passed") is not True:
        return "old_not_passed"
    return "regression" if new.get("passed") is False or scores[0] - scores[1] > minimum_drop else "no_regression"
