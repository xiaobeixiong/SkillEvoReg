"""Deterministic per-skill training masks, with fail-closed exact restoration."""
import hashlib
import math
import random
import re

from .policy import META_SKILLS


def mask_library(files, *, rate, max_atoms, ensure_at_least_one, seed, segment):
    if not 0 <= rate <= 1 or max_atoms < 0:
        raise ValueError("Invalid dropout settings")
    masked = dict(files)
    records = {}
    for path, text in sorted(files.items()):
        if not path.endswith('/SKILL.md') or path.split('/')[0] in META_SKILLS:
            continue
        front = re.match(r'\A---\r?\n.*?\r?\n---(?:\r?\n|\Z)', text, re.S)
        body_start = front.end() if front else 0
        atoms = [a for a in segment(text) if a.start >= body_start and
                 a.span.strip() and not re.match(r'^\s*#{1,6}\s', a.span)]
        count = min(max_atoms, len(atoms), max(
            1 if ensure_at_least_one and rate > 0 else 0,
            math.floor(len(atoms) * rate)))
        digest = hashlib.sha256(f'{seed}:{path}'.encode()).hexdigest()
        chosen = random.Random(int(digest, 16)).sample(atoms, count)
        for i, atom in enumerate(sorted(chosen, key=lambda a: a.start, reverse=True)):
            marker = f'<!-- CONTINUAL_DROPOUT_{digest[:16]}_{i} -->'
            if marker in text:
                raise ValueError('Mask collision')
            masked[path] = masked[path][:atom.start] + marker + '\n' + masked[path][atom.end:]
            records.setdefault(path, {})[marker + '\n'] = atom.span
    return masked, records


def restore_library(candidate, records, *, allow_deleted=False):
    restored = dict(candidate)
    recovered = []
    for path, mapping in records.items():
        if path not in restored:
            if allow_deleted:
                continue  # Native deletion is audited by Regularization, not silently undone.
            raise ValueError('Masked skill deleted before restoration')
        for marker, original in mapping.items():
            # Persistent sessions can reproduce the exact hidden text themselves.
            # Do not insert it twice. No fuzzy matching or guessed restoration.
            if restored[path].count(marker) == 0 and restored[path].count(original) == 1:
                recovered.append((path, marker))
                continue
            if restored[path].count(marker) != 1:
                raise ValueError('Missing or duplicated dropout marker')
            restored[path] = restored[path].replace(marker, original, 1)
            recovered.append((path, marker))
    # A native author may copy a temporary comment into another skill. Remove
    # only standalone, exact copies whose owning atom was successfully restored.
    # Missing owners, unknown IDs and malformed/inline markers still fail closed.
    for owner, marker in recovered:
        pattern = re.compile(r'(?m)^' + re.escape(marker.rstrip('\n')) + r'(?:\n|\Z)')
        for path in restored:
            if path != owner and path.endswith('/SKILL.md') and path.split('/')[0] not in META_SKILLS:
                restored[path] = pattern.sub('', restored[path])
    if any('CONTINUAL_DROPOUT_' in text for text in restored.values()):
        raise ValueError('Residual dropout marker')
    return restored
