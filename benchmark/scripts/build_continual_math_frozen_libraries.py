#!/usr/bin/env python3
"""Materialise frozen skill libraries at a split point, for held-out evaluation.

This is read-only with respect to every existing run: it only consumes artifacts
that are already on disk, and writes a new directory tree.

Sources
-------
regularization      <run>/agent/regularization/step_NNN/final.json          (relative path -> text)
baseline   <run>/agent/checkpoints/step_NNN/state.tar.gz  (container filesystem)

The native create-skill / modify-skill helpers ship inside the image and are
never part of a library transaction, so they are excluded from every artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "benchmark/outputs/continualskillbench"
DEFAULT_OUTPUT_ROOT = REPO / "benchmark/runtime/continualskillbench/frozen_libraries"

HELPERS = {"create-skill", "modify-skill"}
CANONICAL = "root/.agents/skills/"

def _keep(relative: str) -> bool:
    parts = Path(relative).parts
    if not parts or parts[0] in HELPERS:
        return False
    return not Path(relative).is_absolute() and ".." not in parts


def from_final_json(path: Path) -> dict[str, bytes]:
    """Regularization snapshot: mapping of library-relative posix path -> file text."""
    raw = json.loads(path.read_text())
    out: dict[str, bytes] = {}
    for relative, text in raw.items():
        if not _keep(relative):
            continue
        out[relative] = text.encode("utf-8", "surrogateescape")
    return out


def from_state_tar(path: Path) -> dict[str, bytes]:
    """Baseline snapshot: container filesystem archive, canonical skills root."""
    out: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith(CANONICAL):
                continue
            relative = member.name[len(CANONICAL):]
            if not _keep(relative):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            out[relative] = handle.read()
    return out


def write_library(files: dict[str, bytes], target: Path) -> dict:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    digests = {}
    for relative, payload in sorted(files.items()):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        digests[relative] = hashlib.sha256(payload).hexdigest()
    skills = sorted({Path(relative).parts[0] for relative in files})
    return {
        "path": str(target.relative_to(REPO)),
        "skills": skills,
        "n_skills": len(skills),
        "n_files": len(files),
        "chars": sum(len(payload) for payload in files.values()),
        "sha256": digests,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularization-source", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    for label, path in (("regularization", args.regularization_source), ("baseline", args.baseline_source)):
        if not path.is_file():
            print(f"ERROR: {label} source missing: {path}", file=sys.stderr)
            return 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"split": 50, "note": "libraries learned from subtasks 1-50"}

    for name, files, source in (
        ("empty", {}, None),
        ("baseline_step050", from_state_tar(args.baseline_source), args.baseline_source),
        ("regularization_step050", from_final_json(args.regularization_source), args.regularization_source),
    ):
        entry = write_library(files, args.output_root / name)
        if source is not None:
            entry["source"] = str(source.relative_to(REPO))
        manifest[name] = entry
        print(
            f"{name:20s} {entry['n_skills']:3d} skills  "
            f"{entry['n_files']:3d} files  {entry['chars']:7,d} chars"
        )

    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"\nmanifest -> {(args.output_root / 'manifest.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
