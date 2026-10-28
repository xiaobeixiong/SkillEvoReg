"""Freeze a complete first-pass (T1--T3) SkillEvolBench anchor.

The exported directory deliberately has the same ``library/`` and
``stores/replay/`` layout as a run root, so it can be passed directly to
``--shared-learning-anchor-source``.  It contains no evaluation results and
is therefore safe to create while the source run is still evaluating T4--T6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from skillevolbench.schemas import ReplayRecord
from skillevolbench.stores import ReplayStore


_EXPECTED = {
    f"E{env}-LS{family}-T{task}"
    for env in range(1, 7)
    for family in range(1, 6)
    for task in range(1, 4)
}


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def export(source_run: Path, output_dir: Path) -> Path:
    source_run = source_run.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    config_path = source_run / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"source run configuration is missing: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("learning_passes", 1) != 1 or config.get("shared_learning_anchor_source"):
        raise RuntimeError("a learning anchor must be exported from a first-pass run")
    records_dir = source_run / "stores" / "replay" / "records"
    if not records_dir.is_dir():
        raise RuntimeError(f"source replay records are missing: {records_dir}")
    if output_dir.exists():
        raise FileExistsError(f"anchor destination already exists: {output_dir}")

    records: dict[str, Path] = {}
    for path in records_dir.glob("E*-T[123].json"):
        payload = json.loads(path.read_text())
        task_id = payload.get("task_id")
        if task_id in _EXPECTED and payload.get("replay_mode") == "primary":
            records[task_id] = path
    if set(records) != _EXPECTED:
        raise RuntimeError(
            "source must contain exactly the 90 primary T1--T3 records; "
            f"missing={sorted(_EXPECTED - set(records))[:3]} "
            f"extra={sorted(set(records) - _EXPECTED)[:3]}"
        )

    libraries: dict[str, str] = {}
    for env in (f"E{i}" for i in range(1, 7)):
        library = source_run / "library" / env
        if not (library / ".git").is_dir() or not (library / "manifest.yaml").is_file():
            raise RuntimeError(f"source library is incomplete: {library}")
        libraries[env] = _sha256_tree(library)

    try:
        for env in sorted(libraries):
            shutil.copytree(source_run / "library" / env, output_dir / "library" / env)
            # The target run must be able to revise the copied first-pass state.
            (output_dir / "library" / env / ".frozen").unlink(missing_ok=True)
        # ReplayStore owns both the JSON payloads and their SQLite index.
        # Copying just ``records/*.json`` makes a syntactically plausible but
        # unreadable anchor because ReplayStore intentionally queries sqlite.
        target_store = ReplayStore(output_dir / "stores" / "replay")
        try:
            for task_id, path in records.items():
                target_store.persist(ReplayRecord.model_validate(
                    json.loads(path.read_text())
                ))
        finally:
            target_store.close()
        manifest = {
            "schema_version": 1,
            "kind": "complete_t1_t3_learning_anchor",
            "source_run": str(source_run),
            "source_run_id": source_run.name,
            "learning_records": len(records),
            "library_tree_sha256": libraries,
        }
        manifest_path = output_dir / "anchor_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest_path
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Exported complete T1--T3 anchor: {export(args.source_run, args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
