"""Export exact post-T1 skills from a completed environment-scoped run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return proc.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _shared_replay_state(source_run: Path, family_id: str) -> bytes:
    task_id = f"{family_id}-T1"
    source = source_run / "stores" / "replay" / "records" / f"{task_id}.json"
    if not source.is_file():
        raise RuntimeError(f"source T1 replay record missing: {source}")
    payload = json.loads(source.read_text())
    if (
        payload.get("task_id") != task_id
        or payload.get("family_id") != family_id
        or payload.get("task_role") != "canonical"
    ):
        raise RuntimeError(f"source T1 replay identity mismatch: {source}")
    # Preserve exactly the outcome fields and compacted trajectories consumed
    # downstream, but do not attribute the source run's paths, token usage, or
    # cost to the new method.
    outcome = dict(payload.get("outcome") or {})
    outcome.update({
        "trial_dir": None,
        "trajectory_path": None,
        "n_input_tokens": 0,
        "n_output_tokens": 0,
        "n_cache_tokens": 0,
        "cost_usd": 0.0,
        "cost_source": "",
    })
    for key in ("trajectory_compact", "trajectory_compact_rough"):
        compact = dict(payload.get(key) or {})
        compact["raw_path"] = None
        payload[key] = compact
    payload.update({
        "library_hash_pre": None,
        "library_hash_post": None,
        "outcome": outcome,
        "replay_mode": "shared_t1_anchor",
    })
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _family_anchor(
    source_run: Path, family_id: str,
) -> tuple[dict, dict[str, bytes], bytes]:
    env_id = family_id.split("-", 1)[0]
    repo = source_run / "library" / env_id
    task_id = f"{family_id}-T1"
    commits = [
        line.strip()
        for line in _git(
            repo, "log", "--format=%H", "--all", "--fixed-strings",
            f"--grep=(task={task_id})",
        ).splitlines()
        if line.strip()
    ]
    if len(commits) != 1:
        raise RuntimeError(
            f"expected exactly one primary T1 commit for {task_id}, got {commits}"
        )
    commit = commits[0]
    manifest = yaml.safe_load(_git(repo, "show", f"{commit}:manifest.yaml")) or {}
    skills = manifest.get("skills") or {}
    matches = [
        (skill_id, entry)
        for skill_id, entry in skills.items()
        if entry.get("family_id") == family_id
        and entry.get("created_at_task") == task_id
        and int(entry.get("current_version", 0)) == 1
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one version-1 T1 skill for {family_id}, got {len(matches)}"
        )
    skill_id, entry = matches[0]
    slug = skill_id.split(".", 1)[1]
    prefix = f"active/{slug}/"
    tree_paths = [
        line.strip() for line in _git(
            repo, "ls-tree", "-r", "--name-only", commit, "--", prefix,
        ).splitlines() if line.strip()
    ]
    if not tree_paths:
        raise RuntimeError(f"no T1 files found for {family_id} at {commit}")
    files: dict[str, bytes] = {}
    file_hashes: dict[str, str] = {}
    for git_path in tree_paths:
        rel = git_path.removeprefix("active/")
        data = subprocess.run(
            ["git", "show", f"{commit}:{git_path}"],
            cwd=repo, check=True, capture_output=True,
        ).stdout
        data.decode("utf-8")
        files[rel] = data
        file_hashes[rel] = _sha256(data)
    if f"{slug}/SKILL.md" not in files:
        raise RuntimeError(f"T1 anchor {family_id} lacks {slug}/SKILL.md")
    state_data = _shared_replay_state(source_run, family_id)
    state_hash = _sha256(state_data)
    canonical = json.dumps(
        {
            "family_id": family_id,
            "task_id": task_id,
            "skill_id": skill_id,
            "files": file_hashes,
            "t1_state_sha256": state_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    versions = entry.get("versions") or []
    record = {
        "task_id": task_id,
        "skill_id": skill_id,
        "source_commit": commit,
        "source_patch_id": (versions[0].get("patch_id", "") if versions else ""),
        "files": file_hashes,
        "t1_state_file": "t1_state.json",
        "t1_state_sha256": state_hash,
        "anchor_sha256": _sha256(canonical),
    }
    return record, files, state_data


def export(source_run: Path, output_dir: Path, anchor_set_id: str) -> Path:
    source_run = source_run.resolve()
    if not (source_run / "reports" / "full_report.json").is_file():
        raise RuntimeError(f"source run is not complete: {source_run}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    family_records: dict[str, dict] = {}
    for env_index in range(1, 7):
        for skill_index in range(1, 6):
            family_id = f"E{env_index}-LS{skill_index}"
            record, files, state_data = _family_anchor(source_run, family_id)
            for rel, data in files.items():
                target = output_dir / "families" / family_id / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            state_target = output_dir / "families" / family_id / "t1_state.json"
            state_target.write_bytes(state_data)
            family_records[family_id] = record
    benchmark_hash_path = source_run / "benchmark_hash.txt"
    payload = {
        "schema_version": 2,
        "anchor_set_id": anchor_set_id,
        "source_run_id": source_run.name,
        "source_benchmark_hash": (
            benchmark_hash_path.read_text().strip()
            if benchmark_hash_path.is_file() else ""
        ),
        "family_count": len(family_records),
        "families": family_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-set-id", required=True)
    args = parser.parse_args()
    manifest = export(args.source_run, args.output_dir, args.anchor_set_id)
    print(f"Exported shared T1 anchors: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
