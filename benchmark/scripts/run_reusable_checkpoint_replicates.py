#!/usr/bin/env python3
"""Build independent checkpoint reports while reusing compatible observations.

Selection-gate evaluations are deliberately ineligible.  Only summaries below
``checkpoint_evals`` or this script's ``independent_repeats`` directory are
considered, and reuse requires an exact skill hash and evaluation protocol.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLOPT_ROOT = REPO_ROOT / "benchmark/third_party/SkillOpt"
sys.path.insert(0, str(SKILLOPT_ROOT))

from skillopt.config import flatten_config, load_config  # noqa: E402
from skillopt.utils.evaluation_identity import effective_api_route

from evaluate_skillopt_checkpoints import (  # noqa: E402
    checkpoint_skills,
    evaluation_specs,
)


EVALUATOR = (
    REPO_ROOT
    / "benchmark/scripts/evaluate_skillopt_checkpoints.py"
)
PROTOCOL_KEYS = (
    "target_model",
    "target_backend",
    "reasoning_effort",
    "env",
    "data_root",
    "split_mode",
    "split_dir",
    "mode",
    "max_turns",
    "max_completion_tokens",
    "exec_timeout",
)
ALLOWED_SOURCE_MARKERS = {"checkpoint_evals", "independent_repeats"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_path(value: object) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def protocol_payload(config: dict, split_sizes: dict[str, int], *, api_route=None) -> dict:
    payload = {key: config.get(key) for key in PROTOCOL_KEYS}
    for key in ("data_root", "split_dir"):
        if payload.get(key):
            payload[key] = normalized_path(payload[key])
    payload["split_sizes"] = split_sizes
    payload["evaluator"] = "verified400_adapter_v1"
    payload["api_route"] = effective_api_route(config) if api_route is None else api_route
    return payload


def protocol_hash(config: dict, split_sizes: dict[str, int], *, api_route=None) -> str:
    encoded = json.dumps(
        protocol_payload(config, split_sizes, api_route=api_route), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_config(summary_path: Path) -> dict | None:
    for parent in summary_path.parents:
        config_path = parent / "train" / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
    return None


def source_protocol_hash(summary_path: Path, payload: dict) -> str | None:
    provenance = summary_path.parent / "reuse_provenance.json"
    if provenance.exists():
        return json.loads(provenance.read_text(encoding="utf-8")).get(
            "protocol_hash"
        )
    config = source_config(summary_path)
    if config is None:
        return None
    split_name = summary_path.parent.name
    if split_name not in {"train", "val", "test"}:
        return None
    sizes = {split_name: int(payload.get("n_items", 0))}
    # Old checkpoint runs contain all three summaries, so recover all sizes.
    condition_root = summary_path.parent.parent
    for name in ("train", "val", "test"):
        sibling = condition_root / name / "eval_summary.json"
        if sibling.exists():
            sibling_payload = json.loads(sibling.read_text(encoding="utf-8"))
            sizes[name] = int(sibling_payload.get("n_items", 0))
    if set(sizes) != {"train", "val", "test"}:
        return None
    route = payload.get("evaluation_identity", {}).get("api_route")
    if route is None:
        return None
    return protocol_hash(config, sizes, api_route=route)


def discover_observations(search_root: Path, expected_protocol: str) -> list[dict]:
    observations = []
    for summary_path in sorted(search_root.rglob("eval_summary.json")):
        if not ALLOWED_SOURCE_MARKERS.intersection(summary_path.parts):
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not payload.get("evaluation_identity", {}).get("skill_sha256"):
                continue
            skill_path = Path(payload["skill"])
            if not skill_path.exists():
                continue
            observed_protocol = source_protocol_hash(summary_path, payload)
            if observed_protocol != expected_protocol:
                continue
            recorded_hash = payload.get("evaluation_identity", {}).get("skill_sha256")
            if recorded_hash and recorded_hash != file_hash(skill_path):
                continue
            observations.append(
                {
                    "observation_id": hashlib.sha256(
                        str(summary_path.resolve()).encode("utf-8")
                    ).hexdigest(),
                    "skill_hash": recorded_hash or file_hash(skill_path),
                    "split": summary_path.parent.name,
                    "hard": payload.get("hard"),
                    "soft": payload.get("soft"),
                    "n_items": payload.get("n_items"),
                    "summary": str(summary_path.resolve()),
                }
            )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return observations


def write_provenance(output: Path, protocol: str, skill: Path) -> None:
    payload = {
        "kind": "independent_checkpoint_evaluation",
        "used_for_selection": False,
        "protocol_hash": protocol,
        "skill_hash": file_hash(skill),
    }
    (output / "reuse_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Independent reporting evaluations per skill/split (default: 2).",
    )
    parser.add_argument("--condition")
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--search-root", type=Path)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if bool(args.condition) != bool(args.skill):
        raise ValueError("--condition and --skill must be provided together")

    train_root = args.train_root.resolve()
    run_root = train_root.parent
    search_root = (
        args.search_root.resolve() if args.search_root else run_root.parent
    )
    eval_root = (
        args.eval_root.resolve()
        if args.eval_root
        else run_root / "independent_repeats"
    )
    config = flatten_config(load_config(str(args.config.resolve())))
    specs = evaluation_specs(args.config)
    split_sizes = {spec["name"]: int(spec["size"]) for spec in specs}
    expected_protocol = protocol_hash(config, split_sizes)
    checkpoints = (
        [(args.condition, args.skill.resolve())]
        if args.condition
        else checkpoint_skills(train_root, args.config)
    )

    targets_by_hash: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for condition, skill in checkpoints:
        targets_by_hash[file_hash(skill)].append((condition, skill))

    eval_root.mkdir(parents=True, exist_ok=True)
    observations = discover_observations(search_root, expected_protocol)
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for observation in observations:
        by_key[(observation["skill_hash"], observation["split"])].append(
            observation
        )

    planned = []
    jobs: dict[tuple[str, int], dict] = {}
    for skill_hash, targets in targets_by_hash.items():
        skill = targets[0][1]
        for spec in specs:
            split = spec["name"]
            existing = by_key[(skill_hash, split)]
            for index in range(len(existing) + 1, args.repeats + 1):
                condition = f"repeat_{skill_hash[:12]}_{index:02d}"
                output = eval_root / condition / split
                planned.append(
                    {"skill_hash": skill_hash, "split": split, "output": str(output)}
                )
                job = jobs.setdefault(
                    (skill_hash, index),
                    {"condition": condition, "skill": skill, "splits": []},
                )
                job["splits"].append(split)

    if jobs and not args.dry_run and not any(os.environ.get(name) for name in
            ("OPENAI_API_KEY", "YIBU_API_KEY", "AZURE_OPENAI_API_KEY")):
        raise RuntimeError("Set OPENAI_API_KEY (or a supported provider key) to fill repeats")
    for (skill_hash, _), job in jobs.items():
        if args.dry_run:
            continue
        command = [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-u",
                    str(EVALUATOR),
                    "--train-root",
                    str(train_root),
                    "--config",
                    str(args.config.resolve()),
                    "--seed",
                    str(args.seed),
                    "--condition",
                    job["condition"],
                    "--skill",
                    str(job["skill"]),
                    "--eval-root",
                    str(eval_root),
                ]
        # The normal case needs all three splits.  Let the evaluator run them
        # concurrently (8 + 8 + 16 workers) instead of launching them serially.
        if set(job["splits"]) != {spec["name"] for spec in specs}:
            for split in job["splits"]:
                completed = subprocess.run(
                    [*command, "--only-split", split], cwd=REPO_ROOT, check=False
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Independent repeat failed: {skill_hash[:12]} {split}"
                    )
        else:
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Independent repeat failed: {skill_hash[:12]} all splits"
                )
        for split in job["splits"]:
            write_provenance(
                eval_root / job["condition"] / split,
                expected_protocol,
                job["skill"],
            )

    if not args.dry_run:
        observations = discover_observations(search_root, expected_protocol)
        by_key.clear()
        for observation in observations:
            by_key[(observation["skill_hash"], observation["split"])].append(
                observation
            )

    report = {
        "repeats_requested": args.repeats,
        "protocol_hash": expected_protocol,
        "protocol": protocol_payload(config, split_sizes),
        "selection_gate_results_reused": False,
        "dry_run": args.dry_run,
        "planned_new_observations": planned,
        "conditions": [],
    }
    for condition, skill in checkpoints:
        skill_hash = file_hash(skill)
        condition_row = {
            "condition": condition,
            "skill": str(skill),
            "skill_hash": skill_hash,
            "splits": {},
        }
        for spec in specs:
            chosen = by_key[(skill_hash, spec["name"])][: args.repeats]
            hard = [row["hard"] for row in chosen if row["hard"] is not None]
            soft = [row["soft"] for row in chosen if row["soft"] is not None]
            condition_row["splits"][spec["name"]] = {
                "observations": chosen,
                "n": len(chosen),
                "hard_mean": mean(hard) if hard else None,
                "hard_stdev": stdev(hard) if len(hard) > 1 else None,
                "soft_mean": mean(soft) if soft else None,
                "soft_stdev": stdev(soft) if len(soft) > 1 else None,
            }
        report["conditions"].append(condition_row)
    (eval_root / "reuse_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
