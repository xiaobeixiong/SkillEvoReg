#!/usr/bin/env python3
"""Export completed Finance step-50 libraries without changing training runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "benchmark/outputs/continualskillbench"
DESTINATION = ROOT / "benchmark/runtime/continualskillbench/frozen_finance_libraries"
sys.path.insert(0, str(ROOT / "benchmark/scripts"))
from build_continual_math_frozen_libraries import from_final_json, from_state_tar, write_library  # noqa: E402


def unique_trial(job: str) -> Path:
    root = OUTPUTS / job
    trials = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
    if len(trials) != 1:
        raise RuntimeError(f"expected exactly one trial under {root}, found {len(trials)}")
    return trials[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-job', required=True)
    parser.add_argument('--regularization-job', required=True)
    args = parser.parse_args()
    baseline = unique_trial(args.baseline_job)
    regularization = unique_trial(args.regularization_job)
    baseline_source = baseline / "agent/checkpoints/step_050/state.tar.gz"
    regularization_source = regularization / "agent/regularization/step_050/final.json"
    assert baseline_source.is_file(), baseline_source
    assert regularization_source.is_file(), regularization_source
    if DESTINATION.exists():
        raise RuntimeError(f"refusing to overwrite existing frozen libraries: {DESTINATION}")
    DESTINATION.mkdir(parents=True)
    manifest = {"split": 50, "note": "Finance libraries learned only from tasks 1-50"}
    for name, files, source in (
        ("baseline_step050", from_state_tar(baseline_source), baseline_source),
        ("regularization_step050", from_final_json(regularization_source), regularization_source),
    ):
        entry = write_library(files, DESTINATION / name)
        entry["source"] = str(source.relative_to(ROOT))
        manifest[name] = entry
        print(f"{name}: {entry['n_skills']} skills, {entry['chars']} chars")
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {DESTINATION / 'manifest.json'}")


if __name__ == "__main__":
    main()
