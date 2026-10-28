#!/usr/bin/env python3
"""Export exact Healthcare step-50 checkpoints for frozen held-out evaluation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/scripts"))
from build_continual_math_frozen_libraries import from_final_json, from_state_tar, write_library


def trial(job: str) -> Path:
    root = ROOT / "benchmark/outputs/continualskillbench" / job
    found = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
    assert len(found) == 1, (job, found)
    result = json.loads((found[0] / "result.json").read_text())
    assert result.get("exception_info") is None, result.get("exception_info")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-job', required=True)
    parser.add_argument('--regularization-job', required=True)
    args = parser.parse_args()
    baseline_trial = trial(args.baseline_job)
    regularization_trial = trial(args.regularization_job)
    sources = {
        "baseline": baseline_trial / "agent/checkpoints/step_050/state.tar.gz",
        "regularization": regularization_trial / "agent/regularization/step_050/final.json",
    }
    assert all(path.is_file() for path in sources.values()), sources
    assert (regularization_trial / "agent/regularization/step_050/committed.json").is_file()
    target = ROOT / "benchmark/runtime/continualskillbench/frozen_healthcare_libraries"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing frozen library: {target}")
    target.mkdir()
    manifest = {"split": 50, "note": "Healthcare libraries learned only from tasks 1-50"}
    for condition, reader in (("baseline", from_state_tar), ("regularization", from_final_json)):
        source = sources[condition]
        entry = write_library(reader(source), target / f"{condition}_step050")
        entry["source"] = str(source.relative_to(ROOT))
        manifest[f"{condition}_step050"] = entry
        print(condition, entry["n_skills"], "skills", entry["n_files"], "files")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
