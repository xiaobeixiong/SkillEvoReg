"""Expand SkillOpt's official SpreadsheetBench split to 100/100/200.

The official 80 train and 40 validation tasks are preserved.  Exactly 20 and
60 tasks, respectively, are moved from the official test pool using only
static task semantics and workbook structure.  No model outputs, trajectories,
scores, or evolved skills are read.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from build_verified400_compositional_split import (  # noqa: E402
    MANUAL_OPERATION_ATOM_OVERRIDES,
    balance_score,
    normalize_task_metadata,
    operation_atoms,
    output_mode,
    proportion_gap,
    standardize,
    structural_raw,
    summarize,
    token_jaccard,
    vector_gap,
)


DEFAULT_OFFICIAL = REPO_ROOT / (
    "benchmark/data/splits/spreadsheetbench_verified_400/skillopt_official_v1"
)
DEFAULT_FEATURES = REPO_ROOT / (
    "benchmark/data/audit/spreadsheetbench_verified_400/task_features.jsonl"
)
DEFAULT_DATASET = REPO_ROOT / (
    "benchmark/data/extracted/spreadsheetbench_verified_400/dataset.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "benchmark/data/splits/spreadsheetbench_verified_400/"
    "skillopt_official_expanded_100_100_200_v1"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def id_hash(items: list[dict]) -> str:
    payload = "\n".join(str(item["id"]) for item in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def annotate(
    feature_rows: list[dict], source_items: dict[str, dict]
) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    for original in feature_rows:
        item = dict(original)
        item_id = str(item["id"])
        item["id"] = item_id
        item["operation_atoms"] = MANUAL_OPERATION_ATOM_OVERRIDES.get(
            item_id, operation_atoms(item["instruction"])
        )
        item["output_mode"] = output_mode(item)
        item["source_item"] = normalize_task_metadata(source_items[item_id])
        rows.append(item)
    standardize(rows)
    return rows, {item["id"]: item for item in rows}


def choose_expansion(
    *,
    fixed_train: list[dict],
    fixed_val: list[dict],
    official_test: list[dict],
    all_items: list[dict],
    seed: int,
    trials: int,
) -> tuple[list[dict], list[dict], list[dict], float]:
    rng = random.Random(seed)
    best = None
    for _ in range(trials):
        moved = rng.sample(official_test, 80)
        rng.shuffle(moved)
        train = fixed_train + moved[:20]
        val = fixed_val + moved[20:]
        moved_ids = {item["id"] for item in moved}
        test = [item for item in official_test if item["id"] not in moved_ids]
        score = (
            1.25 * balance_score(train, all_items)
            + 1.25 * balance_score(val, all_items)
            + 0.75 * balance_score(test, all_items)
            + 0.50 * balance_score(train, val)
        )
        tie = tuple(sorted(item["id"] for item in moved))
        candidate = (score, tie, train, val, test)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2], best[3], best[4], best[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--trials", type=int, default=20000)
    args = parser.parse_args()

    official_source = {
        split: read_json(args.official_root / split / "items.json")
        for split in ("train", "val", "test")
    }
    source_items = {
        str(item["id"]): item for item in read_json(args.dataset)
    }
    feature_rows = read_jsonl(args.features)
    if len(source_items) != 400 or len(feature_rows) != 400:
        raise ValueError("expected exactly 400 source and audited feature rows")
    all_items, by_id = annotate(feature_rows, source_items)

    official = {
        split: [by_id[str(item["id"])] for item in rows]
        for split, rows in official_source.items()
    }
    train, val, test, objective = choose_expansion(
        fixed_train=official["train"],
        fixed_val=official["val"],
        official_test=official["test"],
        all_items=all_items,
        seed=args.seed,
        trials=args.trials,
    )
    splits = {"train": train, "val": val, "test": test}

    official_train_ids = {item["id"] for item in official["train"]}
    official_val_ids = {item["id"] for item in official["val"]}
    official_test_ids = {item["id"] for item in official["test"]}
    train_extra = [item for item in train if item["id"] in official_test_ids]
    val_extra = [item for item in val if item["id"] in official_test_ids]

    all_ids = [item["id"] for rows in splits.values() for item in rows]
    cross_near = []
    split_by_id = {
        item["id"]: split for split, rows in splits.items() for item in rows
    }
    for index, left in enumerate(all_items):
        for right in all_items[index + 1 :]:
            similarity = token_jaccard(left["instruction"], right["instruction"])
            if similarity >= 0.75 and split_by_id[left["id"]] != split_by_id[right["id"]]:
                cross_near.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "jaccard": round(similarity, 4),
                    }
                )

    checks = {
        "exact_counts_100_100_200": {
            split: len(rows) for split, rows in splits.items()
        } == {"train": 100, "val": 100, "test": 200},
        "all_400_unique": len(all_ids) == 400 and len(set(all_ids)) == 400,
        "official_train_80_preserved": official_train_ids <= {
            item["id"] for item in train
        },
        "official_val_40_preserved": official_val_ids <= {
            item["id"] for item in val
        },
        "train_adds_exactly_20_official_test": len(train_extra) == 20,
        "val_adds_exactly_60_official_test": len(val_extra) == 60,
        "remaining_test_is_official_test_only": {
            item["id"] for item in test
        } <= official_test_ids,
        "no_cross_split_instruction_near_duplicates_at_0_75": not cross_near,
        "train_val_instruction_type_gap_le_0_10": proportion_gap(
            train, val, lambda item: item["instruction_type"]
        ) <= 0.10,
        "train_val_output_mode_gap_le_0_10": proportion_gap(
            train, val, lambda item: item["output_mode"]
        ) <= 0.10,
        "train_val_max_structural_mean_gap_le_0_35": vector_gap(train, val) <= 0.35,
        "train_test_max_structural_mean_gap_le_0_35": vector_gap(train, test) <= 0.35,
        "val_test_max_structural_mean_gap_le_0_35": vector_gap(val, test) <= 0.35,
    }

    output = args.output_dir.resolve()
    for split, rows in splits.items():
        write_json(output / split / "items.json", [item["source_item"] for item in rows])

    manifest = {
        "version": "skillopt_official_expanded_100_100_200_v1",
        "status": "accepted_by_static_checks" if all(checks.values()) else "rejected",
        "seed": args.seed,
        "optimization_trials": args.trials,
        "objective": round(objective, 8),
        "source_split": str(args.official_root.resolve()),
        "source_dataset": str(args.dataset.resolve()),
        "counts": {split: len(rows) for split, rows in splits.items()},
        "provenance": {
            "official_train_preserved": 80,
            "official_val_preserved": 40,
            "official_test_to_train": 20,
            "official_test_to_val": 60,
            "official_test_remaining": 200,
            "train_added_ids": [item["id"] for item in train_extra],
            "val_added_ids": [item["id"] for item in val_extra],
        },
        "selection_policy": {
            "model_outcomes_used": False,
            "historical_curves_used": False,
            "evolved_skills_used": False,
            "features": [
                "instruction_type",
                "operation_atoms",
                "output_mode",
                "workbook_and_answer_size",
                "formula_count",
                "sheet_count",
                "instruction_length",
                "merged_cells",
                "drawings",
            ],
            "near_duplicate_threshold": 0.75,
        },
        "checks": checks,
        "check_details": {
            "cross_split_near_duplicates": cross_near,
            "max_structural_mean_gaps": {
                "train_val": round(vector_gap(train, val), 6),
                "train_test": round(vector_gap(train, test), 6),
                "val_test": round(vector_gap(val, test), 6),
            },
            "instruction_type_gaps": {
                "train_val": round(
                    proportion_gap(train, val, lambda item: item["instruction_type"]), 6
                ),
                "train_test": round(
                    proportion_gap(train, test, lambda item: item["instruction_type"]), 6
                ),
                "val_test": round(
                    proportion_gap(val, test, lambda item: item["instruction_type"]), 6
                ),
            },
        },
        "summaries": {split: summarize(rows) for split, rows in splits.items()},
        "id_sha256": {split: id_hash(rows) for split, rows in splits.items()},
        "task_annotations": {
            item["id"]: {
                "split": split_by_id[item["id"]],
                "official_split": next(
                    split
                    for split, rows in official.items()
                    if item["id"] in {row["id"] for row in rows}
                ),
                "instruction_type": item["instruction_type"],
                "operation_atoms": list(item["operation_atoms"]),
                "output_mode": item["output_mode"],
                "structural_raw": [round(value, 6) for value in structural_raw(item)],
            }
            for item in sorted(all_items, key=lambda row: row["id"])
        },
        "limitations": [
            "This is a benchmark-specific expanded split, not SkillOpt's official split.",
            "Static operation labels are conservative regex annotations.",
            "The split reduces sampling variance but cannot remove stochastic model variance.",
        ],
    }
    write_json(output / "split_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
        "checks": checks,
        "output": str(output),
    }, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
