"""Materialize SkillOpt's official SpreadsheetBench 80/40/280 split.

The released SkillOpt split contains stable IDs only. This script joins those
IDs to the locally prepared Verified-400 task rows without changing membership
or order.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_MANIFEST = (
    ROOT
    / "benchmark"
    / "third_party"
    / "SkillOpt"
    / "data"
    / "spreadsheetbench_id_split"
)
PREPARED_ROOT = (
    ROOT / "benchmark" / "data" / "prepared" / "spreadsheetbench_verified_400"
)
PREPARED_DATASET = PREPARED_ROOT / "dataset.json"
OUTPUT_ROOT = (
    ROOT
    / "benchmark"
    / "data"
    / "splits"
    / "spreadsheetbench_verified_400"
    / "skillopt_official_v1"
)
EXPECTED_COUNTS = {"train": 80, "val": 40, "test": 280}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    prepared_rows = read_json(PREPARED_DATASET)
    prepared_by_id = {str(row["id"]): row for row in prepared_rows}
    if len(prepared_by_id) != 400:
        raise ValueError(
            f"Expected 400 unique prepared tasks, found {len(prepared_by_id)}"
        )

    role_ids: dict[str, list[str]] = {}
    for role, expected_count in EXPECTED_COUNTS.items():
        metadata = read_json(OFFICIAL_MANIFEST / role / "items.json")
        ids = [str(row["id"]) for row in metadata]
        if len(ids) != expected_count or len(set(ids)) != expected_count:
            raise ValueError(
                f"Official {role} expected {expected_count} unique IDs, "
                f"found {len(ids)} rows / {len(set(ids))} unique"
            )
        missing = [task_id for task_id in ids if task_id not in prepared_by_id]
        if missing:
            raise ValueError(f"Official {role} IDs missing locally: {missing}")
        role_ids[role] = ids
        write_json(
            OUTPUT_ROOT / role / "items.json",
            [prepared_by_id[task_id] for task_id in ids],
        )

    all_ids = role_ids["train"] + role_ids["val"] + role_ids["test"]
    if len(set(all_ids)) != 400 or set(all_ids) != set(prepared_by_id):
        raise ValueError("Official roles are not a disjoint cover of Verified-400")

    missing_task_dirs = [
        task_id
        for task_id in all_ids
        if not (PREPARED_ROOT / prepared_by_id[task_id]["spreadsheet_path"]).is_dir()
    ]
    if missing_task_dirs:
        raise FileNotFoundError(
            f"Prepared spreadsheet directories missing: {missing_task_dirs}"
        )

    official_manifest = read_json(OFFICIAL_MANIFEST / "split_manifest.json")
    write_json(
        OUTPUT_ROOT / "split_manifest.json",
        {
            "version": "skillopt_official_v1",
            "source_manifest": str(OFFICIAL_MANIFEST.relative_to(ROOT)),
            "source_revision": official_manifest.get("source_revision"),
            "prepared_dataset": str(PREPARED_DATASET.relative_to(ROOT)),
            "counts": EXPECTED_COUNTS,
            "membership_changed": False,
            "order_changed": False,
            "checks": {
                "all_official_ids_present": True,
                "roles_disjoint": True,
                "covers_verified_400": True,
                "spreadsheet_directories_present": True,
            },
        },
    )
    print(f"Materialized official split at {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
