"""Materialize LiveMathBench 100/80/217 ID and 211 temporal-OOD splits.

The deterministic 50/40/87 core split is preserved.  March and April 2026
are independently ranked by SHA-256 with seed 101; each month contributes 25
examples to train, 20 to validation, and all remaining examples to IID test.
May and June 2026 are isolated in a separate test-only split and are never
read into the train/validation/IID materialization path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLOPT_ROOT = REPO_ROOT / "benchmark" / "third_party" / "SkillOpt"
if str(SKILLOPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILLOPT_ROOT))

from skillopt.envs.livemathematicianbench.dataloader import load_items  # noqa: E402


DEFAULT_CORE = SKILLOPT_ROOT / "data" / "livemathematicianbench_split_50_40_87"
DEFAULT_ID_EXTENSION_RAW = (
    REPO_ROOT / "benchmark" / "data" / "raw" / "livemathbench_id_extension"
)
DEFAULT_OOD_RAW = (
    REPO_ROOT / "benchmark" / "data" / "raw" / "livemathbench_temporal_ood"
)
DEFAULT_ID_OUTPUT = SKILLOPT_ROOT / "data" / "livemathematicianbench_split_100_80_217"
DEFAULT_OOD_OUTPUT = (
    SKILLOPT_ROOT / "data" / "livemathematicianbench_temporal_ood_202605_202606"
)
CORE_COUNTS = {"train": 50, "val": 40, "test": 87}
ID_COUNTS = {"train": 100, "val": 80, "test": 217}
MONTH_COUNTS = {"202603": 112, "202604": 108, "202605": 95, "202606": 116}
SEED = 101
SPLITS = ("train", "val", "test")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_item(item: dict) -> None:
    item_id = str(item.get("id") or "")
    if not item_id or not str(item.get("question") or "").strip():
        raise ValueError(f"Invalid item identity/question: {item_id!r}")
    choices = item.get("choices") or []
    labels = {str(choice.get("label") or "").strip() for choice in choices}
    correct = str((item.get("correct_choice") or {}).get("label") or "").strip()
    if not choices or not correct or correct not in labels:
        raise ValueError(f"Invalid choices/correct label for {item_id}")


def _portable_item(item: dict) -> dict:
    result = dict(item)
    source = Path(str(result["source_path"])).resolve()
    result["source_path"] = str(source.relative_to(REPO_ROOT))
    _validate_item(result)
    return result


def _load_exact_months(raw_root: Path, expected_months: set[str]) -> dict[str, list[dict]]:
    files = sorted(raw_root.glob("*/qa_*_final.json"))
    months = {path.parent.name for path in files}
    if months != expected_months or len(files) != len(expected_months):
        raise FileNotFoundError(
            f"Expected exactly months {sorted(expected_months)} under {raw_root}, got {sorted(months)}"
        )
    grouped = {month: [] for month in expected_months}
    for item in load_items(str(raw_root)):
        month = str(item.get("month") or "")
        if month not in grouped:
            raise ValueError(f"Unexpected month {month!r} in {raw_root}")
        grouped[month].append(_portable_item(item))
    for month, expected in ((m, MONTH_COUNTS[m]) for m in expected_months):
        ids = [str(item["id"]) for item in grouped[month]]
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"{month}: expected {expected} unique items, got {len(ids)}")
    return grouped


def _stable_rank(items: list[dict], month: str) -> list[dict]:
    return sorted(
        items,
        key=lambda item: hashlib.sha256(
            f"livemathbench:{SEED}:{month}:{item['id']}".encode("utf-8")
        ).hexdigest(),
    )


def _write_items(output_root: Path, split: str, items: list[dict]) -> None:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "items.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_partition(split_items: dict[str, list[dict]], expected: dict[str, int]) -> None:
    ids: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = split_items[split]
        row_ids = [str(item["id"]) for item in rows]
        if len(rows) != expected[split] or len(set(row_ids)) != len(row_ids):
            raise ValueError(f"{split}: expected {expected[split]} unique rows, got {len(rows)}")
        ids[split] = set(row_ids)
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = ids[left] & ids[right]
            if overlap:
                raise ValueError(f"Leakage between {left}/{right}: {sorted(overlap)[:5]}")


def materialize(
    core_root: Path,
    id_extension_raw: Path,
    ood_raw: Path,
    id_output: Path,
    ood_output: Path,
) -> dict:
    core = {split: _read_json(core_root / split / "items.json") for split in SPLITS}
    _validate_partition(core, CORE_COUNTS)

    extension = _load_exact_months(id_extension_raw, {"202603", "202604"})
    additions = {split: [] for split in SPLITS}
    per_month_allocation = {}
    for month in ("202603", "202604"):
        ranked = _stable_rank(extension[month], month)
        month_split = {
            "train": ranked[:25],
            "val": ranked[25:45],
            "test": ranked[45:],
        }
        per_month_allocation[month] = {
            split: len(rows) for split, rows in month_split.items()
        }
        for split in SPLITS:
            additions[split].extend(month_split[split])

    id_split = {split: core[split] + additions[split] for split in SPLITS}
    _validate_partition(id_split, ID_COUNTS)

    ood_months = _load_exact_months(ood_raw, {"202605", "202606"})
    ood_items = ood_months["202605"] + ood_months["202606"]
    ood_split = {"train": [], "val": [], "test": ood_items}
    _validate_partition(ood_split, {"train": 0, "val": 0, "test": 211})

    id_ids = {str(item["id"]) for rows in id_split.values() for item in rows}
    ood_ids = {str(item["id"]) for item in ood_items}
    if id_ids & ood_ids:
        raise ValueError("Temporal OOD overlaps the train/validation/IID pool")

    for split in SPLITS:
        _write_items(id_output, split, id_split[split])
        _write_items(ood_output, split, ood_split[split])

    latest_revision = "6f53c5ff7227633ea954b2847cd590314d582047"
    id_manifest = {
        "benchmark": "LiveMathematicianBench",
        "profile": "extended_100_80_217",
        "counts": ID_COUNTS,
        "core_split": str(core_root.relative_to(REPO_ROOT)),
        "core_counts": CORE_COUNTS,
        "core_source_revision": "b72450f6ce96c26158d64d945a5d31ef7727be41",
        "extension_source_revision": latest_revision,
        "extension_months": ["202603", "202604"],
        "extension_allocation": per_month_allocation,
        "ranking": "SHA-256 ascending over livemathbench:<seed>:<month>:<id>",
        "split_seed": SEED,
        "temporal_ood_months": ["202605", "202606"],
        "temporal_ood_count": 211,
        "temporal_ood_used_for_selection": False,
        "raw_files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(id_extension_raw.glob("*/qa_*_final.json"))
        ],
    }
    ood_manifest = {
        "benchmark": "LiveMathematicianBench",
        "profile": "temporal_ood_202605_202606",
        "counts": {"train": 0, "val": 0, "test": 211},
        "source_revision": latest_revision,
        "months": {"202605": 95, "202606": 116},
        "policy": "final-skill evaluation only; never use for training, validation, selection, or tuning",
        "raw_files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(ood_raw.glob("*/qa_*_final.json"))
        ],
    }
    (id_output / "split_manifest.json").write_text(
        json.dumps(id_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ood_output / "split_manifest.json").write_text(
        json.dumps(ood_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"id": id_manifest, "temporal_ood": ood_manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--id-extension-raw", type=Path, default=DEFAULT_ID_EXTENSION_RAW)
    parser.add_argument("--ood-raw", type=Path, default=DEFAULT_OOD_RAW)
    parser.add_argument("--id-output", type=Path, default=DEFAULT_ID_OUTPUT)
    parser.add_argument("--ood-output", type=Path, default=DEFAULT_OOD_OUTPUT)
    args = parser.parse_args()
    result = materialize(
        args.core_root.resolve(),
        args.id_extension_raw.resolve(),
        args.ood_raw.resolve(),
        args.id_output.resolve(),
        args.ood_output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
