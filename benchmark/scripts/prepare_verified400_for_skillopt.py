"""Create a normalized, reproducible SkillOpt data root from Verified 400."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from audit_verified400 import resolve_task_workbooks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("benchmark/data/extracted/spreadsheetbench_verified_400"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/data/prepared/spreadsheetbench_verified_400"),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    tasks = json.loads((source / "dataset.json").read_text(encoding="utf-8"))

    if output.exists():
        shutil.rmtree(output)
    (output / "spreadsheet").mkdir(parents=True)
    shutil.copy2(source / "dataset.json", output / "dataset.json")

    normalized = []
    for task in tasks:
        task_id = str(task["id"])
        source_dir = source / task["spreadsheet_path"]
        output_dir = output / task["spreadsheet_path"]
        output_dir.mkdir(parents=True)

        init_path, golden_path, naming_scheme = resolve_task_workbooks(source_dir, task_id)
        if init_path is None or golden_path is None:
            raise FileNotFoundError(f"Cannot resolve workbook pair for task {task_id}")

        shutil.copy2(source_dir / "prompt.txt", output_dir / "prompt.txt")
        output_init = output_dir / f"1_{task_id}_init.xlsx"
        output_golden = output_dir / f"1_{task_id}_golden.xlsx"
        shutil.copy2(init_path, output_init)
        shutil.copy2(golden_path, output_golden)
        normalized.append(
            {
                "id": task_id,
                "source_naming_scheme": naming_scheme,
                "source_init": init_path.name,
                "source_golden": golden_path.name,
                "prepared_init": output_init.name,
                "prepared_golden": output_golden.name,
            }
        )

    manifest = {
        "source": str(source),
        "output": str(output),
        "task_count": len(tasks),
        "normalization_only": True,
        "original_files_modified": False,
        "tasks": normalized,
    }
    (output / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
