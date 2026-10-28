"""Audit SpreadsheetBench Verified and extract split-relevant features.

The script is read-only with respect to the benchmark workbooks. It parses the
Open XML package directly so formulas and structure are counted without asking
Excel to recalculate or resave the files.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"x": MAIN_NS, "r": DOC_REL_NS}

CELL_REF_RE = re.compile(r"^\$?([A-Z]{1,3})\$?(\d+)$")
RANGE_RE = re.compile(
    r"^\$?([A-Z]{1,3})\$?(\d+)(?::\$?([A-Z]{1,3})\$?(\d+))?$"
)

PROCEDURE_PATTERNS = {
    "lookup_match": r"\b(lookup|xlookup|vlookup|index.?match|match|join|merge|map|correspond)\b",
    "aggregation": r"\b(sum|total|average|mean|count|aggregate|subtotal|maximum|minimum|max|min)\b",
    "sort_filter": r"\b(sort|filter|rank|ascending|descending|highest|lowest)\b",
    "clean_normalize": r"\b(clean|normalize|duplicate|deduplicate|trim|replace|remove blank|missing)\b",
    "extract_parse": r"\b(extract|split|parse|substring|characters?|text to columns|separate)\b",
    "formula_generation": r"\b(formula|function|calculate|calculation|compute)\b",
    "formatting": r"\b(format|highlight|color|font|border|conditional formatting|style)\b",
    "row_column_edit": r"\b(insert|delete|remove|add).{0,20}\b(row|rows|column|columns|cells?)\b",
    "sheet_edit": r"\b(sheet|worksheet|tab)\b",
    "date_time": r"\b(date|time|month|year|day|week|quarter)\b",
}


def col_to_int(col: str) -> int:
    value = 0
    for char in col:
        value = value * 26 + ord(char) - ord("A") + 1
    return value


def parse_range_size(text: str | None) -> tuple[int, int, int]:
    if not text:
        return 0, 0, 0
    match = RANGE_RE.match(text.strip().upper())
    if not match:
        return 0, 0, 0
    c1, r1, c2, r2 = match.groups()
    c2 = c2 or c1
    r2 = r2 or r1
    rows = abs(int(r2) - int(r1)) + 1
    cols = abs(col_to_int(c2) - col_to_int(c1)) + 1
    return rows, cols, rows * cols


def resolve_target(base: str, target: str) -> str:
    base_path = Path(base)
    return (base_path.parent / target).as_posix().lstrip("/")


def parse_workbook(xlsx_path: Path) -> dict:
    with zipfile.ZipFile(xlsx_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"corrupt ZIP member: {bad_member}")

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{REL_NS}}}Relationship")
        }

        sheet_stats = []
        total_formulas = 0
        cross_sheet_formulas = 0
        volatile_formulas = 0
        total_cells = 0
        total_merges = 0
        total_drawings = 0
        max_rows = 0
        max_cols = 0
        cell_types = Counter()

        sheets = workbook.find("x:sheets", NS)
        if sheets is None:
            raise ValueError("workbook has no sheets element")

        for sheet in sheets.findall("x:sheet", NS):
            name = sheet.attrib.get("name", "")
            rel_id = sheet.attrib.get(f"{{{DOC_REL_NS}}}id")
            if not rel_id or rel_id not in rel_targets:
                raise ValueError(f"cannot resolve worksheet relation for {name!r}")
            sheet_xml_path = resolve_target("xl/workbook.xml", rel_targets[rel_id])
            root = ET.fromstring(archive.read(sheet_xml_path))

            dimension = root.find("x:dimension", NS)
            dimension_ref = dimension.attrib.get("ref", "") if dimension is not None else ""
            if ":" in dimension_ref:
                end_ref = dimension_ref.rsplit(":", 1)[1].replace("$", "")
            else:
                end_ref = dimension_ref.replace("$", "")
            end_match = CELL_REF_RE.match(end_ref)
            rows = int(end_match.group(2)) if end_match else 0
            cols = col_to_int(end_match.group(1)) if end_match else 0

            formulas = []
            local_types = Counter()
            for cell in root.findall(".//x:c", NS):
                total_cells += 1
                cell_type = cell.attrib.get("t", "n")
                cell_types[cell_type] += 1
                local_types[cell_type] += 1
                formula = cell.find("x:f", NS)
                if formula is not None:
                    text = formula.text or ""
                    formulas.append(text)
                    total_formulas += 1
                    if "!" in text:
                        cross_sheet_formulas += 1
                    if re.search(r"\b(OFFSET|INDIRECT|TODAY|NOW|RAND|RANDBETWEEN)\s*\(", text, re.I):
                        volatile_formulas += 1

            merge_cells = root.find("x:mergeCells", NS)
            merge_count = (
                int(merge_cells.attrib.get("count", "0"))
                if merge_cells is not None
                else 0
            )
            drawing_count = len(root.findall("x:drawing", NS))
            total_merges += merge_count
            total_drawings += drawing_count
            max_rows = max(max_rows, rows)
            max_cols = max(max_cols, cols)

            sheet_stats.append(
                {
                    "name": name,
                    "dimension": dimension_ref,
                    "rows": rows,
                    "cols": cols,
                    "formula_count": len(formulas),
                    "merge_count": merge_count,
                    "drawing_count": drawing_count,
                    "cell_type_counts": dict(sorted(local_types.items())),
                }
            )

    return {
        "sheet_count": len(sheet_stats),
        "sheets": sheet_stats,
        "max_rows": max_rows,
        "max_cols": max_cols,
        "nonempty_cell_count": total_cells,
        "formula_count": total_formulas,
        "cross_sheet_formula_count": cross_sheet_formulas,
        "volatile_formula_count": volatile_formulas,
        "merge_count": total_merges,
        "drawing_count": total_drawings,
        "cell_type_counts": dict(sorted(cell_types.items())),
    }


def procedure_tags(instruction: str) -> list[str]:
    lowered = instruction.lower()
    return [
        tag
        for tag, pattern in PROCEDURE_PATTERNS.items()
        if re.search(pattern, lowered, flags=re.I | re.S)
    ] or ["other"]


def resolve_task_workbooks(task_dir: Path, task_id: str) -> tuple[Path | None, Path | None, str]:
    exact_init = task_dir / f"1_{task_id}_init.xlsx"
    exact_golden = task_dir / f"1_{task_id}_golden.xlsx"
    if exact_init.is_file() and exact_golden.is_file():
        return exact_init, exact_golden, "numbered_exact"

    bare_init = task_dir / "initial.xlsx"
    bare_golden = task_dir / "golden.xlsx"
    if bare_init.is_file() and bare_golden.is_file():
        return bare_init, bare_golden, "bare"

    init_candidates = sorted(task_dir.glob("*_init.xlsx"))
    golden_candidates = sorted(task_dir.glob("*_golden.xlsx"))
    if len(init_candidates) == 1 and len(golden_candidates) == 1:
        return init_candidates[0], golden_candidates[0], "unique_fallback"

    return None, None, "unresolved"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("benchmark/data/extracted/spreadsheetbench_verified_400"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/data/audit/spreadsheetbench_verified_400"),
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_root / "dataset.json"
    tasks = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("dataset.json must contain a JSON array")

    seen_ids = set()
    features = []
    errors = []
    warnings = []

    for task in tasks:
        task_id = str(task.get("id", ""))
        if not task_id:
            errors.append({"id": None, "error": "missing task id"})
            continue
        if task_id in seen_ids:
            errors.append({"id": task_id, "error": "duplicate task id"})
            continue
        seen_ids.add(task_id)

        task_dir = data_root / str(task.get("spreadsheet_path", ""))
        prompt_path = task_dir / "prompt.txt"
        init_path, golden_path, naming_scheme = resolve_task_workbooks(task_dir, task_id)
        missing = []
        for candidate, label in (
            (prompt_path, "prompt"),
            (init_path, "init workbook"),
            (golden_path, "golden workbook"),
        ):
            if candidate is None:
                missing.append(label)
            elif not candidate.is_file():
                missing.append(str(candidate.relative_to(data_root)))
        if missing:
            errors.append({"id": task_id, "error": "missing files", "files": missing})
            continue

        if naming_scheme != "numbered_exact":
            warnings.append(
                {
                    "id": task_id,
                    "warning": "nonstandard workbook naming",
                    "naming_scheme": naming_scheme,
                    "init": init_path.name,
                    "golden": golden_path.name,
                }
            )

        prompt = prompt_path.read_text(encoding="utf-8").strip()
        instruction = str(task.get("instruction", "")).strip()
        prompt_matches = " ".join(prompt.split()) == " ".join(instruction.split())
        if not prompt_matches:
            warnings.append({"id": task_id, "warning": "prompt differs from dataset instruction"})

        try:
            init_stats = parse_workbook(init_path)
            golden_stats = parse_workbook(golden_path)
        except Exception as exc:
            errors.append({"id": task_id, "error": f"workbook parse failed: {exc}"})
            continue

        answer_rows, answer_cols, answer_cells = parse_range_size(task.get("answer_position"))
        data_rows, data_cols, data_cells = parse_range_size(task.get("data_position"))
        tags = procedure_tags(instruction)

        features.append(
            {
                "id": task_id,
                "instruction": instruction,
                "instruction_type": task.get("instruction_type"),
                "spreadsheet_path": task.get("spreadsheet_path"),
                "answer_sheet": task.get("answer_sheet"),
                "answer_position": task.get("answer_position"),
                "data_position": task.get("data_position"),
                "prompt_matches_instruction": prompt_matches,
                "workbook_naming_scheme": naming_scheme,
                "procedure_tags_heuristic": tags,
                "answer_rows": answer_rows,
                "answer_cols": answer_cols,
                "answer_cells": answer_cells,
                "data_rows": data_rows,
                "data_cols": data_cols,
                "data_cells": data_cells,
                "init": init_stats,
                "golden": golden_stats,
                "derived": {
                    "is_multi_sheet": init_stats["sheet_count"] > 1,
                    "has_formulas": init_stats["formula_count"] > 0,
                    "has_cross_sheet_formulas": init_stats["cross_sheet_formula_count"] > 0,
                    "has_merged_cells": init_stats["merge_count"] > 0,
                    "has_drawings": init_stats["drawing_count"] > 0,
                    "large_target": answer_cells >= 100,
                    "large_workbook": (
                        init_stats["sheet_count"] >= 4
                        or init_stats["nonempty_cell_count"] >= 5000
                    ),
                },
            }
        )

    feature_path = output_dir / "task_features.jsonl"
    with feature_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in features:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    type_counts = Counter(item["instruction_type"] for item in features)
    tag_counts = Counter(
        tag for item in features for tag in item["procedure_tags_heuristic"]
    )
    sheet_counts = [item["init"]["sheet_count"] for item in features]
    formula_counts = [item["init"]["formula_count"] for item in features]
    cell_counts = [item["init"]["nonempty_cell_count"] for item in features]

    report = {
        "source": str(data_root),
        "task_count_declared": len(tasks),
        "task_count_valid": len(features),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "instruction_type_counts": dict(sorted(type_counts.items())),
        "procedure_tag_counts_heuristic": dict(sorted(tag_counts.items())),
        "structure": {
            "multi_sheet_tasks": sum(item["derived"]["is_multi_sheet"] for item in features),
            "formula_tasks": sum(item["derived"]["has_formulas"] for item in features),
            "cross_sheet_formula_tasks": sum(
                item["derived"]["has_cross_sheet_formulas"] for item in features
            ),
            "merged_cell_tasks": sum(
                item["derived"]["has_merged_cells"] for item in features
            ),
            "drawing_tasks": sum(item["derived"]["has_drawings"] for item in features),
            "large_target_tasks": sum(item["derived"]["large_target"] for item in features),
            "large_workbook_tasks": sum(item["derived"]["large_workbook"] for item in features),
            "sheet_count_median": statistics.median(sheet_counts) if sheet_counts else 0,
            "sheet_count_max": max(sheet_counts, default=0),
            "formula_count_median": statistics.median(formula_counts) if formula_counts else 0,
            "formula_count_max": max(formula_counts, default=0),
            "nonempty_cells_median": statistics.median(cell_counts) if cell_counts else 0,
            "nonempty_cells_max": max(cell_counts, default=0),
        },
        "errors": errors,
        "warnings": warnings,
        "outputs": {"features": str(feature_path)},
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
