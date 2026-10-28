"""Build a leakage-resistant compositional split for Verified-400.

This builder deliberately separates two concepts:

* ``test_ood_composition`` contains untouched Verified-400 tasks whose complete
  operation combinations are absent from every in-distribution split, while
  every individual operation remains represented in training.
* ``adv_source`` contains untouched tasks reserved only as scaffolds for a
  later, frozen paired clean/adversarial generator.  It is not itself an
  adversarial test set.

No model outcomes, trajectories, or evolved skills are read by this script.
The manifest contains explicit acceptance checks and exits non-zero if any
hard check fails.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


METADATA_OVERRIDES = {
    "60-7": {
        "answer_position": "'Consolidated Tracker'!A3:E11",
        "answer_sheet": "Consolidated Tracker",
        "reason": "answer_sheet lists source sheets although only Consolidated Tracker is output",
    },
    "283-32": {
        "answer_position": "'Sheet3'!A:G,'Sheet4'!A:G",
        "answer_sheet": "'Sheet3','Sheet4'",
        "reason": "source answer_position has mismatched quotes for two output sheets",
    },
}


def normalize_task_metadata(item: dict) -> dict:
    """Apply the same two explicit metadata fixes as the evaluator adapter."""
    normalized = deepcopy(item)
    override = METADATA_OVERRIDES.get(str(item["id"]))
    if override:
        normalized.update({key: value for key, value in override.items() if key != "reason"})
    return normalized


COUNTS = {
    "train": 200,
    "val": 40,
    "test_iid": 80,
    "test_ood_composition": 40,
    "adv_source": 40,
}

# These patterns intentionally prefer precision over recall.  They describe
# requested operations, not incidental workbook nouns such as "sheet" or
# ambiguous words such as "calculate".
OPERATION_PATTERNS = {
    "lookup_join": re.compile(
        r"\b(?:vlookup|xlookup|hlookup|lookup|index\s*(?:and|&|/|-)?\s*match|"
        r"index.?match|match(?:ing)?\s+(?:records?|rows?|values?|names?|codes?|ids?|data)|"
        r"join\s+(?:tables?|data)|map\s+(?:values?|data))\b",
        re.I,
    ),
    "aggregate": re.compile(
        r"\b(?:sum|sums|summation|subtotal|subtotals|average|mean|running total|cumulative)\b|"
        r"\bcount(?:ing)?\s+(?:the\s+)?(?:number|quantity|occurrences?|rows?|cells?|values?|"
        r"items?|entries?|blanks?|individual|each|all|how many)\b|\bcounts?\s+(?:of|for|by|per)\b|"
        r"\b(?:calculate|compute|show|find|monthly|grand)\s+"
        r"(?:the\s+)?totals?\b|\btotals?\s+(?:of|for|by|per)\b",
        re.I,
    ),
    "sort": re.compile(
        r"\b(?:sort|sorted|sorting|ascending|descending|rank|ranking|alphabetical|"
        r"a\s*(?:to|-)\s*z|z\s*(?:to|-)\s*a|highest\s+to\s+lowest|"
        r"lowest\s+to\s+highest)\b",
        re.I,
    ),
    "filter_select": re.compile(
        r"\b(?:filter|filtered|filtering|criteria range|only rows|rows (?:where|whose|with)|"
        r"select rows|exclude rows|include rows)\b",
        re.I,
    ),
    "deduplicate": re.compile(
        r"\b(?:duplicate|duplicates|duplicated|deduplicate|deduplicated|de-duplicate|unique values?|"
        r"distinct values?|remove duplicates?)\b",
        re.I,
    ),
    "text_parse": re.compile(
        r"\b(?:extract|split|substring|characters?|text to columns|delimiter|delimited|parse|"
        r"separate (?:the )?(?:text|string|contents?|values?|data)|first name|last name|"
        r"middle initial)\b",
        re.I,
    ),
    "clean_replace": re.compile(
        r"\b(?:trim|normalize|clean(?:ing)?|replace|substitute|remove (?:blank|empty|spaces?|"
        r"characters?|text)|leading spaces?|trailing spaces?|missing values?)\b",
        re.I,
    ),
    "format": re.compile(
        r"\b(?:format|formatting|highlight|color|colour|font|border|shade|"
        r"conditional formatting|number format|currency format|accounting format|date format)\b",
        re.I,
    ),
    "row_col_edit": re.compile(
        r"\b(?:(?:insert|delete|remove|add|create)\s+(?:new\s+)?"
        r"(?:row|rows|column|columns|cells)|shift cells|delete these rows)\b",
        re.I,
    ),
    "date_time": re.compile(
        r"\b(?:date|dates|datetime|timestamp|month|months|year|years|week|weeks|"
        r"quarter|quarters|day|days|weekday|weekdays)\b|"
        r"\btime(?:s)?\s+(?:value|values|format|formats|data|entries|difference|duration|stamp)\b",
        re.I,
    ),
    "reshape": re.compile(
        r"\b(?:transpose|pivot|unpivot|reshape|wide format|long format|"
        r"columns? into rows?|rows? into columns?)\b",
        re.I,
    ),
    "formula_logic": re.compile(
        r"\b(?:formula|formulas|function|array formula|dynamic formula)\b",
        re.I,
    ),
}

# Instruction-level manual review overrides for cases where a keyword describes
# background data or a negated operation rather than the requested procedure.
# Keeping these corrections in code makes the judgment reproducible.
MANUAL_OPERATION_ATOM_OVERRIDES = {
    "118-50": ("sort", "text_parse"),  # explicitly says not to add formatting
    "146-49": ("clean_replace", "deduplicate", "sort"),
    "15380": ("date_time", "lookup_join", "text_parse"),
    "209-30": ("text_parse",),  # Rows.Count appears only inside supplied VBA code
    "359-21": ("filter_select", "row_col_edit"),  # Rows.Count is not aggregation
    "402-43": ("row_col_edit",),  # expand rows from an existing count; do not aggregate
    "42216": ("date_time", "format", "reshape", "row_col_edit"),  # "NA mean" is definitional
    "42930": ("sort",),  # "without Find and Replace" is not a cleaning operation
    "45063": ("date_time", "lookup_join"),  # source happens to be pre-sorted
    "488-14": ("sort",),  # "instead of sorting by color" is not formatting
    "57033": ("format", "lookup_join"),  # SUM is Boolean formula machinery, not requested aggregation
    "59511": ("other",),  # explicitly excludes XLOOKUP; task is arithmetic bucketing
    "7665": ("deduplicate", "reshape", "sort", "text_parse"),
}

ATTACK_FAMILIES = (
    "duplicate_collision",
    "identity_collision",
    "blank_zero_boundary",
    "formula_liveness",
    "source_sheet_decoy",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_instruction(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_instruction(left).split())
    right_tokens = set(normalize_instruction(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def operation_atoms(instruction: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, pattern in OPERATION_PATTERNS.items()
            if name != "formula_logic" and pattern.search(instruction)
        )
    ) or ("other",)


def output_mode(item: dict) -> str:
    answer_cells = max(int(item.get("answer_cells") or 0), 1)
    golden_formula_count = int(item["golden"]["formula_count"])
    init_formula_count = int(item["init"]["formula_count"])
    explicit_formula = bool(OPERATION_PATTERNS["formula_logic"].search(item["instruction"]))
    if explicit_formula and golden_formula_count > 0:
        return "formula"
    if golden_formula_count > init_formula_count and golden_formula_count / answer_cells >= 0.25:
        return "formula"
    if golden_formula_count == 0:
        return "static"
    return "mixed_or_existing_formula"


def structural_raw(item: dict) -> tuple[float, ...]:
    init = item["init"]
    return (
        math.log1p(init["nonempty_cell_count"]),
        math.log1p(max(item.get("answer_cells") or 0, 0)),
        math.log1p(init["formula_count"]),
        float(init["sheet_count"]),
        math.log1p(len(item["instruction"])),
        float(bool(item["derived"]["has_merged_cells"])),
        float(bool(item["derived"]["has_drawings"])),
    )


def standardize(items: list[dict]) -> None:
    columns = list(zip(*(structural_raw(item) for item in items)))
    means = [statistics.fmean(column) for column in columns]
    scales = [statistics.pstdev(column) or 1.0 for column in columns]
    for item in items:
        item["structural_vector"] = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(structural_raw(item), means, scales)
        )


def structural_distance(left: dict, right: dict) -> float:
    # Workbook and target size receive the largest weights.  Binary decorations
    # are retained but cannot dominate the match.
    weights = (1.4, 1.4, 0.8, 1.0, 0.7, 0.35, 0.35)
    squared = sum(
        weight * (a - b) ** 2
        for weight, a, b in zip(weights, left["structural_vector"], right["structural_vector"])
    )
    penalty = 0.0
    if left["instruction_type"] != right["instruction_type"]:
        penalty += 2.0
    if left["output_mode"] != right["output_mode"]:
        penalty += 1.25
    penalty += 0.8 * abs(len(left["operation_atoms"]) - len(right["operation_atoms"]))
    return math.sqrt(squared) + penalty


def nearest_controls(task: dict, pool: list[dict]) -> list[tuple[float, dict]]:
    compatible = [
        candidate
        for candidate in pool
        if candidate["instruction_type"] == task["instruction_type"]
        and candidate["output_mode"] == task["output_mode"]
        and abs(len(candidate["operation_atoms"]) - len(task["operation_atoms"])) <= 1
    ]
    if not compatible:
        compatible = pool
    return sorted((structural_distance(task, candidate), candidate) for candidate in compatible)


def mean_vector(items: list[dict]) -> tuple[float, ...]:
    return tuple(statistics.fmean(item["structural_vector"][i] for item in items) for i in range(7))


def vector_gap(left: list[dict], right: list[dict]) -> float:
    a = mean_vector(left)
    b = mean_vector(right)
    return max(abs(x - y) for x, y in zip(a, b))


def assign_controls(
    ood: list[dict], choices: dict[str, list[tuple[float, dict]]]
) -> list[tuple[dict, dict, float]] | None:
    # Hard-to-match tasks claim their best control first; otherwise an easy
    # task can consume it and leave a severe outlier pair.
    ordered = sorted(
        ood,
        key=lambda item: (-choices[item["id"]][0][0], len(choices[item["id"]]), item["id"]),
    )
    used: set[str] = set()
    matches: list[tuple[dict, dict, float]] = []
    for task in ordered:
        match = next(
            ((distance, candidate) for distance, candidate in choices[task["id"]] if candidate["id"] not in used),
            None,
        )
        if match is None:
            return None
        distance, candidate = match
        used.add(candidate["id"])
        matches.append((task, candidate, distance))
    return matches


def select_compositional_ood(
    items: list[dict], combo_counts: Counter, atom_counts: Counter, seed: int
) -> tuple[list[dict], list[dict], dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for item in items:
        atoms = item["operation_atoms"]
        if (
            atoms != ("other",)
            and len(atoms) >= 2
            and combo_counts[atoms] <= 3
            and all(atom_counts[atom] >= 15 for atom in atoms)
        ):
            groups[atoms].append(item)

    rare_ids = {item["id"] for group in groups.values() for item in group}
    control_pool = [
        item
        for item in items
        if item["id"] not in rare_ids and item["operation_atoms"] != ("other",)
    ]
    if sum(len(group) for group in groups.values()) < COUNTS["test_ood_composition"]:
        raise ValueError("fewer than 40 eligible compositional-OOD candidates")

    rng = random.Random(seed)
    group_list = list(groups.values())
    choices = {
        task["id"]: nearest_controls(task, control_pool)
        for group in group_list
        for task in group
    }
    group_base = {
        tuple(task["id"] for task in group): statistics.fmean(
            choices[task["id"]][0][0] for task in group
        )
        for group in group_list
    }
    best = None
    # Randomized, fully seeded search avoids a brittle single greedy ordering.
    # The objective uses only metadata and operation labels, never model scores.
    for _ in range(2500):
        ranked = sorted(
            group_list,
            key=lambda group: (
                group_base[tuple(task["id"] for task in group)] + rng.random() * 1.75
            ),
        )
        chosen: list[dict] = []
        for group in ranked:
            if len(chosen) + len(group) <= COUNTS["test_ood_composition"]:
                chosen.extend(group)
            if len(chosen) == COUNTS["test_ood_composition"]:
                break
        if len(chosen) != COUNTS["test_ood_composition"]:
            continue
        matches = assign_controls(chosen, choices)
        if matches is None:
            continue
        controls = [control for _, control, _ in matches]
        distances = [distance for _, _, distance in matches]
        held_out_ids = {item["id"] for item in chosen + controls}
        remaining_atom_counts = Counter(
            atom
            for item in items
            if item["id"] not in held_out_ids
            for atom in item["operation_atoms"]
        )
        required_atoms = {atom for item in chosen for atom in item["operation_atoms"]}
        if any(remaining_atom_counts[atom] < 8 for atom in required_atoms):
            continue
        atom_hist = Counter(atom for item in chosen for atom in item["operation_atoms"])
        concentration = max(atom_hist.values()) / len(chosen)
        score = (
            statistics.fmean(distances)
            + 0.8 * max(distances)
            + 1.8 * vector_gap(chosen, controls)
            + 0.4 * concentration
        )
        if best is None or score < best[0]:
            best = (score, chosen, matches)

    if best is None:
        raise ValueError("could not construct a 40-task compositional OOD with matched controls")
    score, ood, matches = best
    controls = [control for _, control, _ in matches]
    distances = [distance for _, _, distance in matches]
    diagnostics = {
        "eligible_task_count": sum(len(group) for group in groups.values()),
        "eligible_combination_count": len(groups),
        "objective": round(score, 6),
        "mean_pair_distance": round(statistics.fmean(distances), 6),
        "max_pair_distance": round(max(distances), 6),
        "max_standardized_mean_gap": round(vector_gap(ood, controls), 6),
        "matched_pairs": [
            {"ood_id": task["id"], "iid_control_id": control["id"], "distance": round(distance, 6)}
            for task, control, distance in sorted(matches, key=lambda match: match[0]["id"])
        ],
    }
    return ood, controls, diagnostics


def mentioned_source_sheet(item: dict) -> bool:
    instruction = item["instruction"].casefold()
    return item["init"]["sheet_count"] >= 2 and any(
        len(sheet["name"]) >= 3 and sheet["name"].casefold() in instruction
        for sheet in item["init"]["sheets"]
    )


def attack_suitability(item: dict) -> dict[str, float]:
    atoms = set(item["operation_atoms"])
    instruction = item["instruction"].lower()
    answer_cells = max(int(item.get("answer_cells") or 0), 1)
    moderate = -0.15 * sum(abs(value) for value in item["structural_vector"][:5])
    scores: dict[str, float] = {}
    if "lookup_join" in atoms or "deduplicate" in atoms:
        scores["duplicate_collision"] = 3.0 + moderate
    if "lookup_join" in atoms or (
        "clean_replace" in atoms
        and re.search(r"\b(?:exact match|case-sensitive|case sensitive|trim|whitespace)\b", instruction)
    ):
        scores["identity_collision"] = 2.5 + moderate
    if "aggregate" in atoms and answer_cells > 0:
        scores["blank_zero_boundary"] = 2.5 + moderate
    if (
        item["output_mode"] == "formula"
        and answer_cells <= 250
        and item["golden"]["formula_count"] > 0
    ):
        scores["formula_liveness"] = 3.0 + moderate
    if mentioned_source_sheet(item):
        scores["source_sheet_decoy"] = 3.0 + moderate
    # Explicit duplicate/exact/blank language raises suitability but is not a
    # prerequisite: the later generator adds the same policy clause to both
    # members of a clean/adversarial pair.
    if re.search(r"\b(?:duplicate|first match|last match)\b", instruction):
        scores["duplicate_collision"] = scores.get("duplicate_collision", 0.0) + 0.75
    if re.search(r"\b(?:exact match|case-sensitive|case sensitive|trim|whitespace)\b", instruction):
        scores["identity_collision"] = scores.get("identity_collision", 0.0) + 0.75
    if re.search(r"\b(?:blank|zero|missing|no match)\b", instruction):
        scores["blank_zero_boundary"] = scores.get("blank_zero_boundary", 0.0) + 0.75
    return scores


def select_adv_sources(pool: list[dict], quota: int = 8) -> tuple[list[dict], dict[str, list[str]]]:
    selected: list[dict] = []
    selected_ids: set[str] = set()
    assignments: dict[str, list[str]] = {}
    # Scarcer families go first so a flexible formula task cannot consume a
    # workbook needed by source-sheet attacks.
    family_candidates = {
        family: sorted(
            ((attack_suitability(item).get(family), item) for item in pool if family in attack_suitability(item)),
            key=lambda pair: (-pair[0], pair[1]["id"]),
        )
        for family in ATTACK_FAMILIES
    }
    for family in sorted(ATTACK_FAMILIES, key=lambda name: len(family_candidates[name])):
        chosen = [item for _, item in family_candidates[family] if item["id"] not in selected_ids][:quota]
        if len(chosen) != quota:
            raise ValueError(f"attack family {family} has only {len(chosen)} disjoint eligible sources")
        assignments[family] = [item["id"] for item in chosen]
        selected.extend(chosen)
        selected_ids.update(item["id"] for item in chosen)
    return selected, assignments


def reserve_train_atom_coverage(
    pool: list[dict], required_atoms: set[str], minimum: int = 8
) -> list[dict]:
    """Reserve a compact set guaranteeing each OOD atom is learned in train."""
    remaining = Counter({atom: minimum for atom in required_atoms})
    selected: list[dict] = []
    selected_ids: set[str] = set()
    while any(value > 0 for value in remaining.values()):
        candidates = []
        for item in pool:
            if item["id"] in selected_ids:
                continue
            covered = sum(
                1 for atom in item["operation_atoms"] if atom in remaining and remaining[atom] > 0
            )
            if covered:
                # Prefer multi-coverage tasks, then structurally ordinary tasks.
                ordinary_cost = sum(abs(value) for value in item["structural_vector"][:5])
                candidates.append((covered, -ordinary_cost, item["id"], item))
        if not candidates:
            missing = sorted(atom for atom, value in remaining.items() if value > 0)
            raise ValueError(f"cannot reserve train coverage for atoms: {missing}")
        item = max(candidates)[-1]
        selected.append(item)
        selected_ids.add(item["id"])
        for atom in item["operation_atoms"]:
            if atom in remaining and remaining[atom] > 0:
                remaining[atom] -= 1
    return selected


def stratum(item: dict) -> tuple[str, str, str]:
    return item["instruction_type"], item["output_mode"], item["operation_atoms"][0]


def stratified_take(items: list[dict], target: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in items:
        groups[stratum(item)].append(item)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[dict] = []
    while len(selected) < target:
        available = [key for key, group in groups.items() if group]
        if not available:
            raise ValueError("stratified pool exhausted")
        key = min(available, key=lambda candidate: (len([x for x in selected if stratum(x) == candidate]) / max(1, len(groups[candidate])), candidate))
        selected.append(groups[key].pop())
    selected_ids = {item["id"] for item in selected}
    return selected, [item for item in items if item["id"] not in selected_ids]


def proportion_gap(left: list[dict], right: list[dict], key) -> float:
    left_counts = Counter(key(item) for item in left)
    right_counts = Counter(key(item) for item in right)
    values = set(left_counts) | set(right_counts)
    return max(
        abs(left_counts[value] / len(left) - right_counts[value] / len(right))
        for value in values
    )


def atom_distribution_gap(left: list[dict], right: list[dict]) -> float:
    left_counts = Counter(atom for item in left for atom in item["operation_atoms"])
    right_counts = Counter(atom for item in right for atom in item["operation_atoms"])
    values = set(left_counts) | set(right_counts)
    return sum(
        abs(left_counts[value] / len(left) - right_counts[value] / len(right))
        for value in values
    ) / max(len(values), 1)


def balance_score(selected: list[dict], reference: list[dict]) -> float:
    selected_mean = mean_vector(selected)
    reference_mean = mean_vector(reference)
    structural = sum(abs(a - b) for a, b in zip(selected_mean, reference_mean))
    return (
        2.0 * structural
        + 1.5 * proportion_gap(selected, reference, lambda item: item["instruction_type"])
        + 1.5 * proportion_gap(selected, reference, lambda item: item["output_mode"])
        + 0.8 * proportion_gap(selected, reference, lambda item: len(item["operation_atoms"]))
        + 0.8 * atom_distribution_gap(selected, reference)
    )


def balanced_complete(
    pool: list[dict],
    fixed: list[dict],
    total: int,
    reference: list[dict],
    seed: int,
    trials: int = 8000,
) -> list[dict]:
    needed = total - len(fixed)
    if needed < 0 or needed > len(pool):
        raise ValueError("invalid balanced selection size")
    if needed == 0:
        return list(fixed)
    rng = random.Random(seed)
    best: tuple[float, list[dict]] | None = None
    for _ in range(trials):
        candidate = fixed + rng.sample(pool, needed)
        score = balance_score(candidate, reference)
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    return best[1]


def summarize(items: list[dict]) -> dict:
    vectors = [structural_raw(item) for item in items]
    return {
        "count": len(items),
        "instruction_type": dict(sorted(Counter(item["instruction_type"] for item in items).items())),
        "output_mode": dict(sorted(Counter(item["output_mode"] for item in items).items())),
        "operation_atoms": dict(sorted(Counter(atom for item in items for atom in item["operation_atoms"]).items())),
        "operation_count": dict(sorted(Counter(len(item["operation_atoms"]) for item in items).items())),
        "structure_means": {
            "log1p_nonempty_cells": round(statistics.fmean(row[0] for row in vectors), 4),
            "log1p_answer_cells": round(statistics.fmean(row[1] for row in vectors), 4),
            "log1p_formula_cells": round(statistics.fmean(row[2] for row in vectors), 4),
            "sheet_count": round(statistics.fmean(row[3] for row in vectors), 4),
            "log1p_instruction_chars": round(statistics.fmean(row[4] for row in vectors), 4),
            "merged_fraction": round(statistics.fmean(row[5] for row in vectors), 4),
            "drawing_fraction": round(statistics.fmean(row[6] for row in vectors), 4),
        },
    }


def write_items(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item["source_item"] for item in sorted(items, key=lambda item: item["id"])]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("benchmark/data/audit/spreadsheetbench_verified_400/task_features.jsonl"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmark/data/extracted/spreadsheetbench_verified_400/dataset.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/data/splits/spreadsheetbench_verified_400/v1_compositional"),
    )
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    items = load_jsonl(args.features)
    source_items = {
        str(item["id"]): item for item in json.loads(args.dataset.read_text(encoding="utf-8"))
    }
    if len(items) != sum(COUNTS.values()) or len(source_items) != len(items):
        raise ValueError("expected exactly 400 audited and source tasks")
    for item in items:
        item["operation_atoms"] = MANUAL_OPERATION_ATOM_OVERRIDES.get(
            item["id"], operation_atoms(item["instruction"])
        )
        item["output_mode"] = output_mode(item)
        item["source_item"] = normalize_task_metadata(source_items[item["id"]])
    standardize(items)

    atom_counts = Counter(atom for item in items for atom in item["operation_atoms"] if atom != "other")
    combo_counts = Counter(item["operation_atoms"] for item in items)
    ood, iid_controls, ood_diagnostics = select_compositional_ood(
        items, combo_counts, atom_counts, args.seed
    )
    excluded_ids = {item["id"] for item in ood + iid_controls}

    required_train_atoms = {atom for item in ood for atom in item["operation_atoms"]}
    reserve_pool = [item for item in items if item["id"] not in excluded_ids]
    train_reserve = reserve_train_atom_coverage(reserve_pool, required_train_atoms)
    train_reserve_ids = {item["id"] for item in train_reserve}

    adv_pool = [
        item
        for item in items
        if item["id"] not in excluded_ids and item["id"] not in train_reserve_ids
    ]
    adv_source, adv_assignments = select_adv_sources(adv_pool)
    excluded_ids.update(item["id"] for item in adv_source)

    iid_extra_pool = [
        item
        for item in items
        if item["id"] not in excluded_ids and item["id"] not in train_reserve_ids
    ]
    iid_reference = [item for item in items if item["id"] not in {x["id"] for x in ood + adv_source}]
    iid = balanced_complete(
        iid_extra_pool,
        iid_controls,
        COUNTS["test_iid"],
        iid_reference,
        args.seed + 1,
    )
    iid_extra = [item for item in iid if item["id"] not in {x["id"] for x in iid_controls}]
    excluded_ids.update(item["id"] for item in iid_extra)

    val_pool = [
        item
        for item in items
        if item["id"] not in excluded_ids and item["id"] not in train_reserve_ids
    ]
    val_reference = val_pool + train_reserve
    val = balanced_complete(
        val_pool, [], COUNTS["val"], val_reference, args.seed + 2
    )
    val_ids = {item["id"] for item in val}
    train_remainder = [item for item in val_pool if item["id"] not in val_ids]
    train = train_remainder + train_reserve
    splits = {
        "train": train,
        "val": val,
        "test_iid": iid,
        "test_ood_composition": ood,
        "adv_source": adv_source,
    }

    split_by_id = {
        item["id"]: name for name, split_items in splits.items() for item in split_items
    }
    near_pairs = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            similarity = token_jaccard(left["instruction"], right["instruction"])
            if similarity >= 0.75:
                near_pairs.append(
                    {"left": left["id"], "right": right["id"], "jaccard": round(similarity, 4)}
                )
    cross_split_near_pairs = [
        pair for pair in near_pairs if split_by_id[pair["left"]] != split_by_id[pair["right"]]
    ]

    train_atoms = Counter(atom for item in train for atom in item["operation_atoms"])
    ood_combos = {item["operation_atoms"] for item in ood}
    non_ood = train + val + iid + adv_source
    overlap_combos = sorted(
        "+".join(combo) for combo in ood_combos if any(item["operation_atoms"] == combo for item in non_ood)
    )
    missing_train_atoms = sorted(
        atom
        for atom in {atom for item in ood for atom in item["operation_atoms"]}
        if train_atoms[atom] < 8
    )
    all_ids = [item["id"] for split_items in splits.values() for item in split_items]
    checks = {
        "exact_counts": {name: len(value) for name, value in splits.items()} == COUNTS,
        "all_400_unique": len(all_ids) == 400 and len(set(all_ids)) == 400,
        "ood_combinations_absent_from_all_non_ood": not overlap_combos,
        "every_ood_atom_seen_at_least_8_times_in_train": not missing_train_atoms,
        "no_cross_split_near_duplicates_at_0_75": not cross_split_near_pairs,
        "ood_iid_max_standardized_mean_gap_le_0_50": (
            vector_gap(ood, iid_controls) <= 0.50
        ),
        "ood_iid_max_pair_distance_le_3_00": ood_diagnostics["max_pair_distance"] <= 3.00,
        "train_val_max_standardized_mean_gap_le_0_50": vector_gap(train, val) <= 0.50,
        "train_iid_max_standardized_mean_gap_le_0_50": vector_gap(train, iid) <= 0.50,
        "train_val_instruction_type_gap_le_0_15": proportion_gap(
            train, val, lambda item: item["instruction_type"]
        ) <= 0.15,
        "train_iid_instruction_type_gap_le_0_15": proportion_gap(
            train, iid, lambda item: item["instruction_type"]
        ) <= 0.15,
        "adv_family_quota_8": all(len(ids) == 8 for ids in adv_assignments.values()),
    }

    output_dir = args.output_dir.resolve()
    for name, split_items in splits.items():
        write_items(output_dir / "canonical" / name / "items.json", split_items)
    # This is a derived subset of test_iid, not a sixth disjoint partition.
    write_items(output_dir / "canonical" / "test_iid_matched" / "items.json", iid_controls)
    for view_name, test_name in (
        ("skillopt_iid", "test_iid"),
        ("skillopt_iid_matched", "test_iid_matched"),
        ("skillopt_ood", "test_ood_composition"),
    ):
        write_items(output_dir / view_name / "train" / "items.json", train)
        write_items(output_dir / view_name / "val" / "items.json", val)
        test_items = iid_controls if test_name == "test_iid_matched" else splits[test_name]
        write_items(output_dir / view_name / "test" / "items.json", test_items)

    manifest = {
        "version": "v1_compositional_candidate",
        "seed": args.seed,
        "status": "accepted_by_static_checks" if all(checks.values()) else "rejected",
        "source_dataset": str(args.dataset.resolve()),
        "counts": {name: len(value) for name, value in splits.items()},
        "semantics": {
            "test_ood_composition": (
                "Untouched real tasks with held-out operation combinations; individual atoms remain seen in train."
            ),
            "adv_source": (
                "Untouched real tasks reserved as scaffolds for a later frozen paired generator; not an adversarial test by itself."
            ),
            "test_iid_matched": (
                "Derived 40-task subset of test_iid matched one-to-one to composition OOD; not an additional partition."
            ),
        },
        "selection": {
            "model_outcomes_used": False,
            "rare_combination_max_frequency": 3,
            "minimum_global_atom_frequency": 15,
            "minimum_train_atom_frequency": 8,
            "near_duplicate_threshold": 0.75,
            "operation_patterns": {name: pattern.pattern for name, pattern in OPERATION_PATTERNS.items()},
            "manual_instruction_review_overrides": {
                task_id: list(atoms) for task_id, atoms in MANUAL_OPERATION_ATOM_OVERRIDES.items()
            },
        },
        "checks": checks,
        "check_details": {
            "overlapping_ood_combinations": overlap_combos,
            "ood_atoms_below_train_minimum": missing_train_atoms,
            "cross_split_near_duplicate_pairs": cross_split_near_pairs,
        },
        "ood_matching": ood_diagnostics,
        "derived_views": {"test_iid_matched_count": len(iid_controls)},
        "adv_source_assignments": adv_assignments,
        "summaries": {name: summarize(value) for name, value in splits.items()},
        "task_annotations": {
            item["id"]: {
                "split": split_by_id[item["id"]],
                "operation_atoms": list(item["operation_atoms"]),
                "combination_frequency_in_verified400": combo_counts[item["operation_atoms"]],
                "output_mode": item["output_mode"],
                "attack_family": next(
                    (family for family, ids in adv_assignments.items() if item["id"] in ids), None
                ),
            }
            for item in sorted(items, key=lambda value: value["id"])
        },
        "near_duplicate_pairs": near_pairs,
        "metadata_overrides": METADATA_OVERRIDES,
        "limitations": [
            "Operation labels are conservative regex annotations and the 40 OOD tasks still require instruction-level human review.",
            "Composition novelty is defined relative to Verified-400, not all spreadsheet tasks in the world.",
            "ADV source assignments certify only source suitability; attacks and transformed oracles are not generated by this script.",
            "No split is selected or revised using checkpoint performance.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review_rows = [
        {
            "id": item["id"],
            "operation_atoms": list(item["operation_atoms"]),
            "combination_frequency": combo_counts[item["operation_atoms"]],
            "instruction_type": item["instruction_type"],
            "output_mode": item["output_mode"],
            "instruction": item["instruction"],
        }
        for item in sorted(ood, key=lambda value: value["id"])
    ]
    (output_dir / "ood_manual_review.json").write_text(
        json.dumps(review_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "checks": checks, "counts": manifest["counts"]}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
