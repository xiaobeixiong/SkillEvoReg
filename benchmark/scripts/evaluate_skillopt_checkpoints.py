"""Evaluate checkpoints on the splits declared by an explicit configuration."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

from stream_subprocess import run_and_tee


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLOPT_ROOT = REPO_ROOT / "benchmark/third_party/SkillOpt"
sys.path.insert(0, str(SKILLOPT_ROOT))

from skillopt.config import flatten_config, load_config  # noqa: E402

RUNNER = REPO_ROOT / "benchmark/scripts/run_skillopt.py"
INITIAL_SKILL = (
    REPO_ROOT
    / "benchmark/third_party/SkillOpt/skillopt/envs/spreadsheetbench/skills/initial.md"
)


def checkpoint_skills(train_root: Path, config_path: Path | None = None) -> list[tuple[str, Path]]:
    history = json.loads((train_root / "history.json").read_text(encoding="utf-8"))
    epoch_steps: dict[int, int] = {}
    for record in history:
        epoch = int(record["epoch"])
        epoch_steps[epoch] = max(epoch_steps.get(epoch, 0), int(record["step"]))
    if config_path is not None:
        cfg = flatten_config(load_config(str(config_path.resolve())))
    elif (train_root / "config.json").is_file():
        cfg = json.loads((train_root / "config.json").read_text())
        cfg = flatten_config(cfg)
    else:
        cfg = {}
    initial = Path(cfg.get("skill_init") or INITIAL_SKILL)
    if not initial.is_absolute():
        initial = REPO_ROOT / initial
    skills = [("initial", initial)]
    skills.extend(
        (f"epoch_{epoch:02d}", train_root / "skills" / f"skill_v{step:04d}.md")
        for epoch, step in sorted(epoch_steps.items())
    )
    if len(skills) != 9:
        raise RuntimeError(
            f"Expected initial plus 8 epoch checkpoints, found {len(skills)}"
        )
    missing = [str(path) for _, path in skills if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint skills: {missing}")
    return skills


def evaluation_specs(config_path: Path) -> list[dict]:
    cfg = flatten_config(load_config(str(config_path.resolve())))
    workers = max(1, int(cfg.get("workers", 1)))
    split_root = Path(cfg["split_dir"])
    if not split_root.is_absolute():
        split_root = REPO_ROOT / split_root
    sizes = {
        split: len(json.loads((split_root / split / "items.json").read_text(encoding="utf-8")))
        for split in ("train", "val", "test")
    }
    return [
        {"name": "train", "split": "train", "size": sizes["train"], "workers": workers},
        {"name": "val", "split": "valid_seen", "size": sizes["val"], "workers": workers},
        {"name": "test", "split": "valid_unseen", "size": sizes["test"], "workers": workers},
    ]


def refresh_summaries(eval_root: Path, seed: int) -> None:
    rows = []
    for summary_path in sorted(eval_root.glob("*/*/eval_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "condition": summary_path.parent.parent.name,
                "split": summary_path.parent.name,
                "n_items": payload.get("n_items"),
                "hard": payload.get("hard"),
                "soft": payload.get("soft"),
                "skill": payload.get("skill"),
            }
        )
    (eval_root / "checkpoint_evaluation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (eval_root / "checkpoint_evaluation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "condition",
                "split",
                "n_items",
                "hard",
                "soft",
                "skill",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def run_spec(
    condition: str,
    skill_path: Path,
    spec: dict,
    eval_root: Path,
    seed: int,
    config_path: Path,
) -> tuple[str, int]:
    output = eval_root / condition / spec["name"]
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        str(RUNNER),
        "eval",
        "--config",
        str(config_path),
        "--skill",
        str(skill_path),
        "--split",
        spec["split"],
        "--out_root",
        str(output),
        "--test_env_num",
        str(spec["size"]),
        "--workers",
        str(spec["workers"]),
        "--seed",
        str(seed),
    ]
    with (output / "eval.log").open("a", encoding="utf-8") as log:
        log.write(f"\nCOMMAND: {' '.join(command)}\n")
        log.flush()
        completed = run_and_tee(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            log=log,
        )
    return spec["name"], completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition")
    parser.add_argument("--skill", type=Path)
    parser.add_argument(
        "--only-split",
        choices=("train", "val", "test"),
        help="Evaluate only one reporting split (used to fill missing repeats).",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        help="Optional independent output directory for repeated evaluations.",
    )
    args = parser.parse_args()
    if not any(os.environ.get(name) for name in ("OPENAI_API_KEY", "YIBU_API_KEY", "AZURE_OPENAI_API_KEY")):
        raise RuntimeError("Set OPENAI_API_KEY (or a supported provider key)")
    if bool(args.condition) != bool(args.skill):
        raise ValueError("--condition and --skill must be provided together")

    train_root = args.train_root.resolve()
    eval_root = (
        args.eval_root.resolve()
        if args.eval_root
        else train_root.parent / "checkpoint_evals"
    )
    eval_root.mkdir(parents=True, exist_ok=True)
    checkpoints = (
        [(args.condition, args.skill.resolve())]
        if args.condition
        else checkpoint_skills(train_root, args.config)
    )
    specs = evaluation_specs(args.config)
    if args.only_split:
        specs = [spec for spec in specs if spec["name"] == args.only_split]
    for condition, skill_path in checkpoints:
        # Splits run sequentially, so the config's workers value is also the
        # global pooled-container/API concurrency cap.
        failures = []
        for spec in specs:
            name, exit_code = run_spec(
                condition,
                skill_path,
                spec,
                eval_root,
                args.seed,
                args.config,
            )
            if exit_code != 0:
                failures.append((name, exit_code))
        refresh_summaries(eval_root, args.seed)
        if failures:
            raise RuntimeError(f"Evaluation failed for {condition}: {failures}")
    refresh_summaries(eval_root, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
