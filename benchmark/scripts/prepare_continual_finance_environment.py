#!/usr/bin/env python3
"""Build isolated Finance/Economics train-50 and holdout-50 task copies."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmark/data/raw/continualskillbench/upstream/Continual-Skill-Bench/tasks/finance-econ-100"
TASKS = ROOT / "benchmark/runtime/continualskillbench/tasks"
TRAIN = TASKS / "finance-econ-100-train50"
HOLDOUT = TASKS / "finance-econ-100-holdout50"
DOCKERFILES = ROOT / "benchmark/docker/continual-finance"
SEPARATOR = "---TASK_BREAK---"
OUTPUT = re.compile(r"/app/task_(\d{3})_output\.json")


def chunks(text: str) -> list[str]:
    return [p.strip() for p in re.split(
        rf"^\s*{re.escape(SEPARATOR)}\s*$", text, flags=re.MULTILINE) if p.strip()]


def normalize_skill_frontmatter(target: Path) -> None:
    for path in sorted((target / "environment/skills").glob("*/SKILL.md")):
        text = path.read_text()
        if text.startswith("---\n"):
            continue
        description = next((line.strip() for line in text.splitlines()[1:]
                            if line.strip() and not line.startswith("#")),
                           f"Reusable guidance for {path.parent.name.replace('-', ' ')} tasks.")
        path.write_text("---\n" + f"name: {path.parent.name}\n" +
                        f"description: {json.dumps(description, ensure_ascii=False)}\n---\n\n" + text)


def derive(target: Path, selected: list[str], specs: list[dict], image: str,
           dockerfile: str, note: str, keep_data: set[str] | None = None) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(SOURCE, target)
    shutil.copy2(DOCKERFILES / dockerfile, target / "environment/Dockerfile")
    normalize_skill_frontmatter(target)
    (target / "instruction.md").write_text(f"\n{SEPARATOR}\n".join(selected) + "\n")
    (target / "environment/tasks_spec.json").write_text(
        json.dumps(specs, indent=2, ensure_ascii=False) + "\n")
    toml = (target / "task.toml").read_text()
    assert "docker_image" not in toml
    toml = toml.replace("[environment]\n", f'[environment]\ndocker_image = "{image}"\n', 1)
    (target / "task.toml").write_text(toml + f"\n# Derived protocol split: {note}.\n")
    if keep_data is not None:
        data = target / "environment/data"
        for path in data.iterdir():
            if path.is_file() and path.name not in keep_data:
                path.unlink()


def main() -> None:
    lock = json.loads((ROOT / "benchmark/configs/continual_finance_order_lock.json").read_text())
    for name, expected in lock["sha256"].items():
        observed = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        assert observed == expected, f"raw Finance release changed: {name}"
    task_chunks = chunks((SOURCE / "instruction.md").read_text())
    specs = json.loads((SOURCE / "environment/tasks_spec.json").read_text())
    assert len(task_chunks) == len(specs) == 100
    for chunk, spec in zip(task_chunks, specs):
        match = OUTPUT.search(chunk)
        assert match and int(match.group(1)) == int(spec["task_id"])
    derive(TRAIN, task_chunks[:50], specs[:50], "agent-overfit-continual-finance:baseline-v1",
           "Dockerfile.train50", "official tasks 1-50 only",
           {"hon.json", "btc.json", "uvv.json", "jnj.json", "msft.json"})
    derive(HOLDOUT, task_chunks[50:], specs[50:], "agent-overfit-continual-finance:holdout-v1",
           "Dockerfile.holdout50", "official tasks 51-100 only")
    assert [x["task_id"] for x in json.loads((TRAIN / "environment/tasks_spec.json").read_text())] == list(range(1, 51))
    assert [x["task_id"] for x in json.loads((HOLDOUT / "environment/tasks_spec.json").read_text())] == list(range(51, 101))
    assert {p.name for p in (TRAIN / "environment/data").iterdir() if p.is_file()} == {
        "hon.json", "btc.json", "uvv.json", "jnj.json", "msft.json"}
    print("PASS: raw Finance untouched; isolated official train50/holdout50 tasks prepared")
    print("train: 1-50; holdout: 51-100; task labels remain judge-side only")


if __name__ == "__main__":
    main()
