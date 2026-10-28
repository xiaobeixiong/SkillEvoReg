#!/usr/bin/env python3
"""Derive a tail-holdout task directory from math-100 (new directory; source untouched).

The held-out directory contains only subtasks ``split+1 .. 100``.  That matters for
two independent reasons:

1. ``environment/tasks_spec.json`` is what the verifier aggregates over, and tasks
   without a feedback file score 0 -- so a task directory with all 100 specs would
   silently divide the held-out sum by 100.
2. Keeping the original subtask text (including the ``/app/task_NNN_output.json``
   requirement) preserves the judge task id, which is inferred from that path.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "benchmark/runtime/continualskillbench/tasks"
SOURCE = TASKS / "math-100"
SEPARATOR = "---TASK_BREAK---"
OUTPUT_PATTERN = re.compile(r"/app/task_(\d{1,4})_output\.json")

COPY_ALWAYS = ["tests", "solution"]
COPY_ENVIRONMENT = ["Dockerfile", "claude-settings.json", "skills"]


def split_tasks(instruction: str) -> list[str]:
    parts = re.split(rf"^\s*{re.escape(SEPARATOR)}\s*$", instruction, flags=re.MULTILINE)
    return [part.strip() for part in parts if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TASKS / "math-100-holdout50")
    parser.add_argument("--split", type=int, default=50, help="subtasks 1..split are training")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.target = args.target.resolve()

    if not args.source.is_dir():
        print(f"ERROR: source task missing: {args.source}", file=sys.stderr)
        return 1

    tasks = split_tasks((args.source / "instruction.md").read_text())
    spec = json.loads((args.source / "environment/tasks_spec.json").read_text())
    spec = spec if isinstance(spec, list) else spec.get("tasks", [])
    if len(tasks) != len(spec):
        print(f"ERROR: source mismatch: {len(tasks)} tasks vs {len(spec)} spec entries", file=sys.stderr)
        return 1

    held_tasks = tasks[args.split:]
    held_spec = spec[args.split:]
    if not held_tasks:
        print("ERROR: split leaves no evaluation subtasks", file=sys.stderr)
        return 1

    # Integrity: instruction output path -> spec task id, in order.
    for index, (chunk, entry) in enumerate(zip(held_tasks, held_spec), start=args.split + 1):
        task_id = int(entry["task_id"])
        match = OUTPUT_PATTERN.search(chunk)
        if not match or int(match.group(1)) != task_id:
            print(f"ERROR: subtask {index} output path does not match spec id", file=sys.stderr)
            return 1
        if f"## Task {task_id} " not in chunk.splitlines()[0]:
            print(f"WARNING: task {task_id} heading is {chunk.splitlines()[0]!r}", file=sys.stderr)

    if args.target.exists():
        shutil.rmtree(args.target)
    (args.target / "environment").mkdir(parents=True)

    (args.target / "instruction.md").write_text(
        f"\n{SEPARATOR}\n".join(held_tasks) + "\n"
    )
    (args.target / "environment/tasks_spec.json").write_text(
        json.dumps(held_spec, indent=2, ensure_ascii=False)
    )
    for name in COPY_ENVIRONMENT:
        source = args.source / "environment" / name
        if not source.exists():
            print(f"WARNING: missing environment/{name}", file=sys.stderr)
            continue
        if source.is_dir():
            shutil.copytree(source, args.target / "environment" / name)
        else:
            shutil.copy2(source, args.target / "environment" / name)
    for name in COPY_ALWAYS:
        source = args.source / name
        if not source.exists():
            print(f"WARNING: missing {name}", file=sys.stderr)
            continue
        shutil.copytree(source, args.target / name) if source.is_dir() else shutil.copy2(source, args.target / name)

    # task.toml: keep the image and resource envelope, restate the subtask count.
    toml = (args.source / "task.toml").read_text()
    toml = toml.replace(
        f'author_name = "SkillsBench Law"',
        f'author_name = "SkillsBench Law"',
    )
    toml += (
        f'\n# Derived tail-holdout task: subtasks {args.split + 1}-{args.split + len(held_tasks)} '
        f'of the math-100 sequence, learned on subtasks 1-{args.split}.\n'
    )
    (args.target / "task.toml").write_text(toml)

    print(f"target          {args.target.relative_to(REPO)}")
    print(f"evaluation      task ids {held_spec[0]['task_id']}..{held_spec[-1]['task_id']} ({len(held_tasks)} tasks)")
    print(f"spec entries    {len(held_spec)}")
    print(f"train excluded  subtasks 1..{args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
