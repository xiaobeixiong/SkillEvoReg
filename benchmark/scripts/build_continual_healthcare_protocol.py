#!/usr/bin/env python3
"""Build isolated Healthcare 1--50 training and 51--100 frozen-eval tasks."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "benchmark/data/raw/continualskillbench/upstream/"
          "Continual-Skill-Bench/tasks/healthcare-100")
DEST = ROOT / "benchmark/runtime/continualskillbench/tasks"


def referenced_data(specs: list[dict], data: Path) -> set[Path]:
    text = "\n".join(str(row.get("instruction", "")) for row in specs)
    keep: set[Path] = set()
    for path in data.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(data)
        if path.name in text or rel.as_posix() in text or any(part in text for part in rel.parts[:-1]):
            keep.add(rel)
    return keep


def write_split(name: str, specs: list[dict], parts: list[str], image: str) -> None:
    target = DEST / name
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing task: {target}")
    shutil.copytree(SOURCE, target)
    (target / "instruction.md").write_text("\n---TASK_BREAK---\n".join(parts) + "\n")
    (target / "environment/tasks_spec.json").write_text(
        json.dumps(specs, indent=2, ensure_ascii=False) + "\n")

    source_data = SOURCE / "environment/data"
    keep = referenced_data(specs, source_data)
    data = target / "environment/data"
    shutil.rmtree(data)
    data.mkdir()
    for rel in sorted(keep):
        destination = data / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_data / rel, destination)

    setup = target / "environment/setup_files.sh"
    setup.write_text("""#!/usr/bin/env bash
set -euo pipefail
target="${1:?workspace required}"
mkdir -p "$target"
find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp -a /root/data/. "$target/"
""")
    setup.chmod(0o755)
    (target / "environment/Dockerfile").write_text("""FROM agent-overfit-continual-law:deps-v1
RUN rm -rf /root/data /app/workspace /root/.agents/skills /root/.claude/skills /root/.terminus/skills /root/.codex/skills /root/.opencode/skill /root/.goose/skills /root/.factory/skills /root/.gemini/skills
RUN mkdir -p /root/data /app/workspace
COPY data/ /root/data/
COPY setup_files.sh /root/setup_files.sh
RUN bash /root/setup_files.sh /app/workspace
COPY skills /root/.agents/skills
COPY skills /root/.claude/skills
COPY skills /root/.terminus/skills
COPY skills /root/.codex/skills
COPY skills /root/.opencode/skill
COPY skills /root/.goose/skills
COPY skills /root/.factory/skills
COPY skills /root/.gemini/skills
RUN test ! -e /root/tasks_spec.json && python3 -m pip check
""")
    task_toml = (target / "task.toml").read_text()
    task_toml = re.sub(r'docker_image\s*=\s*"[^"]+"', f'docker_image = "{image}"', task_toml)
    task_toml += f"\n# Derived fixed 50/50 protocol: {name.rsplit('-', 1)[-1]}.\n"
    (target / "task.toml").write_text(task_toml)
    print(name, f"tasks={specs[0]['task_id']}-{specs[-1]['task_id']}",
          f"input_files={len(keep)}", f"image={image}")


def main() -> None:
    specs = json.loads((SOURCE / "environment/tasks_spec.json").read_text())
    parts = [part.strip() for part in (SOURCE / "instruction.md").read_text().split(
        "---TASK_BREAK---") if part.strip()]
    assert len(specs) == len(parts) == 100
    assert [row["task_id"] for row in specs] == list(range(1, 101))
    write_split("healthcare-100-train50", specs[:50], parts[:50],
                "agent-overfit-continual-healthcare:train-v1")
    write_split("healthcare-100-holdout50", specs[50:], parts[50:],
                "agent-overfit-continual-healthcare:holdout-v1")


if __name__ == "__main__":
    main()
