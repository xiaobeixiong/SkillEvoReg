"""Create a separate task copy; never edit the raw release or past results."""
from pathlib import Path
import hashlib
import json
import shutil
import tomllib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmark/data/raw/continualskillbench/upstream/Continual-Skill-Bench/tasks/math-100"
TARGET = ROOT / "benchmark/runtime/continualskillbench/tasks/math-100"
IMAGE = "agent-overfit-continual-math:baseline-v1"
lock = json.loads((ROOT / "benchmark/configs/continual_math_order_lock.json").read_text())
for name, digest in lock["sha256"].items():
    assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
if not TARGET.exists():
    shutil.copytree(SOURCE, TARGET)
    config = TARGET / "task.toml"
    original = config.read_text()
    assert original.count('docker_image = "hb-math"') == 1
    config.write_text(original.replace('docker_image = "hb-math"', f'docker_image = "{IMAGE}"'))
    shutil.copyfile(ROOT / "benchmark/docker/continual-math/Dockerfile", TARGET / "environment/Dockerfile")
else:
    assert tomllib.loads((TARGET / "task.toml").read_text())["environment"]["docker_image"] == IMAGE
for path in SOURCE.rglob("*"):
    if not path.is_file():
        continue
    relative = path.relative_to(SOURCE)
    if relative.as_posix() in {"task.toml", "environment/Dockerfile"}:
        continue
    assert (TARGET / relative).read_bytes() == path.read_bytes(), str(relative)
assert (TARGET / "environment/Dockerfile").read_bytes() == (ROOT / "benchmark/docker/continual-math/Dockerfile").read_bytes()
print("PASS: isolated Math task copy; instructions, grading and meta-skills unchanged")
