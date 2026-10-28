"""Run SkillOpt with the Verified-400 metadata/evaluator compatibility layer."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLOPT_ROOT = REPO_ROOT / "benchmark" / "third_party" / "SkillOpt"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SKILLOPT_ROOT))

from benchmark.evaluation.verified400_adapter import (  # noqa: E402
    evaluate,
    verification_report,
)


def install_adapter() -> None:
    import skillopt.envs.spreadsheetbench.codegen_agent as codegen_agent
    import skillopt.envs.spreadsheetbench.evaluator as evaluator
    import skillopt.envs.spreadsheetbench.rollout as rollout

    evaluator.evaluate = evaluate
    rollout.evaluate = evaluate
    rollout._auto_verify_output = verification_report
    codegen_agent.evaluate = evaluate


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=("train", "eval"))
    args, remaining = parser.parse_known_args()

    install_adapter()
    script_name = "train.py" if args.command == "train" else "eval_only.py"
    script_path = SKILLOPT_ROOT / "scripts" / script_name
    sys.argv = [str(script_path), *remaining]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
