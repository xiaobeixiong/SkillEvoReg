#!/usr/bin/env python3
"""Generate skill-priority Healthcare frozen evaluation configs."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ROOT / "benchmark/runtime/continualskillbench/skillpriority"
LIB = "benchmark/runtime/continualskillbench/frozen_healthcare_libraries"
TEMPLATE = """job_name: {job}
jobs_dir: benchmark/outputs/continualskillbench
n_attempts: 1
timeout_multiplier: 1.0
orchestrator:
  type: local
  n_concurrent_trials: 1
  quiet: true
  retry: {{max_retries: 0}}
environment:
  type: docker
  force_build: false
  delete: false
  override_memory_mb: 3072
agents:
  - import_path: continual_holdout.stateless_agent:HealthcareStatelessHeldout
    model_name: openai/gpt-5.4-mini
    override_setup_timeout_sec: 600
    override_timeout_sec: .inf
    kwargs:
      reasoning_effort: medium
      enable_judge_feedback: true
      judge_model: openai/gpt-5.4-mini
      judge_pass_threshold: 0.6
      enable_skill_probe: false
      compact_every_n: 1
      freeze_skill_updates: true
      frozen_library: {library}
      prefer_skills: true
datasets:
  - path: benchmark/runtime/continualskillbench/tasks
    task_names: [healthcare-100-holdout50]
"""

for condition in ("baseline", "regularization"):
    for repeat in (1, 2):
        job = f"continual_healthcare_frozen_holdout50_{condition}_repeat{repeat:02d}_01_skillpriority"
        TARGETS.mkdir(parents=True, exist_ok=True)
        target = TARGETS / f"{job}.yaml"
        content = TEMPLATE.format(job=job, library=f"{LIB}/{condition}_step050")
        if target.exists() and yaml.safe_load(target.read_text()) != yaml.safe_load(content):
            raise RuntimeError(f"refusing to overwrite semantically different config: {target}")
        if not target.exists():
            target.write_text(content)
        print(target.relative_to(ROOT))
