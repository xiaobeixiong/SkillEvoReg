"""Prepare a fresh job from a published, complete task-boundary checkpoint.

No API calls; the source trial and checkpoint are never modified. Legacy
session reconstruction and partial-task migration are not release entrypoints.
"""
import argparse
from pathlib import Path
import re

import yaml
from continual_regularization.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        raise ValueError('Invalid run name')
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(config['jobs_dir']) / args.run_name
    if output.exists():
        raise ValueError('New run directory already exists')
    checkpoint, data = load_checkpoint(Path(args.checkpoint).resolve())
    if data['recovery_kind'] != 'checkpoint':
        raise ValueError('Only complete native checkpoints are supported by this release helper')
    config['job_name'] = args.run_name
    config['agents'][0]['kwargs'].update(resume_checkpoint=str(checkpoint),
                                        allow_legacy_recovery=False)
    work = Path('benchmark/runtime/continualskillbench/resume') / args.run_name
    work.mkdir(parents=True, exist_ok=True)
    dest = work / 'run.yaml'
    if dest.exists():
        raise ValueError('Prepared config already exists')
    dest.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'Prepared {dest}: resume from task {data["completed_step"]+1}; mode={data["recovery_kind"]}')


if __name__ == '__main__':
    main()
