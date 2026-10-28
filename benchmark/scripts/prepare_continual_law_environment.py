#!/usr/bin/env python3
"""Derive isolated, ordered 50/50 Law tasks from the pinned upstream release."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'benchmark/data/raw/continualskillbench/upstream/Continual-Skill-Bench/tasks/law-100'
TASKS = ROOT / 'benchmark/runtime/continualskillbench/tasks'
DOCKERFILES = ROOT / 'benchmark/docker/continual-law'
SEPARATOR = '---TASK_BREAK---'
FILES = {
    'train50': {'transactions.csv', 'lease_agreement.txt', 'emissions_data.csv',
                'labor_requirements.csv', 'employment_contract.txt', 'system_config.json'},
    'holdout50': {'controls.csv', 'processing_records.csv', 'nda-contract.txt',
                  'merger_agreement.txt', 'license_agreement.txt'},
}


def chunks(text):
    return [p.strip() for p in re.split(r'^\s*---TASK_BREAK---\s*$', text,
            flags=re.MULTILINE) if p.strip()]


def derive(name, items, specs, image):
    target = TASKS / f'law-100-{name}'
    if target.exists():
        raise RuntimeError(f'refusing to replace existing derived task: {target}')
    shutil.copytree(SOURCE, target)
    shutil.copy2(DOCKERFILES / f'Dockerfile.{name}', target / 'environment/Dockerfile')
    shutil.copy2(DOCKERFILES / f'setup_{name}.sh', target / 'environment/setup_files.sh')
    (target / 'instruction.md').write_text(f'\n{SEPARATOR}\n'.join(items) + '\n')
    (target / 'environment/tasks_spec.json').write_text(json.dumps(specs, indent=2) + '\n')
    toml = (target / 'task.toml').read_text()
    assert toml.count('docker_image =') == 1
    toml = re.sub(r'^docker_image\s*=\s*"[^"]+"\s*$',
        f'docker_image = "{image}"', toml, count=1, flags=re.MULTILINE)
    (target / 'task.toml').write_text(toml +
        f'\n# Derived 50/50 protocol: {name}.\n')
    for skill in (target / 'environment/skills').glob('*/SKILL.md'):
        text = skill.read_text()
        if not text.startswith('---\n'):
            description = next((line.strip() for line in text.splitlines()[1:]
                if line.strip() and not line.startswith('#')),
                f'Reusable guidance for {skill.parent.name}.')
            skill.write_text('---\n' + f'name: {skill.parent.name}\n' +
                f'description: {json.dumps(description)}\n---\n\n' + text)
    keep_ids = {int(x['task_id']) for x in specs}
    for path in (target / 'environment/data').iterdir():
        if path.is_dir() and path.name.startswith('task-'):
            if int(path.name.split('-')[-1]) not in keep_ids:
                shutil.rmtree(path)
        elif path.is_file() and path.name not in FILES[name]:
            path.unlink()
    assert {x.name for x in (target / 'environment/data').iterdir() if x.is_file()} == FILES[name]
    assert all(x['task_id'] in keep_ids for x in json.loads((target / 'environment/tasks_spec.json').read_text()))
    return target


def main():
    instruction = (SOURCE / 'instruction.md').read_text()
    specs_text = (SOURCE / 'environment/tasks_spec.json').read_bytes()
    specs = json.loads(specs_text)
    parts = chunks(instruction)
    assert len(parts) == len(specs) == 100
    assert [x['task_id'] for x in specs] == list(range(1, 101))
    for index, (part, row) in enumerate(zip(parts, specs), 1):
        assert f'/app/task_{index:03d}_output.json' in part
        assert part.startswith(f'## Task {index}:')
    lock = ROOT / 'benchmark/configs/continual_law_order_lock.json'
    if lock.exists():
        expected = json.loads(lock.read_text())['sha256']
        assert hashlib.sha256(instruction.encode()).hexdigest() == expected['instruction.md']
        assert hashlib.sha256(specs_text).hexdigest() == expected['environment/tasks_spec.json']
    else:
        lock.write_text(json.dumps({'source_commit': '083f14de18227a1eed15758eb1a80632966525c0',
            'sha256': {'instruction.md': hashlib.sha256(instruction.encode()).hexdigest(),
                       'environment/tasks_spec.json': hashlib.sha256(specs_text).hexdigest()}}, indent=2) + '\n')
    derive('train50', parts[:50], specs[:50], 'agent-overfit-continual-law:baseline-v1')
    derive('holdout50', parts[50:], specs[50:], 'agent-overfit-continual-law:holdout-v1')
    print('PASS: pinned Law order; train IDs 1-50; frozen holdout IDs 51-100; disjoint task data')


if __name__ == '__main__':
    main()
