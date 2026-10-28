"""Check source syntax, configuration paths and accidental credentials."""
import ast
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '__pycache__', '.pytest_cache', '.cache', 'outputs', 'runtime'}
SECRET = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,})\b')


def verify():
    errors = []
    manifest = json.loads((ROOT / 'experiments.json').read_text())
    python_count = 0
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED or part.startswith('.venv') for part in relative.parts):
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if SECRET.search(text):
            errors.append(f'Credential-shaped string: {relative}')
        if not path.is_relative_to(ROOT / 'vendor') and re.search(r'/Users/[A-Za-z][A-Za-z0-9._-]*/', text):
            errors.append(f'Personal absolute path: {relative}')
        # Task fixtures intentionally contain programs for the agent to repair.
        fixture = path.is_relative_to(ROOT / 'benchmark/third_party/SkillEvolBench/benchmark/tasks')
        if path.suffix == '.py' and not fixture:
            python_count += 1
            try:
                ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                errors.append(f'Syntax: {relative}:{exc.lineno}: {exc.msg}')
    for name, entry in manifest['experiments'].items():
        paths = [entry['config']] if 'config' in entry else [entry['baseline'], entry['strategy']]
        for relative in paths:
            path = ROOT / relative
            if not path.is_file():
                errors.append(f'Missing config: {relative}')
                continue
            value = yaml.safe_load(path.read_text())
            if '_base_' in value:
                errors.append(f'Unresolved configuration inheritance: {relative}')
            hook = value.get('evaluation', {}).get('epoch_end_eval_script')
            if hook and not (ROOT / hook).is_file():
                errors.append(f'Missing epoch evaluation script: {relative}: {hook}')
            if name.startswith('continual/') and name.endswith('/test'):
                kwargs = value['agents'][0]['kwargs']
                if not kwargs.get('prefer_skills') or not kwargs.get('freeze_skill_updates'):
                    errors.append(f'Invalid frozen evaluation settings: {relative}')
    result = dict(status='passed' if not errors else 'failed',
                  python_files=python_count, registered_experiments=len(manifest['experiments']),
                  errors=errors)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    raise SystemExit(0 if verify()['status'] == 'passed' else 1)
