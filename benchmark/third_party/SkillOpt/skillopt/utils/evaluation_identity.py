"""Bind a frozen-evaluation output directory to its skill and protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _public_config(value):
    if isinstance(value, dict):
        return {key: _public_config(item) for key, item in value.items()
                if not key.startswith('_') and key not in {'out_root', 'api_key', 'password', 'secret'}
                and not key.endswith(('_api_key', '_password', '_secret'))}
    if isinstance(value, list):
        return [_public_config(item) for item in value]
    return value


def effective_api_route(config):
    shared = (config.get('azure_openai_endpoint') or config.get('azure_endpoint')
              or os.environ.get('OPENAI_BASE_URL') or os.environ.get('YIBU_API_BASE')
              or os.environ.get('AZURE_OPENAI_ENDPOINT', ''))
    return {role: (config.get(role + '_azure_openai_endpoint') or shared
                   or os.environ.get(role.upper() + '_AZURE_OPENAI_ENDPOINT', ''))
            for role in ('optimizer', 'target')}


def evaluation_identity(skill_content, config, split, *, items=None):
    canonical = {'valid_seen': 'val', 'valid_unseen': 'test'}.get(split, split)
    route = effective_api_route(config)
    return dict(schema_version=1,
                skill_sha256=hashlib.sha256(skill_content.encode()).hexdigest(),
                protocol_sha256=_digest(dict(config=_public_config(config), split=canonical, route=route)),
                api_route=route,
                items_sha256=_digest(items) if items is not None else None,
                split=canonical)


def bind_evaluation_output(output, identity):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('history.json', 'runtime_state.json', 'best_skill.md')):
        raise ValueError('Evaluation cannot write into a training directory')
    path = output / 'evaluation_identity.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('Evaluation cache identity mismatch: use a new output directory')
        return
    # The checkpoint wrapper may open its command log before launching eval.
    if any(p.name != 'eval.log' for p in output.iterdir()):
        raise ValueError('Existing evaluation artifacts have no verified identity; use a new output directory')
    with path.open('x') as handle:
        json.dump(identity, handle, indent=2)


def completed_evaluation(output, identity, expected_items):
    path = Path(output) / 'eval_summary.json'
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    if summary.get('evaluation_identity') != identity or summary.get('n_items') != expected_items:
        raise ValueError('Completed evaluation summary does not match the requested identity/items')
    return True
