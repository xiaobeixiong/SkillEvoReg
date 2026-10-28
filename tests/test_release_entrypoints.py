"""Entrypoint checks without model calls, Docker or dataset downloads."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run import environment, plan


def registry():
    return json.loads((ROOT / 'experiments.json').read_text())['experiments']


def test_skillopt_config_paths():
    for name in registry():
        if name.startswith('skillopt/'):
            _, dataset, method = name.split('/')
            command, cwd, env = plan(argparse.Namespace(
                framework='skillopt', dataset=dataset, method=method,
                evaluate=False, skill=None, split='test', cfg_options=[]))
            assert cwd == ROOT
            assert (ROOT / command[command.index('--config') + 1]).is_file()
            assert str(ROOT / 'benchmark/third_party/SkillOpt') in env['PYTHONPATH'].split(os.pathsep)


def test_host_specific_runtime():
    assert environment('continual')['PYTHONPATH'].split(os.pathsep)[0] == str(ROOT / 'vendor/harbor_continual/src')
    assert environment('evol')['PYTHONPATH'].split(os.pathsep)[0] == str(ROOT / 'vendor/harbor_evol/src')


def test_planning_is_read_only(tmp_path):
    args = argparse.Namespace(framework='continual', domain='law', method='regularization', phase='test',
        frozen_library=str(tmp_path / 'library'), run_name='test_plan', checkpoint=None)
    command, _, env = plan(args)
    assert '_RELEASE_PREPARED_CONFIG' in env
    assert not Path(command[-1]).exists()


def test_second_pass_requires_anchor():
    args = argparse.Namespace(framework='evol', method='skillevoreg', run_id='test', anchor=None)
    with pytest.raises(ValueError, match='complete shared'):
        plan(args)


def test_supported_entries():
    expected = {f'skillopt/{dataset}/{method}'
                for dataset in ('spreadsheet', 'searchqa', 'livemath')
                for method in ('baseline', 'skillevoreg')}
    expected.update(f'skillevolbench/{method}' for method in ('baseline', 'twopass', 'skillevoreg'))
    expected.update(f'continual/{domain}/{method}/{phase}'
                    for domain in ('math', 'finance', 'law', 'healthcare', 'office')
                    for method in ('baseline', 'regularization')
                    for phase in ('train', 'test'))
    assert set(registry()) == expected


def test_frozen_configs():
    for name, entry in registry().items():
        if name.startswith('continual/') and name.endswith('/test'):
            kwargs = yaml.safe_load((ROOT / entry['config']).read_text())['agents'][0]['kwargs']
            assert kwargs['freeze_skill_updates'] and kwargs['prefer_skills']
            assert not kwargs.get('resume_checkpoint')


def test_no_bundled_run_outputs():
    for name in ('results', 'prompts', 'tools', 'docs/provenance',
                 'benchmark/outputs',
                 'benchmark/third_party/SkillEvolBench/configs/t1_anchors'):
        assert not (ROOT / name).exists()


def test_model_prompts_remain_in_implementation():
    for name in ('benchmark/third_party/SkillOpt/skillopt/engine/trainer.py',
                 'benchmark/third_party/SkillEvolBench/skillevolbench/components/ccv.py',
                 'benchmark/integrations/continual_regularization/ccv.py'):
        assert (ROOT / name).is_file()
        ast.parse((ROOT / name).read_text())
