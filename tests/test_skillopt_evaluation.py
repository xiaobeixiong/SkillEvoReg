"""Publication entrypoint regressions; all rollout/model calls are stubbed."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'benchmark/scripts'),
                str(ROOT / 'benchmark/third_party/SkillOpt')]
from run import environment, plan
from skillopt.utils.evaluation_identity import (
    bind_evaluation_output, completed_evaluation, evaluation_identity,
)
import evaluate_skillopt_checkpoints as checkpoints


def eval_args(skill, **changes):
    values = dict(framework='skillopt', dataset='searchqa', method='skillevoreg',
                  evaluate=True, skill=str(skill), split='test', cfg_options=[],
                  eval_repeat=None, eval_out_root=None)
    values.update(changes)
    return argparse.Namespace(**values)


def test_evaluation_plan_is_fresh_and_read_only(tmp_path):
    skill = tmp_path / 'skill.md'; skill.write_text('Rule A')
    first, _, _ = plan(eval_args(skill))
    second, _, _ = plan(eval_args(skill))
    a = Path(first[first.index('--out_root') + 1])
    b = Path(second[second.index('--out_root') + 1])
    assert a != b
    assert 'evaluations' in a.parts
    assert not a.exists() and not b.exists()


def test_repeat_skill_and_protocol_are_separate(tmp_path):
    skill = tmp_path / 'skill.md'; skill.write_text('Rule A')
    def output(**kw):
        command, _, _ = plan(eval_args(skill, eval_repeat=1, **kw))
        return command[command.index('--out_root') + 1]
    original = output()
    assert original == output()
    assert original != output(cfg_options=['env.workers=1'])
    assert original != output(split='val')
    skill.write_text('Rule B')
    assert original != output()
    with pytest.raises(ValueError, match='positive'):
        plan(eval_args(skill, eval_repeat=0))


def test_api_environment_reaches_the_effective_clients(monkeypatch):
    monkeypatch.setattr(os, 'environ', {})
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://review.invalid/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'dummy-test-key')
    monkeypatch.setenv('YIBU_API_BASE', 'https://other.invalid/v1')
    child = environment('skillopt')
    assert child['TARGET_AZURE_OPENAI_ENDPOINT'] == 'https://review.invalid/v1'
    from skillopt.model import azure_openai as backend
    for name in tuple(vars(backend)):
        if name.isupper() or name in ('_optimizer_client', '_target_client'):
            monkeypatch.setattr(backend, name, getattr(backend, name))
    backend.configure_azure_openai(endpoint=None, auth_mode='openai_compatible')
    assert backend._role_config('optimizer')['endpoint'] == 'https://review.invalid/v1'
    assert backend._role_config('target')['endpoint'] == 'https://review.invalid/v1'
    assert backend._role_config('target')['api_key'] == 'dummy-test-key'
    for p in (ROOT / 'configs/skillopt').rglob('*.yaml'):
        assert not yaml.safe_load(p.read_text())['model']['azure_openai_endpoint']


@pytest.mark.parametrize('dataset', ['spreadsheet', 'searchqa', 'livemath'])
def test_epoch_hook_resolves_for_each_dataset(tmp_path, dataset):
    for method in ('baseline', 'skillevoreg'):
        config = yaml.safe_load((ROOT / f'configs/skillopt/{dataset}/{method}.yaml').read_text())
        hook = config['evaluation']['epoch_end_eval_script']
        assert (ROOT / hook).is_file()
    split_dir = tmp_path / 'splits'
    for split, count in [('train', 2), ('val', 1), ('test', 3)]:
        (split_dir / split).mkdir(parents=True)
        (split_dir / split / 'items.json').write_text(json.dumps([{'id': i} for i in range(count)]))
    config['env']['split_dir'] = str(split_dir)
    path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(config))
    specs = checkpoints.evaluation_specs(path)
    assert [(s['split'], s['size']) for s in specs] == [('train', 2), ('valid_seen', 1), ('valid_unseen', 3)]


def test_wrapper_does_not_blindly_skip_an_existing_summary(tmp_path, monkeypatch):
    output = tmp_path / 'epoch_01/test'; output.mkdir(parents=True)
    (output / 'eval_summary.json').write_text('{}')
    captured = []
    def fake(command, **kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(checkpoints, 'run_and_tee', fake)
    checkpoints.run_spec('epoch_01', tmp_path / 'skill.md',
        dict(name='test', split='valid_unseen', size=3, workers=1),
        tmp_path, 101, tmp_path / 'config.yaml')
    assert len(captured) == 1
    assert '--out_root' in captured[0]


def test_identity_rejects_foreign_skill_protocol_and_unmarked_data(tmp_path):
    identity = evaluation_identity('Rule A', dict(target_model='mini'), 'test')
    output = tmp_path / 'eval'
    bind_evaluation_output(output, identity)
    bind_evaluation_output(output, identity)
    with pytest.raises(ValueError, match='mismatch'):
        bind_evaluation_output(output, evaluation_identity('Rule B', dict(target_model='mini'), 'test'))
    with pytest.raises(ValueError, match='mismatch'):
        bind_evaluation_output(output, evaluation_identity('Rule A', dict(target_model='other'), 'test'))
    stale = tmp_path / 'stale'; stale.mkdir()
    (stale / 'results.jsonl').write_text('{"id":"old"}\n')
    with pytest.raises(ValueError, match='no verified identity'):
        bind_evaluation_output(stale, identity)
    train = tmp_path / 'train'; train.mkdir()
    (train / 'runtime_state.json').write_text('{}')
    with pytest.raises(ValueError, match='training directory'):
        bind_evaluation_output(train, identity)


def test_completed_summary_is_checked_and_not_overwritten(tmp_path):
    identity = evaluation_identity('Rule A', {}, 'test')
    bind_evaluation_output(tmp_path, identity)
    summary = dict(evaluation_identity=identity, n_items=1, hard=1.0, usage=dict(total_tokens=123))
    path = tmp_path / 'eval_summary.json'; path.write_text(json.dumps(summary))
    assert completed_evaluation(tmp_path, identity, 1)
    assert json.loads(path.read_text())['usage']['total_tokens'] == 123
    with pytest.raises(ValueError, match='does not match'):
        completed_evaluation(tmp_path, identity, 2)


def test_same_task_id_with_changed_input_is_a_different_identity():
    first = evaluation_identity('Rule A', {}, 'test', items=[dict(id='one', question='A')])
    second = evaluation_identity('Rule A', {}, 'test', items=[dict(id='one', question='B')])
    assert first != second


def test_endpoint_changes_but_key_rotation_does_not_reuse_foreign_protocol(monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://one.invalid/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'first-dummy-key')
    first = evaluation_identity('Rule A', dict(target_model='mini'), 'test')
    monkeypatch.setenv('OPENAI_API_KEY', 'second-dummy-key')
    assert first == evaluation_identity('Rule A', dict(target_model='mini'), 'valid_unseen')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://two.invalid/v1')
    assert first != evaluation_identity('Rule A', dict(target_model='mini'), 'test')


def test_initial_checkpoint_comes_from_its_dataset_config(tmp_path):
    initial = tmp_path / 'math_initial.md'; initial.write_text('Math initial')
    train = tmp_path / 'train'; (train / 'skills').mkdir(parents=True)
    rows = []
    for epoch in range(1, 9):
        rows.append(dict(epoch=epoch, step=epoch))
        (train / 'skills' / f'skill_v{epoch:04d}.md').write_text(str(epoch))
    (train / 'history.json').write_text(json.dumps(rows))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(dict(env=dict(name='livemathematicianbench', skill_init=str(initial)))))
    assert checkpoints.checkpoint_skills(train, config)[0] == ('initial', initial)


@pytest.fixture
def native_eval(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('native_eval_test',
        ROOT / 'benchmark/third_party/SkillOpt/scripts/eval_only.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for name in vars(module):
        if name.startswith('configure_'):
            monkeypatch.setattr(module, name, lambda *a, **kw: None)
    calls = []
    class Adapter:
        def setup(self, config): pass
        def build_eval_env(self, *args): return [{'id': 'one'}]
        def rollout(self, items, skill, output):
            calls.append((skill, output))
            return [dict(id='one', hard=1, soft=1.0)]
    monkeypatch.setattr(module, 'get_adapter', lambda config: Adapter())
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(dict(model=dict(backend='openai_chat', optimizer='gpt-5.4-mini',
        target='gpt-5.4-mini'), env=dict(name='searchqa', out_root=str(tmp_path / 'train')))))
    skill = tmp_path / 'skill.md'; skill.write_text('Rule A')
    def run(output=None):
        argv = ['eval_only.py', '--config', str(config), '--skill', str(skill), '--split', 'test']
        if output: argv += ['--out_root', str(output)]
        monkeypatch.setattr(sys, 'argv', argv)
        module.main()
    return run, calls, skill


def test_native_eval_reuses_only_identical_completed_run(native_eval, tmp_path):
    run, calls, skill = native_eval
    output = tmp_path / 'evaluation'
    run(output); run(output)
    assert len(calls) == 1
    skill.write_text('Rule B')
    with pytest.raises(ValueError, match='identity mismatch'):
        run(output)
    assert len(calls) == 1


def test_native_default_does_not_use_training_output(native_eval, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run, calls, _ = native_eval
    train = tmp_path / 'train'; train.mkdir()
    (train / 'runtime_state.json').write_text('unchanged')
    run(); run()
    assert len(calls) == 2 and calls[0][1] != calls[1][1]
    assert all(Path(output) != train for _, output in calls)
    assert (train / 'runtime_state.json').read_text() == 'unchanged'


def test_native_eval_refuses_unmarked_cached_answers(native_eval, tmp_path):
    run, calls, _ = native_eval
    output = tmp_path / 'evaluation'; output.mkdir()
    (output / 'results.jsonl').write_text('{"id":"one","hard":1}\n')
    with pytest.raises(ValueError, match='no verified identity'):
        run(output)
    assert not calls
