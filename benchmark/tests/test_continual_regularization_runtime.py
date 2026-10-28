import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from continual_regularization import container_io
from continual_regularization.ports import Retrieval
from continual_regularization.transactions import Window, apply_repair


def test_mirrored_commit_and_concurrent_change(tmp_path, monkeypatch):
    import base64
    roots = [str(tmp_path / 'a'), str(tmp_path / 'b')]
    monkeypatch.setattr(container_io, 'ROOTS', roots)
    for root in roots:
        p = Path(root) / 'skill/SKILL.md'
        p.parent.mkdir(parents=True)
        p.write_text('old')
    snapshots = {r: container_io.read(r) for r in roots}
    request = dict(op='write', expected=snapshots, changes={'skill/SKILL.md': 'new'})
    monkeypatch.setenv('CONTINUAL_IO', base64.b64encode(json.dumps(request).encode()).decode())
    container_io.main()
    assert all((Path(r) / 'skill/SKILL.md').read_text() == 'new' for r in roots)
    with pytest.raises(ValueError, match='concurrently'):
        container_io.main()


def test_rewrite_cannot_touch_history_in_same_file():
    with pytest.raises(ValueError, match='Only MERGE'):
        apply_repair({'s/SKILL.md': 'old new'},
                     {'operation': 'REWRITE', 'changes': {'history_0': 'bad'}},
                     source='s/SKILL.md', history=set(), mode='modify',
                     windows={'history_0': Window('s/SKILL.md', 0, 3, 'old')})


def test_large_payload_uses_file(tmp_path, monkeypatch):
    monkeypatch.setattr(container_io, 'ROOTS', [str(tmp_path / 'library')])
    request = tmp_path / 'payload.json'
    request.write_text(json.dumps(dict(op='write', expected={str(tmp_path / 'library'): {}},
                                      changes={'s/SKILL.md': 'x' * 200000})))
    monkeypatch.setenv('CONTINUAL_IO_PATH', str(request))
    container_io.main()
    assert (tmp_path / 'library/s/SKILL.md').stat().st_size == 200000
    assert not request.exists()


def test_regularization_class_does_not_change_baseline_prompts(tmp_path, monkeypatch):
    from continual_regularization.agent import MathRegularization
    from continual_regularization.baseline import MathBaseline
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://yibuapi.com/v1')
    params = dict(logs_dir=tmp_path, model_name='openai/gpt-5.4-mini', task_tests_dir=tmp_path)
    baseline = MathBaseline(**params)
    regularization = MathRegularization(**params, regularization_config_path='benchmark/configs/continual_math_regularization_policy.json')
    assert baseline._build_judge_feedback_turn(1, 100) == regularization._build_judge_feedback_turn(1, 100)
    assert [c.command for c in baseline.create_run_agent_commands('task')] == [c.command for c in regularization.create_run_agent_commands('task')]


def test_source_pool_contains_only_passed_training_task(tmp_path):
    import asyncio
    from continual_regularization.agent import RegularizationEnvironment
    class Environment:
        async def exec(self, **kwargs):
            return SimpleNamespace(return_code=0, stdout=kwargs['command'], stderr='')
    agent = SimpleNamespace(_display_plan={1: (1, '评分'), 2: (2, '评分')},
                            _task_tests_dir=tmp_path,
                            regularization_config={},
                            _grading_env_for_task=lambda i: {'HARBOR_GRADING_SPEC_JSON': json.dumps({'task_id': i})})
    wrapper = RegularizationEnvironment(Environment(), agent, 'first\n---TASK_BREAK---\nsecond')
    asyncio.run(wrapper.exec(command='CONTINUAL_SCORE {"task_id":1,"score":0}'))
    assert wrapper.successful_sources == []
    wrapper.rollout_trace = 'only second task trace'
    asyncio.run(wrapper.exec(command='CONTINUAL_SCORE {"task_id":2,"score":1}'))
    assert len(wrapper.successful_sources) == 1
    assert wrapper.successful_sources[0]['instruction'] == 'second'
    assert wrapper.successful_sources[0]['trajectory'] == 'only second task trace'


@pytest.mark.parametrize('instruction', ['Solve the task.', 'Solve by checking edge cases.'])
def test_task_text_does_not_bypass_regularization_plan(tmp_path, monkeypatch, instruction):
    import asyncio
    from continual_regularization import agent as runtime

    class Environment:
        async def exec(self, **kwargs):
            stdout = ('CONTINUAL_SCORE {"task_id":1,"score":1}'
                      if kwargs['command'] == 'judge' else 'training trace')
            return SimpleNamespace(return_code=0, stdout=stdout, stderr='')

    agent = SimpleNamespace(
        _display_plan={1: (1, '做题'), 2: (1, '评分')},
        logs_dir=tmp_path, _task_tests_dir=tmp_path,
        regularization_config={'seed': 101, 'dropout': {}},
        _grading_env_for_task=lambda i: {'HARBOR_GRADING_SPEC_JSON': '{}'},
    )
    wrapper = runtime.RegularizationEnvironment(Environment(), agent, instruction)
    files = {'s/SKILL.md': 'Existing skill.'}
    commits, masks = [], []

    async def read_snapshot(request):
        assert request == {'op': 'read'}
        return {'/root/.agents/skills': files}

    async def record_commit(snapshots, desired):
        commits.append(dict(desired))

    def record_mask(old, **kwargs):
        masks.append(dict(old))
        return dict(old), {}

    wrapper.io = read_snapshot
    wrapper.commit = record_commit
    monkeypatch.setattr(runtime, 'mask_library', record_mask)
    asyncio.run(wrapper.exec(command='codex exec ' + instruction))
    assert wrapper.index == 1
    assert masks == commits == [files]
    assert wrapper.rollout_trace == 'training trace'
    asyncio.run(wrapper.exec(command='judge'))
    assert wrapper.index == 2
    assert len(wrapper.successful_sources) == 1
    assert wrapper.successful_sources[0]['instruction'] == instruction
    assert wrapper.successful_sources[0]['trajectory'] == 'training trace'


def test_preflight_without_plan_does_not_advance_regularization(tmp_path):
    import asyncio
    import shlex
    from continual_regularization.agent import RegularizationEnvironment
    from continual_regularization.baseline import API_PREFLIGHT

    calls = []

    class Environment:
        async def exec(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(return_code=0, stdout='OK', stderr='')

    wrapper = RegularizationEnvironment(
        Environment(), SimpleNamespace(_display_plan={}), 'task')
    command = 'python3 -c ' + shlex.quote(API_PREFLIGHT)
    result = asyncio.run(wrapper.exec(command=command))
    assert result.stdout == 'OK'
    assert calls == [{'command': command}]
    assert wrapper.index == 0
    assert wrapper.successful_sources == []
    assert not hasattr(wrapper, 'records')
