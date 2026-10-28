import asyncio
import os
import types
from pathlib import Path

from harbor.environments.base import ExecResult

from continual_holdout.stateless_agent import (
    FrozenCheckedEnvironment,
    MathStatelessHeldout,
    ScienceStatelessHeldout,
)


def test_strict_plan_has_no_learning_or_memory(tmp_path):
    os.environ.setdefault('OPENAI_BASE_URL', 'https://example.invalid/v1')
    task = Path('benchmark/runtime/continualskillbench/tasks/math-100-holdout50')
    agent = MathStatelessHeldout(
        logs_dir=tmp_path, model_name='openai/gpt-5.4-mini',
        task_tests_dir=task/'tests', tasks_spec_path=task/'environment/tasks_spec.json',
        reasoning_effort='medium', enable_judge_feedback=True,
        judge_model='openai/gpt-5.4-mini', compact_every_n=1,
        frozen_library=Path('benchmark/runtime/continualskillbench/frozen_libraries/empty'))
    commands = agent.create_run_agent_commands((task/'instruction.md').read_text())
    assert len(commands) == 103
    assert sum(phase == '做题' for _, phase in agent._display_plan.values()) == 50
    assert sum(phase == '评分' for _, phase in agent._display_plan.values()) == 50
    text = '\n'.join(c.command for c in commands)
    assert 'Based on this feedback, update the skill library' not in text
    assert 'python3 /root/append_task_memory.py --task-num' not in text
    assert 'There is no feedback or skill-maintenance turn' in text
    assert 'Reply exactly READY. Do not call tools' in text


def test_skill_priority_changes_only_solve_instruction(tmp_path):
    os.environ.setdefault('OPENAI_BASE_URL', 'https://example.invalid/v1')
    task = Path('benchmark/runtime/continualskillbench/tasks/math-100-holdout50')
    def make(enabled):
        agent = MathStatelessHeldout(
            logs_dir=tmp_path / str(enabled), model_name='openai/gpt-5.4-mini',
            task_tests_dir=task/'tests', tasks_spec_path=task/'environment/tasks_spec.json',
            reasoning_effort='medium', enable_judge_feedback=True,
            judge_model='openai/gpt-5.4-mini', compact_every_n=1,
            prefer_skills=enabled)
        return agent, agent.create_run_agent_commands((task/'instruction.md').read_text())
    old, old_commands = make(False)
    new, new_commands = make(True)
    assert old._display_plan == new._display_plan
    assert len(old_commands) == len(new_commands) == 103
    for i, (a, b) in enumerate(zip(old_commands, new_commands), 1):
        if new._display_plan[i][1] == '做题':
            assert b.command == a.command.replace(
                'Learned skills may be read as reference.',
                'Prefer skills found under $HOME/.agents/skills/. Inspect them before inventing a new approach.')
        else:
            assert a.command == b.command


class _ScriptedEnvironment:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


def _encrypted_failure():
    return ExecResult(
        return_code=1,
        stdout='{"type":"error","code":"invalid_encrypted_content",'
               '"message":"Encrypted content could not be decrypted or parsed"}',
        stderr='',
    )


def test_encrypted_transport_failure_retries_same_task_in_fresh_session(tmp_path):
    environment = _ScriptedEnvironment([
        _encrypted_failure(),
        ExecResult(return_code=0, stdout='', stderr=''),  # partial-output cleanup
        ExecResult(return_code=0, stdout='retry succeeded', stderr=''),
    ])
    wrapped = FrozenCheckedEnvironment(
        environment, tmp_path, plan={1: (52, '做题')},
        agent=object(), expected_digest='frozen',
    )
    checks = []

    async def check(_self, number, boundary):
        checks.append((number, boundary))

    wrapped._check = types.MethodType(check, wrapped)
    command = "codex exec resume --last -- '/app/task_052_output.json'"
    result = asyncio.run(wrapped.exec(command=command, timeout_sec=600))

    assert result.return_code == 0
    assert wrapped.index == 1
    assert checks == [(52, 'before_solve'), (52, 'after_solve')]
    assert environment.calls[1]['command'] == 'rm -f /app/task_052_output.json'
    assert 'codex exec resume --last' not in environment.calls[2]['command']
    assert 'codex exec' in environment.calls[2]['command']
    records = [__import__('json').loads(line) for line in
               (tmp_path / 'frozen_transport_retries.jsonl').read_text().splitlines()]
    assert records == [{
        'task': 52,
        'attempt': 1,
        'fresh_session': True,
        'backoff_sec': 0,
        'removed_partial_output': '/app/task_052_output.json',
        'return_code': 0,
        'retryable_transport_failure': False,
    }]


def test_encrypted_initialization_failure_retries_in_fresh_session(tmp_path):
    environment = _ScriptedEnvironment([
        _encrypted_failure(),
        ExecResult(return_code=0, stdout='', stderr=''),  # no task output to clean
        ExecResult(return_code=0, stdout='initialization retry succeeded', stderr=''),
    ])
    wrapped = FrozenCheckedEnvironment(
        environment, tmp_path, plan={1: (0, '初始化会话')},
        agent=object(), expected_digest='frozen',
    )
    result = asyncio.run(wrapped.exec(command="codex exec -- 'initialize'", timeout_sec=600))

    assert result.return_code == 0
    assert wrapped.index == 1
    assert environment.calls[1]['command'] == 'true'
    assert environment.calls[2]['command'] == "codex exec -- 'initialize'"
    records = [__import__('json').loads(line) for line in
               (tmp_path / 'frozen_transport_retries.jsonl').read_text().splitlines()]
    assert records[0]['task'] == 0
    assert records[0]['removed_partial_output'] is None
    assert records[0]['fresh_session'] is True


def test_invalid_judge_response_is_resampled_without_rerunning_task(tmp_path, monkeypatch):
    judge_failure = ExecResult(
        return_code=1,
        stdout='Judge score for task 91: 0.0\nAssertionError: Judge API failure: batch not scoreable',
        stderr='',
    )
    environment = _ScriptedEnvironment([
        judge_failure,
        ExecResult(return_code=0, stdout='CONTINUAL_SCORE {"task_id":91,"score":0.8}', stderr=''),
    ])
    wrapped = FrozenCheckedEnvironment(
        environment, tmp_path, plan={1: (91, '评分')},
        agent=object(), expected_digest='frozen',
    )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, 'sleep', no_sleep)
    result = asyncio.run(wrapped.exec(command='python3 /root/judge_subtask.py --task-id 91'))

    assert result.return_code == 0
    assert len(environment.calls) == 2
    assert not any('codex exec' in call['command'] for call in environment.calls)
    records = [__import__('json').loads(line) for line in
               (tmp_path / 'frozen_judge_retries.jsonl').read_text().splitlines()]
    assert records == [{
        'task': 91, 'attempt': 1, 'backoff_sec': 5, 'return_code': 0,
        'retryable_judge_failure': False,
    }]


def test_non_encrypted_failure_is_not_retried(tmp_path):
    environment = _ScriptedEnvironment([
        ExecResult(return_code=1, stdout='', stderr='ordinary solver failure'),
    ])
    wrapped = FrozenCheckedEnvironment(
        environment, tmp_path, plan={1: (82, '做题')},
        agent=object(), expected_digest='frozen',
    )

    async def check(_self, number, boundary):
        return None

    wrapped._check = types.MethodType(check, wrapped)
    try:
        asyncio.run(wrapped.exec(
            command="codex exec -- '/app/task_082_output.json'", timeout_sec=600))
    except RuntimeError as error:
        assert 'stopping before further skill updates' in str(error)
    else:
        raise AssertionError('ordinary failures must remain fail-closed')
    assert len(environment.calls) == 1
    assert not (tmp_path / 'frozen_transport_retries.jsonl').exists()


def test_science_uses_the_same_transport_retry_regularization():
    assert issubclass(ScienceStatelessHeldout, MathStatelessHeldout)
    assert ScienceStatelessHeldout.run is MathStatelessHeldout.run


def test_stream_disconnect_and_rate_limit_are_retryable():
    result = ExecResult(
        return_code=1,
        stdout='Reconnecting: requests exceeded rate limit; '
               'stream disconnected before completion',
        stderr='',
    )
    assert FrozenCheckedEnvironment._retryable_transport_failure(result)
    assert FrozenCheckedEnvironment._transport_retry_delay(result, 1) == 30
    assert FrozenCheckedEnvironment._transport_retry_delay(result, 2) == 90
    assert FrozenCheckedEnvironment._transport_retry_delay(result, 3) == 180
    assert FrozenCheckedEnvironment._transport_retry_delay(result, 4) == 300
    assert FrozenCheckedEnvironment._transport_retry_delay(result, 8) == 300
    assert FrozenCheckedEnvironment._transport_retry_delay(_encrypted_failure(), 1) == 0
