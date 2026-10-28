from pathlib import Path
import pytest
from continual_regularization.baseline import MathBaseline
from continual_regularization.agent import MathRegularization
from continual_regularization.checkpoint import protocol


@pytest.mark.parametrize('regularization', [False, True])
def test_independent_tasks_keep_same_task_maintenance_and_checkpoints(tmp_path, monkeypatch, regularization):
    monkeypatch.setenv('OPENAI_API_KEY', 'dummy-test-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://yibuapi.com/v1')
    kwargs = dict(logs_dir=tmp_path, model_name='openai/gpt-5.4-mini',
                  reasoning_effort='medium', task_tests_dir=tmp_path,
                  compact_every_n=1)
    if regularization:
        kwargs['regularization_config_path'] = Path(__file__).resolve().parents[1] / 'configs/continual_math_regularization_policy.json'
    agent = (MathRegularization if regularization else MathBaseline)(**kwargs)
    commands = agent.create_run_agent_commands('first\n---TASK_BREAK---\nsecond\n---TASK_BREAK---\nthird')
    phases = agent._display_plan
    solves = [c.command for i, c in enumerate(commands, 1) if phases[i][1] == '做题']
    updates = [c.command for i, c in enumerate(commands, 1) if phases[i][1] == '技能更新']
    assert len(solves) == len(updates) == 3
    # Task 1 continues the intro-only session. No prior task history exists.
    assert 'codex exec resume --last' in solves[0]
    for command in solves[1:]:
        assert 'codex exec' in command and 'codex exec resume' not in command
        assert '/root/task_memory.md' in command
    assert all('codex exec resume --last' in command for command in updates)
    assert len(agent._checkpoint_boundaries) == 3
    assert agent._checkpoint_total_steps == 3
    assert [phases[i][0] for i in sorted(agent._checkpoint_boundaries)] == [1, 2, 3]


def test_session_mode_is_part_of_resume_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'dummy-test-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://yibuapi.com/v1')
    agent = MathBaseline(logs_dir=tmp_path, model_name='openai/gpt-5.4-mini',
                         task_tests_dir=tmp_path, compact_every_n=None)
    before = protocol(agent, 'task')
    agent._compact_every_n = 1
    assert protocol(agent, 'task') != before
