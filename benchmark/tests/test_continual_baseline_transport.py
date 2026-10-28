import asyncio
from types import SimpleNamespace
import pytest
from continual_regularization.baseline import CheckedEnvironment, MathBaseline


def test_nonzero_command_stops(tmp_path):
    class Environment:
        async def exec(self, **kwargs):
            return SimpleNamespace(return_code=1, stdout="", stderr="transport failed")
    with pytest.raises(RuntimeError, match="stopping"):
        asyncio.run(CheckedEnvironment(Environment(), tmp_path).exec(command="false"))
    assert (tmp_path / "infrastructure_failure.json").is_file()


def test_judge_env_not_command_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-in-command")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://yibuapi.com/v1")
    agent = MathBaseline(logs_dir=tmp_path, model_name="openai/gpt-5.4-mini",
                         judge_model="openai/gpt-5.4-mini", reasoning_effort="medium",
                         task_tests_dir=tmp_path)
    commands = agent.create_run_agent_commands("Write /app/task_001_output.json")
    assert all("test-secret-never-in-command" not in c.command for c in commands)
    judge = next(c for c in commands if c.sync_task_tests_dir)
    assert "judge_error" in judge.command
    assert judge.env["OPENAI_API_BASE"] == "https://yibuapi.com/v1"
    assert all(c.command.startswith("bash -o pipefail") for c in commands)
    assert all(". ~/.nvm/nvm.sh;" not in c.command for c in commands)
    assert all("unset HTTP_PROXY HTTPS_PROXY ALL_PROXY" in c.command for c in commands)
    assert all(c.env["NO_PROXY"] == "*" and c.env["no_proxy"] == "*" for c in commands)
    cli = [c for c in commands if "codex exec" in c.command]
    assert cli
    assert all("model_provider=" in c.command and "continual_yibu.base_url=" in c.command for c in cli)
    assert all("https://yibuapi.com/v1" in c.command for c in cli)


def test_per_task_progress_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://yibuapi.com/v1")
    agent = MathBaseline(logs_dir=tmp_path, model_name="openai/gpt-5.4-mini",
                         task_tests_dir=tmp_path)
    agent._display_plan = {}
    agent.create_run_agent_commands("task one\n---TASK_BREAK---\ntask two")
    labels = list(agent._display_plan.values())
    for n in (1, 2):
        assert (n, "做题") in labels
        assert (n, "评分") in labels
        assert (n, "技能更新") in labels
        assert (n, "确认技能写入") in labels
    # Verification commands have identical text: progress must use execution order,
    # not command text alone, for repeated commands.
