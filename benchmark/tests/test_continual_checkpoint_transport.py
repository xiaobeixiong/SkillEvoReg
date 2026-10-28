import asyncio
from types import SimpleNamespace

from harbor.environments.base import ExecResult

from continual_regularization.checkpoint import CheckpointEnvironment


def encrypted_failure():
    return ExecResult(
        return_code=1,
        stdout='{"code":"invalid_encrypted_content",'
               '"message":"Encrypted content could not be decrypted or parsed"}',
        stderr='',
    )


class RawEnvironment:
    def __init__(self):
        self.results = [
            encrypted_failure(),
            ExecResult(return_code=0, stdout='', stderr=''),
            ExecResult(return_code=0, stdout='clean retry trajectory', stderr=''),
        ]
        self.calls = []

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


class RegularizationEnvironment:
    def __init__(self, raw):
        self.environment = raw
        self.calls = 0
        self.rollout_trace = ''

    async def exec(self, **kwargs):
        self.calls += 1
        return await self.environment.exec(**kwargs)


def test_training_solve_retry_does_not_advance_regularization_or_checkpoint(tmp_path):
    raw = RawEnvironment()
    regularization = RegularizationEnvironment(raw)
    agent = SimpleNamespace(
        _display_plan={1: (4, '做题')},
        _checkpoint_boundaries=set(),
        enable_checkpoints=False,
        logs_dir=tmp_path,
    )
    wrapped = CheckpointEnvironment.__new__(CheckpointEnvironment)
    wrapped.environment = regularization
    wrapped.agent = agent
    wrapped.instruction = ''
    wrapped.index = 0
    wrapped.receipts = {}
    wrapped.resume = None

    command = "codex exec resume --last -- '/app/task_004_output.json'"
    result = asyncio.run(wrapped.exec(command=command, timeout_sec=600))

    assert result.return_code == 0
    assert wrapped.index == 1
    assert regularization.calls == 1
    assert len(raw.calls) == 3
    assert raw.calls[1]['command'] == 'rm -f /app/task_004_output.json'
    assert 'resume --last' not in raw.calls[2]['command']
    assert wrapped.receipts['1']['return_code'] == 0
    assert regularization.rollout_trace == 'clean retry trajectory'


def test_training_non_transport_failure_is_not_retried(tmp_path):
    raw = RawEnvironment()
    raw.results = [ExecResult(return_code=1, stdout='', stderr='ordinary failure')]
    regularization = RegularizationEnvironment(raw)
    agent = SimpleNamespace(
        _display_plan={1: (4, '做题')}, _checkpoint_boundaries=set(),
        enable_checkpoints=False, logs_dir=tmp_path,
    )
    wrapped = CheckpointEnvironment.__new__(CheckpointEnvironment)
    wrapped.environment = regularization
    wrapped.agent = agent
    wrapped.instruction = ''
    wrapped.index = 0
    wrapped.receipts = {}
    wrapped.resume = None

    result = asyncio.run(wrapped.exec(
        command="codex exec -- '/app/task_004_output.json'", timeout_sec=600))
    assert result.return_code == 1
    assert len(raw.calls) == 1
    assert wrapped.receipts['1']['return_code'] == 1


def test_training_stream_disconnect_is_retryable():
    result = ExecResult(
        return_code=1,
        stdout='Your requests exceeded rate limit; stream disconnected before completion',
        stderr='',
    )
    assert CheckpointEnvironment._retryable_transport_failure(result)
    assert CheckpointEnvironment._transport_retry_delay(result, 1) == 30
    assert CheckpointEnvironment._transport_retry_delay(result, 2) == 90
    assert CheckpointEnvironment._transport_retry_delay(result, 3) == 180
    assert CheckpointEnvironment._transport_retry_delay(encrypted_failure(), 1) == 0
