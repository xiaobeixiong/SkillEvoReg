"""Offline checks for Codex timeout diagnostics; no CLI or model is called."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark/third_party/SkillOpt'))
from skillopt.model import codex_harness


@pytest.fixture
def codex_workdir(tmp_path, monkeypatch):
    workdir = tmp_path / 'work'
    workdir.mkdir()
    monkeypatch.setattr(codex_harness, 'get_codex_exec_config', lambda: {
        'path': 'codex', 'sandbox': 'workspace-write', 'profile': '',
        'reasoning_effort': 'medium', 'full_auto': False,
        'approval_policy': 'never',
    })
    return workdir


@pytest.mark.parametrize('existing_trace', [False, True])
@pytest.mark.parametrize('stdout,stderr,expected', [
    (b'partial output', b'', 'partial output'),
    (b'', b'partial error', 'partial error'),
    (b'partial output', b'partial error', 'partial output\n[stderr]\npartial error'),
    ('partial output', 'partial error', 'partial output\n[stderr]\npartial error'),
    (b'partial output', 'partial error', 'partial output\n[stderr]\npartial error'),
    ('partial output', b'partial error', 'partial output\n[stderr]\npartial error'),
    (None, None, ''),
    (b'output:\xff', b'error:\xfe', 'output:\ufffd\n[stderr]\nerror:\ufffd'),
])
def test_timeout_writes_text_and_preserves_exception(
    codex_workdir, monkeypatch, existing_trace, stdout, stderr, expected,
):
    raw_path = codex_workdir.parent / 'codex_raw.txt'
    if existing_trace:
        raw_path.write_text('previous trace', encoding='utf-8')
    error = subprocess.TimeoutExpired(['codex'], 1, output=stdout, stderr=stderr)
    calls = []

    def timeout(*args, **kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(codex_harness.subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        codex_harness._run_codex_cli_exec(
            work_dir=str(codex_workdir), prompt='offline fixture',
            model='gpt-5.4-mini', timeout=1,
        )

    assert caught.value is error
    assert error.stdout == stdout and error.stderr == stderr
    assert len(calls) == 1 and calls[0]['text'] is True
    prefix = 'previous trace\n\n===== TURN BREAK =====\n\n' if existing_trace else ''
    assert raw_path.read_text(encoding='utf-8') == prefix + expected
    assert (codex_workdir.parent / 'codex_trace_summary.txt').is_file()


def test_completed_cli_behavior_is_unchanged(codex_workdir, monkeypatch):
    calls = []

    def completed(command, **kwargs):
        calls.append(command)
        (codex_workdir / 'codex_last_message.txt').write_text('OK', encoding='utf-8')
        return subprocess.CompletedProcess(command, 0, stdout='completed trace', stderr='')

    monkeypatch.setattr(codex_harness.subprocess, 'run', completed)
    response, raw = codex_harness._run_codex_cli_exec(
        work_dir=str(codex_workdir), prompt='offline fixture',
        model='gpt-5.4-mini', timeout=1,
    )
    assert len(calls) == 1
    assert response == 'OK'
    assert raw == 'completed trace'
