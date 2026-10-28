"""Persist subprocess logs while also streaming progress to the terminal."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Mapping, TextIO


def run_and_tee(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log: TextIO,
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # Give each streamed child its own process group.  This lets Ctrl+C
        # terminate the complete nested evaluation tree instead of orphaning
        # run_skillopt.py and its Harbor workers when the outer driver exits.
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    return subprocess.CompletedProcess(command, returncode)
