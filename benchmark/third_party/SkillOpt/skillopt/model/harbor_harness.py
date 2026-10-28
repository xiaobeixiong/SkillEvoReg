"""Pooled Docker Codex execution for SkillOpt target rollouts.

The historical ``harbor_codex_exec`` name is retained for config compatibility,
but execution is lighter than a Harbor trial: one persistent agent-runtime
container is shared by this Python process and every sample starts a fresh,
isolated ``codex exec`` session inside it.
"""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any


_START_LOCK = threading.Lock()
_POOL: "_DockerCodexPool | None" = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _docker_error(proc: subprocess.CompletedProcess[str]) -> str:
    return (proc.stderr or proc.stdout or "unknown Docker error").strip()


class _DockerCodexPool:
    def __init__(self, config: dict[str, Any]) -> None:
        self.workers = max(1, int(config.get("workers", 1)))
        self.image = str(
            config.get("image") or "agent-overfit-skillopt-runtime:latest"
        ).strip()
        self.infrastructure_retries = max(0, int(config.get("infrastructure_retries", 2)))
        self.repo_root = _repo_root()
        self.runtime_root = self.repo_root / "benchmark" / "outputs" / ".skillopt_codex_pool"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.container_name = f"skillopt-codex-pool-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._semaphore = threading.BoundedSemaphore(self.workers)
        self._stopped = False
        self._start_container()

    def _start_container(self) -> None:
        env = os.environ.copy()
        base_url = env.get("YIBU_API_BASE") or env.get("OPENAI_BASE_URL")
        api_key = env.get("YIBU_API_KEY") or env.get("OPENAI_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError("YIBU_API_BASE/YIBU_API_KEY are required for pooled Codex")
        # ``--env NAME`` copies the value from this subprocess environment and
        # avoids placing the credential itself in the command line.
        env["OPENAI_BASE_URL"] = base_url
        env["OPENAI_API_KEY"] = api_key
        env.pop("AZURE_OPENAI_API_VERSION", None)
        cmd = [
            "docker", "run", "--detach", "--rm", "--name", self.container_name,
            "--label", "agent-overfit.role=skillopt-codex-pool",
            "--label", f"agent-overfit.owner-pid={os.getpid()}",
            "--cap-add", "SYS_ADMIN",
            "--mount", f"type=bind,source={self.repo_root},target={self.repo_root}",
            "--env", "OPENAI_BASE_URL", "--env", "OPENAI_API_KEY",
            "--workdir", str(self.repo_root), self.image,
            "sh", "-c", "while :; do sleep 3600; done",
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"failed to start pooled Codex container: {_docker_error(proc)}")
        probe = subprocess.run(
            ["docker", "exec", self.container_name, "sh", "-c", "command -v codex"],
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            self.stop()
            raise RuntimeError(f"Codex is unavailable in {self.image}: {_docker_error(probe)}")
        print(
            f"[Codex pool] container={self.container_name} workers={self.workers} image={self.image}",
            flush=True,
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        subprocess.run(
            ["docker", "rm", "-f", self.container_name], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False,
        )

    def _kill_session(self, pid_path: Path) -> None:
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return
        subprocess.run(
            ["docker", "exec", self.container_name, "kill", "-TERM", str(pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        time.sleep(0.5)
        subprocess.run(
            ["docker", "exec", self.container_name, "kill", "-KILL", str(pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )

    def _one_attempt(
        self, *, work_dir: str, prompt: str, model: str, timeout: int, request_id: str
    ) -> tuple[str, str]:
        workspace = Path(work_dir).resolve()
        try:
            workspace.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f"pooled Codex workspace must be inside repository: {workspace}") from exc

        session_root = self.runtime_root / request_id
        home = session_root / "home"
        codex_home = session_root / "codex-home"
        tmp_dir = session_root / "tmp"
        for path in (home, codex_home, tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        last_message_path = workspace / f".codex_last_message_{request_id}.txt"
        pid_path = session_root / "codex.pid"
        reasoning_effort = str(os.environ.get("CODEX_EXEC_REASONING_EFFORT") or "high").strip()
        # The prior Harbor Codex adapter used high effort unconditionally.
        # Preserve that behavior when inherited legacy configs say "none".
        if not reasoning_effort or reasoning_effort == "none":
            reasoning_effort = "high"
        base_url = os.environ.get("YIBU_API_BASE") or os.environ.get("OPENAI_BASE_URL")

        codex_args = [
            "codex", "exec", "--skip-git-repo-check", "--color", "never",
            "--dangerously-bypass-approvals-and-sandbox", "-C", str(workspace),
            "-m", model, "-c", 'model_provider="yibu"',
            "-c", 'model_providers.yibu.name="Yibu"',
            "-c", f'model_providers.yibu.base_url="{base_url}"',
            "-c", 'model_providers.yibu.env_key="OPENAI_API_KEY"',
            "-c", 'model_providers.yibu.wire_api="responses"',
            "-c", 'model_verbosity="medium"',
        ]
        if reasoning_effort and reasoning_effort != "none":
            codex_args.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        codex_args.extend(["--output-last-message", str(last_message_path), prompt])
        session_wrapper = (
            self.repo_root / "benchmark" / "scripts"
            / "skillopt_codex_isolated_session.sh"
        )
        if not session_wrapper.is_file():
            raise RuntimeError(f"missing isolated-session wrapper: {session_wrapper}")
        # Each exec gets its own mount namespace: workspace/session/tmp remain
        # writable while the rest of the shared container is read-only.
        cmd = [
            "docker", "exec", "--workdir", str(workspace),
            "--env", f"HOME={home}", "--env", f"CODEX_HOME={codex_home}",
            "--env", f"TMPDIR={tmp_dir}", self.container_name,
            "sh", str(session_wrapper), str(self.repo_root), str(workspace),
            str(session_root), str(tmp_dir), str(pid_path), "--", *codex_args,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1, int(timeout)))
        except subprocess.TimeoutExpired as exc:
            self._kill_session(pid_path)
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            raise TimeoutError(
                f"pooled Codex session timed out after {timeout}s: {(stderr or stdout)[-2000:]}"
            ) from exc

        raw = proc.stdout or ""
        if proc.stderr:
            raw = f"{raw}\n[stderr]\n{proc.stderr}" if raw else proc.stderr
        response = ""
        if last_message_path.is_file():
            response = last_message_path.read_text(encoding="utf-8")
            last_message_path.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"pooled codex exec failed with exit code {proc.returncode}: {raw[-4000:]}"
            )
        if not response.strip():
            raise RuntimeError("pooled codex exec completed without a final message")
        return response, raw

    def run(self, *, work_dir: str, prompt: str, model: str, timeout: int) -> tuple[str, str]:
        errors: list[str] = []
        with self._semaphore:
            for attempt in range(self.infrastructure_retries + 1):
                request_id = f"{uuid.uuid4().hex}-a{attempt + 1}"
                try:
                    return self._one_attempt(
                        work_dir=work_dir, prompt=prompt, model=model,
                        timeout=timeout, request_id=request_id,
                    )
                except (RuntimeError, TimeoutError) as exc:
                    errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                    if attempt < self.infrastructure_retries:
                        time.sleep(min(2 ** attempt, 8))
                finally:
                    shutil.rmtree(self.runtime_root / request_id, ignore_errors=True)
            raise RuntimeError("pooled Codex execution failed; " + " | ".join(errors))


def _get_pool(config: dict[str, Any]) -> _DockerCodexPool:
    global _POOL
    requested_workers = max(1, int(config.get("workers", 1)))
    with _START_LOCK:
        if _POOL is None or _POOL._stopped:
            _POOL = _DockerCodexPool(config)
            atexit.register(_POOL.stop)
        elif _POOL.workers != requested_workers:
            raise RuntimeError(
                f"Codex pool already has workers={_POOL.workers}; cannot change to "
                f"{requested_workers} in the same process"
            )
        return _POOL


def stop_harbor_codex_pool() -> None:
    """Stop this process's shared container (mainly for tests and cleanup)."""
    global _POOL
    with _START_LOCK:
        if _POOL is not None:
            _POOL.stop()
            _POOL = None


def run_harbor_codex_exec(
    *, work_dir: str, prompt: str, model: str, timeout: int, config: dict[str, Any]
) -> tuple[str, str]:
    """Run one new Codex session in the process-wide pooled container."""
    return _get_pool(config).run(work_dir=work_dir, prompt=prompt, model=model, timeout=timeout)
