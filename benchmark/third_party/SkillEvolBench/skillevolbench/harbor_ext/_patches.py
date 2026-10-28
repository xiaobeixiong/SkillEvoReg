"""Runtime patches for Harbor 0.6+.

Harbor 0.6 caches ``Task.instruction`` as a plain ``str`` attribute set in
``Task.__init__`` (``models/task/task.py:56``):

    self.instruction = strip_canary(self.paths.instruction_path.read_text())

This read happens BEFORE the ``TrialEvent.START`` hook fires. Our START
hook (``RuntimeBuilder.build()`` -> ``PromptBuilder.write_*``) overwrites
``instruction.md`` on disk with a retrieval-augmented version, but
``Trial._execute_agent`` later passes the cached attribute to
``agent.run``:

    self._agent.run(instruction=self._task.instruction, ...)

So the on-disk update is silently ignored. Every Path-A / Path-B baseline
(``raw_trajectory_rag``, ``curated_*``, ``history_context_control``,
``selfgen_*``) was effectively running zero-shot -- the agents never saw
any of the retrieved skills / past-trajectory / history blocks the hook
believed it had injected. ``injection-context.json`` was a wishful audit
log.

This module monkey-patches the single-step agent entrypoint to re-read
``instruction.md`` from disk just before delegating to Harbor, so the
START-hook write actually reaches the agent. Harbor 0.6 used
``Trial._execute_agent``; Harbor 0.22 uses ``SingleStepTrial._run_agent``.

Idempotent. Apply once per process (Harbor instantiates many ``Trial``
objects but they all share the same class). Call from
``LifelongRunner.run()`` right after Harbor is first imported.
"""

from __future__ import annotations

import logging
import re

_LOG = logging.getLogger(__name__)
_PATCHED: bool = False


def _invalid_response_item_id(output: str) -> bool:
    return bool(re.search(r"Invalid.{0,12}input\[\d+\]\.id", output, re.S)
                and re.search(r"Expected an ID that begins with.{0,10}(?:rs|msg)", output, re.S))


def _unavailable_model_channel(output: str) -> bool:
    return '503 Service Unavailable' in output and 'No available channel for model' in output


def _refresh_instruction(trial, strip_canary) -> None:
    """Refresh Harbor's cached instruction across old/new Task attributes."""
    task = getattr(trial, "task", None) or getattr(trial, "_task", None)
    if task is None:
        raise RuntimeError(
            "Harbor instruction integrity failure: trial exposes neither "
            "task nor _task"
        )
    paths = getattr(task, "paths", None)
    instr_path = getattr(paths, "instruction_path", None)
    if instr_path is None or not instr_path.is_file():
        raise RuntimeError(
            "Harbor instruction integrity failure: injected instruction.md "
            f"is unavailable at {instr_path!s}"
        )
    refreshed = strip_canary(instr_path.read_text())
    if not refreshed.strip():
        raise RuntimeError(
            "Harbor instruction integrity failure: injected instruction.md "
            f"is empty at {instr_path!s}"
        )
    task.instruction = refreshed


def apply_harbor_patches() -> None:
    """Install runtime patches against Harbor 0.6+. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    # Imported here, not at module top, so unit tests that don't have
    # Harbor on PYTHONPATH can still import this file.
    from harbor.models.task.task import strip_canary
    from harbor.trial.trial import Trial
    from harbor.agents.installed.base import BaseInstalledAgent, ApiInternalServerError
    original_classifier = BaseInstalledAgent._classify_exec_error

    def classify(self, command, result):
        classified = original_classifier(self, command, result)
        output = f"{result.stdout or ''}\n{result.stderr or ''}"
        if _invalid_response_item_id(output) or _unavailable_model_channel(output):
            # Existing retry policy discards the attempt before scoring/learning.
            return ApiInternalServerError(str(classified))
        return classified

    BaseInstalledAgent._classify_exec_error = classify

    if hasattr(Trial, "_execute_agent"):
        # Harbor 0.6 compatibility.
        _orig_execute_agent = Trial._execute_agent

        async def _execute_agent_with_fresh_instruction(self) -> None:
            _refresh_instruction(self, strip_canary)
            await _orig_execute_agent(self)

        Trial._execute_agent = _execute_agent_with_fresh_instruction
        patched_entrypoint = "Trial._execute_agent"
    else:
        # Harbor 0.22 moved the single-step entrypoint onto its concrete
        # trial class. All SkillEvolBench tasks are single-step tasks.
        from harbor.trial.single_step import SingleStepTrial

        _orig_run_agent = SingleStepTrial._run_agent

        async def _run_agent_with_fresh_instruction(self) -> None:
            _refresh_instruction(self, strip_canary)
            await _orig_run_agent(self)

        SingleStepTrial._run_agent = _run_agent_with_fresh_instruction
        patched_entrypoint = "SingleStepTrial._run_agent"

    _PATCHED = True
    _LOG.info(
        "harbor_ext: patched %s to refresh task.instruction from disk "
        "before agent.run",
        patched_entrypoint,
    )


__all__ = ["apply_harbor_patches", "_refresh_instruction"]
