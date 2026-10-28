"""RuntimeBuilder -- per-trial runtime task copy + instruction injection (Part 6 §6.2).

The Part 4 ``build_job_config`` already pre-copied each task's
``benchmark/tasks/<slug>/`` directory into
``workspace/runs/<run_id>/runtime/<task_id>/harbor-task-copy/`` *before*
Harbor started. This builder runs *per trial* in
``SkillEvolBenchHooks.on_trial_started`` and only:

1. Asserts the skeleton is present (sanity check).
2. Composes the injected ``instruction.md`` via ``PromptBuilder``.
3. Overwrites the runtime ``instruction.md`` with the injected version.
4. Writes a per-trial ``injection-context.json`` audit snapshot for the
   container's ``/context/injection.json`` mount.

Returns the runtime task directory path so the hook can store it in the
pre-state cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from skillevolbench.prompting.prompt_builder import PromptBuilder
from skillevolbench.schemas import BaselineConfig


_LOG = logging.getLogger(__name__)


class RuntimeBuilder:
    """Compose + write the runtime ``instruction.md`` plus an audit snapshot."""

    def __init__(self, prompt_builder: Optional[PromptBuilder] = None) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        task: Any,
        run_root: Path,
        baseline: BaselineConfig,
        retrieved_skills: list[Any],
        retrieved_trajectories: list[Any],
        history_context: Optional[str],
        library_frozen: bool,
        runtime_basename: Optional[str] = None,
    ) -> Path:
        run_root = Path(run_root)
        # Replay / shadow trials use a runtime dir basename like
        # "E1-LS1-T1__replay" or "E1-LS1-T6__oracle_shadow", NOT the bare
        # task_id. If the caller passes ``runtime_basename`` explicitly,
        # use that. Otherwise fall back to ``task.task_id`` for primary
        # trials (the common path).
        basename = runtime_basename or task.task_id
        runtime_dir = run_root / "runtime" / basename

        # 1. Skeleton sanity check
        if not runtime_dir.exists():
            raise FileNotFoundError(
                f"Runtime task skeleton not pre-copied: {runtime_dir}. "
                f"Did Part 4 build_job_config run?"
            )
        if not (runtime_dir / "task.toml").exists():
            raise FileNotFoundError(
                f"Runtime task skeleton incomplete: {runtime_dir / 'task.toml'} missing"
            )

        # 2. Read original instruction
        original_instruction = self._read_original_instruction(task, runtime_dir)

        # 3. Compose injected instruction
        injected_instruction = self.prompt_builder.build(
            original_instruction=original_instruction,
            baseline=baseline,
            task=task,
            retrieved_skills=retrieved_skills,
            retrieved_trajectories=retrieved_trajectories,
            history_context=history_context,
            library_frozen=library_frozen,
        )

        # 4. Overwrite runtime instruction.md
        instruction_path = runtime_dir / task.harbor.instruction
        instruction_path.write_text(injected_instruction)

        # 5. Audit snapshot
        injection_audit_path = runtime_dir / "injection-context.json"
        injection_audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit = {
            "task_id": task.task_id,
            "task_slug": task.task_slug,
            "baseline": baseline.name,
            "retrieved_skill_ids": [
                getattr(s, "skill_id", str(s)) for s in retrieved_skills
            ],
            "n_trajectories_retrieved": len(retrieved_trajectories or []),
            "history_context_chars": len(history_context or ""),
            "library_frozen": bool(library_frozen),
            "injected_instruction_chars": len(injected_instruction),
        }
        injection_audit_path.write_text(json.dumps(audit, indent=2))

        return runtime_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_original_instruction(task: Any, runtime_dir: Path) -> str:
        """Read the un-overwritten instruction.

        Harbor may invoke ``on_trial_started`` more than once for the same
        runtime directory when a transient infrastructure failure is retried.
        After the first invocation, ``instruction.md`` contains the injected
        prompt, so reading it again would recursively duplicate the task and
        evaluation constraint. Preserve the pristine skeleton instruction in
        an audit-only sidecar on first use and always read that sidecar on
        subsequent attempts.

        The runtime directory is freshly copied by ``build_job_config`` for a
        new run, so the first read remains byte-identical to the immutable
        benchmark asset. The sidecar is not part of the benchmark tree or the
        mounted skill library.
        """
        instr_path = runtime_dir / task.harbor.instruction
        pristine_path = runtime_dir / ".skillevol-original-instruction.md"
        if pristine_path.is_file():
            return pristine_path.read_text()
        original = instr_path.read_text()
        pristine_path.write_text(original)
        return original


__all__ = ["RuntimeBuilder"]
