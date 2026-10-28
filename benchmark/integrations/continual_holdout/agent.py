"""Frozen-library held-out evaluation.

New module: it subclasses the existing baseline agent and never modifies it, so no
other experiment changes behaviour.

What this agent does differently
--------------------------------
1. **Frozen library.** A declared skill library is provisioned into the container's
   skill roots before the run starts, and the roots are pruned first, so a condition
   is exactly its declared library.
2. **No library writes.** The judge-feedback / skill-maintenance turn ("技能更新")
   and the skill-verification command ("确认技能写入") are dropped from the command
   plan, so nothing is created, merged, rewritten or deleted during evaluation.
3. **No Regularization runtime.** It extends ``MathBaseline``, so CCV, the local edit and
   dropout never run.  The library is the only thing that varies.

The purpose is to separate *what was learned* from *how it was learned*: the same
held-out subtasks are solved under different frozen libraries, so any difference is
attributable to the library rather than to the process that produced it.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import tarfile
import tempfile
from pathlib import Path

from continual_regularization.baseline import MathBaseline
from continual_regularization.container_io import ROOTS

# Phases that write to the library.  They are removed for frozen evaluation.
SKIP_PHASES = {"技能更新", "确认技能写入"}

HELPERS = {"create-skill", "modify-skill"}
CANONICAL = "/root/.agents/skills"
ARCHIVE_TARGET = "/tmp/continual-frozen-library.tar.gz"


def library_digest(root: Path) -> tuple[str, list[str]]:
    """Digest a library directory exactly like the container-side verifier does."""
    accumulator = hashlib.sha256()
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] not in HELPERS
    )
    for relative in files:
        accumulator.update(relative.encode("utf-8"))
        accumulator.update(b"\0")
        accumulator.update((root / relative).read_bytes())
        accumulator.update(b"\0")
    return accumulator.hexdigest(), files


class MathHeldout(MathBaseline):
    """Baseline sequential agent with a declared, frozen skill library."""

    def __init__(self, *args, frozen_library=None, freeze_skill_updates=True,
                 limit_tasks=None, **kwargs):
        self.frozen_library = Path(frozen_library) if frozen_library else None
        self.freeze_skill_updates = freeze_skill_updates
        self.limit_tasks = limit_tasks
        kwargs.setdefault("enable_checkpoints", False)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ plan
    def create_run_agent_commands(self, instruction):
        commands = super().create_run_agent_commands(instruction)
        if not self.freeze_skill_updates:
            return commands
        plan = getattr(self, "_display_plan", {})
        kept, rebuilt = [], {}
        for index, command in enumerate(commands, start=1):
            entry = plan.get(index)
            if entry is not None and entry[1] in SKIP_PHASES:
                continue
            kept.append(command)
            rebuilt[len(kept)] = entry if entry is not None else (0, "初始化")
        # CheckedEnvironment already holds a reference to this dict, so update it
        # in place rather than rebinding, or the displayed phase would drift.
        plan.clear()
        plan.update(rebuilt)
        return kept

    # ------------------------------------------------------------- injection
    def _pack_library(self, target: Path) -> tuple[str, list[str]]:
        source = self.frozen_library
        digest, files = library_digest(source) if source else (
            hashlib.sha256().hexdigest(), []
        )
        with tarfile.open(target, "w:gz") as archive:
            for relative in files:
                archive.add(source / relative, arcname=relative)
        return digest, files

    async def _provision(self, environment):
        record: dict = {
            "frozen_library": str(self.frozen_library) if self.frozen_library else None,
            "freeze_skill_updates": self.freeze_skill_updates,
            "skip_phases": sorted(SKIP_PHASES),
        }
        with tempfile.TemporaryDirectory(prefix="continual-frozen-") as directory:
            archive = Path(directory) / "library.tar.gz"
            expected_digest, expected_files = self._pack_library(archive)
            record["declared_digest"] = expected_digest
            record["declared_files"] = len(expected_files)
            await environment.upload_file(source_path=str(archive), target_path=ARCHIVE_TARGET)

        script = Path(__file__).with_name("frozen_io.py").read_text()
        result = await environment.exec(
            command="python3 -c " + shlex.quote(script),
            env={
                "CONTINUAL_FROZEN_ARCHIVE": ARCHIVE_TARGET,
                "CONTINUAL_FROZEN_ROOTS": json.dumps(ROOTS),
            },
            timeout_sec=300,
        )
        if result.return_code:
            raise RuntimeError(
                "Frozen library provisioning failed: "
                + ((result.stderr or "") + (result.stdout or ""))[-1500:]
            )
        try:
            observed = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as error:
            raise RuntimeError(
                f"Frozen library provisioning returned unreadable output: {result.stdout!r}"
            ) from error

        record["observed"] = observed
        record["digest_match"] = observed.get("canonical_digest") == expected_digest
        if not record["digest_match"]:
            raise RuntimeError(
                "Frozen library digest mismatch: declared "
                f"{expected_digest[:16]} vs observed {str(observed.get('canonical_digest'))[:16]}"
            )
        (self.logs_dir / "frozen_library.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )
        print(
            f"[Frozen] {record['declared_files']} files, "
            f"{observed.get('n_skills')} skills in {CANONICAL}; "
            f"pruned {sum(observed.get('pruned', {}).values())}; digest verified",
            flush=True,
        )

    # ------------------------------------------------------- end-of-run check
    async def _verify_intact(self, environment, expected_digest: str):
        """Prove the library the agent was graded against is the declared one."""
        script = Path(__file__).with_name("frozen_verify.py").read_text()
        result = await environment.exec(
            command="python3 -c " + shlex.quote(script),
            env={"CONTINUAL_FROZEN_ROOTS": json.dumps(ROOTS)},
            timeout_sec=180,
        )
        if result.return_code:
            return {
                "verified": False,
                "error": ((result.stderr or "") + (result.stdout or ""))[-800:],
            }
        observed = json.loads(result.stdout.strip().splitlines()[-1])
        roots = {root: entry["digest"] for root, entry in observed["roots"].items()}
        intact = observed["canonical_digest"] == expected_digest and len(set(roots.values())) == 1
        record = {
            "verified": True,
            "library_unchanged": intact,
            "declared_digest": expected_digest,
            "canonical_digest": observed["canonical_digest"],
            "n_files": observed["n_files"],
            "n_skills": len(observed["skills"]),
            "roots_identical": len(set(roots.values())) == 1,
            "helpers_present": observed["helpers_present"],
        }
        if not intact:
            print(
                "[Frozen] WARNING: the skill library changed during evaluation -- "
                "this trial is not a valid frozen-library measurement",
                flush=True,
            )
        else:
            print(
                f"[Frozen] library intact after {self._task_count} task(s): "
                f"{observed['n_files']} files, digest verified",
                flush=True,
            )
        return record

    # ------------------------------------------------------------------- run
    async def run(self, instruction, environment, context):
        from harbor.agents.installed.codex_seq import _split_tasks
        tasks = _split_tasks(instruction)
        if self.limit_tasks:
            tasks = tasks[: self.limit_tasks]
            instruction = "\n---TASK_BREAK---\n".join(tasks) + "\n"
        self._task_count = len(tasks)
        await self._provision(environment)
        declared = json.loads((self.logs_dir / "frozen_library.json").read_text())[
            "declared_digest"
        ]
        try:
            await super().run(instruction, environment, context)
        finally:
            try:
                final = await self._verify_intact(environment, declared)
            except Exception as error:  # never mask the run's own outcome
                final = {"verified": False, "error": f"{type(error).__name__}: {error}"[:800]}
            (self.logs_dir / "frozen_library_final.json").write_text(
                json.dumps(final, indent=2, ensure_ascii=False)
            )
