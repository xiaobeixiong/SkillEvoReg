"""Strict frozen-library evaluation with no held-out feedback carry-over.

This module is intentionally separate from ``MathHeldout`` so existing and
running experiments keep their exact behaviour.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from pathlib import Path

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from harbor.agents.installed.codex_seq import CodexSeq, _split_tasks
from continual_regularization.baseline import (
    API_PREFLIGHT, DIRECT_PREFIX, CheckedEnvironment, codex_route_flags,
)
from .agent import MathHeldout


EVAL_INTRO = """\
You are evaluating a fixed, read-only skill library on held-out tasks.
Each task is an independent evaluation item. You will receive the task, solve it,
and write its required output file. A judge scores the output outside your session.
There is no feedback or skill-maintenance turn. Do not create, modify, move, or
delete anything under any skill directory. Learned skills may be read as reference.
"""

INITIALIZATION_PROMPT = EVAL_INTRO + """\

This is only an initialization handshake; no evaluation task has been supplied.
Reply exactly READY. Do not call tools, inspect files, browse, or begin any task.
"""


class FrozenCheckedEnvironment(CheckedEnvironment):
    """Fail closed if a solve command changes the declared skill library."""

    def __init__(self, *args, agent, expected_digest, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.expected_digest = expected_digest

    @staticmethod
    def _retryable_transport_failure(result) -> bool:
        blob = ((result.stdout or '') + '\n' + (result.stderr or '')).lower()
        return ('invalid_encrypted_content' in blob
                or 'encrypted content could not be decrypted or parsed' in blob
                or 'stream disconnected before completion' in blob
                or 'stream closed before response.completed' in blob
                or 'exceeded rate limit' in blob
                or '502 bad gateway' in blob
                or '503 service unavailable' in blob
                or '504 gateway timeout' in blob)

    @staticmethod
    def _fresh_session(command: str) -> str:
        """Retry the same task prompt without referencing a failed Codex thread."""
        return command.replace('codex exec resume --last', 'codex exec')

    @staticmethod
    def _task_output(command: str) -> str | None:
        match = re.search(r'/app/task_(\d{1,4})_output\.json', command)
        return f'/app/task_{int(match.group(1)):03d}_output.json' if match else None

    @staticmethod
    def _transport_retry_delay(result, attempt: int) -> int:
        """Back off only for transient provider congestion/stream failures.

        Invalid encrypted history is avoided by the fresh session itself and
        therefore needs no delay.  Rate limiting and stream instability do:
        immediate retries merely hit the same unhealthy provider window.
        """
        blob = ((result.stdout or '') + '\n' + (result.stderr or '')).lower()
        if ('exceeded rate limit' in blob
                or 'stream disconnected before completion' in blob
                or 'stream closed before response.completed' in blob
                or '502 bad gateway' in blob
                or '503 service unavailable' in blob
                or '504 gateway timeout' in blob):
            return (30, 90, 180, 300, 300, 300, 300, 300)[min(attempt - 1, 7)]
        return 0

    async def _command_with_transport_retry(self, kwargs, first_result, task_number):
        """Retry a proven transport failure in a fresh Codex session.

        This covers both the one-time session initialization and task solves.
        A possibly written task output is removed first, so a solve retry cannot
        inherit a partial answer. Skill roots are checked by the normal
        before/after frozen digest audit and are never modified here.
        """
        result = first_result
        records = []
        retry_kwargs = dict(kwargs)
        retry_kwargs['command'] = self._fresh_session(retry_kwargs['command'])
        output = self._task_output(retry_kwargs['command'])
        for attempt in range(1, 9):
            if not self._retryable_transport_failure(result):
                break
            delay_sec = self._transport_retry_delay(result, attempt)
            if delay_sec:
                await asyncio.sleep(delay_sec)
            cleanup = f'rm -f {shlex.quote(output)}' if output else 'true'
            cleaned = await self.environment.exec(command=cleanup, timeout_sec=30)
            if cleaned.return_code:
                break
            result = await self.environment.exec(**retry_kwargs)
            records.append({
                'task': task_number,
                'attempt': attempt,
                'fresh_session': True,
                'backoff_sec': delay_sec,
                'removed_partial_output': output,
                'return_code': result.return_code,
                'retryable_transport_failure': self._retryable_transport_failure(result),
            })
            if result.return_code == 0:
                break
        if records:
            with (self.logs_dir / 'frozen_transport_retries.jsonl').open('a') as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return result

    @staticmethod
    def _retryable_judge_failure(result) -> bool:
        blob = ((result.stdout or '') + '\n' + (result.stderr or '')).lower()
        return 'judge api failure: batch not scoreable' in blob

    async def _judge_with_retry(self, kwargs, first_result, task_number):
        """Resample only an infrastructure-invalid judge result.

        The task output and frozen library are left untouched. The judge command
        overwrites its own feedback artifact, so a malformed/empty model response
        cannot remain as a scored zero when a later attempt succeeds.
        """
        result = first_result
        records = []
        for attempt, delay_sec in enumerate((5, 15, 30), start=1):
            if not self._retryable_judge_failure(result):
                break
            await asyncio.sleep(delay_sec)
            result = await self.environment.exec(**kwargs)
            records.append({
                'task': task_number,
                'attempt': attempt,
                'backoff_sec': delay_sec,
                'return_code': result.return_code,
                'retryable_judge_failure': self._retryable_judge_failure(result),
            })
            if result.return_code == 0:
                break
        if records:
            with (self.logs_dir / 'frozen_judge_retries.jsonl').open('a') as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return result

    async def _check(self, number, boundary):
        script = Path(__file__).with_name('frozen_verify.py').read_text()
        result = await self.environment.exec(
            command='python3 -c ' + shlex.quote(script),
            env={'CONTINUAL_FROZEN_ROOTS': json.dumps(
                __import__('continual_regularization.container_io', fromlist=['ROOTS']).ROOTS)},
            timeout_sec=180)
        if result.return_code:
            record = {'verified': False, 'library_unchanged': False,
                      'error': ((result.stderr or '') + (result.stdout or ''))[-800:]}
        else:
            observed = json.loads(result.stdout.strip().splitlines()[-1])
            roots = {root: row['digest'] for root, row in observed['roots'].items()}
            record = {
                'verified': True,
                'library_unchanged': observed['canonical_digest'] == self.expected_digest
                                     and len(set(roots.values())) == 1,
                'canonical_digest': observed['canonical_digest'],
                'roots_identical': len(set(roots.values())) == 1,
            }
        row = dict(task=number, boundary=boundary, **record)
        with (self.logs_dir / 'frozen_step_integrity.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if record.get('verified') is not True or record.get('library_unchanged') is not True:
            raise RuntimeError(f'Frozen library changed at task {number} ({boundary})')

    async def exec(self, **kwargs):
        number, phase = self.plan.get(self.index + 1, (0, '初始化'))
        if phase == '做题':
            await self._check(number, 'before_solve')
        # Inline the small CheckedEnvironment operation so a failed solve can
        # be classified before fail-closed propagation.  Other phases retain
        # the exact original behavior and are never retried here.
        self.index += 1
        if self.display:
            progress, task = self.display
            progress.update(task, phase=f'第 {number} 题 · {phase}' if number else phase)
        result = await self.environment.exec(**kwargs)
        if (phase in ('初始化会话', '做题') and result.return_code != 0
                and self._retryable_transport_failure(result)):
            result = await self._command_with_transport_retry(kwargs, result, number)
        if (phase == '评分' and result.return_code != 0
                and self._retryable_judge_failure(result)):
            result = await self._judge_with_retry(kwargs, result, number)
        if result.return_code != 0:
            (self.logs_dir / 'infrastructure_failure.json').write_text(json.dumps({
                'command_index': self.index, 'return_code': result.return_code,
                'stdout': result.stdout, 'stderr': result.stderr,
            }, ensure_ascii=False))
            raise RuntimeError('Continual command failed; stopping before further skill updates')
        if phase in ('初始化会话', '做题'):
            from .usage import record_usage
            record_usage(self, result, phase, number, kwargs.get('command', ''))
        if 'HARBOR_GRADING_SPEC_JSON' in (kwargs.get('env') or {}):
            for line in (result.stdout or '').splitlines():
                if line.startswith('CONTINUAL_SCORE '):
                    data = json.loads(line[len('CONTINUAL_SCORE '):])
                    self.scores[data['task_id']] = float(data['score'])
            if self.display and self.scores:
                mean = sum(self.scores.values()) / len(self.scores)
                progress.update(task, score=f'已评分 {len(self.scores)} 题均分 {mean:.1%}')
        if phase == '做题':
            await self._check(number, 'after_solve')
        if self.display and phase == '评分':
            progress, task = self.display
            progress.update(task, completed=number, phase=f'第 {number} 题完成')
        return result


class MathStatelessHeldout(MathHeldout):
    """Frozen tail evaluation where held-out labels never reach later tasks."""

    frozen_display_name = "Frozen Math"

    def __init__(self, *args, prefer_skills=False, **kwargs):
        self.prefer_skills = prefer_skills
        super().__init__(*args, **kwargs)

    def _task_prompt(self, task_num, n_tasks, sub_instruction):
        return (
            (EVAL_INTRO.replace(
                'Learned skills may be read as reference.',
                'Prefer skills found under $HOME/.agents/skills/. '
                'Inspect them before inventing a new approach.'
            ) if self.prefer_skills else EVAL_INTRO) + '\n'
            f'[Task {task_num} of {n_tasks}] [held-out, stateless]\n\n{sub_instruction}'
        )

    def _build_task_instruction(self, task_num, n_tasks, sub_instruction,
                                include_memory_hint, include_skill_probe=False):
        if include_skill_probe:
            raise ValueError('Skill probes are disabled in strict frozen evaluation')
        return self._task_prompt(task_num, n_tasks, sub_instruction)

    def _build_compaction_task_instruction(self, task_num, n_tasks, sub_instruction,
                                           include_skill_probe=False):
        if include_skill_probe:
            raise ValueError('Skill probes are disabled in strict frozen evaluation')
        return self._task_prompt(task_num, n_tasks, sub_instruction)

    def create_run_agent_commands(self, instruction):
        commands = super().create_run_agent_commands(instruction)
        plan = self._display_plan
        kept, rebuilt = [], {}
        intro_count = memory_count = 0
        for index, command in enumerate(commands, start=1):
            number, phase = plan[index]
            # Remove only the 49 feedback-memory append executions, not setup or
            # credential cleanup commands that share a display label.
            if phase == '保存上下文' and 'python3 /root/append_task_memory.py ' in command.command:
                memory_count += 1
                continue
            if phase == '初始化会话':
                intro_count += 1
                model = self.model_name.split('/')[-1]
                raw = self._build_codex_exec_command(INITIALIZATION_PROMPT, model)
                raw = raw.replace('codex exec', 'codex exec ' + codex_route_flags(), 1)
                command.command = 'bash -o pipefail -c ' + shlex.quote(DIRECT_PREFIX + raw)
            kept.append(command)
            rebuilt[len(kept)] = (number, phase)
        if intro_count != 1 or memory_count != max(0, len(_split_tasks(instruction)) - 1):
            raise RuntimeError(f'Unexpected frozen plan: intro={intro_count}, memory={memory_count}')
        plan.clear()
        plan.update(rebuilt)
        return kept

    async def run(self, instruction, environment, context):
        tasks = _split_tasks(instruction)
        if self.limit_tasks:
            tasks = tasks[:self.limit_tasks]
            instruction = '\n---TASK_BREAK---\n'.join(tasks) + '\n'
        self._task_count = len(tasks)
        await self._provision(environment)
        declared = json.loads((self.logs_dir / 'frozen_library.json').read_text())['declared_digest']
        final = None
        try:
            for attempt in range(4):
                probe = await environment.exec(
                    command='python3 -c ' + shlex.quote(API_PREFLIGHT),
                    env={'OPENAI_API_KEY': os.environ['OPENAI_API_KEY'],
                         'OPENAI_BASE_URL': os.environ['OPENAI_BASE_URL'],
                         'NO_PROXY': '*', 'no_proxy': '*'}, timeout_sec=60)
                if not probe.return_code:
                    break
                (self.logs_dir / f'preflight_error_{attempt+1}.txt').write_text(
                    (probe.stderr or '') + (probe.stdout or ''))
                if attempt == 3 or not FrozenCheckedEnvironment._retryable_transport_failure(probe):
                    break
                delay = (30, 90, 180)[attempt]
                print(f'[Preflight retry] waiting {delay}s; no evaluation started', flush=True)
                await asyncio.sleep(delay)
            if probe.return_code:
                raise RuntimeError('Container API preflight failed; no evaluation started')
            print((probe.stdout or '').strip(), flush=True)
            self._display_plan = {}
            with Progress(TextColumn(self.frozen_display_name), BarColumn(),
                          TextColumn('{task.completed:.0f}/{task.total:.0f}'),
                          TextColumn('{task.fields[phase]}'),
                          TextColumn('{task.fields[score]}'), TimeElapsedColumn()) as progress:
                task = progress.add_task(self.frozen_display_name, total=len(tasks),
                                         phase='初始化', score='尚无评分')
                wrapped = FrozenCheckedEnvironment(
                    environment, self.logs_dir, (progress, task), self._display_plan,
                    agent=self, expected_digest=declared)
                await CodexSeq.run(self, instruction, wrapped, context)
        finally:
            try:
                final = await self._verify_intact(environment, declared)
            except Exception as error:
                final = {'verified': False, 'error': f'{type(error).__name__}: {error}'[:800]}
            (self.logs_dir / 'frozen_library_final.json').write_text(
                json.dumps(final, indent=2, ensure_ascii=False))


class ScienceStatelessHeldout(MathStatelessHeldout):
    """The same strict frozen evaluator for Science tasks 51-100."""

    frozen_display_name = "Frozen Science"


class FinanceStatelessHeldout(MathStatelessHeldout):
    """Strict frozen evaluator for Finance/Economics tasks 51-100."""

    frozen_display_name = "Frozen Finance"


class LawStatelessHeldout(MathStatelessHeldout):
    """Strict frozen evaluator for the published Law tasks 51--100."""

    frozen_display_name = "Frozen Law"


class OfficeStatelessHeldout(MathStatelessHeldout):
    """Strict frozen evaluator for the published Office tasks 51--100."""

    frozen_display_name = "Frozen Office"


class HealthcareStatelessHeldout(MathStatelessHeldout):
    """Strict frozen evaluator for the published Healthcare tasks 51--100."""

    frozen_display_name = "Frozen Healthcare"
