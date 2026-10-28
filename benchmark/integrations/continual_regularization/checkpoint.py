"""Command-boundary checkpoints shared by sequential baseline and Regularization.

Never replay a failed maintenance command on its partially edited filesystem.
Restart in a new container at the last completed *task*, not the failed command.
"""
import asyncio
import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protocol(agent, instruction):
    tests = getattr(agent, '_task_tests_dir', None)
    tests_hash = {str(p.relative_to(tests)): file_digest(p)
                  for p in sorted(Path(tests).rglob('*')) if p.is_file()} if tests else {}
    return dict(instruction=digest(instruction), tests=digest(tests_hash),
                model=agent.model_name, effort=agent._reasoning_effort,
                judge=agent._judge_model, pass_threshold=agent._judge_pass_threshold,
                judge_feedback=agent._enable_judge_feedback,
                skill_probe=agent._enable_skill_probe, compact_every_n=agent._compact_every_n,
                regularization=getattr(agent, 'regularization_config', None))


def load_checkpoint(path):
    path = Path(path).resolve()
    data = json.loads((path / 'manifest.json').read_text())
    if data.get('version') != 1 or data.get('status') != 'complete':
        raise ValueError('Not a complete Continual checkpoint')
    if not (0 < data['completed_step'] <= data['total_steps']):
        raise ValueError('Invalid checkpoint step')
    if file_digest(path / 'state.tar.gz') != data['archive_sha256']:
        raise ValueError('Checkpoint archive hash mismatch')
    if set(data['receipts']) != {str(i) for i in range(1, data['command_index'] + 1)}:
        raise ValueError('Checkpoint command prefix has gaps')
    if any(r['return_code'] != 0 for r in data['receipts'].values()):
        raise ValueError('Failed command in checkpoint prefix')
    return path, data


# Deliberately excludes credentials, installed tools and caches. Includes task
# files in /root and /app, memory, all real skill directories and Codex sessions.
# CODEX_HOME is /logs/agent in this harness, not the host's Codex App directory.
CAPTURE = r'''
import os, tarfile
from pathlib import Path
skip = {'root/.codex', 'root/.cache', 'root/.local', 'root/.nvm',
        'root/.rustup', 'root/.cargo', 'root/.npm', 'root/.ssh'}
def keep(info):
    if any(info.name == s or info.name.startswith(s + '/') for s in skip): return None
    return info
with tarfile.open('/tmp/continual-state.tar.gz', 'w:gz') as out:
    for name in ['root', 'app', 'logs/agent/sessions', 'logs/agent/skills', 'logs/verifier']:
        if Path('/' + name).exists(): out.add('/' + name, arcname=name, filter=keep)
'''

RESTORE = r'''
import tarfile
from pathlib import PurePosixPath
with tarfile.open('/tmp/continual-state.tar.gz') as archive:
    for member in archive.getmembers():
        p = PurePosixPath(member.name)
        if p.is_absolute() or '..' in p.parts or p.parts[0] not in ('root','app','logs'):
            raise ValueError('Unsafe checkpoint member')
        if member.issym() or member.islnk():
            # Checkpoints may contain ordinary relative workspace links only.
            q = PurePosixPath(member.linkname)
            if q.is_absolute() or '..' in q.parts: raise ValueError('Unsafe checkpoint link')
    archive.extractall('/', filter='data')
'''


class CheckpointEnvironment:
    def __init__(self, environment, agent, instruction):
        self.environment, self.agent, self.instruction = environment, agent, instruction
        self.index, self.receipts = 0, {}
        self.expected = protocol(agent, instruction)
        self.resume = None
        if agent.resume_checkpoint:
            self.path, self.resume = load_checkpoint(agent.resume_checkpoint)
            if self.resume['protocol'] != self.expected:
                raise ValueError('Resume protocol mismatch: task order/model/Regularization/grading changed')
            if self.resume.get('recovery_kind') == 'legacy_reconstructed' and not agent.allow_legacy_recovery:
                raise ValueError('Legacy recovery lacks workspace snapshot; explicit opt-in required')

    def __getattr__(self, name):
        return getattr(self.environment, name)

    async def _restore(self):
        await self.environment.upload_file(source_path=str(self.path / 'state.tar.gz'),
                                           target_path='/tmp/continual-state.tar.gz')
        # Bypass Regularization command accounting for filesystem-only restoration.
        raw = getattr(self.environment, 'environment', self.environment)
        result = await raw.exec(command='python3 -c ' + shlex.quote(RESTORE), timeout_sec=180)
        if result.return_code:
            raise RuntimeError('Checkpoint restore failed: ' + (result.stderr or '')[-1000:])
        # The archive contains the judge script that was deployed when the
        # checkpoint was created.  Reinstall only the current judge after the
        # state restore so evaluator fixes apply to resumed tasks without
        # overwriting the restored skill library, memory, or Codex sessions.
        build_judge = getattr(self.agent, '_build_deploy_judge_command', None)
        if callable(build_judge):
            result = await raw.exec(command=build_judge(), timeout_sec=180)
            if result.return_code:
                raise RuntimeError('Judge redeploy after checkpoint restore failed: '
                                   + ((result.stderr or '') + (result.stdout or ''))[-1000:])
        if hasattr(self.environment, 'successful_sources'):
            self.environment.successful_sources = self.resume['successful_sources']
        if self.resume.get('recovery_kind') == 'legacy_reconstructed':
            for step in range(1, self.resume['completed_step'] + 1):
                result = await raw.exec(command=self.agent._build_write_memory_command(step), timeout_sec=30)
                if result.return_code: raise RuntimeError('Legacy memory reconstruction failed')
        source = Path(self.resume['source_agent'])
        # Preserve regularization prefix for ordinary analysis; never copy failed tail.
        for step in range(1, self.resume['completed_step'] + 1):
            src = source / 'regularization' / f'step_{step:03d}'
            dst = self.agent.logs_dir / 'regularization' / src.name
            if src.exists(): shutil.copytree(src, dst, dirs_exist_ok=False)
        (self.agent.logs_dir / 'resume_provenance.json').write_text(json.dumps(dict(
            checkpoint=str(self.path), completed_step=self.resume['completed_step'],
            recovery_kind=self.resume.get('recovery_kind', 'checkpoint'),
            source_agent=str(source), archive_sha256=self.resume['archive_sha256'],
            judge_redeployed_after_restore=callable(build_judge)), indent=2))
        print(f'[Resume] restored through task {self.resume["completed_step"]}; '
              f'next task {self.resume["completed_step"] + 1}; prior scores reused', flush=True)

    async def _save(self, step):
        raw = getattr(self.environment, 'environment', self.environment)
        result = await raw.exec(command='python3 -c ' + shlex.quote(CAPTURE), timeout_sec=180)
        if result.return_code:
            raise RuntimeError('Checkpoint capture failed: ' + (result.stderr or '')[-1000:])
        root = self.agent.logs_dir / 'checkpoints'
        root.mkdir(exist_ok=True)
        target = root / f'step_{step:03d}'
        pending = root / f'.step_{step:03d}.pending'
        pending.mkdir(exist_ok=False)
        await raw.download_file(source_path='/tmp/continual-state.tar.gz', target_path=pending / 'state.tar.gz')
        data = dict(version=1, status='complete', completed_step=step,
                    total_steps=self.agent._checkpoint_total_steps, command_index=self.index,
                    protocol=self.expected, source_agent=str(self.agent.logs_dir.resolve()),
                    receipts=self.receipts, successful_sources=getattr(self.environment, 'successful_sources', []),
                    archive_sha256=file_digest(pending / 'state.tar.gz'), recovery_kind='checkpoint')
        if self.resume:
            data['ancestor_recovery_kind'] = self.resume.get('ancestor_recovery_kind', self.resume['recovery_kind'])
        (pending / 'manifest.json').write_text(json.dumps(data))
        pending.rename(target)  # Publish only after archive and manifest are durable files.
        print(f'[Checkpoint] task {step} saved', flush=True)

    @staticmethod
    def _retryable_transport_failure(result):
        blob = ((result.stdout or '') + '\n' + (result.stderr or '')).lower()
        return ('invalid_encrypted_content' in blob
                or 'encrypted content could not be decrypted or parsed' in blob
                or 'stream disconnected before completion' in blob
                or 'stream closed before response.completed' in blob
                or 'exceeded rate limit' in blob)

    @staticmethod
    def _fresh_session(command):
        return command.replace('codex exec resume --last', 'codex exec')

    @staticmethod
    def _task_output(command):
        match = re.search(r'/app/task_(\d{1,4})_output\.json', command)
        return f'/app/task_{int(match.group(1)):03d}_output.json' if match else None

    @staticmethod
    def _transport_retry_delay(result, attempt):
        """Avoid immediately re-entering the same congested provider window."""
        blob = ((result.stdout or '') + '\n' + (result.stderr or '')).lower()
        if ('exceeded rate limit' in blob
                or 'stream disconnected before completion' in blob
                or 'stream closed before response.completed' in blob):
            return (30, 90, 180)[min(attempt - 1, 2)]
        return 0

    async def _retry_solve_transport(self, kwargs, first_result, step):
        """Retry only encrypted-content transport failures, outside accounting.

        The Regularization has already prepared the masked library for this solve. Calling
        the raw environment avoids advancing Regularization/checkpoint command indices or
        applying dropout twice. The successful retry replaces the failed receipt.
        """
        result = first_result
        command = self._fresh_session(kwargs['command'])
        output = self._task_output(command)
        # Bypass RegularizationEnvironment; its per-step setup must happen exactly once.
        raw = getattr(self.environment, 'environment', self.environment)
        records = []
        for attempt in range(1, 4):
            if not self._retryable_transport_failure(result):
                break
            delay_sec = self._transport_retry_delay(result, attempt)
            if delay_sec:
                await asyncio.sleep(delay_sec)
            cleanup = f'rm -f {shlex.quote(output)}' if output else 'true'
            cleaned = await raw.exec(command=cleanup, timeout_sec=30)
            if cleaned.return_code:
                break
            retry_kwargs = dict(kwargs, command=command)
            result = await raw.exec(**retry_kwargs)
            records.append(dict(
                task=step, attempt=attempt, fresh_session=True,
                backoff_sec=delay_sec,
                removed_partial_output=output, return_code=result.return_code,
                retryable_transport_failure=self._retryable_transport_failure(result),
            ))
            if result.return_code == 0:
                # RegularizationEnvironment did not observe the bypassed retry directly.
                if hasattr(self.environment, 'rollout_trace'):
                    self.environment.rollout_trace = result.stdout or ''
                break
        if records:
            path = self.agent.logs_dir / 'transport_retries.jsonl'
            with path.open('a') as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return result

    async def exec(self, **kwargs):
        self.index += 1
        phase = self.agent._display_plan[self.index]
        if self.resume and self.index <= self.resume['command_index']:
            if self.index == 1:
                # Deploy official runtime/judge/meta-skills, then restore state.
                result = await self.environment.exec(**kwargs)
                if result.return_code: return result
                await self._restore()
            if hasattr(self.environment, 'index'):
                self.environment.index = self.index
            receipt = self.resume['receipts'][str(self.index)]
            self.receipts[str(self.index)] = receipt
            return SimpleNamespace(**receipt)
        result = await self.environment.exec(**kwargs)
        if (phase[1] == '做题' and result.return_code != 0
                and self._retryable_transport_failure(result)):
            result = await self._retry_solve_transport(kwargs, result, phase[0])
        self.receipts[str(self.index)] = dict(return_code=result.return_code,
                                             stdout=result.stdout, stderr=result.stderr)
        if result.return_code == 0 and self.agent.enable_checkpoints and self.index in self.agent._checkpoint_boundaries:
            await self._save(phase[0])
        return result
