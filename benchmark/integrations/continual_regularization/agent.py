"""Regularization-only command-boundary adapter. Baseline class is unmodified."""
import asyncio
import base64
import json
import shlex
import tempfile
import uuid
from pathlib import Path

from .baseline import MathBaseline
from .dropout import mask_library, restore_library
from .engine import Settings, StepRegularization
from .ports import Model, Retrieval, measure, measure_scope, quick


class RegularizationEnvironment:
    def __init__(self, env, agent, instruction):
        self.environment, self.agent = env, agent
        self.index = 0
        from harbor.agents.installed.codex_seq import _split_tasks
        self.tasks = _split_tasks(instruction)
        self.script = Path(__file__).with_name('container_io.py').read_text()
        self.successful_sources = []
        self.rollout_trace = ''

    def __getattr__(self, name):
        return getattr(self.environment, name)

    async def io(self, request):
        # Whole mirrored snapshots exceed Linux's per-environment-variable limit.
        remote = '/tmp/continual-io-' + uuid.uuid4().hex + '.json'
        with tempfile.TemporaryDirectory(prefix='continual-io-') as directory:
            payload = Path(directory) / 'request.json'
            payload.write_text(json.dumps(request))
            await self.environment.upload_file(source_path=str(payload), target_path=remote)
            result = await self.environment.exec(command='python3 -c ' + shlex.quote(self.script),
                env={'CONTINUAL_IO_PATH': remote}, timeout_sec=60)
        if result.return_code:
            raise RuntimeError('Regularization library transaction failed: ' + ((result.stderr or '') + (result.stdout or ''))[-1500:])
        return json.loads(result.stdout)

    async def commit(self, snapshots, desired):
        canonical = snapshots['/root/.agents/skills']
        changes = {p: desired.get(p) for p in canonical.keys() | desired.keys()
                   if canonical.get(p) != desired.get(p)}
        if changes:
            await self.io({'op': 'write', 'expected': snapshots, 'changes': changes})

    async def exec(self, **kwargs):
        # MathBaseline API preflight is not one of the planned commands.
        if not getattr(self.agent, '_display_plan', {}):
            return await self.environment.exec(**kwargs)
        self.index += 1
        number, phase = self.agent._display_plan[self.index]
        cfg = self.agent.regularization_config
        if phase == '做题':
            self.directory = self.agent.logs_dir / 'regularization' / f'step_{number:03d}'
            self.directory.mkdir(parents=True, exist_ok=False)
            self.snapshots = await self.io({'op': 'read'})
            self.old = self.snapshots['/root/.agents/skills']
            from skillevolbench.components.online_regularization import rule_segment
            masked, self.records = mask_library(self.old, seed=cfg['seed'] + number,
                segment=rule_segment, **cfg['dropout'])
            (self.directory / 'before.json').write_text(json.dumps(self.snapshots))
            (self.directory / 'dropout.json').write_text(json.dumps(self.records))
            await self.commit(self.snapshots, masked)
            # Tell the native author markers are temporary, not missing knowledge.
            # We preserve the official task and maintenance prompt otherwise.
        if phase == '确认技能写入':
            snapshots = await self.io({'op': 'read'})
            raw = snapshots['/root/.agents/skills']
            (self.directory / 'raw.json').write_text(json.dumps(raw))
            restored = restore_library(raw, self.records, allow_deleted=True)
            (self.directory / 'restored.json').write_text(json.dumps(restored))
            model = Model(self.directory / 'model')
            ccv = self.agent.build_ccv(
                model, self.directory / 'ccv', cfg['probe_timeout'],
                minimum_drop=cfg['budgets']['ccv_minimum_drop'])
            engine = StepRegularization(Settings(**cfg['budgets']), measure=measure,
                measure_scope=measure_scope,
                retrieve=Retrieval(**cfg['retrieval']), local_edit=lambda req: quick(model, req),
                ccv=ccv, repair_regression=ccv.repair)
            final, events = await asyncio.to_thread(engine.finish, self.old, restored,
                masked_paths=set(self.records), source={'pool': list(self.successful_sources)})
            (self.directory / 'events.json').write_text(json.dumps(events, indent=2))
            (self.directory / 'final.json').write_text(json.dumps(final))
            await self.commit(snapshots, final)
            (self.directory / 'committed.json').write_text(json.dumps({'status': 'committed'}))
            print(f'[Regularization] step {number}: {len(events)-1} update records; CCV={events[-1]["status"]}', flush=True)
        if phase == '技能更新' and self.records:
            from .baseline import DIRECT_PREFIX, codex_route_flags
            prompt = self.agent._build_judge_feedback_turn(number, len(self.tasks)) + (
                '\nRuntime-only instruction, never copy into skill content: '
                'Existing temporary HTML comment placeholders must stay exactly once in their current locations. '
                'Their original text will be restored externally. Do not invent, duplicate, move or explain placeholders.')
            command = self.agent._build_codex_resume_command(prompt, 'gpt-5.4-mini').replace('. ~/.nvm/nvm.sh; ', '')
            command = command.replace('codex exec resume --last', 'codex exec resume --last ' + codex_route_flags(), 1)
            kwargs['command'] = 'bash -o pipefail -c ' + shlex.quote(DIRECT_PREFIX + command)
            (self.directory / 'maintenance_command.txt').write_text(kwargs['command'])
        result = await self.environment.exec(**kwargs)
        if result.return_code == 0 and phase == '做题':
            self.rollout_trace = result.stdout or ''
        if result.return_code == 0 and phase == '评分':
            for line in (result.stdout or '').splitlines():
                if line.startswith('CONTINUAL_SCORE '):
                    score = json.loads(line[len('CONTINUAL_SCORE '):])
                    if float(score['score']) >= .6:
                        self.successful_sources.append({'instruction': self.tasks[number-1],
                            'task_id': score['task_id'], 'training_score': score['score'],
                            'trajectory': self.rollout_trace,
                            'grading': json.loads(self.agent._grading_env_for_task(score['task_id'])['HARBOR_GRADING_SPEC_JSON']),
                            'tests': str(self.agent._task_tests_dir)})
        return result


class MathRegularization(MathBaseline):
    def __init__(self, *args, regularization_config_path, smoke_tasks=None, **kwargs):
        self.regularization_config = json.loads(Path(regularization_config_path).read_text())
        self.smoke_tasks = smoke_tasks
        super().__init__(*args, **kwargs)

    def build_ccv(self, model, directory, timeout, minimum_drop):
        from .ccv import MathCCV
        return MathCCV(model, directory, timeout, minimum_drop=minimum_drop)

    async def run(self, instruction, environment, context):
        if self.smoke_tasks:
            from harbor.agents.installed.codex_seq import _split_tasks
            instruction = '\n---TASK_BREAK---\n'.join(_split_tasks(instruction)[:self.smoke_tasks])
        await super().run(instruction, RegularizationEnvironment(environment, self, instruction), context)


class ScienceRegularization(MathRegularization):
    """Continual Regularization with a Science-specific CCV adapter."""

    display_name = "Science"

    def build_ccv(self, model, directory, timeout, minimum_drop):
        from .ccv import ScienceCCV
        return ScienceCCV(model, directory, timeout, minimum_drop=minimum_drop)


class FinanceRegularization(MathRegularization):
    """Continual Regularization with a Finance/Economics-specific CCV adapter."""

    display_name = "Finance"

    def build_ccv(self, model, directory, timeout, minimum_drop):
        from .ccv import FinanceCCV
        return FinanceCCV(model, directory, timeout, minimum_drop=minimum_drop)


class LawRegularization(MathRegularization):
    """Continual Regularization with a Law-specific CCV policy."""

    display_name = "Law"

    def build_ccv(self, model, directory, timeout, minimum_drop):
        from .ccv import LawCCV
        return LawCCV(model, directory, timeout, minimum_drop=minimum_drop)


class OfficeRegularization(MathRegularization):
    """Continual Regularization with an Office-workflow-specific CCV policy."""

    display_name = "Office"

    def build_ccv(self, model, directory, timeout, minimum_drop):
        from .ccv import OfficeCCV
        return OfficeCCV(model, directory, timeout, minimum_drop=minimum_drop)


class HealthcareRegularization(MathRegularization):
    """Continual Regularization with a healthcare-specific CCV policy."""

    display_name = "Healthcare"

    def build_ccv(self, model, directory, timeout, minimum_drop):
        from .ccv import HealthcareCCV
        return HealthcareCCV(model, directory, timeout, minimum_drop=minimum_drop)
