import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from continual_regularization import checkpoint as cp


def bundle(path, **changes):
    path.mkdir()
    (path/'state.tar.gz').write_bytes(b'test archive')
    data = dict(version=1, status='complete', completed_step=1, total_steps=2,
                command_index=2, protocol={'test':1}, source_agent=str(path/'source'),
                successful_sources=[{'task_id':1}], recovery_kind='checkpoint',
                receipts={'1':dict(return_code=0,stdout='setup',stderr=None),
                          '2':dict(return_code=0,stdout='prior',stderr=None)},
                archive_sha256=cp.file_digest(path/'state.tar.gz'))
    data.update(changes)
    (path/'manifest.json').write_text(json.dumps(data))
    return data


@pytest.mark.parametrize('change', [dict(status='pending'),dict(command_index=3),
    dict(archive_sha256='bad'),dict(completed_step=0),
    dict(receipts={'1':dict(return_code=1),'2':dict(return_code=0)})])
def test_reject_invalid_checkpoint(tmp_path, change):
    bundle(tmp_path/'c',**change)
    with pytest.raises(ValueError): cp.load_checkpoint(tmp_path/'c')


def test_skip_prefix_without_retraining_or_duplicate_source_pool(tmp_path, monkeypatch):
    bundle(tmp_path/'c')
    monkeypatch.setattr(cp,'protocol',lambda *a: {'test':1})
    class Env:
        index=0
        calls=[]
        async def exec(self, **kw):
            self.index += 1; self.calls.append(kw['command'])
            return NS(return_code=0,stdout='new',stderr=None)
    agent=NS(resume_checkpoint=tmp_path/'c',allow_legacy_recovery=False,
             enable_checkpoints=True,
             _display_plan={1:(0,'setup'),2:(1,'done'),3:(2,'train')},
             _checkpoint_boundaries=set())
    env=Env(); w=cp.CheckpointEnvironment(env,agent,'tasks')
    async def restore(): env.successful_sources=w.resume['successful_sources']
    monkeypatch.setattr(w,'_restore',restore)
    async def run():
        assert (await w.exec(command='setup')).stdout=='setup'
        assert (await w.exec(command='old train')).stdout=='prior'
        assert (await w.exec(command='new train')).stdout=='new'
    asyncio.run(run())
    assert env.calls==['setup','new train']
    assert env.index==3
    assert env.successful_sources==[{'task_id':1}]


def test_protocol_and_legacy_optin(tmp_path, monkeypatch):
    bundle(tmp_path/'c',recovery_kind='legacy_reconstructed')
    agent=NS(resume_checkpoint=tmp_path/'c',allow_legacy_recovery=False)
    monkeypatch.setattr(cp,'protocol',lambda *a: {'test':2})
    with pytest.raises(ValueError,match='protocol mismatch'):cp.CheckpointEnvironment(None,agent,'')
    monkeypatch.setattr(cp,'protocol',lambda *a: {'test':1})
    with pytest.raises(ValueError,match='explicit opt-in'):cp.CheckpointEnvironment(None,agent,'')


def test_resume_helper_requires_explicit_checkpoint(monkeypatch):
    spec=importlib.util.spec_from_file_location('prepare',Path('benchmark/scripts/prepare_continual_resume.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr('sys.argv', ['prepare_continual_resume.py', '--config', 'unused.yaml',
                                    '--run-name', 'new_run'])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2
    assert not hasattr(mod, 'legacy')


def test_checkpoint_boundaries_preserve_native_command_sequence(tmp_path, monkeypatch):
    from continual_regularization.baseline import MathBaseline
    monkeypatch.setenv('OPENAI_API_KEY','dummy');monkeypatch.setenv('OPENAI_BASE_URL','https://yibuapi.com/v1')
    agent=MathBaseline(logs_dir=tmp_path,model_name='openai/gpt-5.4-mini',task_tests_dir=tmp_path)
    commands=agent.create_run_agent_commands('task1\n---TASK_BREAK---\ntask2')
    assert [agent._display_plan[i] for i in sorted(agent._checkpoint_boundaries)]==[
        (1,'保存上下文'),(2,'确认技能写入')]
    assert len(commands)==12


def test_restore_redeploys_current_judge_after_archived_state(tmp_path, monkeypatch):
    bundle(tmp_path/'c')
    monkeypatch.setattr(cp, 'protocol', lambda *a: {'test': 1})
    class Env:
        calls = []
        async def upload_file(self, **kwargs):
            return None
        async def exec(self, **kwargs):
            self.calls.append(kwargs['command'])
            return NS(return_code=0, stdout='', stderr='')
    agent = NS(
        resume_checkpoint=tmp_path/'c', allow_legacy_recovery=False,
        logs_dir=tmp_path/'logs',
        _build_deploy_judge_command=lambda: 'DEPLOY_CURRENT_JUDGE',
    )
    agent.logs_dir.mkdir()
    env = Env()
    wrapper = cp.CheckpointEnvironment(env, agent, 'tasks')
    asyncio.run(wrapper._restore())
    assert env.calls[-1] == 'DEPLOY_CURRENT_JUDGE'
    provenance = json.loads((agent.logs_dir/'resume_provenance.json').read_text())
    assert provenance['judge_redeployed_after_restore'] is True
