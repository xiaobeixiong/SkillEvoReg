#!/usr/bin/env python3
"""Portable, explicit launcher around the unchanged native experiment engines."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
EVOL = ROOT/'benchmark/third_party/SkillEvolBench'

def shared_anchor_source(anchor):
    """Resolve a complete first-pass export and verify its persistent libraries."""
    anchor=Path(anchor).resolve()
    manifest_path=anchor/'anchor_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('A complete shared T1--T3 export with anchor_manifest.json is required')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('kind')!='complete_t1_t3_learning_anchor' or manifest.get('learning_records')!=90:
        raise ValueError('Expected a complete shared anchor with 90 first-pass learning records')
    source=Path(manifest.get('source_run','')).resolve()
    config_path=source/'config.json'
    if not config_path.is_file() or not (source/'reports/full_report.json').is_file():
        raise ValueError('The shared anchor source must be a completed first-pass run')
    config=json.loads(config_path.read_text())
    if config.get('learning_passes',1)!=1 or config.get('shared_learning_anchor_source'):
        raise ValueError('The shared anchor must originate from one learning pass, not a second-pass run')
    def snapshot(path):
        if not (path/'manifest.yaml').is_file() or not (path/'.git').exists():
            raise ValueError(f'Incomplete shared environment library: {path}')
        return {p.relative_to(path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in path.rglob('*') if p.is_file() and '.git' not in p.relative_to(path).parts
                and p.name!='.frozen'}
    for index in range(1,7):
        env_id=f'E{index}'
        if snapshot(source/'library'/env_id)!=snapshot(anchor/'library'/env_id):
            raise ValueError(f'Shared anchor differs from its first-pass source: {env_id}')
    return source

def environment(framework):
    env = os.environ.copy()
    paths=[ROOT, ROOT/'benchmark/third_party/SkillOpt', EVOL, ROOT/'benchmark/integrations']
    if framework in ('evol','continual'):
        paths.insert(0, ROOT/f'vendor/harbor_{"evol" if framework=="evol" else "continual"}/src')
    env['PYTHONPATH']=os.pathsep.join(map(str, paths))
    env['PYTHONUNBUFFERED']='1'
    env['PYTHONUTF8']='1'
    key=env.get('OPENAI_API_KEY') or env.get('YIBU_API_KEY')
    base=env.get('OPENAI_BASE_URL') or env.get('YIBU_API_BASE')
    if key:
        for name in ('OPENAI_API_KEY','YIBU_API_KEY','AZURE_OPENAI_API_KEY',
                     'OPTIMIZER_AZURE_OPENAI_API_KEY','TARGET_AZURE_OPENAI_API_KEY'):
            env[name]=key
    if base:
        for name in ('OPENAI_BASE_URL','OPENAI_API_BASE','YIBU_API_BASE',
                     'AZURE_OPENAI_ENDPOINT','OPTIMIZER_AZURE_OPENAI_ENDPOINT',
                     'TARGET_AZURE_OPENAI_ENDPOINT'):
            env[name]=base
    return env

def plan(args):
    manifest=json.loads((ROOT/'experiments.json').read_text())
    env=environment(args.framework)
    cwd=ROOT
    if args.framework=='skillopt':
        entry=manifest['experiments'][f'skillopt/{args.dataset}/{args.method}']
        command=[sys.executable,'-u','benchmark/scripts/run_skillopt.py',
                 'eval' if args.evaluate else 'train','--config',entry['config']]
        if args.evaluate:
            if not args.skill:
                raise ValueError('--skill is required for frozen evaluation')
            skill=Path(args.skill).resolve()
            if not skill.is_file():
                raise ValueError(f'Skill file is missing: {skill}')
            import yaml
            config=yaml.safe_load((ROOT/entry['config']).read_text())
            payload=dict(config=config,overrides=args.cfg_options,split=args.split,
                         endpoint=env.get('OPENAI_BASE_URL',''))
            protocol=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()[:16]
            skill_hash=hashlib.sha256(skill.read_bytes()).hexdigest()[:16]
            repeat=getattr(args,'eval_repeat',None)
            if repeat is not None and repeat < 1:
                raise ValueError('--eval-repeat must be positive')
            explicit=getattr(args,'eval_out_root',None)
            if any(option.split('=',1)[0] in ('env.out_root','out_root') for option in args.cfg_options):
                raise ValueError('Use --eval-out-root to choose an evaluation directory')
            observation=f'repeat_{repeat:02d}' if repeat is not None else 'observation_'+uuid.uuid4().hex
            output=(Path(explicit).resolve() if explicit else
                    ROOT/'benchmark/outputs/evaluations'/args.dataset/args.method/
                    skill_hash/protocol/args.split/observation)
            command += ['--skill',str(skill),'--split',args.split,'--out_root',str(output)]
        if args.cfg_options:
            command += ['--cfg-options',*args.cfg_options]
    elif args.framework=='evol':
        if args.method in ('baseline','twopass','skillevoreg'):
            if not args.anchor:
                raise ValueError('A complete shared T1--T3 export is required via --anchor')
            source=shared_anchor_source(args.anchor)
        if args.method=='baseline':
            if args.run_id:
                raise ValueError('The one-pass baseline is frozen evaluation; use --repeat, not --run-id')
            command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_evol.py'),
                     '--source',str(source),'--method','baseline','--block','all',
                     '--repeat',str(args.repeat),'--output-root',
                     str(EVOL/'workspace/frozen_evaluation'/source.name),'--execute']
            return command,ROOT,env
        if not args.run_id:
            raise ValueError('--run-id is required for learning runs')
        method='baseline' if args.method=='anchor' else args.method
        entry=manifest['experiments'][f'skillevolbench/{method}']
        command=[sys.executable,'-u','-m','scripts.run','--baseline',str(ROOT/entry['baseline']),
            '--strategy',str(ROOT/entry['strategy']), '--model-yaml',
            'configs/models/gpt-5.4-mini-yibu-medium.yaml', '--order-seed','A',
            '--run-id',args.run_id,'--workspace-root','workspace/runs']
        if args.method=='anchor':
            command+=['--anchor-source-only']
        if args.method in ('twopass','skillevoreg'):
            command+=['--learning-passes','2','--shared-learning-anchor-source',str(Path(args.anchor).resolve())]
        cwd=EVOL
    else:
        key=f'continual/{args.domain}/{args.method}/{args.phase}'
        entry=manifest['experiments'][key]
        config=ROOT/entry['config']
        if args.frozen_library or args.run_name or args.checkpoint:
            import yaml
            value=yaml.safe_load(config.read_text())
            if args.frozen_library:
                value['agents'][0]['kwargs']['frozen_library']=str(Path(args.frozen_library).resolve())
            if args.checkpoint:
                value['agents'][0]['kwargs']['resume_checkpoint']=str(Path(args.checkpoint).resolve())
            if args.run_name:
                value['job_name']=args.run_name
            config=ROOT/'benchmark/runtime/release_configs'/f'{value["job_name"]}.yaml'
            # Planning must be read-only. The prepared file is written only on execute.
            env['_RELEASE_PREPARED_CONFIG']=yaml.safe_dump(value,sort_keys=False)
        command=[sys.executable,'-u','-m','harbor.cli.main','run','-c',str(config)]
    return command,cwd,env

def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='framework',required=True)
    subs.add_parser('list')
    s=subs.add_parser('skillopt')
    s.add_argument('dataset',choices=['spreadsheet','searchqa','livemath'])
    s.add_argument('method',choices=['baseline','skillevoreg'])
    s.add_argument('--evaluate',action='store_true')
    s.add_argument('--skill')
    s.add_argument('--eval-repeat',type=int,help='Stable positive observation index for resumable evaluation')
    s.add_argument('--eval-out-root',help='Explicit evaluation directory; identity is checked before reuse')
    s.add_argument('--split',choices=['train','val','test'],default='test')
    s.add_argument('--cfg-options',nargs='+',default=[])
    e=subs.add_parser('evol')
    e.add_argument('method',choices=['anchor','baseline','twopass','skillevoreg'])
    e.add_argument('--anchor')
    e.add_argument('--run-id',help='Output run ID for anchor or second-pass learning')
    e.add_argument('--repeat',type=int,choices=[1,2],default=1,help='Frozen one-pass baseline observation')
    c=subs.add_parser('continual')
    c.add_argument('domain',choices=['math','finance','law','healthcare','office'])
    c.add_argument('method',choices=['baseline','regularization'])
    c.add_argument('phase',choices=['train','test'])
    c.add_argument('--frozen-library')
    c.add_argument('--run-name')
    c.add_argument('--checkpoint')
    for sub in (s,e,c):
        sub.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.framework=='list':
        for key in json.loads((ROOT/'experiments.json').read_text())['experiments']:
            print(key)
        return
    try:
        command,cwd,env=plan(args)
    except (KeyError, ValueError) as exc:
        p.error(str(exc))
    print('Working directory:',cwd)
    print('Command:',shlex.join(command),flush=True)
    if not args.execute:
        print('Plan only. Add --execute to start the workload.')
        return
    if not env.get('OPENAI_API_KEY') or not env.get('OPENAI_BASE_URL'):
        p.error('Export OPENAI_API_KEY and OPENAI_BASE_URL before execution')
    prepared=env.pop('_RELEASE_PREPARED_CONFIG',None)
    if prepared:
        target=Path(command[-1])
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_text()!=prepared:
            p.error('Prepared configuration already exists with different contents')
        target.write_text(prepared)
    if args.framework=='continual' and args.phase=='test':
        import yaml
        config=yaml.safe_load(Path(command[-1]).read_text())
        kwargs=config['agents'][0]['kwargs']
        if not kwargs.get('freeze_skill_updates') or not kwargs.get('prefer_skills'):
            p.error('Frozen evaluation requires unchanged library and skill-priority prompt')
        if not Path(kwargs['frozen_library']).exists():
            p.error('Frozen library is missing; export/stage the step-50 state first')
    raise SystemExit(subprocess.call(command,cwd=cwd,env=env))

if __name__=='__main__':
    main()
