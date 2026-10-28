"""Supply new source roots to the original frozen SkillEvolBench evaluator."""
import argparse
import asyncio
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from run import ROOT, EVOL, environment

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--method',choices=['baseline','noreg','regularization'],required=True)
    p.add_argument('--block',choices=['t1-t3','t4-t6','all'],required=True)
    p.add_argument('--repeat',type=int,choices=[1,2],default=2)
    p.add_argument('--output-root',type=Path,default=EVOL/'workspace/frozen_evaluation')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    expected=environment('evol')
    if os.environ.get('_RELEASE_EVOL_EVAL_CHILD')!='1':
        expected['_RELEASE_EVOL_EVAL_CHILD']='1'
        raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env=expected,cwd=ROOT))
    location=EVOL/'scripts/evaluate_frozen.py'
    spec=importlib.util.spec_from_file_location('release_frozen_matrix',location)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    blocks=('t4-t6','t1-t3') if args.block=='all' else (args.block,)
    targets=[module.EvalTarget(method=args.method,block=block,source_root=args.source.resolve(),
                              repeat_index=args.repeat,output_root=args.output_root.resolve())
             for block in blocks]
    if args.execute:
        if not expected.get('OPENAI_API_KEY') or not expected.get('OPENAI_BASE_URL'):
            p.error('Export API credentials before execution')
        for target in targets:
            asyncio.run(module._run_target(target))
    else:
        for target in targets:
            module._check_target(target)

if __name__=='__main__': main()
