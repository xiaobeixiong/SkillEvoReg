"""Run selected existing method tests in the release's own import context."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from run import ROOT, environment

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('framework',choices=['skillopt','skillevolbench','continual'])
    args=p.parse_args()
    framework='evol' if args.framework=='skillevolbench' else args.framework
    if args.framework=='skillopt':
        base=ROOT/'benchmark/third_party/SkillOpt/tests'
        files=['test_skillevoreg.py','test_gate.py','test_qa_math_adapter.py','test_json_utils.py']
    elif args.framework=='skillevolbench':
        base=ROOT/'benchmark/third_party/SkillEvolBench/tests'
        files=['test_online_regularization.py','test_serial_execution.py','test_frozen_evaluation.py']
    else:
        base=ROOT/'benchmark/tests'
        files=['test_continual_regularization_policy.py','test_continual_regularization_engine.py',
               'test_continual_regularization_transactions.py','test_continual_quick_validation.py',
               'test_continual_regularization_attacks.py','test_continual_regularization_boundary.py',
               'test_continual_dropout_copied_marker.py','test_continual_holdout_usage.py',
               'test_continual_judge_full_output.py','test_continual_per_task_sessions.py',
               'test_continual_checkpoint.py','test_continual_checkpoint_transport.py']
    env=environment(framework)
    env['PYTHONDONTWRITEBYTECODE']='1'
    if args.framework=='continual':
        import hashlib
        cache=Path(os.environ.get('TIKTOKEN_CACHE_DIR',str(ROOT/'.cache/tiktoken')))
        name=hashlib.sha1(b'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken').hexdigest()
        if not (cache/name).is_file():
            p.error('Prefetch tokenizer vocabulary once: python scripts/cache_tokenizer.py. Offline tests never fetch it automatically.')
        env['TIKTOKEN_CACHE_DIR']=str(cache)
    command=[sys.executable,'-m','pytest','-q','--tb=short','-o','addopts=','-p','no:cacheprovider',*[str(base/f) for f in files]]
    raise SystemExit(subprocess.call(command,cwd=ROOT,env=env))

if __name__=='__main__': main()
