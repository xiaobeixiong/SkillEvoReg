"""Stage declared data inputs from an existing project; never overwrite data."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA_PATHS=[
    'benchmark/data/prepared/spreadsheetbench_verified_400',
    'benchmark/data/splits/spreadsheetbench_verified_400/skillopt_official_expanded_100_100_200_v1',
    'benchmark/third_party/SkillOpt/data/searchqa_split',
    'benchmark/third_party/SkillOpt/data/livemathematicianbench_split_100_80_217',
]
TASKS=['math-100','math-100-holdout50','finance-econ-100-train50','finance-econ-100-holdout50',
       'law-100-train50','law-100-holdout50','healthcare-100-train50','healthcare-100-holdout50',
       'office-100-train50','office-100-holdout50']
FROZEN=['frozen_libraries','frozen_finance_libraries','frozen_law_libraries',
        'frozen_healthcare_libraries','frozen_office_libraries']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-project',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--with-frozen-libraries',action='store_true')
    args=p.parse_args()
    source=args.from_project.resolve()
    if source==ROOT:
        p.error('Source and release must differ')
    paths=DATA_PATHS+[f'benchmark/runtime/continualskillbench/tasks/{x}' for x in TASKS]
    if args.with_frozen_libraries:
        paths += [f'benchmark/runtime/continualskillbench/{x}' for x in FROZEN]
    actions=[]
    for relative in paths:
        base=source/relative
        if not base.is_dir():
            actions.append(dict(path=relative,status='missing_source'))
            continue
        files=[x for x in base.rglob('*') if x.is_file() and not x.is_symlink() and
               not any(part in {'.git','__pycache__','.DS_Store'} for part in x.relative_to(base).parts)]
        actions.append(dict(path=relative,files=len(files),bytes=sum(x.stat().st_size for x in files),status='planned' if args.dry_run else 'staged'))
        if args.dry_run:
            continue
        for path in files:
            target=ROOT/relative/path.relative_to(base)
            raw=path.read_bytes()
            if path.suffix in {'.json','.yaml','.yml','.toml'}:
                try:
                    raw=raw.decode().replace(str(source),str(ROOT)).encode()
                except UnicodeDecodeError:
                    pass
            if target.exists():
                if target.read_bytes()!=raw:
                    raise RuntimeError(f'Refusing to overwrite different data: {target.relative_to(ROOT)}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw)
                shutil.copymode(path,target)
    if not args.dry_run:
        out=ROOT/'benchmark/runtime/data_staging_report.json'
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(actions,indent=2)+'\n')
    print(json.dumps(actions,indent=2))

if __name__=='__main__': main()
