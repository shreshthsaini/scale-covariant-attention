#!/usr/bin/env python3
"""Prepare spool candidates without submitting jobs or touching a live queue."""
import argparse
import json
import math
from pathlib import Path
import shlex
from types import SimpleNamespace

import run

PYTHON='/work/09032/saini_2/vista/library/uv-envs/sca/bin/python'
EXTRA='/work/09032/saini_2/vista/library/uv-envs/sca-baseline-expansion-20261017/lib/python3.11/site-packages'
HUB=Path('/scratch/09032/saini_2/tmp/hf/hub')
PROFILES=('sigma_dev_matched','hrdit_dev_author')


def required_files():
    return [run.ROOT/x for x in ['run.py','sources.json','model_manifest.json','dependency_overlay.json','pilot46.jsonl']]+[
        p for p in sorted((run.ROOT/'vendor').rglob('*')) if p.is_file() and '__pycache__' not in p.parts]


def verify_freeze(freeze):
    for source in required_files():
        target=freeze/source.relative_to(run.ROOT)
        if not target.is_file() or run.sha(target)!=run.sha(source):
            raise ValueError(f'Freeze missing or different: {target}')


def smoke_ready(profile,out_root):
    row=run.read_prompts(run.ROOT/'pilot46.jsonl',0,1)[0]
    args=SimpleNamespace(profile=profile,prompts=run.ROOT/'pilot46.jsonl',seed=0)
    config=run.make_config(args,row,run.verify_sources())
    stem=run.stem_for(row['id'],0)
    out=out_root/profile/'4096x4096'
    image=out/(stem+'.png');sidecar=out/(stem+'.json')
    if not run.check_resume(image,sidecar,config):
        raise ValueError(f'{profile} GPU smoke is incomplete')
    record=json.loads(sidecar.read_text())
    seconds=record.get('wall_seconds',0)
    if not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('Smoke lacks a positive measured pipeline duration')
    return seconds


def task_text(freeze,out_root,profile,start,end,minimum):
    checks='\n'.join(f'{run.sha(p)}  {freeze/p.relative_to(run.ROOT)}' for p in required_files())
    command=['uv','run','--no-project','--python',PYTHON,'python',str(freeze/'run.py'),
        '--profile',profile,'--prompts',str(freeze/'pilot46.jsonl'),'--start',str(start),'--end',str(end),
        '--seed','0','--out',str(out_root),'--dev-snapshot',str(HUB/'models--black-forest-labs--FLUX.1-dev/snapshots'/run.DEV),
        '--mode','generate']
    if profile=='hrdit_dev_author': command+=['--extra-site',EXTRA]
    if start>0: command+=['--wandb']
    return ('#!/bin/bash\n'+f'# SCA_MIN_SECONDS={minimum}\nset -euo pipefail\n'
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1\n'
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false\n'
        'export HF_HOME=/scratch/09032/saini_2/tmp/hf\n'
        'export UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache\n'
        'export TORCHINDUCTOR_CACHE_DIR=/scratch/09032/saini_2/sca/cache/baseline_expansion_20261017/inductor\n'
        'export TRITON_CACHE_DIR=/scratch/09032/saini_2/sca/cache/baseline_expansion_20261017/triton\n'
        'export CC=/usr/bin/gcc CXX=/usr/bin/g++\n'
        '"$CC" --version | head -n 1\n'
        'printf "int main(void){return 0;}\\n" | "$CC" -x c -Wno-psabi -fsyntax-only -\n'
        "sha256sum -c <<'SCA_BASELINE_HASHES'\n"+checks+'\nSCA_BASELINE_HASHES\n'+shlex.join(command)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--tasks-dir',type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','pilot12'],default='smoke')
    a=p.parse_args()
    freeze=a.frozen_root.resolve();out=a.out_root.resolve()
    verify_freeze(freeze)
    durations={x:smoke_ready(x,out) for x in PROFILES} if a.stage=='pilot12' else {}
    a.tasks_dir.mkdir(parents=True,exist_ok=True)
    created=[]
    for profile in PROFILES:
        ranges=[(0,1)] if a.stage=='smoke' else [(1,5),(5,9),(9,12)]
        for start,end in ranges:
            minimum=3600 if a.stage=='smoke' else max(2700,math.ceil(durations[profile]*2*(end-start)+300))
            path=a.tasks_dir/f'a_b_{a.stage}_{profile}_{start:03d}_{end:03d}.sh'
            with path.open('x') as f:f.write(task_text(freeze,out,profile,start,end,minimum))
            created.append(dict(path=str(path),sha256=run.sha(path),minimum_seconds=minimum))
    print(json.dumps(dict(status='candidates_only_not_enqueued',stage=a.stage,tasks=created),indent=2))


if __name__=='__main__':main()
