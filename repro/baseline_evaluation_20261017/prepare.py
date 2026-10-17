#!/usr/bin/env python3
"""Validate exact baseline pairs and prepare isolated evaluation commands, no jobs."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[2]
DEPLOY=REPO/'results/evidence/continuation/baseline_smoke_deployment_20261017.json'
NEW=('sigma_dev_matched','hrdit_dev_author')
CONTROLS=('full','area','dype','sega_official','ultraimage','sca_nb')
PY='/work/09032/saini_2/vista/library/uv-envs/sca/bin/python'
EXTRA_PY='/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adapter():
    receipt=json.loads(DEPLOY.read_text());root=Path(receipt['frozen_root'])
    for name,digest in receipt['files'].items():
        if sha(root/name)!=digest:raise ValueError(f'Frozen generation file changed: {name}')
    spec=importlib.util.spec_from_file_location('frozen_baseline',root/'run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return receipt,module


def check_sample(image,method,row):
    image=Path(image);sidecar=image.with_suffix('.json')
    if not image.is_file() or not sidecar.is_file():raise ValueError(f'Missing pair: {image}')
    record=json.loads(sidecar.read_text());cfg=record['config']
    expected=dict(method=method,prompt_id=str(row['id']),prompt=row['prompt'],res='4096x4096',seed=0)
    if record.get('complete') is not True or any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError(f'Incomplete or wrong identity: {sidecar}')
    digest=sha(image)
    if record.get('png_sha256')!=digest:raise ValueError(f'PNG checksum mismatch: {image}')
    from PIL import Image
    with Image.open(image) as opened:
        if opened.size!=(4096,4096):raise ValueError(f'Wrong pixel dimensions: {image}')
    return dict(method=method,prompt_id=str(row['id']),image=str(image.resolve()),
        image_sha256=digest,sidecar=str(sidecar.resolve()),sidecar_sha256=sha(sidecar))


def costs(receipt,adapter):
    row=adapter.read_prompts(adapter.ROOT/'pilot46.jsonl',0,1)[0];result={}
    for method in NEW:
        path=Path(receipt['output_root'])/method/'4096x4096'/(adapter.stem_for(row['id'],0)+'.png')
        if not path.is_file() or not path.with_suffix('.json').is_file():
            result[method]={'status':'smoke_not_complete','estimate':None};continue
        args=SimpleNamespace(profile=method,prompts=adapter.ROOT/'pilot46.jsonl',seed=0)
        if not adapter.check_resume(path,path.with_suffix('.json'),adapter.make_config(args,row,adapter.verify_sources())):
            raise ValueError('Smoke provenance is incomplete')
        check_sample(path,method,row)
        value=json.loads(path.with_suffix('.json').read_text())['seconds']
        if not math.isfinite(value) or value<=0:raise ValueError('Invalid smoke time')
        result[method]=dict(status='one_observed_smoke',seconds=value,
            additional_after_first_smoke_gpu_hours={str(n):dict(central=(n-1)*value/3600,
            planning_range=[(n-1)*value/3600,2*(n-1)*value/3600]) for n in [12,46,191]},
            caveat='First image includes compilation, excludes loading. Evaluation/loading/retry cost unknown; this range is not a confidence interval.')
    return result


def gather(size,receipt,adapter):
    rows=adapter.read_prompts(adapter.ROOT/'pilot46.jsonl',0,size)
    selected=[];sources=adapter.verify_sources()
    for method in NEW:
        args=SimpleNamespace(profile=method,prompts=adapter.ROOT/'pilot46.jsonl',seed=0)
        for row in rows:
            image=Path(receipt['output_root'])/method/'4096x4096'/(adapter.stem_for(row['id'],0)+'.png')
            expected=adapter.make_config(args,row,sources)
            if not adapter.check_resume(image,image.with_suffix('.json'),expected):
                raise ValueError(f'Barrier closed: {method}/{row["id"]} has not completed')
            selected.append(check_sample(image,method,row))
    lookup={}
    for group in ['final4k_nb_base','final4k_full_base','e1_baselines_base','e1_ultraimage_base']:
        report=json.loads((REPO/'results/evidence/eval'/group/'metrics.json').read_text())
        for record in report['per_image']:
            key=(record['method'],record['prompt_id'],str(record['seed']))
            path=str(Path(record['path']).resolve())
            if key in lookup and lookup[key]!=path:raise ValueError(f'Ambiguous control path: {key}')
            lookup[key]=path
    for method in CONTROLS:
        for row in rows:selected.append(check_sample(lookup[method,str(row['id']),'0'],method,row))
    return rows,selected


def freeze_evaluators(destination):
    files=[REPO/x for x in ['repro/evaluate_corrected.py','repro/eval_extra_portable.py',
        'repro/extra_report_identity.py','scripts/eval_extra.py','scripts/paired_bootstrap.py',
        'repro/baseline_evaluation_20261017/check_scores.py']]
    files+=list((REPO/'repro/evaluator_snapshot').rglob('*.py'))
    manifest={}
    for source in files:
        relative=source.relative_to(REPO);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);manifest[str(relative)]=sha(source)
    return manifest


def command(python,script,args):return shlex.join(['uv','run','--no-project','--python',python,'python',str(script),*map(str,args)])


def prepare(workspace,rows,selected):
    workspace=workspace.resolve()
    if workspace.exists():raise ValueError('Use a new evaluation workspace; existing results are preserved')
    workspace.mkdir(parents=True);code=workspace/'code';manifest=freeze_evaluators(code)
    view=workspace/'samples';view.mkdir()
    for item in selected:
        folder=view/item['method'];folder.mkdir(exist_ok=True)
        for kind in ['image','sidecar']:
            source=Path(item[kind]);(folder/source.name).symlink_to(source)
    prompts=workspace/'prompts.jsonl';prompts.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    references=workspace/'references.jsonl';shutil.copyfile(REPO/'results/evidence/prompts/aesthetic4k_refs.jsonl',references)
    receipt=dict(size=len(rows),methods=list(NEW+CONTROLS),inputs=selected,sources=manifest,
        prompts_sha256=sha(prompts),references_sha256=sha(references),status='inputs_complete_scoring_not_run')
    (workspace/'inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')
    common=['--root',view,'--prompts',prompts,'--reference-method','sca_nb','--res','4096x4096']
    base=common+['--references',references,'--reference-clusters','images','--out',workspace/'base']
    extra=common+['--crope-root',code/'repro/evaluator_snapshot','--out',workspace/'extra']
    header='#!/bin/bash\nset -euo pipefail\nexport HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n'
    check=command(PY,code/'repro/baseline_evaluation_20261017/check_scores.py',[workspace])
    gpu=[check+' --inputs-only',command(PY,code/'repro/evaluate_corrected.py',base+['--metrics','clip,pickscore,musiq,fid,kid,patch_fid,patch_kid','--device','cuda','--bootstrap','0']),
        command(EXTRA_PY,code/'repro/eval_extra_portable.py',extra+['--metrics','ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa','--device','cuda','--bootstrap','0']),check]
    cpu=[check,command(PY,code/'repro/evaluate_corrected.py',base+['--metrics','fid,kid,patch_fid,patch_kid','--device','cpu','--bootstrap','1000'])]
    for method in NEW:
        cpu.append(command(PY,code/'scripts/paired_bootstrap.py',['--method',method,'--reference','sca_nb','--draws','2000','--out',workspace/(method+'_paired.csv'),workspace/'base/scored_per_image.csv',workspace/'extra/extra_per_image.csv']))
    for name,lines in [('score_gpu.sh',gpu),('intervals_cpu.sh',cpu)]:
        (workspace/name).write_text(header+'\n'.join(lines)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',type=int,choices=[12,46],default=12)
    p.add_argument('--workspace',type=Path);p.add_argument('--cost-only',action='store_true');a=p.parse_args()
    deployment,adapter=load_adapter()
    if a.cost_only:print(json.dumps(costs(deployment,adapter),indent=2))
    else:
        if a.workspace is None:p.error('--workspace required')
        rows,samples=gather(a.size,deployment,adapter)
        result=prepare(a.workspace,rows,samples)
        print(json.dumps(dict(status=result['status'],samples=len(samples),workspace=str(a.workspace))))
