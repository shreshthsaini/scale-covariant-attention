#!/usr/bin/env python3
"""Input, metric coverage and feature-cache barrier; contains no metric estimator."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_csv(path,expected,metrics):
    with path.open() as f:rows=list(csv.DictReader(f))
    keys=[(r['method'],r['prompt_id']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError(f'Wrong paired coverage: {path}')
    for row in rows:
        if row['res']!='4096x4096' or str(row['seed'])!='0':raise ValueError('Wrong resolution/seed')
        for metric in metrics:
            if not math.isfinite(float(row[metric])):raise ValueError(f'Missing/nonfinite {metric}')


def main(workspace,inputs_only=False):
    receipt=json.loads((workspace/'inputs.json').read_text());code=workspace/'code'
    for name,value in receipt['sources'].items():
        if sha(code/name)!=value:raise ValueError(f'Evaluator source changed: {name}')
    for name in ['prompts','references']:
        if sha(workspace/(name+'.jsonl'))!=receipt[name+'_sha256']:raise ValueError(f'{name} changed')
    for row in receipt['inputs']:
        for kind in ['image','sidecar']:
            if sha(row[kind])!=row[kind+'_sha256']:raise ValueError(f'Input changed: {row[kind]}')
    if inputs_only:return
    expected={(x['method'],x['prompt_id']) for x in receipt['inputs']}
    check_csv(workspace/'base/per_image.csv',expected,['clip','pickscore','musiq','musiq_crop1024'])
    check_csv(workspace/'extra/extra_per_image.csv',expected,['ir','aesthetic','hpsv2','hpsv21','clipiqa','arniqa'])
    report=json.loads((workspace/'base/metrics.json').read_text())
    extra=json.loads((workspace/'extra/extra_metrics.json').read_text())
    if not report.get('complete') or report.get('errors') or not extra.get('complete') or extra.get('errors'):
        raise ValueError('Scoring reports are incomplete')
    if report['config']['bootstrap']!=0 or report['config']['device']!='cuda':
        raise ValueError('Expected GPU point-only scoring report')
    sys.path.insert(0,str(code/'repro/evaluator_snapshot'))
    from crope.metrics.io import digest,file_signature,load_references
    from crope.metrics.evaluate import package_versions,cached_model_revisions
    if report['versions']!=package_versions() or report['model_revisions']!=cached_model_revisions():
        raise ValueError('CPU cache consumer environment differs from GPU extraction')
    settings=dict(schema=report['metric_schema'],crop_seed=report['config']['crop_seed'],
        crops=report['config']['crops'],versions=report['versions'],model_revisions=report['model_revisions'])
    import numpy as np
    files={}
    for record in report['per_image']+load_references(workspace/'references.jsonl'):
        for patch in [False,True]:
            key=digest(dict(file=file_signature(record['path']),prompt=record.get('prompt'),
                prompt_id=record['prompt_id'],metric='inception_features',settings=settings,
                source_config_hash=record.get('config_hash'),extra={'patch':patch}))
            path=workspace/'base/features'/(key+'.npz')
            with np.load(path,allow_pickle=False) as saved:
                values=saved['features']
                if values.shape!=(4 if patch else 1,2048) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid Inception feature cache: {path}')
            files[str(path)]=sha(path)
    for name in ['metrics.json','per_image.csv','summary.csv']:
        source=workspace/'base'/name;target=source.with_name('scored_'+name)
        if not target.exists():
            with target.open('xb') as f:f.write(source.read_bytes())
    marker=dict(inputs_sha256=sha(workspace/'inputs.json'),features=files,
        reports={name:sha(workspace/name) for name in ['base/scored_metrics.json','base/scored_per_image.csv',
        'base/scored_summary.csv','extra/extra_metrics.json','extra/extra_per_image.csv']})
    path=workspace/'GPU_SCORES_COMPLETE.json'
    if path.exists() and json.loads(path.read_text())!=marker:
        raise ValueError('Previously completed score/cache hashes changed')
    if not path.exists():
        with path.open('x') as f:f.write(json.dumps(marker,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('workspace',type=Path)
    p.add_argument('--inputs-only',action='store_true');a=p.parse_args();main(a.workspace,a.inputs_only)
    print('input barrier passed' if a.inputs_only else 'GPU score and feature-cache barrier passed')
