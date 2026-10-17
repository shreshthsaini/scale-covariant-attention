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


def check_csv(path,expected,metrics,paths=None):
    with path.open() as f:rows=list(csv.DictReader(f))
    keys=[(r['method'],r['prompt_id']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError(f'Wrong paired coverage: {path}')
    for row in rows:
        if paths is not None and str(Path(row['path']).resolve())!=paths[row['method'],row['prompt_id']]:
            raise ValueError('CSV image path differs from input receipt')
        if row['res']!='4096x4096' or str(row['seed'])!='0':raise ValueError('Wrong resolution/seed')
        for metric in metrics:
            if not math.isfinite(float(row[metric])):raise ValueError(f'Missing/nonfinite {metric}')


METHODS={'sigma_dev_matched','hrdit_dev_author','full','area','dype','sega_official','ultraimage','sca_nb'}


def validate_view(receipt,records,prompts):
    size=receipt['size'];expected={(r['method'],r['prompt_id']):r for r in receipt['inputs']}
    ids={str(r['id']) for r in prompts}
    if size not in {12,46} or set(receipt['methods'])!=METHODS or len(ids)!=size:
        raise ValueError('Expected exact 12/46 prompt, eight-method protocol')
    if len(expected)!=8*size or len(receipt['inputs'])!=8*size or set(expected)!={(m,i) for m in METHODS for i in ids}:
        raise ValueError('Receipt lacks exact paired coverage')
    actual={(r['method'],r['prompt_id']):r for r in records}
    if len(actual)!=len(records) or set(actual)!=set(expected):raise ValueError('Discovered sample view differs')
    for key,record in actual.items():
        original=expected[key]
        if (record['path']!=original['image'] or record['sidecar']!=original['sidecar'] or
            record['config_hash']!=original['config_hash'] or record['res']!='4096x4096' or str(record['seed'])!='0'):
            raise ValueError('Discovered path/config differs from receipt')
    return expected


def scoring_paths(workspace):
    marker=workspace/'GPU_SCORES_COMPLETE.json'
    if not marker.exists():return workspace/'base/metrics.json',workspace/'base/per_image.csv'
    old=json.loads(marker.read_text())
    if sha(workspace/'inputs.json')!=old['inputs_sha256']:raise ValueError('Completed input receipt changed')
    for name,value in old['reports'].items():
        if sha(workspace/name)!=value:raise ValueError('Immutable scoring report changed')
    return workspace/'base/scored_metrics.json',workspace/'base/scored_per_image.csv'


def preserve_scores(workspace):
    completed=(workspace/'GPU_SCORES_COMPLETE.json').exists()
    for name in ['metrics.json','per_image.csv','summary.csv']:
        source=workspace/'base'/name;target=source.with_name('scored_'+name)
        if target.exists():
            if not completed and sha(target)!=sha(source):
                raise ValueError('Partial immutable snapshot differs from current GPU report')
        else:
            if completed:raise ValueError('Completed immutable snapshot is missing')
            with target.open('xb') as f:f.write(source.read_bytes())


def main(workspace,inputs_only=False):
    receipt=json.loads((workspace/'inputs.json').read_text());code=workspace/'code'
    for name,value in receipt['sources'].items():
        if sha(code/name)!=value:raise ValueError(f'Evaluator source changed: {name}')
    for name in ['prompts','references']:
        if sha(workspace/(name+'.jsonl'))!=receipt[name+'_sha256']:raise ValueError(f'{name} changed')
    for row in receipt['inputs']:
        for kind in ['image','sidecar']:
            if sha(row[kind])!=row[kind+'_sha256']:raise ValueError(f'Input changed: {row[kind]}')
    sys.path.insert(0,str(code/'repro/evaluator_snapshot'))
    sys.path.insert(0,str(code/'repro'))
    from crope.metrics.io import digest,file_signature,load_references,discover_samples,read_jsonl
    from extra_report_identity import identities,REPORT_FILES
    records=discover_samples(workspace/'samples',read_jsonl(workspace/'prompts.jsonl'),'4096x4096')
    original=validate_view(receipt,records,read_jsonl(workspace/'prompts.jsonl'))
    if inputs_only:return
    report_path,csv_path=scoring_paths(workspace)
    paths={key:row['image'] for key,row in original.items()}
    expected={(x['method'],x['prompt_id']) for x in receipt['inputs']}
    check_csv(csv_path,expected,['clip','pickscore','musiq','musiq_crop1024'],paths)
    check_csv(workspace/'extra/extra_per_image.csv',expected,['ir','aesthetic','hpsv2','hpsv21','clipiqa','arniqa'],paths)
    report=json.loads(report_path.read_text())
    extra=json.loads((workspace/'extra/extra_metrics.json').read_text())
    if not report.get('complete') or report.get('errors') or not extra.get('complete') or extra.get('errors'):
        raise ValueError('Scoring reports are incomplete')
    for item in [report,extra]:
        if item['config']['bootstrap']!=0 or item['config']['device']!='cuda':
            raise ValueError('Expected GPU point-only scoring reports')
    validate_view(receipt,report['per_image'],read_jsonl(workspace/'prompts.jsonl'))
    extra_identity=json.loads((workspace/'extra/extra_input_identities.json').read_text())
    if extra_identity.get('phase')!='complete' or extra_identity.get('identities')!=identities(records):
        raise ValueError('Extra scoring input identity barrier failed')
    if set(extra_identity.get('report_files',{}))!=set(REPORT_FILES):raise ValueError('Extra report hashes missing')
    for name,value in extra_identity['report_files'].items():
        if sha(workspace/'extra'/name)!=value:raise ValueError('Extra report changed')
    from crope.metrics.evaluate import package_versions,cached_model_revisions
    if report['versions']!=package_versions() or report['model_revisions']!=cached_model_revisions():
        raise ValueError('CPU cache consumer environment differs from GPU extraction')
    settings=dict(schema=report['metric_schema'],crop_seed=report['config']['crop_seed'],
        crops=report['config']['crops'],versions=report['versions'],model_revisions=report['model_revisions'])
    import numpy as np
    files={}
    for record in records+load_references(workspace/'references.jsonl'):
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
    preserve_scores(workspace)
    marker=dict(inputs_sha256=sha(workspace/'inputs.json'),features=files,
        reports={name:sha(workspace/name) for name in ['base/scored_metrics.json','base/scored_per_image.csv',
        'base/scored_summary.csv','extra/extra_metrics.json','extra/extra_per_image.csv','extra/extra_input_identities.json']})
    path=workspace/'GPU_SCORES_COMPLETE.json'
    if path.exists() and json.loads(path.read_text())!=marker:
        raise ValueError('Previously completed score/cache hashes changed')
    if not path.exists():
        with path.open('x') as f:f.write(json.dumps(marker,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('workspace',type=Path)
    p.add_argument('--inputs-only',action='store_true');a=p.parse_args();main(a.workspace,a.inputs_only)
    print('input barrier passed' if a.inputs_only else 'GPU score and feature-cache barrier passed')
