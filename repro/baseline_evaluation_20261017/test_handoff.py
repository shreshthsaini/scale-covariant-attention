import csv
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

HERE=Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_sample_hash_and_identity(tmp_path):
    from PIL import Image
    p=load('prepare');image=tmp_path/'sample.png';Image.new('RGB',(4096,4096)).save(image)
    row={'id':'x','prompt':'text'};config=dict(method='m',prompt_id='x',prompt='text',res='4096x4096',seed=0)
    record=dict(complete=True,config=config,png_sha256=p.sha(image));sidecar=image.with_suffix('.json')
    sidecar.write_text(json.dumps(record));assert p.check_sample(image,'m',row)['prompt_id']=='x'
    record['png_sha256']='bad';sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='checksum'):p.check_sample(image,'m',row)
    record['png_sha256']=p.sha(image);record['config']['seed']=1;sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='identity'):p.check_sample(image,'m',row)


def test_csv_requires_exact_pairs_and_finite_scores(tmp_path):
    p=load('check_scores');path=tmp_path/'scores.csv'
    path.write_text('method,prompt_id,res,seed,x\na,p,4096x4096,0,1.2\n')
    p.check_csv(path,{('a','p')},['x'])
    with pytest.raises(ValueError):p.check_csv(path,{('b','p')},['x'])
    path.write_text('method,prompt_id,res,seed,x\na,p,4096x4096,0,nan\n')
    with pytest.raises(ValueError):p.check_csv(path,{('a','p')},['x'])


def test_generated_scripts_use_split_compute_and_preserved_scalars(tmp_path):
    p=load('prepare');out=tmp_path/'new'
    p.prepare(out,[{'id':'fixture','prompt':'fixture'}],[])
    for name in ['score_gpu.sh','intervals_cpu.sh']:
        subprocess.run(['bash','-n',str(out/name)],check=True)
        assert 'sbatch' not in (out/name).read_text()
    gpu=(out/'score_gpu.sh').read_text();cpu=(out/'intervals_cpu.sh').read_text()
    assert '--bootstrap 0' in gpu and '--device cuda' in gpu
    assert '--bootstrap 1000' in cpu and '--reference-clusters images' in cpu
    assert '--draws 2000' in cpu and 'scored_per_image.csv' in cpu
    with pytest.raises(ValueError,match='new evaluation workspace'):p.prepare(out,[],[])


def test_missing_smoke_produces_no_cost_number(tmp_path):
    from types import SimpleNamespace
    p=load('prepare');adapter=SimpleNamespace(ROOT=tmp_path,
        read_prompts=lambda *args:[{'id':'x','prompt':'p'}],stem_for=lambda *args:'x')
    result=p.costs({'output_root':str(tmp_path)},adapter)
    assert all(r['estimate'] is None and r['status']=='smoke_not_complete' for r in result.values())


def fixture_view():
    p=load('check_scores');prompts=[{'id':str(i),'prompt':'p'} for i in range(12)]
    records=[dict(method=m,prompt_id=str(i),path=f'/image/{m}/{i}.png',sidecar=f'/image/{m}/{i}.json',
                  config_hash='hash',res='4096x4096',seed=0) for m in p.METHODS for i in range(12)]
    receipt=dict(size=12,methods=sorted(p.METHODS),inputs=[dict(method=r['method'],prompt_id=r['prompt_id'],
        image=r['path'],sidecar=r['sidecar'],config_hash='hash') for r in records])
    return p,receipt,records,prompts


def test_current_view_rejects_changed_extra_or_subset():
    p,receipt,records,prompts=fixture_view();p.validate_view(receipt,records,prompts)
    changed=[dict(r) for r in records];changed[0]['path']='/other/image.png'
    with pytest.raises(ValueError,match='path/config'):p.validate_view(receipt,changed,prompts)
    changed[0]=dict(records[0],config_hash='different')
    with pytest.raises(ValueError,match='path/config'):p.validate_view(receipt,changed,prompts)
    with pytest.raises(ValueError,match='view'):p.validate_view(receipt,records+[records[0]],prompts)
    receipt['inputs']=receipt['inputs'][:-1]
    with pytest.raises(ValueError,match='coverage'):p.validate_view(receipt,records,prompts)


def test_cpu_rerun_uses_immutable_scored_report(tmp_path):
    p=load('check_scores');(tmp_path/'base').mkdir();(tmp_path/'inputs.json').write_text('{}')
    scored=tmp_path/'base/scored_metrics.json';scored.write_text('{"device":"cuda"}')
    live=tmp_path/'base/metrics.json';live.write_text('{"device":"cpu"}')
    marker=dict(inputs_sha256=p.sha(tmp_path/'inputs.json'),reports={'base/scored_metrics.json':p.sha(scored)})
    (tmp_path/'GPU_SCORES_COMPLETE.json').write_text(json.dumps(marker))
    assert p.scoring_paths(tmp_path)[0]==scored
    scored.write_text('changed')
    with pytest.raises(ValueError,match='Immutable'):p.scoring_paths(tmp_path)


def test_cache_only_constructor_and_metric_guards():
    from types import SimpleNamespace
    p=load('cpu_cache_only');args=SimpleNamespace(device='cpu',metrics='fid,kid,patch_fid,patch_kid',
        bootstrap=1000,reference_clusters='images');p.validate_args(args)
    with pytest.raises(RuntimeError,match='refuses Inception'):p.deny_encoder('cpu')
    args.metrics='fid,musiq'
    with pytest.raises(ValueError,match='distribution'):p.validate_args(args)
    args.metrics='fid';args.device='cuda'
    with pytest.raises(ValueError):p.validate_args(args)


def test_csv_rejects_changed_image_path(tmp_path):
    p=load('check_scores');path=tmp_path/'scores.csv'
    path.write_text('method,prompt_id,res,seed,x,path\na,p,4096x4096,0,1.2,/wrong.png\n')
    with pytest.raises(ValueError,match='image path'):p.check_csv(path,{('a','p')},['x'],{('a','p'):'/right.png'})


def test_stale_partial_snapshot_refused(tmp_path):
    p=load('check_scores');base=tmp_path/'base';base.mkdir()
    for name in ['metrics.json','per_image.csv','summary.csv']:(base/name).write_text('current')
    (base/'scored_metrics.json').write_text('old')
    with pytest.raises(ValueError,match='Partial immutable'):p.preserve_scores(tmp_path)
    (base/'scored_metrics.json').write_text('current');p.preserve_scores(tmp_path)
    assert (base/'scored_per_image.csv').read_text()=='current'
