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
    assert gpu.count('--threads 4')==2 and cpu.count('--threads 4')==1
    assert 'MKL_NUM_THREADS=4' in cpu
    receipt=json.loads((out/'inputs.json').read_text())
    assert len(receipt['reference_inputs'])==195
    assert len({r['image'] for r in receipt['reference_inputs']})==168
    with pytest.raises(ValueError,match='new evaluation workspace'):p.prepare(out,[],[])


def test_missing_smoke_produces_no_cost_number(tmp_path):
    from types import SimpleNamespace
    p=load('prepare');adapter=SimpleNamespace(ROOT=tmp_path,
        read_prompts=lambda *args:[{'id':'x','prompt':'p'}],stem_for=lambda *args:'x')
    result=p.costs({'output_root':str(tmp_path)},adapter)
    assert all(r['estimate'] is None and r['status']=='smoke_not_complete' for r in result.values())


def fixture_view(size=12):
    p=load('check_scores');prompts=[{'id':str(i),'prompt':'p'} for i in range(size)]
    records=[dict(method=m,prompt_id=str(i),path=f'/image/{m}/{i}.png',sidecar=f'/image/{m}/{i}.json',
                  config_hash='hash',res='4096x4096',seed=0) for m in p.METHODS for i in range(size)]
    receipt=dict(size=size,methods=sorted(p.METHODS),inputs=[dict(method=r['method'],prompt_id=r['prompt_id'],
        image=r['path'],sidecar=r['sidecar'],config_hash='hash') for r in records])
    return p,receipt,records,prompts


@pytest.mark.parametrize('size',[12,46])
def test_current_view_rejects_changed_extra_or_subset(size):
    p,receipt,records,prompts=fixture_view(size);p.validate_view(receipt,records,prompts)
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
    for name in p.BASE_REPORTS:(base/name).write_text('current')
    (base/'scored_metrics.json').write_text('old')
    with pytest.raises(ValueError,match='Partial immutable'):p.preserve_scores(tmp_path)
    (base/'scored_metrics.json').write_text('current');p.preserve_scores(tmp_path)
    assert (base/'scored_per_image.csv').read_text()=='current'


def test_gpu_paired_deltas_survive_live_cpu_report_replacement(tmp_path):
    p=load('check_scores');base=tmp_path/'base';base.mkdir()
    for name in p.BASE_REPORTS:(base/name).write_text('gpu-'+name)
    p.preserve_scores(tmp_path)
    (tmp_path/'GPU_SCORES_COMPLETE.json').write_text('{}')
    for name in p.BASE_REPORTS:(base/name).write_text('cpu-'+name)
    p.preserve_scores(tmp_path)
    for name in p.BASE_REPORTS:
        assert (base/('scored_'+name)).read_text()=='gpu-'+name


def test_reference_bytes_are_bound_even_with_same_size_and_timestamp(tmp_path):
    import os
    p=load('prepare');image=tmp_path/'reference.jpg';image.write_bytes(b'original')
    manifest=tmp_path/'references.jsonl'
    manifest.write_text(''.join(json.dumps(dict(id=i,path=image.name))+'\n' for i in ['a','b']))
    original=p.reference_inputs(manifest);stat=image.stat()
    assert len(original)==2 and original[0]['image_sha256']==original[1]['image_sha256']
    image.write_bytes(b'changed!');os.utime(image,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert p.reference_inputs(manifest)!=original
    manifest.write_text(json.dumps(dict(id='a',path=image.name))+'\n'+json.dumps(dict(id='a',path=image.name))+'\n')
    with pytest.raises(ValueError,match='Duplicate reference'):p.reference_inputs(manifest)


def test_input_barrier_rejects_changed_reference_before_scoring(tmp_path):
    p=load('prepare');checks=load('check_scores')
    image=tmp_path/'reference.jpg';image.write_bytes(b'original')
    references=tmp_path/'references.jsonl';references.write_text(json.dumps(dict(id='a',path=str(image)))+'\n')
    prompts=tmp_path/'prompts.jsonl';prompts.write_text('{}\n')
    receipt=dict(sources={},inputs=[],prompts_sha256=p.sha(prompts),references_sha256=p.sha(references),
        reference_inputs=p.reference_inputs(references))
    (tmp_path/'inputs.json').write_text(json.dumps(receipt))
    image.write_bytes(b'changed!')
    with pytest.raises(ValueError,match='Reference image bytes'):
        checks.main(tmp_path,inputs_only=True)
