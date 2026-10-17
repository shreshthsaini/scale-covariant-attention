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
