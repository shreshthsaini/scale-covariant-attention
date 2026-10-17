"""CPU safety/provenance tests; these do not establish GPU baseline quality."""
import importlib.util
import json
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('baseline_adapter',HERE/'run.py')
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_pinned_sources():
    receipt=adapter.verify_sources()
    assert set(receipt['repositories'])=={'hrdit','sigma'}
    assert all(len(row['commit'])==40 for row in receipt['repositories'].values())


@pytest.mark.parametrize('rows',[
    [{'id':'x','prompt':'a'},{'id':'x','prompt':'b'}],
    [{'id':'x','prompt':''}],
    [],
])
def test_bad_prompts(tmp_path,rows):
    path=tmp_path/'p.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError):adapter.read_prompts(path)


def test_seed_agrees_with_existing_campaign():
    import sys
    sys.path.insert(0,str(HERE.parents[1]))
    from sca.runtime import image_seed,image_stem
    for prompt_id in ['abc','../bad path','a4k_eval_4096_123']:
        assert adapter.seed_for(prompt_id,0)==image_seed(prompt_id,0)
        assert adapter.stem_for(prompt_id,0)==image_stem(prompt_id,0)
        assert '/' not in adapter.stem_for(prompt_id,0)


def test_partial_and_mismatched_output_fail_closed(tmp_path):
    image=tmp_path/'x.png';sidecar=tmp_path/'x.json'
    assert not adapter.check_resume(image,sidecar,{'x':1})
    image.write_bytes(b'partial')
    with pytest.raises(ValueError,match='Partial'):adapter.check_resume(image,sidecar,{'x':1})
    sidecar.write_text(json.dumps({'config':{'x':2},'complete':True,'png_sha256':adapter.sha(image)}))
    with pytest.raises(ValueError,match='provenance'):adapter.check_resume(image,sidecar,{'x':1})


def test_resume_checks_pixel_dimensions_and_hash(tmp_path):
    from PIL import Image
    image=tmp_path/'x.png';sidecar=tmp_path/'x.json';config={'x':1}
    Image.new('RGB',(16,16)).save(image)
    sidecar.write_text(json.dumps({'config':config,'complete':True,'png_sha256':adapter.sha(image)}))
    with pytest.raises(ValueError,match='dimensions'):adapter.check_resume(image,sidecar,config)
    Image.new('RGB',(4096,4096)).save(image)
    sidecar.write_text(json.dumps({'config':config,'complete':True,'png_sha256':adapter.sha(image)}))
    assert adapter.check_resume(image,sidecar,config)
    image.write_bytes(image.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='provenance'):adapter.check_resume(image,sidecar,config)


def test_profiles_do_not_hide_multistage_budget():
    assert adapter.PROFILES['hrdit_dev_author']['steps']==[30,17,10]
    assert adapter.PROFILES['sigma_dev_matched']['mu']=='max_shift'
    assert adapter.PROFILES['sigma_krea_native']['mu']=='upstream_target_length_cap'


def load_preparer():
    import sys
    sys.path.insert(0,str(HERE))
    spec=importlib.util.spec_from_file_location('prepare_baselines',HERE/'prepare_tasks.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_pilot_gate_requires_completed_smoke(tmp_path):
    with pytest.raises(ValueError,match='incomplete'):
        load_preparer().smoke_ready('sigma_dev_matched',tmp_path)


def test_task_has_frozen_sources_and_scoped_command(tmp_path):
    import subprocess
    text=load_preparer().task_text(Path('/scratch/frozen'),Path('/scratch/new_outputs'),
        'hrdit_dev_author',0,1,3600)
    path=tmp_path/'task.sh';path.write_text(text)
    subprocess.run(['bash','-n',str(path)],check=True)
    assert '# SCA_MIN_SECONDS=3600' in text
    assert 'export CC=/usr/bin/gcc' in text
    assert 'sha256sum -c' in text
    assert '--start 0 --end 1' in text
    assert '--mode generate' in text
    assert '--extra-site' in text
    assert 'sbatch' not in text and 'scancel' not in text


def test_freeze_rejects_missing_source(tmp_path):
    with pytest.raises(ValueError,match='Freeze missing'):
        load_preparer().verify_freeze(tmp_path)


def test_worklist_is_disjoint_from_calibration():
    root=HERE.parents[1]/'results/evidence/prompts'
    dev={str(x['id']) for x in adapter.read_prompts(root/'dev4.jsonl')}
    pilot=adapter.read_prompts(HERE/'pilot46.jsonl')
    assert len(pilot)==46
    assert not dev&{str(x['id']) for x in pilot}


def test_hrdit_global_blur_rng_is_independent_of_order():
    import torch
    from torchvision.transforms import GaussianBlur
    image=torch.arange(256,dtype=torch.float32).reshape(1,1,16,16)
    blur=GaussianBlur(3)
    adapter.seed_global(torch,123)
    expected=blur(image)
    adapter.seed_global(torch,999)
    blur(image)
    adapter.seed_global(torch,123)
    assert torch.equal(expected,blur(image))


def test_runtime_versions_are_resume_identity(tmp_path,monkeypatch):
    from types import SimpleNamespace
    path=tmp_path/'p.jsonl';path.write_text('{"id":"x","prompt":"a"}\n')
    args=SimpleNamespace(profile='sigma_dev_matched',prompts=path,seed=0)
    sources=adapter.verify_sources()
    first=adapter.make_config(args,{'id':'x','prompt':'a'},sources)
    monkeypatch.setattr(adapter,'versions',lambda:dict(first['runtime_versions'],torch='changed'))
    second=adapter.make_config(args,{'id':'x','prompt':'a'},sources)
    assert first!=second


def test_nonfinite_latent_and_decoder_are_rejected():
    import torch
    values={'latents':torch.ones(2)}
    assert adapter.finite_latents(None,0,None,values) is values
    adapter.finite_decoder(None,None,torch.ones(2))
    for bad in [float('nan'),float('inf')]:
        with pytest.raises(FloatingPointError):
            adapter.finite_latents(None,2,None,{'latents':torch.tensor([bad])})
        with pytest.raises(FloatingPointError):
            adapter.finite_decoder(None,None,torch.tensor([bad]))


def test_only_pilot_tasks_opt_into_wandb():
    prepare=load_preparer()
    common=(Path('/scratch/frozen'),Path('/scratch/new'),'sigma_dev_matched')
    assert '--wandb' not in prepare.task_text(*common,0,1,3600)
    assert '--wandb' in prepare.task_text(*common,1,5,2700)


@pytest.mark.parametrize('fail',[False,True])
def test_generation_records_timing_and_finishes_wandb(tmp_path,monkeypatch,fail):
    """Exercise orchestration with fake weights/GPU, never contact WandB."""
    import sys
    from types import SimpleNamespace
    import torch
    from PIL import Image
    calls=[]
    tracker=SimpleNamespace(log=lambda row:calls.append(('log',row)),
        finish=lambda **kw:calls.append(('finish',kw)))
    def init(**kw):
        calls.append(('init',kw));return tracker
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init,Settings=lambda **kw:kw))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda seed:None)
    monkeypatch.setattr(adapter,'verify_model',lambda *args:None)
    class Scheduler:
        config={}
        timesteps=torch.tensor([1000.,500.])
        sigmas=torch.tensor([1.,.5,0.])
        def register_to_config(self,**kw):self.config.update(kw)
    class Pipeline:
        scheduler=Scheduler()
        vae=SimpleNamespace(decoder=SimpleNamespace(register_forward_hook=lambda fn:None))
        @classmethod
        def from_pretrained(cls,*args,**kwargs):return cls()
        def enable_vae_tiling(self):pass
        def to(self,device):return self
        def __call__(self,**kw):
            if fail:raise FloatingPointError('fixture failure')
            kw['callback_on_step_end'](self,0,None,{'latents':torch.ones(2)})
            return SimpleNamespace(images=[Image.new('RGB',(4096,4096))])
    Transformer=SimpleNamespace(from_pretrained=lambda *args,**kwargs:None)
    monkeypatch.setattr(adapter,'classes',lambda *args:(Pipeline,Transformer))
    prompts=tmp_path/'p.jsonl';prompts.write_text('{"id":"p","prompt":"fixture"}\n')
    model=tmp_path/adapter.DEV;model.mkdir();out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['run.py','--profile','sigma_dev_matched','--prompts',str(prompts),
        '--out',str(out),'--dev-snapshot',str(model),'--mode','generate','--wandb'])
    if fail:
        with pytest.raises(FloatingPointError):adapter.main()
        assert calls[-1]==('finish',{'exit_code':1})
    else:
        adapter.main()
        assert calls[-1]==('finish',{'exit_code':0})
        assert any(kind=='log' and data['complete']==1 for kind,data in calls)
        record=json.loads(next(out.rglob('*seed0.json')).read_text())
        assert record['seconds']==record['wall_seconds'] and record['seconds']>0
        assert record['scheduler_sigmas']==[1.,.5,0.]
        assert record['finite_checks']==['latents_each_step','vae_decoder_before_pixel_conversion']
