#!/usr/bin/env python3
"""Pinned official baseline adapters. Plan and import checks never load weights."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
DEV = '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'
KREA = '8162a9c7b05a641be098422bf2fcf335615c2f28'
PROFILES = {
    'sigma_dev_matched': dict(method='sigma', model='dev', dtype='bf16', mu='max_shift', steps=[28], guidance=[4.5]),
    'sigma_krea_matched': dict(method='sigma', model='krea', dtype='bf16', mu='max_shift', steps=[28], guidance=[4.5]),
    'sigma_krea_native': dict(method='sigma', model='krea', dtype='bf16', mu='upstream_target_length_cap', steps=[28], guidance=[4.5]),
    'sigma_dev_paper_settings': dict(method='sigma', model='dev', dtype='bf16', mu='upstream_target_length_cap', steps=[28], guidance=[3.5]),
    'hrdit_dev_author': dict(method='hrdit', model='dev', dtype='fp16', mu='upstream_scheduler_attribute_assignment', steps=[30,17,10], guidance=[3.5,4.5,6.0]),
}
VERSION_PACKAGES=('torch','triton','diffusers','transformers','accelerate','numpy','pillow','torchvision')


def versions():
    return {x:importlib.metadata.version(x) for x in VERSION_PACKAGES}


def seed_global(torch, seed):
    # Upstream HRDiT GaussianBlur samples sigma from the global Torch RNG.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_latents(pipe, step, timestep, values):
    import torch
    if not torch.isfinite(values['latents']).all():
        raise FloatingPointError(f'Non-finite latents after step {step}')
    return values


def finite_decoder(module, inputs, output):
    import torch
    if not torch.isfinite(output).all():
        raise FloatingPointError('Non-finite VAE decoder output before pixel conversion')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    receipt = json.loads((ROOT/'sources.json').read_text())
    for name, repo in receipt['repositories'].items():
        for row in repo['files']:
            if sha(ROOT/'vendor'/name/row['path']) != row['sha256']:
                raise ValueError(f'Upstream source changed: {name}/{row["path"]}')
    return receipt


def verify_model(snapshot, repo_id):
    manifest=json.loads((ROOT/'model_manifest.json').read_text())
    model=next(x for x in manifest['models'] if x['repo_id']==repo_id)
    for record in model['files']:
        path=snapshot/record['path']
        if not path.is_file() or path.stat().st_size!=record['bytes']:
            raise ValueError(f'Missing or resized model file: {path}')
        if path.resolve().name!=Path(record['resolved_path']).name:
            raise ValueError(f'Model blob identity differs: {path}')
        if record.get('metadata_sha256') and sha(path)!=record['metadata_sha256']:
            raise ValueError(f'Model metadata checksum differs: {path}')
    return model


def read_prompts(path, start=0, end=None):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    ids = [str(x['id']) for x in rows]
    if len(set(ids)) != len(ids) or any(not str(x['prompt']).strip() for x in rows):
        raise ValueError('Duplicate IDs or empty prompts')
    if start < 0 or (end is not None and (end <= start or end > len(rows))):
        raise ValueError('Invalid prompt slice')
    selected = rows[start:end]
    if not selected:
        raise ValueError('Empty prompt slice')
    return selected


def seed_for(prompt_id, seed):
    value = json.dumps([str(prompt_id), int(seed)], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big') % (2**63-1)


def stem_for(prompt_id, seed):
    key = str(prompt_id)
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', key)[:80].strip('.') or 'prompt'
    return f'{safe}_{hashlib.sha256(key.encode()).hexdigest()[:10]}_seed{seed}'


def atomic_json(path, value):
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def check_resume(image, sidecar, config):
    if not image.exists() and not sidecar.exists():
        return False
    if not image.is_file() or not sidecar.is_file():
        raise ValueError(f'Partial output retained; use a new output root: {image}')
    record = json.loads(sidecar.read_text())
    if record.get('config') != config or record.get('complete') is not True or record.get('png_sha256') != sha(image):
        raise ValueError(f'Output provenance mismatch: {image}')
    from PIL import Image
    with Image.open(image) as opened:
        if opened.size != (4096,4096):
            raise ValueError(f'Wrong output dimensions: {image}')
    return True


def classes(method, extra_site=None):
    if method=='hrdit':
        if not extra_site:
            raise ValueError('HRDiT requires the separately pinned wavelet dependency overlay')
        overlay=json.loads((ROOT/'dependency_overlay.json').read_text())
        for row in overlay['files']:
            path=Path(extra_site)/row['path']
            if not path.is_file() or sha(path)!=row['sha256']:
                raise ValueError(f'Dependency overlay missing or changed: {path}')
    if extra_site:
        sys.path.insert(0, str(Path(extra_site).resolve()))
    sys.path.insert(0, str(ROOT/'vendor'/method))
    if method == 'hrdit':
        from hrdit import FluxPipeline, FluxTransformer2DModel
    else:
        from flux.pipeline_flux import FluxPipeline
        from flux.transformer_flux import FluxTransformer2DModel
    return FluxPipeline, FluxTransformer2DModel


def make_config(args, row, sources):
    profile = PROFILES[args.profile]
    return dict(schema_version=1, method=args.profile, prompt_id=str(row['id']), prompt=row['prompt'],
        res='4096x4096', seed=args.seed, derived_seed=seed_for(row['id'],args.seed), profile=profile,
        model_revision=KREA if profile['model']=='krea' else DEV, component_revision=DEV,
        upstream_commit=sources['repositories'][profile['method']]['commit'],
        adapter_sha256=sha(__file__), source_receipt_sha256=sha(ROOT/'sources.json'),
        model_manifest_sha256=sha(ROOT/'model_manifest.json'),
        dependency_overlay_sha256=sha(ROOT/'dependency_overlay.json') if profile['method']=='hrdit' else None,
        runtime_versions=versions(),
        prompt_manifest_sha256=sha(args.prompts), generator_device='cpu', vae_tiling=True,
        execution='resident_cuda', scope_plan_sha256=sha(ROOT/'vendor/hrdit/configs/scope_plan_flux.json') if profile['method']=='hrdit' else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=PROFILES, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dev-snapshot', type=Path)
    p.add_argument('--krea-snapshot', type=Path)
    p.add_argument('--extra-site', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['plan','import-check','generate'], default='plan')
    p.add_argument('--wandb', action='store_true', help='Log pilot/campaign completion and timings, no images')
    args = p.parse_args()
    sources = verify_sources()
    rows = read_prompts(args.prompts,args.start,args.end)
    configs = [make_config(args,row,sources) for row in rows]
    profile = PROFILES[args.profile]
    if args.mode == 'plan':
        print(json.dumps(dict(status='planned_not_run', samples=configs),indent=2)); return
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:
        os.environ[key] = '1'
    Pipeline, Transformer = classes(profile['method'],args.extra_site)
    import torch
    runtime_versions = versions()
    if args.mode == 'import-check':
        print(json.dumps(dict(status='imports_only_no_gpu_generation', versions=runtime_versions, profile=args.profile,
            extra_site=str(args.extra_site), pipeline_module=Pipeline.__module__, transformer_module=Transformer.__module__))); return
    if not torch.cuda.is_available():
        raise RuntimeError('Generation requires a GPU allocation')
    for snapshot, revision in [(args.dev_snapshot,DEV)]+([(args.krea_snapshot,KREA)] if profile['model']=='krea' else []):
        if snapshot is None or not snapshot.is_dir() or snapshot.resolve().name != revision:
            raise ValueError(f'Use the pinned existing local snapshot {revision}')
    verify_model(args.dev_snapshot,'black-forest-labs/FLUX.1-dev')
    if profile['model']=='krea':
        verify_model(args.krea_snapshot,'black-forest-labs/FLUX.1-Krea-dev')
    args.out.mkdir(parents=True,exist_ok=True)
    out = args.out/args.profile/'4096x4096'
    out.mkdir(parents=True,exist_ok=True)
    pending=[]
    for row, config in zip(rows,configs):
        stem = stem_for(row['id'],args.seed)
        if not check_resume(out/(stem+'.png'),out/(stem+'.json'),config):
            pending.append((row,config,stem))
    if not pending:
        print('All selected outputs have matching provenance'); return
    dtype = torch.float16 if profile['dtype']=='fp16' else torch.bfloat16
    transformer_root = args.krea_snapshot if profile['model']=='krea' else args.dev_snapshot
    options = dict(sigma=True,method='yarn') if profile['method']=='sigma' else {}
    transformer = Transformer.from_pretrained(str(transformer_root),subfolder='transformer',torch_dtype=dtype,local_files_only=True,**options)
    pipe = Pipeline.from_pretrained(str(args.dev_snapshot),transformer=transformer,torch_dtype=dtype,local_files_only=True)
    pipe.enable_vae_tiling()
    pipe.vae.decoder.register_forward_hook(finite_decoder)
    pipe.to('cuda')
    if profile['mu']=='max_shift':
        pipe.scheduler.register_to_config(max_image_seq_len=65536)
    elif profile['method']=='hrdit':
        pipe.scheduler.use_dynamic_shifting = False  # Exact upstream inference.py assignment.
    tracker=None
    completed=False
    if args.wandb:
        import wandb
        logdir=args.out/'wandb'
        logdir.mkdir(parents=True,exist_ok=True)
        tracker=wandb.init(project='scale-covariant-attention',group='baseline-expansion-20261017',
            job_type='baseline-generation',dir=str(logdir),settings=wandb.Settings(init_timeout=45),config=dict(profile=args.profile,
            start=args.start,end=args.end,seed=args.seed,source_receipt_sha256=sha(ROOT/'sources.json')))
    try:
        for row,config,stem in pending:
            seed_global(torch,config['derived_seed'])
            generator = torch.Generator('cpu').manual_seed(config['derived_seed'])
            kwargs = dict(prompt=row['prompt'],generator=generator,
                callback_on_step_end=finite_latents,callback_on_step_end_tensor_inputs=['latents'])
            intermediate=None
            if profile['method']=='sigma':
                kwargs.update(height=4096,width=4096,num_inference_steps=profile['steps'][0],guidance_scale=profile['guidance'][0])
            else:
                intermediate = args.out/'intermediates'/stem/f'attempt_{time.time_ns()}_{os.getpid()}'
                intermediate.parent.mkdir(parents=True,exist_ok=True)
                kwargs.update(height=1024,width=1024,guidance_scale=3.5,num_inference_steps=30,max_sequence_length=512,
                    ntk_factor=[4.0,10.0],proportional_attention=True,text_duplication=False,swin_pachify=False,
                    target_heights=[2048,4096],target_widths=[2048,4096],num_inference_steps_highres=[17,10],
                    filter_ratio=[0.2,0.2],guidance_scale_highres=[4.5,6],structure_guidance='fft',alphas=[1.0,0.25],
                    betas=[0.5,0.5],upsampling_choice='latent',flow_choice='v_theta',
                    scope_plan=json.loads((ROOT/'vendor/hrdit/configs/scope_plan_flux.json').read_text()),
                    hap_step_ranges=[[0,999],[0,999]],hap_anchor_stride=32,spa_step_ranges=[[0,3],[0,0]],
                    group_nums=[80,80],save_dir=str(intermediate),img_index=0)
            torch.cuda.synchronize(); start=time.perf_counter()
            image = pipe(**kwargs).images[0]
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            if image.size != (4096,4096):
                raise ValueError(f'Unexpected image dimensions {image.size}')
            image_path=out/(stem+'.png'); temporary=out/(stem+f'.tmp.{os.getpid()}.png')
            image.save(temporary); os.replace(temporary,image_path)
            atomic_json(out/(stem+'.json'),dict(complete=True,config=config,png_sha256=sha(image_path),
                runtime=runtime_versions,scheduler_config=dict(pipe.scheduler.config),wall_seconds=elapsed,seconds=elapsed,
                finite_checks=['latents_each_step','vae_decoder_before_pixel_conversion'],
                scheduler_timesteps=pipe.scheduler.timesteps.detach().cpu().tolist(),
                scheduler_sigmas=pipe.scheduler.sigmas.detach().cpu().tolist(),
                timing_scope='pipeline_including_text_encoding_and_decode_not_model_load_no_excluded_warmup',
                upstream_source=sources['repositories'][profile['method']],extra_site=str(args.extra_site),
                intermediates_prefix=str(intermediate) if intermediate else None))
            print(json.dumps(dict(prompt_id=row['id'],seconds=elapsed,path=str(image_path))),flush=True)
            if tracker:
                tracker.log(dict(prompt_id=str(row['id']),pipeline_seconds=elapsed,complete=1))
        completed=True
    finally:
        if tracker:
            tracker.finish(exit_code=0 if completed else 1)


if __name__ == '__main__':
    main()
