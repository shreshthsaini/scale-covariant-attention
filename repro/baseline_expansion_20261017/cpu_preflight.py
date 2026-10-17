#!/usr/bin/env python3
"""Import and tiny transformer-forward compatibility check, no weights or CUDA."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import torch
from run import classes, verify_sources

p=argparse.ArgumentParser()
p.add_argument('--method',choices=['hrdit','sigma'],required=True)
p.add_argument('--extra-site',type=Path)
p.add_argument('--report',type=Path,required=True)
a=p.parse_args()
verify_sources()
Pipeline,Transformer=classes(a.method,a.extra_site)
torch.manual_seed(0)
model=Transformer(in_channels=4,num_layers=1,num_single_layers=1,
    attention_head_dim=16,num_attention_heads=2,joint_attention_dim=32,
    pooled_projection_dim=32,guidance_embeds=True,axes_dims_rope=(4,4,8),
    **(dict(sigma=True,method='yarn') if a.method=='sigma' else {}))
ids=torch.zeros(16,3)
axis=torch.linspace(0,255,4) if a.method=='sigma' else torch.arange(4)
ids[:,1]=axis.repeat_interleave(4)
ids[:,2]=axis.repeat(4)
with torch.no_grad():
    result=model(hidden_states=torch.randn(1,16,4),encoder_hidden_states=torch.randn(1,8,32),
        pooled_projections=torch.randn(1,32),timestep=torch.tensor([.5]),img_ids=ids,
        txt_ids=torch.zeros(8,3),guidance=torch.tensor([4.5])).sample
assert result.shape==(1,16,4) and torch.isfinite(result).all()
report=dict(status='cpu_tiny_transformer_forward_passed',method=a.method,shape=list(result.shape),
    absolute_max=float(result.abs().max()),versions={x:importlib.metadata.version(x) for x in
    ['torch','diffusers','transformers','accelerate']},extra_site=str(a.extra_site),
    sigma_enabled=a.method=='sigma',coordinate_max=float(ids.max()),
    excluded=['GPU kernels','FlexAttention HAP mask path','4K generation','pretrained weights','quality evaluation'])
a.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
