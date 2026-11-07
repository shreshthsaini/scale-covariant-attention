#!/usr/bin/env python3
"""GPU check of the per-query mix (sca/qmix.py) at the real 4096^2 shapes, random q/k/v.

Why: on GPU, f13 images carry token-sized colour blocks (borders, dark areas) that the CPU brute-force tests do not
show. This script compares, on CUDA:
  A. the compiled sca_qmix_fn against the eager sca_qmix_forward (same inputs), for f1 and f13;
  B. each component on sampled queries (border and interior) against a float32 brute force:
     text part (true ids), near part (bridge + window, true ids), far lattice part (outside-window block centres,
     coarse ids), and the in-window lattice part used by f1.
Usage: python scripts/r7_qmix_gpu_check.py [--code DIR] [--window 64]
"""
import argparse
import math
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--code", default="/scratch/09032/saini_2/sca/code/8ace484")
p.add_argument("--window", type=int, default=64)
p.add_argument("--heads", type=int, default=24)
a = p.parse_args()
sys.path.insert(0, a.code)
from sca.attn import SCAContext, bridge_index, flex_layout, local_branch_flex, rope  # noqa: E402
from sca import qmix as qm, textgate as tg  # noqa: E402

try:
    from diffusers.models.embeddings import FluxPosEmbed
except ImportError:  # older layout
    from diffusers.models.transformers.transformer_flux import FluxPosEmbed

dev = "cuda"
H = W = 256
L, h, D = 512, a.heads, 128
f, tile, win = (4, 4), (8, 8), (a.window, a.window)
cells, per = (8, 8), 16
torch.manual_seed(0)
q, k, v = (torch.randn(1, L + H * W, h, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
ids = torch.zeros(H * W, 3, device=dev)
ids[:, 1] = torch.arange(H, device=dev).repeat_interleave(W).float()
ids[:, 2] = torch.arange(W, device=dev).repeat(H).float()
pe = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])
rot = pe(torch.cat([torch.zeros(L, 3, device=dev), ids]))
rot = tuple(t.to(dev) for t in rot)
ctx = SCAContext(mode="sca", known_grid=(H, W))
ctx.pos_embed = pe
ctx.set_inputs(ids)
rq, rk = ctx.coarse_rope(f, dev)
rq, rk = tuple(t.to(dev) for t in rq), tuple(t.to(dev) for t in rk)
idx = bridge_index(k[:, L:], (H, W), cells, per)
R = idx.shape[1]
layout = flex_layout(R, (H, W), tile, win, dev)
lat_in = tg.lattice_layout((H, W), tile, win, f, dev, inside=True)
lat_out = tg.lattice_layout((H, W), tile, win, f, dev)
pi = torch.full((1, 1, h, 1), 0.7, device=dev)
args = (q, k, v, rot, rq, rk, pi, L, (H, W), f, "center", layout, cells, per, lat_in, lat_out)

# A. compiled vs eager
for three in (False, True):
    oc = qm.sca_qmix_fn(dev)(*args, 1.0, three, 0.0).float()
    oe = qm.sca_qmix_forward(*args, 1.0, three, 0.0).float()
    d = (oc - oe).abs()
    name = "f13" if three else "f1"
    print(f"A {name}: compiled vs eager max {d.max().item():.4g}, mean {d.mean().item():.4g}, "
          f"finite {torch.isfinite(oc).all().item()}, |o| max {oc.abs().max().item():.3g}", flush=True)
    img = d[0, L:].amax((1, 2)).view(H, W)
    bad = (img > 0.05).nonzero()
    print(f"  tokens with diff > 0.05: {bad.shape[0]}; first {bad[:8].tolist()}", flush=True)

# B. components on sampled queries
cos, sin = rot
qi = rope(q[:, L:], (cos[L:], sin[L:]))
kr = rope(k, rot)
kt, vt = kr[:, :L], v[:, :L]
qg = rope(q[:, L:], rq)
cen = tg.lattice_index((H, W), f, dev)
kb, vb = rope(k[:, L:][:, cen], rk), v[:, L:][:, cen]
o_t, lse_t = tg.text_part(qi, kt, vt)
kimg, vimg = kr[:, L:], v[:, L:]
g = idx[:, :, None, None].expand(-1, -1, h, D)
kl, vl = torch.cat([torch.gather(kimg, 1, g), kimg], 1), torch.cat([torch.gather(vimg, 1, g), vimg], 1)
o_n, lse_n = local_branch_flex(qi, kl, vl, R, layout, True, 0.0, None, return_lse=True)
o_n = o_n.transpose(1, 2).index_select(1, layout.q_inv)
lse_n = lse_n.index_select(2, layout.q_inv)
o_far, lse_far = tg.lattice_part(qg, kb, vb, lat_out)
o_in, lse_in = tg.lattice_part(qg, kb, vb, lat_in)

rows = [0, 1, 7, 8, 127, 128, 248, 255]
sample = torch.tensor([r * W + c for r in rows for c in rows], device=dev)
Wh = win[0]
r0 = (torch.arange(H, device=dev) // tile[0] * tile[0] + tile[0] // 2 - Wh // 2).clamp(0, H - Wh)
c0 = (torch.arange(W, device=dev) // tile[1] * tile[1] + tile[1] // 2 - Wh // 2).clamp(0, W - Wh)
kr_c, kc_c = cen // W, cen % W
sc = D ** -0.5
err = {"text": 0.0, "near": 0.0, "far": 0.0, "in": 0.0, "lse_text": 0.0, "lse_near": 0.0, "lse_far": 0.0, "lse_in": 0.0}
for n in sample.tolist():
    r, c = n // W, n % W
    qn = qi[0, n].float()                                     # [h, D]
    st = torch.einsum("hd,lhd->hl", qn, kt[0].float()) * sc
    ot = torch.einsum("hl,lhd->hd", st.softmax(-1), vt[0].float())
    rr = torch.arange(r0[r], r0[r] + Wh, device=dev)
    cc = torch.arange(c0[c], c0[c] + Wh, device=dev)
    wi = (rr[:, None] * W + cc[None, :]).reshape(-1)
    keys = torch.cat([torch.gather(kimg, 1, g)[0], kimg[0, wi]], 0).float()
    vals = torch.cat([torch.gather(vimg, 1, g)[0], vimg[0, wi]], 0).float()
    sn = torch.einsum("hd,mhd->hm", qn, keys) * sc
    on = torch.einsum("hm,mhd->hd", sn.softmax(-1), vals)
    qgn = qg[0, n].float()
    sb = torch.einsum("hd,mhd->hm", qgn, kb[0].float()) * sc
    inside = (kr_c >= r0[r]) & (kr_c < r0[r] + Wh) & (kc_c >= c0[c]) & (kc_c < c0[c] + Wh)
    sfar = sb.masked_fill(inside[None], float("-inf"))
    sin_ = sb.masked_fill(~inside[None], float("-inf"))
    ofar = torch.einsum("hm,mhd->hd", sfar.softmax(-1), vb[0].float())
    oin = torch.einsum("hm,mhd->hd", sin_.softmax(-1), vb[0].float())
    for key, got, ref in (("text", o_t[0, n], ot), ("near", o_n[0, n], on), ("far", o_far[0, n], ofar),
                          ("in", o_in[0, n], oin)):
        err[key] = max(err[key], (got.float() - ref).abs().max().item())
    for key, got, ref in (("lse_text", lse_t[0, :, n], st.logsumexp(-1)), ("lse_near", lse_n[0, :, n], sn.logsumexp(-1)),
                          ("lse_far", lse_far[0, :, n], sfar.logsumexp(-1)), ("lse_in", lse_in[0, :, n], sin_.logsumexp(-1))):
        err[key] = max(err[key], (got.float() - ref).abs().max().item())
print("B max abs error over", sample.numel(), "sampled queries (border and interior):",
      {k_: round(v_, 5) for k_, v_ in err.items()}, flush=True)
print("B ln-vs-log2 hint: lse_far ratio", (lse_far[0, :, sample[:4]].float() /
                                            torch.stack([torch.einsum("hd,mhd->hm", qg[0, n].float(), kb[0].float()).mul(sc)
                                                         .masked_fill(((kr_c >= r0[n // W]) & (kr_c < r0[n // W] + Wh) &
                                                                       (kc_c >= c0[n % W]) & (kc_c < c0[n % W] + Wh))[None],
                                                                      float("-inf")).logsumexp(-1)
                                                         for n in sample[:4].tolist()], -1)).mean().item())
