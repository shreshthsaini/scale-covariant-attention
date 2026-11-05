#!/usr/bin/env python3
"""Round 6, Step 2: where attention goes at the place a copy forms (.agent/notes/duplication_fix_round6.md).

Re-runs the adopted operator (sca_v2, campaign flags, seed 0) on the prompts of a spec file and, at the chosen steps, in
every block, takes three 4 x 4-token query patches inside one query tile: "copy" (centred on a human-labeled copy in
the campaign image), "main" (centre of the main subject's detector box) and "bg" (background far from every detected
box). For each patch and head it measures, on the same q and k, three operators:
  full  full attention at true positions (text keys at id 0, all 65,536 image keys);
  loc   the window branch of sca_v2: text keys, the bridge tokens and the current window, one softmax, true positions;
  glob  the global branch: coarse query ids, text keys and the block-centre keys at coarse ids;
and three quantities: text (softmax mass on the 512 text keys), subj (mass on the T5 tokens of the subject words) and
main (mass on image keys inside the main subject's box; block keys by their centre). The mix weight pi of each head is
stored too. Output per prompt: <out>/<id>/probe.npz with array stats [steps, layers, heads, 3 regions, 3 operators,
3 quantities], pi [steps, layers, heads], sigma [steps], and final.png (checked against the campaign PNG).
The probe only reads q and k; the image is the campaign image up to run-to-run noise.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

REGIONS = ("copy", "main", "bg")
OPS = ("full", "loc", "glob")
QTY = ("text", "subj", "main")


def patch_indices(xy_px, grid, tile=8, side=4):
    """Image-token indices of a side x side patch inside the query tile that contains pixel xy (16 px per token)."""
    H, W = grid
    r, c = int(xy_px[1]) // 16, int(xy_px[0]) // 16
    r, c = min(max(r, 0), H - 1), min(max(c, 0), W - 1)
    r0 = (r // tile) * tile + min(max(r % tile - side // 2, 0), tile - side)
    c0 = (c // tile) * tile + min(max(c % tile - side // 2, 0), tile - side)
    return torch.tensor([(r0 + a) * W + c0 + b for a in range(side) for b in range(side)])


def box_mask(box_px, grid):
    """Bool [H*W]: image tokens whose centre lies inside the pixel box (x0, y0, x1, y1)."""
    H, W = grid
    rows = (torch.arange(H) * 16 + 8).float()
    cols = (torch.arange(W) * 16 + 8).float()
    x0, y0, x1, y1 = box_px
    m = ((rows >= y0) & (rows <= y1))[:, None] & ((cols >= x0) & (cols <= x1))[None, :]
    return m.flatten()


def window_keys(idx, grid, tile, win):
    """Image-token indices of the window shared by the query tile of idx[0] (window_index rule)."""
    H, W = grid
    (th, tw), (Wh, Ww) = tile, win
    r, c = int(idx[0]) // W, int(idx[0]) % W
    r0 = min(max((r // th) * th + th // 2 - Wh // 2, 0), H - Wh)
    c0 = min(max((c // tw) * tw + tw // 2 - Ww // 2, 0), W - Ww)
    rr = torch.arange(r0, r0 + Wh)[:, None] * W + torch.arange(c0, c0 + Ww)[None, :]
    return rr.flatten()


def _masses(qi, keys, scale, groups):
    """qi [n, h, D], keys [M, h, D]; groups: list of index tensors into keys -> [len(groups), h] mean softmax mass."""
    p = torch.einsum("nhd,mhd->hnm", qi, keys).mul(scale).softmax(-1)          # [h, n, M]
    return torch.stack([p[..., g].sum(-1).mean(-1) if len(g) else torch.zeros(p.shape[0], device=p.device)
                        for g in groups])


@torch.no_grad()
def measure(q, k, rot, ctx, regions, subj_idx, main_mask):
    """[regions, 3 operators, 3 quantities, heads] for one block. q, k pre-RoPE [1, L + N, h, D] (text first);
    rot = (cos, sin) of all L + N tokens; regions: list of image-token index tensors (each inside one query tile);
    subj_idx: text-token indices of the subject words; main_mask: bool [N] image tokens of the main subject."""
    from sca.attn import bridge_index, pool_tokens, rope
    H, W = ctx.grid
    N = H * W
    L = q.shape[1] - N
    dev = q.device
    cos, sin = rot
    D = q.shape[-1]
    scale = D ** -0.5
    kr = rope(k, rot)[0].float()                                                  # [L + N, h, D]
    f, tile, win = ctx.geometry()
    rq, rk = ctx.coarse_rope(f, dev)
    kp = pool_tokens(k[:, L:], (H, W), f, ctx.pool)
    kg = torch.cat([kr[:L], rope(kp, rk)[0].float()], 0)                         # [L + Nb, h, D]
    centres = torch.arange(N, device=dev).view(H, W)[f[0] // 2::f[0], f[1] // 2::f[1]].reshape(-1)
    main_mask = main_mask.to(dev)
    subj = torch.as_tensor(subj_idx, dtype=torch.long, device=dev)
    txt = torch.arange(L, device=dev)
    cells, per = ctx.bridge_now()
    bidx = bridge_index(k[:, L:], (H, W), cells, per)[0] if per else torch.zeros(0, dtype=torch.long, device=dev)
    out = torch.zeros(len(regions), 3, 3, q.shape[2], device=dev)
    for ri, idx in enumerate(regions):
        idx = idx.to(dev)
        qi = rope(q[:, L + idx], (cos[L + idx], sin[L + idx]))[0].float()        # [n, h, D] true positions
        main_full = L + torch.nonzero(main_mask).flatten()
        out[ri, 0] = _masses(qi, kr, scale, [txt, subj, main_full])
        wk = window_keys(idx.cpu(), (H, W), tile, win).to(dev)
        keys = torch.cat([kr[:L], kr[L + bidx], kr[L + wk]], 0)
        img = torch.cat([bidx, wk])
        main_loc = L + torch.nonzero(main_mask[img]).flatten()
        out[ri, 1] = _masses(qi, keys, scale, [txt, subj, main_loc])
        qg = rope(q[:, L + idx], (rq[0][idx], rq[1][idx]))[0].float()
        main_glob = L + torch.nonzero(main_mask[centres]).flatten()
        out[ri, 2] = _masses(qg, kg, scale, [txt, subj, main_glob])
    return out


def subject_tokens(tokenizer, prompt, words, max_len=512):
    """T5 token indices whose character span overlaps a word starting with any of `words` (case-insensitive)."""
    enc = tokenizer(prompt, padding="max_length", max_length=max_len, truncation=True, return_offsets_mapping=True)
    spans = [m.span() for w in words for m in re.finditer(r"\b" + re.escape(w) + r"\w*", prompt, flags=re.I)]
    return [j for j, (a, b) in enumerate(enc["offset_mapping"]) if b > a and any(a < e and b > s for s, e in spans)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", type=Path, required=True, help="JSON list of {prompt_id, copy_xy, main_box, bg_xy, words}")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", default="6,8,9,10,11,12,13,14,16,18,20,24")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    import sample
    from sca.attn import SCAProcessor, install
    from sca.runtime import image_seed, image_stem, load_pipeline, read_prompts, runtime_environment, set_mu
    from PIL import Image
    runtime_environment()
    steps = [int(s) for s in a.steps.split(",")]
    spec = json.loads(a.spec.read_text())
    prompts = {r["id"]: r for r in read_prompts(Path("/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl"))}
    args = sample.parser().parse_args(["--method", "sca", "--res", "4096x4096", "--seed", "0", "--guidance", "4.5",
                                       "--pi-table", "/scratch/09032/saini_2/sca/calib/k1_1024.npz", "--gate-sigma", "0.89",
                                       "--label", "sca_v2", "--bridge-cells", "8", "--bridge-k", "16", "--win-sched",
                                       "0.88:128,0.86:112,0.84:96,0.82:80", "--win-pi-match", "--win-pi-interp",
                                       "--prompts", "x", "--out", str(a.out), "--device", a.device])
    h, w = args.res
    grid = (h // 16, w // 16)
    ctx, _ = sample.build_context(args, grid)
    pipe = load_pipeline(a.device, "dev")
    install(pipe.transformer, ctx)
    ctx.scheduler = pipe.scheduler
    pipe.vae.enable_tiling()
    set_mu(pipe, h, w, "max_shift")
    state = {}
    orig = SCAProcessor.sca

    def probed(self, q, k, v, image_rotary_emb):
        o = orig(self, q, k, v, image_rotary_emb)
        st = state.get("cur")
        if st is not None and ctx.step in st["steps"]:
            s = st["steps"].index(ctx.step)
            st["stats"][s, self.layer] = measure(q, k, image_rotary_emb, ctx, st["regions"], st["subj"],
                                                 st["main"]).permute(3, 0, 1, 2).cpu().numpy()
            st["pi"][s, self.layer] = ctx.pi_now(self.layer, q.shape[2], q.device).float().cpu().numpy()
            st["sigma"][s] = ctx.sigma
        return o
    SCAProcessor.sca = probed
    for item in spec:
        pid = item["prompt_id"]
        out = a.out / pid
        if (out / "probe.npz").exists():
            print("skip", pid, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        row = prompts[pid]
        subj = subject_tokens(pipe.tokenizer_2, row["prompt"], item["words"])
        regions = [patch_indices(item["copy_xy"], grid), patch_indices(item["main_xy"], grid),
                   patch_indices(item["bg_xy"], grid)]
        nl, nh = ctx.n_layers or 57, 24
        state["cur"] = {"steps": steps, "regions": regions, "subj": subj, "main": box_mask(item["main_box"], grid),
                        "stats": np.zeros((len(steps), nl, nh, 3, 3, 3), np.float32),
                        "pi": np.zeros((len(steps), nl, nh), np.float32), "sigma": np.zeros(len(steps))}
        g = torch.Generator(device="cpu").manual_seed(image_seed(pid, 0))
        with torch.inference_mode():
            image = pipe(prompt=row["prompt"], height=h, width=w, num_inference_steps=28, guidance_scale=4.5,
                         generator=g, max_sequence_length=512).images[0]
        image.save(out / "final.png")
        st = state.pop("cur")
        camp = Path(f"/scratch/09032/saini_2/sca/out/dupfix/sca_v2/4096x4096/{image_stem(pid, 0)}.png")
        psnr = None
        if camp.exists():
            x = np.asarray(image.convert("RGB"), np.float64)
            y = np.asarray(Image.open(camp).convert("RGB"), np.float64)
            psnr = float(10 * math.log10(255 ** 2 / max(((x - y) ** 2).mean(), 1e-12)))
        np.savez_compressed(out / "probe.npz", stats=st["stats"], pi=st["pi"], sigma=st["sigma"], steps=np.array(steps),
                            subj=np.array(subj), regions=np.stack([r.numpy() for r in regions]),
                            main_box=np.array(item["main_box"]), psnr_vs_campaign=np.array(psnr if psnr else np.nan))
        print("probe", pid, "subject tokens", subj, "psnr vs campaign", psnr, flush=True)


if __name__ == "__main__":
    main()
