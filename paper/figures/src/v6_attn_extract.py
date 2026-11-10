"""Attention rows of all 24 heads of block 12 at one query, at three noise levels (v6 Figures 2 and 3).

Input: the CPU capture of FLUX.1-dev at 1024^2 written by scripts/attn_focus.py for the Aesthetic-4K prompt
a4k_eval_4096_minus8343371271909676658, seed 0 (the native 1024^2 stage of progressive SCA: 30 steps, guidance 3.5,
mu 1.15), /scratch/09032/saini_2/sca/out/attn_focus/sheep_church_seed0/qk_step{08,20,26}.pt. The same capture and
query as fig_attn_focus.py (query token row 43, column 11: the face of the leftmost sheep; block 12 one-based).

For every head of block 12 and every captured step this writes, exactly as attn_focus_data.Capture computes them:
  full        the full-attention row over the 64 x 64 image keys (Eq. 1), and the text mass;
  share16     the share of the image attention within Chebyshev distance 16 tokens of the query (the quantity of
              Figure 2b and of the variance split), and m4, the share inside the emulated r = 4 window;
and at step 20 (sigma 0.612, the noise bin of the J curves) the emulated local (window) and global (block-centre)
operators at r = 4, which panel (b) of the method figure draws.

Output (small; float32): paper/figures/src/data/v6_attn_block12.npz and v6_attn_block12.json (provenance and the
scalar values). CPU only, about a minute per step; run on a compute node:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I paper/figures/src/v6_attn_extract.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CAPTURE = Path("/scratch/09032/saini_2/sca/out/attn_focus/sheep_church_seed0")
OUT = HERE / "data" / "v6_attn_block12.npz"
STEPS = (8, 20, 26)
STEP_OPS = 20             # step whose local and global operators are stored
QUERY = (43, 11)          # token (row, col), as fig_attn_focus.QUERY
BLOCK = 11                # zero-based (block 12 one-based), as fig_attn_focus.BLOCK
RATIO = 4
RADIUS = 16


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if limit is None:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        else:
            h.update(fh.read(limit))
    return h.hexdigest()


def share_within(img: np.ndarray, row: int, col: int, radius: int) -> float:
    n = img.shape[0]
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    inside = np.maximum(np.abs(rr - row), np.abs(cc - col)) <= radius
    return float(img[inside].sum() / img.sum())


def main() -> None:
    import torch
    from attn_focus_data import Capture
    torch.set_num_threads(32)
    meta = json.loads((CAPTURE / "meta.json").read_text())
    out: dict[str, np.ndarray] = {}
    rec = {"capture": str(CAPTURE), "prompt_id": meta["prompt_id"], "prompt": meta["prompt"], "seed": meta["seed"],
           "steps_total": meta["steps"], "guidance": meta["guidance"], "mu": meta.get("mu"),
           "query_row_col": list(QUERY), "block_one_based": BLOCK + 1, "ratio": RATIO, "radius": RADIUS,
           "image_1024_sha256": sha256(CAPTURE / "image_1024.png"), "steps": {}}
    sigmas, fulls, texts, s16, m4 = [], [], [], [], []
    for step in STEPS:
        cap = Capture(CAPTURE, step)
        n_heads = cap.q.shape[2]
        r0, c0, w, t = cap.window(*QUERY, RATIO)
        f_step, t_step, s_step, m_step = [], [], [], []
        for h in range(n_heads):
            full, txt = cap.full(BLOCK, h, *QUERY)
            _, _, m = cap.split(full, *QUERY, RATIO)
            f_step.append(full.astype(np.float32))
            t_step.append(txt)
            s_step.append(share_within(full, *QUERY, RADIUS))
            m_step.append(m)
        if step == STEP_OPS:
            loc, glo, ltxt, gtxt = [], [], [], []
            for h in range(n_heads):
                lo, lt = cap.local(BLOCK, h, *QUERY, RATIO)
                go, gt = cap.global_(BLOCK, h, *QUERY, RATIO)
                loc.append(lo[r0:r0 + w, c0:c0 + w].astype(np.float32))
                glo.append(go.astype(np.float32))
                ltxt.append(lt)
                gtxt.append(gt)
            out["local_r4"] = np.stack(loc)            # [24, w, w] window crop, rows r0.., cols c0..
            out["global_r4"] = np.stack(glo)           # [24, 16, 16] block-centre keys
            out["local_r4_text"] = np.array(ltxt)
            out["global_r4_text"] = np.array(gtxt)
            out["window_r4"] = np.array([r0, c0, w, t])
        sigmas.append(cap.sigma)
        fulls.append(np.stack(f_step))
        texts.append(t_step)
        s16.append(s_step)
        m4.append(m_step)
        rec["steps"][str(step)] = {"sigma": cap.sigma, "qk_bytes": (CAPTURE / f"qk_step{step:02d}.pt").stat().st_size}
        del cap
        print(f"step {step}: sigma {sigmas[-1]:.4f}; share16 head 4 {s_step[3]:.4f}, head 7 {s_step[6]:.4f}",
              flush=True)
    out.update(steps=np.array(STEPS), sigma=np.array(sigmas), query=np.array(QUERY), block=np.array(BLOCK),
               full=np.stack(fulls), text_mass=np.array(texts), share16=np.array(s16), m4=np.array(m4))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    rec["arrays"] = {k: list(v.shape) for k, v in out.items()}
    rec["share16"] = {f"head{h + 1}": [round(float(x), 6) for x in out["share16"][:, h]] for h in range(24)}
    rec["m4"] = {f"head{h + 1}": [round(float(x), 6) for x in out["m4"][:, h]] for h in range(24)}
    rec["output_sha256"] = sha256(OUT)
    OUT.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
