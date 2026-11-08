"""Attention rows from the captured FLUX.1-dev queries and keys (scripts/attn_focus.py), for the attention-focus figure.

Input: a capture directory with qk_step<j>.pt ({"q", "k": [57, T + N, 24, 128] bf16, pre-rotation, text first;
"txt_ids", "img_ids"; "sigma"}) and image_1024.png. For one image query token and one (block, head) this module
returns, exactly as FLUX.1-dev computes them at that step:

  full    softmax over all T + N keys with FLUX's RoPE (Eq. 1): the image part as a 64 x 64 map, and the text mass;
  near    the image part inside the window of the emulated local operator at ratio r (side n / r tokens around the
          query's tile of round(t n / (r n0)) tokens, clamped to the grid; Section 2.4), and far = the rest, with
          m = the near share of the image attention (the split of Lemma 1 restricted to image keys);
  local   the emulated local operator at ratio r: softmax over the text keys and the window keys at their own
          positions (no bridge, as in reach emulation);
  global  the emulated global operator at ratio r: softmax over the text keys and the block-centre keys
          kappa(b) = r b + floor(r / 2), keys at coarse position b and the query at c(p) = floor(p / r) (Eq. 2).

Everything runs on CPU in float32 from the bf16 tensors. Run as a script to survey all 1,368 heads at one query:
  <sca env python> attn_focus_data.py --capture DIR --step 20 --query ROW COL [--ratio 4] --out CSV
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

N0, TILE = 64, 8          # FLUX.1 native grid side and inference tile, Section 2.2
AXES_DIM, THETA = (16, 56, 56), 10000  # FLUX.1 RoPE (FluxTransformer2DModel defaults)


def rope_tables(ids: torch.Tensor):
    from diffusers.models.transformers.transformer_flux import FluxPosEmbed
    return FluxPosEmbed(theta=THETA, axes_dim=list(AXES_DIM))(ids)


def rotate(x: torch.Tensor, cs) -> torch.Tensor:
    """x [S, D] float32 rotated by FLUX's RoPE tables (cos, sin) [S, D] (diffusers apply_rotary_emb, real form)."""
    from diffusers.models.embeddings import apply_rotary_emb
    return apply_rotary_emb(x[None, :, None, :], cs, sequence_dim=1)[0, :, 0, :]


class Capture:
    def __init__(self, directory: Path, step: int):
        self.dir = Path(directory)
        blob = torch.load(self.dir / f"qk_step{step:02d}.pt", map_location="cpu", weights_only=False)
        self.q, self.k = blob["q"], blob["k"]                 # [57, T + N, 24, 128] bf16
        self.sigma, self.step = float(blob["sigma"]), int(blob["step"])
        self.T = int(blob["text_len"])
        self.img_ids = blob["img_ids"].float()                 # [N, 3] = (0, row, col)
        self.txt_ids = blob["txt_ids"].float()
        self.N = self.img_ids.shape[0]
        self.n = int(round(math.sqrt(self.N)))
        assert self.n * self.n == self.N and self.q.shape[1] == self.T + self.N
        ids = torch.cat([self.txt_ids, self.img_ids], 0)
        self.cs = rope_tables(ids)
        self.d = self.q.shape[-1]

    # ----------------------------------------------------------------------------------------------- geometry
    def window(self, row: int, col: int, r: int):
        """Top-left corner and side of the emulated local window at ratio r for the query token (row, col)."""
        w = self.n // r
        t = max(1, round(TILE * w / N0))
        def start(p):
            tau = p // t
            return int(np.clip(t * tau + t // 2 - w // 2, 0, self.n - w))
        return start(row), start(col), w, t

    def centres(self, r: int):
        """Block-centre token (rows, cols) of the emulated global operator at ratio r: kappa(b) = r b + floor(r/2)."""
        g = np.arange(self.n // r) * r + r // 2
        return g, g

    # ----------------------------------------------------------------------------------------------- attention rows
    def qk(self, block: int, head: int):
        return self.q[block, :, head].float(), self.k[block, :, head].float()

    def full(self, block: int, head: int, row: int, col: int):
        q, k = self.qk(block, head)
        i = self.T + row * self.n + col
        qr = rotate(q[i:i + 1], (self.cs[0][i:i + 1], self.cs[1][i:i + 1]))
        kr = rotate(k, self.cs)
        a = torch.softmax((kr @ qr[0]) / math.sqrt(self.d), dim=0).numpy().astype(np.float64)
        return a[self.T:].reshape(self.n, self.n), float(a[:self.T].sum())

    def local(self, block: int, head: int, row: int, col: int, r: int):
        """Emulated local operator: softmax over text keys and the window keys at their own positions."""
        q, k = self.qk(block, head)
        r0, c0, w, _ = self.window(row, col, r)
        rows, cols = np.meshgrid(np.arange(r0, r0 + w), np.arange(c0, c0 + w), indexing="ij")
        idx = self.T + (rows * self.n + cols).ravel()
        i = self.T + row * self.n + col
        qr = rotate(q[i:i + 1], (self.cs[0][i:i + 1], self.cs[1][i:i + 1]))[0]
        keys = torch.cat([torch.arange(self.T), torch.as_tensor(idx)])
        kr = rotate(k[keys], (self.cs[0][keys], self.cs[1][keys]))
        a = torch.softmax((kr @ qr) / math.sqrt(self.d), dim=0).numpy().astype(np.float64)
        out = np.zeros((self.n, self.n))
        out[r0:r0 + w, c0:c0 + w] = a[self.T:].reshape(w, w)
        return out, float(a[:self.T].sum())

    def global_(self, block: int, head: int, row: int, col: int, r: int):
        """Emulated global operator: block-centre keys at coarse positions b, the query at c(p) = floor(p / r)."""
        q, k = self.qk(block, head)
        g = self.n // r
        cr, cc = self.centres(r)
        rows, cols = np.meshgrid(cr, cc, indexing="ij")
        idx = torch.as_tensor(self.T + (rows * self.n + cols).ravel())
        b1, b2 = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        kid = torch.zeros(g * g, 3)
        kid[:, 1], kid[:, 2] = torch.as_tensor(b1.ravel()).float(), torch.as_tensor(b2.ravel()).float()
        qid = torch.tensor([[0.0, row // r, col // r]])
        cs_k, cs_q = rope_tables(kid), rope_tables(qid)
        cs_t = (self.cs[0][:self.T], self.cs[1][:self.T])
        i = self.T + row * self.n + col
        qr = rotate(q[i:i + 1], cs_q)[0]
        kr = torch.cat([rotate(k[:self.T], cs_t), rotate(k[idx], cs_k)], 0)
        a = torch.softmax((kr @ qr) / math.sqrt(self.d), dim=0).numpy().astype(np.float64)
        return a[self.T:].reshape(g, g), float(a[:self.T].sum())

    def split(self, full_img: np.ndarray, row: int, col: int, r: int):
        """near and far parts of the image attention, each renormalised, and m = near share of the image mass."""
        r0, c0, w, _ = self.window(row, col, r)
        inside = np.zeros_like(full_img, dtype=bool)
        inside[r0:r0 + w, c0:c0 + w] = True
        tot = full_img.sum()
        m = float(full_img[inside].sum() / tot)
        near = np.where(inside, full_img, 0.0)
        far = np.where(inside, 0.0, full_img)
        return near / max(near.sum(), 1e-30), far / max(far.sum(), 1e-30), m


def survey(cap: Capture, row: int, col: int, r: int, out: Path, archive: Path | None = None, bin_: int = 8):
    """Per head: near share m at this query (ratio r window), text mass, and the archive's pi and share within 16."""
    z = np.load(archive) if archive is not None else None
    rows = []
    for block in range(cap.q.shape[0]):
        for head in range(cap.q.shape[2]):
            img, txt = cap.full(block, head, row, col)
            _, _, m = cap.split(img, row, col, r)
            rec = dict(block=block, head=head, m_query=m, text_mass_query=txt,
                       peak_share=float(img.max() / img.sum()))
            if z is not None:
                mm = z["mass_mean"][:, block, head, bin_]
                rec.update(pi_r4=float(z["s4_pi_head"][block, head, bin_]), pi_r2=float(z["s2_pi_head"][block, head, bin_]),
                           share16=float(mm[4] / (1 - mm[0])))
            rows.append(rec)
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--query", type=int, nargs=2, required=True, metavar=("ROW", "COL"))
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cap = Capture(a.capture, a.step)
    print(f"sigma {cap.sigma:.4f} step {cap.step} T {cap.T} n {cap.n}")
    survey(cap, a.query[0], a.query[1], a.ratio, a.out, a.archive)
