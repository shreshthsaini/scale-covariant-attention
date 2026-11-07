"""Variance split of per-head attention reach and of the calibrated mix (CPU only).

For each calibration archive, every quantity X[layer, head, noise bin] is split into three parts of its total
variance over all (layer, head, bin) cells: between layers, between heads within a layer, and across noise levels
within a head (the three sum to 1). "Head" in the paper means layer plus head within layer, i.e. 1 - noise_within_head.
head_avg_curve is the variance of the head-averaged curve over noise levels, as a share of the same total: the
part a schedule shared by all heads could follow.

Quantities: the image share of each head's attention within r tokens of the query (mass_mean[r] / (1 - text mass),
r in the archive's radii), the text mass, and the fitted weights pi at emulated ratios 2 and 4 where present.
Two noise ranges: all populated bins, and the post-gate bins (sigma < 0.89).

Inputs (read-only): calib/{k1_1024,krea_1024,flux2_1024,k1_1280b}.npz.
Output: results/heads_view/variance_split.csv with columns
  checkpoint,res,sigma_range,quantity,radius,layer,head_within_layer,noise_within_head,head_avg_curve,n_heads,n_bins
and the head-averaged curve range in curve_min, curve_max (for the paper's "it narrows" sentence).

Run from the repository root:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I scripts/variance_split.py
Moved from .agent/notes/positioning_20261105/evidence/variance_split.py (same arithmetic; that script printed
evidence/variance_split_out.txt, which this CSV reproduces).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = [("flux1_dev", "1024x1024", "calib/k1_1024.npz"), ("flux1_krea_dev", "1024x1024", "calib/krea_1024.npz"),
            ("flux2_dev", "1024x1024", "calib/flux2_1024.npz"), ("flux1_dev", "1280x1280", "calib/k1_1280b.npz")]
OUT = ROOT / "results" / "heads_view" / "variance_split.csv"
GATE = 0.89


def split(X: np.ndarray):
    """(layer, head within layer, noise within head, head-averaged curve, curve min, curve max) of X[l, h, b]."""
    v = X.reshape(-1)
    tot = v.var()
    lm = X.mean(axis=(1, 2))
    hm = X.mean(axis=2)
    bl = ((lm[:, None, None] - v.mean()) ** 2 * np.ones_like(X)).mean()
    hl = ((hm[:, :, None] - lm[:, None, None]) ** 2 * np.ones_like(X)).mean()
    wh = ((X - hm[:, :, None]) ** 2).mean()
    t = X.mean(axis=(0, 1))
    te = ((t - t.mean()) ** 2).mean()
    return bl / tot, hl / tot, wh / tot, te / tot, float(t.min()), float(t.max())


def main():
    rows = []
    for ckpt, res, rel in ARCHIVES:
        z = np.load(ROOT / rel, allow_pickle=False)
        sig = z["sigma_centers"]
        for label, sel in (("all", z["populated"]), ("post_gate", z["populated"] & (sig < GATE))):
            b = np.where(sel)[0]
            mm = z["mass_mean"][:, :, :, b]
            text = mm[0]
            qs = [("image_share", r, mm[i + 1] / np.clip(1 - text, 1e-6, None))
                  for i, r in enumerate(z["radii"].tolist())]
            qs += [("text_mass", "", text)]
            qs += [(f"pi_ratio{s}", "", z[f"s{s}_pi_head"][:, :, b]) for s in (2, 4) if f"s{s}_pi_head" in z.files]
            for name, radius, X in qs:
                bl, hl, wh, te, lo, hi = split(X)
                rows.append(dict(checkpoint=ckpt, res=res, sigma_range=label, quantity=name, radius=radius,
                                 layer=f"{bl:.6f}", head_within_layer=f"{hl:.6f}", noise_within_head=f"{wh:.6f}",
                                 head_avg_curve=f"{te:.6f}", n_heads=X.shape[0] * X.shape[1], n_bins=X.shape[2],
                                 mean=f"{X.mean():.6f}", curve_min=f"{lo:.6f}", curve_max=f"{hi:.6f}"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
