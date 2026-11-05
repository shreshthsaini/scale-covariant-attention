"""Split the variance of per-head attention reach and of the calibrated mix into
between-layer, between-head-within-layer, and across-noise-level-within-head parts.
Input: calib/k1_1024.npz (FLUX.1-dev, 32 prompts at 1024^2). CPU only.
Run: <sca env python> -I variance_split.py calib/k1_1024.npz"""
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
sig = z['sigma_centers']
def split(X):
    v = X.reshape(-1); tot = v.var()
    lm = X.mean(axis=(1, 2)); hm = X.mean(axis=2)
    bl = ((lm[:, None, None] - v.mean()) ** 2 * np.ones_like(X)).mean()
    hl = ((hm[:, :, None] - lm[:, None, None]) ** 2 * np.ones_like(X)).mean()
    wh = ((X - hm[:, :, None]) ** 2).mean()
    t = X.mean(axis=(0, 1)); te = ((t - t.mean()) ** 2).mean()
    return bl / tot, hl / tot, wh / tot, te / tot, float(t.min()), float(t.max())
for label, sel in (('all populated bins', z['populated']), ('post-gate bins (sigma < 0.89)', z['populated'] & (sig < 0.89))):
    b = np.where(sel)[0]
    print(f'== {label}: sigma {np.round(sig[b], 2).tolist()}')
    mm = z['mass_mean'][:, :, :, b]; text = mm[0]
    rows = [(f'measured image share within {r} tokens', mm[i + 1] / np.clip(1 - text, 1e-6, None)) for i, r in enumerate(z['radii'].tolist())]
    rows += [('measured text mass', text)] + [(f'fitted pi, ratio {s}', z[f's{s}_pi_head'][:, :, b]) for s in (2, 4) if f's{s}_pi_head' in z.files]
    for name, X in rows:
        bl, hl, wh, te, lo, hi = split(X)
        print(f'  {name:40s} mean {X.mean():.3f} | layer {bl:.3f} | head within layer {hl:.3f} | noise level within head {wh:.3f} | head-averaged curve {te:.4f} (range {lo:.2f} to {hi:.2f})')
