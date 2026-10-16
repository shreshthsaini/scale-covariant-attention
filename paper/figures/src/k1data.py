"""K1 calibration data for the paper figures.

Source: FLUX.1-dev, 1024x1024, 32 prompts, guidance 4.5, 28 steps, mu = max_shift.
data/k1_1024_pi.npz is a compact copy of /scratch/09032/saini_2/sca/calib/k1_1024.npz
(pi tables, sigma bins, and the per-cell sufficient statistics); data/k1_1024.json is the
merge report. Errors are recomputed here from the statistics with the same algebra as
sca/calib.py and checked against the report, so a figure never shows a copied number.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, ".."))
N_DOUBLE = 19  # FLUX.1: 19 double-stream blocks, then 38 single-stream blocks


def load():
    z = dict(np.load(os.path.join(HERE, "data", "k1_1024_pi.npz")))
    with open(os.path.join(HERE, "data", "k1_1024.json")) as fh:
        report = json.load(fh)
    return z, report


def rel_err(S, pi, axis=None):
    """sum ||O_full - (pi O_loc + (1 - pi) O_glob)||^2 / sum ||O_full||^2 (image queries)."""
    err = S[2] - 2 * pi * S[1] + pi ** 2 * S[0]
    return err.sum(axis) / np.maximum(S[4].sum(axis), 1e-300)


def errors(z, s):
    """Relative error of every pi variant at emulated scale s, recomputed from statistics."""
    S, pop = z[f"s{s}_stats"], z["populated"]
    centers = z["sigma_centers"]
    out = {
        "per_head": rel_err(S, z[f"s{s}_pi_head"]),
        "per_layer": rel_err(S, z[f"s{s}_pi_layer"][:, None]),
        "uniform": rel_err(S, z[f"s{s}_pi_uniform"][None, None]),
        "loc": rel_err(S, 1.0),
        "glob": rel_err(S, 0.0),
    }
    cs = np.sort(centers[pop])
    cands = np.concatenate([[0.0], (cs[1:] + cs[:-1]) / 2, [1.0]])
    sw = [(rel_err(S, (centers <= c).astype(float)[None, None]), c) for c in cands]
    out["switch"], out["sigma_star"] = min(sw)
    other = 4 if s == 2 else 2
    out["transfer_from"] = rel_err(S, z[f"s{other}_pi_head"])
    return {k: float(v) for k, v in out.items()}


def check(z, report):
    for s in (2, 4):
        e = errors(z, s)
        ref = report["scales"][str(s)]["rel_err"]
        for k in ("per_head", "per_layer", "uniform", "switch", "glob", "loc"):
            assert abs(e[k] - ref[k]) < 1e-9, (s, k, e[k], ref[k])
    return True
