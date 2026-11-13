"""Data of the sampler-stages explorer (SCAComp.stages, web/assets/js/c_stages.js): method step 3, one table for every
size and sampler stage.

The run: progressive SCA at 4096^2 for the prompt of the paper's attention figures (Aesthetic-4K
a4k_eval_4096_minus8343371271909676658, seed 0; fig_attn_focus.TARGET). Its sidecar JSON records every stage's size
and the noise level of every step (stage_sigmas: 30 steps at 1024^2 from sigma 1, then 17 at 2048^2 from 0.81 and 10
at 4096^2 from 0.61), each SCA stage's settings (pi_scale 2 and 4, window 64 at these noise levels: the hand-set
wider windows act only above sigma 0.82, before these stages start), the sha256 of the weight table it read, and
pi_trace, the mean weight over all 1,368 heads at every SCA step.
The table: results/evidence/calib/k1_1024.npz (asserted to be the file whose sha256 the run recorded). Every weight
is computed by the code the sampler runs, sca.attn.PiTable (linear in sigma between bin centres), at the stage's
scale, and the mean over all heads is asserted equal to the run's pi_trace at every step. Nothing is typed.
Exported: the step list (stage, size, ratio r, sigma, the mean weight over all heads), block 12's 24 heads (the
block of the heads and emulation components) with their weight at every SCA step, and block 12's rows of the r = 2
and r = 4 tables over the populated noise bins, heads ordered by their mean r = 4 weight (as the paper's method
figure orders them).

Run (repository root, CPU, about half a minute, mostly importing torch):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_stages.py
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
REPO = os.path.dirname(WEB)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "paper", "figures", "src"))

import fig_attn_focus as faf  # noqa: E402
from sca.attn import PiTable  # noqa: E402

OUT = os.path.join(WEB, "assets", "data", "stages")
TABLE = os.path.join(REPO, "results", "evidence", "calib", "k1_1024.npz")
BLOCK = faf.BLOCK            # zero-based block 12, the block of the paper's attention figures


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    side = json.load(open(str(faf.TARGET).replace(".png", ".json")))
    cfg = side["config"]
    assert sha256(TABLE) == cfg["sca"]["pi_table_sha256"], "the run read another table"
    shapes = [int(s["res"].split("x")[0]) for s in side["stage_shapes"]]
    sca_stages = {int(s["res"].split("x")[0]): s for s in cfg["sca"]["stages"]}
    trace = {(t["stage"], round(t["sigma"], 9)): t["pi_mean"] for t in side["pi_trace"]}
    z = np.load(TABLE)
    centers, pop = z["sigma_centers"], z["populated"].astype(bool)
    steps, b12 = [], []
    tables = {}
    for si, (size, sig) in enumerate(zip(shapes, side["stage_sigmas"])):
        st = sca_stages.get(size)
        if st is not None:
            assert all(s < min(w[0] for w in st["win_sched"]) for s in sig), "a wider window would act in this stage"
            assert st["pi_variant"] == "head" and st["window"] == st["n0"]
            r = int(st["pi_scale"])
            tab = PiTable.load(TABLE, "head", target_scale=r)
            assert tab.scale == r
            tables[r] = tab
        for s in sig:
            rec = dict(stage=si, size=size, sigma=round(float(s), 5))
            if st is not None:
                pi = np.asarray(tab(float(s)))            # [57, 24], the sampler's own lookup
                mean = float(pi.mean())
                ref = trace[(si, round(float(s), 9))]
                assert abs(mean - ref) < 1e-9, (si, s, mean, ref)
                rec.update(r=r, pi_mean=round(mean, 4))
                b12.append([round(float(v), 3) for v in pi[BLOCK]])
            else:
                b12.append(None)
            steps.append(rec)
    cols = np.where(pop)[0][::-1]                         # populated bins, sigma falling left to right
    order = np.argsort(-tables[4].table[BLOCK][:, cols].mean(1), kind="stable")
    rec = dict(
        block=BLOCK + 1, order=[int(h) + 1 for h in order], steps=steps, b12=b12,
        bins=[round(float(centers[c]), 4) for c in cols],
        table={str(r): [[round(float(t.table[BLOCK, h, c]), 3) for c in cols] for h in range(t.table.shape[1])] for r, t in tables.items()},
        stages=[dict(size=size, n=len(sig), sigma0=round(float(sig[0]), 3), r=int(sca_stages[size]["pi_scale"]) if size in sca_stages else None)
                for size, sig in zip(shapes, side["stage_sigmas"])],
        prompt=cfg["prompt"],
        source=dict(script="web/build/v5_build_stages.py", run=os.path.relpath(str(faf.TARGET).replace(".png", ".json"), "/scratch/09032/saini_2"),
                    table="results/evidence/calib/k1_1024.npz", lookup="sca/attn.py PiTable"))
    with open(os.path.join(OUT, "data.js"), "w") as f:
        f.write("/* Generated by web/build/v5_build_stages.py; do not edit. */\n")
        f.write("window.SCA_DATA = window.SCA_DATA || {};\n")
        f.write("window.SCA_DATA.stages = " + json.dumps(rec, separators=(",", ":")) + ";\n")
    print("stages", rec["stages"], "steps", len(steps), "bins", rec["bins"])
    print("pi_mean", [s.get("pi_mean") for s in steps if "pi_mean" in s][:5], "...")
    print("data", os.path.getsize(os.path.join(OUT, "data.js")), "bytes")


if __name__ == "__main__":
    main()
