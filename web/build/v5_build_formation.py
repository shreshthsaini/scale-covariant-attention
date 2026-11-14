"""Data of the formation explorer (SCAComp.formation, web/assets/js/c_formation.js): pick a patch, watch it form.

Runs: the recorded progressive SCA traces (/scratch/09032/saini_2/sca/web_assets/trace/sca_v2_prog/<id>: the
decoded estimate x0 of each of the 57 steps as a 1024 px preview, a 512 px window at native size decoded at every
step, the mix weight pi of all 57 x 24 heads that every step multiplied in, and the final 4096^2 PNG, pixel-identical
to the campaign image). Prompts: the first three of the process videos' rule (web/build/v5_process.py selection:
traced showcase prompts, one per category by the paper hero's seeded draw, in the hero's category order), loaded and
checked with v5_process.load_trace (complete, not synthetic, final equals the campaign PNG).
Keyframes per prompt (the rule): 7 steps of the 1024^2 stage, 5 of the 2048^2 stage and 4 of the 4096^2 stage,
evenly spaced within each stage with its first and last step, then the final image.
Weights: block 12's heads 4 and 7 (the paper's near- and far-reaching examples, v6_method_assets.NEAR_HEAD and
FAR_HEAD), read from the trace at each keyframe (what the run used; none in the 1024^2 stage, which uses the model's
own attention).
Geometry constants come from code: n0 and the query tile from attn_focus_data (N0, TILE), pixels per token from
v6_scaling.PATCH.
Images (web/assets/data/formation/img/<id>/): k<j>.webp, the keyframe's x0 preview (1024 px); c<j>.webp, the
trace's own window decoded at that keyframe, stored at the stage's true resolution (the trace upsamples it by its
crop_scale); f_<c>_<r>.webp, the final image in 1024 px native tiles; f.webp, the final image at 1024 px.
Presets: the trace's window, then two windows of highest edge energy of the final image (zoomlib.regions, 512 px,
at least 1.4 spans from each other and from the trace's window).

Run (repository root, CPU, about a minute):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_formation.py
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
REPO = os.path.dirname(WEB)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "paper", "figures", "src"))

from PIL import Image  # noqa: E402

import attn_focus_data as afd  # noqa: E402
import v5_process as vp  # noqa: E402
import v6_method_assets as ma  # noqa: E402
import v6_scaling as vs  # noqa: E402
import zoomlib  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(WEB, "assets", "data", "formation")
PER_STAGE = (7, 5, 4)
VIEW_Q, CROP_Q, TILE_Q = 72, 80, 70


def keyframes(steps):
    out = []
    for s, k in enumerate(PER_STAGE):
        idx = [st["index"] for st in steps if st["stage"] == s]
        pick = sorted(set(int(round(x)) for x in np.linspace(idx[0], idx[-1], k)))
        assert len(pick) == k, (s, pick)
        out += pick
    return out


def presets(final, box, k=2, span=512):
    regs = zoomlib.regions([zoomlib.edge_map(final)], final.size, k=k + 6, span=span)
    bx, by = box[0] + box[2] / 2, box[1] + box[2] / 2
    out = [[int(bx), int(by)]]
    for x, y, _ in regs:
        if all((x - a) ** 2 + (y - b) ** 2 >= (1.4 * span) ** 2 for a, b in out):
            out.append([int(x), int(y)])
        if len(out) == k + 1:
            break
    return out


def one(sid):
    t = vp.load_trace("sca_v2_prog", sid)
    d = vp.TRACE / "sca_v2_prog" / sid
    odir = os.path.join(OUT, "img", sid)
    os.makedirs(odir, exist_ok=True)
    steps = t["steps"]
    kf = keyframes(steps)
    bl, h4, h7 = ma.BLOCK, ma.NEAR_HEAD, ma.FAR_HEAD
    frames = []
    for j, i in enumerate(kf):
        st = steps[i]
        assert st["index"] == i
        v = Image.open(d / st["preview"]).convert("RGB")
        v.save(os.path.join(odir, f"k{j}.webp"), "WEBP", quality=VIEW_Q, method=6)
        c = Image.open(d / st["crop"]).convert("RGB")
        sc = float(st["crop_scale"])
        true_px = int(round(c.size[0] / sc))
        if true_px != c.size[0]:
            c = c.resize((true_px, true_px), Image.LANCZOS)
        c.save(os.path.join(odir, f"c{j}.webp"), "WEBP", quality=CROP_Q, method=6)
        rec = dict(i=i, step=int(st["step"]), stage=int(st["stage"]), size=int(st["stage_res"].split("x")[0]),
                   sigma=round(float(st["sigma"]), 3), view=f"img/{sid}/k{j}.webp", crop=f"img/{sid}/c{j}.webp",
                   crop_px=true_px)
        pi = st.get("pi")
        if pi is not None:
            assert st["phase"] == "mixed", (sid, i, st["phase"])
            rec.update(near=round(float(pi[bl][h4]), 3), far=round(float(pi[bl][h7]), 3))
        frames.append(rec)
    fin = Image.open(d / t["final"]["png"]).convert("RGB")
    assert fin.size == (4096, 4096)
    fin.resize((1024, 1024), Image.LANCZOS).save(os.path.join(odir, "f.webp"), "WEBP", quality=VIEW_Q, method=6)
    T = zoomlib.TILE
    for r in range(4096 // T):
        for c in range(4096 // T):
            fin.crop((c * T, r * T, (c + 1) * T, (r + 1) * T)).save(os.path.join(odir, f"f_{c}_{r}.webp"), "WEBP",
                                                                     quality=TILE_Q, method=6)
    box = [int(v) for v in t["box_xys"]]
    rec = dict(sid=sid, title=vp.showcase()[sid]["title"], prompt=t["prompt"], seed=int(t["seed"]), frames=frames,
               final=dict(view=f"img/{sid}/f.webp", tiles=dict(path=f"img/{sid}/f_{{c}}_{{r}}.webp", size=T,
                                                                 cols=4096 // T, rows=4096 // T)),
               box=box, presets=presets(fin, box), size=4096)
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    sids = vp.selection(3)
    with Pool(len(sids)) as pool:
        recs = pool.map(one, sids)
    stages = []
    t = vp.load_trace("sca_v2_prog", sids[0])
    for s in range(3):
        sts = [st for st in t["steps"] if st["stage"] == s]
        stages.append(dict(size=int(sts[0]["stage_res"].split("x")[0]), steps=len(sts)))
    rec = dict(n0=afd.N0, tile=afd.TILE, patch=vs.PATCH, block=ma.BLOCK + 1, near=ma.NEAR_HEAD + 1, far=ma.FAR_HEAD + 1,
               stages=stages, steps_total=int(t["steps_total"]), runs=recs,
               source=dict(script="web/build/v5_build_formation.py", traces="/scratch/09032/saini_2/sca/web_assets/trace/sca_v2_prog",
                           rule="web/build/v5_process.py selection(3)"))
    with open(os.path.join(OUT, "data.js"), "w") as f:
        f.write("/* Generated by web/build/v5_build_formation.py; do not edit. */\n")
        f.write("window.SCA_DATA = window.SCA_DATA || {};\n")
        f.write("window.SCA_DATA.formation = " + json.dumps(rec, separators=(",", ":")) + ";\n")
    for r in recs:
        print(r["sid"], r["title"], "keyframes", [f["i"] for f in r["frames"]], "box", r["box"], "presets", r["presets"])
        print("   weights", [(f["i"], f.get("near"), f.get("far")) for f in r["frames"] if "near" in f])
    print(f"data {zoomlib.dir_bytes(OUT) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
