"""Process videos of the v5 page ("each image beside the run that made it"): web/assets/video/v5/process_<id>.mp4.

What each shows (1600 x 900, three 488 px panels): left, progressive SCA's final 4096^2 image; middle, its recorded
run, the decoded estimate x0 of each of its 57 steps (HRDiT's three stages: 30 steps at 1024^2 with the model's own
attention, 17 at 2048^2 reading the reach table at r = 2, 10 at 4096^2 at r = 4), with the stage and the table ratio
in a corner chip and the stage rail below; right, AReA's run of the same prompt and seed (28 steps at 4096^2: pooled
coarse keys, then shrinking fine windows, from its trace). Both runs carry the trace's native-pixel window (a 512 px
box at the same place) and an inset of that window as decoded at each step (tagged 1:1, or 2x / 4x upsampled while
progressive SCA is at 2048^2 / 1024^2). One clock normalised to the video: both runs spread evenly over RUN seconds
(real clean seconds per image are printed in the panel heads), a hold on the finals, and a cross-fade back to the
first frame (seamless loop).

Prompts (rule): the showcase prompts with both a progressive SCA trace and an AReA trace
(/scratch/09032/saini_2/sca/web_assets/trace/{sca_v2_prog,area_sc}), one per category by the paper hero's seeded
draw (random.Random("0:<category>").choice over the category's sorted traced ids), the first five of the hero's
category order (portrait, animal, landscape, architecture, dense). Traces are checked complete, not synthetic, and
pixel-identical to the campaign PNG (trace.json final.pixels_identical).

Run from the repository root:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_process.py [--only <id>] [--k 5]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_media as M  # noqa: E402

W, H = 1600, 900
FPS = 30
PANEL, GAP = 488, 18
INSET = 150
LEAD, RUN, HOLD, BACK = 0.4, 11.0, 2.2, 0.6
MAX_BYTES = 2_500_000
TRACE = Path("/scratch/09032/saini_2/sca/web_assets/trace")
SHOWCASE = M.WEB / "build" / "showcase_prompts.jsonl"
ORDER = ["portrait", "animal", "landscape", "architecture", "dense", "product"]
BUILD = M.BUILD / "process"


# ------------------------------------------------------------------------------------------------ selection
def showcase() -> dict:
    return {json.loads(l)["id"]: json.loads(l) for l in SHOWCASE.read_text().splitlines() if l.strip()}


def traced() -> list:
    return sorted(p.name for p in (TRACE / "sca_v2_prog").iterdir()
                  if (p / "trace.json").exists() and (TRACE / "area_sc" / p.name / "trace.json").exists())


def selection(k: int) -> list:
    sc, tr = showcase(), traced()
    out = []
    for c in ORDER:
        ids = sorted(i for i in tr if sc[i]["category"] == c)
        if ids:
            out.append(random.Random(f"0:{c}").choice(ids))
    return out[:k]


def load_trace(method: str, sid: str) -> dict:
    d = TRACE / method / sid
    t = json.loads((d / "trace.json").read_text())
    assert t["complete"] and not t["synthetic"], (method, sid)
    assert t["final"].get("pixels_identical") is True, (method, sid, "final differs from the campaign PNG")
    assert t["steps_total"] == len(t["steps"]), (method, sid)
    return t


# ------------------------------------------------------------------------------------------------ context
def context(sid: str) -> dict:
    p, a = load_trace("sca_v2_prog", sid), load_trace("area_sc", sid)
    assert p["prompt"] == a["prompt"] and p["seed"] == a["seed"] and p["derived_seed"] == a["derived_seed"]
    assert p["box_xys"] == a["box_xys"] and p["res"] == a["res"] == "4096x4096"
    dp, da = TRACE / "sca_v2_prog" / sid, TRACE / "area_sc" / sid

    def prev(d, st):
        return Image.open(d / st["preview"]).convert("RGB").resize((PANEL, PANEL), Image.LANCZOS)

    def crop(d, st):
        return Image.open(d / st["crop"]).convert("RGB").resize((INSET, INSET), Image.LANCZOS)

    fin = Image.open(dp / "final.png").convert("RGB")
    final = fin.reduce(4).resize((PANEL, PANEL), Image.LANCZOS)
    return dict(sid=sid, p=p, a=a, pp=[prev(dp, s) for s in p["steps"]], pc=[crop(dp, s) for s in p["steps"]],
                ap=[prev(da, s) for s in a["steps"]], ac=[crop(da, s) for s in a["steps"]], final=final,
                title=showcase()[sid]["title"], sec=dict(prog=M.num("main.prog.time_clean", "display"),
                                                         area=M.num("main.area.time_clean", "display")))


# ------------------------------------------------------------------------------------------------ drawing
def step_at(t: float, n: int) -> tuple:
    """(index, blend from the previous index) of a run of n steps spread over RUN seconds after LEAD."""
    if t < LEAD:
        return 0, 1.0
    f = (t - LEAD) / RUN * n
    i = min(n - 1, int(f))
    a = float(M.smoothstep((f - i) / 0.35)) if i > 0 else 1.0
    if t >= LEAD + RUN:
        return n - 1, 1.0
    return i, a


def chip(img, xy, text, size=22, color=None, anchor="lt"):
    M.chip(img, xy, text, size, anchor=anchor, bg=(10, 12, 11, 190), fg=M.rgb(color or M.PAL["ink"]) + (255,),
           weight=600, radius=6)


def panel_run(img, d, x, y, frames, crops, st_list, i, a, box, inset_corner, kind, ctx):
    im = frames[i] if a >= 1 or i == 0 else Image.blend(frames[i - 1], frames[i], a)
    mask = Image.new("L", (PANEL, PANEL), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, PANEL - 1, PANEL - 1], radius=8, fill=255)
    img.paste(im, (x, y), mask)
    st = st_list[i]
    # the native window: box on the run, inset of the step's decode of it
    s = PANEL / 4096
    bx, by, bs = box
    d.rectangle([x + bx * s, y + by * s, x + (bx + bs) * s, y + (by + bs) * s], outline=M.rgb(M.PAL["crop"]),
                width=2)
    cr = crops[i] if a >= 1 or i == 0 else Image.blend(crops[i - 1], crops[i], a)
    ix = x + PANEL - INSET - 10 if inset_corner[0] == "r" else x + 10
    iy = y + PANEL - INSET - 10 if inset_corner[1] == "b" else y + 10
    img.paste(cr, (ix, iy))
    d.rectangle([ix - 1, iy - 1, ix + INSET, iy + INSET], outline=M.rgb(M.PAL["crop"]), width=2)
    if kind == "prog":
        scale = st.get("crop_scale", 1.0)
        tag = "1:1" if abs(scale - 1) < 1e-6 else f"{scale:g}× upsampled"
        stage = st["stage"]
        lab = {0: "1024² · own attention", 1: "2048² · table r = 2", 2: "4096² · table r = 4"}[stage]
        col = M.PAL["ink"] if stage == 0 else M.PAL["sca"]
    else:
        tag = "1:1"
        lab = (f"coarse · pooled keys, f = {st['area_size']}" if st["phase"] == "coarse"
               else f"fine · windows of {st['area_size']}")
        col = M.PAL["area"]
    chip(img, (ix + 6, iy + 6), tag, size=18, color=M.PAL["ink"])
    chip(img, (x + 12, y + 12), lab, size=22, color=col)


def rail(d, x, y, w, counts, names, cur, colors, n_total):
    gap = 6
    span = w - gap * (len(counts) - 1)
    xx = x
    for si, (cnt, nm, col) in enumerate(zip(counts, names, colors)):
        sw = span * cnt / n_total
        d.rounded_rectangle([xx, y, xx + sw, y + 8], radius=4, fill=M.rgb(M.PAL["rule"]))
        first = sum(counts[:si])
        if cur >= first:
            frac = min(1.0, (cur - first + 1) / cnt)
            d.rounded_rectangle([xx, y, xx + max(8, sw * frac), y + 8], radius=4, fill=M.rgb(col))
        on = first <= cur < first + cnt
        d.text((xx, y + 18), nm, font=M.font(20, "sans", 600 if on else 400),
               fill=M.rgb(M.PAL["ink"] if on else M.PAL["ink3"]), anchor="la")
        xx += sw + gap


def core(ctx, t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), M.rgb(M.PAL["screen"]))
    d = ImageDraw.Draw(img, "RGBA")
    x0 = (W - 3 * PANEL - 2 * GAP) // 2
    y0 = 200
    xs = [x0 + i * (PANEL + GAP) for i in range(3)]
    box = ctx["p"]["box_xys"]
    # inset corner: bottom-right unless the box sits there
    s = PANEL / 4096
    bx1, by1 = (box[0] + box[2]) * s, (box[1] + box[2]) * s
    corner = "rb" if not (bx1 > PANEL - INSET - 20 and by1 > PANEL - INSET - 20) else "lb"
    # left: the final image
    mask = Image.new("L", (PANEL, PANEL), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, PANEL - 1, PANEL - 1], radius=8, fill=255)
    img.paste(ctx["final"], (xs[0], y0), mask)
    d.rectangle([xs[0] + box[0] * s, y0 + box[1] * s, xs[0] + (box[0] + box[2]) * s, y0 + (box[1] + box[2]) * s],
                outline=M.rgb(M.PAL["crop"]), width=2)
    np_, na = len(ctx["pp"]), len(ctx["ap"])
    ip, ap_ = step_at(t, np_)
    ia, aa = step_at(t, na)
    heads = [("Final image", "progressive SCA \u00b7 4096 \u00d7 4096", M.PAL["ink"]),
             ("Progressive SCA", f"{ctx['sec']['prog']} s per image \u00b7 step {ip + 1} of {np_}", M.PAL["sca"]),
             ("AReA", f"{ctx['sec']['area']} s per image \u00b7 step {ia + 1} of {na}", M.PAL["area"])]
    for x, (h1, h2, col) in zip(xs, heads):
        d.text((x, y0 - 50), h1, font=M.font(30, "sans", 600), fill=M.rgb(col), anchor="ls")
        d.text((x, y0 - 16), h2, font=M.font(23, "sans", 400), fill=M.rgb(M.PAL["ink2"]), anchor="ls",
               features=["tnum"])
    panel_run(img, d, xs[1], y0, ctx["pp"], ctx["pc"], ctx["p"]["steps"], ip, ap_, box, corner, "prog", ctx)
    panel_run(img, d, xs[2], y0, ctx["ap"], ctx["ac"], ctx["a"]["steps"], ia, aa, box, corner, "area", ctx)
    # rails and step counters
    ry = y0 + PANEL + 22
    stg = ctx["p"]["progressive"]
    counts = [stg["base_steps"]] + list(stg["highres_steps"])
    assert sum(counts) == np_
    rail(d, xs[1], ry, PANEL, counts, ["1024²", "2048²", "4096²"], ip,
         [M.PAL["ink3"], M.PAL["sca"], M.PAL["sca"]], np_)
    nco = sum(1 for st in ctx["a"]["steps"] if st["phase"] == "coarse")
    rail(d, xs[2], ry, PANEL, [nco, na - nco], ["coarse", "fine"], ia, [M.PAL["far"], M.PAL["area"]], na)
    # footer: the prompt's title and the clock note
    fy = ry + 82
    d.text((xs[0], fy), ctx["title"], font=M.font(28, "serif", 600), fill=M.rgb(M.PAL["ink"]), anchor="ls")
    d.text((xs[0], fy + 32), "same prompt and seed; both runs shown over the same 11 s; red box: the 512 px "
                             "window of the insets", font=M.font(21, "sans", 400), fill=M.rgb(M.PAL["ink3"]),
           anchor="ls")
    return img


def draw(ctx, k: int) -> Image.Image:
    t = k / FPS
    t_back = LEAD + RUN + HOLD
    total = t_back + BACK
    if t < t_back:
        return core(ctx, t)
    a = float(M.smoothstep((t - t_back) / (total - t_back - 1 / FPS)))
    return Image.blend(core(ctx, t_back - 1 / FPS), core(ctx, 0.0), a)


_CTX: dict = {}


def _work(task):
    sid, idxs, outdir = task
    if sid not in _CTX:
        _CTX.clear()
        _CTX[sid] = context(sid)
    for k in idxs:
        draw(_CTX[sid], k).save(Path(outdir) / f"{k:05d}.png", compress_level=1)
    return len(idxs)


def build(sid: str, order: int, procs: int):
    n = round((LEAD + RUN + HOLD + BACK) * FPS)
    name = f"process_{sid.split('_', 2)[-1]}"
    fdir = BUILD / name / f"frames_n{n}"
    fdir.mkdir(parents=True, exist_ok=True)
    chunk = max(6, math.ceil(n / procs))
    tasks = [(sid, list(range(i, min(n, i + chunk))), str(fdir)) for i in range(0, n, chunk)]
    t0 = time.time()
    with Pool(min(procs, len(tasks))) as pool:
        sum(pool.imap_unordered(_work, tasks))
    print(f"{name}: {n} frames in {time.time() - t0:.1f} s", flush=True)
    mp4 = M.OUT / f"{name}.mp4"
    enc = M.encode_to_size(fdir / "%05d.png", mp4, FPS, n, MAX_BYTES, crf_floor=20, preset="veryslow", gop=n,
                           passlog=BUILD / name / "x264pass", x264="aq-mode=3", threads=16)
    p = M.check_video(mp4, max_bytes=MAX_BYTES, w=W, h=H)
    t_rest = LEAD + RUN + HOLD * 0.5
    poster = M.poster_webp(Image.open(fdir / f"{round(t_rest * FPS):05d}.png"), M.OUT / f"{name}.webp",
                           quality=84)
    pt, at = load_trace("sca_v2_prog", sid), load_trace("area_sc", sid)
    title = showcase()[sid]["title"]
    alt = (f"{title}: progressive SCA's final 4096 by 4096 image beside its recorded run, the estimate growing from "
           f"1024 to 2048 to 4096 pixels over 57 steps, and AReA's 28-step run of the same prompt and seed, each with "
           f"a native-pixel inset of the same window.")
    cap = (f"{title}: the final image beside the runs that made it. Progressive SCA: 30 steps at 1024 x 1024, 17 at "
           f"2048 x 2048 (reach table at r = 2), 10 at 4096 x 4096 (r = 4). AReA: 28 steps at 4096 x 4096. Both runs "
           f"shown over the same 11 s; prompts: one traced showcase prompt per category by seeded draw.")
    e = M.entry_for(name, mp4, poster, loop=True, alt=alt, caption=cap,
                    sources=dict(prompt_id=sid, prompt=pt["prompt"], seed=pt["seed"], derived_seed=pt["derived_seed"],
                                 progressive_trace=str(TRACE / "sca_v2_prog" / sid),
                                 area_trace=str(TRACE / "area_sc" / sid), final_png=pt["campaign"]["png"],
                                 box_xys=pt["box_xys"],
                                 rule=("traced showcase prompts, one per category by random.Random('0:<category>')"
                                       ".choice over the category's sorted traced ids, hero category order")),
                    script="v5_process.py", order=60 + order,
                    extra=dict(role=f"process_{order + 1}", title=title, encode=enc, poster_time=round(t_rest, 3),
                               timing=dict(lead=LEAD, run=RUN, hold=HOLD, back=BACK)))
    M.manifest_put(e)
    sd = M.BUILD / "sheets"
    M.contact_sheet(mp4, sd / f"{name}_sheet.png", n=12, cols=4, tile_w=480)
    M.seam_sheet(mp4, sd / f"{name}_seam.png", tile_w=640)
    print(json.dumps(dict(name=name, **p, enc=enc)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--only", default=None)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--look", type=str, default=None, help="comma list of times: look frames of the first pick")
    a = ap.parse_args()
    sel = selection(a.k)
    print("selection:", sel, flush=True)
    if a.look:
        ctx = context(a.only or sel[0])
        outs = []
        for tv in a.look.split(","):
            p = BUILD / "look" / f"{ctx['sid']}_{float(tv):.2f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            draw(ctx, round(float(tv) * FPS)).save(p)
            outs.append(p)
        M.frames_sheet(outs, BUILD / "look" / f"{ctx['sid']}_look.png", cols=2, tile_w=800, label=ctx["sid"])
        print(BUILD / "look" / f"{ctx['sid']}_look.png")
        return
    for i, sid in enumerate(sel):
        if a.only and sid != a.only:
            continue
        build(sid, i, a.procs)


if __name__ == "__main__":
    main()
