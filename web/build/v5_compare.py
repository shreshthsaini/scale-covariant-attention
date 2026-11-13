"""Comparison clips of the v5 page: web/assets/video/v5/compare_p25.mp4, compare_p50.mp4, compare_p75.mp4.

What each shows. The three prompts of the paper's Figure 5 (paper/figures/v6/qual_main_provenance.json: the
evaluation prompts at the 25th, 50th and 75th percentile, nearest rank, of the paired ImageReward difference of
progressive SCA minus AReA over the 195 prompts at seed 0). For one prompt: AReA, HRDiT and progressive SCA side by
side, each lane headed by the method and its clean seconds per 4096^2 image; the three whole images, then one
synchronized slow push-in to native pixels (one video pixel per image pixel) on the same window of all three, a hold,
and a cross-fade back to the whole images (seamless loop). The only text: method names, seconds, the prompt's
percentile and the scale ("whole image" / "1:1 native pixels").

Window (rule): the panel-sized native window of highest mean edge energy averaged over the three images (the paper's
like-for-like crop rule, v6_common.best_windows over the three detail maps, 1/16 border margin).
Seconds: numbers.json main.area.time_clean, main.hrdit.time_clean, main.prog.time_clean (display strings), asserted
equal to the paper's \\hlSecArea, \\hlSecHrdit, \\hlSecProg. Images: the PNGs listed in the provenance file.

Run from the repository root:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_compare.py [--only 25|50|75]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_media as M  # noqa: E402

sys.path.insert(0, str(M.FIGSRC))
import v6_common as vc  # noqa: E402

W, H = 1600, 900
FPS = 30
PANEL = 488                       # square panel side (px) = the native window side
GAP = 18
MAX_BYTES = 2_500_000
T_WHOLE, T_IN, T_NATIVE, T_BACK = 1.0, 6.2, 1.8, 0.8
METHODS = [("area", "AReA", "area", "hlSecArea"), ("hrdit_dev_author", "HRDiT", "hrdit", "hlSecHrdit"),
           ("sca_v2_prog", "Progressive SCA", "prog", "hlSecProg")]
COLOR = {"area": "area", "hrdit_dev_author": "hrdit", "sca_v2_prog": "sca"}
BUILD = M.BUILD / "compare"
PAD = 32
ORD = {25: "25th", 50: "50th", 75: "75th"}


def rows() -> dict:
    prov = json.loads((M.V6 / "qual_main_provenance.json").read_text())
    return {r["percentile"]: r for r in prov["rows"]}, prov["selection"]["rule"]


def seconds() -> dict:
    mac = M.headline_macros()
    out = {}
    for m, name, key, macro in METHODS:
        disp = M.num(f"main.{key}.time_clean", "display")
        assert disp == mac[macro], (key, disp, mac[macro])
        out[m] = disp
    return out


def padded(path: str) -> Path:
    """Cached padded copy, keyed by the full source path (methods share file names, so the stem alone collides)."""
    import hashlib
    key = hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:12]
    out = BUILD / "src" / f"{key}_{Path(path).stem}_pad{PAD}.npy"
    if not out.exists():
        im = Image.open(path).convert("RGB")
        assert im.size == (4096, 4096)
        g = Image.new("RGB", (4096 + 2 * PAD, 4096 + 2 * PAD), M.rgb(M.PAL["screen"]))
        g.paste(im, (PAD, PAD))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.asarray(g))
    return out


def view(src: Image.Image, s: float, cx: float, cy: float, n: int) -> Image.Image:
    """An n x n view of the padded source at scale s (panel px per image px) centred on (cx, cy) image px."""
    if abs(s - 1.0) < 1e-9:
        x0, y0 = round(cx - n / 2) + PAD, round(cy - n / 2) + PAD
        return src.crop((x0, y0, x0 + n, y0 + n))
    u0, v0 = cx - n / 2 / s + PAD, cy - n / 2 / s + PAD
    u1, v1 = cx + n / 2 / s + PAD, cy + n / 2 / s + PAD
    return src.resize((n, n), Image.LANCZOS, box=(max(0.0, u0), max(0.0, v0), min(float(src.width), u1),
                                                  min(float(src.height), v1)))


def path_state(t: float, win: tuple) -> tuple:
    """(s, cx, cy, native) at video time t: whole, eased push-in to the window centre at s = 1, hold."""
    s0 = PANEL / 4096
    T = np.array([win[0] + PANEL / 2, win[1] + PANEL / 2])
    c0 = np.array([2048.0, 2048.0])
    p0 = PANEL / 2 + s0 * (T - c0)
    if t < T_WHOLE:
        return s0, 2048.0, 2048.0, False
    if t < T_WHOLE + T_IN:
        e = float(M.smootherstep((t - T_WHOLE) / T_IN))
        s = math.exp(math.log(s0) * (1 - e))
        p = p0 + (PANEL / 2 - p0) * e
        c = T - (p - PANEL / 2) / s
        return s, float(c[0]), float(c[1]), False
    return 1.0, float(T[0]), float(T[1]), True


_SRC: dict = {}


def _src(p):
    if p not in _SRC:
        _SRC[p] = Image.fromarray(np.load(p))
    return _SRC[p]


def core(job: dict, t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), M.rgb(M.PAL["screen"]))
    d = ImageDraw.Draw(img, "RGBA")
    s, cx, cy, native = path_state(t, job["win"])
    x0 = (W - 3 * PANEL - 2 * GAP) // 2
    y0 = 196
    mask = Image.new("L", (PANEL, PANEL), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, PANEL - 1, PANEL - 1], radius=8, fill=255)
    for i, (m, name, key, macro) in enumerate(METHODS):
        x = x0 + i * (PANEL + GAP)
        v = view(_src(job["srcs"][m]), s, cx, cy, PANEL)
        img.paste(v, (x, y0), mask)
        col = M.PAL[COLOR[m]]
        f1 = M.font(30, "sans", 600)
        d.text((x, y0 - 22), name, font=f1, fill=M.rgb(col), anchor="ls")
        d.text((x + PANEL, y0 - 22), f"{job['sec'][m]} s per image", font=M.font(24, "sans", 400),
               fill=M.rgb(M.PAL["ink2"]), anchor="rs", features=["tnum"])
    # footer: the percentile (selection) and the scale
    fy = y0 + PANEL + 44
    pc = job["pct"]
    d.text((x0, fy), f"{ORD[pc]} percentile prompt" + (" (median)" if pc == 50 else ""), font=M.font(26, "sans", 600),
           fill=M.rgb(M.PAL["ink"]), anchor="ls")
    d.text((x0, fy + 34), "of the paired ImageReward gain of progressive SCA over AReA, 195 prompts",
           font=M.font(22, "sans", 400), fill=M.rgb(M.PAL["ink3"]), anchor="ls")
    xr = x0 + 3 * PANEL + 2 * GAP
    zoom = s / (PANEL / 4096)
    lab = "1:1 native pixels" if native else ("whole image, 4096 × 4096" if zoom < 1.01 else f"{zoom:.1f}×")
    d.text((xr, fy), lab, font=M.font(26, "sans", 600), fill=M.rgb(M.PAL["sca"] if native else M.PAL["ink2"]),
           anchor="rs", features=["tnum"])
    return img


def draw(job: dict, k: int) -> Image.Image:
    t = k / FPS
    t_back = T_WHOLE + T_IN + T_NATIVE
    total = t_back + T_BACK
    if t < t_back:
        return core(job, t)
    a = float(M.smoothstep((t - t_back) / (total - t_back - 1 / FPS)))
    return Image.blend(core(job, t_back - 1 / FPS), core(job, 0.0), a)


def _work(task):
    job, idxs, outdir = task
    for k in idxs:
        draw(job, k).save(Path(outdir) / f"{k:05d}.png", compress_level=1)
    return len(idxs)


def build(pc: int, procs: int):
    rr, rule = rows()
    r = rr[pc]
    sec = seconds()
    srcs = {m: r["sources"][m] for m, *_ in METHODS}
    meta = vc.same_prompt_and_noise({m: srcs[m] for m in srcs})
    maps = [vc.detail_map(Image.open(srcs[m]).convert("RGB")) for m in srcs]
    win = vc.best_windows(maps, PANEL, PANEL, k=1)[0][:2]
    job = dict(pct=pc, srcs={m: str(padded(p)) for m, p in srcs.items()}, sec=sec, win=win)
    n = round((T_WHOLE + T_IN + T_NATIVE + T_BACK) * FPS)
    name = f"compare_p{pc}"
    fdir = BUILD / name / f"frames_n{n}"
    fdir.mkdir(parents=True, exist_ok=True)
    tasks = [(job, list(range(i, min(n, i + 6))), str(fdir)) for i in range(0, n, 6)]
    t0 = time.time()
    with Pool(procs) as pool:
        sum(pool.imap_unordered(_work, tasks))
    print(f"{name}: {n} frames in {time.time() - t0:.1f} s", flush=True)
    mp4 = M.OUT / f"{name}.mp4"
    enc = M.encode_to_size(fdir / "%05d.png", mp4, FPS, n, MAX_BYTES, crf_floor=20, preset="veryslow", gop=n,
                           passlog=BUILD / name / "x264pass", x264="aq-mode=3", threads=16)
    p = M.check_video(mp4, max_bytes=MAX_BYTES, w=W, h=H)
    poster = M.poster_webp(Image.open(fdir / "00000.png"), M.OUT / f"{name}.webp", quality=84)
    alt = (f"AReA, HRDiT and progressive SCA images of the same prompt side by side ({ORD[pc]} percentile prompt), "
           f"zooming together from the whole 4096 by 4096 images into the same detail at native pixels.")
    cap = (f"{ORD[pc]} percentile prompt of the paired ImageReward gain of progressive SCA over AReA (the rows of the "
           f"paper's Figure 5). Synchronized push-in to native pixels on the window of highest edge energy averaged "
           f"over the three images. Clean seconds per 4096 x 4096 image on one GH200.")
    e = M.entry_for(name, mp4, poster, loop=True, alt=alt, caption=cap,
                    sources=dict(images=srcs, prompt=meta["prompt"], prompt_id=meta["prompt_id"], seed=meta["seed"],
                                 ir_difference=r["ir_difference"], selection_rule=rule,
                                 provenance="paper/figures/v6/qual_main_provenance.json",
                                 window_px=[int(win[0]), int(win[1]), PANEL, PANEL],
                                 window_rule="v6_common.best_windows over the three images' detail maps",
                                 seconds={m: dict(display=sec[m]) for m in sec}),
                    script="v5_compare.py", order=40 + [25, 50, 75].index(pc),
                    extra=dict(role=f"compare_{[25, 50, 75].index(pc) + 1}", percentile=pc, encode=enc,
                               timing=dict(whole=T_WHOLE, push_in=T_IN, native=T_NATIVE, back=T_BACK)))
    M.manifest_put(e)
    sd = M.BUILD / "sheets"
    M.contact_sheet(mp4, sd / f"{name}_sheet.png", n=12, cols=4, tile_w=480)
    M.seam_sheet(mp4, sd / f"{name}_seam.png", tile_w=640)
    print(json.dumps(dict(name=name, **p, enc=enc)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, choices=[25, 50, 75], default=None)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    for pc in ([a.only] if a.only else [25, 50, 75]):
        build(pc, a.procs)


if __name__ == "__main__":
    main()
