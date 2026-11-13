"""Speed race loop of the v5 page: web/assets/video/v5/speedrace_16x9.mp4 and speedrace_9x16.mp4 (+ .webp posters).

What it shows. Four lanes on one time axis: progressive SCA, single-pass SCA, HRDiT and AReA, each making 4096^2
images back to back at its measured clean seconds per image (batch 1, one GH200). Every image is one segment of its
lane; the segment fills as time runs, and when it completes, that image appears above its end and the lane's counter
goes up. The race runs until AReA's first image is done (progressive SCA has finished four by then), holds, and
fades back to the empty start, so the loop is seamless. Time runs COMPRESSION times faster than real time, stated on
screen with a clock of real seconds.

Numbers (never typed). Seconds per image are numbers.json main.prog.time_clean, main.sca.time_clean,
main.hrdit.time_clean, main.area.time_clean ("value" drives the motion, "display" is printed), checked against the
paper's macros \\hlSecProg, \\hlSecSca, \\hlSecHrdit, \\hlSecArea (paper/figures/snippets/main_tables/headline.tex).

Images (rule). Every lane works through the same prompt queue: the paper's decile prompts of the paired ImageReward
gain of progressive SCA over AReA (v6_gallery deciles_a/deciles_b provenance; 195 evaluation prompts, seed 0) in order
of distance from the median, upper first: 50th (the median, also the median row of the paper's Figure 5), 60th, 40th,
70th. So the first image of every lane is the same median prompt; single-pass SCA shows only queue positions 1 and 2,
both prompts that name no person (the page's rule for single-pass images). Images are the 4096^2 PNGs of the paper's
evaluation pool (v6_common.pool195), Lanczos thumbnails.

Run from the repository root:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_speedrace.py [--cut 16x9|9x16|both]
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

FPS = 30
SS = 2                       # supersampling of every frame
COMPRESSION = 20             # real seconds per video second
T_LEAD, T_HOLD, T_FADE = 0.5, 2.6, 0.55
POP = 0.35                   # s of video for an image to appear
MAX_BYTES = 2_500_000
BUILD = M.BUILD / "speedrace"
METHODS = [  # key in numbers.json, label, macro, colour, style
    ("prog", "Progressive SCA", "hlSecProg", M.PAL["sca"], "solid"),
    ("sca", "Single-pass SCA", "hlSecSca", M.PAL["sca"], "hollow"),
    ("hrdit", "HRDiT", "hlSecHrdit", M.PAL["hrdit"], "solid"),
    ("area", "AReA", "hlSecArea", M.PAL["area"], "solid"),
]
POOL_KEY = {"prog": "sca_v2_prog", "sca": "sca_v2", "hrdit": "hrdit_dev_author", "area": "area"}
QUEUE_PCTS = [50, 60, 40, 70]


# ------------------------------------------------------------------------------------------------ data
def seconds() -> dict:
    mac = M.headline_macros()
    out = {}
    for key, label, macro, col, style in METHODS:
        v, disp = M.num(f"main.{key}.time_clean"), M.num(f"main.{key}.time_clean", "display")
        if disp != mac[macro]:
            raise ValueError(f"main.{key}.time_clean {disp} != paper macro {macro} {mac[macro]}")
        out[key] = dict(value=float(v), display=disp)
    return out


def queue() -> list:
    dec = {}
    for f in ("deciles_a", "deciles_b"):
        for r in json.loads((M.V6 / f"{f}_provenance.json").read_text())["rows"]:
            dec[r["percentile"]] = r["key"]
    pool = vc.pool195()
    texts = vc.prompts195(pool)
    out = []
    for i, p in enumerate(QUEUE_PCTS):
        k = dec[p]
        e = dict(pct=p, key=k, prompt=texts[k], person=vc.names_person(texts[k]),
                 paths={m: pool[POOL_KEY[m]][k]["path"] for m, *_ in METHODS})
        out.append(e)
    return out


def thumbs(q: list, sec: dict, size: int) -> dict:
    """{(method, queue index): thumbnail path} for every image that completes during the race."""
    t_end = sec["area"]["value"]
    need = {}
    for m, *_ in METHODS:
        n = int(math.floor(t_end / sec[m]["value"] + 1e-9))
        for i in range(n):
            if m == "sca" and q[i]["person"]:
                raise ValueError(f"single-pass SCA would show a prompt that names a person: {q[i]['key']}")
            need[(m, i)] = q[i]["paths"][m]
    out = {}
    d = BUILD / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    for (m, i), src in need.items():
        p = d / f"{m}_{i}_{size}.png"
        if not p.exists():
            im = Image.open(src).convert("RGB")
            im.reduce(4).resize((size, size), Image.LANCZOS).save(p)
        out[(m, i)] = str(p)
    return out


# ------------------------------------------------------------------------------------------------ layout
def layout(cut: str) -> dict:
    if cut == "16x9":
        W, H = 1600, 900
        L = dict(W=W, H=H, mx=96, top=64, kick=24, clock=46, clock_sub=24, lane_y0=196, lane_h=150, name=30,
                 secs=24, a0=440, a1=1330, bar_h=14, bar_dy=118, thumb=96, thumb_dy=10, cnt_x=1504, cnt=64,
                 cnt_sub=24, tick=24, name_dy=58, secs_dy=96, stacked=False)
    else:
        W, H = 900, 1600
        L = dict(W=W, H=H, mx=56, top=96, kick=28, clock=60, clock_sub=28, lane_y0=340, lane_h=282, name=36,
                 secs=28, a0=56, a1=770, bar_h=16, bar_dy=240, thumb=140, thumb_dy=88, cnt_x=844, cnt=60,
                 cnt_sub=28, tick=28, name_dy=36, secs_dy=36, stacked=True)
    return L


# ------------------------------------------------------------------------------------------------ state
def race_state(t_video: float, sec: dict) -> dict:
    """Real seconds on the clock and the fade-out factor at a video time."""
    t_race = sec["area"]["value"] / COMPRESSION
    if t_video < T_LEAD:
        return dict(t=0.0, fade=1.0, end=False)
    if t_video < T_LEAD + t_race:
        return dict(t=(t_video - T_LEAD) * COMPRESSION, fade=1.0, end=False)
    if t_video < T_LEAD + t_race + T_HOLD:
        return dict(t=sec["area"]["value"], fade=1.0, end=True)
    f = (t_video - T_LEAD - t_race - T_HOLD) / (T_FADE - 1.0 / FPS)   # empty again on the last frame
    return dict(t=sec["area"]["value"], fade=float(1 - M.smoothstep(f)), end=True)


def total_seconds(sec: dict) -> float:
    return T_LEAD + sec["area"]["value"] / COMPRESSION + T_HOLD + T_FADE


# ------------------------------------------------------------------------------------------------ drawing
def _alpha(c, a):
    c = M.rgb(c) if isinstance(c, str) else c
    return c[:3] + (int(round(255 * max(0.0, min(1.0, a)))),)


def draw_frame(t_video: float, sec: dict, L: dict, thumbs_: dict, q: list) -> Image.Image:
    s = SS
    W, H = L["W"] * s, L["H"] * s
    img = Image.new("RGB", (W, H), M.rgb(M.PAL["screen"]))   # RGB base: RGBA colours blend (an RGBA base would not)
    d = ImageDraw.Draw(img, "RGBA")
    st = race_state(t_video, sec)
    t, fade = st["t"], st["fade"]
    t_end = sec["area"]["value"]
    t_axis = t_end * 1.07                         # the axis runs a little past AReA's image; tracks fade out there
    mx = L["mx"] * s
    a0, a1 = L["a0"] * s, L["a1"] * s

    def X(tt):
        return a0 + (a1 - a0) * tt / t_axis

    ink, ink2, ink3, rule = M.PAL["ink"], M.PAL["ink2"], M.PAL["ink3"], M.PAL["rule"]
    # header: kicker (left), clock and compression (right)
    kick = "SECONDS PER 4096 × 4096 IMAGE · ONE GH200"
    fk = M.font(L["kick"] * s, "sans", 600)
    top = L["top"] * s
    if L["stacked"]:
        d.text((mx, top), "SECONDS PER 4096 × 4096 IMAGE", font=fk, fill=M.rgb(ink2), anchor="la",
               features=["tnum"])
        d.text((mx, top + 1.5 * L["kick"] * s), "ONE GH200, BATCH 1", font=fk, fill=M.rgb(ink2), anchor="la")
    else:
        d.text((mx, top), kick, font=fk, fill=M.rgb(ink2), anchor="la")
    clock = sec["area"]["display"] + " s" if st["end"] else f"{int(t)} s"
    clocks = [(clock, fade)] + ([("0 s", 1 - fade)] if fade < 1 else [])   # the clock cross-fades back to 0
    fc = M.font(L["clock"] * s, "serif", 600)
    fcs = M.font(L["clock_sub"] * s, "sans", 400)
    if L["stacked"]:
        cy = top + 3.4 * L["kick"] * s
        for txt, al in clocks:
            M.text_alpha(img, (mx, cy), txt, fc, ink, al, anchor="la", features=["tnum"])
        d.text((mx, cy + 1.25 * L["clock"] * s), f"real time, shown {COMPRESSION}× faster", font=fcs,
               fill=M.rgb(ink2), anchor="la")
    else:
        rx = L["cnt_x"] * s
        for txt, al in clocks:
            M.text_alpha(img, (rx, top - 8 * s), txt, fc, ink, al, anchor="ra", features=["tnum"])
        d.text((rx, top + 1.18 * L["clock"] * s), f"real time, shown {COMPRESSION}× faster", font=fcs,
               fill=M.rgb(ink2), anchor="ra")
    # lanes
    lane_y0, lane_h = L["lane_y0"] * s, L["lane_h"] * s
    bar_h = L["bar_h"] * s
    gap = 4 * s
    yb = lane_y0 + 4 * lane_h
    grid = [tv for tv in range(0, int(t_axis) + 1, 50)]
    for tv in grid:
        xx = X(tv)
        if L["stacked"]:
            for li in range(4):
                by_ = lane_y0 + li * lane_h + L["bar_dy"] * s
                d.line([(xx, by_ - 12 * s), (xx, by_ + bar_h + 12 * s)], fill=_alpha(rule, 1.0), width=max(1, s))
        else:
            d.line([(xx, lane_y0 - 6 * s), (xx, yb - 18 * s)], fill=_alpha(rule, 0.9), width=max(1, s))
    for li, (key, label, macro, col, style) in enumerate(METHODS):
        y = lane_y0 + li * lane_h
        per = sec[key]["value"]
        fn = M.font(L["name"] * s, "sans", 600)
        fs = M.font(L["secs"] * s, "sans", 400)
        if L["stacked"]:
            d.text((mx, y + L["name_dy"] * s), label, font=fn, fill=M.rgb(col), anchor="ls")
            d.text((mx, y + (L["name_dy"] + 38) * s), f"{sec[key]['display']} s per image", font=fs,
                   fill=M.rgb(ink2), anchor="ls", features=["tnum"])
        else:
            d.text((mx, y + L["name_dy"] * s), label, font=fn, fill=M.rgb(col), anchor="ls")
            d.text((mx, y + L["secs_dy"] * s), f"{sec[key]['display']} s per image", font=fs, fill=M.rgb(ink2),
                   anchor="ls", features=["tnum"])
        by = y + L["bar_dy"] * s
        k = 0
        done = 0
        while k * per < min(t_axis, t_end):
            sx0, sx1 = X(k * per) + gap / 2, X(min((k + 1) * per, t_axis)) - gap / 2
            cut_end = (k + 1) * per > t_axis
            r = bar_h / 2
            d.rounded_rectangle([sx0, by, sx1 + (r if cut_end else 0), by + bar_h], radius=r, fill=M.rgb(rule))
            fill_to = X(min(t, (k + 1) * per)) - (gap / 2 if t >= (k + 1) * per else 0)
            if t > k * per and fill_to > sx0:
                fa = fade
                if style == "hollow":
                    d.rounded_rectangle([sx0, by, fill_to, by + bar_h], radius=r, fill=_alpha(col, 0.22 * fa),
                                        outline=_alpha(col, fa), width=max(1, round(2 * s)))
                else:
                    d.rounded_rectangle([sx0, by, fill_to, by + bar_h], radius=r, fill=_alpha(col, fa))
            if t >= (k + 1) * per - 1e-9 and (k + 1) * per <= t_end + 1e-9:
                done += 1
            k += 1
        # images of the completed segments
        for i in range(done):
            tk = (i + 1) * per
            age = (t - tk) / COMPRESSION        # video seconds since completion
            if st["end"] and tk >= t_end - 1e-9:
                age = max(age, (t_video - T_LEAD - t_end / COMPRESSION))
            a = float(M.ease_out_cubic(age / POP)) if age < POP else 1.0
            a *= fade
            if a <= 0:
                continue
            ts = L["thumb"] * s
            sc = 0.9 + 0.1 * a
            tw = ts * sc
            tx1 = X(tk) - gap / 2
            ty0 = y + L["thumb_dy"] * s + (1 - a) * 10 * s + (ts - tw)
            th = Image.open(thumbs_[(key, i)]).convert("RGB").resize((round(tw), round(tw)), Image.LANCZOS)
            mask = Image.new("L", th.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, th.size[0] - 1, th.size[1] - 1], radius=6 * s,
                                                   fill=int(255 * a))
            img.paste(th, (round(tx1 - tw), round(ty0)), mask)
            d.rounded_rectangle([round(tx1 - tw), round(ty0), round(tx1 - tw) + th.size[0] - 1,
                                 round(ty0) + th.size[1] - 1], radius=6 * s, outline=_alpha(col, 0.9 * a),
                                width=max(1, round(1.5 * s)))
        # counter
        last_tk = done * per
        age = (t - last_tk) / COMPRESSION if done else 9.0
        pop = 1.0 + 0.14 * (1 - M.smoothstep(age / 0.35)) if done and age < 0.35 else 1.0
        fcn = M.font(L["cnt"] * s * pop, "serif", 600)
        fcs2 = M.font(L["cnt_sub"] * s, "sans", 400)
        # the counter cross-fades back to a grey 0 with everything else
        vals = [(done, fade, col if done else ink3)] + ([(0, 1 - fade, ink3)] if fade < 1 and done else [])
        cx = L["cnt_x"] * s
        cyy = y + L["name_dy"] * s if L["stacked"] else by + bar_h - 26 * s
        for v, al, cc in vals:
            M.text_alpha(img, (cx, cyy), f"{v}", fcn, cc, al, anchor="rs", features=["tnum"])
        if not L["stacked"]:
            for v, al, cc in vals:
                M.text_alpha(img, (cx, by + bar_h + 4 * s), "images" if v != 1 else "image", fcs2, ink2, al,
                             anchor="ra")
    # fade-out of the tracks at the axis end (bar rows only)
    fade_w = 64 * s
    grad = Image.linear_gradient("L").rotate(90).resize((round(fade_w), round(bar_h + 8 * s)), Image.BILINEAR)
    grad = Image.eval(grad, lambda v: 255 - v)
    over = Image.new("RGBA", grad.size, M.rgb(M.PAL["screen"]) + (255,))
    over.putalpha(grad)
    for li in range(4):
        by_ = lane_y0 + li * lane_h + L["bar_dy"] * s
        img.paste(over, (round(a1 - fade_w + 14 * s), round(by_ - 4 * s)), over)
        d.rectangle([a1 + 14 * s, by_ - 4 * s, a1 + 40 * s, by_ + bar_h + 4 * s], fill=M.rgb(M.PAL["screen"]))
    # axis labels and the playhead
    ft = M.font(L["tick"] * s, "sans", 400)
    for tv in grid:
        lab = f"{tv} s" if tv else "0"
        d.text((X(tv), yb + 2 * s), lab, font=ft, fill=M.rgb(ink3), anchor="ma", features=["tnum"])
    if 0 < t and fade > 0:
        xx = X(t)
        if L["stacked"]:
            for li in range(4):
                by_ = lane_y0 + li * lane_h + L["bar_dy"] * s
                d.line([(xx, by_ - 14 * s), (xx, by_ + bar_h + 14 * s)], fill=_alpha(ink, 0.8 * fade),
                       width=max(1, round(2 * s)))
        else:
            d.line([(xx, lane_y0 - 14 * s), (xx, yb - 10 * s)], fill=_alpha(ink, 0.55 * fade),
                   width=max(1, round(1.5 * s)))
    return img.convert("RGB").resize((L["W"], L["H"]), Image.LANCZOS)


# ------------------------------------------------------------------------------------------------ render
def _work(task):
    idxs, cut, sec, thumbs_, q, outdir = task
    L = layout(cut)
    for k in idxs:
        draw_frame(k / FPS, sec, L, thumbs_, q).save(Path(outdir) / f"{k:05d}.png", compress_level=1)
    return len(idxs)


def build(cut: str, procs: int) -> dict:
    sec = seconds()
    q = queue()
    L = layout(cut)
    th = thumbs(q, sec, L["thumb"] * SS)
    n = round(total_seconds(sec) * FPS)
    fdir = BUILD / cut / f"frames_c{COMPRESSION}_n{n}"
    fdir.mkdir(parents=True, exist_ok=True)
    tasks = [(list(range(i, min(n, i + 8))), cut, sec, th, q, str(fdir)) for i in range(0, n, 8)]
    t0 = time.time()
    with Pool(procs) as pool:
        sum(pool.imap_unordered(_work, tasks))
    print(f"{cut}: {n} frames in {time.time() - t0:.1f} s", flush=True)
    mp4 = M.OUT / f"speedrace_{cut}.mp4"
    enc = M.encode_to_size(fdir / "%05d.png", mp4, FPS, n, MAX_BYTES, crf_floor=18, preset="veryslow",
                           gop=n, passlog=BUILD / cut / "x264pass", x264="aq-mode=3", threads=16)
    p = M.check_video(mp4, max_bytes=MAX_BYTES, w=L["W"], h=L["H"])
    t_rest = T_LEAD + sec["area"]["value"] / COMPRESSION + T_HOLD * 0.5
    poster = M.poster_webp(Image.open(fdir / f"{round(t_rest * FPS):05d}.png"), M.OUT / f"speedrace_{cut}.webp",
                           quality=84)
    counts = {m: int(math.floor(sec["area"]["value"] / sec[m]["value"] + 1e-9)) for m, *_ in METHODS}
    alt = (f"Race of four methods making 4096 by 4096 images, shown {COMPRESSION} times faster than real time: "
           f"in the {sec['area']['display']} seconds AReA takes for one image, progressive SCA finishes "
           f"{counts['prog']}, single-pass SCA {counts['sca']} and HRDiT {counts['hrdit']}.")
    cap = (f"Clean seconds per 4096 x 4096 image on one GH200 (batch 1), time shown {COMPRESSION}x faster. Every lane "
           f"works through the same prompts: the paper's decile prompts of the ImageReward gain over AReA from the "
           f"median outward (50th, 60th, 40th, 70th percentile).")
    e = M.entry_for(f"speedrace_{cut}", mp4, poster, loop=True, alt=alt, caption=cap,
                    sources=dict(seconds={m: dict(numbers_key=f"main.{m}.time_clean", **sec[m]) for m in sec},
                                 paper_macros="paper/figures/snippets/main_tables/headline.tex",
                                 queue=[dict(percentile=e["pct"], key=e["key"], prompt=e["prompt"],
                                             images={m: e["paths"][m] for m in e["paths"]}) for e in q],
                                 decile_rule=json.loads((M.V6 / "deciles_a_provenance.json").read_text())
                                 ["selection"]["rule"]),
                    script="v5_speedrace.py", order=20 + (0 if cut == "16x9" else 1),
                    extra=dict(role="speedrace" if cut == "16x9" else "speedrace9x16", compression=COMPRESSION,
                               counts=counts, timing=dict(lead=T_LEAD, race=round(sec["area"]["value"] / COMPRESSION, 3),
                                                          hold=T_HOLD, fade=T_FADE),
                               poster_time=round(t_rest, 3), encode=enc))
    M.manifest_put(e)
    sd = M.BUILD / "sheets"
    M.contact_sheet(mp4, sd / f"speedrace_{cut}_sheet.png", n=12, cols=4, tile_w=480 if cut == "16x9" else 270)
    M.seam_sheet(mp4, sd / f"speedrace_{cut}_seam.png", tile_w=640 if cut == "16x9" else 360)
    print(json.dumps(dict(cut=cut, **p, enc=enc)), flush=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="both", choices=["16x9", "9x16", "both"])
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--frame", type=float, default=None, help="render one frame at this video time to BUILD/look")
    a = ap.parse_args()
    cuts = ["16x9", "9x16"] if a.cut == "both" else [a.cut]
    if a.frame is not None:
        sec, q = seconds(), queue()
        for cut in cuts:
            L = layout(cut)
            th = thumbs(q, sec, L["thumb"] * SS)
            out = BUILD / "look" / f"{cut}_{a.frame:.2f}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            draw_frame(a.frame, sec, L, th, q).save(out)
            print(out)
        return
    for cut in cuts:
        build(cut, a.procs)


if __name__ == "__main__":
    main()
