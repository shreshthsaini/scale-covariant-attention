"""Explainer of the v5 page (paper v6.2): web/assets/video/v5/explainer.mp4, explainer.webp, explainer.vtt.

About 86 s, 1600 x 900, 30 fps, captions burned in and in a WebVTT file, no sound. Scenes:
  title        "Reach Is a Property of the Head", subtitle, "Anonymous authors, under review".
  problem      the run's native 1024^2 stage (64 x 64 tokens) beside full attention run directly at 4096^2 on the same
               prompt and seed (256 x 256 tokens, offsets up to 255 where training saw 63).
  observation  the near share of all 1,368 heads against the noise level (32 prompts, after the gate; the data of the
               paper's Figure 2b, v6_observation.load) and the variance split (head / noise level / shared schedule).
  views        method_window frames (v5_method.py): two native-size views per head, mixed by the head's reach.
  measure      band 1, "Measure once at 1024^2": method_emulation frames.
  generate     band 2, "Generate at any size": method_stages frames.
  speed        speedrace_16x9 frames (v5_speedrace.py).
  claims       the headline claims of paper v6.2 (Holm-corrected comparisons): better than AReA on all ten per-image
               metrics at each of three seeds; statistically indistinguishable from HRDiT on ImageReward, HPSv2.1 and
               PickScore (intervals that include zero, asserted) in 46% of its time.
  showcase     hero_16x9 frames of the first image (v5_hero.py): whole image to native pixels.
  end          title card.
Every number is read from the paper's generated macros (paper/figures/snippets/main_tables/headline.tex) or the data
files, never typed; the reused scenes are the frames of the v5 loops (run v5_hero.py, v5_speedrace.py and
v5_method.py first; their frame folders live under the scratch build folder).

Run from the repository root:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_explainer.py [--look t,t,...]
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

W, H = 1600, 900
FPS = 30
XF = 0.35                     # dissolve into each scene (s)
MAX_BYTES = 10_000_000
BUILD = M.BUILD / "explainer"
STAGE = (112, 8, 1376, 774)   # where a reused 1600 x 900 loop frame is placed (scaled 0.86)
CAP_Y = 852
IMG1024 = M.REPO / "paper" / "figures" / "attn_focus" / "image_1024.png"
FULL4096 = Path("/scratch/09032/saini_2/sca/out/main/full/4096x4096/"
                "a4k_eval_4096_minus8343371271909676658_0325f0a692_seed0.png")
FRAMES = {   # name: (frame folder, frames used = the loop before its loop-closing tail, how to regenerate)
    "views": (M.BUILD / "method" / "method_window" / "frames_n240", 240, "v5_method.py window"),
    "measure": (M.BUILD / "method" / "method_emulation" / "frames_n294", 276, "v5_method.py emulation"),
    "generate": (M.BUILD / "method" / "method_stages" / "frames_n303", 285, "v5_method.py stages"),
    "speed": (M.BUILD / "speedrace" / "16x9" / "frames_c20_n413", 396, "v5_speedrace.py --cut 16x9"),
    "showcase": (M.BUILD / "hero" / "16x9" / "frames_k3_f30_0.8_2.3_1.4_1.6_0.6", 183, "v5_hero.py --cut 16x9"),
}


# ------------------------------------------------------------------------------------------------ numbers
def facts() -> dict:
    m = M.headline_macros()
    for k in ("hlProgHrditIr", "hlProgHrditHps", "hlProgHrditPick"):
        lo, hi = float(m[k + "Lo"]), float(m[k + "Hi"])
        assert lo < 0 < hi, f"{k}: the interval [{lo}, {hi}] excludes zero; the wording would be wrong"
    wins = [m["hlProgWinsArea"], m["hlProgSeedOneWinsArea"], m["hlProgSeedTwoWinsArea"]]
    assert len(set(wins)) == 1, wins
    return dict(sec_prog=m["hlSecProg"], sec_area=m["hlSecArea"], sec_hrdit=m["hlSecHrdit"],
                x_area=m["hlSpeedProgArea"], x_hrdit=m["hlSpeedProgHrdit"], var_head=m["hlVarHead"],
                var_step=m["hlVarStep"], var_avg=m["hlVarAvg"], wins=wins[0], wins_word=m["hlProgWinsAreaWord"],
                pct_hrdit=m["hlCampProgHrditPct"],
                n_area=int(math.floor(float(m["hlSecArea"]) / float(m["hlSecProg"]))))


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def scenes(f: dict) -> list:
    """(name, seconds, [(start, end, caption)]) in order."""
    return [
        ("title", 5.0, [(0.6, 4.8, "FLUX.1-dev at 4096 × 4096 without training: only the attention changes.")]),
        ("problem", 8.0, [(0.0, 3.9, "FLUX.1-dev is trained near 1024 × 1024, a grid of 64 × 64 tokens."),
                          (3.9, 8.0, "Run directly at 4096 × 4096, full attention sees 16 times the tokens and "
                                     "offsets up to 255, where training saw 63.")]),
        ("observation", 14.0, [
            (0.0, 4.6, "Split one query's keys into near and far: the near share is that head's reach."),
            (4.6, 9.4, f"Over 1,368 heads and every noise level, which head it is explains {f['var_head']}% of "
                       f"the variance."),
            (9.4, 14.0, f"The noise level explains {f['var_step']}%; a schedule shared by all heads, "
                        f"{f['var_avg']}%.")]),
        ("views", 8.0, [(0.0, 4.0, "SCA gives each head two native-size views: a window around the query and a grid "
                                   "over the whole image."),
                        (4.0, 8.0, "Each head mixes the two by its own reach π.")]),
        ("measure", 9.2, [(0.0, 4.6, "Measure once at 1024 × 1024, where full attention is still the reference: "
                                     "both views shrink by r."),
                          (4.6, 9.2, "The best weight has a closed form, π* = clip(B/A, 0, 1), and fills the "
                                     "reach table.")]),
        ("generate", 9.5, [(0.0, 4.75, "Generate at any size: progressive SCA keeps HRDiT's three sampler stages."),
                            (4.75, 9.5, "The 2048² and 4096² stages read the same reach table, at r = 2 and "
                                        "r = 4.")]),
        ("speed", 13.2, [(0.0, 9.0, f"{f['sec_prog']} s per 4096² image on one GH200: {f['x_area']}× "
                                     f"faster than AReA, {f['x_hrdit']}× faster than HRDiT."),
                          (9.0, 13.2, f"In AReA's {f['sec_area']} s for one image, progressive SCA finishes "
                                       f"{WORDS[f['n_area']]}.")]),
        ("claims", 7.5, [(0.0, 3.7, f"Better than AReA on all {f['wins_word']} per-image metrics, at each of three "
                                    f"seeds."),
                         (3.7, 7.5, f"Statistically indistinguishable from HRDiT on three preference scores, in "
                                    f"{f['pct_hrdit']}% of its time.")]),
        ("showcase", 6.1, [(0.0, 6.1, "A 4096 × 4096 progressive SCA image, from the whole frame to native "
                                      "pixels.")]),
        ("end", 4.0, []),
    ]


# ------------------------------------------------------------------------------------------------ helpers
def text(d, xy, s, size, weight=600, fill="ink", anchor="la", family="sans"):
    d.text(xy, s, font=M.font(size, family, weight), fill=M.rgb(M.PAL[fill]) if fill in M.PAL else fill,
           anchor=anchor, features=["tnum"])


def reticle(d, box, col, arm=22, w=2):
    x0, y0, x1, y1 = box
    c = M.rgb(col) if isinstance(col, str) else tuple(col)
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(x, y), (x + dx * arm, y)], fill=c, width=w)
        d.line([(x, y), (x, y + dy * arm)], fill=c, width=w)


def blank() -> Image.Image:
    return Image.new("RGB", (W, H), M.rgb(M.PAL["screen"]))


# ------------------------------------------------------------------------------------------------ native scenes
def s_title(ctx, u, dur):
    img = blank()
    d = ImageDraw.Draw(img)
    a = float(M.smoothstep(u / 0.8))
    M.text_alpha(img, (800, 360), "Reach Is a Property of the Head", M.font(70, "serif", 600), M.PAL["ink"], a,
                 anchor="ms")
    M.text_alpha(img, (800, 420), "Scale-Covariant Attention for High-Resolution Diffusion Transformers",
                 M.font(30, "sans", 400), M.PAL["ink2"], float(M.smoothstep((u - 0.3) / 0.8)), anchor="ms")
    M.text_alpha(img, (800, 482), "Anonymous authors, under review", M.font(24, "sans", 400), M.PAL["ink3"],
                 float(M.smoothstep((u - 0.6) / 0.8)), anchor="ms")
    if a > 0.5:
        reticle(d, (300, 270, 1300, 520), M.mix(M.PAL["screen"], M.PAL["sca"], float(M.smoothstep((u - 0.4) / 0.8))))
    return img


def s_problem(ctx, u, dur):
    img = blank()
    d = ImageDraw.Draw(img, "RGBA")
    sm, big = 300, 600
    yb = 730                                   # common bottom edge, above the caption zone
    xl = 230
    xr = xl + sm + 170
    a1 = float(M.smoothstep(u / 0.6))
    a2 = float(M.smoothstep((u - 3.9) / 0.9))
    # left: the native 1024^2 stage of the run (drawn at a quarter of the 4096^2 side, its true relative size)
    im1 = ctx["im1024"].resize((sm, sm), Image.LANCZOS)
    if a1 > 0:
        img.paste(Image.blend(Image.new("RGB", im1.size, M.rgb(M.PAL["screen"])), im1, a1), (xl, yb - sm))
    text(d, (xl, yb - sm - 40), "1024 × 1024", 30, 600, "ink", anchor="ls")
    text(d, (xl, yb - sm - 12), "trained size · 64 × 64 tokens", 23, 400, "ink2", anchor="ls")
    # right: full attention run directly at 4096^2, the same prompt and seed
    d.rectangle([xr, yb - big, xr + big, yb], outline=M.rgb(M.PAL["rule"]), width=2)
    if a2 > 0:
        im2 = ctx["full4096"]
        img.paste(Image.blend(Image.new("RGB", im2.size, M.rgb(M.PAL["screen"])), im2, a2), (xr, yb - big))
    text(d, (xr, yb - big - 40), "4096 × 4096, full attention", 30, 600, "ink" if a2 > 0.5 else "ink3",
         anchor="ls")
    text(d, (xr, yb - big - 12), "256 × 256 tokens · offsets up to 255", 23, 400, "ink2", anchor="ls")
    text(d, (xl, yb + 30), "same prompt, same seed", 22, 400, "ink3", anchor="ls")
    return img


def s_observation(ctx, u, dur):
    img = blank()
    S = 2
    big = Image.new("RGB", (W * S, H * S), M.rgb(M.PAL["screen"]))
    d = ImageDraw.Draw(big, "RGBA")
    x0, y0, x1, y1 = 150 * S, 120 * S, 900 * S, 700 * S
    sig = ctx["sig"]
    flat = ctx["flat"]
    nb = len(sig)
    reveal = float(M.smoothstep(u / 2.6))

    def X(j):
        return x0 + (x1 - x0) * j / (nb - 1)

    def Y(v):
        return y1 - (y1 - y0) * v
    # axes
    d.line([(x0, y1), (x1, y1)], fill=M.rgb(M.PAL["rule"]), width=2 * S)
    for v in (0, 0.5, 1):
        d.line([(x0, Y(v)), (x1, Y(v))], fill=M.rgb(M.PAL["rule"]) + (110,), width=S)
        d.text((x0 - 14 * S, Y(v)), f"{v:g}", font=M.font(22 * S), fill=M.rgb(M.PAL["ink3"]), anchor="rm")
    for j in (0, nb // 2, nb - 1):
        d.text((X(j), y1 + 14 * S), f"{sig[j]:.2f}", font=M.font(22 * S), fill=M.rgb(M.PAL["ink3"]), anchor="ma")
    d.text((x1, y1 + 48 * S), "noise level σ (falling as denoising runs)", font=M.font(22 * S),
           fill=M.rgb(M.PAL["ink2"]), anchor="ra")
    d.text((x0, y0 - 64 * S), "Near share of 1,368 heads", font=M.font(30 * S, "sans", 600), fill=M.rgb(M.PAL["ink"]),
           anchor="la")
    d.text((x0, y0 - 28 * S), "share of attention within 16 tokens of the query, 32 prompts, 1024²",
           font=M.font(22 * S), fill=M.rgb(M.PAL["ink2"]), anchor="la")
    # curves: each head one line, coloured by its mean share (far blue -> near orange)
    xr = x0 + (x1 - x0) * reveal
    means = flat.mean(axis=1)
    far, near = np.array(M.rgb(M.PAL["far"])), np.array(M.rgb(M.PAL["near"]))
    hi_a = float(M.smoothstep((u - 3.0) / 0.8))
    for i in ctx["order"]:
        pts = [(X(j), Y(flat[i, j])) for j in range(nb) if X(j) <= xr + 1]
        if len(pts) < 2:
            continue
        c = tuple(int(v) for v in far + (near - far) * float(means[i]))
        d.line(pts, fill=c + (int(46 * (1 - 0.5 * hi_a)) + 10,), width=S)
    for h, key in ((ctx["near_row"], "near"), (ctx["far_row"], "far")):
        if hi_a > 0:
            pts = [(X(j), Y(flat[h, j])) for j in range(nb)]
            d.line(pts, fill=M.rgb(M.PAL[key]) + (int(255 * hi_a),), width=4 * S, joint="curve")
    if hi_a > 0:
        cv = flat.mean(axis=0)
        d.line([(X(j), Y(cv[j])) for j in range(nb)], fill=M.rgb(M.PAL["ink"]) + (int(230 * hi_a),), width=3 * S,
               joint="curve")
        d.text((X(nb - 1) + 10 * S, Y(flat[ctx["near_row"], -1])), "head 4", font=M.font(22 * S, "sans", 600),
               fill=M.rgb(M.PAL["near"]) + (int(255 * hi_a),), anchor="lm")
        d.text((X(nb - 1) + 10 * S, Y(flat[ctx["far_row"], -1])), "head 7", font=M.font(22 * S, "sans", 600),
               fill=M.rgb(M.PAL["far"]) + (int(255 * hi_a),), anchor="lm")
        d.text((X(nb - 1) + 10 * S, Y(cv[-1])), "average", font=M.font(22 * S, "sans", 600),
               fill=M.rgb(M.PAL["ink"]) + (int(255 * hi_a),), anchor="lm")
    img = big.resize((W, H), Image.LANCZOS)
    # variance split (right)
    f = ctx["facts"]
    rows = [(4.6, f"{f['var_head']}%", "which head it is", "sca"), (9.4, f"{f['var_step']}%", "the noise level", "ink"),
            (10.6, f"{f['var_avg']}%", "a schedule shared by all heads", "ink2")]
    bx = 1010
    for k, (t0, big_s, lab, col) in enumerate(rows):
        a = float(M.smoothstep((u - t0) / 0.7))
        y = 250 + k * 160
        M.text_alpha(img, (bx, y), big_s, M.font(72 if k == 0 else 56, "serif", 600), M.PAL[col], a, anchor="ls")
        M.text_alpha(img, (bx, y + 40), lab, M.font(26, "sans", 400), M.PAL["ink2"], a, anchor="ls")
    d2 = ImageDraw.Draw(img)
    text(d2, (bx, 140), "Variance of the near share", 26, 600, "ink")
    return img


def s_claims(ctx, u, dur):
    img = blank()
    f = ctx["facts"]
    rows = [(0.0, f"{f['wins']} of {f['wins']}", "per-image metrics better than AReA,", "at each of three seeds"),
            (3.7, f"{f['pct_hrdit']}%", "of HRDiT's time, statistically indistinguishable",
             "on ImageReward, HPSv2.1 and PickScore")]
    for k, (t0, big_s, l1, l2) in enumerate(rows):
        a = float(M.smoothstep((u - t0) / 0.7))
        y = 330 + k * 230
        M.text_alpha(img, (560, y), big_s, M.font(96, "serif", 600), M.PAL["sca"], a, anchor="rs")
        M.text_alpha(img, (610, y - 46), l1, M.font(32, "sans", 600), M.PAL["ink"], a, anchor="ls")
        M.text_alpha(img, (610, y - 4), l2, M.font(28, "sans", 400), M.PAL["ink2"], a, anchor="ls")
    d = ImageDraw.Draw(img)
    text(d, (160, 130), "PROGRESSIVE SCA AT 4096 × 4096 · PAIRED TESTS, HOLM-CORRECTED", 24, 600, "ink2")
    return img


def s_end(ctx, u, dur):
    return s_title(ctx, 3.0 + u, dur)


# ------------------------------------------------------------------------------------------------ reused scenes
_FR: dict = {}


def reused(name: str, u: float) -> Image.Image:
    fdir, n, how = FRAMES[name]
    k = min(n - 1, max(0, round(u * FPS)))
    p = fdir / f"{k:05d}.png"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run web/build/{how} first")
    im = Image.open(p).convert("RGB")
    img = blank()
    x, y, w, h = STAGE
    img.paste(im.resize((w, h), Image.LANCZOS), (x, y))
    return img


def band(img: Image.Image, label: str):
    M.chip(img, (STAGE[0] + STAGE[2] - 4, 36), label, 24, anchor="rt", bg=(15, 42, 31, 235),
           fg=M.rgb(M.PAL["sca_ink"]) + (255,), weight=600, radius=6)


NATIVE = {"title": s_title, "problem": s_problem, "observation": s_observation, "claims": s_claims, "end": s_end}


def scene_frame(ctx, name: str, u: float, dur: float) -> Image.Image:
    if name in NATIVE:
        return NATIVE[name](ctx, u, dur)
    img = reused(name, u)
    if name == "measure":
        band(img, "1  Measure once at 1024 × 1024")
    elif name == "generate":
        band(img, "2  Generate at any size")
    return img


# ------------------------------------------------------------------------------------------------ timeline
def timeline(f: dict):
    sc = scenes(f)
    for name, dur, _ in sc:
        if name in FRAMES:
            assert round(dur * FPS) <= FRAMES[name][1], (name, dur, FRAMES[name][1])
    starts, t = [], 0.0
    for name, dur, cues in sc:
        starts.append(t)
        t += dur
    cues = [(s + a, s + b, txt) for (name, dur, cs), s in zip(sc, starts) for a, b, txt in cs]
    return sc, starts, t, cues


def context() -> dict:
    import v6_observation as vo
    f = facts()
    obs = vo.load()
    flat = obs["flat"]
    nh = 24
    near_row, far_row = 11 * nh + 3, 11 * nh + 6          # block 12, heads 4 and 7 (vo.HEADS)
    assert tuple(vo.HEADS) == (3, 6)
    im1024 = Image.open(IMG1024).convert("RGB")
    full = Image.open(FULL4096).convert("RGB").reduce(4).resize((620, 620), Image.LANCZOS)
    return dict(facts=f, sig=obs["sig_cols"], flat=flat, order=obs["order"][::-1], near_row=near_row, far_row=far_row,
                im1024=im1024, full4096=full)


def frame(ctx, k: int) -> Image.Image:
    sc, starts, total, cues = timeline(ctx["facts"])
    t = k / FPS
    i = max(j for j, s in enumerate(starts) if s <= t + 1e-9)
    name, dur, _ = sc[i]
    u = t - starts[i]
    img = scene_frame(ctx, name, u, dur)
    if i > 0 and u < XF:
        pname, pdur, _ = sc[i - 1]
        prev = scene_frame(ctx, pname, pdur - 1 / FPS, pdur)
        img = Image.blend(prev, img, float(M.smoothstep(u / XF)))
    # caption (burned in), with a short fade at its edges
    for a, b, txt in cues:
        if a <= t < b:
            al = min(1.0, (t - a) / 0.25, (b - t) / 0.25)
            caption(img, txt, max(0.0, al))
    return img


def caption(img: Image.Image, txt: str, al: float, max_w: int = 1380):
    """One centred line, or two balanced lines when it would be wider than max_w."""
    f = M.font(30, "sans", 500)
    if f.getlength(txt) <= max_w:
        M.text_alpha(img, (W // 2, CAP_Y), txt, f, M.PAL["ink"], al, anchor="ms")
        return
    words = txt.split(" ")
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        cost = max(f.getlength(a), f.getlength(b))
        if best is None or cost < best[0]:
            best = (cost, a, b)
    M.text_alpha(img, (W // 2, CAP_Y - 19), best[1], f, M.PAL["ink"], al, anchor="ms")
    M.text_alpha(img, (W // 2, CAP_Y + 19), best[2], f, M.PAL["ink"], al, anchor="ms")


def _work(task):
    idxs, outdir = task
    global _C
    try:
        _C
    except NameError:
        _C = context()
    for k in idxs:
        frame(_C, k).save(Path(outdir) / f"{k:05d}.png", compress_level=1)
    return len(idxs)


def stamp(t: float) -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", type=str, default=None)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    f = facts()
    sc, starts, total, cues = timeline(f)
    for _, _, c in cues:
        M.check_words(c, where="explainer caption")
    if a.look:
        ctx = context()
        outs = []
        for tv in a.look.split(","):
            p = BUILD / "look" / f"t{float(tv):06.2f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            frame(ctx, round(float(tv) * FPS)).save(p)
            outs.append(p)
        M.frames_sheet(outs, BUILD / "look" / "look.png", cols=2, tile_w=800, label="explainer look")
        print(BUILD / "look" / "look.png")
        return
    n = round(total * FPS)
    fdir = BUILD / f"frames_n{n}"
    fdir.mkdir(parents=True, exist_ok=True)
    chunk = max(8, math.ceil(n / (a.procs * 3)))
    tasks = [(list(range(i, min(n, i + chunk))), str(fdir)) for i in range(0, n, chunk)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        sum(pool.imap_unordered(_work, tasks))
    print(f"explainer: {n} frames in {time.time() - t0:.1f} s", flush=True)
    mp4 = M.OUT / "explainer.mp4"
    enc = M.encode_to_size(fdir / "%05d.png", mp4, FPS, n, MAX_BYTES, crf_floor=21, preset="veryslow", gop=150,
                           passlog=BUILD / "x264pass", x264="aq-mode=3", threads=24)
    p = M.check_video(mp4, max_bytes=MAX_BYTES, w=W, h=H)
    poster = M.poster_webp(Image.open(fdir / f"{round(3.5 * FPS):05d}.png"), M.OUT / "explainer.webp", quality=84)
    vtt = M.OUT / "explainer.vtt"
    lines = ["WEBVTT", ""]
    for j, (a0, b0, txt) in enumerate(cues, 1):
        lines += [str(j), f"{stamp(a0)} --> {stamp(b0)}", txt, ""]
    vtt.write_text("\n".join(lines))
    alt = ("An 86-second explainer without sound, with captions: the problem at 4096 by 4096, the observation that "
           "reach belongs to the head, the two native-size views, reach emulation at 1024 by 1024, the three sampler "
           "stages, the speed race and the headline results.")
    cap = ("The method in 86 seconds: reach is a property of the head; measure it once at 1024 x 1024; generate at "
           "any size. Captions burned in and in explainer.vtt.")
    e = M.entry_for("explainer", mp4, poster, loop=False, alt=alt, caption=cap,
                    sources=dict(macros="paper/figures/snippets/main_tables/headline.tex",
                                 scenes=[dict(name=nm, start=round(s, 3), seconds=dur) for (nm, dur, _), s in
                                         zip(sc, starts)],
                                 reused_frames={k: str(v[0]) for k, v in FRAMES.items()},
                                 problem_images=dict(native_1024=M.rel_repo(IMG1024), full_4096=str(FULL4096)),
                                 observation="paper/figures/src/v6_observation.py load() (k1_1024.npz, "
                                             "variance_split.csv)"),
                    script="v5_explainer.py", order=90,
                    extra=dict(role="explainer", vtt="explainer.vtt", controls=True, encode=enc,
                               cues=[dict(start=round(a0, 3), end=round(b0, 3), text=txt) for a0, b0, txt in cues]))
    M.manifest_put(e)
    sd = M.BUILD / "sheets"
    M.contact_sheet(mp4, sd / "explainer_sheet.png", n=20, cols=4, tile_w=480)
    print(json.dumps(dict(name="explainer", **p, enc=enc)), flush=True)


if __name__ == "__main__":
    main()
