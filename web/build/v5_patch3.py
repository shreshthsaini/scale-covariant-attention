""""One patch, three sizes": web/assets/video/v5/patch3_<id>.mp4 (+ _9x16 later), role "patch3".

A motion graphic that teaches why SCA's two views change share of the image but never size. The model was trained on
64 x 64 tokens. Through progressive SCA's three stages (1K = 1024^2, 2K = 2048^2, 4K = 4096^2) one patch always reads
64 x 64 tokens around it (local, every token: the native-size window) and 64 x 64 tokens spread over the whole image
(global, coarse: the native-size grid of block centres). The canvases are drawn TO SCALE with one token = 2.5 px at
every stage, so the canvas grows (160, 320, 640 px) while the window stays 160 px: the whole canvas at 1K, half the
side at 2K, a quarter at 4K; the grid takes every token at 1K, one per 2 x 2 block at 2K, one per 4 x 4 block at 4K.
Geometry is the sampler's (window: 64 tokens around the query's 8 x 8 tile, clamped; grid: block centres at stride
n/64, offset stride/2). Frames are the recorded run (x0 previews per step), the patch zoom is the run's own native
window decoded at every step (4x / 2x upsampled at 1K / 2K, native at 4K), the head bars are the run's recorded
weights of block 12's heads 4 and 7 (median over each stage's steps). The reading strokes show WHICH tokens the patch
reads, drawn uniform; they are not attention strengths, and no attention map is drawn (none exists at 2K or 4K).
Patch: the formation component's first preset (v5_build_formation.presets: the trace's own window); scene 3's two
extra patches are its other presets.

Scenes (s): 0 final image with the patch (2) | 1 1K (6) | 2 growth to 2K (8) | 3 growth to 4K, then the focus visits
the two other patches (8) | 4 recap, three canvases to scale (3) | cross-fade to the start (0.6).

Run from the repository root:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_patch3.py [--only <id>] [--look t,t,...]
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
import v5_process as VP  # noqa: E402

sys.path.insert(0, str(M.FIGSRC))

W, H, S, FPS = 1600, 900, 2, 30
TP = 5                                   # 2x px per token at every stage (2.5 px on screen)
NT = {0: 64, 1: 128, 2: 256}
PZ = 280                                 # zoom panel side
# layouts (1x px): canvas anchor (bottom-left corner of every canvas = the 4K canvas's), the 4K extent, zoom panel,
# bars, legend, title, stage indicator, caption (centre and wrap width), recap (scale, x of each canvas, bottom, labels)
LAYOUTS = {
    "16x9": dict(W=1600, H=900, AX=120, AY=750, K4=(120, 110, 760, 750), PX0=860, PY0=330, BX0=1176, LEG_X=860,
                 LEG_Y=160, TITLE=(96, 70), IND=(1504, 70), CAP=(800, 852), CAP_W=1450,
                 RECAP=dict(scale=1.0, xs=(200, 400, 760), bottom=730, lab_y=768,
                            labs=("64 × 64 tokens", "128 × 128 tokens", "256 × 256 tokens"))),
    "9x16": dict(W=900, H=1600, AX=130, AY=860, K4=(130, 220, 770, 860), PX0=54, PY0=1110, BX0=380, LEG_X=54,
                 LEG_Y=950, TITLE=(54, 100), IND=(846, 100), CAP=(450, 1525), CAP_W=800,
                 RECAP=dict(scale=0.66, xs=(54, 185, 421), bottom=1040, lab_y=1078,
                            labs=("64 × 64", "128 × 128", "256 × 256 tokens"))),
}


def use(cut: str):
    """Set the module's layout constants (each worker renders one cut)."""
    g = globals()
    L = LAYOUTS[cut]
    for k in ("W", "H", "AX", "AY", "K4", "PX0", "PY0", "BX0", "LEG_X", "LEG_Y", "TITLE", "IND", "CAP", "CAP_W",
              "RECAP"):
        g[k] = L[k]
    g["CUT"] = cut


use("16x9")
T0, T1, T2, T3, T4, TB = 2.0, 6.0, 8.0, 8.0, 3.0, 0.6
BLOCK, H4, H7 = 11, 3, 6
MAX_BYTES = 6_000_000
BUILD = M.BUILD / "patch3"
CAPS = ["Follow one patch as the image is made.",
        "At 1K, the size the model was trained on, the patch reads every token.",
        "At 2K the image has grown; the window has not.",
        "At 4K the same window gives the detail; the same grid holds the layout.",
        "Every size reads what the model was trained to read."]
C = {k: np.array(M.rgb(v), dtype=np.float64) / 255 for k, v in M.PAL.items()}


# ------------------------------------------------------------------------------------------------ data
def context(sid: str) -> dict:
    import v5_build_formation as BF
    t = VP.load_trace("sca_v2_prog", sid)
    d = VP.TRACE / "sca_v2_prog" / sid
    final = Image.open(d / "final.png").convert("RGB")
    box = [int(v) for v in t["box_xys"]]
    centres = [tuple(float(v) for v in c) for c in BF.presets(final, box)]
    assert abs(centres[0][0] - (box[0] + box[2] / 2)) < 1 and abs(centres[0][1] - (box[1] + box[2] / 2)) < 1
    steps = t["steps"]
    pv = [Image.open(d / st["preview"]).convert("RGB") for st in steps]
    crops = [Image.open(d / st["crop"]).convert("RGB") for st in steps]
    w = {}
    for s in (1, 2):
        vals = [st["pi"][BLOCK] for st in steps if st["stage"] == s]
        w[s] = (float(np.median([v[H4] for v in vals])), float(1 - np.median([v[H7] for v in vals])))
    finals = []
    for cx, cy in centres:
        x0, y0 = int(round(cx - 256)), int(round(cy - 256))
        finals.append(final.crop((x0, y0, x0 + 512, y0 + 512)))
    return dict(sid=sid, t=t, pv=pv, crops=crops, final=final, final_sm=final.reduce(2), centres=centres,
                finals=finals, w=w, title=VP.showcase()[sid]["title"],
                first={s: [i for i, st in enumerate(steps) if st["stage"] == s] for s in (0, 1, 2)})


# ------------------------------------------------------------------------------------------------ geometry
def win_exact(c, n):
    """Window start (row, col) in tokens: the sampler's rule (64 tokens around the query's 8 x 8 tile, clamped)."""
    qc, qr = c[0] / 4096 * n, c[1] / 4096 * n

    def st(p):
        return int(np.clip(8 * int(p // 8) + 4 - 32, 0, n - 64))
    return st(qr), st(qc)


def win_cont(c, n):
    qc, qr = c[0] / 4096 * n, c[1] / 4096 * n
    return float(np.clip(qr - 32, 0, n - 64)), float(np.clip(qc - 32, 0, n - 64))


def kernel(r):
    yy, xx = np.mgrid[0:TP, 0:TP] + 0.5
    dd = np.sqrt((yy - TP / 2) ** 2 + (xx - TP / 2) ** 2)
    return np.clip(r + 0.5 - dd, 0, 1)


KER = {k: kernel(r) for k, r in (("dense", 1.05), ("g0", 1.15), ("g1", 1.6), ("g2", 2.15))}


def canvas(img: Image.Image, n: int, stage_key: str, wrc, a_dense, thin, a_grid, a_tint, dense_col="near",
           a_frame=0.0):
    """Compose the canvas at its stage size (n tokens, TP px per token at 2x): image, tints, token dots."""
    side = n * TP
    base = np.asarray(img.resize((side, side), Image.LANCZOS), dtype=np.float64) / 255
    stride = n // 64
    inwin = np.zeros((n, n), bool)
    if wrc is not None:
        r0, c0 = int(round(wrc[0])), int(round(wrc[1]))
        inwin[r0:r0 + 64, c0:c0 + 64] = True
    grid = np.zeros((n, n), bool)
    grid[stride // 2::stride, stride // 2::stride] = True
    out = base
    if a_tint > 0:
        out = out * (1 - 0.05 * a_tint) + C["far"] * 0.05 * a_tint
        m = np.kron(inwin, np.ones((TP, TP)))[..., None] * 0.12 * a_tint
        out = out * (1 - m) + C["near"] * m
    if a_dense > 0:   # every token: inside the window stays, outside three of every four (or 15 of 16) fade
        A = np.where(inwin, 1.0, np.where(grid, 1.0, 1.0 - thin)) * a_dense
        A = np.where(grid & ~inwin, A * (1 - thin), A)          # outside grid dots are drawn by the grid layer
        col = np.where(inwin[..., None], C[dense_col], C["ink"] * 0.85)
        al = np.kron(A, KER["dense"])[..., None] * 0.75
        out = out * (1 - al) + np.kron(col, np.ones((TP, TP, 1))) * al
    if a_grid > 0:
        al = np.kron(grid * a_grid, KER[stage_key])[..., None] * 0.95
        out = out * (1 - al) + C["far"] * al
    return Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8))


# ------------------------------------------------------------------------------------------------ timeline
def stage_at(t):
    """Scene, local time and its constants."""
    if t < T0:
        return 0, t
    if t < T0 + T1:
        return 1, t - T0
    if t < T0 + T1 + T2:
        return 2, t - T0 - T1
    if t < T0 + T1 + T2 + T3:
        return 3, t - T0 - T1 - T2
    return 4, t - T0 - T1 - T2 - T3


def step_frame(ctx, s, u, t0, t1):
    idx = ctx["first"][s]
    if u <= t0:
        return idx[0], 1.0, idx[0]
    f = min(1.0, (u - t0) / (t1 - t0)) * (len(idx) - 1)
    i = int(f)
    a = float(M.smoothstep((f - i) / 0.6)) if i < len(idx) - 1 else 1.0
    return idx[min(i + 1, len(idx) - 1)] if a > 0 and i < len(idx) - 1 else idx[i], (a if i < len(idx) - 1 else 1.0), \
        idx[i]


def preview(ctx, s, u, t0, t1):
    nxt, a, cur = step_frame(ctx, s, u, t0, t1)
    if nxt == cur or a <= 0:
        return ctx["pv"][cur], cur
    return Image.blend(ctx["pv"][cur], ctx["pv"][nxt], a), nxt


def crop_at(ctx, s, u, t0, t1):
    nxt, a, cur = step_frame(ctx, s, u, t0, t1)
    if nxt == cur or a <= 0:
        return ctx["crops"][cur]
    return Image.blend(ctx["crops"][cur], ctx["crops"][nxt], a)


# ------------------------------------------------------------------------------------------------ drawing helpers
def brackets(d, box, col, arm=10, w=2):
    x0, y0, x1, y1 = [v * S for v in box]
    arm, w = arm * S, round(w * S)
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(x, y), (x + dx * arm, y)], fill=col, width=w)
        d.line([(x, y), (x, y + dy * arm)], fill=col, width=w)


def strokes(d, src, target, period, travel, length, col, width, u, alpha, glow=None):
    """A steady flow of uniform pulses, one per source, staggered evenly in time (which tokens are read, never how
    strongly). Each source fires once per period; the pulse travels for `travel` of the period, then rests. With
    glow=(radius, colour), the source dot is drawn brighter, pulsing as it fires."""
    tx, ty = target
    n = max(1, len(src))
    for k, (sx, sy) in enumerate(src):
        ph = ((u / period) + k / n) % 1.0
        if glow:
            g = 0.65 + 0.35 * max(0.0, 1 - ph / 0.25)
            r = glow[0] * S
            d.ellipse([sx * S - r, sy * S - r, sx * S + r, sy * S + r], fill=glow[1] + (int(255 * alpha * g),))
        if ph > travel:
            continue
        q = ph / travel
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy)
        if dist < 3:
            continue
        a = q * (1 - length / max(dist, length))
        b = min(1.0, a + length / dist)
        p0 = (sx + dx * a, sy + dy * a)
        p1 = (sx + dx * b, sy + dy * b)
        fade = min(1.0, q / 0.2, (1 - q) / 0.2)
        d.line([(p0[0] * S, p0[1] * S), (p1[0] * S, p1[1] * S)], fill=col + (int(255 * alpha * fade),),
               width=round(width * S))


def caption(img, txt, al):
    f = M.font(30, "sans", 500)
    x, y = CAP
    if f.getlength(txt) <= CAP_W:
        M.text_alpha(img, (x, y), txt, f, M.PAL["ink"], al, anchor="ms")
        return
    words = txt.split(" ")
    i = min(range(1, len(words)), key=lambda j: max(f.getlength(" ".join(words[:j])), f.getlength(" ".join(words[j:]))))
    M.text_alpha(img, (x, y - 19), " ".join(words[:i]), f, M.PAL["ink"], al, anchor="ms")
    M.text_alpha(img, (x, y + 19), " ".join(words[i:]), f, M.PAL["ink"], al, anchor="ms")


def text(img, xy, s, size, weight=600, col="ink", anchor="ls", alpha=1.0, family="sans"):
    M.text_alpha(img, xy, s, M.font(size, family, weight), M.PAL[col] if col in M.PAL else col, alpha, anchor=anchor)


# ------------------------------------------------------------------------------------------------ frame
def frame(ctx, t: float) -> Image.Image:
    total = T0 + T1 + T2 + T3 + T4
    if t >= total:
        a = float(M.smoothstep((t - total) / (TB - 1 / FPS)))
        return Image.blend(frame(ctx, total - 1 / FPS), frame(ctx, 0.0), a)
    sc, u = stage_at(t)
    big = Image.new("RGB", (W * S, H * S), M.rgb(M.PAL["screen"]))
    ov = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov, "RGBA")
    pc = ctx["centres"][0]
    focus_c = pc
    zoom_img, zoom_lab, zoom_a, one = None, "the patch, magnified", 0.0, False
    bars_a, legend_a, legend = 0.0, 0.0, None
    inset = None
    recap = sc == 4
    if sc == 0:
        side, s_content, img = 640.0, 2, ctx["final_sm"]
        layers = dict(n=256, key="g2", wrc=None, a_dense=0, thin=0, a_grid=0, a_tint=0)
        a_win = 0.0
        zoom_img, zoom_a, one = ctx["finals"][0].crop((116, 116, 396, 396)), 1.0, True
        zoom_lab = "the patch, 1:1"
    elif sc == 1:
        g = float(M.smoothstep(u / 0.8))
        side = 640 + (160 - 640) * g
        pv, _ = preview(ctx, 0, u, 0.8, 5.4)
        img = Image.blend(ctx["final_sm"].resize((1024, 1024), Image.LANCZOS), pv, g) if g < 1 else pv
        on = float(M.smoothstep((u - 1.0) / 0.6))
        layers = dict(n=64, key="g0", wrc=(0, 0), a_dense=0, thin=0, a_grid=on, a_tint=0)
        a_win = on
        inset = ("comp", on)
        legend_a, legend = on, ("the whole 1K canvas", "every token of the 1K canvas")
        zoom_a = 1 - on
        zoom_img, one, zoom_lab = ctx["finals"][0].crop((116, 116, 396, 396)), True, "the patch, 1:1"
    elif sc in (2, 3):
        s = sc - 1
        n_from, n_to = NT[s - 1], NT[s]
        grow = float(M.smoothstep(u / 1.2))
        side = n_from * 2.5 + (n_to - n_from) * 2.5 * grow
        if u < 1.2:
            img = ctx["pv"][ctx["first"][s - 1][-1]]
        else:
            img, _ = preview(ctx, s, u, 1.2, 7.4 if s == 1 else 4.4)
        dense = float(M.smoothstep((u - 1.2) / 0.4))
        thin = float(M.smoothstep((u - 1.7) / 0.9))
        on = float(M.smoothstep((u - 2.6) / 0.5))
        travel = s == 2 and u >= 5.0
        if travel:                       # scene 3: the focus visits the two other presets
            k = 1 if u < 6.5 else 2
            m = float(M.smoothstep((u - (5.0 if k == 1 else 6.5)) / 0.55))
            a_c, b_c = ctx["centres"][k - 1], ctx["centres"][k]
            focus_c = (a_c[0] + (b_c[0] - a_c[0]) * m, a_c[1] + (b_c[1] - a_c[1]) * m)
        wrc = win_cont(focus_c, n_to)
        if not travel:
            ex = win_exact(focus_c, n_to)
            e = float(M.smoothstep((u - 1.0) / 0.3))
            wrc = (wrc[0] + (ex[0] - wrc[0]) * e, wrc[1] + (ex[1] - wrc[1]) * e)
        layers = dict(n=n_to, key=f"g{s}", wrc=wrc, a_dense=dense * (1 - 0.0), thin=thin, a_grid=thin, a_tint=on)
        a_win = 1.0
        legend_a = 1.0
        legend = (("every token in the window: half the side", "one per 2 × 2 block, the whole image") if s == 1 else
                  ("every token in the window: a quarter of the side", "one per 4 × 4 block, the whole image"))
        bars_a = on
        if s == 1 and u < 1.2:
            zoom_img = crop_at(ctx, 0, 9, 0, 1)
        elif not travel:
            zoom_img = crop_at(ctx, s, u, 1.2, 7.4 if s == 1 else 4.4)
        zoom_a = 1.0 if sc == 3 else float(M.smoothstep(u / 0.5))
        zoom_lab = "the patch, magnified"
        if s == 2 and u >= 4.4 and not travel:          # end of the run: zoom to native pixels
            z = float(M.smoothstep((u - 4.4) / 0.5))
            span = 512 + (PZ - 512) * z
            fin = ctx["finals"][0]
            zoom_img = fin.resize((PZ, PZ), Image.LANCZOS, box=(256 - span / 2, 256 - span / 2, 256 + span / 2,
                                                                256 + span / 2))
            one = z >= 1
        if travel:
            k = 1 if u < 6.5 else 2
            zoom_img = ctx["finals"][k].crop((116, 116, 396, 396))
            one = True
        if one:
            zoom_lab = "the patch, 1:1"
    else:   # recap
        side, img = 640.0, ctx["final_sm"]
        layers = None
        a_win = 1.0
    # ---------------------------------------------------------------- canvas (to scale)
    if not recap:
        n = layers["n"]
        comp = canvas(img, n, layers["key"], layers["wrc"], layers["a_dense"], layers["thin"], layers["a_grid"],
                      layers["a_tint"])
        sd = round(side * S)
        cx0, cy0 = AX, AY - side
        if sc in (1, 2):                 # a very faint outline of the 4K extent: room to grow
            d.rectangle([K4[0] * S, K4[1] * S, K4[2] * S, K4[3] * S], outline=(236, 238, 233, 18), width=round(1.5 * S))
        big.paste(comp.resize((sd, sd), Image.LANCZOS), (round(cx0 * S), round(cy0 * S)))
        ctx["_comp"] = comp
        tok = side / n
        # window frame (64 tokens = 160 px at every stage)
        if a_win > 0:
            if sc == 1 or sc == 0:
                wr, wc = 0.0, 0.0
                wside = side
            else:
                wr, wc = layers["wrc"] if layers["wrc"] else (0, 0)
                wside = 64 * tok
            wx0, wy0 = cx0 + wc * tok, cy0 + wr * tok
            d.rectangle([wx0 * S, wy0 * S, (wx0 + wside) * S, (wy0 + wside) * S],
                        outline=M.rgb(M.PAL["near"]) + (int(255 * a_win),), width=round(2.4 * S))
        # patch brackets (green accent)
        pb = side / 8
        px_, py_ = cx0 + focus_c[0] / 4096 * side, cy0 + focus_c[1] / 4096 * side
        brackets(d, (px_ - pb / 2, py_ - pb / 2, px_ + pb / 2, py_ + pb / 2), M.rgb(M.PAL["sca"]) + (255,),
                 arm=max(6, pb / 4), w=2.2)
        # reading strokes: which tokens the patch reads (uniform)
        if sc in (2, 3) and bars_a > 0:
            wr, wc = layers["wrc"]
            loc = [(cx0 + (wc + c + 0.5) * tok, cy0 + (wr + r + 0.5) * tok) for r in (6, 19, 32, 45, 58)
                   for c in (6, 19, 32, 45, 58) if not (r == 32 and c == 32)]
            st = n // 64
            glo = [(cx0 + (st * b2 + st // 2 + 0.5) * tok, cy0 + (st * b1 + st // 2 + 0.5) * tok)
                   for b1 in (4, 18, 32, 46, 60) for b2 in (4, 18, 32, 46, 60)]
            strokes(d, glo, (px_, py_), 2.6, 0.55, 22, M.rgb(M.PAL["far"]), 1.4, u, 0.9 * bars_a,
                    glow=(2.6, M.rgb("#9CC3EE")))
            strokes(d, loc, (px_, py_), 1.6, 0.45, 7, M.rgb(M.PAL["near"]), 1.4, u, 0.9 * bars_a)
    else:
        a_in = float(M.smoothstep(u / 0.8))
        RC = RECAP
        for s, base, xr in zip((0, 1, 2), (160, 320, 640), RC["xs"]):
            n = NT[s]
            al = 1.0 if s == 2 else a_in
            sd_end = base * RC["scale"]
            if s == 2:                    # the 4K canvas glides from its place to the recap row
                sd1 = 640 + (sd_end - 640) * a_in
                x0 = AX + (xr - AX) * a_in
                y0 = (AY - 640) + ((RC["bottom"] - sd_end) - (AY - 640)) * a_in
            else:
                sd1, x0, y0 = sd_end, xr, RC["bottom"] - sd_end
            idx = ctx["first"][s][-1]
            im = ctx["final_sm"] if s == 2 else ctx["pv"][idx]
            wrc = win_exact(pc, n)
            comp = canvas(im, n, f"g{s}", wrc, 0, 1, 1.0, 0.0)
            ci = comp.resize((round(sd1 * S), round(sd1 * S)), Image.LANCZOS)
            if al < 1:
                ci = Image.blend(Image.new("RGB", ci.size, M.rgb(M.PAL["screen"])), ci, al)
            big.paste(ci, (round(x0 * S), round(y0 * S)))
            tok = sd1 / n
            wx0, wy0 = x0 + wrc[1] * tok, y0 + wrc[0] * tok
            d.rectangle([wx0 * S, wy0 * S, (wx0 + 64 * tok) * S, (wy0 + 64 * tok) * S],
                        outline=M.rgb(M.PAL["near"]) + (int(255 * al),), width=round(2.4 * S))
            pb = sd1 / 8
            px_, py_ = x0 + pc[0] / 4096 * sd1, y0 + pc[1] / 4096 * sd1
            brackets(d, (px_ - pb / 2, py_ - pb / 2, px_ + pb / 2, py_ + pb / 2), M.rgb(M.PAL["sca"]) + (int(255 * al),),
                     arm=max(5, pb / 4), w=2.0)
            ctx.setdefault("_recap_lab", {})[s] = (x0, al)
    big = big.convert("RGBA")
    big.alpha_composite(ov)
    img1 = big.convert("RGB").resize((W, H), Image.LANCZOS)
    # ---------------------------------------------------------------- right panel
    if not recap:
        dd = ImageDraw.Draw(img1, "RGBA")
        if inset is not None and inset[1] > 0:
            ins = ctx["_comp"].resize((PZ, PZ), Image.LANCZOS)
            img1.paste(Image.blend(Image.new("RGB", ins.size, M.rgb(M.PAL["screen"])), ins, inset[1]), (PX0, PY0))
            dd.rectangle([PX0 - 1, PY0 - 1, PX0 + PZ, PY0 + PZ], outline=M.rgb(M.PAL["near"]) + (int(255 * inset[1]),),
                         width=3)
            text(img1, (PX0, PY0 - 14), "the 1K canvas, magnified", 26, 600, "ink2", alpha=inset[1])
        if zoom_img is not None and zoom_a > 0:
            z = zoom_img.resize((PZ, PZ), Image.LANCZOS)
            img1.paste(Image.blend(Image.new("RGB", z.size, M.rgb(M.PAL["screen"])), z, zoom_a), (PX0, PY0))
            dd.rectangle([PX0 - 1, PY0 - 1, PX0 + PZ, PY0 + PZ], outline=M.rgb(M.PAL["sca"]) + (int(255 * zoom_a),),
                         width=2)
            text(img1, (PX0, PY0 - 14), zoom_lab, 26, 600, "ink2", alpha=zoom_a)
        if legend_a > 0 and legend:
            ly = LEG_Y
            al = legend_a
            lay = Image.new("RGBA", img1.size, (0, 0, 0, 0))
            dl = ImageDraw.Draw(lay)
            dl.rectangle([PX0, ly - 22, PX0 + 22, ly], outline=M.rgb(M.PAL["near"]) + (int(255 * al),), width=3)
            for yy in (ly + 52, ly + 62):
                for xx in (PX0 + 3, PX0 + 11, PX0 + 19):
                    dl.ellipse([xx - 2.5, yy - 2.5, xx + 2.5, yy + 2.5], fill=M.rgb(M.PAL["far"]) + (int(255 * al),))
            img1 = img1.convert("RGBA")
            img1.alpha_composite(lay)
            img1 = img1.convert("RGB")
            text(img1, (PX0 + 40, ly), "local: 64 × 64 tokens", 26, 600, "near", alpha=al)
            if bars_a > 0 and sc in (2, 3):
                text(img1, (LEG_X + 330, ly), "reads nearby", 26, 400, "near", alpha=bars_a)
                text(img1, (LEG_X + 346, ly + 72), "reads everywhere", 26, 400, "far", alpha=bars_a)
            text(img1, (PX0 + 40, ly + 32), legend[0], 26, 400, "ink2", alpha=al)
            text(img1, (PX0 + 40, ly + 72), "global: 64 × 64 tokens", 26, 600, "far", alpha=al)
            text(img1, (PX0 + 40, ly + 104), legend[1], 26, 400, "ink2", alpha=al)
        if bars_a > 0 and sc in (2, 3):
            s = sc - 1
            loc4, glo7 = ctx["w"][s]
            dd = ImageDraw.Draw(img1, "RGBA")
            for j, (lab, share_local, right, col) in enumerate(
                    (("a near-looking head", loc4, f"{round(100 * loc4)}% local", "near"),
                     ("a far-looking head", 1 - glo7, f"{round(100 * glo7)}% global", "far"))):
                y = PY0 + 30 + j * 130
                text(img1, (BX0, y), lab, 26, 600, "ink", alpha=bars_a)
                text(img1, (BX0, y + 34), right, 26, 600, col, alpha=bars_a)
                bw, bh, by = 320, 14, y + 52
                cut = BX0 + bw * share_local
                if share_local > 0.005:
                    dd.rectangle([BX0, by, cut, by + bh], fill=M.rgb(M.PAL["near"]) + (int(255 * bars_a),))
                if share_local < 0.995:
                    dd.rectangle([cut, by, BX0 + bw, by + bh], fill=M.rgb(M.PAL["far"]) + (int(255 * bars_a),))
            text(img1, (BX0, PY0 + 30 + 2 * 130 - 10), "heads 4 and 7 of block 12", 26, 400, "ink3",
                 alpha=bars_a)
        if one and zoom_a > 0.5:
            M.chip(img1, (PX0 + 10, PY0 + 10), "1:1", 22, anchor="lt", bg=(8, 10, 9, 200), fg=(236, 238, 233, 255),
                   weight=600, radius=5)
    else:
        for s, (x0, al) in ctx.get("_recap_lab", {}).items():
            text(img1, (x0, RECAP["lab_y"]), ["1K", "2K", "4K"][s], 26, 600, "ink", alpha=al)
            text(img1, (x0, RECAP["lab_y"] + 32), RECAP["labs"][s], 26, 400, "ink2",
                 alpha=al)
    # ---------------------------------------------------------------- title, stage indicator, caption
    text(img1, TITLE, "One patch, three sizes", 34, 600, "ink", family="serif")
    labs = ["1K", "2K", "4K"]
    cur = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3}[sc]
    x = IND[0]
    xs = []
    for j in (2, 1, 0):
        f = M.font(26, "sans", 600)
        wdt = f.getlength(labs[j])
        x -= wdt
        xs.insert(0, (x, labs[j]))
        x -= 40
    for j, (xx, lab) in enumerate(xs):
        on = j == cur or cur == 3
        text(img1, (xx, IND[1]), lab, 26, 600, "sca" if on else "ink3")
        if j < 2:
            text(img1, (xx + M.font(26, "sans", 600).getlength(lab) + 20, IND[1]), "·", 26, 600, "ink3", anchor="ms")
    prog = float(np.clip((t - T0) / (T1 + T2 + T3), 0, 1))
    dd = ImageDraw.Draw(img1)
    x_a, x_b = xs[0][0], IND[0]
    dd.rectangle([x_a, IND[1] + 14, x_b, IND[1] + 17], fill=M.rgb(M.PAL["rule"]))
    if prog > 0:
        dd.rectangle([x_a, IND[1] + 14, x_a + (x_b - x_a) * prog, IND[1] + 17], fill=M.rgb(M.PAL["sca"]))
    t0s = [0, T0, T0 + T1, T0 + T1 + T2, T0 + T1 + T2 + T3, total]
    al = min(1.0, (t - t0s[sc]) / 0.35, (t0s[sc + 1] - t) / 0.35) if sc > 0 else min(1.0, (t0s[1] - t) / 0.35)
    if sc == 0:
        al = min(1.0, (t0s[1] - t) / 0.35)
    caption(img1, CAPS[sc], max(0.0, al))
    return img1


# ------------------------------------------------------------------------------------------------ build
_CTX: dict = {}


def _work(task):
    sid, cut, idxs, outdir = task
    use(cut)
    key = (sid, cut)
    if key not in _CTX:
        _CTX.clear()
        _CTX[key] = context(sid)
    for k in idxs:
        frame(_CTX[key], k / FPS).save(Path(outdir) / f"{k:05d}.png", compress_level=1)
    return len(idxs)


def build(sid: str, order: int, procs: int, cut: str = "16x9"):
    use(cut)
    total = T0 + T1 + T2 + T3 + T4 + TB
    n = round(total * FPS)
    name = f"patch3_{sid.split('_', 2)[-1]}" + ("" if cut == "16x9" else "_9x16")
    fdir = BUILD / name / f"frames_n{n}"
    fdir.mkdir(parents=True, exist_ok=True)
    chunk = max(4, math.ceil(n / procs))
    tasks = [(sid, cut, list(range(i, min(n, i + chunk))), str(fdir)) for i in range(0, n, chunk)]
    t0 = time.time()
    with Pool(min(procs, len(tasks))) as pool:
        sum(pool.imap_unordered(_work, tasks))
    print(f"{name}: {n} frames in {time.time() - t0:.1f} s", flush=True)
    mp4 = M.OUT / f"{name}.mp4"
    enc = M.encode_to_size(fdir / "%05d.png", mp4, FPS, n, MAX_BYTES, crf_floor=20, preset="veryslow", gop=n,
                           passlog=BUILD / name / "x264pass", x264="aq-mode=3", threads=16)
    p = M.check_video(mp4, max_bytes=MAX_BYTES, w=W, h=H)
    t_rest = T0 + T1 + T2 + 3.8
    poster = M.poster_webp(Image.open(fdir / f"{round(t_rest * FPS):05d}.png"), M.OUT / f"{name}.webp", quality=84)
    ctx = context(sid)
    title = ctx["title"]
    alt = (f"{title}: one patch followed through progressive SCA's three sizes. The canvas grows from 1K to 2K to 4K "
           f"while the orange window the patch reads stays 64 by 64 tokens, covering the whole image, then half, then a "
           f"quarter of its side, and the blue grid of 64 by 64 keys spreads to one token per 2 by 2 and per 4 by 4 "
           f"block.")
    cap = (f"{title}: one patch, three sizes. Local: 64 x 64 tokens around the patch; global: 64 x 64 tokens over the "
           f"whole image. Canvases to scale; strokes show which tokens the patch reads; positions exact; weights from "
           f"the run.")
    e = M.entry_for(name, mp4, poster, loop=True, alt=alt, caption=cap,
                    sources=dict(trace=str(VP.TRACE / "sca_v2_prog" / sid), prompt_id=sid,
                                 patch_centres=[list(c) for c in ctx["centres"]],
                                 patch_rule="formation component presets (v5_build_formation.presets)",
                                 weights={"2K": dict(head4_local=ctx["w"][1][0], head7_global=ctx["w"][1][1]),
                                          "4K": dict(head4_local=ctx["w"][2][0], head7_global=ctx["w"][2][1])},
                                 weights_rule="median over the stage's steps of the trace's per-step pi, block 12"),
                    script="v5_patch3.py", order=45 + order + (0 if cut == "16x9" else 5),
                    extra=dict(role="patch3" if cut == "16x9" else "patch3_9x16", title=title, prompt_id=sid,
                               captions=CAPS, encode=enc, poster_time=round(t_rest, 3),
                               timing=dict(final=T0, k1=T1, k2=T2, k4=T3, recap=T4, back=TB)))
    M.manifest_put(e)
    sd = M.BUILD / "sheets"
    M.contact_sheet(mp4, sd / f"{name}_sheet.png", n=20, cols=4, tile_w=480 if cut == "16x9" else 270)
    M.seam_sheet(mp4, sd / f"{name}_seam.png", tile_w=640 if cut == "16x9" else 360)
    print(json.dumps(dict(name=name, **p, enc=enc)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cut", default="16x9", choices=["16x9", "9x16", "both"])
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--look", type=str, default=None)
    a = ap.parse_args()
    sel = VP.selection(a.k)
    if a.only:
        sel = [s for s in sel if s == a.only] or [a.only]
    for c in CAPS:
        M.check_words(c, where="patch3 caption")
    cuts = ["16x9", "9x16"] if a.cut == "both" else [a.cut]
    if a.look:
        for cut in cuts:
            use(cut)
            ctx = context(sel[0])
            outs = []
            for tv in a.look.split(","):
                p = BUILD / "look" / f"{sel[0]}_{cut}_{float(tv):05.2f}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                frame(ctx, float(tv)).save(p)
                outs.append(p)
            M.frames_sheet(outs, BUILD / "look" / f"{sel[0]}_{cut}_look.png", cols=2 if cut == "16x9" else 4,
                           tile_w=800 if cut == "16x9" else 400, label=f"{sel[0]} {cut}")
            print(BUILD / "look" / f"{sel[0]}_{cut}_look.png")
        return
    for cut in cuts:
        for i, sid in enumerate(sel):
            build(sid, i, a.procs, cut)


if __name__ == "__main__":
    main()
