"""Method loops of the v5 page (8 to 10 s each, seamless, 1600 x 900, on the darkroom ground):

  method_window.mp4      (a) one head's two native-size views on a 4096^2 image while the query moves: the 64 x 64
                         token window around the query's 8 x 8 tile (near view, orange) and the 64 x 64 grid of block
                         centres over the whole image (far view, blue), for a near-reaching head (block 12, head 4)
                         and a far-reaching head (head 7). Each view is drawn at the head's measured weight pi at
                         ratio 4 (the reach table, noise bin sigma = 0.62): window opacity grows with pi, grid opacity
                         with 1 - pi. Geometry exactly as Section 3.1 (tile 8, window 64 clamped to the grid,
                         block-centre subsampling at stride 4).
  method_emulation.mp4   (b) reach emulation at 1024^2 for block 12, head 10 (the paper's Figure 3 head): its real
                         full attention at the query (the face of the leftmost sheep, step 20, sigma 0.61), the two
                         views shrinking by r (window 64 -> 32 -> 16 tokens, grid stride 1 -> 2 -> 4), the real
                         emulated near and far attention at r = 4, then the mix pi * near + (1 - pi) * far as pi
                         sweeps and settles at pi* = clip(B / A, 0, 1) on the real J(pi) = E - 2 pi B + pi^2 A of
                         that cell (32 emulation prompts).
  method_stages.mp4      (c) progressive SCA's three sampler stages on a recorded run: the decoded estimate of every
                         step at its stage size (1024^2, 2048^2, 4096^2 drawn to scale), and the weight pi of all
                         1,368 heads (57 blocks x 24 heads) that the step read from the reach table, at r = 2 for the
                         2048^2 stage and r = 4 for the 4096^2 stage; the 1024^2 stage runs the model's own attention.

Data (real, never typed): paper/figures/src/data/v6_attn_block12.npz (attention capture), results/evidence/calib/
k1_1024.npz (the reach table and the A, B, E sums), paper/figures/v6/method_assets/provenance.json (J sums of the
illustrated heads, checked against k1_1024), paper/figures/attn_focus/image_1024.png (the native stage of the run),
the 4096^2 output of that run (fig_attn_focus.TARGET), and the progressive trace
/scratch/09032/saini_2/sca/web_assets/trace/sca_v2_prog/<id>/ (x0 previews and per-step pi).
Run (prompt for (c): the first draw of the paper hero's rule that has a recorded run):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_method.py [window|emulation|stages|all]
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
SS = 2
MAX_BYTES = 2_500_000
BUILD = M.BUILD / "method"
ATTN = M.FIGSRC / "data" / "v6_attn_block12.npz"
CALIB = M.REPO / "results" / "evidence" / "calib" / "k1_1024.npz"
MASSETS = M.V6 / "method_assets" / "provenance.json"
IMG1024 = M.REPO / "paper" / "figures" / "attn_focus" / "image_1024.png"
TARGET = Path("/scratch/09032/saini_2/sca/out/dupfix_prog/sca_v2_prog/4096x4096/"
              "a4k_eval_4096_minus8343371271909676658_0325f0a692_seed0.png")
TRACE = Path("/scratch/09032/saini_2/sca/web_assets/trace/sca_v2_prog")
BLOCK, BIN = 11, 8                        # block 12 (zero-based 11); noise bin centred at sigma = 0.618
NEAR_HEAD, FAR_HEAD, MID_HEAD = 3, 6, 9   # heads 4, 7, 10 (zero-based), as the paper's method assets

C = {k: np.array(M.rgb(v), dtype=np.float64) / 255 for k, v in M.PAL.items()}


# ------------------------------------------------------------------------------------------------ helpers
def label(d, xy, s, size, weight=600, fill="ink", anchor="la", family="sans"):
    d.text(xy, s, font=M.font(size * SS, family, weight), fill=M.rgb(M.PAL[fill]), anchor=anchor,
           features=["tnum"])


def kicker(d, s):
    label(d, (96 * SS, 60 * SS), s.upper(), 24, 600, "ink2")


def pi_color(p):
    """Diverging colour of a weight: far blue (pi = 0), a dark neutral (1/2), near orange (pi = 1)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)[..., None]
    mid = np.array(M.rgb("#3B423E"), dtype=np.float64) / 255
    lo = mid + (C["far"] - mid) * np.clip(1 - 2 * p, 0, 1)
    hi = mid + (C["near"] - mid) * np.clip(2 * p - 1, 0, 1)
    return np.where(p < 0.5, lo, hi)


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def heat_alpha(dens, dmax=100.0):
    v = np.clip(np.log10(np.maximum(dens, 1.0)) / np.log10(dmax), 0, 1)
    return np.minimum(1.0, v ** 0.6)


def finish(img: Image.Image) -> Image.Image:
    return img.resize((W, H), Image.LANCZOS) if img.size != (W, H) else img


def encode(name: str, fdir: Path, n: int, poster_frame: int, alt: str, caption: str, sources, role: str,
           order: int, extra: dict | None = None):
    mp4 = M.OUT / f"{name}.mp4"
    enc = M.encode_to_size(fdir / "%05d.png", mp4, FPS, n, MAX_BYTES, crf_floor=18, preset="veryslow", gop=n,
                           passlog=BUILD / name / "x264pass", x264="aq-mode=3", threads=16)
    p = M.check_video(mp4, max_bytes=MAX_BYTES, w=W, h=H)
    poster = M.poster_webp(Image.open(fdir / f"{poster_frame:05d}.png"), M.OUT / f"{name}.webp", quality=84)
    e = M.entry_for(name, mp4, poster, loop=True, alt=alt, caption=caption, sources=sources, script="v5_method.py",
                    order=order, extra=dict(role=role, encode=enc, **(extra or {})))
    M.manifest_put(e)
    sd = M.BUILD / "sheets"
    M.contact_sheet(mp4, sd / f"{name}_sheet.png", n=12, cols=4, tile_w=480)
    M.seam_sheet(mp4, sd / f"{name}_seam.png", tile_w=640)
    print(json.dumps(dict(name=name, **p, enc=enc)), flush=True)


def render_all(name: str, fn, n: int, procs: int = 48) -> Path:
    fdir = BUILD / name / f"frames_n{n}"
    fdir.mkdir(parents=True, exist_ok=True)
    tasks = [(name, list(range(i, min(n, i + 6))), str(fdir)) for i in range(0, n, 6)]
    t0 = time.time()
    with Pool(procs) as pool:
        sum(pool.imap_unordered(_work, tasks))
    print(f"{name}: {n} frames in {time.time() - t0:.1f} s", flush=True)
    return fdir


_CTX: dict = {}


def xfade_loop(core, ctx, t: float, t_back: float, total: float) -> Image.Image:
    """Frames before t_back come from core(t); after it, the rest frame cross-fades into frame 0, reaching it on the
    last frame, so the loop closes on itself."""
    if t < t_back:
        return core(ctx, t)
    a = float(M.smoothstep((t - t_back) / max(1e-6, total - t_back - 1.0 / FPS)))
    return Image.blend(core(ctx, t_back - 1.0 / FPS), core(ctx, 0.0), a)


def _work(task):
    name, idxs, outdir = task
    if name not in _CTX:
        _CTX[name] = CONTEXTS[name]()
    ctx = _CTX[name]
    for k in idxs:
        DRAW[name](ctx, k).save(Path(outdir) / f"{k:05d}.png", compress_level=1)
    return len(idxs)


def reach_table() -> dict:
    z = np.load(CALIB)
    return dict(s2=z["s2_pi_head"], s4=z["s4_pi_head"], sigma=z["sigma_centers"], populated=z["populated"],
                stats=z["stats"])


# ================================================================================================ (a) window
N4096, TILE, WIN = 256, 8, 64
A_SECONDS = 8.0
A_PANEL = 580


def a_context():
    t = reach_table()
    im = Image.open(TARGET).convert("RGB")
    base = im.reduce(4).resize((A_PANEL * SS, A_PANEL * SS), Image.LANCZOS)
    base = Image.blend(base, Image.new("RGB", base.size, M.rgb(M.PAL["screen"])), 0.38)
    pis = {h: float(t["s4"][BLOCK, h, BIN]) for h in (NEAR_HEAD, FAR_HEAD)}
    return dict(base=base, pi=pis, sigma=float(t["sigma"][BIN]))


def a_query(k: int, n: int) -> tuple:
    """The query's token (row, col) on a closed Lissajous path through the image (period = the loop)."""
    th = 2 * math.pi * k / n
    r = 128 + 92 * math.sin(2 * th)
    c = 128 + 100 * math.cos(th)
    return r, c


def a_window(r, c):
    def start(p):
        return int(np.clip(TILE * (int(p) // TILE) + TILE // 2 - WIN // 2, 0, N4096 - WIN))
    return start(r), start(c)


def a_draw(ctx, k):
    n = round(A_SECONDS * FPS)
    img = Image.new("RGB", (W * SS, H * SS), M.rgb(M.PAL["screen"]))
    d = ImageDraw.Draw(img, "RGBA")
    kicker(d, f"One head, two native-size views · 4096 × 4096 · block 12")
    qr, qc = a_query(k, n)
    wr, wc = a_window(qr, qc)
    s = A_PANEL * SS / N4096                       # px per token
    xs = [150, 870]
    y0 = 172
    for x0, h in zip(xs, (NEAR_HEAD, FAR_HEAD)):
        X0, Y0 = x0 * SS, y0 * SS
        p = ctx["pi"][h]
        img.paste(ctx["base"], (X0, Y0))
        # far view: the 64 x 64 block centres (stride 4, centre offset 2) over the whole image
        ga = 0.10 + 0.90 * (1 - p)
        rad = 1.55 * SS
        col = M.rgb(M.PAL["far"]) + (int(255 * ga),)
        for b1 in range(64):
            cy = Y0 + (4 * b1 + 2 + 0.5) * s
            for b2 in range(64):
                cx = X0 + (4 * b2 + 2 + 0.5) * s
                d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col)
        # near view: the 64-token window of the query's tile, clamped to the grid
        wa = 0.08 + 0.42 * p
        bx0, by0 = X0 + wc * s, Y0 + wr * s
        bx1, by1 = bx0 + WIN * s, by0 + WIN * s
        d.rectangle([bx0, by0, bx1, by1], fill=M.rgb(M.PAL["near"]) + (int(255 * wa),))
        d.rectangle([bx0, by0, bx1, by1], outline=M.rgb(M.PAL["near"]) + (int(255 * (0.35 + 0.65 * p)),),
                    width=round(2.5 * SS))
        # the query's tile and the query
        tr, tc = TILE * (int(qr) // TILE), TILE * (int(qc) // TILE)
        d.rectangle([X0 + tc * s, Y0 + tr * s, X0 + (tc + TILE) * s, Y0 + (tr + TILE) * s],
                    outline=M.rgb(M.PAL["ink"]) + (230,), width=round(1.5 * SS))
        qx, qy = X0 + (qc + 0.5) * s, Y0 + (qr + 0.5) * s
        rq = 4.5 * SS
        d.ellipse([qx - rq, qy - rq, qx + rq, qy + rq], fill=M.rgb(M.PAL["ink"]), outline=M.rgb(M.PAL["screen"]),
                  width=round(1.2 * SS))
        # heading and weight bar
        name = "near-reaching" if h == NEAR_HEAD else "far-reaching"
        label(d, (X0, Y0 - 52 * SS), f"Head {h + 1}", 30, 600, "ink")
        hw = M.font(30 * SS, "sans", 600).getlength(f"Head {h + 1}")
        label(d, (X0 + hw + 14 * SS, Y0 - 50 * SS), name, 26, 400, "ink2")
        label(d, (X0 + A_PANEL * SS, Y0 - 50 * SS), f"π = {p:.2f}", 26, 600, "ink2", anchor="ra")
        by = Y0 + A_PANEL * SS + 26 * SS
        bw = A_PANEL * SS
        bh = 12 * SS
        cut = X0 + bw * p
        if p > 0.004:
            d.rounded_rectangle([X0, by, max(cut, X0 + bh), by + bh], radius=bh / 2, fill=M.rgb(M.PAL["near"]))
        if p < 0.996:
            d.rounded_rectangle([min(cut, X0 + bw - bh), by, X0 + bw, by + bh], radius=bh / 2,
                                fill=M.rgb(M.PAL["far"]))
        label(d, (X0, by + bh + 12 * SS), f"window {round(100 * p)}%", 24, 600, "near")
        label(d, (X0 + bw, by + bh + 12 * SS), f"grid {round(100 * (1 - p))}%", 24, 600, "far", anchor="ra")
    return finish(img)


# ================================================================================================ (b) emulation
N1024 = 64
B_PANEL = 576   # a multiple of 32 so the 64-token maps tile it exactly at 2x
T_FULL, T_SHRINK, T_VIEWS, T_MIX, T_HOLD, T_BACK = 1.6, 2.3, 1.3, 2.9, 1.1, 0.6


def b_times():
    a = 0.0
    out = {}
    for k, v in (("full", T_FULL), ("shrink", T_SHRINK), ("views", T_VIEWS), ("mix", T_MIX), ("hold", T_HOLD),
                 ("back", T_BACK)):
        out[k] = (a, a + v)
        a += v
    return out, a


def b_context():
    z = np.load(ATTN)
    assert int(z["block"]) == BLOCK and tuple(int(v) for v in z["query"]) == (43, 11)
    step_i = 1                                     # step 20, sigma 0.612
    full = z["full"][step_i, MID_HEAD].astype(np.float64)
    loc = z["local_r4"][MID_HEAD].astype(np.float64)
    glob = z["global_r4"][MID_HEAD].astype(np.float64)
    r0, c0, w = (int(v) for v in z["window_r4"][:3])
    t = reach_table()
    prov = json.loads(MASSETS.read_text())["j_curves"]["Mid"]
    A, B, E = prov["A"], prov["B"], prov["E"]
    # check the sums against the table: stats[r index 1 (r = 4)] holds A, B, E per block, head, bin
    pi_star = float(np.clip(B / A, 0, 1))
    assert abs(pi_star - float(t["s4"][BLOCK, MID_HEAD, BIN])) < 1e-6, "J sums disagree with the reach table"
    img = Image.open(IMG1024).convert("RGB").resize((B_PANEL * SS, B_PANEL * SS), Image.LANCZOS)
    g = np.asarray(img.convert("L"), dtype=np.float64) / 255
    base = np.repeat((0.10 + 0.55 * g)[..., None], 3, -1)       # a dim greyscale ground so heat reads
    return dict(full=full, loc=loc, glob=glob, win=(r0, c0, w), A=A, B=B, E=E, pi_star=pi_star, base=base,
                sigma=float(z["sigma"][step_i]), sigma_bin=float(t["sigma"][BIN]))


def b_heat_full(ctx):
    """Full attention: orange inside the r = 4 window, blue outside (the near / far split of Lemma 1)."""
    full = ctx["full"]
    dens = full / full.sum() * N1024 ** 2
    a = heat_alpha(dens)
    r0, c0, w = ctx["win"]
    inside = np.zeros_like(a, bool)
    inside[r0:r0 + w, c0:c0 + w] = True
    return a, inside


def b_draw(ctx, k):
    times, total = b_times()
    return xfade_loop(b_core, ctx, k / FPS, times["back"][0], total)


def b_core(ctx, t):
    times, total = b_times()
    img = Image.new("RGB", (W * SS, H * SS), M.rgb(M.PAL["screen"]))
    X0, Y0 = 120 * SS, 196 * SS
    P = B_PANEL * SS
    s = P / N1024
    r0, c0, w = ctx["win"]
    qr, qc = 43, 11

    def phase(name):
        a, b = times[name]
        return float(np.clip((t - a) / (b - a), 0, 1))

    # weights of the three layers on the left panel
    f_full = 1.0 if t < times["full"][1] else 1 - float(M.smoothstep(phase("shrink") / 0.35))
    back = float(M.smoothstep(phase("back")))
    f_full = max(f_full, back)
    u = phase("shrink")                                           # r: hold 1, ease to 2, hold, ease to 4, hold
    if u < 0.125:
        r_now, r_lab = 1.0, 1
    elif u < 0.375:
        r_now, r_lab = 2 ** float(M.ease_in_out_sine((u - 0.125) / 0.25)), 2
    elif u < 0.54:
        r_now, r_lab = 2.0, 2
    elif u < 0.79:
        r_now, r_lab = 2 * 2 ** float(M.ease_in_out_sine((u - 0.54) / 0.25)), 4
    else:
        r_now, r_lab = 4.0, 4
    if t >= times["shrink"][1]:
        r_now, r_lab = 4.0, 4
    views = float(M.smoothstep(phase("views")))
    if t >= times["views"][1]:
        views = 1.0
    views *= (1 - back)
    # pi: hidden before the mix; sweeps 0 -> 1 -> settles at pi* (damped)
    m = phase("mix")
    ps = ctx["pi_star"]
    if t < times["mix"][0]:
        pi = None
    elif m < 0.42:
        pi = float(M.ease_in_out_sine(m / 0.42))
    else:
        u = (m - 0.42) / 0.58
        pi = ps + (1 - ps) * math.exp(-5.0 * u) * math.cos(2.6 * math.pi * u)
        if u >= 1:
            pi = ps
    if t >= times["mix"][1]:
        pi = ps
    # ---- left panel pixels
    base = ctx["base"].copy()
    lay = base
    a_full, inside = b_heat_full(ctx)
    kk = P // N1024
    if f_full > 0:
        A_ = np.kron(a_full, np.ones((kk, kk)))[..., None] * f_full
        IN = np.kron(inside, np.ones((kk, kk), bool))[..., None]
        colr = np.where(IN, C["near"], C["far"])
        lay = lay * (1 - A_) + colr * A_
    if views > 0 and pi is not None or (views > 0 and t < times["mix"][0]):
        pw = 1.0 if pi is None else pi
        loc = np.zeros((N1024, N1024))
        loc[r0:r0 + w, c0:c0 + w] = ctx["loc"]
        dens = loc / loc.sum() * N1024 ** 2
        a_loc = heat_alpha(dens)
        a_loc[~np.pad(np.ones((w, w), bool), ((r0, N1024 - r0 - w), (c0, N1024 - c0 - w)))] = 0
        A_ = np.kron(a_loc, np.ones((kk, kk)))[..., None] * views * (0.25 + 0.75 * pw if pi is not None else 1.0)
        lay = lay * (1 - A_) + C["near"] * A_
    img.paste(to_img(lay), (X0, Y0))
    d = ImageDraw.Draw(img, "RGBA")
    # grid dots: stride follows the shrink (1 -> 2 -> 4); at r = 4 their area follows the far view's attention
    stride = 1 if r_now < 1.999 else (2 if r_now < 3.999 else 4)
    grid_a = float(np.clip((t - times["shrink"][0]) / 0.4, 0, 1)) * (1 - back)
    if grid_a > 0:
        gl = ctx["glob"] / ctx["glob"].sum()
        pw = 0.0 if pi is None else pi
        for b1 in range(N1024 // stride):
            for b2 in range(N1024 // stride):
                cy = Y0 + (stride * b1 + stride // 2 + 0.5) * s
                cx = X0 + (stride * b2 + stride // 2 + 0.5) * s
                if stride == 4 and views > 0:
                    rad = (1.6 + 26.0 * math.sqrt(gl[b1, b2]) * views) * SS
                    al = grid_a * (0.5 + 0.5 * views) * ((1 - pw) * 0.75 + 0.25 if pi is not None else 1.0)
                else:
                    rad = (0.9 if stride == 1 else 1.3 if stride == 2 else 1.7) * SS
                    al = grid_a * 0.85
                d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=M.rgb(M.PAL["far"]) + (int(255 * al),))
    # the window: from the whole grid (r = 1) to 32 (r = 2) to 16 tokens (r = 4), around the query's tile
    if t >= times["shrink"][0] and back < 1:
        def win_at(rr):
            ww = N1024 / rr
            tt = max(1.0, 8.0 * ww / 64)
            def st(p):
                return float(np.clip(tt * (p // tt) + tt / 2 - ww / 2, 0, N1024 - ww))
            return st(qr), st(qc), ww
        wr0, wc0, ww = win_at(r_now)
        al = 1 - back
        d.rectangle([X0 + wc0 * s, Y0 + wr0 * s, X0 + (wc0 + ww) * s, Y0 + (wr0 + ww) * s],
                    outline=M.rgb(M.PAL["near"]) + (int(255 * al),), width=round(2.5 * SS))
    qx, qy = X0 + (qc + 0.5) * s, Y0 + (qr + 0.5) * s
    rq = 4.5 * SS
    d.ellipse([qx - rq, qy - rq, qx + rq, qy + rq], fill=M.rgb(M.PAL["ink"]), outline=M.rgb(M.PAL["screen"]),
              width=round(1.2 * SS))
    # ---- captions over the panel
    kicker(d, "Reach emulation at 1024 × 1024 · block 12, head 10")
    if t < times["shrink"][0] + 0.3 or back > 0.5:
        cap, sub = "Full attention", "the reference at the native size"
    elif t < times["views"][0]:
        cap, sub = f"Both views shrink by r = {r_lab}", \
            f"window {64 // r_lab} tokens, grid of {64 // r_lab} × {64 // r_lab} keys"
    elif pi is None:
        cap, sub = "Two views at r = 4", "window 16 tokens, grid of 16 × 16 keys"
    else:
        cap, sub = "Mix: π · window + (1 − π) · grid", "fitted to full attention"
    label(d, (X0, Y0 - 52 * SS), cap, 30, 600, "ink")
    label(d, (X0, Y0 + P + 18 * SS), sub, 24, 400, "ink2")
    label(d, (X0, Y0 + P + 52 * SS), f"query: the face of the leftmost sheep · σ = {ctx['sigma']:.2f}",
          22, 400, "ink3")
    # ---- right: J(pi) on [0, 1]
    gx0, gy0, gx1, gy1 = 900 * SS, 250 * SS, 1490 * SS, 700 * SS
    A, B, E = ctx["A"], ctx["B"], ctx["E"]
    J = lambda p: E - 2 * p * B + p * p * A  # noqa: E731
    pp = np.linspace(0, 1, 201)
    jj = J(pp)
    lo, hi = jj.min(), jj.max()
    def gxy(p, j):
        return gx0 + (gx1 - gx0) * p, gy1 - (gy1 - gy0) * (j - lo) / (hi - lo) * 0.92
    plot_a = float(np.clip(t / 0.6, 0, 1)) if t < times["back"][0] else 1.0
    d.line([(gx0, gy1), (gx1, gy1)], fill=M.rgb(M.PAL["rule"]), width=round(1.5 * SS))
    d.line([(gx0, gy0 - 10 * SS), (gx0, gy1)], fill=M.rgb(M.PAL["rule"]), width=round(1.5 * SS))
    for tv in (0, 0.5, 1):
        x, _ = gxy(tv, lo)
        label(d, (x, gy1 + 14 * SS), f"{tv:g}", 24, 400, "ink3", anchor="ma")
    label(d, (gx1, gy1 + 50 * SS), "weight π of the window", 24, 400, "ink2", anchor="ra")
    label(d, (gx0, gy0 - 64 * SS), "Error against full attention", 28, 600, "ink")
    label(d, (gx0, gy0 - 30 * SS), "J(π) = E − 2πB + π²A", 24, 400, "ink2")
    pts = [gxy(p, j) for p, j in zip(pp, jj)]
    d.line(pts, fill=M.rgb(M.PAL["ink2"]) + (int(255 * plot_a),), width=round(2.5 * SS), joint="curve")
    if pi is not None:
        x, y = gxy(pi, J(pi))
        rr = 9 * SS
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=M.rgb(M.PAL["sca"]), outline=M.rgb(M.PAL["screen"]),
                  width=round(2 * SS))
        settled = t >= times["mix"][0] + 0.86 * T_MIX
        fa = float(np.clip((t - (times["mix"][0] + 0.86 * T_MIX)) / 0.4, 0, 1)) * (1 - back) if settled else 0.0
        M.text_alpha(img, (x, y - 24 * SS), f"π = {pi:.2f}", M.font(26 * SS, "sans", 600), M.PAL["sca"],
                     1 - fa, anchor="md")
        if settled:
            xs, ys = gxy(ps, J(ps))
            ly = ys - 150 * SS                     # well above the minimum, where the parabola is low
            for yy in np.arange(ly + 14 * SS, ys - 14 * SS, 9 * SS):
                d.line([(xs, yy), (xs, min(yy + 4 * SS, ys - 14 * SS))], fill=M.rgb(M.PAL["sca"]) + (int(220 * fa),),
                       width=round(1.5 * SS))
            M.text_alpha(img, (xs, ly), f"π* = clip(B/A, 0, 1) = {ps:.2f}", M.font(26 * SS, "sans", 600),
                         M.PAL["sca"], fa, anchor="md")
    return finish(img)


# ================================================================================================ (c) stages
C_PANEL = 600


def c_pick() -> str:
    """The first image of the paper hero's draw (its category order) that has a recorded progressive run."""
    import v6_hero
    picks = v6_hero.draw()
    for cat in v6_hero.ORDER:
        if (TRACE / picks[cat] / "trace.json").exists():
            return picks[cat]
    raise FileNotFoundError("no hero draw has a trace")


def c_context():
    sid = c_pick()
    tr = json.loads((TRACE / sid / "trace.json").read_text())
    assert tr["complete"] and not tr["synthetic"] and tr["steps_total"] == len(tr["steps"]) == 57
    steps = tr["steps"]
    pre = []
    for st in steps:
        im = Image.open(TRACE / sid / st["preview"]).convert("RGB")
        pre.append(im)
    pis = [None if st.get("pi") is None else np.array(st["pi"], dtype=np.float64) for st in steps]
    return dict(sid=sid, tr=tr, steps=steps, pre=pre, pis=pis)


def c_schedule():
    """Video time per step: [(t_start, step index)], hold and back."""
    per = [0.075] * 30 + [0.18] * 17 + [0.27] * 10
    ts, a = [], 0.3
    for i, p in enumerate(per):
        ts.append(a)
        a += p
    return ts, a, a + 1.2, a + 1.2 + 0.6


def c_step_at(t, ts):
    i = 0
    for k, a in enumerate(ts):
        if t >= a:
            i = k
    return i


def c_draw(ctx, k):
    ts, t_end, t_hold, total = c_schedule()
    return xfade_loop(c_core, ctx, k / FPS, t_hold, total)


def c_core(ctx, t):
    ts, t_end, t_hold, total = c_schedule()
    steps = ctx["steps"]
    back = float(M.smoothstep((t - t_hold) / (total - t_hold - 1 / FPS))) if t > t_hold else 0.0
    i = c_step_at(t, ts) if t < t_end else 56
    if back > 0:
        i_show = 56
    else:
        i_show = i
    st = steps[i_show]
    stage = st["stage"]
    img = Image.new("RGB", (W * SS, H * SS), M.rgb(M.PAL["screen"]))
    d = ImageDraw.Draw(img, "RGBA")
    kicker(d, "Progressive SCA: three sampler stages read one reach table")
    # ---- image at its stage size, drawn to scale (1024^2 : 2048^2 : 4096^2 = 1 : 2 : 4 in side)
    P = C_PANEL * SS
    X0, Y0 = 110 * SS, 160 * SS
    side_of = {0: P / 4, 1: P / 2, 2: P}
    # grow between stages over 0.35 s
    t_stage = {0: ts[0], 1: ts[30], 2: ts[47]}
    cur = side_of[stage]
    if stage > 0:
        g = float(M.smoothstep((t - t_stage[stage]) / 0.35)) if back == 0 else 1.0
        cur = side_of[stage - 1] + (side_of[stage] - side_of[stage - 1]) * g
    # the outline of the 4096^2 canvas, and the stage sizes as faint frames
    for sv in (P / 4, P / 2, P):
        d.rectangle([X0, Y0 + P - sv, X0 + sv, Y0 + P], outline=M.rgb(M.PAL["rule"]), width=round(1.2 * SS))
    im = ctx["pre"][i_show].resize((round(cur), round(cur)), Image.LANCZOS)
    if back > 0:
        first = ctx["pre"][0].resize((round(side_of[0]), round(side_of[0])), Image.LANCZOS)
        blank = Image.new("RGB", im.size, M.rgb(M.PAL["screen"]))
        im = Image.blend(im, blank, back)
        img.paste(im, (X0, round(Y0 + P - cur)))
        if back > 0.5:
            f2 = Image.blend(Image.new("RGB", first.size, M.rgb(M.PAL["screen"])), first, (back - 0.5) * 2)
            img.paste(f2, (X0, round(Y0 + P - side_of[0])))
    else:
        img.paste(im, (X0, round(Y0 + P - cur)))
    res = st["stage_res"].replace("x", " × ")
    label(d, (X0, Y0 - 44 * SS), res, 30, 600, "ink")
    label(d, (X0 + P, Y0 - 42 * SS), f"σ = {st['sigma']:.2f}", 26, 400, "ink2", anchor="ra")
    # ---- right: pi of all heads at this step (57 blocks x 24 heads)
    gx0, gy0 = 820 * SS, 210 * SS
    cw = 11.6 * SS
    ch = 12.0 * SS
    pi = ctx["pis"][i_show]
    head = "Model's own attention: no table" if stage == 0 else \
        f"Reach table at r = {2 if stage == 1 else 4}"
    label(d, (gx0, gy0 - 92 * SS), head, 30, 600, "ink" if stage == 0 else "sca")
    label(d, (gx0, gy0 - 54 * SS), "weight π of every head this step reads (57 blocks × 24 heads)", 24,
          400, "ink2")
    gw, gh = 57 * cw, 24 * ch
    if pi is None:
        d.rectangle([gx0, gy0, gx0 + gw, gy0 + gh], fill=M.rgb(M.PAL["raised"]))
        label(d, (gx0 + gw / 2, gy0 + gh / 2), "full attention at 1024 × 1024", 26, 400, "ink3", anchor="mm")
    else:
        cols = pi_color(pi.T)                       # 24 x 57
        cell = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
        tile = Image.fromarray(cell).resize((round(gw), round(gh)), Image.NEAREST)
        img.paste(tile, (round(gx0), round(gy0)))
        # thin separators
        for b in range(1, 57):
            x = gx0 + b * cw
            d.line([(x, gy0), (x, gy0 + gh)], fill=M.rgb(M.PAL["screen"]) + (150,), width=1)
        for hh in range(1, 24):
            y = gy0 + hh * ch
            d.line([(gx0, y), (gx0 + gw, y)], fill=M.rgb(M.PAL["screen"]) + (150,), width=1)
    label(d, (gx0, gy0 + gh + 12 * SS), "block 1", 22, 400, "ink3")
    label(d, (gx0 + gw, gy0 + gh + 12 * SS), "block 57", 22, 400, "ink3", anchor="ra")
    # colour key
    ky = gy0 + gh + 64 * SS
    kw = 300 * SS
    ramp = pi_color(np.linspace(0, 1, 256))[None, :, :].repeat(10, 0)
    img.paste(to_img(ramp).resize((round(kw), round(12 * SS)), Image.BILINEAR), (round(gx0), round(ky)))
    label(d, (gx0, ky + 22 * SS), "grid (π = 0)", 22, 400, "far")
    label(d, (gx0 + kw, ky + 22 * SS), "window (π = 1)", 22, 400, "near", anchor="ra")
    # ---- stage rail
    rx0, rx1, ry = 820 * SS, 1490 * SS, 690 * SS
    counts = [30, 17, 10]
    names = ["1024² · own attention", "2048² · r = 2", "4096² · r = 4"]
    gap = 10 * SS
    tot = sum(counts)
    span = rx1 - rx0 - 2 * gap
    x = rx0
    for si, (cnt, nm) in enumerate(zip(counts, names)):
        sw = span * cnt / tot
        on = si < stage or (si == stage)
        col = M.PAL["ink3"] if si == 0 else M.PAL["sca"]
        d.rounded_rectangle([x, ry, x + sw, ry + 10 * SS], radius=5 * SS, fill=M.rgb(M.PAL["rule"]))
        first = sum(counts[:si])
        if i_show >= first:
            frac = min(1.0, (i_show - first + 1) / cnt)
            d.rounded_rectangle([x, ry, x + max(10 * SS, sw * frac), ry + 10 * SS], radius=5 * SS,
                                fill=M.rgb(col) + (int(255 * (1 - back)),))
        label(d, (x, ry + 26 * SS), nm, 22, 600 if si == stage else 400, "ink" if si == stage else "ink3")
        label(d, (x, ry + 56 * SS), f"{cnt} steps", 22, 400, "ink3")
        x += sw + gap
    label(d, (rx0, ry - 40 * SS), f"step {i_show + 1} of 57", 24, 600, "ink2")
    return finish(img)


CONTEXTS = {"method_window": a_context, "method_emulation": b_context, "method_stages": c_context}
DRAW = {"method_window": a_draw, "method_emulation": b_draw, "method_stages": c_draw}


def build(which: str, procs: int):
    t = reach_table()
    if which in ("window", "all"):
        n = round(A_SECONDS * FPS)
        fdir = render_all("method_window", a_draw, n, procs)
        p4, p7 = float(t["s4"][BLOCK, NEAR_HEAD, BIN]), float(t["s4"][BLOCK, FAR_HEAD, BIN])
        encode("method_window", fdir, n, 0,
               alt=("Two panels of the same 4096 by 4096 image with a moving query point: for head 4 the orange "
                    "window around the query dominates; for head 7 only the blue grid of keys over the whole image "
                    "carries weight."),
               caption=(f"Each head mixes a native-size window (orange, 64 x 64 tokens around the query's tile) and "
                        f"a native-size grid (blue, 64 x 64 keys over the whole image) by its reach pi. Block 12 at "
                        f"4096 x 4096: head 4 pi = {p4:.2f}, head 7 pi = {p7:.2f} (reach table at r = 4, "
                        f"sigma = {float(t['sigma'][BIN]):.2f})."),
               sources=dict(image=str(TARGET), reach_table=M.rel_repo(CALIB),
                            cells={"head 4": dict(block=12, head=4, r=4, sigma_bin=float(t["sigma"][BIN]), pi=p4),
                                   "head 7": dict(block=12, head=7, r=4, sigma_bin=float(t["sigma"][BIN]), pi=p7)},
                            geometry="Section 3.1: tile 8, window 64 clamped to the grid, block centres at stride 4"),
               role="method_window", order=30)
    if which in ("emulation", "all"):
        _, total = b_times()
        n = round(total * FPS)
        fdir = render_all("method_emulation", b_draw, n, procs)
        prov = json.loads(MASSETS.read_text())["j_curves"]["Mid"]
        encode("method_emulation", fdir, n, round((T_FULL + T_SHRINK + T_VIEWS + T_MIX + 0.5) * FPS),
               alt=("Reach emulation at 1024 by 1024: a head's full attention at one query, its window and grid "
                    "shrinking by a factor of four, and the weight between them sliding along an error parabola "
                    "until it settles at its minimum."),
               caption=(f"Reach emulation: at 1024 x 1024 both views shrink by r (window 16 tokens, grid stride 4 "
                        f"for r = 4) and the weight that best matches full attention has a closed form, "
                        f"pi* = clip(B/A, 0, 1) = {float(np.clip(prov['B'] / prov['A'], 0, 1)):.2f} for block 12, "
                        f"head 10. Real attention of one query, real sums over 32 prompts."),
               sources=dict(attention=M.rel_repo(ATTN), j_sums=M.rel_repo(MASSETS), reach_table=M.rel_repo(CALIB),
                            image=M.rel_repo(IMG1024), head=dict(block=12, head=10, r=4, step=20,
                                                                  query_token=[43, 11], **prov)),
               role="method_emulation", order=31)
    if which in ("stages", "all"):
        ts, t_end, t_hold, total = c_schedule()
        n = round(total * FPS)
        fdir = render_all("method_stages", c_draw, n, procs)
        sid = c_pick()
        encode("method_stages", fdir, n, round((t_end + 0.6) * FPS),
               alt=("A progressive SCA run: the image estimate grows from 1024 to 2048 to 4096 pixels, and a grid "
                    "of 1,368 head weights lights up in orange and blue once the 2048 and 4096 stages read the "
                    "reach table."),
               caption=("Progressive SCA keeps HRDiT's three stages: 30 steps at 1024 x 1024 with the model's own "
                        "attention, then 17 at 2048 x 2048 reading the reach table at r = 2 and 10 at 4096 x 4096 "
                        "at r = 4. Recorded run of the first paper-hero prompt with a trace."),
               sources=dict(trace=str(TRACE / sid), prompt_id=sid,
                            rule="first draw of the paper hero's rule (v6_hero.draw, category order) with a trace"),
               role="method_stages", order=32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["window", "emulation", "stages", "all"])
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--frames", type=str, default=None, help="comma list of times (s): render look frames only")
    a = ap.parse_args()
    if a.frames:
        names = {"window": ["method_window"], "emulation": ["method_emulation"], "stages": ["method_stages"],
                 "all": list(DRAW)}[a.which]
        for nm in names:
            ctx = CONTEXTS[nm]()
            outs = []
            for tv in a.frames.split(","):
                k = round(float(tv) * FPS)
                p = BUILD / "look" / f"{nm}_{float(tv):.2f}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                DRAW[nm](ctx, k).save(p)
                outs.append(p)
            M.frames_sheet(outs, BUILD / "look" / f"{nm}_look.png", cols=2, tile_w=800, label=nm)
            print(BUILD / "look" / f"{nm}_look.png")
        return
    build(a.which, a.procs)


if __name__ == "__main__":
    main()
