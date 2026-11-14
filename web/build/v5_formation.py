"""Formation videos of the v5 page (SCA's "each image beside the run that made it"): formation_<id>.mp4, plus the
patch data for the interactive inspector (web/assets/data/formation_media/).

Video (1600 x 900): left, the final 4096^2 progressive SCA image (small, constant); centre, the image as it forms (the
recorded run's decoded estimate of every one of its 57 steps); right, three zoom-in patches of the forming image,
ending on the final image's native pixels. Three beats with one caption each:
  1. 1024^2 stage (30 steps, the model's own attention): no overlay; the layout forms.
  2. 2048^2 stage (17 steps) and 3. 4096^2 stage (10 steps): a focus moves patch to patch. At the focused patch: the
     query tile (8 x 8 tokens at the patch centre), the native-size window around it (64 x 64 tokens, clamped to the
     grid: half the image side at 2048^2, a quarter at 4096^2; orange) and the native-size grid of block centres over
     the whole image (64 x 64 keys at stride 2 or 4; blue dots), plus two chips with the weights the run read for the
     paper's two example heads (block 12, heads 4 and 7) at that step (the trace's per-step pi: the reach table at
     r = 2 or r = 4). Positions exact; weights from the reach table; no attention map is drawn (none exists at 4096^2).
Patches (rule): three 512 x 512 windows of the final image of highest mean edge energy (v6_common.detail_map, 1/16
border margin), each at least one window from the others (a 512 px gap in x or y), ordered top to bottom. During the
run a patch shows its region of the step's 1024 px preview (128 x 128 preview pixels, upscaled); the hold zooms each
patch into the final PNG's own pixels (1:1).
Prompts (rule): as v5_process.py, the traced showcase prompts, one per category by the paper hero's seeded draw,
first five of the hero category order.

Data for the inspector (web/assets/data/formation_media/<id>/): formation.json (patch boxes, per-step stage, size, sigma,
the two example heads' weights, trace file locations and sizes), patch<k>_steps.webp (a 57-tile horizontal sprite,
128 x 128 per step: the patch region of each step's preview, native preview pixels) and patch<k>_final.webp (the
512 x 512 window of the final PNG, 1:1); index.json lists the prompts.

Run from the repository root:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_formation.py data
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_formation.py video [--only <id>] [--look t,t]
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
import v6_common as vc  # noqa: E402

W, H = 1600, 900
FPS = 30
PATCH = 512
TRACE = VP.TRACE / "sca_v2_prog"
DATA = M.WEB / "assets" / "data" / "formation_media"   # (formation/ belongs to the interactive owner)
BUILD = M.BUILD / "formation"
BLOCK, HEAD_NEAR, HEAD_FAR = 11, 3, 6          # block 12, heads 4 and 7 (zero-based), the paper's example heads
MAX_BYTES = 4_000_000


# ------------------------------------------------------------------------------------------------ patches
def patches(final: Image.Image, k: int = 3, w: int = PATCH, f: int = 8) -> list:
    """k windows of highest mean edge energy, each at least one window from the others; top to bottom."""
    a = vc.detail_map(final, f)
    kw = w // f
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[kw:, kw:] - ii[:-kw, kw:] - ii[kw:, :-kw] + ii[:-kw, :-kw]
    m = max(1, round(a.shape[0] / 16))
    s[:m, :] = s[-m:, :] = -np.inf
    s[:, :m] = s[:, -m:] = -np.inf
    out = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(s), s.shape)
        out.append([int(x * f), int(y * f), w, w])
        s[max(0, y - 2 * kw + 1): y + 2 * kw, max(0, x - 2 * kw + 1): x + 2 * kw] = -np.inf   # one window apart
    return sorted(out, key=lambda b: (b[1], b[0]))


def aligned_patches(final: Image.Image, t: dict) -> list:
    """The inspector's presets (web/build/v5_build_formation.py presets: the trace's own window, then two windows of
    highest edge energy at least 1.4 spans from each other and from it), as 512 px boxes, so the video and the
    inspector show the same three patches; ordered top to bottom."""
    import v5_build_formation as BF
    box = [int(v) for v in t["box_xys"]]
    out = []
    for cx, cy in BF.presets(final, box):
        x = int(np.clip(cx - PATCH // 2, 0, 4096 - PATCH))
        y = int(np.clip(cy - PATCH // 2, 0, 4096 - PATCH))
        out.append([x, y, PATCH, PATCH])
    return sorted(out, key=lambda b: (b[1], b[0]))


def load(sid: str) -> dict:
    t = VP.load_trace("sca_v2_prog", sid)
    d = TRACE / sid
    final = Image.open(d / "final.png").convert("RGB")
    assert final.size == (4096, 4096)
    return dict(sid=sid, t=t, d=d, final=final, boxes=aligned_patches(final, t))


# ------------------------------------------------------------------------------------------------ data export
def export(sid: str) -> dict:
    L = load(sid)
    t, d = L["t"], L["d"]
    out = DATA / sid
    out.mkdir(parents=True, exist_ok=True)
    steps = []
    for st in t["steps"]:
        pi = st.get("pi")
        steps.append(dict(step=st["step"], stage=st["stage"], size=st["stage_res"], sigma=round(st["sigma"], 4),
                          preview=st["preview"], crop=st["crop"],
                          head4_window=None if pi is None else round(pi[BLOCK][HEAD_NEAR], 4),
                          head7_window=None if pi is None else round(pi[BLOCK][HEAD_FAR], 4)))
    prev = [Image.open(d / st["preview"]).convert("RGB") for st in t["steps"]]
    pw = prev[0].width
    sc = pw / 4096
    files = []
    for k, (x, y, w, h) in enumerate(L["boxes"]):
        tw = round(w * sc)
        sprite = Image.new("RGB", (tw * len(prev), tw))
        for i, im in enumerate(prev):
            sprite.paste(im.crop((round(x * sc), round(y * sc), round(x * sc) + tw, round(y * sc) + tw)), (i * tw, 0))
        sp = out / f"patch{k + 1}_steps.webp"
        sprite.save(sp, "WEBP", quality=88, method=6)
        fp = out / f"patch{k + 1}_final.webp"
        L["final"].crop((x, y, x + w, y + h)).save(fp, "WEBP", quality=92, method=6)
        files.append(dict(box=[x, y, w, h], steps_sprite=sp.name, sprite_tile=[tw, tw], sprite_tiles=len(prev),
                          final=fp.name, final_size=[w, h]))
    rec = dict(
        id=sid, title=VP.showcase()[sid]["title"], prompt=t["prompt"], seed=t["seed"], size=[4096, 4096],
        patch_rule=("the inspector's presets (v5_build_formation.presets): the trace's own 512 px window, then two "
                    "windows of highest edge energy at least 1.4 spans from each other and from it; top to bottom"),
        patches=files, steps=steps, stages=t["progressive"]["stages"],
        stage_steps=[t["progressive"]["base_steps"]] + list(t["progressive"]["highres_steps"]),
        heads=dict(block=12, near_head=4, far_head=7, note="weight on the window, read from the reach table at the "
                   "step's ratio (r = 2 at 2048^2, r = 4 at 4096^2); null at 1024^2 (the model's own attention)"),
        trace=dict(dir=str(d), preview_px=t["preview_px"], preview_note="decoded estimate x0 of the whole image at each "
                   "step, resized to 1024 px (all stages)", crop_box=t["box_xys"], crop_px=t["crop_px"],
                   crop_note="native-resolution window crop_box at each step, upsampled 4x / 2x at 1024^2 / 2048^2",
                   final="final.png (4096 x 4096, identical to the campaign PNG)"),
        geometry=dict(tile_tokens=8, window_tokens=64, grid_keys=64, token_px_at_stage=16,
                      note="window: 64 x 64 tokens around the query's 8 x 8 tile, clamped; grid: block centres at "
                           "stride n/64 (2 at 2048^2, 4 at 4096^2)"))
    (out / "formation.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


# ------------------------------------------------------------------------------------------------ video
T1, T2, T3, T4, TB = 3.6, 6.3, 6.3, 3.4, 0.6   # beats: 1024^2, 2048^2, 4096^2, hold, back
CAPS = ["The layout forms at the size the model was trained on.",
        "Near-looking heads read a native-size window around each patch.",
        "Far-looking heads read a coarse grid over the whole image, so the layout holds.",
        "Each patch ends at the final image's native pixels."]
# layouts: centre image (CX, CY, CS), final image (LX, LY, LS), patches (side RS at patch_xy(k)), connectors to the
# patches' left edges ("right") or top edges ("down"), caption centre and width, legend corner, sizes
LAYOUTS = {
    "16x9": dict(W=1600, H=900, CX=435, CY=56, CS=720, LX=115, LY=56, LS=280, RS=200, conn="right",
                 patch_xy=[(1285, 86), (1285, 316), (1285, 546)], cap_xy=(795, 852), cap_w=1300, cap=30,
                 legend_xy=(115, 456), legend=20, flabel_xy=(115, 366), flabel=22, chip=20, chip_w=270, stage=22,
                 one=18),
    "9x16": dict(W=900, H=1600, CX=40, CY=96, CS=820, LX=40, LY=1262, LS=160, RS=250, conn="down",
                 patch_xy=[(40, 966), (325, 966), (610, 966)], cap_xy=(450, 1522), cap_w=820, cap=30,
                 legend_xy=(232, 1280), legend=24, flabel_xy=(40, 1452), flabel=22, chip=24, chip_w=330, stage=26,
                 one=22),
}


def schedule(t: float) -> dict:
    """Beat, step index, blend and focus at video time t (before the loop-back)."""
    if t < T1:
        f = t / T1 * 30
        return dict(beat=0, i=min(29, int(f)), a=float(M.smoothstep((f - int(f)) / 0.5)), focus=None)
    if t < T1 + T2 + T3:
        beat = 1 if t < T1 + T2 else 2
        t0 = T1 if beat == 1 else T1 + T2
        dur = T2 if beat == 1 else T3
        n0, n = (30, 17) if beat == 1 else (47, 10)
        u = (t - t0) / dur
        f = u * n
        i = n0 + min(n - 1, int(f))
        a = float(M.smoothstep((f - int(f)) / 0.5))
        fp = u * 3                                  # focus position 0..3 over three patches
        k = min(2, int(fp))
        move = float(M.smoothstep((fp - k) / 0.18)) if k > 0 else 1.0
        return dict(beat=beat, i=i, a=a, focus=k, move=move)
    return dict(beat=3, i=56, a=1.0, focus=None, hold=(t - T1 - T2 - T3) / T4)


def frame_preview(ctx, i, a):
    pv = ctx["pv"]
    if i == 0 or a >= 1:
        return pv[i]
    return Image.blend(pv[i - 1], pv[i], a)


def window_geom(box, stage):
    """(tile, window, grid stride, n) in tokens of the stage grid for a query at the patch centre."""
    n = {1: 128, 2: 256}[stage]
    x, y, w, h = box
    qc, qr = (x + w / 2) / 4096 * n, (y + h / 2) / 4096 * n
    tr, tc = 8 * int(qr // 8), 8 * int(qc // 8)

    def start(p):
        return int(np.clip(8 * int(p // 8) + 4 - 32, 0, n - 64))
    return (tr, tc), (start(qr), start(qc)), n // 64, n


def caption(img, L, txt, al):
    f = M.font(L["cap"], "sans", 500)
    x, y = L["cap_xy"]
    if f.getlength(txt) <= L["cap_w"]:
        M.text_alpha(img, (x, y), txt, f, M.PAL["ink"], al, anchor="ms")
        return
    words = txt.split(" ")
    best = min(((max(f.getlength(" ".join(words[:i])), f.getlength(" ".join(words[i:]))), i)
                for i in range(1, len(words))))
    i = best[1]
    M.text_alpha(img, (x, y - 19), " ".join(words[:i]), f, M.PAL["ink"], al, anchor="ms")
    M.text_alpha(img, (x, y + 19), " ".join(words[i:]), f, M.PAL["ink"], al, anchor="ms")


def draw(ctx, t: float) -> Image.Image:
    L = ctx["L"]
    W_, H_, CX, CY, CS, LX, LY, LS, RS = (L[k] for k in ("W", "H", "CX", "CY", "CS", "LX", "LY", "LS", "RS"))
    total = T1 + T2 + T3 + T4
    if t >= total:
        a = float(M.smoothstep((t - total) / (TB - 1 / FPS)))
        return Image.blend(draw(ctx, total - 1 / FPS), draw(ctx, 0.0), a)
    sch = schedule(t)
    img = Image.new("RGB", (W_, H_), M.rgb(M.PAL["screen"]))
    S = 2
    big = Image.new("RGBA", (W_ * S, H_ * S), (0, 0, 0, 0))   # vector overlay at 2x
    d = ImageDraw.Draw(big, "RGBA")
    # the final image, small
    img.paste(ctx["final_small"], (LX, LY))
    dd = ImageDraw.Draw(img)
    fx, fy = L["flabel_xy"]
    if L["conn"] == "right":
        dd.text((fx, fy), "final image", font=M.font(L["flabel"], "sans", 600), fill=M.rgb(M.PAL["ink2"]),
                anchor="ls")
        dd.text((fx, fy + 28), "4096 × 4096", font=M.font(L["flabel"], "sans", 400), fill=M.rgb(M.PAL["ink3"]),
                anchor="ls")
    else:
        dd.text((fx, fy), "final image · 4096 × 4096", font=M.font(L["flabel"], "sans", 600),
                fill=M.rgb(M.PAL["ink2"]), anchor="ls")
    # the forming image
    pv = frame_preview(ctx, sch["i"], sch["a"])
    img.paste(pv.resize((CS, CS), Image.LANCZOS), (CX, CY))
    st = ctx["t"]["steps"][sch["i"]]
    stage = st["stage"]
    sc = CS / 4096
    boxes = ctx["boxes"]
    hold = sch.get("hold", 0.0)
    ov_a = 1.0 if sch["beat"] < 3 else 1 - float(M.smoothstep(hold / 0.12))
    # patch frames and connectors (always), the focused one emphasised
    for k, (x, y, w, h) in enumerate(boxes):
        foc = sch["focus"] == k
        bx0, by0 = (CX + x * sc) * S, (CY + y * sc) * S
        bx1, by1 = (CX + (x + w) * sc) * S, (CY + (y + h) * sc) * S
        al = 1.0 if (foc or sch["focus"] is None) else 0.55
        d.rectangle([bx0, by0, bx1, by1], outline=M.rgb(M.PAL["near"]) + (int(255 * al),), width=round(1.6 * S))
        px, py = L["patch_xy"][k]
        if L["conn"] == "right":
            a_, b_ = (bx1, (by0 + by1) / 2), (px * S, (py + RS / 2) * S)
        else:
            a_, b_ = ((bx0 + bx1) / 2, by1), ((px + RS / 2) * S, py * S)
        d.line([a_, b_], fill=M.rgb(M.PAL["near"]) + (int(255 * (0.9 if foc else (0.4 if sch["focus"] is None
                                                                                   else 0.18))),),
               width=round((1.6 if foc else 1.0) * S))
    # focus overlay: grid (blue dots), window (orange), tile (white), head chips
    chips = None
    if sch["focus"] is not None and stage > 0 and ov_a > 0:
        k = sch["focus"]
        (tr, tc), (wr, wc), stride, n = window_geom(boxes[k], stage)
        if k > 0 and sch["move"] < 1:                        # slide from the previous patch's window
            (ptr, ptc), (pwr, pwc), _, _ = window_geom(boxes[k - 1], stage)
            m = sch["move"]
            tr, tc = ptr + (tr - ptr) * m, ptc + (tc - ptc) * m
            wr, wc = pwr + (wr - pwr) * m, pwc + (wc - pwc) * m
        tok = CS / n
        dot_a = 0.62 if stage == 2 else 0.42
        rad = 1.35 * S * CS / 720
        col = M.rgb(M.PAL["far"]) + (int(255 * dot_a * ov_a),)
        for b1 in range(64):
            yy = (CY + (stride * b1 + stride // 2 + 0.5) * tok) * S
            for b2 in range(64):
                xx = (CX + (stride * b2 + stride // 2 + 0.5) * tok) * S
                d.ellipse([xx - rad, yy - rad, xx + rad, yy + rad], fill=col)
        wx0, wy0 = (CX + wc * tok) * S, (CY + wr * tok) * S
        wx1, wy1 = wx0 + 64 * tok * S, wy0 + 64 * tok * S
        d.rectangle([wx0, wy0, wx1, wy1], fill=M.rgb(M.PAL["near"]) + (int(40 * ov_a),),
                    outline=M.rgb(M.PAL["near"]) + (int(255 * ov_a),), width=round(2.6 * S))
        tx0, ty0 = (CX + tc * tok) * S, (CY + tr * tok) * S
        d.rectangle([tx0, ty0, tx0 + 8 * tok * S, ty0 + 8 * tok * S], fill=M.rgb(M.PAL["ink"]) + (90,),
                    outline=M.rgb(M.PAL["ink"]) + (255,), width=round(1.6 * S))
        chips = (wx0 / S, wy0 / S, wx1 / S, wy1 / S, st, ov_a)
    img = img.convert("RGBA")
    img.alpha_composite(big.resize((W_, H_), Image.LANCZOS))
    img = img.convert("RGB")
    # chips beside the window: the weights the run read for the two example heads at this step
    if chips:
        wx0, wy0, wx1, wy1, st_, al = chips
        pi = st_["pi"]
        p4, p7 = pi[BLOCK][HEAD_NEAR], pi[BLOCK][HEAD_FAR]
        labs = [(f"head 4: {round(100 * p4)}% window", M.PAL["near"]),
                (f"head 7: {round(100 * (1 - p7))}% grid", M.PAL["far"])]
        ch = L["chip"]
        cx = max(CX + 8, min(wx0, CX + CS - L["chip_w"]))
        need = 2 * (ch * 1.7) + 16
        below = wy1 + need < CY + CS
        cy = wy1 + 10 if below else max(CY + 8, wy0 - need)
        for j, (lab, col) in enumerate(labs):
            M.chip(img, (cx, cy + j * ch * 1.7), lab, ch, anchor="lt", bg=(8, 10, 9, int(215 * al)),
                   fg=M.rgb(M.PAL["ink"]) + (int(255 * al),), weight=600, radius=6, dot=M.rgb(col) + (255,))
    # legend while the focus overlay is on
    if sch["focus"] is not None and stage > 0:
        la = ov_a * float(M.smoothstep((t - T1) / 0.4))
        gx, ly = L["legend_xy"]
        fs = L["legend"]
        step_y = round(fs * 2.3)
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dl = ImageDraw.Draw(lay)
        a8 = int(255 * la)
        dl.rectangle([gx, ly - 13, gx + 26, ly + 13], fill=M.rgb(M.PAL["near"]) + (int(40 * la),),
                     outline=M.rgb(M.PAL["near"]) + (a8,), width=2)
        for yy in (ly + step_y - 4, ly + step_y + 4):
            for xx in (gx + 5, gx + 13, gx + 21):
                dl.ellipse([xx - 2, yy - 2, xx + 2, yy + 2], fill=M.rgb(M.PAL["far"]) + (a8,))
        dl.rectangle([gx + 7, ly + 2 * step_y - 6, gx + 19, ly + 2 * step_y + 6], fill=M.rgb(M.PAL["ink"]) +
                     (int(90 * la),), outline=M.rgb(M.PAL["ink"]) + (a8,), width=2)
        for j, (l1, l2) in enumerate((("native-size window", "around the patch"),
                                      ("native-size grid", "over the whole image"),
                                      ("query tile", "at the patch centre"))):
            yy = ly + j * step_y + 6
            dl.text((gx + 40, yy - 2), l1, font=M.font(fs, "sans", 600), fill=M.rgb(M.PAL["ink"]) + (a8,),
                    anchor="ls")
            dl.text((gx + 40, yy + fs), l2, font=M.font(fs, "sans", 400), fill=M.rgb(M.PAL["ink2"]) + (a8,),
                    anchor="ls")
        img = img.convert("RGBA")
        img.alpha_composite(lay)
        img = img.convert("RGB")
    # stage chip
    M.chip(img, (CX + 12, CY + 12), {0: "1024 × 1024", 1: "2048 × 2048", 2: "4096 × 4096"}[stage], L["stage"],
           anchor="lt", bg=(8, 10, 9, 200), fg=M.rgb(M.PAL["ink"]) + (255,), weight=600, radius=6)
    # the zoom-in patches
    pscale = ctx["pv"][sch["i"]].width / 4096
    zoom = float(M.smoothstep((hold - 0.15) / 0.45)) if sch["beat"] == 3 else 0.0
    d2 = ImageDraw.Draw(img, "RGBA")
    for k, (x, y, w, h) in enumerate(boxes):
        px, py = L["patch_xy"][k]
        if zoom <= 0:
            reg = frame_preview(ctx, sch["i"], sch["a"]).crop((x * pscale, y * pscale, (x + w) * pscale,
                                                                (y + h) * pscale))
            tile = reg.resize((RS, RS), Image.LANCZOS)
        else:
            # zoom into the final PNG: from the 512 window to its central RS x RS at 1:1
            span = w + (RS - w) * zoom
            cxp, cyp = x + w / 2, y + h / 2
            tile = ctx["final"].resize((RS, RS), Image.LANCZOS, box=(cxp - span / 2, cyp - span / 2,
                                                                     cxp + span / 2, cyp + span / 2))
            if zoom < 0.25:
                reg = frame_preview(ctx, 56, 1.0).crop((x * pscale, y * pscale, (x + w) * pscale, (y + h) * pscale))
                tile = Image.blend(reg.resize((RS, RS), Image.LANCZOS), tile, zoom / 0.25)
        img.paste(tile, (px, py))
        foc = sch["focus"] == k
        d2.rectangle([px - 1, py - 1, px + RS, py + RS], outline=M.rgb(M.PAL["near"]) + (255 if foc or
                     sch["focus"] is None else 150,), width=3 if foc else 2)
        if zoom >= 1:
            M.chip(img, (px + 8, py + 8), "1:1", L["one"], anchor="lt", bg=(8, 10, 9, 200), fg=(236, 238, 233, 255),
                   weight=600, radius=5)
    # caption
    cap = CAPS[sch["beat"]]
    t_beat0 = [0, T1, T1 + T2, T1 + T2 + T3][sch["beat"]]
    t_beat1 = [T1, T1 + T2, T1 + T2 + T3, T1 + T2 + T3 + T4][sch["beat"]]
    al = 1.0 if (sch["beat"] == 0 and t < 0.3) else min(1.0, (t - t_beat0) / 0.3)
    al = min(al, (t_beat1 - t) / 0.3) if sch["beat"] < 3 else al
    caption(img, L, cap, max(0.0, min(1.0, al)))
    return img


def context(sid: str, cut: str = "16x9") -> dict:
    Ld = load(sid)
    t, d = Ld["t"], Ld["d"]
    L = LAYOUTS[cut]
    pv = [Image.open(d / st["preview"]).convert("RGB") for st in t["steps"]]
    return dict(sid=sid, t=t, pv=pv, final=Ld["final"], boxes=Ld["boxes"], L=L, cut=cut,
                final_small=Ld["final"].reduce(4).resize((L["LS"], L["LS"]), Image.LANCZOS))


_CTX: dict = {}


def _work(task):
    sid, cut, idxs, outdir = task
    key = (sid, cut)
    if key not in _CTX:
        _CTX.clear()
        _CTX[key] = context(sid, cut)
    for k in idxs:
        draw(_CTX[key], k / FPS).save(Path(outdir) / f"{k:05d}.png", compress_level=1)
    return len(idxs)


def build(sid: str, order: int, procs: int, cut: str = "16x9"):
    L = LAYOUTS[cut]
    total = T1 + T2 + T3 + T4 + TB
    n = round(total * FPS)
    name = f"formation_{sid.split('_', 2)[-1]}" + ("" if cut == "16x9" else "_9x16")
    fdir = BUILD / name / f"frames_n{n}"
    fdir.mkdir(parents=True, exist_ok=True)
    chunk = max(6, math.ceil(n / procs))
    tasks = [(sid, cut, list(range(i, min(n, i + chunk))), str(fdir)) for i in range(0, n, chunk)]
    t0 = time.time()
    with Pool(min(procs, len(tasks))) as pool:
        sum(pool.imap_unordered(_work, tasks))
    print(f"{name}: {n} frames in {time.time() - t0:.1f} s", flush=True)
    mp4 = M.OUT / f"{name}.mp4"
    enc = M.encode_to_size(fdir / "%05d.png", mp4, FPS, n, MAX_BYTES, crf_floor=20, preset="veryslow", gop=n,
                           passlog=BUILD / name / "x264pass", x264="aq-mode=3", threads=16)
    p = M.check_video(mp4, max_bytes=MAX_BYTES, w=L["W"], h=L["H"])
    t_rest = T1 + T2 + T3 + T4 * 0.8
    poster = M.poster_webp(Image.open(fdir / f"{round(t_rest * FPS):05d}.png"), M.OUT / f"{name}.webp", quality=84)
    Ld = load(sid)
    title = VP.showcase()[sid]["title"]
    side = "on the right" if cut == "16x9" else "below"
    alt = (f"{title}: a progressive SCA image forming over 57 steps at 1024, 2048 and 4096 pixels. A focus moves "
           f"across three detailed patches, showing the native-size window and the coarse grid each head reads, "
           f"while zoomed patches {side} sharpen and end at native pixels.")
    cap = (f"{title}: how a 4096 x 4096 image forms with progressive SCA. Orange: the native-size window around a "
           f"patch; blue: the native-size grid over the whole image; chips: how much two heads use each "
           f"(positions exact; weights from the reach table). Patches: the run's recorded window and two of highest "
           f"edge energy.")
    e = M.entry_for(name, mp4, poster, loop=True, alt=alt, caption=cap,
                    sources=dict(trace=str(TRACE / sid), final_png=Ld["t"]["campaign"]["png"], prompt_id=sid,
                                 patches=Ld["boxes"], data=f"web/assets/data/formation_media/{sid}/formation.json",
                                 rule="traced showcase prompts, one per category by seeded draw (as process videos)",
                                 patch_rule=("the trace's own window and two of highest edge energy, 1.4 spans apart "
                                             "(inspector presets)")),
                    script="v5_formation.py", order=50 + order + (0 if cut == "16x9" else 5),
                    extra=dict(role="formation" if cut == "16x9" else "formation9x16", title=title, prompt_id=sid,
                               encode=enc, poster_time=round(t_rest, 3), captions=CAPS,
                               timing=dict(stage1024=T1, stage2048=T2, stage4096=T3, hold=T4, back=TB)))
    M.manifest_put(e)
    sd = M.BUILD / "sheets"
    M.contact_sheet(mp4, sd / f"{name}_sheet.png", n=16, cols=4, tile_w=480 if cut == "16x9" else 270)
    M.seam_sheet(mp4, sd / f"{name}_seam.png", tile_w=640 if cut == "16x9" else 360)
    print(json.dumps(dict(name=name, **p, enc=enc)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "video"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cut", default="16x9", choices=["16x9", "9x16", "both"])
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--look", type=str, default=None)
    a = ap.parse_args()
    sel = VP.selection(a.k)
    if a.only:
        sel = [s for s in sel if s == a.only] or [a.only]
    cuts = ["16x9", "9x16"] if a.cut == "both" else [a.cut]
    if a.what == "data":
        index = []
        for sid in sel:
            rec = export(sid)
            index.append(dict(id=sid, title=rec["title"], json=f"{sid}/formation.json",
                              patches=[p["box"] for p in rec["patches"]]))
            print(sid, [p["box"] for p in rec["patches"]], flush=True)
        if not a.only:
            (DATA / "index.json").write_text(json.dumps(dict(about="formation patches per prompt (v5_formation.py)",
                                                             items=index), indent=1) + "\n")
        return
    if a.look:
        for cut in cuts:
            ctx = context(sel[0], cut)
            outs = []
            for tv in a.look.split(","):
                p = BUILD / "look" / f"{sel[0]}_{cut}_{float(tv):05.2f}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                draw(ctx, float(tv)).save(p)
                outs.append(p)
            M.frames_sheet(outs, BUILD / "look" / f"{sel[0]}_{cut}_look.png", cols=2 if cut == "16x9" else 3,
                           tile_w=800 if cut == "16x9" else 450, label=f"{sel[0]} {cut}")
            print(BUILD / "look" / f"{sel[0]}_{cut}_look.png")
        return
    for cut in cuts:
        for i, sid in enumerate(sel):
            build(sid, i, a.procs, cut)


if __name__ == "__main__":
    main()
