"""Hero deep-zoom loops of the v5 page: web/assets/video/v5/hero_16x9.mp4 and hero_9x16.mp4 (+ .webp posters).

What it shows. Whole 4096^2 progressive SCA images (FLUX.1-dev, HRDiT's three-stage sampler 1024^2 -> 2048^2 ->
4096^2 with SCA attention at 2048^2 and 4096^2, seed 0) floating on the page's darkroom ground (#0B0D0C), each zoomed
into native pixels (one video pixel per image pixel), held, zoomed back out, then dissolved into the next image. The
last dissolve lands on the first image's whole view, which is also the first frame, so the loop is seamless. No text is
burned in (design.md: the page overlays the title and a readout); the manifest carries a readout track (image, size,
zoom factor per tenth of a second, and the native-pixel holds) for the page's "4096 x 4096" tag and zoom indicator.

Images (rule). The paper hero's rule (paper/figures/src/v6_hero.py, hero_provenance.json): one showcase prompt per
category, random.Random("0:<category>").choice over the category's sorted ids, categories in the hero's order
portrait, animal, landscape, architecture, dense, product; the loop takes the first K of that order (default 3:
blacksmith, owl, ice cave; three keep a 1600 x 900 loop under 2.5 MB at a quality that holds up at 100 percent).
Sources are the 4096^2 PNGs the paper hero uses, sidecar-checked by v6_hero.source().

Zoom target (rule). The video-sized window (W x H native pixels, the 1:1 view) of highest mean edge energy, the
paper's crop rule (v6_common.detail_map / best_windows: PIL FIND_EDGES of the 8x box-downsampled greyscale image, at
least 1/16 of the side from every border).

Pixels. Every frame is resampled from the full 4096^2 PNG (PIL Lanczos with the source box of that frame, so the
filter widens with the downscale and nothing aliases); the native hold is the PNG's own pixels (an exact crop). Zoom
is log-uniform in scale with a smootherstep ease, and the zoom target glides from its place in the whole view to its
place in the native window on the same ease.

Run from the repository root (CPU; renders in parallel processes, frames under the scratch build folder):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_hero.py [--cut 16x9|9x16|both]
      [--k 4] [--fps 30] [--preview]
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
import v6_hero  # noqa: E402

CUTS = {"16x9": (1600, 900), "9x16": (900, 1600)}
MAX_BYTES = 2_500_000
PAD = 48              # source padding (px) in the ground colour, so the image edge antialiases into the ground
WHOLE_FILL = 0.92     # the whole image spans this fraction of the frame's short side
CORNER_PX = 10        # corner radius of the whole image on screen (px), the page's --radius at video scale
# timeline (s): whole hold, zoom in, native hold, zoom out, dissolve to the next image
T_WHOLE, T_IN, T_NATIVE, T_OUT, T_XF = 0.8, 2.3, 1.4, 1.6, 0.6
BUILD = M.BUILD / "hero"


# ------------------------------------------------------------------------------------------------ selection
def selection(k: int) -> list:
    picks = v6_hero.draw()
    out = []
    for cat in v6_hero.ORDER[:k]:
        sid = picks[cat]
        png, meta = v6_hero.source(sid)
        out.append(dict(id=sid, category=cat, png=str(png), png_sha256=meta["png_sha256"],
                        prompt=meta["config"]["prompt"], title=v6_hero.showcase()[sid].get("title", sid)))
    return out


def padded_source(item: dict) -> Path:
    """The PNG on a PAD-px ground border with rounded corners (radius in source px for CORNER_PX at the whole view),
    cached as .npy for the workers."""
    out = BUILD / "src" / f"{item['id']}_pad{PAD}.npy"
    if out.exists():
        return out
    im = Image.open(item["png"]).convert("RGB")
    assert im.size == (4096, 4096), im.size
    n = 4096 + 2 * PAD
    ground = Image.new("RGB", (n, n), M.rgb(M.PAL["screen"]))
    # corner radius in source px: CORNER_PX on screen at the smallest whole-view scale of the two cuts
    s_whole = min(WHOLE_FILL * min(w, h) / 4096 for w, h in CUTS.values())
    r = CORNER_PX / s_whole
    mask = Image.new("L", (4096, 4096), 0)   # aliasing at source scale is below a screen pixel after the downscale
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, 4095, 4095], radius=r, fill=255)
    ground.paste(im, (PAD, PAD), mask)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(ground))
    return out


def target(item: dict, w: int, h: int) -> tuple:
    im = Image.open(item["png"]).convert("RGB")
    (x, y, ww, hh), = vc.best_windows([vc.detail_map(im)], w, h, k=1)
    return x, y


# ------------------------------------------------------------------------------------------------ timeline
def timeline(items: list, W: int, H: int, fps: int) -> tuple[list, list]:
    """Frame specs and the readout track."""
    s0 = WHOLE_FILL * min(W, H) / 4096
    frames, track = [], []
    t = 0.0

    def nf(sec):
        return max(1, round(sec * fps))

    for i, it in enumerate(items):
        tx, ty = it["window"]
        c1 = (tx + W / 2, ty + H / 2)            # native view centre (window centre), integer box at s = 1
        c0 = (2048.0, 2048.0)
        T = np.array(c1)                          # the zoom target point: the window centre
        p0 = np.array([W / 2, H / 2]) + s0 * (T - np.array(c0))   # target's screen position in the whole view
        p1 = np.array([W / 2, H / 2])                              # ... and in the native view

        def view(e):
            s = math.exp(math.log(s0) + (math.log(1.0) - math.log(s0)) * e)
            p = p0 + (p1 - p0) * e
            c = T - (p - np.array([W / 2, H / 2])) / s
            return s, float(c[0]), float(c[1])

        seg = dict(id=it["id"], title=it["title"], category=it["category"], t0=round(t, 3))
        n = nf(T_WHOLE)
        frames += [dict(kind="view", img=i, s=s0, cx=c0[0], cy=c0[1])] * n
        n_in = nf(T_IN)
        seg["zoom_in"] = [round(t + n / fps, 3), round(t + (n + n_in) / fps, 3)]
        for j in range(1, n_in + 1):
            e = float(M.smootherstep(j / n_in))
            s, cx, cy = view(e)
            frames.append(dict(kind="view", img=i, s=s, cx=cx, cy=cy))
        n_nat = nf(T_NATIVE)
        frames += [dict(kind="view", img=i, s=1.0, cx=c1[0], cy=c1[1])] * n_nat
        start_nat = t + (n + n_in) / fps
        seg["native"] = [round(start_nat - 1 / fps, 3), round(start_nat + n_nat / fps, 3)]
        n_out = nf(T_OUT)
        for j in range(1, n_out + 1):
            e = float(M.smootherstep(1 - j / n_out))
            s, cx, cy = view(e)
            frames.append(dict(kind="view", img=i, s=s, cx=cx, cy=cy))
        seg["zoom_out"] = [round(start_nat + n_nat / fps, 3), round(start_nat + (n_nat + n_out) / fps, 3)]
        n_xf = nf(T_XF)
        nxt = (i + 1) % len(items)
        for j in range(1, n_xf + 1):
            a = float(M.smoothstep(j / n_xf))
            frames.append(dict(kind="blend", a=i, b=nxt, alpha=a))
        t = len(frames) / fps
        seg["t1"] = round(t, 3)
        seg["window_px"] = [tx, ty, W, H]
        track.append(seg)
    return frames, track


def zoom_curve(frames: list, fps: int, W: int, H: int, every: int = 3) -> list:
    """[[t, scale]] every `every` frames: scale = video pixels per image pixel (1 = native); dissolves sit at the
    whole-view scale."""
    s0 = WHOLE_FILL * min(W, H) / 4096
    return [[round(k / fps, 3), round(f["s"] if f["kind"] == "view" else s0, 4)]
            for k, f in enumerate(frames) if k % every == 0]


# ------------------------------------------------------------------------------------------------ rendering
_SRC: dict = {}


def _src(path: str) -> Image.Image:
    if path not in _SRC:
        _SRC[path] = Image.fromarray(np.load(path))
    return _SRC[path]


def render_view(src: Image.Image, s: float, cx: float, cy: float, W: int, H: int) -> Image.Image:
    ground = Image.new("RGB", (W, H), M.rgb(M.PAL["screen"]))
    n = src.width                       # padded side
    if abs(s - 1.0) < 1e-9:             # native: the PNG's own pixels
        x0, y0 = round(cx - W / 2) + PAD, round(cy - H / 2) + PAD
        return src.crop((x0, y0, x0 + W, y0 + H))
    # screen rect covered by the padded source; source box of exactly those screen pixels (padded coordinates)
    X0 = max(0, math.ceil(W / 2 + s * (-PAD - cx)))
    X1 = min(W, math.floor(W / 2 + s * (4096 + PAD - cx)))
    Y0 = max(0, math.ceil(H / 2 + s * (-PAD - cy)))
    Y1 = min(H, math.floor(H / 2 + s * (4096 + PAD - cy)))
    u0, u1 = cx + (X0 - W / 2) / s + PAD, cx + (X1 - W / 2) / s + PAD
    v0, v1 = cy + (Y0 - H / 2) / s + PAD, cy + (Y1 - H / 2) / s + PAD
    u0, v0 = max(0.0, u0), max(0.0, v0)
    u1, v1 = min(float(n), u1), min(float(n), v1)
    region = src.resize((X1 - X0, Y1 - Y0), Image.LANCZOS, box=(u0, v0, u1, v1))
    ground.paste(region, (X0, Y0))
    return ground


def _work(task):
    cut, W, H, idxs, specs, srcs, wholes, outdir = task
    for k, f in zip(idxs, specs):
        if f["kind"] == "view":
            im = render_view(_src(srcs[f["img"]]), f["s"], f["cx"], f["cy"], W, H)
        else:
            a = Image.open(wholes[f["a"]])
            b = Image.open(wholes[f["b"]])
            im = Image.blend(a, b, f["alpha"])
        im.save(Path(outdir) / f"{k:05d}.png", compress_level=1)
    return len(idxs)


def build_cut(cut: str, items: list, fps: int, procs: int, preview: bool, render: bool = True) -> dict:
    W, H = CUTS[cut]
    for it in items:
        it["window"] = target(it, W, H)
    frames, track = timeline(items, W, H, fps)
    if not render:
        return dict(W=W, H=H, frames=frames, track=track, n=len(frames))
    root = BUILD / cut
    tag = f"k{len(items)}_f{fps}_" + "_".join(f"{v:g}" for v in (T_WHOLE, T_IN, T_NATIVE, T_OUT, T_XF))
    fdir = root / ("preview" if preview else f"frames_{tag}")
    fdir.mkdir(parents=True, exist_ok=True)
    srcs = [str(padded_source(it)) for it in items]
    # whole views (rest frames of the dissolves), rendered once
    wholes = []
    s0 = WHOLE_FILL * min(W, H) / 4096
    for i, it in enumerate(items):
        p = root / f"whole_{i}.png"
        render_view(Image.fromarray(np.load(srcs[i])), s0, 2048.0, 2048.0, W, H).save(p)
        wholes.append(str(p))
    idx = list(range(len(frames)))
    if preview:
        idx = idx[::max(1, len(idx) // 24)]
    chunk = 6
    tasks = [(cut, W, H, idx[i:i + chunk], [frames[j] for j in idx[i:i + chunk]], srcs, wholes, str(fdir))
             for i in range(0, len(idx), chunk)]
    t0 = time.time()
    with Pool(procs) as pool:
        done = sum(pool.imap_unordered(_work, tasks))
    print(f"{cut}: {done} frames in {time.time() - t0:.1f} s -> {fdir}", flush=True)
    return dict(W=W, H=H, frames=frames, track=track, fdir=fdir, n=len(frames))


def encode_cut(cut: str, info: dict, fps: int, items: list) -> dict:
    W, H = info["W"], info["H"]
    mp4 = M.OUT / f"hero_{cut}.mp4"
    enc = M.encode_to_size(info["fdir"] / "%05d.png", mp4, fps, info["n"], MAX_BYTES, crf_floor=24,
                           preset="veryslow", gop=info["n"], passlog=BUILD / cut / "x264pass", x264="aq-mode=3",
                           threads=16)
    p = M.check_video(mp4, max_bytes=MAX_BYTES, w=W, h=H)
    poster = M.OUT / f"hero_{cut}.webp"
    M.poster_webp(Image.open(info["fdir"] / "00000.png"), poster, quality=80)
    return dict(mp4=mp4, poster=poster, enc=enc, probe=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="both", choices=["16x9", "9x16", "both"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--manifest-only", action="store_true", help="rewrite the manifest entries from the files present")
    a = ap.parse_args()
    items = selection(a.k)
    for it in items:
        M.check_words(it["title"], where=it["id"])
    cuts = ["16x9", "9x16"] if a.cut == "both" else [a.cut]
    for order, cut in enumerate(cuts):
        info = build_cut(cut, [dict(it) for it in items], a.fps, a.procs, a.preview, render=not a.manifest_only)
        if a.preview:
            continue
        if a.manifest_only:
            r = dict(mp4=M.OUT / f"hero_{cut}.mp4", poster=M.OUT / f"hero_{cut}.webp",
                     enc=json.loads(M.MANIFEST.read_text())["items"][[e["id"] for e in json.loads(
                         M.MANIFEST.read_text())["items"]].index(f"hero_{cut}")]["encode"],
                     probe=M.check_video(M.OUT / f"hero_{cut}.mp4", max_bytes=MAX_BYTES))
        else:
            r = encode_cut(cut, info, a.fps, items)
        W, H = info["W"], info["H"]
        role = "hero16x9" if cut == "16x9" else "hero9x16"
        titles = ", ".join(s["title"].lower() for s in info["track"])
        alt = (f"Deep zoom into {len(items)} images generated at 4096 by 4096 pixels with progressive SCA ({titles}): "
               f"each starts whole, zooms to native pixels at a detailed spot, and zooms back out.")
        cap = (f"Progressive SCA at 4096 x 4096, from the whole image to native pixels and back. Images: one showcase "
               f"prompt per category by seeded draw (the paper hero's rule); zoom target: the {W} x {H} window of "
               f"highest edge energy.")
        rec = dict(
            role=role, readout=dict(label="4096 x 4096", native_label="1:1 native pixels",
                                    zoom="scale = video pixels per image pixel; 1.0 = native",
                                    whole_scale=round(WHOLE_FILL * min(W, H) / 4096, 5), segments=info["track"],
                                    zoom_curve=zoom_curve(info["frames"], a.fps, W, H)),
            selection=dict(rule=("paper hero rule (v6_hero.draw): one showcase prompt per category, "
                                 "random.Random('0:<category>').choice over the category's sorted ids; first "
                                 f"{len(items)} of the order {', '.join(v6_hero.ORDER)}"),
                           zoom_rule=(f"{W} x {H} native window of highest mean edge energy "
                                      "(v6_common.best_windows on detail_map, 1/16 border margin)")),
            encode=r["enc"], timing=dict(whole=T_WHOLE, zoom_in=T_IN, native=T_NATIVE, zoom_out=T_OUT,
                                         dissolve=T_XF, fps=a.fps))
        e = M.entry_for(f"hero_{cut}", r["mp4"], r["poster"], loop=True, alt=alt, caption=cap,
                        sources=[dict(id=it["id"], png=it["png"], png_sha256=it["png_sha256"],
                                      window_px=s["window_px"]) for it, s in zip(items, info["track"])],
                        script="v5_hero.py", order=10 + order, extra=rec)
        M.manifest_put(e)
        if a.manifest_only:
            continue
        sheet_dir = M.BUILD / "sheets"
        M.contact_sheet(r["mp4"], sheet_dir / f"hero_{cut}_sheet.png", n=16, cols=4,
                        tile_w=400 if cut == "16x9" else 240)
        M.seam_sheet(r["mp4"], sheet_dir / f"hero_{cut}_seam.png", tile_w=640 if cut == "16x9" else 360)
        print(json.dumps(dict(cut=cut, **{k: v for k, v in r["probe"].items()}, enc=r["enc"])), flush=True)


if __name__ == "__main__":
    main()
