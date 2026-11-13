"""Tagged variants of the hero loops: web/assets/video/v5/hero_16x9_tagged.mp4, hero_9x16_tagged.mp4 (+ posters).

The same frames as v5_hero.py (run it first; its frame folders are read from the scratch build folder) with two
burned-in elements, for a page that does not overlay its own readout: a "4096 x 4096" tag (top left) and a zoom
indicator (bottom right on 16:9, bottom centre on 9:16): a track from "whole" to "1:1" with a dot at the current
scale (log scale) and a label that reads the zoom factor relative to the whole view and turns into "1:1 native pixels"
in the SCA green on the native hold. The clean loops stay the default for the page (design.md: no burned-in text).

Run from the repository root:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_hero_tagged.py [--cut 16x9|9x16|both]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_hero as HZ  # noqa: E402
import v5_media as M  # noqa: E402

MAX_BYTES = 2_500_000


def overlay(img: Image.Image, s: float, s0: float, cut: str) -> Image.Image:
    W, H = img.size
    phone = cut == "9x16"
    fs = 28 if phone else 22
    mx, my = round(0.06 * W), round(0.06 * H) if not phone else round(0.05 * H)
    M.chip(img, (mx, my), "4096 × 4096", fs, anchor="lt", bg=(8, 10, 9, 170), fg=(236, 238, 233, 255),
           weight=600, radius=6)
    native = abs(s - 1.0) < 1e-6
    zoom = s / s0
    tw = 300 if phone else 240
    if phone:
        x1 = W // 2 + tw // 2
    else:
        x1 = W - mx
    x0 = x1 - tw
    yb = H - my
    # backing so the indicator reads on bright snow and on dark forges alike
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    pad = 14
    dl.rounded_rectangle([x0 - pad, yb - (fs * 3.1) - pad, x1 + pad, yb + pad], radius=8, fill=(8, 10, 9, 150))
    u = math.log(s / s0) / math.log(1.0 / s0)              # 0 at the whole view, 1 at native
    ty = yb - fs * 1.15
    dl.rounded_rectangle([x0, ty - 2, x1, ty + 2], radius=2, fill=M.rgb(M.PAL["rule"]) + (255,))
    xd = x0 + tw * max(0.0, min(1.0, u))
    if xd > x0 + 1:
        dl.rounded_rectangle([x0, ty - 2, xd, ty + 2], radius=2, fill=M.rgb(M.PAL["sca"]) + (255,))
    r = fs * 0.32
    dl.ellipse([xd - r, ty - r, xd + r, ty + r], fill=M.rgb(M.PAL["sca"] if native else M.PAL["ink"]) + (255,))
    f_small = M.font(fs * 0.82, "sans", 500)
    dl.text((x0, yb), "whole", font=f_small, fill=M.rgb(M.PAL["ink2"]) + (255,), anchor="ls")
    dl.text((x1, yb), "1:1", font=f_small, fill=M.rgb(M.PAL["ink2"]) + (255,), anchor="rs")
    lab = "1:1 native pixels" if native else (f"{zoom:.1f}×" if zoom > 1.005 else "whole image")
    dl.text((x1, ty - fs * 0.75), lab, font=M.font(fs, "sans", 600),
            fill=M.rgb(M.PAL["sca"] if native else M.PAL["ink"]) + (255,), anchor="rs", features=["tnum"])
    img = img.convert("RGBA")
    img.alpha_composite(layer)
    return img.convert("RGB")


def _work(task):
    cut, idxs, scales, s0, src, dst = task
    for k, s in zip(idxs, scales):
        im = Image.open(Path(src) / f"{k:05d}.png").convert("RGB")
        overlay(im, s, s0, cut).save(Path(dst) / f"{k:05d}.png", compress_level=1)
    return len(idxs)


def build(cut: str, k: int, fps: int, procs: int):
    items = HZ.selection(k)
    info = HZ.build_cut(cut, [dict(it) for it in items], fps, procs, False, render=False)
    W, H = info["W"], info["H"]
    s0 = HZ.WHOLE_FILL * min(W, H) / 4096
    tag = f"k{len(items)}_f{fps}_" + "_".join(f"{v:g}" for v in (HZ.T_WHOLE, HZ.T_IN, HZ.T_NATIVE, HZ.T_OUT, HZ.T_XF))
    src = HZ.BUILD / cut / f"frames_{tag}"
    if not (src / f"{info['n'] - 1:05d}.png").exists():
        raise FileNotFoundError(f"{src}: run web/build/v5_hero.py --cut {cut} first")
    dst = HZ.BUILD / cut / f"frames_tagged_{tag}"
    dst.mkdir(parents=True, exist_ok=True)
    scales = [f["s"] if f["kind"] == "view" else s0 for f in info["frames"]]
    n = info["n"]
    tasks = [(cut, list(range(i, min(n, i + 8))), scales[i:i + 8], s0, str(src), str(dst)) for i in range(0, n, 8)]
    t0 = time.time()
    with Pool(procs) as pool:
        sum(pool.imap_unordered(_work, tasks))
    print(f"{cut} tagged: {n} frames in {time.time() - t0:.1f} s", flush=True)
    name = f"hero_{cut}_tagged"
    mp4 = M.OUT / f"{name}.mp4"
    enc = M.encode_to_size(dst / "%05d.png", mp4, fps, n, MAX_BYTES, crf_floor=24, preset="veryslow", gop=n,
                           passlog=HZ.BUILD / cut / "x264pass_tagged", x264="aq-mode=3", threads=16)
    p = M.check_video(mp4, max_bytes=MAX_BYTES, w=W, h=H)
    poster = M.poster_webp(Image.open(dst / "00000.png"), M.OUT / f"{name}.webp", quality=80)
    clean = next(e for e in json.loads(M.MANIFEST.read_text())["items"] if e["id"] == f"hero_{cut}")
    e = M.entry_for(name, mp4, poster, loop=True, alt=clean["alt"] + " A tag reads 4096 by 4096 and a zoom "
                    "indicator reaches 1:1 native pixels.", caption=clean["caption"], sources=clean["sources"],
                    script="v5_hero_tagged.py", order=12 + (0 if cut == "16x9" else 1),
                    extra=dict(role=f"hero{cut.replace('x', 'x')}_tagged", variant_of=f"hero_{cut}",
                               note="fallback with the tag and zoom indicator burned in; the clean loop is the "
                                    "default (design.md)", encode=enc))
    M.manifest_put(e)
    sd = M.BUILD / "sheets"
    M.contact_sheet(mp4, sd / f"{name}_sheet.png", n=12, cols=4, tile_w=400 if cut == "16x9" else 240)
    print(json.dumps(dict(name=name, **p, enc=enc)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="both", choices=["16x9", "9x16", "both"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    for cut in (["16x9", "9x16"] if a.cut == "both" else [a.cut]):
        build(cut, a.k, a.fps, a.procs)


if __name__ == "__main__":
    main()
