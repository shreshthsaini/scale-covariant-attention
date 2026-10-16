"""Selection aid (not a paper figure): cached thumbnails and numbered contact sheets.

  python contact_sheet.py --labels sca area --res 4096x4096 --out DIR [--ids a b c] [--per 30]
Each sheet tiles the methods side by side per prompt, labelled with the short prompt id.
"""

import argparse
import os
from multiprocessing import Pool

from PIL import Image, ImageDraw

import imgsel

CACHE = "/scratch/09032/saini_2/sca/figcache/thumbs"


def thumb(args):
    label, res, sid, png, side = args
    dst = os.path.join(CACHE, label, res, f"{sid}_{side}.jpg")
    if not os.path.exists(dst):
        imgsel.save_jpeg(imgsel.fit(imgsel.load(png), side), dst, quality=85)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--res", default="4096x4096")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--per", type=int, default=30)
    ap.add_argument("--cols", type=int, default=0)
    ap.add_argument("--side", type=int, default=384)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    idx = {l: {imgsel.short_id(k): v for k, v in imgsel.index(l, a.res).items()} for l in a.labels}
    ids = a.ids or sorted(set.intersection(*[set(v) for v in idx.values()]))
    jobs = [(l, a.res, s, idx[l][s], a.side) for s in ids for l in a.labels]
    with Pool(a.workers) as pool:
        paths = pool.map(thumb, jobs, chunksize=2)
    th = {(j[0], j[2]): p for j, p in zip(jobs, paths)}
    os.makedirs(a.out, exist_ok=True)
    cols = a.cols or (6 if len(a.labels) == 1 else max(1, 6 // len(a.labels)))
    for k in range(0, len(ids), a.per):
        chunk = ids[k:k + a.per]
        tiles = []
        for s in chunk:
            ims = [Image.open(th[(l, s)]) for l in a.labels]
            w = sum(i.width for i in ims) + 4 * (len(ims) - 1)
            h = max(i.height for i in ims) + 14
            t = Image.new("RGB", (w, h), "white")
            x = 0
            for i in ims:
                t.paste(i, (x, 14)); x += i.width + 4
            ImageDraw.Draw(t).text((2, 1), s, fill="black")
            tiles.append(t)
        tw = max(t.width for t in tiles); thh = max(t.height for t in tiles)
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (tw + 8), rows * (thh + 8)), "white")
        for n, t in enumerate(tiles):
            sheet.paste(t, ((n % cols) * (tw + 8), (n // cols) * (thh + 8)))
        sheet.save(os.path.join(a.out, f"sheet_{'_'.join(a.labels)}_{a.res}_{k // a.per:02d}.jpg"), quality=80)
    print(len(ids), "prompts")


if __name__ == "__main__":
    main()
