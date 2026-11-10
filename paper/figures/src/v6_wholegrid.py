"""Whole-image comparison grids of the v6 family (Figure 5's style), used by the appendix pages of v6_more.py.

Each cell is a method's full image; that image's own 256 x 256 window of highest edge energy is boxed in red and shown at
native pixels as an inset in the bottom-right corner (bottom-left, top-right, top-left when the inset would cover the
box). Under each image sits one number (ImageReward), best of the row bold. Columns are methods (optionally grouped
under a spanning head, for example a seed), rows are examples with a two-line label at left.
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

from PIL import Image

import v6_common as vc

Image.MAX_IMAGE_PIXELS = None
WIN = 256   # inset window side (native px); Figure 1 and fig:sizes_models use the same size


def _edge(p):
    return vc.detail_map(Image.open(p).convert("RGB"))


def _tile(job):
    path, w, h, win, tn, cn, assets, dpi, q = job
    im = Image.open(path).convert("RGB")
    vc.save_jpeg(vc.thumb(im, w, h, dpi=dpi), Path(assets) / tn, vc.THUMB_Q)
    x, y = win[0], win[1]
    vc.save_jpeg(im.crop((x, y, x + WIN, y + WIN)), Path(assets) / cn, q, subsampling=2)  # 0.47 in insets: 4:2:0
    return im.size


def build(rows: list, cols: list, pdf: Path, assets: Path, stem: str, groups: list | None = None,
          secs: dict | None = None, label_w: float = 45.0, cell: float | None = None, inset: float = 34.0,
          group_gap: float = 6.0, fmt: str = "%.2f", dpi: int = 250, crop_q: int = 85,
          width: float = vc.TEXT_PT, bold_sets: list | None = None) -> dict:
    """rows: [dict(label=(line1, line2), cells={col: dict(path=..., value=float)})];
    cols: [(key, head TeX, ours bool)]; groups: [(head TeX, [col keys])] for a spanning line above the heads;
    secs: {col: TeX} grey seconds line under each head. Returns a record."""
    G = vc.G
    keys = [c[0] for c in cols]
    gidx = {}
    for gi, (_, members) in enumerate(groups or []):
        for k in members:
            gidx[k] = gi
    n_gaps = sum(1 for a, b in zip(keys, keys[1:]) if gidx.get(a) != gidx.get(b)) if groups else 0
    if cell is None:
        cell = (width - label_w - G - (len(keys) - 1) * G - n_gaps * group_gap) / len(keys)
    xs, x = [], label_w + G
    for i, k in enumerate(keys):
        if i and groups and gidx.get(k) != gidx.get(keys[i - 1]):
            x += group_gap
        xs.append(x)
        x += cell + G
    used = x - G
    x0 = (width - used) / 2 if used < width - 0.5 else 0.0      # centre a narrower grid
    xs = [v + x0 for v in xs]
    g_h = 10.0 if groups else 0.0
    head_h = g_h + (18.0 if secs else 11.0)
    val_h, row_gap = 8.0, 3.0
    H = head_h + len(rows) * (cell + val_h) + (len(rows) - 1) * row_gap
    cv = vc.Canvas(width, H, assets)
    for gtext, members in groups or []:
        ix = [keys.index(k) for k in members]
        gx0, gx1 = xs[min(ix)], xs[max(ix)] + cell
        cv.text((gx0 + gx1) / 2, g_h - 2.5, gtext, anchor="base", bold=True)
        cv.line([(gx0, g_h - 0.5), (gx1, g_h - 0.5)], color=vc.MID, lw=0.4)
    for (k, head, ours), x in zip(cols, xs):
        cv.text(x + cell / 2, head_h - (10.5 if secs else 3.0), head, anchor="base",
                color=vc.SCA if ours else vc.INK, bold=ours and "Progressive" in head)
        if secs and k in secs:
            cv.text(x + cell / 2, head_h - 3.0, secs[k], size=6.5, anchor="base", color=vc.MID)
    jobs = [(r_i, k) for r_i, r in enumerate(rows) for k in keys if k in r["cells"]]
    with Pool(min(48, len(jobs))) as mp:
        maps = dict(zip(jobs, mp.map(_edge, [rows[r_i]["cells"][k]["path"] for r_i, k in jobs])))
    wins = {j: vc.best_windows([maps[j]], WIN, WIN, k=1)[0] for j in jobs}
    tiles = []
    for r_i, k in jobs:
        tiles.append((rows[r_i]["cells"][k]["path"], cell, cell, wins[(r_i, k)], f"r{r_i}_{k}_full.jpg",
                      f"r{r_i}_{k}_inset_{wins[(r_i, k)][0]}_{wins[(r_i, k)][1]}.jpg", str(assets), dpi, crop_q))
    with Pool(min(48, len(tiles))) as mp:
        sizes = dict(zip(jobs, mp.map(_tile, tiles)))
    rec = dict(rows=[])
    M = 2.5
    for r_i, r in enumerate(rows):
        y = head_h + r_i * (cell + val_h + row_gap)
        l1, l2 = r["label"]
        cv.text(x0 + label_w / 2, y + cell / 2 - 5.0, l1, anchor="base")
        if l2:
            cv.text(x0 + label_w / 2, y + cell / 2 + 5.5, l2, anchor="base")
        vals = {k: c["value"] for k, c in r["cells"].items()}
        sets = bold_sets or [list(vals)]          # bold the best value within each set (default: the whole row)
        bests = {max([k for k in st if k in vals], key=vals.get) for st in sets}
        best = sorted(bests)
        cells = {}
        for k, x in zip(keys, xs):
            if k not in r["cells"]:
                continue
            wpx, hpx = sizes[(r_i, k)]
            assert wpx == hpx, (r_i, k, wpx, hpx)
            bx = wins[(r_i, k)]
            sc = cell / wpx
            b0, b1 = (x + bx[0] * sc, y + bx[1] * sc), (x + (bx[0] + WIN) * sc, y + (bx[1] + WIN) * sc)
            for corner in ("br", "bl", "tr", "tl"):
                ix = x + M if corner[1] == "l" else x + cell - M - inset
                iy = y + M if corner[0] == "t" else y + cell - M - inset
                if not (ix < b1[0] + 1 and b0[0] - 1 < ix + inset and iy < b1[1] + 1 and b0[1] - 1 < iy + inset):
                    break
            cv.image(x, y, cell, cell, f"r{r_i}_{k}_full.jpg")
            cv.box(*b0, *b1, inside=False)
            cv.image(ix, iy, inset, inset, f"r{r_i}_{k}_inset_{bx[0]}_{bx[1]}.jpg")
            cv.box(ix, iy, ix + inset, iy + inset, inside=True)
            cv.text(x + cell / 2, y + cell + val_h - 1.6, fmt % vals[k], size=6.5, anchor="base", color=vc.MID,
                    bold=(k in bests))
            cells[k] = dict(path=r["cells"][k]["path"], value=vals[k], window_px=list(bx), inset_corner=corner,
                            png_sha256=vc.sidecar(r["cells"][k]["path"])["png_sha256"])
        rec["rows"].append(dict(label=list(r["label"]), best=best, cells=cells, **r.get("meta", {})))
    cv.compile(pdf, stem)
    rec.update(pdf=str(pdf), pdf_bytes=pdf.stat().st_size, size_in=[round(width / 72.27, 3), round(H / 72.27, 3)],
               cell_pt=round(cell, 2), inset_pt=inset, used_width_pt=round(used, 2), thumb_dpi=dpi,
               inset_jpeg_q=crop_q, window_rule=f"each image's own {WIN} x {WIN} window of highest edge energy")
    return rec
