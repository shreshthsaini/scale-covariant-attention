"""The comparison grid of the v6 qualitative family: methods x examples, each cell a full image with a red 1 pt box
and the native 1:1 crop of that box. Used by Figure 5 (v6_qual_main.py) and every appendix sample page
(v6_gallery.py), so they share one grid, one label style, one box style and one crop rule.

Two orientations:
* "rows": methods are rows (rotated labels in a left column), examples are columns; each cell is the full image
  followed on its right by the crop, both as tall as the row.
* "cols": methods are columns (names above), examples are rows; each cell is the full image with the crop beneath it,
  both as wide as the column.

Crop rule (one clause in every caption): the window of fixed size with the highest mean edge energy averaged over the
methods shown for that example (v6_common.best_windows over v6_common.detail_map of each image), at least 1/16 of the
side from every border. Images of one example must share prompt and initial noise (asserted from the sidecars).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

import v6_common as vc

Image.MAX_IMAGE_PIXELS = None
WORKERS = int(os.environ.get("V6_WORKERS", "48"))


@dataclass
class Example:
    key: str                       # file-safe id (short prompt id)
    paths: dict                    # method -> png
    header: str | None = None      # optional TeX text above the example (cols: none)
    window: tuple | None = None    # filled by the rule
    meta: dict = field(default_factory=dict)


def _prep(job):
    """Load one image, return its edge map (for the window rule) and its size."""
    path = job
    im = Image.open(path).convert("RGB")
    return vc.detail_map(im), im.size


def _tiles(job):
    """Write the thumbnail and the crop of one image; returns file names."""
    path, thumb_name, crop_name, w_pt, h_pt, box, assets = job
    im = Image.open(path).convert("RGB")
    vc.save_jpeg(vc.thumb(im, w_pt, h_pt), Path(assets) / thumb_name, vc.THUMB_Q)
    x, y, w, h = box
    vc.save_jpeg(im.crop((x, y, x + w, y + h)), Path(assets) / crop_name, vc.CROP_Q)
    return thumb_name, crop_name


def windows(examples: list[Example], crop_px: tuple[int, int], check_noise: bool = True) -> None:
    """Fill ex.window with the rule's window for every example (parallel over all images)."""
    jobs = [(ex.key, m, p) for ex in examples for m, p in ex.paths.items()]
    with Pool(min(WORKERS, len(jobs))) as pool:
        res = pool.map(_prep, [p for _, _, p in jobs])
    maps: dict = {}
    sizes: dict = {}
    for (k, m, _), (a, size) in zip(jobs, res):
        maps.setdefault(k, []).append(a)
        sizes.setdefault(k, set()).add(size)
    for ex in examples:
        assert len(sizes[ex.key]) == 1, (ex.key, sizes[ex.key])
        if check_noise:
            ex.meta.update(vc.same_prompt_and_noise(ex.paths))
        ex.meta["size"] = list(next(iter(sizes[ex.key])))
        ex.window = vc.best_windows(maps[ex.key], crop_px[0], crop_px[1], k=1)[0]


def build(examples: list[Example], methods: list[str], labels: dict, orient: str, assets: Path, pdf: Path,
          stem: str, full_h: float, crop_w: float, crop_h: float | None = None, label_w: float = 10.0,
          ours: dict | None = None, group_gap: float = vc.GG, head_h: float = 11.0, width: float = vc.TEXT_PT,
          crop_px: tuple[int, int] | None = None, blocks: int = 1, block_gap: float = 8.0) -> dict:
    """Compose the grid, compile it, return the provenance record.

    orient "rows": row height full_h; each example column is full (aspect kept) + G + crop_w; `blocks` > 1 stacks the
    examples in that many bands of rows (each band repeats the methods), separated by block_gap.
    orient "cols": column width = full image width; the crop (crop_w = column width, crop_h tall) sits beneath."""
    ours = ours or {}
    G = vc.G
    W0, H0 = examples[0].meta["size"]
    aspect = W0 / H0
    rec = dict(examples=[], methods=methods, orient=orient)
    jobs = []
    if orient == "rows":
        fw = full_h * aspect                         # full image width in the row
        cw, ch = crop_w, full_h
        per_block = (len(examples) + blocks - 1) // blocks
        band_h = len(methods) * full_h + (len(methods) - 1) * G
        H = blocks * band_h + (blocks - 1) * block_gap
        cv = vc.Canvas(width, H, assets)
        x_start = label_w + G
        for b in range(blocks):
            y0 = b * (band_h + block_gap)
            for r, m in enumerate(methods):
                y = y0 + r * (full_h + G)
                cv.text(label_w / 2 + 0.6, y + full_h / 2, labels[m], size=vc.FS_LABEL, anchor="center", rotate=90,
                        color=ours.get(m, vc.INK), bold=(m == "sca_v2_prog"))
            for j, ex in enumerate(examples[b * per_block:(b + 1) * per_block]):
                gx = x_start + j * (fw + G + cw + group_gap)
                for r, m in enumerate(methods):
                    y = y0 + r * (full_h + G)
                    jobs.append((ex, m, (gx, y, fw, full_h), (gx + fw + G, y, cw, ch)))
        used_w = x_start + per_block * (fw + G + cw) + (per_block - 1) * group_gap
    else:  # cols
        n = len(methods)
        colw = (width - (n - 1) * G) / n
        fw, fh = colw, colw / aspect
        cw, ch = colw, crop_h
        ex_h = fh + G + ch
        H = head_h + len(examples) * ex_h + (len(examples) - 1) * group_gap
        cv = vc.Canvas(width, H, assets)
        for c, m in enumerate(methods):
            x = c * (colw + G)
            cv.text(x + colw / 2, head_h - 3.0, labels[m], size=vc.FS_LABEL, anchor="base",
                    color=ours.get(m, vc.INK), bold=(m == "sca_v2_prog"))
        for i, ex in enumerate(examples):
            y = head_h + i * (ex_h + group_gap)
            for c, m in enumerate(methods):
                x = c * (colw + G)
                jobs.append((ex, m, (x, y, fw, fh), (x, y + fh + G, cw, ch)))
        used_w = width
    # tiles
    work = []
    for ex, m, full, crop in jobs:
        bx, by, bw, bh = ex.window
        work.append((ex.paths[m], f"{ex.key}_{m}_full.jpg", f"{ex.key}_{m}_crop_{bx}_{by}_{bw}x{bh}.jpg",
                     full[2], full[3], ex.window, str(assets)))
    with Pool(min(WORKERS, len(work))) as pool:
        names = pool.map(_tiles, work)
    seen = set()
    for (ex, m, full, crop), (tn, cn) in zip(jobs, names):
        x, y, w, h = full
        cv.image(x, y, w, h, tn)
        bx, by, bw, bh = ex.window
        s = w / ex.meta["size"][0]
        cv.box(x + bx * s, y + by * s, x + (bx + bw) * s, y + (by + bh) * s, inside=False)
        x2, y2, w2, h2 = crop
        cv.image(x2, y2, w2, h2, cn)
        cv.box(x2, y2, x2 + w2, y2 + h2, inside=True)
        if ex.key not in seen:
            seen.add(ex.key)
            rec["examples"].append(dict(key=ex.key, window_px=list(ex.window), size_px=ex.meta["size"],
                                        crop_tile_pt=[round(v, 2) for v in (w2, h2)],
                                        crop_ppi=round(bw / (w2 / 72.27), 1), full_tile_pt=[round(w, 2), round(h, 2)],
                                        prompt=ex.meta.get("prompt"), prompt_id=ex.meta.get("prompt_id"),
                                        derived_seed=ex.meta.get("derived_seed"),
                                        sources={mm: str(p) for mm, p in ex.paths.items()}))
    cv.compile(pdf, stem)
    rec.update(pdf=str(pdf), pdf_bytes=pdf.stat().st_size, size_pt=[round(width, 2), round(cv.h, 2)],
               size_in=[round(width / 72.27, 3), round(cv.h / 72.27, 3)], used_width_pt=round(used_w, 2))
    return rec
