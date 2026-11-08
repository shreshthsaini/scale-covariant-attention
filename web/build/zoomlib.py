"""Shared pieces of the page's deep-zoom grids (web/assets/js/zoomgrid.js): tour regions and native-pixel tiles.

Tour regions. Every image of one grid (one prompt and seed, several methods at the same size) gets an edge-energy
map (absolute horizontal plus vertical differences of the grey image at half resolution). Each map is divided by
its own mean, so no method weighs more than another, and the grid's maps are averaged. The k windows of `span`
native pixels with the largest summed energy, at least 1.4 spans apart and away from the border, are the stops of
the auto-zoom tour; every method in the grid uses the same stops.

Tiles. `cut_grid` splits an image into 1024 px native-pixel WebP tiles at quality 80 (4 x 4 at 4096^2), which the
page loads only when a view is zoomed past what the overview image holds. `cut_regions` cuts one 1024 px tile
centred on each tour stop, for sets too large to tile whole (the 96-prompt page) or images whose tiles are large
(full attention's noise).
"""
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TILE = 1024
QUALITY = 80


def edge_map(im, f=2):
    """Mean-normalised edge energy of an RGB image at 1/f resolution."""
    g = np.asarray(im.convert("L").reduce(f), dtype=np.float32)
    e = np.abs(np.diff(g, axis=1, append=g[:, -1:])) + np.abs(np.diff(g, axis=0, append=g[-1:, :]))
    return e / max(float(e.mean()), 1e-6)


def regions(maps, size, k=3, span=640, f=2, margin=0.05):
    """Tour stops [cx, cy, span] (native px) from a list of edge maps of one grid (same shape)."""
    W, H = size
    e = np.mean(np.stack(maps), axis=0)
    h, w = e.shape
    s = max(8, span // f)
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    st = max(1, s // 8)
    ys, xs = np.arange(0, h - s + 1, st), np.arange(0, w - s + 1, st)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    box = ii[Y + s, X + s] - ii[Y, X + s] - ii[Y + s, X] + ii[Y, X]
    cx, cy = (X + s / 2) * f, (Y + s / 2) * f
    ok = ((cx >= margin * W + span / 2) & (cx <= W * (1 - margin) - span / 2)
          & (cy >= margin * H + span / 2) & (cy <= H * (1 - margin) - span / 2))
    box = np.where(ok, box, -np.inf).ravel()
    cx, cy = cx.ravel(), cy.ravel()
    md2 = (1.4 * span) ** 2
    out = []
    for i in np.argsort(box)[::-1]:
        if not np.isfinite(box[i]):
            break
        x, y = float(cx[i]), float(cy[i])
        if all((x - a) ** 2 + (y - b) ** 2 >= md2 for a, b in out):
            out.append((x, y))
            if len(out) == k:
                break
    # visit order: strongest first, then the nearest unvisited stop
    order = [out[0]] if out else []
    rest = out[1:]
    while rest:
        a, b = order[-1]
        j = min(range(len(rest)), key=lambda t: (rest[t][0] - a) ** 2 + (rest[t][1] - b) ** 2)
        order.append(rest.pop(j))
    return [[int(round(x)), int(round(y)), int(span)] for x, y in order]


def fresh(path, src):
    return os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(src)


def cut_grid(im, outdir, src=None, tile=TILE, q=QUALITY):
    """Whole image in tile x tile native-pixel WebP tiles named <col>_<row>.webp."""
    os.makedirs(outdir, exist_ok=True)
    W, H = im.size
    cols, rows = -(-W // tile), -(-H // tile)
    for r in range(rows):
        for c in range(cols):
            p = os.path.join(outdir, f"{c}_{r}.webp")
            if src and fresh(p, src):
                continue
            im.crop((c * tile, r * tile, min(W, (c + 1) * tile), min(H, (r + 1) * tile))).save(
                p, "WEBP", quality=q, method=6)
    return {"size": tile, "cols": cols, "rows": rows}


def cut_regions(im, regs, outdir, src=None, tile=TILE, q=QUALITY):
    """One tile x tile native-pixel tile centred on each stop, clamped inside the image: [[x0, y0, w, h, name]]."""
    os.makedirs(outdir, exist_ok=True)
    W, H = im.size
    out = []
    for k, (cx, cy, _) in enumerate(regs):
        tw, th = min(tile, W), min(tile, H)
        x0 = int(min(max(0, cx - tw // 2), W - tw))
        y0 = int(min(max(0, cy - th // 2), H - th))
        p = os.path.join(outdir, f"r{k}.webp")
        if not (src and fresh(p, src)):
            im.crop((x0, y0, x0 + tw, y0 + th)).save(p, "WEBP", quality=q, method=6)
        out.append([x0, y0, tw, th, f"r{k}.webp"])
    return out


def save_view(im, path, long_side=1600, q=80, src=None):
    """Overview WebP, long side `long_side` px (never upscaled). Returns its [w, h]."""
    if not (src and fresh(path, src)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        W, H = im.size
        s = min(1.0, long_side / max(W, H))
        im.resize((round(W * s), round(H * s)), Image.LANCZOS).save(path, "WEBP", quality=q, method=6)
    with Image.open(path) as v:
        return list(v.size)


def dir_bytes(d):
    return sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0
