#!/usr/bin/env python
"""Render the SCA method explainer video for the project webpage.

Output: a silent 1920x1080 H.264 MP4 at 30 fps (about 77 s) with burned-in captions, SRT and WebVTT
copies of those captions, and a 1280 px WebP poster frame. Every frame is drawn with matplotlib (Agg)
and piped as raw RGB into the vendored ffmpeg (libx264, yuv420p, BT.709 tags, -threads 4).

Scenes
  1 Two ways to fail at 4096^2. FLUX.1-dev with full attention (stripes, flat field) beside the
    per-head mix with native-size windows from step 1 (five Moai), then a slow push-in to a 1:1 crop of
    the stripes.
  2 What a head wants. One query token, its native-size local window (amber) and the block-centre
    global keys (blue, one per 4x4 block) on a real SCA 4096^2 image (schematic grid, s = 4), then the
    per-head weight pi on a slider.
  3 Calibration by emulation. At 1024^2 both views shrink by 4 (window 64 -> 16 tokens, global grid
    64x64 -> 16x16 keys); one head's output error against full attention as a function of pi, from the
    real sufficient statistics; then the real scale-4 pi table (57 blocks x 24 heads, the 13 populated
    noise bins; heads sorted within each block as in the paper) filling in bin by bin.
  4 The opening phase. The 28-step sigma schedule (sigma-linear axis) for windows-from-step-1 and for
    SCA (8 steps global only, 4 shrinking-window steps, then native windows), with the Moai pair (the
    right image is SCA without the schedule: the dev prompt has no final-operator image; the opening it
    shows is the same); the small second Moai is framed, not hidden.
  5 The schedule at 4096^2 (final operator, sca_v2). A step counter 1..28 with the 57x24 pi map at each
    step's sigma: the scale-4 table on native steps and, on the four wide-window steps, the log-linear mix
    of the scale-2 and scale-4 tables at ratio max(grid) / w (sca/attn.py table_interp), all checked
    against the sampler's pi_trace in the final image's sidecar; the window side per step; the final SCA
    image with a 1:1 crop, then an end card with the speed line from --numbers. No intermediate sampler
    states are shown.

Inputs (read only)
  /scratch/09032/saini_2/sca/out/{main/full,abl/abl_v0,main/sca_nb}/4096x4096/a4k_eval_4096_<id>_*_seed0.png
      and the JSON sidecar of the final image (28 sigmas, gate, pi_trace); ids in SOURCES below.
  /scratch/09032/saini_2/sca/calib/k1_1024.npz: s4_pi_head [57, 24, 14], sigma_centers, populated, stats
      (falls back to the identical compact copy paper/figures/src/data/k1_1024_pi.npz).
  paper/figures/src/scafig.py: palette and PI_CMAP.  --numbers JSON: speed.sca_s, speed.area_s, speed.ratio.
Outputs
  <out>/sca_method.mp4, sca_method.srt, sca_method.vtt, sca_method_poster.webp (default out:
  web/assets/video/method). <work>/cache holds the downscaled images and native crops, made once from
  the 4096^2 PNGs (manifest.json records source paths and SHA256); <work>/preview holds --preview frames.
Run (repository root, CPU only, one process)
  nice -n 19 /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/render_method_video.py \
      --numbers web/build/numbers.json
  --preview 3,20,40 writes those frames (seconds) as PNG to <work>/preview and skips the encode.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper" / "figures" / "src"))
import scafig as sf  # noqa: E402  (paper palette and the pi colormap)

# ----------------------------------------------------------------------------- inputs
OUT_ROOT = Path("/scratch/09032/saini_2/sca/out")
CALIB = Path("/scratch/09032/saini_2/sca/calib/k1_1024.npz")
CALIB_COPY = REPO / "paper" / "figures" / "src" / "data" / "k1_1024_pi.npz"
WORK = Path("/scratch/09032/saini_2/sca/web_assets/method_video")
FFMPEG = Path("/work/09032/saini_2/vista/library/ffmpeg-7.1-shared")
METHOD_DIRS = {"full": "main/full", "abl_v0": "abl/abl_v0", "sca": "main/sca_nb", "sca_v2": "dupfix/sca_v2"}
SOURCES = {  # name -> (method, prompt id); every image is FLUX.1-dev, 4096^2, seed 0
    "stripes": ("full", "6648013380537069782"),      # full attention: vertical stripes
    "flat": ("full", "1638867997226496674"),         # full attention: flat field
    "moai_v0": ("abl_v0", "3614061571823823193"),    # per-head mix from step 1: five Moai
    "moai_sca": ("sca", "3614061571823823193"),      # SCA without the schedule: one large Moai, a small one far off
    "sheep": ("sca_v2", "minus8343371271909676658"),  # backdrop of the schematic (scenes 2 and 3), final operator
    "guitar": ("sca_v2", "5849277588805396896"),     # final image (final operator); its sidecar gives sigmas, schedule
}
STRIPES_C = (1480, 2048)            # native centre of the 1:1 crop that ends the push-in
GUITAR_CROP = (1540, 880, 600)      # native x0, y0, side of the 1:1 inset
MOAI_SMALL = (1200, 3450, 340, 400)  # native box around the small second Moai in SCA's image
MID_BIN = 7                         # heads are sorted within each block by pi at this bin (paper)

# ----------------------------------------------------------------------------- style
W, H, DPI = 1920, 1080, 100
MX, TITLE_Y, BAND_Y = 80, 84, 924
XF = 0.6                            # cross-fade between scenes, seconds
INK, MID, RULE = sf.INK, sf.MID, sf.RULE
GLOBAL, GLOBAL_LIGHT, LOCAL, LOCAL_LIGHT = sf.GLOBAL, sf.GLOBAL_LIGHT, sf.LOCAL, sf.LOCAL_LIGHT
SCA, FULL, ZOOM = sf.SCA, sf.METHOD_COLOR["full"], sf.ZOOM
GLOBAL_DARK, LOCAL_DARK = sf.pi_color(0.0), sf.pi_color(1.0)  # the ends of PI_CMAP, for text
WIDE = sf.pi_color(0.72)            # shrinking-window steps of the schedule (between LOCAL_LIGHT and LOCAL)
BAND, EMPTY = "#F3F4F6", "#ECEEF1"
PI_LUT = (sf.PI_CMAP(np.linspace(0, 1, 256))[:, :3] * 255 + 0.5).astype(np.uint8)
FONT_DIR = Path("/usr/share/fonts/urw-base35")


def setup_fonts():
    names = ["NimbusSans-Regular.otf", "NimbusSans-Bold.otf"]
    if all((FONT_DIR / n).exists() for n in names):
        for n in names:
            font_manager.fontManager.addfont(str(FONT_DIR / n))
        family = "Nimbus Sans"
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "text.antialiased": True})
    return family


def pt(px):
    """Font size in points for a size in pixels at DPI 100."""
    return px * 72.0 / DPI


def rgb(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def ramp(t, t0, t1):
    return ease((t - t0) / (t1 - t0))


def lerp(a, b, u):
    return a + (b - a) * u


def keyframes(t, keys):
    """Eased piecewise interpolation through (time, value) pairs."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return lerp(v0, v1, ramp(t, t0, t1))
    return keys[-1][1]


def pi_rgb(p):
    return PI_LUT[np.clip(np.rint(np.asarray(p) * 255), 0, 255).astype(int)]


def cells(rgb_img, c):
    return np.repeat(np.repeat(rgb_img, c, axis=0), c, axis=1)


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


# ----------------------------------------------------------------------------- cache and data
def find_png(method, pid):
    hits = sorted(glob.glob(str(OUT_ROOT / METHOD_DIRS[method] / "4096x4096" / f"a4k_eval_4096_{pid}_*_seed0.png")))
    if len(hits) != 1:
        raise FileNotFoundError(f"{method} {pid}: {hits}")
    return Path(hits[0])


def stripes_region():
    side = 1600
    x0 = int(min(max(STRIPES_C[0] - side // 2, 0), 4096 - side))
    y0 = int(min(max(STRIPES_C[1] - side // 2, 0), 4096 - side))
    return x0, y0, side


def cache_files(name):
    files = [f"{name}_1024.png"]
    if name == "stripes":
        x0, y0, s = stripes_region()
        files += [f"{name}_2048.png", f"{name}_native_{x0}_{y0}_{s}.png"]
    if name == "guitar":
        files.append("guitar_crop_%d_%d_%d.png" % GUITAR_CROP)
    return files


def build_cache(work, rebuild=False):
    """Decode each 4096^2 PNG at most once and keep the small versions the scenes need."""
    cdir = work / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    mpath = cdir / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() and not rebuild else {}
    for name, (method, pid) in SOURCES.items():
        png = find_png(method, pid)
        files = cache_files(name)
        rec = man.get(name, {})
        if not rebuild and rec.get("png") == str(png) and all((cdir / f).exists() for f in files):
            continue
        t0 = time.time()
        im = Image.open(png).convert("RGB")
        assert im.size == (4096, 4096), (png, im.size)
        im.resize((1024, 1024), Image.LANCZOS).save(cdir / files[0])
        if name == "stripes":
            im.resize((2048, 2048), Image.LANCZOS).save(cdir / files[1])
            x0, y0, s = stripes_region()
            im.crop((x0, y0, x0 + s, y0 + s)).save(cdir / files[2])
        if name == "guitar":
            x, y, s = GUITAR_CROP
            im.crop((x, y, x + s, y + s)).save(cdir / files[1])
        man[name] = {"method": method, "prompt_id": pid, "png": str(png), "sha256": sha256(png), "files": files}
        print(f"cache {name}: {png.name} ({time.time() - t0:.1f} s)", flush=True)
    mpath.write_text(json.dumps(man, indent=1) + "\n")
    return man


def interp_pi(table, centers, sigma):
    """sca/attn.py PiTable.__call__: linear in sigma between bin centres, constant beyond them."""
    j = int(np.searchsorted(centers, sigma))
    if j <= 0:
        return table[..., 0]
    if j >= len(centers):
        return table[..., -1]
    a = (sigma - centers[j - 1]) / (centers[j] - centers[j - 1])
    return (1 - a) * table[..., j - 1] + a * table[..., j]


class Data:
    def __init__(self, work, numbers):
        cdir = work / "cache"
        man = json.loads((cdir / "manifest.json").read_text())
        self.img = {k: Image.open(cdir / v["files"][0]).convert("RGB") for k, v in man.items()}
        x0, y0, s = stripes_region()
        self.stripes2048 = Image.open(cdir / man["stripes"]["files"][1]).convert("RGB")
        self.stripes_native = (Image.open(cdir / man["stripes"]["files"][2]).convert("RGB"), x0, y0)
        self.guitar_crop = np.asarray(Image.open(cdir / man["guitar"]["files"][1]).convert("RGB"))
        self.manifest = man
        # calibration table (scale 4 is the table used at 4096^2)
        path = CALIB if CALIB.exists() else CALIB_COPY
        z = np.load(path)
        self.calib_path = path
        self.pi = z["s4_pi_head"].astype(np.float64)            # [57 blocks, 24 heads, 14 bins]
        self.pi2 = z["s2_pi_head"].astype(np.float64)           # scale-2 table, for the wide-window steps
        self.centers = z["sigma_centers"].astype(np.float64)
        self.populated = z["populated"].astype(bool)
        if "stats" in z.files:
            self.stats = z["stats"][[int(s) for s in z["scales"]].index(4)]
        else:
            self.stats = z["s4_stats"]
        assert self.pi.shape == (57, 24, 14) and self.stats.shape[1:] == (57, 24, 14)
        self.order = np.argsort(-self.pi[:, :, MID_BIN], axis=1)  # per block, highest pi first
        # sampler schedule and pi trace of the final image
        side = Path(man["guitar"]["png"]).with_suffix(".json")
        meta = json.loads(side.read_text())
        self.sigmas = np.array(meta["sigmas"], dtype=np.float64)
        self.gate = float(meta["config"]["sca"]["gate_sigma"])
        assert int(meta["config"]["sca"]["pi_scale"]) == 4 and len(self.sigmas) == 28
        self.sched = [(float(a), int(b)) for a, b in sorted(meta["config"]["sca"].get("win_sched", []), reverse=True)]
        self.native = int(meta["config"]["sca"]["window"])
        self.side = max(meta["config"]["grid"])
        self.bridge = int(meta["config"]["sca"].get("bridge_cells", 0)) ** 2 * int(meta["config"]["sca"].get("bridge_k", 0))
        err = max(abs(float(self.pi_at(tr["sigma"]).mean()) - tr["pi_mean"]) for tr in meta["pi_trace"])
        self.n_open = int((self.sigmas > self.gate).sum())
        self.windows = [None if sg > self.gate else self.window_at(sg) for sg in self.sigmas]
        self.n_wide = sum(1 for w in self.windows if w is not None and w != self.native)
        print(f"calib {path}; sigma schedule from {side.name}; opening steps {self.n_open}; "
              f"pi_trace max |diff| {err:.2e}", flush=True)
        assert err < 1e-9 and self.n_open == 8
        self.example = self.pick_example()
        self.speed = read_speed(numbers)

    def window_at(self, sigma):
        """SCAContext.window_now: the win_sched side while sigma > its bound, else the native window."""
        for bound, w in self.sched:
            if sigma > bound:
                return w
        return self.native

    def pi_at(self, sigma):
        """SCAContext.pi_table_now: on a wide-window step the log-linear mix of the scale-2 and scale-4 tables
        at ratio side / w (table_interp), else the scale-4 table; linear in sigma between bin centres."""
        w = self.window_at(sigma)
        p4 = interp_pi(self.pi, self.centers, sigma)
        if w == self.native:
            return p4
        lam = min(max(math.log(self.side / w / 2) / math.log(2), 0.0), 1.0)
        return (1 - lam) * interp_pi(self.pi2, self.centers, sigma) + lam * p4

    def fit(self, name, size, wash=0.0):
        a = np.asarray(self.img[name].resize((size, size), Image.LANCZOS)).astype(np.float32)
        return (a * (1 - wash) + 255 * wash + 0.5).astype(np.uint8)

    def heat(self, p, c):
        """[57, 24] pi map -> RGB image with heads sorted within each block (best at the bottom)."""
        s = np.take_along_axis(p, self.order, axis=1)          # [57, 24], rank 0 = highest pi
        return cells(pi_rgb(s.T[::-1]), c)

    def bin_map(self, b):
        return self.pi[:, :, b]

    def pick_example(self):
        """One (block, head) cell at MID_BIN with an interior optimum and a clear minimum."""
        A, B, E, F = (self.stats[i][:, :, MID_BIN] for i in (0, 1, 2, 4))
        pistar = np.clip(B / np.maximum(A, 1e-300), 0, 1)
        J = lambda p: (E - 2 * p * B + p * p * A) / np.maximum(F, 1e-300)  # noqa: E731
        score = np.minimum(J(0.0), J(1.0)) / np.maximum(J(pistar), 1e-12)
        score[(pistar < 0.4) | (pistar > 0.6)] = -1
        l, h = np.unravel_index(int(np.argmax(score)), score.shape)
        assert abs(pistar[l, h] - self.pi[l, h, MID_BIN]) < 1e-9
        ps = np.linspace(0, 1, 121)
        curve = np.array([J(p)[l, h] for p in ps])
        print(f"example head: block {l}, head {h}, sigma bin {MID_BIN}, pi* {pistar[l, h]:.3f}, "
              f"rel. error {curve.min():.4f} vs {J(0.0)[l, h]:.4f} (global) and {J(1.0)[l, h]:.4f} (local)")
        return {"block": int(l), "head": int(h), "pistar": float(pistar[l, h]), "pi": ps, "err": curve}


def read_speed(path):
    if path is None or not Path(path).exists():
        print(f"numbers: {path} not found; the end card carries no speed line", flush=True)
        return None
    d = json.loads(Path(path).read_text())

    def get(key):  # flat {"speed.sca_s": {"value", "display"}} or nested {"speed": {"sca_s": number}}
        v = d[key] if key in d else d["speed"][key.split(".", 1)[1]]
        return (float(v["value"]), v.get("display")) if isinstance(v, dict) else (float(v), None)

    (sca, sca_d), (area, area_d), (ratio, ratio_d) = get("speed.sca_s"), get("speed.area_s"), get("speed.ratio")
    sca_d, area_d = sca_d or f"{sca:.1f}", area_d or f"{area:.1f}"
    ratio_d = (ratio_d or f"{ratio:.2f}").rstrip("×") + "×"
    if abs(area / sca - ratio) > 0.01:
        print(f"warning: speed.ratio {ratio} differs from area_s / sca_s = {area / sca:.3f}")
    print(f"numbers: {path}: SCA {sca_d} s, AReA {area_d} s, ratio {ratio_d}", flush=True)
    return sca_d, area_d, ratio_d


class Zoomer:
    """A square view (centre cx, cy and side L in native 4096^2 pixels) rendered at D x D pixels."""

    def __init__(self, data, D):
        self.D = D
        self.levels = [(data.img["stripes"], 0.25, 0, 0), (data.stripes2048, 0.5, 0, 0)]
        self.native = data.stripes_native

    def view(self, cx, cy, L):
        x0, y0 = cx - L / 2, cy - L / 2
        for im, f, ox, oy in self.levels:
            if L * f >= self.D * 0.999:
                box = ((x0 - ox) * f, (y0 - oy) * f, (x0 - ox + L) * f, (y0 - oy + L) * f)
                return np.asarray(im.resize((self.D, self.D), Image.BICUBIC, box=box))
        im, ox, oy = self.native
        if x0 >= ox and y0 >= oy and x0 + L <= ox + im.width and y0 + L <= oy + im.height:
            box = (x0 - ox, y0 - oy, x0 - ox + L, y0 - oy + L)
            return np.asarray(im.resize((self.D, self.D), Image.BICUBIC, box=box))
        im, f = self.levels[1][0], 0.5
        return np.asarray(im.resize((self.D, self.D), Image.BICUBIC, box=(x0 * f, y0 * f, (x0 + L) * f, (y0 + L) * f)))


# ----------------------------------------------------------------------------- drawing
class Canvas:
    """A 1920x1080 figure with one full-frame axes in pixel coordinates (y grows downwards)."""

    def __init__(self, band=True):
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
        ax.set_autoscale_on(False)
        self.ax = ax
        if band:
            self.rect(0, BAND_Y, W, H - BAND_Y, fc=BAND, z=0)

    def text(self, x, y, s, size, color=INK, weight="normal", ha="left", va="baseline", z=8, **kw):
        return self.ax.text(x, y, s, fontsize=pt(size), color=color, fontweight=weight, ha=ha, va=va,
                            zorder=z, **kw)

    def image(self, arr, x, y, z=1):
        h, w = arr.shape[:2]
        return self.ax.imshow(arr, extent=(x, x + w, y + h, y), interpolation="nearest", aspect="auto",
                              zorder=z, resample=False)

    def rect(self, x, y, w, h, fc="none", ec="none", lw=0.0, z=2, alpha=1.0, **kw):
        p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw * 0.72, zorder=z, alpha=alpha,
                      **kw)
        self.ax.add_patch(p)
        return p

    def line(self, xs, ys, color=INK, lw=1.0, z=3, **kw):
        ln = Line2D(xs, ys, color=color, linewidth=lw * 0.72, zorder=z, **kw)
        self.ax.add_line(ln)
        return ln

    def dots(self, xs, ys, d, color, edge="white", ew=1.5, z=6):
        return self.ax.scatter(xs, ys, s=pt(d) ** 2, c=[color] * len(xs), edgecolors=edge,
                               linewidths=ew * 0.72, zorder=z)

    def grab(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3]

    def close(self):
        plt.close(self.fig)


def set_alpha(arts, a):
    for x in arts:
        x.set_alpha(a)


class Scene:
    title, dur, band, cues = "", 10.0, True, []

    def __init__(self, data):
        self.d, self.c = data, None

    def render(self, t):
        if self.c is None:
            self.c = Canvas(self.band)
            if self.title:
                self.c.text(MX, TITLE_Y, self.title, 44, weight="bold")
            self.build()
        self.update(t)
        return self.c.grab()

    def release(self):
        if self.c is not None:
            self.c.close()
            self.c = None

    def build(self):
        raise NotImplementedError

    def update(self, t):
        raise NotImplementedError


def label_row(c, x, y, main, sub, color, marker="square"):
    """Colour marker, a main label and a grey second line; returns the artists."""
    arts = []
    if marker == "square":
        arts.append(c.rect(x, y - 19, 19, 19, fc=color))
    arts.append(c.text(x + 31, y, main, 30))
    if sub:
        arts.append(c.text(x + 31, y + 38, sub, 26, color=MID))
    return arts


# ----------------------------------------------------------------------------- scene 1
class SceneFail(Scene):
    title = "Two ways to fail at 4096²"
    dur = 13.5
    S, Y, X = 560, 160, (80, 680, 1280)
    Z0, Z1 = 7.6, 12.4
    cues = [(0.5, 4.8, "At 4096², FLUX.1-dev with full attention", "returns stripes or a flat field."),
            (5.0, 8.3, "Native-size windows from step 1", "draw five Moai instead of one."),
            (8.5, 12.8, "Up close, the stripes hold no detail.")]

    def build(self):
        c, d, S = self.c, self.d, self.S
        self.zoom = Zoomer(d, S)
        self.ims = [c.image(self.zoom.view(2048, 2048, 4096), self.X[0], self.Y),
                    c.image(d.fit("flat", S), self.X[1], self.Y),
                    c.image(d.fit("moai_v0", S), self.X[2], self.Y)]
        rows = [("Full attention", "stripes", FULL), ("Full attention", "flat field", FULL),
                ("Windows from step 1", "five Moai", LOCAL)]
        self.labels = [label_row(c, x, self.Y + S + 50, *r) for x, r in zip(self.X, rows)]
        self.frame = c.rect(0, 0, 1, 1, ec=ZOOM, lw=4, z=5)
        x, y = self.X[0] + 14, self.Y + 14
        self.tag = [c.rect(x, y, 64, 40, fc=ZOOM, z=6), c.text(x + 32, y + 30, "1:1", 26, weight="bold", ha="center")]

    def update(self, t):
        S, X, Y = self.S, self.X, self.Y
        for im, labs, a in zip(self.ims, self.labels, (ramp(t, 0.3, 1.1), ramp(t, 0.6, 1.4), ramp(t, 4.9, 5.7))):
            im.set_alpha(a)
            set_alpha(labs, a)
        u = ramp(t, self.Z0, self.Z1)
        L = 4096.0 * (S / 4096.0) ** u
        k = (L - S) / (4096.0 - S)
        cx, cy = lerp(STRIPES_C[0], 2048, k), lerp(STRIPES_C[1], 2048, k)
        self.ims[0].set_data(self.zoom.view(cx, cy, L))
        s = S / L
        fx, fy = X[0] + (STRIPES_C[0] - S / 2 - (cx - L / 2)) * s, Y + (STRIPES_C[1] - S / 2 - (cy - L / 2)) * s
        self.frame.set_bounds(fx, fy, S * s, S * s)
        self.frame.set_alpha(ramp(t, 7.0, 7.6))
        set_alpha(self.tag, ramp(t, 12.2, 12.8))


# ----------------------------------------------------------------------------- scene 2
class SceneHead(Scene):
    title = "What a head wants"
    dur = 14.5
    P, PX, PY, N, Q = 640, 100, 170, 32, (15, 14)  # photo side, origin, schematic cells, query (row, col)
    cues = [(0.5, 3.2, "Each head sees two views of the image."),
            (3.4, 6.4, "Local: an exact native-size window", "of 64×64 tokens around the query."),
            (6.6, 9.6, "Global: one key per 4×4 block,", "the whole image on a 64×64 grid."),
            (9.8, 13.8, "A per-head weight π mixes the two.")]
    SX0, SX1, SY = 900, 1760, 742
    PI_KEYS = [(10.5, 0.5), (11.3, 0.9), (12.2, 0.15), (13.1, 0.62)]

    def build(self):
        c, P, N = self.c, self.P, self.N
        cell = P / N
        self.photo = c.image(self.d.fit("sheep", P, wash=0.30), self.PX, self.PY)
        segs = [[(self.PX + i * cell, self.PY), (self.PX + i * cell, self.PY + P)] for i in range(N + 1)]
        segs += [[(self.PX, self.PY + i * cell), (self.PX + P, self.PY + i * cell)] for i in range(N + 1)]
        self.grid = LineCollection(segs, colors="white", linewidths=0.8 * 0.72, zorder=3)
        c.ax.add_collection(self.grid)
        self.note = c.text(self.PX, self.PY + P + 40, "Schematic token grid at s = 4", 24, color=MID)
        qr, qc = self.Q
        self.qcell = (self.PX + qc * cell, self.PY + qr * cell, cell)
        self.win_box = (self.PX + (qc - 4) * cell, self.PY + (qr - 4) * cell, 8 * cell)
        self.win_fill = c.rect(0, 0, 1, 1, fc=LOCAL, z=4)
        self.win_edge = c.rect(0, 0, 1, 1, ec=LOCAL, lw=4, z=5)
        self.query = c.rect(*self.qcell[:2], cell, cell, fc=INK, ec="white", lw=2, z=7)
        b = np.arange(8) * 4 + 2  # block-centre token of each 4x4 block
        gy, gx = np.meshgrid(b, b, indexing="ij")
        xs, ys = self.PX + (gx.ravel() + 0.5) * cell, self.PY + (gy.ravel() + 0.5) * cell
        self.dots = c.dots(xs, ys, 12, GLOBAL, ew=2)
        dist = np.hypot(gx.ravel() - qc, gy.ravel() - qr)
        self.dist = dist / dist.max()
        self.dot_rgba = np.tile(np.r_[sf_rgb(GLOBAL), 1.0], (len(xs), 1))
        self.dot_edge = np.tile(np.r_[1.0, 1.0, 1.0, 1.0], (len(xs), 1))
        lx = 900
        self.leg = [
            [c.rect(lx, 262 - 21, 21, 21, fc=INK, z=6)] + label_row(c, lx, 262, "Query token", "one of 256×256 tokens at 4096²", None, None)[0:],
            [c.rect(lx - 2, 392 - 23, 25, 25, fc=LOCAL_LIGHT, ec=LOCAL, lw=3, z=6)]
            + label_row(c, lx, 392, "Local view", "exact window of 64×64 tokens", None, None),
            [c.dots([lx + 10.5], [392 + 130 - 10.5], 20, GLOBAL, ew=2)]
            + label_row(c, lx, 522, "Global view", "one key per 4×4 block, 64×64 grid", None, None),
        ]
        # pi slider
        ramp_img = pi_rgb(np.linspace(0, 1, self.SX1 - self.SX0))[None].repeat(16, axis=0)
        self.track = c.image(ramp_img, self.SX0, self.SY - 8, z=3)
        self.knob = Circle((self.SX0, self.SY), 17, facecolor="white", edgecolor=INK, linewidth=3 * 0.72, zorder=7)
        c.ax.add_patch(self.knob)
        self.knob_lab = c.text(self.SX0, self.SY - 34, "π", 38, ha="center")
        self.ends = [c.text(self.SX0, self.SY + 52, "global", 26, color=GLOBAL_DARK),
                     c.text(self.SX1, self.SY + 52, "local", 26, color=LOCAL_DARK, ha="right")]

    def update(self, t):
        a = ramp(t, 0.1, 0.8)
        set_alpha([self.photo, self.grid, self.note], a)
        self.grid.set_alpha(0.55 * a)
        aq = ramp(t, 0.8, 1.3)
        self.query.set_alpha(aq)
        set_alpha(self.leg[0], aq)
        # pi and the emphasis it gives each view once the slider is in
        pi = keyframes(t, self.PI_KEYS)
        k = ramp(t, 9.8, 10.4)
        w_emph = lerp(1.0, 0.35 + 0.65 * pi, k)
        g_emph = lerp(1.0, 0.35 + 0.65 * (1 - pi), k)
        # local window grows out of the query cell
        g = ramp(t, 3.5, 4.6)
        qx, qy, qs = self.qcell
        wx, wy, ws = self.win_box
        x, y, s = lerp(qx, wx, g), lerp(qy, wy, g), lerp(qs, ws, g)
        aw = ramp(t, 3.4, 3.8)
        self.win_fill.set_bounds(x, y, s, s)
        self.win_edge.set_bounds(x, y, s, s)
        self.win_fill.set_alpha(0.22 * aw * w_emph)
        self.win_edge.set_alpha(aw * w_emph)
        set_alpha(self.leg[1], ramp(t, 3.4, 4.0))
        # global keys ripple outwards from the query
        ad = np.array([ramp(t, 6.7 + 1.1 * r, 7.1 + 1.1 * r) for r in self.dist]) * g_emph
        self.dot_rgba[:, 3] = ad
        self.dot_edge[:, 3] = ad
        self.dots.set_facecolors(self.dot_rgba)
        self.dots.set_edgecolors(self.dot_edge)
        set_alpha(self.leg[2], ramp(t, 6.6, 7.2))
        # slider
        asl = ramp(t, 9.8, 10.4)
        self.knob.center = (lerp(self.SX0, self.SX1, pi), self.SY)
        self.knob_lab.set_x(lerp(self.SX0, self.SX1, pi))
        set_alpha([self.track, self.knob, self.knob_lab] + self.ends, asl)


def sf_rgb(hex_color):
    return rgb(hex_color) / 255.0


# ----------------------------------------------------------------------------- scene 3
class SceneCalib(Scene):
    title = "Calibration by emulation"
    dur = 20.5
    SQ, LX, RX, SY = 460, 100, 640, 240
    QR, QC = 15, 14  # the query of scene 2, in 32-cell schematic units
    cues = [(0.5, 3.5, "π is fitted once, at 1024²,", "where full attention still works."),
            (3.7, 6.8, "Shrink both views by 4 (and by 2)", "to mimic a 4× (2×) larger image."),
            (7.0, 10.0, "Each head's π is the best fit", "to full attention, in closed form."),
            (10.4, 15.0, "One π for each of 1368 heads,", "noise level and ratio."),
            (15.2, 19.8, "Heads differ: most lean local,", "a few lean global.")]
    PX0, PX1, PY0, PY1 = 1260, 1800, 430, 700  # fit plot area
    CELL, TX0, TY0, TGX, TGY = 6, 61, 250, 22, 58  # table cell, origin, gaps

    def build(self):
        c, d, SQ = self.c, self.d, self.SQ
        photo = d.fit("sheep", SQ, wash=0.30)
        A = []
        wbox = lambda x0: (x0 + (self.QC - 4) / 32 * SQ, self.SY + (self.QR - 4) / 32 * SQ, SQ / 4)  # noqa: E731
        # left: the target geometry (window = 1/4 of the side, global keys on a 64x64 grid)
        A += [c.image(photo, self.LX, self.SY), c.text(self.LX, self.SY - 24, "Target 4096²", 30)]
        x, y, s = wbox(self.LX)
        self.l_fill = c.rect(x, y, s, s, fc=LOCAL, z=4, alpha=0.22)
        A += [self.l_fill, c.rect(x, y, s, s, ec=LOCAL, lw=4, z=5)]
        g = (np.arange(64) * 4 + 2 + 0.5) / 256 * SQ
        gx, gy = np.meshgrid(g, g)
        self.l_dots = c.dots(self.LX + gx.ravel(), self.SY + gy.ravel(), 2.8, GLOBAL, edge="none", ew=0)
        A += [c.text(self.LX, self.SY + SQ + 40, "64-token window", 26, color=MID),
              c.text(self.LX, self.SY + SQ + 76, "64×64 global keys", 26, color=MID)]
        self.left = A
        # right: calibration image, both views shrink by 4
        self.r_img = c.image(photo, self.RX, self.SY)
        self.r_lab = c.text(self.RX, self.SY - 24, "Calibration 1024²", 30)
        self.r_fill = c.rect(0, 0, 1, 1, fc=LOCAL, z=4)
        self.r_edge = c.rect(0, 0, 1, 1, ec=LOCAL, lw=4, z=5)
        self.r_win0, self.r_win1 = (self.RX, self.SY, SQ), wbox(self.RX)
        i = np.arange(64)
        rx, ry = np.meshgrid((i + 0.5) / 64 * SQ, (i + 0.5) / 64 * SQ)
        keep = np.outer(i % 4 == 2, i % 4 == 2).ravel()
        self.r_keep = keep
        self.r_dense = c.dots(self.RX + rx.ravel()[~keep], self.SY + ry.ravel()[~keep], 3.2, GLOBAL, edge="none", ew=0)
        self.r_kept = c.dots(self.RX + rx.ravel()[keep], self.SY + ry.ravel()[keep], 3.2, GLOBAL, edge="white", ew=0)
        self.r_sub = [c.text(self.RX, self.SY + SQ + 40, "16-token window", 26, color=MID),
                      c.text(self.RX, self.SY + SQ + 76, "16×16 global keys", 26, color=MID)]
        # fit plot: one head's output error against full attention as a function of pi
        P = [c.text(self.PX0 - 50, self.PY0 - 66, "Error against full attention", 30),
             c.text(self.PX0 - 50, self.PY0 - 30, "one head, one noise level", 26, color=MID),
             c.line([self.PX0, self.PX0, self.PX1], [self.PY0 - 10, self.PY1, self.PY1], color=INK, lw=1.6),
             c.text(self.PX0, self.PY1 + 36, "0", 24, color=MID, ha="center"),
             c.text(self.PX1, self.PY1 + 36, "1", 24, color=MID, ha="center"),
             c.text(self.PX0, self.PY1 + 70, "global", 24, color=GLOBAL_DARK, ha="center"),
             c.text(self.PX1, self.PY1 + 70, "local", 24, color=LOCAL_DARK, ha="center"),
             c.text(self.PX0 - 22, (self.PY0 + self.PY1) / 2, "error", 24, color=MID, ha="center", va="center",
                    rotation=90)]
        self.plot_static = P
        ex = d.example
        top = ex["err"].max() * 1.08
        self.cx = self.PX0 + ex["pi"] * (self.PX1 - self.PX0)
        self.cy = self.PY1 - ex["err"] / top * (self.PY1 - self.PY0)
        self.curve = c.line(self.cx[:1], self.cy[:1], color=INK, lw=3.5, z=5)
        j = int(np.argmin(ex["err"]))
        mx, my = self.PX0 + ex["pistar"] * (self.PX1 - self.PX0), self.cy[j]
        self.opt = [c.line([mx, mx], [my, self.PY1], color=SCA, lw=2.2, z=4, linestyle=(0, (4, 3))),
                    c.dots([mx], [my], 20, SCA, ew=2.5, z=7),
                    c.text(mx, self.PY1 + 38, "π", 30, color=SCA, ha="center")]
        # the pi table: 13 populated noise bins, sigma high to low, 57 blocks x 24 heads each
        self.bins = [int(b) for b in np.argsort(-d.centers) if d.populated[b]]
        cw, ch = 57 * self.CELL, 24 * self.CELL
        self.empty = np.empty((ch, cw, 3), np.uint8)
        self.empty[:] = rgb(EMPTY).astype(np.uint8)
        self.full_maps, self.panels, self.tab_static = [], [], []
        for j, b in enumerate(self.bins):
            r, col = divmod(j, 5)
            x, y = self.TX0 + col * (cw + self.TGX), self.TY0 + r * (ch + self.TGY)
            self.full_maps.append(d.heat(d.bin_map(b), self.CELL))
            self.panels.append(c.image(self.empty.copy(), x, y, z=2))
            self.tab_static.append(c.text(x, y - 12, f"σ {d.centers[b]:.2f}", 24, color=MID))
        lx0 = self.TX0 + 3 * (cw + self.TGX) + 40
        lx1 = self.TX0 + 4 * (cw + self.TGX) + cw - 40
        ly = self.TY0 + 2 * (ch + self.TGY)
        bar = pi_rgb(np.linspace(0, 1, int(lx1 - lx0)))[None].repeat(22, axis=0)
        self.tab_static += [c.image(bar, lx0, ly + 40, z=2),
                            c.text((lx0 + lx1) / 2, ly + 28, "π", 30, ha="center"),
                            c.text(lx0, ly + 94, "global", 24, color=GLOBAL_DARK),
                            c.text(lx1, ly + 94, "local", 24, color=LOCAL_DARK, ha="right"),
                            c.text((lx0 + lx1) / 2, ly + 134, "57 blocks × 24 heads per panel,", 24, color=MID,
                                   ha="center"),
                            c.text((lx0 + lx1) / 2, ly + 166, "heads sorted within each block", 24, color=MID,
                                   ha="center")]

    def update(self, t):
        out = 1 - ramp(t, 9.9, 10.5)
        set_alpha(self.left, ramp(t, 0.2, 0.9) * out)
        self.l_fill.set_alpha(0.22 * ramp(t, 0.2, 0.9) * out)
        self.l_dots.set_alpha(0.6 * ramp(t, 0.2, 0.9) * out)
        ar = ramp(t, 0.9, 1.6) * out
        set_alpha([self.r_img, self.r_lab], ar)
        # both views appear at native size (window = whole image, every token a global key), then shrink
        av = ramp(t, 3.7, 4.3) * out
        g = ramp(t, 4.4, 5.8)
        (x0, y0, s0), (x1, y1, s1) = self.r_win0, self.r_win1
        x, y, s = lerp(x0, x1, g), lerp(y0, y1, g), lerp(s0, s1, g)
        self.r_fill.set_bounds(x, y, s, s)
        self.r_edge.set_bounds(x, y, s, s)
        self.r_fill.set_alpha(0.22 * av)
        self.r_edge.set_alpha(av)
        self.r_dense.set_alpha(0.6 * av * (1 - g))
        self.r_kept.set_alpha(lerp(0.6, 1.0, g) * av)
        dsz = lerp(2.8, 7.5, g)
        self.r_kept.set_sizes([pt(dsz) ** 2] * int(self.r_keep.sum()))
        self.r_kept.set_linewidths([lerp(0, 1.6, g) * 0.72])
        set_alpha(self.r_sub, ramp(t, 5.8, 6.4) * out)
        # the fit
        set_alpha(self.plot_static, ramp(t, 7.0, 7.6) * out)
        n = max(2, int(round(ramp(t, 7.4, 8.8) * len(self.cx))))
        self.curve.set_data(self.cx[:n], self.cy[:n])
        self.curve.set_alpha(ramp(t, 7.3, 7.5) * out)
        set_alpha(self.opt, ramp(t, 8.8, 9.3) * out)
        # the table fills in, noise bin by noise bin
        ab = ramp(t, 10.3, 10.9)
        set_alpha(self.tab_static, ab)
        for j, (im, full) in enumerate(zip(self.panels, self.full_maps)):
            im.set_alpha(ab)
            k = ramp(t, 11.0 + 0.5 * j, 11.45 + 0.5 * j)
            kp = int(round(k * full.shape[1]))
            arr = self.empty.copy()
            arr[:, :kp] = full[:, :kp]
            im.set_data(arr)


# ----------------------------------------------------------------------------- scene 4
class SceneOpen(Scene):
    title = "The opening phase"
    dur = 12.5
    X0, X1, R1, R2 = 560, 1760, 172, 228
    IMG, IY, IXL, IXR = 480, 410, 450, 990
    cues = [(0.5, 4.2, "Mixing windows in from step 1", "draws five Moai."),
            (4.4, 8.2, "SCA runs 8 of 28 steps globally,", "then shrinks its windows over 4 steps."),
            (8.4, 11.9, "The opening keeps one large statue;", "a small second Moai remains far off.")]

    def build(self):
        c, d = self.c, self.d
        xs = self.X0 + (1 - d.sigmas) * (self.X1 - self.X0)
        self.ticks = []
        for row, ry in enumerate((self.R1, self.R2)):
            arts = []
            for k, x in enumerate(xs):
                col = GLOBAL if (row == 1 and k < d.n_open) else (
                    WIDE if (row == 1 and k < d.n_open + d.n_wide) else LOCAL)
                arts.append(c.rect(x - 3, ry - 18, 6, 36, fc=col, z=3))
            self.ticks.append(arts)
        self.rowlab = [c.text(self.X0 - 34, self.R1 + 10, "Windows from step 1", 28, ha="right"),
                       c.text(self.X0 - 34, self.R2 + 10, "SCA", 28, ha="right", weight="bold")]
        ya = self.R2 + 40
        self.axis = [c.line([self.X0, self.X1], [ya, ya], color=RULE, lw=2),
                     c.text(self.X0, ya + 32, "1", 24, color=MID, ha="center"),
                     c.text((self.X0 + self.X1) / 2, ya + 32, "σ", 26, color=MID, ha="center"),
                     c.text(self.X1, ya + 32, "0", 24, color=MID, ha="center")]
        xo = (xs[0] + xs[d.n_open - 1]) / 2
        self.lab_open = c.text(xo, ya + 66, "global only", 24, color=GLOBAL_DARK, ha="center")
        self.lab_mix = c.text((xs[d.n_open + d.n_wide] + xs[-1]) / 2, ya + 66, "per-head mix, native windows", 24,
                              color=LOCAL_DARK, ha="center")
        self.lab_wide = c.text((xs[d.n_open] + xs[d.n_open + d.n_wide - 1]) / 2, ya + 98, "shrinking windows", 22,
                               color=LOCAL_DARK, ha="center")
        self.imL = [c.image(d.fit("moai_v0", self.IMG), self.IXL, self.IY),
                    c.text(self.IXL, self.IY - 18, "Windows from step 1", 28)]
        self.imR = [c.image(d.fit("moai_sca", self.IMG), self.IXR, self.IY),
                    c.text(self.IXR, self.IY - 18, "Global-only opening", 28, weight="bold")]
        s = self.IMG / 4096
        x, y, w, h = MOAI_SMALL
        self.small = c.rect(self.IXR + x * s, self.IY + y * s, w * s, h * s, ec=ZOOM, lw=3.5, z=5)

    def update(self, t):
        set_alpha(self.axis, ramp(t, 0.2, 0.8))
        self.rowlab[0].set_alpha(ramp(t, 0.4, 1.0))
        for k, r in enumerate(self.ticks[0]):
            r.set_alpha(ramp(t, 0.6 + 0.045 * k, 0.9 + 0.045 * k))
        self.lab_mix.set_alpha(ramp(t, 1.4, 2.0))
        set_alpha(self.imL, ramp(t, 1.6, 2.4))
        self.rowlab[1].set_alpha(ramp(t, 4.3, 4.9))
        for k, r in enumerate(self.ticks[1]):
            r.set_alpha(ramp(t, 4.5 + 0.045 * k, 4.8 + 0.045 * k))
        self.lab_open.set_alpha(ramp(t, 4.8, 5.4))
        self.lab_wide.set_alpha(ramp(t, 5.2, 5.8))
        set_alpha(self.imR, ramp(t, 5.4, 6.2))
        self.small.set_alpha(ramp(t, 8.6, 9.2))


# ----------------------------------------------------------------------------- scene 5
class SceneMixed(Scene):
    title = "The schedule at 4096²"
    dur = 16.0
    C, HX, HY = 20, 640, 196
    T1, D_OPEN, D_WIDE, D_MIX = 0.8, 0.36, 0.62, 0.22
    FADE = 10.4
    TX0, TX1, TY = 640, 1780, 806
    IM, IMX, IMY, INX, INY = 720, 150, 170, 1080, 200
    cues = [(0.5, 3.6, "Steps 1 to 8 skip the local branch:", "every head is global."),
            (3.8, 6.4, "Then the windows shrink, 128 to 80 tokens;", "each head's π is read at that ratio."),
            (6.6, 10.2, "The last 16 steps use native windows. Every", "window also sees 1,024 bridge tokens."),
            (10.7, 15.4, "The final 4096² image,", "with a 1:1 crop of its detail.")]

    def __init__(self, data):
        super().__init__(data)
        d = data
        n_nat = len(d.sigmas) - d.n_open - d.n_wide
        wide = [w for w in d.windows if w is not None and w != d.native]
        self.cues = [(0.5, 3.6, f"Steps 1 to {d.n_open} skip the local branch:", "every head is global."),
                     (3.8, 6.4, f"Then the windows shrink, {wide[0]} to {wide[-1]} tokens;",
                      "each head's π is read at that ratio."),
                     (6.6, 10.2, f"The last {n_nat} steps use native windows. Every",
                      f"window also sees {d.bridge:,} bridge tokens."),
                     (10.7, 15.4, "The final 4096² image,", "with a 1:1 crop of its detail.")]

    def build(self):
        c, d = self.c, self.d
        n = len(d.sigmas)
        dur = [self.D_OPEN if k < d.n_open else (self.D_WIDE if k < d.n_open + d.n_wide else self.D_MIX)
               for k in range(n)]
        self.t_step = self.T1 + np.cumsum([0.0] + dur)
        self.maps = [np.zeros((57, 24)) if k < d.n_open else d.pi_at(d.sigmas[k]) for k in range(n)]
        A = []
        self.heat = c.image(d.heat(self.maps[0], self.C), self.HX, self.HY, z=2)
        hw, hh = 57 * self.C, 24 * self.C
        self.open_lab = c.text(self.HX + hw / 2, self.HY + hh / 2 + 14, "global only", 40, color="white",
                               ha="center", weight="bold")
        A += [self.heat, c.text(self.HX + hw / 2, self.HY + hh + 38, "57 blocks", 24, color=MID, ha="center"),
              c.text(self.HX - 18, self.HY + hh / 2, "24 heads, sorted", 24, color=MID, ha="center",
                     va="center", rotation=90)]
        self.step_lab = c.text(MX, 300, "Step", 28, color=MID)
        self.num = c.text(MX + 180, 424, "1", 120, weight="bold", ha="right")
        self.of = c.text(MX + 196, 424, "/ 28", 44, color=MID)
        self.sig = c.text(MX, 500, "σ 1.000", 40)
        self.ph_mark = c.rect(MX, 556, 22, 22, fc=GLOBAL, z=4)
        self.ph_text = c.text(MX + 36, 578, "global only", 30)
        bx0, bx1, by = MX, MX + 440, 700
        bar = pi_rgb(np.linspace(0, 1, bx1 - bx0))[None].repeat(18, axis=0)
        A += [self.step_lab, self.num, self.of, self.sig, self.ph_mark, self.ph_text,
              c.image(bar, bx0, by, z=2), c.text((bx0 + bx1) / 2, by - 14, "π", 30, ha="center"),
              c.text(bx0, by + 52, "global", 24, color=GLOBAL_DARK),
              c.text(bx1, by + 52, "local", 24, color=LOCAL_DARK, ha="right")]
        xs = self.TX0 + (1 - d.sigmas) * (self.TX1 - self.TX0)
        self.tl = [c.rect(x - 2.5, self.TY - 16, 5, 32, fc=GLOBAL if k < d.n_open else (
                       WIDE if k < d.n_open + d.n_wide else LOCAL), z=3) for k, x in enumerate(xs)]
        self.tl_x = xs
        self.head = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, facecolor=INK, edgecolor="none", zorder=5)
        c.ax.add_patch(self.head)
        A += [self.head, c.line([self.TX0, self.TX1], [self.TY + 26, self.TY + 26], color=RULE, lw=2),
              c.text(self.TX0, self.TY + 58, "1", 24, color=MID, ha="center"),
              c.text((self.TX0 + self.TX1) / 2, self.TY + 58, "σ", 26, color=MID, ha="center"),
              c.text(self.TX1, self.TY + 58, "0", 24, color=MID, ha="center")]
        self.partA = A
        # the final image with its 1:1 inset
        s = self.IM / 4096
        x, y, w = GUITAR_CROP
        fx, fy, fw = self.IMX + x * s, self.IMY + y * s, w * s
        self.partB = [c.image(d.fit("guitar", self.IM), self.IMX, self.IMY, z=2)]
        self.frameB = [c.rect(fx, fy, fw, fw, ec=ZOOM, lw=3.5, z=5)]
        self.inset = [c.image(d.guitar_crop, self.INX, self.INY, z=2),
                      c.rect(self.INX, self.INY, w, w, ec=ZOOM, lw=4.5, z=5),
                      c.line([fx + fw, self.INX], [fy, self.INY], color=ZOOM, lw=2, z=4),
                      c.line([fx + fw, self.INX], [fy + fw, self.INY + w], color=ZOOM, lw=2, z=4),
                      c.text(self.INX, self.INY + w + 46, "1:1 crop", 28, color=MID)]

    def update(self, t):
        d = self.d
        k = int(np.clip(np.searchsorted(self.t_step, t, side="right") - 1, 0, len(d.sigmas) - 1))
        u = t - self.t_step[k]
        blend = 0.3 if k == d.n_open else 0.12
        m = self.maps[k] if k == 0 else lerp(self.maps[k - 1], self.maps[k], ease(u / blend))
        self.heat.set_data(d.heat(m, self.C))
        self.num.set_text(str(k + 1))
        self.sig.set_text(f"σ {d.sigmas[k]:.3f}")
        mixed = k >= d.n_open
        self.ph_mark.set_facecolor((WIDE if d.windows[k] != d.native else LOCAL) if mixed else GLOBAL)
        w = d.windows[k]
        self.ph_text.set_text("global only" if not mixed else (
            f"window {w}, ratio {d.side / w:.2f}".rstrip("0").rstrip(".") if w != d.native
            else f"native {w}-token window"))
        a = 1 - ramp(t, self.FADE, self.FADE + 0.5)
        set_alpha(self.partA, a)
        self.open_lab.set_alpha(a * (1 - ramp(t, self.t_step[d.n_open], self.t_step[d.n_open] + 0.3)))
        for j, r in enumerate(self.tl):
            r.set_alpha(a * (1.0 if j <= k else 0.22))
        x = self.tl_x[k]
        self.head.set_xy([[x - 9, self.TY - 34], [x + 9, self.TY - 34], [x, self.TY - 20]])
        set_alpha(self.partB, ramp(t, self.FADE + 0.4, self.FADE + 1.2))
        set_alpha(self.frameB, ramp(t, self.FADE + 1.1, self.FADE + 1.6))
        set_alpha(self.inset, ramp(t, self.FADE + 1.5, self.FADE + 2.3))


class SceneEnd(Scene):
    title, dur, band, cues = "", 5.5, False, []

    def build(self):
        c, sp = self.c, self.d.speed
        y = 520 if sp else 560
        self.arts = [c.text(W / 2, y - 112, "Scale-Covariant Attention", 32, color=SCA, weight="bold", ha="center"),
                     c.text(W / 2, y, "Fitted once at 1024². No training.", 64, weight="bold", ha="center")]
        if sp:
            line = f"{sp[0]} s per 4096² image, {sp[2]} faster than AReA ({sp[1]} s)"
            self.arts.append(c.text(W / 2, y + 84, line, 36, color=MID, ha="center"))

    def update(self, t):
        for i, a in enumerate(self.arts):
            a.set_alpha(ramp(t, 0.2 + 0.35 * i, 0.8 + 0.35 * i))


# ----------------------------------------------------------------------------- captions and timeline
def caption_coverage(lines):
    """Antialiased coverage (0..1) of one caption, rendered once, for the band below the scenes."""
    bh = H - BAND_Y
    fig = plt.figure(figsize=(W / DPI, bh / DPI), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(bh, 0)
    ax.set_axis_off()
    bases = [bh / 2 + 14] if len(lines) == 1 else [bh / 2 - 12, bh / 2 + 40]
    for yb, s in zip(bases, lines):
        ax.text(W / 2, yb, s, fontsize=pt(40), color="black", ha="center", va="baseline")
    fig.canvas.draw()
    g = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(np.float32)
    plt.close(fig)
    return 1.0 - g / 255.0


class Timeline:
    def __init__(self, scenes):
        self.scenes = scenes
        self.starts, t = [], 0.0
        for s in scenes:
            self.starts.append(t)
            t += s.dur - XF
        self.total = self.starts[-1] + scenes[-1].dur
        self.cues = [(round(st + c[0], 3), round(st + c[1], 3), list(c[2:])) for st, s in zip(self.starts, scenes)
                     for c in s.cues]
        for _, _, lines in self.cues:
            assert all(len(x.split()) <= 8 for x in lines) and len(lines) <= 2, lines
        self.cov = [caption_coverage(lines) for _, _, lines in self.cues]
        self.ink = rgb(INK)

    def frame(self, T, release=False):
        k = max(i for i, s in enumerate(self.starts) if s <= T)
        img = self.scenes[k].render(T - self.starts[k]).astype(np.float32)
        if k > 0 and T < self.starts[k] + XF:
            a = ease((T - self.starts[k]) / XF)
            prev = self.scenes[k - 1].render(T - self.starts[k - 1]).astype(np.float32)
            img = prev * (1 - a) + img * a
        if release:
            for i in range(k - (1 if T < self.starts[k] + XF else 0)):
                self.scenes[i].release()
        band = img[BAND_Y:]
        for (t0, t1, _), cov in zip(self.cues, self.cov):
            if t0 <= T <= t1:
                a = min(ramp(T, t0, t0 + 0.25), 1 - ramp(T, t1 - 0.25, t1))
                w = (a * cov)[..., None]
                band[:] = band * (1 - w) + self.ink * w
        if T < 0.5:
            img = img * ease(T / 0.5) + 255.0 * (1 - ease(T / 0.5))
        return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def stamp(t, sep):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_subtitles(cues, srt, vtt):
    with open(srt, "w") as fh:
        for i, (t0, t1, lines) in enumerate(cues, 1):
            fh.write(f"{i}\n{stamp(t0, ',')} --> {stamp(t1, ',')}\n" + "\n".join(lines) + "\n\n")
    with open(vtt, "w") as fh:
        fh.write("WEBVTT\n\n")
        for i, (t0, t1, lines) in enumerate(cues, 1):
            fh.write(f"{i}\n{stamp(t0, '.')} --> {stamp(t1, '.')}\n" + "\n".join(lines) + "\n\n")


def encode(tl, mp4, fps, crf, poster_t, poster_path):
    n = int(round(tl.total * fps))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{FFMPEG}/lib:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [str(FFMPEG / "bin" / "ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-filter_threads", "4",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+bicubic,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-tune", "animation", "-crf", str(crf), "-profile:v", "high",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", "-threads", "4", str(mp4)]
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, env=env)
    pf = int(round(poster_t * fps))
    t0 = time.time()
    try:
        for f in range(n):
            fr = tl.frame(f / fps, release=True)
            proc.stdin.write(fr.tobytes())
            if f == pf:
                Image.fromarray(fr).resize((1280, 720), Image.LANCZOS).save(poster_path, "WEBP", quality=90, method=6)
            if f % 150 == 0:
                print(f"frame {f}/{n}  {time.time() - t0:.0f} s", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    print(f"encoded {n} frames ({n / fps:.2f} s) in {time.time() - t0:.0f} s -> {mp4}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--numbers", default=str(REPO / "web" / "build" / "numbers.json"),
                    help="JSON with speed.sca_s, speed.area_s, speed.ratio (end card); missing -> no speed line")
    ap.add_argument("--out", default=str(REPO / "web" / "assets" / "video" / "method"))
    ap.add_argument("--work", default=str(WORK), help="scratch dir for the image cache and previews")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--poster-scene", type=int, default=2, help="1-based scene of the poster frame")
    ap.add_argument("--poster-at", type=float, default=13.4, help="seconds into that scene")
    ap.add_argument("--preview", default="", help="comma-separated seconds; write PNG frames and stop")
    ap.add_argument("--rebuild-cache", action="store_true")
    a = ap.parse_args()
    family = setup_fonts()
    work = Path(a.work)
    build_cache(work, a.rebuild_cache)
    d = Data(work, a.numbers)
    tl = Timeline([SceneFail(d), SceneHead(d), SceneCalib(d), SceneOpen(d), SceneMixed(d), SceneEnd(d)])
    print(f"font {family}; total {tl.total:.2f} s; scene starts " + ", ".join(f"{s:.1f}" for s in tl.starts))
    if a.preview:
        pdir = work / "preview"
        pdir.mkdir(parents=True, exist_ok=True)
        for T in (float(x) for x in a.preview.split(",")):
            p = pdir / f"t{T:06.2f}.png"
            Image.fromarray(tl.frame(T)).save(p)
            print(p)
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_subtitles(tl.cues, out / "sca_method.srt", out / "sca_method.vtt")
    poster_t = tl.starts[a.poster_scene - 1] + a.poster_at
    encode(tl, out / "sca_method.mp4", a.fps, a.crf, poster_t, out / "sca_method_poster.webp")
    for (t0, t1, lines) in tl.cues:
        print(f"{t0:6.2f} {t1:6.2f}  " + " / ".join(lines))


if __name__ == "__main__":
    main()
