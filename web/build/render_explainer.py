#!/usr/bin/env python
"""Render the SCA method explainer video and the heads loop of the project page (CPU only, real data).

Outputs
  web/assets/video/method/sca_explainer.mp4   1920x1080, 30 fps, H.264 (libx264, yuv420p, BT.709, +faststart),
                                              silent, short captions drawn in the frame
  web/assets/video/method/sca_explainer.vtt, sca_explainer.srt (the same cues), sca_explainer_poster.webp (1280 px)
  web/assets/video/teaser/heads_loop.mp4      960x960, 8 s: the 1,368 per-head weight curves with a noise-level
                                              cursor that sweeps forth and back on a cosine, so the last frame leads
                                              into the first; heads_loop.webp poster
Scenes (start times are printed by every run)
  1 Full attention fails at 4096^2: token count, attention cost and offsets from the paper's arithmetic; real
    FLUX.1-dev full-attention outputs (stripes, flat field).
  2 Every head mixes near and far: the softmax split O_full = m O_near + (1 - m) O_far (paper Lemma, Section 2.1);
    m of three real heads at 1024^2 (results/heads_view/rings.csv); then SCA's two views at 4096^2, the native
    window (with text and bridge tokens, AReA's rule) and the 64x64 grid of block-centre keys, mixed by pi_h.
  3 Reach is set by the head: all 1,368 curves pi_h(sigma) of results/heads_view/concept_curves.csv (ratio 4),
    coloured by each head's mean weight, the median, AReA's single switch; the variance split of numbers.json
    (heads.var.*, from results/heads_view/variance_split.csv).
  4 Reach emulation at 1024^2: both views shrunk by r (r = 4, then 2), beside the target geometry at 4096^2
    (2048^2); full attention and both views see the same queries, keys and values.
  5 Fit each head in closed form: A, B, E, J(pi) = E - 2 pi B + pi^2 A, pi* = clip(B/A, 0, 1); J of the three
    heads of scene 2 from the stored sums (results/evidence/calib/k1_1024.npz, `stats`), then the reach table
    (r = 2 and 4) filling bin by bin.
  6 Progressive sampling at 4096^2 (HRDiT's released three-stage schedule): the three stages on a noise-level
    timeline from the final image's sidecar, the 57x24 pi map each SCA stage reads (checked against the sidecar's
    pi_trace), the final progressive SCA image and a native 1:1 crop. With --trace DIR the decoded previews of a real
    progressive trace play during the sweep (layout below); without it no stage image is shown.
  7 End card: seconds per 4096^2 image on one GH200 (prog.clean.*) and the seed and HRGen-1K line (prog.seeds.*,
    prog.later.hrgen.*).
Every number the video prints that numbers.json holds is read from it (display strings); the rest is arithmetic on
the paper's geometry (16 px per token, 64 tokens per side at 1024^2) or read from the data files above.

--trace DIR (scene 6): DIR/trace.json with "steps": [{"stage": 0|1|2 (or "res": "1024x1024" ...), "sigma": s,
"preview": "steps/x0_NN.webp"}, ...] in sampling order (30 + 17 + 10 entries; previews are decoded x0 images, any
square size; with fewer previews than steps, steps map to previews in order), optional "prompt_id" (a warning is
printed unless it is the final image's prompt) and "synthetic": true (refused for outputs inside web/); or, without
trace.json, DIR/stage0/, DIR/stage1/, DIR/stage2/ holding one image per step, sorted by name. The scene ends on the
final image of SOURCES["final"] and its crop either way.

Run from the repository root (CPU only, one python process per --jobs worker, ffmpeg at 4 threads):
  nice -n 19 /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/render_explainer.py --jobs 10
  --preview 3,20,40 writes those frames (seconds) as PNG to <work>/preview and skips the encode;
  --preview-loop 0,2 does the same for the loop; --only explainer|loop limits the encode.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import multiprocessing as mp
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper" / "figures" / "src"))
import scafig as sf  # noqa: E402  (paper palette and the pi colormap)

# ----------------------------------------------------------------------------- inputs
OUT_ROOT = Path("/scratch/09032/saini_2/sca/out")
TABLES = [REPO / "results" / "evidence" / "calib" / "k1_1024.npz", Path("/scratch/09032/saini_2/sca/calib/k1_1024.npz")]
CURVES = REPO / "results" / "heads_view" / "concept_curves.csv"
RINGS = REPO / "results" / "heads_view" / "rings.csv"
VARSPLIT = REPO / "results" / "heads_view" / "variance_split.csv"
NUMBERS = REPO / "web" / "build" / "numbers.json"
WORK = Path("/scratch/09032/saini_2/sca/web_assets/explainer")
FFMPEG = Path("/work/09032/saini_2/vista/library/ffmpeg-7.1-shared")
SOURCES = {  # name -> (glob under OUT_ROOT, prompt id); FLUX.1-dev, 4096^2, seed 0
    "stripes": ("main/full", "6648013380537069782"),     # full attention: vertical stripes
    "flat": ("main/full", "1638867997226496674"),        # full attention: flat field
    "final": ("dupfix_prog/sca_v2_prog", "minus3112502764021036396"),  # progressive SCA: waterfall, stone bridge
}
FINAL_CROP = (1560, 540, 512)        # native x0, y0, side of the 1:1 crop (stone arch of the bridge)
SORT_BIN = 8                          # heads are sorted within each block by their ratio-4 weight in this bin

# ----------------------------------------------------------------------------- style
W, H, DPI, FPS = 1920, 1080, 100, 30
MX, TITLE_Y, BAND_Y = 96, 100, 968
XF = 0.6                              # cross-fade between scenes, seconds
CAP_PX = 38
INK, MID, RULE, GRIDC = sf.INK, sf.MID, sf.RULE, sf.GRID
GLOBAL, GLOBAL_LIGHT, LOCAL, LOCAL_LIGHT = sf.GLOBAL, sf.GLOBAL_LIGHT, sf.LOCAL, sf.LOCAL_LIGHT
SCA, SCA_LIGHT, ZOOM = sf.SCA, sf.SCA_LIGHT, sf.ZOOM
FULL = sf.METHOD_COLOR["full"]
GRIDLINE = sf.GRIDLINE
LOCAL_TEXT, GLOBAL_TEXT = "#A85F12", "#245A96"   # darker amber and blue for text (paper figures)
BAND, FAINT, EMPTY, CANVAS = "#F4F5F7", "#F8F9FA", "#ECEEF1", "#E4E8EC"
MASS_CMAP = LinearSegmentedColormap.from_list("mass", ["#FFFBF5", LOCAL_LIGHT, LOCAL, "#9C5512", "#4A2606"])
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
    plt.rcParams.update({"font.family": family, "text.antialiased": True, "mathtext.fontset": "cm",
                         "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt"})
    return family


def pt(px):
    """Font size in points for a size in pixels at DPI 100."""
    return px * 72.0 / DPI


def rgb(color):
    """Colour -> float32 RGB in 0..255."""
    return np.array(to_rgb(color), dtype=np.float32) * 255.0


def rgba(color, a=1.0):
    return np.r_[np.array(to_rgb(color)), a]


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def ramp(t, t0, t1):
    return ease((t - t0) / (t1 - t0))


def vis(t, t_in, t_out=None, d=0.45):
    """Opacity that eases in over d seconds from t_in and out over d seconds before t_out."""
    a = ramp(t, t_in, t_in + d)
    if t_out is not None:
        a = min(a, 1.0 - ramp(t, t_out - d, t_out))
    return a


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


def r2_points(n, seed):
    """n points in [0, 1)^2 from the additive R2 sequence (plastic number), offset by seed: an even spread."""
    g = 1.32471795724474602596
    a1, a2 = 1.0 / g, 1.0 / (g * g)
    k = np.arange(n) + 1 + seed * 7919
    return np.mod(0.5 + a1 * k, 1.0), np.mod(0.5 + a2 * k, 1.0)


# ----------------------------------------------------------------------------- numbers and data
class Numbers:
    """numbers.json: key -> {"value", "display", ...}; the video prints the display strings."""

    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text())

    def __call__(self, key):
        v = self.d[key]
        s = v.get("display") if isinstance(v, dict) else v
        if s is None or str(s).strip() in ("", "pending"):
            raise ValueError(f"{self.path}: {key} is {s!r}")
        return str(s)

    def val(self, key):
        v = self.d[key]
        return v["value"] if isinstance(v, dict) else v


def find_png(sub, pid):
    hits = sorted(glob.glob(str(OUT_ROOT / sub / "4096x4096" / f"a4k_eval_4096_{pid}_*_seed0.png")))
    if len(hits) != 1:
        raise FileNotFoundError(f"{sub} {pid}: {hits}")
    return Path(hits[0])


def cache_files(name):
    if name == "final":
        return ["final_2048.png", "final_crop_%d_%d_%d.png" % FINAL_CROP]
    return [f"{name}_1024.png"]


def build_cache(work, rebuild=False):
    """Decode each 4096^2 PNG at most once and keep the small versions the scenes need (manifest with SHA256)."""
    cdir = work / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    mpath = cdir / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() and not rebuild else {}
    changed = False
    for name, (sub, pid) in SOURCES.items():
        png = find_png(sub, pid)
        files = cache_files(name)
        rec = man.get(name, {})
        if not rebuild and rec.get("png") == str(png) and rec.get("files") == files and \
                all((cdir / f).exists() for f in files):
            continue
        t0 = time.time()
        im = Image.open(png).convert("RGB")
        assert im.size == (4096, 4096), (png, im.size)
        if name == "final":
            im.resize((2048, 2048), Image.LANCZOS).save(cdir / files[0])
            x, y, s = FINAL_CROP
            im.crop((x, y, x + s, y + s)).save(cdir / files[1])
        else:
            im.resize((1024, 1024), Image.LANCZOS).save(cdir / files[0])
        man[name] = {"dir": sub, "prompt_id": pid, "png": str(png), "sha256": sha256(png), "files": files}
        changed = True
        print(f"cache {name}: {png.name} ({time.time() - t0:.1f} s)", flush=True)
    if changed or not mpath.exists():
        tmp = mpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(man, indent=1) + "\n")
        tmp.replace(mpath)
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


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class Trace:
    """Decoded per-step previews of one progressive run (scene 6, --trace DIR); see the module docstring."""

    def __init__(self, path, stage_steps):
        self.path = Path(path)
        self.meta, stages = {}, [[] for _ in stage_steps]
        tj = self.path / "trace.json"
        if tj.exists():
            self.meta = json.loads(tj.read_text())
            res_stage = {"1024x1024": 0, "2048x2048": 1, "4096x4096": 2}
            steps = self.meta.get("steps", [])
            bounds = np.cumsum([0] + list(stage_steps))
            for i, st in enumerate(steps):
                if "stage" in st:
                    s = int(st["stage"])
                elif "res" in st:
                    s = res_stage[str(st["res"])]
                else:
                    s = int(np.searchsorted(bounds, i, side="right") - 1)
                stages[min(s, len(stages) - 1)].append(self.path / st["preview"])
        else:
            for s in range(len(stage_steps)):
                files = sorted(glob.glob(str(self.path / f"stage{s}" / "*.webp")) +
                               glob.glob(str(self.path / f"stage{s}" / "*.png")))
                stages[s] = [Path(f) for f in files]
        missing = [p for st in stages for p in st if not p.exists()]
        if missing or any(len(st) == 0 for st in stages):
            raise FileNotFoundError(f"trace {self.path}: per-stage previews {[len(s) for s in stages]}, "
                                    f"missing {missing[:3]}")
        for s, (st, n) in enumerate(zip(stages, stage_steps)):
            if len(st) != n:
                print(f"trace: stage {s} has {len(st)} previews for {n} steps; steps map to previews in order",
                      flush=True)
        self.stages, self.steps = stages, list(stage_steps)
        self.synthetic = bool(self.meta.get("synthetic", False))
        pid = str(self.meta.get("prompt_id", ""))
        if pid and not pid.endswith(SOURCES["final"][1]):
            print(f"trace: prompt {pid} is not the final image's prompt {SOURCES['final'][1]}", flush=True)
        self._cache = {}

    def image(self, stage, k, size):
        """Preview of step k (0-based within its stage) as a size x size RGB array."""
        st = self.stages[stage]
        i = min(int(k * len(st) / self.steps[stage]), len(st) - 1)
        key = (stage, i, size)
        if key not in self._cache:
            if len(self._cache) > 24:
                self._cache.pop(next(iter(self._cache)))
            im = Image.open(st[i]).convert("RGB")
            self._cache[key] = np.asarray(im.resize((size, size), Image.LANCZOS))
        return self._cache[key]


class Data:
    def __init__(self, work, numbers, trace=None, build=True):
        N = self.N = Numbers(numbers)
        # geometry: 16 px per token; 64 x 64 tokens at 1024^2 (the native grid and the native window)
        self.r0 = N("calib.res")
        self.n0 = int(N.val("calib.res")) // 16
        self.G = int(N.val("config.grid_side_4096"))
        self.win = int(N.val("config.window"))
        assert self.win == self.n0 == int(N.val("prog.cfg.window")), "window != native grid side"
        self.block = int(N.val("config.native_ratio"))                 # 4: one key per 4 x 4 block at 4096^2
        assert self.block * self.n0 == self.G
        self.tok_x = (self.G // self.n0) ** 2                          # 16
        self.tokens_t = N("latency.tokens.4096")                        # 65,536
        assert self.tok_x * self.n0 ** 2 == int(N.val("latency.tokens.4096"))
        self.cost_x = N("config.cost_x_4096")                           # 256
        assert int(N.val("config.cost_x_4096")) == self.tok_x ** 2
        self.blocks, self.heads = int(N.val("config.blocks")), int(N.val("config.heads_per_block"))
        self.n_heads = N("calib.n_heads_c")                             # 1,368
        assert self.blocks * self.heads == int(N.val("calib.n_heads")) == int(N.val("heads.var.n_heads"))
        self.bridge = N("prog.cfg.bridge_tokens")                       # 1,024
        assert N("config.bridge_tokens") == self.bridge
        self.gate = N("config.gate_sigma")                              # 0.89
        self.n_open = N("config.opening_steps")                         # 8
        self.radius = N("heads.var.radius")                             # 16
        self.n_prompts = N("calib.n_prompts")                           # 32
        self.split = N("calib.split_half.s2")                           # 0.989
        self.var = {k: N(f"heads.var.{k}") for k in ("head", "noise", "avg", "between_blocks", "within_block")}
        self.check_variance()
        self.load_tables()
        self.load_curves()
        self.load_rings()
        self.load_prog()
        self.load_images(work, build)
        self.trace = Trace(trace, self.stage_steps) if trace else None
        self.end = dict(prog=N("prog.clean.prog"), hrdit=N("prog.clean.hrdit"), area=N("prog.clean.area"),
                        v=[float(N.val(k)) for k in ("prog.clean.prog", "prog.clean.hrdit", "prog.clean.area")],
                        all_word=N("prog.seeds.all_word"), n_word=N("prog.seeds.n_word"),
                        hrgen=N("prog.later.hrgen.n_better"))
        assert N.val("prog.later.hrgen.n_better") == N.val("prog.later.hrgen.n_total"), "HRGen-1K not all better"
        assert N.val("prog.later.hrgen.n_worse") == 0 and N.val("prog.later.hrgen.n_unresolved") == 0
        assert self.end["all_word"].split()[-1] == {10: "ten"}[int(N.val("prog.later.hrgen.n_total"))]

    def check_variance(self):
        rows = [r for r in read_csv(VARSPLIT) if r["checkpoint"] == "flux1_dev" and r["res"] == "1024x1024" and
                r["sigma_range"] == "post_gate" and r["quantity"] == "image_share" and r["radius"] == self.radius]
        assert len(rows) == 1, rows
        r = rows[0]
        got = {"between_blocks": 100 * float(r["layer"]), "within_block": 100 * float(r["head_within_layer"]),
               "noise": 100 * float(r["noise_within_head"]), "avg": 100 * float(r["head_avg_curve"])}
        got["head"] = got["between_blocks"] + got["within_block"]
        for k, v in got.items():
            assert abs(v - float(self.N.val(f"heads.var.{k}"))) < 0.01, (k, v)

    def load_tables(self):
        path = next(p for p in TABLES if p.exists())
        z = np.load(path)
        self.table_path, self.table_sha = path, sha256(path)
        scales = [int(s) for s in z["scales"]]
        self.tab = {s: z[f"s{s}_pi_head"].astype(np.float64) for s in (2, 4)}
        self.stats = {s: z["stats"][scales.index(s)] for s in (2, 4)}   # [A, B, E, E_fl, F2, count]
        self.centers = z["sigma_centers"].astype(np.float64)
        self.populated = z["populated"].astype(bool)
        assert int(z["n_prompts"]) == int(self.N.val("calib.n_prompts"))
        for s in (2, 4):
            A, B = self.stats[s][0], self.stats[s][1]
            pi = np.where(A > 0, np.clip(B / np.maximum(A, 1e-300), 0, 1), 1.0)
            assert np.abs(pi - self.tab[s])[..., self.populated].max() < 1e-12, "table != clip(B / A, 0, 1)"
        self.order = np.argsort(-self.tab[4][:, :, SORT_BIN], axis=1, kind="stable")   # per block, high pi first
        self.bins = [int(b) for b in np.argsort(-self.centers) if self.populated[b]]  # sigma high to low

    def heat(self, p, c):
        """[57, 24] pi map -> RGB, blocks along x, heads (sorted within each block, highest pi on top) along y."""
        s = np.take_along_axis(p, self.order, axis=1)                  # [57, 24]
        return cells(pi_rgb(s.T), c)

    def load_curves(self):
        a = np.loadtxt(CURVES, delimiter=",", skiprows=1)              # layer, head, sigma, pi, mean_after_gate
        nb, nh = self.blocks, self.heads
        a = a.reshape(nb, nh, -1, 5)
        assert np.all(a[..., 0] == np.arange(nb)[:, None, None]) and np.all(a[..., 1] == np.arange(nh)[None, :, None])
        self.sig = a[0, 0, :, 2]
        assert np.all(a[..., 2] == self.sig)
        self.Y = a[..., 3].reshape(nb * nh, -1)
        self.mean = a[..., 0, 4].reshape(-1)
        self.block_of = np.repeat(np.arange(nb), nh)
        assert np.allclose(self.Y.mean(axis=1), self.mean, atol=1e-5)
        assert abs(self.sig[0] - float(self.N.val("config.gate_sigma"))) < 1e-9
        self.median = np.median(self.Y, axis=0)

    def load_rings(self):
        self.rings = {}
        for r in read_csv(RINGS):
            d = self.rings.setdefault(r["role"], dict(block=int(r["block"]), head=int(r["head"]), bin=int(r["bin"]),
                                                      rings=[]))
            d["rings"].append((int(r["cheb_from"]), int(r["cheb_to"]), int(r["tokens"]), float(r["share"])))
        for d in self.rings.values():
            d["m"] = sum(x[3] for x in d["rings"])                        # share of image attention within 16
            assert d["rings"][-1][1] == int(self.radius)
        b = {d["bin"] for d in self.rings.values()}
        assert len(b) == 1
        self.ring_bin = b.pop()
        self.ring_sigma = self.centers[self.ring_bin]

    def load_prog(self):
        N = self.N
        png = find_png(*SOURCES["final"])
        meta = json.loads(png.with_suffix(".json").read_text())
        cfg = meta["config"]
        self.prog_meta = meta
        self.stage_sigmas = [np.array(s, dtype=np.float64) for s in meta["stage_sigmas"]]
        self.stage_steps = [len(s) for s in self.stage_sigmas]
        self.stage_res = [s["res"].split("x")[0] for s in meta["stage_shapes"]]
        self.stage_ratio = [None] + [int(s["pi_scale"]) for s in cfg["sca"]["stages"]]
        assert self.stage_steps == [int(N.val("prog.cfg.base_steps")), int(N.val("prog.cfg.steps_2048")),
                                    int(N.val("prog.cfg.steps_4096"))]
        assert sum(self.stage_steps) == int(N.val("prog.cfg.total_steps"))
        assert self.stage_ratio[1:] == [int(N.val("prog.cfg.ratio_2048")), int(N.val("prog.cfg.ratio_4096"))]
        assert f"{self.stage_sigmas[1][0]:.2f}" == N("prog.cfg.sigma_2048")
        assert f"{self.stage_sigmas[2][0]:.2f}" == N("prog.cfg.sigma_4096")
        assert self.stage_res == ["1024", "2048", "4096"]
        assert meta["config"]["sca"]["pi_table_sha256"] == self.table_sha, "table differs from the run's table"
        err = 0.0
        for tr in meta["pi_trace"]:
            err = max(err, abs(self.pi_at(tr["stage"], tr["sigma"]).mean() - tr["pi_mean"]))
        assert err < 1e-9, err
        self.s2048, self.s4096 = N("prog.cfg.sigma_2048"), N("prog.cfg.sigma_4096")
        self.r2048, self.r4096 = N("prog.cfg.ratio_2048"), N("prog.cfg.ratio_4096")
        self.steps_c = [N("prog.cfg.base_steps"), N("prog.cfg.steps_2048"), N("prog.cfg.steps_4096")]
        self.total_c = N("prog.cfg.total_steps")

    def pi_at(self, stage, sigma):
        """The [57, 24] weights a progressive SCA stage reads: its ratio's table, linear in sigma."""
        return interp_pi(self.tab[self.stage_ratio[stage]], self.centers, sigma)

    def load_images(self, work, build=True):
        cdir = work / "cache"
        man = build_cache(work) if build else json.loads((cdir / "manifest.json").read_text())
        self.manifest = man
        self.img = {k: Image.open(cdir / man[k]["files"][0]).convert("RGB") for k in man}
        self.final_crop = np.asarray(Image.open(cdir / man["final"]["files"][1]).convert("RGB"))

    def fit(self, name, size, wash=0.0):
        a = np.asarray(self.img[name].resize((size, size), Image.LANCZOS)).astype(np.float32)
        return (a * (1 - wash) + 255 * wash + 0.5).astype(np.uint8)

    def rel_err(self, s, block, head, b, ps):
        A, B, E, F = (self.stats[s][i][block, head, b] for i in (0, 1, 2, 4))
        return (E - 2 * ps * B + ps * ps * A) / F, float(np.clip(B / A, 0, 1))


# ----------------------------------------------------------------------------- drawing
class Layer:
    """An axes over a pixel rectangle of the frame; data coordinates (xlim, ylim) with y growing downwards."""

    def __init__(self, fig, rect, xlim=None, ylim=None, z=0, size=(W, H)):
        self.fw, self.fh = size
        x, y, w, h = rect
        self.ax = fig.add_axes(self._pos(rect))
        self.ax.set_xlim(*(xlim or (0, w)))
        self.ax.set_ylim(*(ylim or (h, 0)))
        self.ax.set_axis_off()
        self.ax.set_autoscale_on(False)
        self.ax.set_zorder(z)
        self.ax.patch.set_alpha(0.0)
        self.box = rect

    def _pos(self, rect):
        x, y, w, h = rect
        return [x / self.fw, 1 - (y + h) / self.fh, w / self.fw, h / self.fh]

    def place(self, rect):
        self.box = rect
        self.ax.set_position(self._pos(rect))

    def text(self, x, y, s, size, color=INK, weight="normal", ha="left", va="baseline", z=8, alpha=1.0, **kw):
        return self.ax.text(x, y, s, fontsize=pt(size), color=color, fontweight=weight, ha=ha, va=va, zorder=z,
                            alpha=alpha, **kw)

    def image(self, arr, x, y, w=None, h=None, z=1, interp="nearest"):
        hh, ww = arr.shape[:2]
        w = ww if w is None else w
        h = hh if h is None else h
        return self.ax.imshow(arr, extent=(x, x + w, y + h, y), interpolation=interp, aspect="auto", zorder=z,
                              resample=False)

    def rect(self, x, y, w, h, fc="none", ec="none", lw=0.0, z=2, alpha=1.0, **kw):
        p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw * 0.72, zorder=z, alpha=alpha, **kw)
        self.ax.add_patch(p)
        return p

    def rbox(self, x, y, w, h, fc="none", ec="none", lw=0.0, z=2, pad=0.0, r=10, alpha=1.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={r}", facecolor=fc, edgecolor=ec,
                           linewidth=lw * 0.72, zorder=z, alpha=alpha, mutation_aspect=1)
        self.ax.add_patch(p)
        return p

    def line(self, xs, ys, color=INK, lw=1.0, z=3, **kw):
        ln = Line2D(xs, ys, color=color, linewidth=lw * 0.72, zorder=z, **kw)
        self.ax.add_line(ln)
        return ln

    def lines(self, segs, colors, lw=1.0, z=2, alpha=None):
        lc = LineCollection(segs, colors=colors, linewidths=lw * 0.72, zorder=z, alpha=alpha)
        self.ax.add_collection(lc)
        return lc

    def dots(self, xs, ys, d, color, edge="none", ew=0.0, z=6):
        n = len(xs)
        fc = np.tile(rgba(color), (n, 1)) if isinstance(color, str) else np.asarray(color, dtype=float)
        ec = np.tile(rgba(edge), (n, 1)) if edge != "none" else "none"
        return self.ax.scatter(xs, ys, s=pt(d) ** 2, c=fc, edgecolors=ec, linewidths=ew * 0.72, zorder=z)

    def arrow(self, p0, p1, color=INK, lw=2.0, style="-|>", head=16, z=5, rad=0.0, shrink=0.0, **kw):
        a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=head, color=color, linewidth=lw * 0.72,
                            zorder=z, connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink, **kw)
        self.ax.add_patch(a)
        return a


class Canvas(Layer):
    """A frame-sized figure; the base layer is in frame pixels."""

    def __init__(self, band=True, size=(W, H)):
        self.fig = plt.figure(figsize=(size[0] / DPI, size[1] / DPI), dpi=DPI, facecolor="white")
        super().__init__(self.fig, (0, 0, size[0], size[1]), z=0, size=size)
        if band:
            self.rect(0, BAND_Y, size[0], size[1] - BAND_Y, fc=BAND, z=0)
        self.top = Layer(self.fig, (0, 0, size[0], size[1]), z=10, size=size)   # above every sub-layer

    def layer(self, rect, xlim=None, ylim=None, z=1):
        return Layer(self.fig, rect, xlim, ylim, z, size=(self.fw, self.fh))

    def grab(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3]

    def close(self):
        plt.close(self.fig)


class Group:
    """Artists that fade together; each keeps its own base opacity."""

    def __init__(self, *arts):
        self.items = []
        for a in arts:
            self.add(a)

    def add(self, art, base=1.0):
        if isinstance(art, (list, tuple)):
            for a in art:
                self.add(a, base)
        elif isinstance(art, Group):
            self.items += art.items
        else:
            self.items.append((art, base))
        return art

    def __call__(self, a):
        for art, b in self.items:
            art.set_alpha(a * b)


def set_dot_alpha(coll, base_rgba, a, edge_rgba=None):
    """Per-point opacity for a scatter: base_rgba [n, 4], a scalar or [n]."""
    fc = base_rgba.copy()
    fc[:, 3] = base_rgba[:, 3] * a
    coll.set_facecolors(fc)
    if edge_rgba is not None:
        ec = edge_rgba.copy()
        ec[:, 3] = edge_rgba[:, 3] * a
        coll.set_edgecolors(ec)


def colorbar(c, x, y, w, h=16, labels=True, size=26, z=3):
    """Horizontal PI_CMAP bar with 'global' and 'local' at its ends; returns a Group."""
    g = Group(c.image(pi_rgb(np.linspace(0, 1, int(w)))[None].repeat(int(h), axis=0), x, y, z=z))
    if labels:
        g.add(c.text(x, y + h + size + 6, "0 global", size, color=GLOBAL_TEXT, z=z))
        g.add(c.text(x + w, y + h + size + 6, "local 1", size, color=LOCAL_TEXT, ha="right", z=z))
    return g


def chip_square(c, x, y, s, fc, ec, lw=2.5, z=6):
    return c.rect(x, y - s, s, s, fc=fc, ec=ec, lw=lw, z=z)


# ----------------------------------------------------------------------------- scenes
class Scene:
    title, dur, band = "", 10.0, True

    def __init__(self, data):
        self.d, self.c = data, None
        self.cues = self.make_cues()

    def make_cues(self):
        return []

    def render(self, t):
        if self.c is None:
            self.c = Canvas(self.band)
            if self.title:
                self.c.text(MX, TITLE_Y, self.title, 46, weight="bold")
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


# ----------------------------------------------------------------------------- scene 1
class SceneProblem(Scene):
    title = "Full attention fails at 4096²"
    dur = 13.0
    PW, PH, PY = 900, 700, 150          # schematic layer (frame px)
    XC, X0, X1 = 680, 510, 84           # layer x: grid centred, grid and stats centred, beside the images
    SQ, OX, OY = 500, 30, 80            # big square side and origin inside the layer
    IM, IMX, IMY, GAP = 392, 1040, 250, 20
    T_MOVE = 9.8

    def make_cues(self):
        d = self.d
        return [(0.5, 3.6, f"FLUX.1-dev is trained near {d.r0}², a grid of {d.n0}×{d.n0} tokens."),
                (3.8, 7.0, f"At 4096² the grid has {d.tok_x}× the tokens; attention costs about {d.cost_x}× as much."),
                (7.2, 9.7, f"Offsets between tokens reach {d.G - 1}, where training saw at most {d.n0 - 1}."),
                (9.9, 12.7, "Full attention then returns stripes or a flat field.")]

    def build(self):
        c, d = self.c, self.d
        L = self.L = c.layer((self.XC, self.PY, self.PW, self.PH), z=1)
        S, ox, oy = self.SQ, self.OX, self.OY
        cell = S / (d.G / 16)
        # 4096^2: 256 x 256 tokens, lines every 16 tokens
        segs = [[(ox + i * cell, oy), (ox + i * cell, oy + S)] for i in range(1, 16)]
        segs += [[(ox, oy + i * cell), (ox + S, oy + i * cell)] for i in range(1, 16)]
        self.big = Group(L.rect(ox, oy, S, S, fc=FAINT, z=1), L.lines(segs, RULE, 1.0, z=2),
                         L.rect(ox, oy, S, S, ec=MID, lw=1.6, z=3))
        # 1024^2: its own layer (64 tokens), shrinking from the big square's size to a quarter of its side
        self.small = c.layer((self.XC + ox, self.PY + oy, S, S), (0, d.n0), (d.n0, 0), z=2)
        sm = self.small
        segs = [[(i, 0), (i, d.n0)] for i in range(16, d.n0, 16)] + [[(0, i), (d.n0, i)] for i in range(16, d.n0, 16)]
        self.small_g = Group(sm.rect(0, 0, d.n0, d.n0, fc=CANVAS, z=1), sm.lines(segs, "#C9D0D6", 1.0, z=2),
                             sm.rect(0, 0, d.n0, d.n0, ec=INK, lw=2.6, z=3))
        # legend under the squares
        y1, y2 = oy + S + 52, oy + S + 98
        self.leg1 = Group(L.rect(ox, y1 - 22, 22, 22, fc=CANVAS, ec=INK, lw=2.0, z=4),
                          L.text(ox + 36, y1, f"{d.r0}²: {d.n0} × {d.n0} = {d.n0 ** 2:,} tokens", 28, z=4))
        self.leg2 = Group(L.rect(ox, y2 - 22, 22, 22, fc=FAINT, ec=MID, lw=1.6, z=4),
                          L.text(ox + 36, y2, f"4096²: {d.G} × {d.G} = {d.tokens_t} tokens", 28, z=4))
        # the largest offset: across the whole grid
        self.off = Group(L.arrow((ox, oy - 26), (ox + S, oy - 26), style="<|-|>", lw=2.0, head=14, color=INK),
                         L.text(ox + S / 2, oy - 40, f"{d.G - 1}", 28, ha="center", weight="bold"))
        # stats column
        sx = ox + S + 60
        rows = [(f"{d.tok_x}×", ["the tokens"]), (f"≈ {d.cost_x}×", ["the attention cost"]),
                (f"{d.G - 1}", ["the largest offset", f"(training: at most {d.n0 - 1})"])]
        self.stats = []
        for i, (a, subs) in enumerate(rows):
            y = oy + 80 + 160 * i
            g = Group(L.text(sx, y, a, 64, weight="bold"))
            for j, s in enumerate(subs):
                g.add(L.text(sx, y + 42 + 34 * j, s, 28, color=MID))
            self.stats.append(g)
        # the failures
        self.ims = []
        for i, (name, lab) in enumerate((("stripes", "stripes"), ("flat", "flat field"))):
            x = self.IMX + i * (self.IM + self.GAP)
            self.ims.append(Group(c.image(d.fit(name, self.IM), x, self.IMY, z=2),
                                  c.rect(x, self.IMY, self.IM, self.IM, ec=RULE, lw=1.0, z=3),
                                  c.text(x, self.IMY + self.IM + 46, lab, 30, z=3)))
        self.im_head = Group(c.text(self.IMX, self.IMY - 26, "FLUX.1-dev with full attention at 4096²", 28,
                                    color=MID))

    def update(self, t):
        d = self.d
        x = keyframes(t, [(3.9, self.XC), (4.9, self.X0), (self.T_MOVE, self.X0), (self.T_MOVE + 1.0, self.X1)])
        self.L.place((x, self.PY, self.PW, self.PH))
        u = ramp(t, 3.9, 5.2)
        s = lerp(self.SQ, self.SQ * d.n0 / d.G, u)
        self.small.place((x + self.OX, self.PY + self.OY, s, s))
        a_small = vis(t, 0.3)
        self.small_g(a_small)
        self.leg1(a_small)
        self.big(ramp(t, 4.1, 4.9))
        self.leg2(ramp(t, 4.5, 5.3))
        self.stats[0](vis(t, 4.2))
        self.stats[1](vis(t, 5.5))
        self.stats[2](vis(t, 7.3))
        self.off(vis(t, 7.3))
        self.im_head(vis(t, self.T_MOVE + 0.6))
        for i, g in enumerate(self.ims):
            g(vis(t, self.T_MOVE + 0.7 + 0.3 * i))


# ----------------------------------------------------------------------------- scene 2
class SceneNearFar(Scene):
    title = "Every head mixes near and far"
    dur = 21.5
    GX, GY, GS = 128, 176, 640          # grid square (frame px)
    RX = 860                            # right column
    TB = 11.5                           # start of the second beat (SCA's two views)
    T_GLOB = 15.0

    def make_cues(self):
        d = self.d
        m = {k: f"{v['m']:.2f}" for k, v in d.rings.items()}
        b = d.block
        return [(0.5, 3.8, "Split one query's keys into a near set and a far set."),
                (4.0, 7.6, "Full attention is exactly a mix of both, weighted by the near mass m."),
                (7.8, 11.3, f"Measured at {d.r0}²: m is {m['local']} for a local head, {m['global']} for a global "
                            "head."),
                (11.5, 14.8, f"Local view: an exact {d.win}×{d.win} window, the text and {d.bridge} bridge tokens."),
                (15.0, 18.0, f"Global view: one token per {b}×{b} block, the whole image as a {d.n0}×{d.n0} grid."),
                (18.2, 21.2, f"Each head has its own weight π. Window and grid offsets stay within {d.n0 - 1}.")]

    def win_rect(self):
        """The native window of the query at 4096^2 (tile 8, centred, clamped), in 256-token coordinates."""
        d = self.d
        t = 8
        return [int(min(max(t * (v // t) + t // 2 - d.win // 2, 0), d.G - d.win)) for v in self.q256]

    def build(self):
        c, d = self.c, self.d
        n, G = d.n0, d.G
        self.q256 = (100, 120)
        wr, wc = self.win_rect()
        # the query sits at the same token inside the window as on the 64 x 64 grid of the first beat
        self.QR, self.QC = self.q256[0] - wr, self.q256[1] - wc
        k = G / self.GS                                   # tokens per px on the 256 grid
        self.wrect = (self.GX + wc / k, self.GY + wr / k, d.win / k)
        # ---- beat B background: the 256 x 256 grid (below the 64-token layer)
        B = self.B = c.layer((self.GX, self.GY, self.GS, self.GS), (0, G), (G, 0), z=1)
        self.gB = Group(B.rect(0, 0, G, G, fc=FAINT, z=1), B.rect(0, 0, G, G, ec=MID, lw=1.6, z=7))
        cb = np.arange(n) * d.block + d.block // 2 + 0.5
        gx, gy = np.meshgrid(cb, cb)
        gx, gy = gx.ravel(), gy.ravel()
        self.gdots = B.dots(gx, gy, 3.4, GLOBAL, z=5)
        self.gdots_rgba = np.tile(rgba(GLOBAL), (len(gx), 1))
        dist = np.maximum(np.abs(gx - self.q256[1]), np.abs(gy - self.q256[0]))
        self.gdist = dist / dist.max()
        # bridge tokens: 16 per cell of an 8 x 8 partition (positions illustrative)
        bx, by = [], []
        cell = G // 8
        for i in range(8):
            for j in range(8):
                u, v = r2_points(16, i * 8 + j)
                bx += list(j * cell + np.floor(u * cell) + 0.5)
                by += list(i * cell + np.floor(v * cell) + 0.5)
        self.bdots = B.dots(np.array(bx), np.array(by), 5.0, LOCAL, edge="white", ew=0.9, z=6)
        self.bdots_rgba = np.tile(rgba(LOCAL), (len(bx), 1))
        self.bdots_edge = np.tile(rgba("#FFFFFF"), (len(bx), 1))
        # ---- the 64 x 64 grid: near / far in beat A, the native window in beat B
        A = self.A = c.layer((self.GX, self.GY, self.GS, self.GS), (0, n), (n, 0), z=2)
        r = int(d.radius)
        segs = [[(i, 0), (i, n)] for i in range(4, n, 4)] + [[(0, i), (n, i)] for i in range(4, n, 4)]
        self.far = A.rect(0, 0, n, n, fc=GLOBAL_LIGHT, z=1)
        self.near = A.rect(self.QC - r, self.QR - r, 2 * r + 1, 2 * r + 1, fc=LOCAL_LIGHT, z=2)
        self.near_edge = A.rect(self.QC - r, self.QR - r, 2 * r + 1, 2 * r + 1, ec=LOCAL, lw=3.0, z=4)
        self.gl = A.lines(segs, "#FFFFFF", 1.0, z=3)
        self.win_fill = A.rect(0, 0, n, n, fc=LOCAL_LIGHT, z=1)
        self.win_edge = A.rect(0, 0, n, n, ec=INK, lw=1.6, z=5)
        self.q = c.top.dots([0], [0], 16, INK, edge="white", ew=2.6, z=9)
        # legends under the grid
        ly = self.GY + self.GS + 46
        self.legA = Group(chip_square(c, self.GX, ly, 24, LOCAL_LIGHT, LOCAL),
                          c.text(self.GX + 38, ly, f"near: within {d.radius} tokens of the query", 28),
                          chip_square(c, self.GX, ly + 40, 24, GLOBAL_LIGHT, GLOBAL_LIGHT, lw=0),
                          c.text(self.GX + 38, ly + 40, f"far: every other key of the {n} × {n} grid at {d.r0}²", 28))
        self.legB = Group(c.text(self.GX, ly, f"Token grid at 4096², {G} × {G}; bridge positions illustrative", 26,
                                 color=MID))
        # ---- right column, beat A: the identity and three real heads
        x = self.RX
        self.eq1 = Group(c.text(x, 300, r"$O_{\mathrm{full}} \;=\; m\,O_{\mathrm{near}} \;+\; (1-m)\,O_{\mathrm{far}}$",
                                62))
        self.eq2 = Group(c.text(x, 384, r"$m \;=\; \sum_{j\,\in\,\mathrm{near}} e^{\ell_j} \;/\; \sum_{j} e^{\ell_j}$",
                                42),
                         c.text(x + 470, 380, "softmax mass of the near keys", 28, color=MID))
        self.heads = []
        roles = [("local", "local head"), ("mid", "in between"), ("global", "global head")]
        T, tx0, gap = 192, x + 20, 70
        for i, (role, lab) in enumerate(roles):
            rg = d.rings[role]
            tx = tx0 + i * (T + gap)
            ty = 560
            arts = Group(c.image(self.ring_tile(rg, T), tx, ty, z=3),
                         c.rect(tx, ty, T, T, ec=RULE, lw=1.0, z=4),
                         c.text(tx, ty - 18, lab, 28, z=4))
            qq = T / n
            arts.add(c.rect(tx + (n // 2 - r) * qq, ty + (n // 2 - r) * qq, (2 * r + 1) * qq, (2 * r + 1) * qq,
                            ec=LOCAL, lw=1.6, z=5))
            by = ty + T + 22
            arts.add(c.rect(tx, by, T * rg["m"], 14, fc=LOCAL, z=4))
            arts.add(c.rect(tx + T * rg["m"], by, T * (1 - rg["m"]), 14, fc=GLOBAL, z=4))
            arts.add(c.text(tx, by + 52, f"m = {rg['m']:.2f}", 30, z=4))
            self.heads.append(arts)
        self.heads_note = Group(c.text(tx0, 900, f"FLUX.1-dev at {d.r0}², σ = {d.ring_sigma:.2f}. Colour: attention "
                                                 "density over image keys", 26, color=MID))
        # ---- right column, beat B: the two views, the mix, the offsets
        y0 = 268
        self.vL = Group(chip_square(c, x, y0 + 2, 28, LOCAL_LIGHT, LOCAL, lw=3.0),
                        c.text(x + 46, y0, "Local view", 34, weight="bold"),
                        c.text(x + 46, y0 + 42, f"an exact {d.win} × {d.win} window around the query,", 28, color=MID),
                        c.text(x + 46, y0 + 80, f"plus the text and {d.bridge} bridge tokens (AReA's rule)", 28,
                               color=MID))
        y1 = 450
        self.vG = Group(c.dots([x + 14], [y1 - 12], 14, GLOBAL, z=9),
                        c.text(x + 46, y1, "Global view", 34, weight="bold"),
                        c.text(x + 46, y1 + 42, f"one real token per {d.block} × {d.block} block:", 28, color=MID),
                        c.text(x + 46, y1 + 80, f"the whole image as a {n} × {n} grid", 28, color=MID))
        self.eqB = Group(c.text(x, 668, r"$O \;=\; \pi_h\,O_{\mathrm{loc}} \;+\; (1-\pi_h)\,O_{\mathrm{glob}}$", 62))
        self.SX0, self.SX1, self.SY = x, x + 560, 748
        self.slider = colorbar(c, self.SX0, self.SY, self.SX1 - self.SX0, 16)
        self.knob = Circle((self.SX0, self.SY + 8), 15, facecolor="white", edgecolor=INK, linewidth=3 * 0.72, zorder=7)
        c.ax.add_patch(self.knob)
        self.knob_lab = c.text(self.SX0, self.SY - 18, r"$\pi_h$", 36, ha="center")
        self.slider.add([self.knob, self.knob_lab])
        self.noteB = Group(c.text(x, 892, f"Window and grid offsets stay within the trained {n - 1}.", 28, color=MID))
        mids = [d.rel_err(2, d.rings[k]["block"], d.rings[k]["head"], d.ring_bin, 0.0)[1]
                for k in ("global", "mid", "local")]
        t0 = 18.8
        self.PI_KEYS = [(t0, mids[1]), (t0 + 0.9, mids[2]), (t0 + 1.8, mids[0]), (t0 + 2.5, mids[1])]

    def ring_tile(self, rg, T):
        n = self.d.n0
        ii, jj = np.mgrid[0:n, 0:n]
        cheb = np.maximum(np.abs(ii - n // 2), np.abs(jj - n // 2))
        dens = np.zeros((n, n))
        inner = 0
        for lo, hi, tokens, share in rg["rings"]:
            dens[(cheb >= lo) & (cheb <= hi)] = share / (tokens / n ** 2)
            inner += tokens
        dens[cheb > rg["rings"][-1][1]] = (1 - rg["m"]) / ((n ** 2 - inner) / n ** 2)
        col = MASS_CMAP(LogNorm(0.1, 100.0, clip=True)(dens))[..., :3]
        img = (col * 255 + 0.5).astype(np.uint8)
        return np.asarray(Image.fromarray(img).resize((T, T), Image.NEAREST))

    def update(self, t):
        d, TB = self.d, self.TB
        n = d.n0
        # ---- 64-token layer: full size in beat A, then the native window inside the 256 grid
        u = ramp(t, TB + 0.2, TB + 1.6)
        wx, wy, ws = self.wrect
        rect = (lerp(self.GX, wx, u), lerp(self.GY, wy, u), lerp(self.GS, ws, u), lerp(self.GS, ws, u))
        self.A.place(rect)
        a0 = vis(t, 0.2)
        aout = 1 - ramp(t, TB - 0.2, TB + 0.4)
        self.far.set_alpha(0.75 * vis(t, 0.9) * aout)
        self.near.set_alpha(0.95 * vis(t, 0.9) * aout)
        self.near_edge.set_alpha(vis(t, 0.9) * aout)
        self.gl.set_alpha(0.9 * a0 * (1 - u))
        self.win_fill.set_alpha(0.8 * ramp(t, TB + 0.9, TB + 1.6))
        self.win_edge.set_alpha(a0)
        self.win_edge.set_edgecolor(LOCAL if u > 0.5 else INK)
        self.win_edge.set_linewidth((1.6 + 1.6 * ramp(t, TB + 0.8, TB + 1.4)) * 0.72)
        # query marker follows the layer
        qx = rect[0] + (self.QC + 0.5) / n * rect[2]
        qy = rect[1] + (self.QR + 0.5) / n * rect[3]
        self.q.set_offsets([[qx, qy]])
        self.q.set_alpha(ramp(t, 0.3, 0.8))
        self.legA(vis(t, 0.9, TB))
        # ---- right column, beat A
        self.eq1(vis(t, 3.0, TB))
        self.eq2(vis(t, 5.0, TB))
        for i, g in enumerate(self.heads):
            g(vis(t, 7.9 + 0.35 * i, TB))
        self.heads_note(vis(t, 8.6, TB))
        # ---- beat B
        self.gB(ramp(t, TB + 0.3, TB + 1.1))
        self.legB(vis(t, TB + 1.0))
        ab = 0.95 * ramp(t, TB + 1.8, TB + 2.6) * (1 - 0.55 * ramp(t, self.T_GLOB, self.T_GLOB + 0.8))
        set_dot_alpha(self.bdots, self.bdots_rgba, ab, self.bdots_edge)
        tg = self.T_GLOB + 0.1
        ad = np.clip((t - (tg + 1.2 * self.gdist)) / 0.4, 0, 1)
        set_dot_alpha(self.gdots, self.gdots_rgba, ad * ad * (3 - 2 * ad))
        self.vL(vis(t, TB + 0.6))
        self.vG(vis(t, self.T_GLOB))
        self.eqB(vis(t, 18.2))
        pi = keyframes(t, self.PI_KEYS)
        self.knob.center = (lerp(self.SX0, self.SX1, pi), self.SY + 8)
        self.knob_lab.set_x(lerp(self.SX0, self.SX1, pi))
        self.slider(vis(t, 18.5))
        self.noteB(vis(t, 19.4))


# ----------------------------------------------------------------------------- scene 3
class SceneHeads(Scene):
    title = "Reach is set by the head"
    dur = 19.5
    PX0, PX1, PY0, PY1 = 236, 1150, 196, 800      # plot area (frame px)
    S0, S1, P0, P1 = 1.0, 0.07, -0.05, 1.08       # sigma at the left and right edge; pi at the bottom and top
    RX = 1264

    def make_cues(self):
        d = self.d
        return [(0.5, 3.6, f"FLUX.1-dev has {d.blocks} blocks of {d.heads} heads: {d.n_heads} heads in all."),
                (3.8, 7.2, "Each curve is one head's weight π on its window, over the noise level σ."),
                (7.4, 10.6, f"AReA gives every head one switch: weight 0 above σ = {d.gate}, then 1."),
                (10.8, 13.8, f"Take the share of attention within {d.radius} tokens, after the first {d.n_open} "
                             "steps."),
                (14.0, 16.9, f"Which head it is explains {d.var['head']}% of its variance; the noise level, "
                             f"{d.var['noise']}%."),
                (17.1, 19.3, "Reach is a property of the head, not of the step.")]

    def X(self, s):
        return self.PX0 + (self.S0 - np.asarray(s)) / (self.S0 - self.S1) * (self.PX1 - self.PX0)

    def Yp(self, p):
        return self.PY1 - (np.asarray(p) - self.P0) / (self.P1 - self.P0) * (self.PY1 - self.PY0)

    def build(self):
        c, d = self.c, self.d
        X, Y = self.X, self.Yp
        # axes, gridlines, ticks
        st = Group()
        for p in (0.0, 0.5, 1.0):
            st.add(c.line([self.PX0, self.PX1], [Y(p), Y(p)], color=GRIDLINE, lw=1.2, z=1))
            st.add(c.text(self.PX0 - 16, Y(p) + 9, f"{p:g}", 26, color=MID, ha="right"))
        st.add(c.line([self.PX0, self.PX1], [self.PY1, self.PY1], color=INK, lw=1.6, z=4))
        for s in (1.0, float(d.gate), 0.6, 0.3, 0.1):
            st.add(c.line([X(s), X(s)], [self.PY1, self.PY1 + 9], color=INK, lw=1.6, z=4))
            st.add(c.text(X(s), self.PY1 + 42, d.gate if abs(s - float(d.gate)) < 1e-9 else f"{s:g}", 26,
                          color=MID, ha="center"))
        st.add(c.text((self.PX0 + self.PX1) / 2, self.PY1 + 92, r"noise level $\sigma$", 28, ha="center"))
        st.add(c.text(self.PX0 - 92, (self.PY0 + self.PY1) / 2, r"weight on the window, $\pi_h$", 28, ha="center",
                      va="center", rotation=90))
        self.static = st
        # 1,368 curves: extreme heads first, mid heads on top (paper Figure 2); they appear block by block
        order = np.argsort(np.abs(d.mean - 0.5))[::-1]
        xs = X(d.sig)
        self.segs = [np.column_stack([xs, Y(d.Y[k])]) for k in order]
        self.base = np.array([rgba(sf.pi_color(d.mean[k]), 0.16) for k in order])
        self.t_head = 0.6 + 2.8 * d.block_of[order] / (d.blocks - 1)
        self.lc = c.lines(self.segs, self.base, lw=1.3, z=2)
        self.count = c.text(self.PX1, self.PY0 - 8, "", 30, color=MID, ha="right")
        # median and AReA
        self.med = c.line(xs, Y(d.median), color=SCA, lw=5.0, z=6)
        self.med_lab = c.text(X(0.5), Y(np.interp(0.5, d.sig[::-1], d.median[::-1])) + 46, "median head", 28,
                              color=SCA, ha="center", z=7, weight="bold",
                              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3))
        g = float(d.gate)
        self.area = c.line(X([1.0, g, g, 0.1]), Y([0.0, 0.0, 1.0, 1.0]), color=INK, lw=3.2, z=7,
                           linestyle=(0, (5, 3.5)))
        self.area_lab = c.text(X(0.86), Y(1.0) - 20, "AReA: one switch for every head", 28, z=7)
        # colour legend (right column, top)
        x = self.RX
        self.leg = Group(c.text(x, 250, "Colour: the head's mean weight", 28),
                         colorbar(c, x, 276, 520, 16))
        # variance split (right column)
        self.var_head = Group(c.text(x, 430, f"Attention within {d.radius} tokens, at {d.r0}²", 30, weight="bold"),
                              c.text(x, 468, f"variance over heads and noise levels, after step {d.n_open}", 26,
                                     color=MID))
        self.bw, self.by = 520, 500
        self.vh = float(d.N.val("heads.var.head")) / 100
        self.bar_h = c.rect(x, self.by, 0, 40, fc=INK, z=4)
        self.bar_n = c.rect(x, self.by, 0, 40, fc=RULE, z=4)
        self.bars = Group(self.bar_h, self.bar_n)
        self.num_head = Group(c.text(x, 640, f"{d.var['head']}%", 78, weight="bold"),
                              c.text(x + 250, 612, "which head it is", 28),
                              c.text(x + 250, 646, f"{d.var['between_blocks']}% between blocks,", 26, color=MID),
                              c.text(x + 250, 678, f"{d.var['within_block']}% within a block", 26, color=MID))
        self.num_noise = Group(c.text(x, 790, f"{d.var['noise']}%", 78, weight="bold", color=MID),
                               c.text(x + 250, 762, "the noise level", 28),
                               c.text(x + 250, 796, f"{d.var['avg']}% shared by all heads", 26, color=MID))

    def update(self, t):
        d = self.d
        self.static(vis(t, 0.2))
        a = np.clip((t - self.t_head) / 0.35, 0, 1)
        a = a * a * (3 - 2 * a)
        col = self.base.copy()
        col[:, 3] = self.base[:, 3] * a
        self.lc.set_color(col)
        nshow = int((t >= self.t_head + 0.05).sum())
        self.count.set_text(f"{nshow:,} heads" if nshow < len(a) else f"{d.n_heads} heads")
        self.count.set_alpha(vis(t, 0.6))
        am = vis(t, 5.0)
        self.med.set_alpha(am)
        self.med_lab.set_alpha(am)
        self.med_lab.get_bbox_patch().set_alpha(0.85 * am)
        aa = vis(t, 7.5)
        self.area.set_alpha(aa)
        self.area_lab.set_alpha(aa)
        self.leg(vis(t, 3.9))
        self.var_head(vis(t, 10.9))
        g = ramp(t, 14.1, 15.3)
        x, bw = self.RX, self.bw
        self.bar_h.set_bounds(x, self.by, bw * self.vh * g, 40)
        self.bar_n.set_bounds(x + bw * self.vh * g + 2, self.by, max(bw * (1 - self.vh) * g - 2, 0), 40)
        self.bars(vis(t, 14.0))
        self.num_head(vis(t, 14.4))
        self.num_noise(vis(t, 15.2))


# ----------------------------------------------------------------------------- scene 4
class SceneEmulate(Scene):
    title = "Reach emulation at 1024²"
    dur = 18.5
    S, Y0, LX, RX = 430, 214, 214, 1276          # grid squares (frame px)
    QF = (0.5, 0.4375)                           # query at this fraction of each grid (row, col)
    T_VIEWS, T_RATIO, T_R2, T_FLOW = 7.1, 10.5, 13.7, 16.1

    def make_cues(self):
        d = self.d
        q = d.n0 // int(d.r4096)
        return [(0.5, 3.6, "At 4096², full attention is the failure, so it cannot serve as the reference."),
                (3.8, 6.8, f"At {d.r0}², a {d.n0}×{d.n0} grid, the model's own full attention is still valid."),
                (7.0, 10.2, f"Shrink both views by r = {d.r4096}: a {q}×{q} window and a {q}×{q} grid."),
                (10.4, 13.4, f"The image-to-window ratio is that of 4096²: {d.n0}/{q} = {d.G}/{d.win} = "
                             f"{d.r4096}."),
                (13.6, 15.8, f"r = {d.r2048} emulates 2048² the same way."),
                (16.0, 18.3, "Full attention and both views see the same queries, keys and values.")]

    def views(self, layer, n, r, t_tile, dot_px):
        """Window (side n / r, centred on the query's tile) and block-centre keys (every r-th token) on an n-grid."""
        w = n // r
        qr, qc = int(self.QF[0] * n), int(self.QF[1] * n)
        st = [int(min(max(t_tile * (v // t_tile) + t_tile // 2 - w // 2, 0), n - w)) for v in (qr, qc)]
        g = Group(layer.rect(st[1], st[0], w, w, fc=LOCAL_LIGHT, z=3, alpha=0.75),
                  layer.rect(st[1], st[0], w, w, ec=LOCAL, lw=3.0, z=6))
        cb = np.arange(n // r) * r + r // 2 + 0.5
        gx, gy = np.meshgrid(cb, cb)
        g.add(layer.dots(gx.ravel(), gy.ravel(), dot_px, GLOBAL, z=5))
        return g

    def grid(self, x, n, step):
        c = self.c
        L = c.layer((x, self.Y0, self.S, self.S), (0, n), (n, 0), z=1)
        segs = [[(i, 0), (i, n)] for i in range(step, n, step)] + [[(0, i), (n, i)] for i in range(step, n, step)]
        g = Group(L.rect(0, 0, n, n, fc=FAINT, z=1), L.lines(segs, "#D5DADF", 0.9, z=2),
                  L.rect(0, 0, n, n, ec=MID, lw=1.6, z=7))
        return L, g

    def build(self):
        c, d = self.c, self.d
        n, G, S = d.n0, d.G, self.S
        r4, r2 = int(d.r4096), int(d.r2048)
        # left: 1024^2 at r = 4 and r = 2 (tile side round(t n / (r n0)) = 2 and 4)
        self.Lg, self.gL = self.grid(self.LX, n, 4)
        self.L4 = self.views(self.Lg, n, r4, round(8 * n / (r4 * n)), 8.0)
        self.L2 = self.views(self.Lg, n, r2, round(8 * n / (r2 * n)), 5.4)
        # right: 4096^2 (r = 4) and 2048^2 (r = 2), native window, block-centre keys
        self.Rg4, self.gR4 = self.grid(self.RX, G, 16)
        self.R4 = self.views(self.Rg4, G, G // d.win, 8, 2.9)
        self.Rg2, self.gR2 = self.grid(self.RX, G // 2, 8)
        self.R2 = self.views(self.Rg2, G // 2, G // 2 // d.win, 8, 2.9)
        self.stripes = self.Rg4.image(d.fit("stripes", S, wash=0.15), 0, 0, G, G, z=1.5)
        ql = [self.LX + (int(self.QF[1] * n) + 0.5) / n * S, self.Y0 + (int(self.QF[0] * n) + 0.5) / n * S]
        qr = [self.RX + (int(self.QF[1] * G) + 0.5) / G * S, self.Y0 + (int(self.QF[0] * G) + 0.5) / G * S]
        self.q = c.top.dots([ql[0], qr[0]], [ql[1], qr[1]], 13, INK, edge="white", ew=2.2, z=9)
        # labels
        y = self.Y0 + S + 50
        self.labL = Group(c.text(self.LX, y, f"{d.r0}²: {n} × {n} tokens", 30, weight="bold"))
        self.labL_ok = Group(c.text(self.LX, y + 38, "full attention is valid here", 28, color=MID))
        self.labR = Group(c.text(self.RX, y, f"4096²: {G} × {G} tokens", 30, weight="bold"))
        self.labR_fail = Group(c.text(self.RX, y + 38, "full attention fails here", 28, color=MID))
        self.labR2 = Group(c.text(self.RX, y, f"2048²: {G // 2} × {G // 2} tokens", 30, weight="bold"))
        w4, w2 = n // r4, n // r2
        self.subL4 = Group(c.text(self.LX, y + 38, f"window {w4} × {w4}, grid {w4} × {w4} keys", 28, color=MID))
        self.subL2 = Group(c.text(self.LX, y + 38, f"window {w2} × {w2}, grid {w2} × {w2} keys", 28, color=MID))
        self.subR = Group(c.text(self.RX, y + 38, f"window {d.win} × {d.win}, grid {n} × {n} keys", 28, color=MID))
        # the shared ratio, between the grids
        cx = (self.LX + S + self.RX) / 2
        self.mid = Group(c.text(cx, 360, "image side / window side", 28, color=MID, ha="center"),
                         c.arrow((self.LX + S + 30, 470), (self.RX - 30, 470), lw=2.0, head=18, color=MID))
        self.ratio4 = Group(c.text(cx, 430, f"{n}/{w4}  =  {G}/{d.win}  =  {r4}", 46, ha="center", weight="bold"),
                            c.text(cx, 522, f"r = {r4}", 40, ha="center"))
        self.ratio2 = Group(c.text(cx, 430, f"{n}/{w2}  =  {G // 2}/{d.win}  =  {r2}", 46, ha="center",
                                   weight="bold"),
                            c.text(cx, 522, f"r = {r2}", 40, ha="center"))
        # same queries, keys and values
        y3 = 900
        pills = [(r"$O_{\mathrm{full}}$: all keys", INK), (r"$O_{\mathrm{loc}}$: the window", LOCAL),
                 (r"$O_{\mathrm{glob}}$: the grid", GLOBAL)]
        self.flow = Group(c.text(self.LX, y3, f"At {d.r0}², every block and step:", 28))
        px = self.LX + 470
        for s, col in pills:
            w = 300
            self.flow.add([c.text(px + 22, y3, s, 30, z=8), c.rbox(px, y3 - 34, w, 48, fc="white", ec=col, lw=2.4,
                                                                   r=12, z=7)])
            px += w + 30

    def update(self, t):
        a0 = vis(t, 0.2)
        self.gL(a0)
        self.labL(a0)
        self.q.set_alpha(vis(t, 0.5))
        sw = ramp(t, self.T_R2 + 0.1, self.T_R2 + 0.9)             # r = 4 -> r = 2
        self.gR4(a0 * (1 - sw))
        self.gR2(sw)
        self.stripes.set_alpha(0.85 * vis(t, 0.3, self.T_VIEWS + 0.3))
        self.labR(a0 * (1 - sw))
        self.labR2(sw)
        self.labR_fail(vis(t, 0.5, self.T_VIEWS + 0.2))
        self.labL_ok(vis(t, 3.9, self.T_VIEWS + 0.2))
        av = ramp(t, self.T_VIEWS + 0.2, self.T_VIEWS + 1.0)
        self.L4(av * (1 - sw))
        self.R4(av * (1 - sw))
        self.L2(sw)
        self.R2(sw)
        self.subL4(vis(t, self.T_VIEWS + 0.6) * (1 - sw))
        self.subL2(sw)
        self.subR(vis(t, self.T_VIEWS + 0.6))
        self.mid(vis(t, self.T_RATIO))
        self.ratio4(vis(t, self.T_RATIO + 0.1) * (1 - sw))
        self.ratio2(sw)
        self.flow(vis(t, self.T_FLOW))


# ----------------------------------------------------------------------------- scene 5
class SceneFit(Scene):
    title = "Fit each head in closed form"
    dur = 19.0
    PX0, PX1, PY0, PY1 = 1060, 1600, 300, 760     # J plot
    TB = 10.4                                     # tables
    CELL, GAP = 4, 10

    def make_cues(self):
        d = self.d
        return [(0.5, 4.0, "Sum over prompts, steps, queries and channels, per block, head, noise level and ratio."),
                (4.2, 7.2, "The squared error J(π) of a weight π is a parabola."),
                (7.4, 10.2, "On [0, 1], J is minimized by π* = clip(B/A, 0, 1), in closed form."),
                (10.4, 13.6, f"One emulation pass over {d.n_prompts} prompts at {d.r0}² fills the reach table."),
                (13.8, 16.4, f"Tables fitted on disjoint prompt halves agree: correlation {d.split} at r = {d.r2048}."),
                (16.6, 18.8, "One reach table per model, for every target size.")]

    def build(self):
        c, d = self.c, self.d
        x = MX + 24
        eqs = [r"$A \;=\; \sum\, \Vert O_{\mathrm{loc}} - O_{\mathrm{glob}} \Vert^2$",
               r"$B \;=\; \sum\, \langle\, O_{\mathrm{full}} - O_{\mathrm{glob}},\; "
               r"O_{\mathrm{loc}} - O_{\mathrm{glob}}\,\rangle$",
               r"$E \;=\; \sum\, \Vert O_{\mathrm{full}} - O_{\mathrm{glob}} \Vert^2$"]
        self.eq_sums = [Group(c.text(x, 270 + 86 * i, s, 46)) for i, s in enumerate(eqs)]
        self.eq_note = Group(c.text(x, 540, "sums over prompts, steps, queries and channels,", 28, color=MID),
                             c.text(x, 576, "one per block, head, noise level and ratio r", 28, color=MID))
        self.eq_J = Group(c.text(x, 690, r"$J(\pi) \;=\; E \,-\, 2\pi B \,+\, \pi^{2} A$", 54))
        self.eq_pi = Group(c.text(x, 800, r"$\pi^{*} \;=\; \mathrm{clip}(B/A,\; 0,\; 1)$", 54, color=SCA))
        # J of the three heads of scene 2 (ratio 2, their noise bin), relative to sum ||O_full||^2
        ps = np.linspace(0, 1, 161)
        self.curves = []
        cols = {"local": LOCAL, "mid": INK, "global": GLOBAL}
        labs = {"local": "local head", "mid": "in between", "global": "global head"}
        J = {k: d.rel_err(2, v["block"], v["head"], d.ring_bin, ps) for k, v in d.rings.items()}
        top = max(j.max() for j, _ in J.values()) * 1.1
        X = lambda p: self.PX0 + np.asarray(p) * (self.PX1 - self.PX0)  # noqa: E731
        Y = lambda e: self.PY1 - np.asarray(e) / top * (self.PY1 - self.PY0)  # noqa: E731
        st = Group()
        for e in np.arange(0, top, 0.2):
            st.add(c.line([self.PX0, self.PX1], [Y(e), Y(e)], color=GRIDLINE, lw=1.2, z=1))
            st.add(c.text(self.PX0 - 14, Y(e) + 9, f"{e:.1f}", 26, color=MID, ha="right"))
        st.add(c.line([self.PX0, self.PX1], [self.PY1, self.PY1], color=INK, lw=1.6, z=4))
        for p in (0, 0.5, 1):
            st.add(c.line([X(p), X(p)], [self.PY1, self.PY1 + 9], color=INK, lw=1.6, z=4))
            st.add(c.text(X(p), self.PY1 + 40, f"{p:g}", 26, color=MID, ha="center"))
        st.add(c.text(X(0), self.PY1 + 78, "global", 26, color=GLOBAL_TEXT, ha="center"))
        st.add(c.text(X(1), self.PY1 + 78, "local", 26, color=LOCAL_TEXT, ha="center"))
        st.add(c.text(X(0.5), self.PY1 + 78, r"weight $\pi$", 28, ha="center"))
        st.add(c.text(self.PX0 - 70, self.PY0 - 52, r"relative error $J(\pi)$", 30))
        st.add(c.text(self.PX0 - 70, self.PY0 - 16, f"three heads, r = {d.r2048}, σ = {d.ring_sigma:.2f}", 26,
                      color=MID))
        self.plot = st
        box = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.5)
        for k in ("global", "mid", "local"):
            j, pstar = J[k]
            xs, ys = X(ps), Y(j)
            ln = c.line(xs[:2], ys[:2], color=cols[k], lw=4.0, z=5)
            jm = float(np.interp(pstar, ps, j))
            mk = Group(c.line([X(pstar), X(pstar)], [Y(jm), self.PY1], color=SCA, lw=2.0, z=4,
                              linestyle=(0, (4, 3))),
                       c.dots([X(pstar)], [Y(jm)], 17, SCA, edge="white", ew=2.4, z=8))
            if k == "local":
                lab = c.text(X(0.06), Y(j[int(0.06 * 160)]) - 14, labs[k], 28, color=LOCAL_TEXT, z=8)
            else:
                lab = c.text(X(1) + 14, ys[-1] + 9, labs[k], 28, color=GLOBAL_TEXT if k == "global" else INK, z=8)
            val = c.text(X(pstar), Y(jm) - 24, f"{pstar:.2f}", 26, color=SCA, ha="center", z=9, weight="bold",
                         bbox=box)
            mk.add(val)
            mk.add(val.get_bbox_patch(), 0.9)
            self.curves.append(dict(line=ln, xs=xs, ys=ys, mark=mk, lab=lab))
        # ---- the two tables: blocks down, (noise bin x heads) across, sigma high to low, r = 2 above r = 4
        cw = d.heads * self.CELL
        nb = len(d.bins)
        self.TW = nb * cw + (nb - 1) * self.GAP
        self.TX = (W - self.TW) / 2
        th = d.blocks * self.CELL
        self.TY = [262, 262 + th + 92]
        self.full_maps, self.panels = [], []
        tg = Group()
        emp = np.empty((th, self.TW, 3), np.uint8)
        emp[:] = 255
        for j in range(nb):
            x0 = j * (cw + self.GAP)
            emp[:, x0:x0 + cw] = rgb(EMPTY).astype(np.uint8)
        self.empty = emp
        for row, s in enumerate((int(d.r2048), int(d.r4096))):
            img = emp.copy()
            for j, b in enumerate(d.bins):
                x0 = j * (cw + self.GAP)
                p = np.take_along_axis(d.tab[s][:, :, b], d.order, axis=1)   # [57, 24]
                img[:, x0:x0 + cw] = cells(pi_rgb(p), self.CELL)
            self.full_maps.append(img)
            self.panels.append(c.image(emp.copy(), self.TX, self.TY[row], z=2))
            res = {2: "2048²", 4: "4096²"}[s]
            tg.add(c.text(self.TX, self.TY[row] - 18, f"r = {s}", 30, weight="bold"))
            tg.add(c.text(self.TX + 90, self.TY[row] - 18, f"emulates {res}", 28, color=MID))
        y = self.TY[1] + th + 38
        for j, b in enumerate(d.bins):
            if j % 2 == 0 or j == nb - 1:
                tg.add(c.text(self.TX + j * (cw + self.GAP) + cw / 2, y, f"{d.centers[b]:.2f}", 26, color=MID,
                              ha="center"))
        tg.add(c.text(self.TX - 20, y, "σ", 28, color=MID, ha="right"))
        tg.add(c.text(self.TX + self.TW, self.TY[0] - 18,
                      f"each panel: {d.blocks} blocks (down) × {d.heads} heads, sorted within each block", 26,
                      color=MID, ha="right"))
        self.tab_static = tg
        self.tab_bar = colorbar(c, self.TX + self.TW - 420, y + 24, 420, 14, size=26)

    def update(self, t):
        d, TB = self.d, self.TB
        out = 1 - ramp(t, TB - 0.5, TB)
        for i, g in enumerate(self.eq_sums):
            g(vis(t, 0.4 + 0.5 * i) * out)
        self.eq_note(vis(t, 1.8) * out)
        self.eq_J(vis(t, 4.2) * out)
        self.eq_pi(vis(t, 7.4) * out)
        self.plot(vis(t, 4.4) * out)
        for i, cv in enumerate(self.curves):
            u = ramp(t, 4.8 + 0.5 * i, 6.2 + 0.5 * i)
            k = max(2, int(round(u * len(cv["xs"]))))
            cv["line"].set_data(cv["xs"][:k], cv["ys"][:k])
            cv["line"].set_alpha(ramp(t, 4.7, 4.9) * out)
            cv["lab"].set_alpha(vis(t, 5.8 + 0.5 * i) * out)
            cv["mark"](vis(t, 7.8 + 0.4 * i) * out)
        ab = ramp(t, TB, TB + 0.5)
        self.tab_static(ab)
        self.tab_bar(ab)
        cw = d.heads * self.CELL
        for full, im in zip(self.full_maps, self.panels):
            im.set_alpha(ab)
            arr = self.empty.copy()
            for j in range(len(d.bins)):
                k = ramp(t, TB + 0.4 + 0.22 * j, TB + 0.7 + 0.22 * j)
                if k <= 0:
                    continue
                x0 = j * (cw + self.GAP)
                rows = int(round(k * full.shape[0]))
                arr[:rows, x0:x0 + cw] = full[:rows, x0:x0 + cw]
            im.set_data(arr)


# ----------------------------------------------------------------------------- scene 6
class SceneProgressive(Scene):
    title = "Progressive sampling at 4096²"
    dur = 22.0
    SX0, SX1 = 470, 1050                 # sigma 1 and 0 on the timeline
    ROWS = (360, 505, 650)
    T_STAGE = [(3.7, 6.6), (7.1, 10.1), (10.6, 13.2)]    # sweep of each stage
    T_FINAL = 16.2
    C = 12                               # pi map cell (no trace)

    def make_cues(self):
        d = self.d
        return [(0.5, 3.4, "The main configuration keeps HRDiT's released three-stage schedule."),
                (3.6, 6.8, f"{d.steps_c[0]} steps at {d.r0}² with the model's own full attention set the layout."),
                (7.0, 10.3, f"Decode, upsample, sharpen, re-encode, re-noise to σ = {d.s2048}: the last "
                            f"{d.steps_c[1]} steps at 2048²."),
                (10.5, 13.4, f"Re-noise to σ = {d.s4096}: the last {d.steps_c[2]} steps at 4096²."),
                (13.6, 15.9, f"Each SCA stage reads the reach table at its ratio: {d.r2048}, then {d.r4096}."),
                (16.2, 21.8, "The final 4096² image, with a crop at native resolution.")]

    def sx(self, s):
        return self.SX0 + (1.0 - np.asarray(s)) * (self.SX1 - self.SX0)

    def build(self):
        c, d = self.c, self.d
        self.has_trace = d.trace is not None
        if self.has_trace:
            self.ROWS = (250, 345, 440)
        rows = self.ROWS
        A = Group()
        labels = [(f"{d.r0}²", "full attention"), ("2048²", f"SCA, ratio {d.r2048}"),
                  ("4096²", f"SCA, ratio {d.r4096}")]
        self.ticks, self.tick_x = [], []
        for s, (y, (res, sub)) in enumerate(zip(rows, labels)):
            A.add(c.text(MX + 24, y + 2, res, 34, weight="bold"))
            A.add(c.text(MX + 24, y + 38, f"{sub}, {d.steps_c[s]} steps", 26, color=MID))
            A.add(c.line([self.SX0, self.SX1], [y + 6, y + 6], color=RULE, lw=1.4, z=2))
            col = FULL if s == 0 else SCA
            xs = self.sx(d.stage_sigmas[s])
            self.ticks.append([(c.rbox(x - 3.5, y - 16, 7, 44, fc=col, r=3.5, z=4), col) for x in xs])
            self.tick_x.append(xs)
        self.ticks_static = A
        # re-noise: from the end of one stage back to the start of the next, and guides at those noise levels
        ya = rows[2] + 82
        self.trans = []
        for s in (1, 2):
            y0, y1 = rows[s - 1], rows[s]
            x1 = self.sx(d.stage_sigmas[s][0])
            self.trans.append(Group(
                c.arrow((self.SX1 + 8, y0 + 22), (x1 - 2, y1 - 30), rad=-0.12, lw=2.0, head=15, color=MID, z=3),
                c.line([x1, x1], [y1 - 20, ya], color=RULE, lw=1.4, z=1, linestyle=(0, (3, 3)))))
        ax = Group(c.line([self.SX0, self.SX1], [ya, ya], color=INK, lw=1.6, z=3))
        for s in (1.0, float(d.s2048), float(d.s4096), 0.0):
            lab = {float(d.s2048): d.s2048, float(d.s4096): d.s4096}.get(s, f"{s:g}")
            ax.add(c.line([self.sx(s), self.sx(s)], [ya, ya + 9], color=INK, lw=1.6, z=3))
            ax.add(c.text(self.sx(s), ya + 40, lab, 26, color=MID, ha="center"))
        ax.add(c.text(self.SX0 - 24, ya + 40, "σ", 28, color=MID, ha="right"))
        self.axis = ax
        self.step = c.text(MX + 24, ya + 40, "", 30, z=8)
        self.credit = Group(c.text(MX + 24, 196, "Schedule: HRDiT's, as released; SCA replaces only the attention", 28,
                                   color=MID))
        # pi map of the current SCA step
        C = self.C if not self.has_trace else 8
        mw, mh = d.blocks * C, d.heads * C
        if self.has_trace:
            self.MX0, self.MY0 = MX + 24, ya + 126
        else:
            self.MX0, self.MY0 = 1140, rows[0]
        self.map_c = C
        empty = np.empty((mh, mw, 3), np.uint8)
        empty[:] = rgb(EMPTY).astype(np.uint8)
        self.empty = empty
        self.map = c.image(empty, self.MX0, self.MY0, z=3)
        self.map_frame = c.rect(self.MX0, self.MY0, mw, mh, ec=RULE, lw=1.0, z=4)
        self.map_title = c.text(self.MX0, self.MY0 - 22, f"Weights π from the reach table ({d.blocks} blocks × "
                                f"{d.heads} heads)", 28, z=4)
        self.map_sub = c.text(self.MX0, self.MY0 + mh + 40, "", 28, color=MID, z=4)
        self.map_full = c.text(self.MX0 + mw / 2, self.MY0 + mh / 2 + 12, "full attention: no table", 30, color=MID,
                               ha="center", z=5)
        self.mapg = Group(self.map, self.map_frame, self.map_title, self.map_sub)
        if not self.has_trace:
            self.mapg.add(colorbar(c, self.MX0, self.MY0 + mh + 70, min(mw, 420), 14, size=26))
        # step timing: (stage, k, start time)
        self.steps = []
        for s, (t0, t1) in enumerate(self.T_STAGE):
            n = d.stage_steps[s]
            for k in range(n):
                self.steps.append((s, k, t0 + (t1 - t0) * k / n))
        self.step_t = np.array([x[2] for x in self.steps])
        self.maps = [None if s == 0 else d.pi_at(s, d.stage_sigmas[s][k]) for s, k, _ in self.steps]
        # previews of a real progressive trace
        if self.has_trace:
            P = 640
            self.PV = (1170, 176, P)
            self.pv = c.image(np.full((P, P, 3), 255, np.uint8), 1170, 176, z=3)
            self.pv_frame = c.rect(1170, 176, P, P, ec=RULE, lw=1.0, z=4)
            self.pv_lab = c.text(1170, 176 + P + 40, "", 28, color=MID, z=4)
            self.pvg = Group(self.pv, self.pv_frame, self.pv_lab)
        # the final image and its native crop
        IM, IX, IY = 760, MX + 24, 168
        s = IM / 4096
        x, y, w = FINAL_CROP
        fx, fy, fw = IX + x * s, IY + y * s, w * s
        final = np.asarray(d.img["final"].resize((IM, IM), Image.LANCZOS))
        CX, CY = 1130, 220
        self.fin = Group(c.image(final, IX, IY, z=5), c.rect(IX, IY, IM, IM, ec=RULE, lw=1.0, z=6))
        self.fin_frame = Group(c.rect(fx, fy, fw, fw, ec=ZOOM, lw=3.5, z=7))
        self.fin_crop = Group(c.image(d.final_crop, CX, CY, z=5),
                              c.rect(CX, CY, w, w, ec=ZOOM, lw=4.5, z=7),
                              c.line([fx + fw, CX], [fy, CY], color=ZOOM, lw=2.0, z=6),
                              c.line([fx + fw, CX], [fy + fw, CY + w], color=ZOOM, lw=2.0, z=6),
                              c.text(CX, CY + w + 46, "native pixels, 1:1", 28, color=MID, z=7),
                              c.text(CX, CY - 22, "Progressive SCA, 4096², final image", 28, z=7))

    def update(self, t):
        d = self.d
        out = 1 - ramp(t, self.T_FINAL - 0.6, self.T_FINAL)
        a0 = vis(t, 0.3) * out
        self.credit(vis(t, 0.6) * out)
        self.ticks_static(a0)
        k = int(np.searchsorted(self.step_t, t, side="right") - 1)
        idx = 0
        for s in range(3):
            for tk, col in self.ticks[s]:
                tk.set_alpha(a0 * (1.0 if idx <= k else 0.22))
                tk.set_facecolor(INK if idx == k else col)
                idx += 1
        for s, g in enumerate(self.trans):
            g(vis(t, self.T_STAGE[s + 1][0] - 0.5) * out)
        self.axis(a0)
        s, j = 0, 0
        if k >= 0:
            s, j, _ = self.steps[k]
            self.step.set_text(f"step {k + 1} of {d.total_c}")
        self.step.set_alpha(out if k >= 0 else 0.0)
        # pi map
        am = vis(t, 3.6) * out
        self.mapg(am)
        if k >= 0 and s > 0:
            prev = self.maps[k - 1] if self.maps[k - 1] is not None else self.maps[k]
            u = ease((t - self.step_t[k]) / 0.15)
            self.map.set_data(d.heat(lerp(prev, self.maps[k], u), self.map_c))
            self.map_sub.set_text(f"{d.stage_res[s]}² stage, ratio {d.stage_ratio[s]}, "
                                  f"σ = {d.stage_sigmas[s][j]:.2f}")
            self.map_full.set_alpha(0)
        else:
            self.map.set_data(self.empty)
            self.map_sub.set_text("")
            self.map_full.set_alpha(am)
        if self.has_trace:
            self.pvg(vis(t, 3.6) * out)
            if k >= 0:
                self.pv.set_data(d.trace.image(s, j, self.PV[2]))
                self.pv_lab.set_text(f"Decoded preview, step {k + 1}, {d.stage_res[s]}²")
        self.fin(ramp(t, self.T_FINAL - 0.2, self.T_FINAL + 0.6))
        self.fin_frame(vis(t, self.T_FINAL + 1.2))
        self.fin_crop(vis(t, self.T_FINAL + 1.7))


# ----------------------------------------------------------------------------- scene 7
class SceneEnd(Scene):
    title, dur, band = "", 6.5, False

    def make_cues(self):
        return []

    def build(self):
        c, e = self.c, self.d.end
        self.arts = []
        cx = W / 2
        self.arts.append(Group(c.text(cx, 236, "Scale-Covariant Attention", 34, color=SCA, weight="bold",
                                      ha="center")))
        self.arts.append(Group(c.text(cx, 316, "Seconds per 4096² image on one GH200, timed back to back", 40,
                                      ha="center")))
        names = ["Progressive SCA", "HRDiT", "AReA"]
        disp = [e["prog"], e["hrdit"], e["area"]]
        cols = [SCA, "#AEB5BD", "#AEB5BD"]
        x0, wmax, y0 = 760, 760, 420
        vmax = max(e["v"])
        self.bars = []
        for i, (nm, v, ds, col) in enumerate(zip(names, e["v"], disp, cols)):
            y = y0 + 96 * i
            g = Group(c.text(x0 - 28, y + 40, nm, 34, ha="right", weight="bold" if i == 0 else "normal",
                             color=INK if i == 0 else MID))
            bar = c.rect(x0, y, 0, 56, fc=col, z=3)
            val = c.text(x0, y + 40, f"{ds} s", 34, weight="bold" if i == 0 else "normal",
                         color=SCA if i == 0 else MID)
            g.add([bar, val])
            self.bars.append((g, bar, val, wmax * v / vmax, y))
        self.arts += [b[0] for b in self.bars]
        line = (f"Beats AReA on {e['all_word']} per-image metrics at each of {e['n_word']} seeds and on HRGen-1K.")
        self.arts.append(Group(c.text(cx, 780, line, 36, ha="center")))
        self.arts.append(Group(c.text(cx, 836, "Training-free. One reach table serves every target size.", 32,
                                      color=MID, ha="center")))

    def update(self, t):
        for i, g in enumerate(self.arts):
            g(vis(t, 0.2 + 0.3 * i))
        for i, (g, bar, val, w, y) in enumerate(self.bars):
            u = ramp(t, 1.0 + 0.3 * i, 2.2 + 0.3 * i)
            bar.set_bounds(760, y, w * u, 56)
            val.set_x(760 + w * u + 18)


# ----------------------------------------------------------------------------- captions and timeline
def caption_coverage(text):
    """Antialiased coverage (0..1) of one caption line, rendered once, for the band below the scenes."""
    bh = H - BAND_Y
    fig = plt.figure(figsize=(W / DPI, bh / DPI), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(bh, 0)
    ax.set_axis_off()
    ax.text(W / 2, bh / 2 + 13, text, fontsize=pt(CAP_PX), color="black", ha="center", va="baseline")
    fig.canvas.draw()
    g = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(np.float32)
    plt.close(fig)
    cov = 1.0 - g / 255.0
    cols = np.where(cov.max(axis=0) > 0.02)[0]
    assert len(cols) and cols[0] > 60 and cols[-1] < W - 60, f"caption too wide: {text}"
    return cov


class Timeline:
    def __init__(self, scenes):
        self.scenes = scenes
        self.starts, t = [], 0.0
        for s in scenes:
            self.starts.append(t)
            t += s.dur - XF
        self.total = self.starts[-1] + scenes[-1].dur
        self.cues = [(round(st + c[0], 3), round(st + c[1], 3), c[2]) for st, s in zip(self.starts, scenes)
                     for c in s.cues]
        for _, _, line in self.cues:
            assert len(line.split()) <= 15, line
        self.cov = [caption_coverage(line) for _, _, line in self.cues]
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
    with open(srt, "w", encoding="utf-8") as fh:
        for i, (t0, t1, line) in enumerate(cues, 1):
            fh.write(f"{i}\n{stamp(t0, ',')} --> {stamp(t1, ',')}\n{line}\n\n")
    with open(vtt, "w", encoding="utf-8") as fh:
        fh.write("WEBVTT\n\n")
        for i, (t0, t1, line) in enumerate(cues, 1):
            fh.write(f"{i}\n{stamp(t0, '.')} --> {stamp(t1, '.')}\n{line}\n\n")


# ----------------------------------------------------------------------------- the heads loop
class HeadsLoop:
    """960 x 960: the 1,368 per-head curves. A noise-level cursor sweeps right and back on a cosine (one period per
    loop, so the last frame leads into the first); curves left of it are drawn in their colour, each head's current
    weight is a dot on the cursor."""
    S, DUR = 960, 8.0
    PX0, PX1, PY0, PY1 = 110, 900, 150, 840
    S0, S1, P0, P1 = 0.89, 0.10, -0.03, 1.04

    def __init__(self, d):
        self.d = d
        self.c = None

    def X(self, s):
        return self.PX0 + (self.S0 - np.asarray(s)) / (self.S0 - self.S1) * (self.PX1 - self.PX0)

    def Yp(self, p):
        return self.PY1 - (np.asarray(p) - self.P0) / (self.P1 - self.P0) * (self.PY1 - self.PY0)

    def build(self):
        d = self.d
        c = self.c = Canvas(band=False, size=(self.S, self.S))
        X, Y = self.X, self.Yp
        for p in (0.0, 0.5, 1.0):
            c.line([self.PX0, self.PX1], [Y(p), Y(p)], color=GRIDLINE, lw=1.4, z=1)
        c.line([self.PX0, self.PX1], [self.PY1 + 14, self.PY1 + 14], color=INK, lw=2.0, z=4)
        c.text(self.PX0 - 18, Y(1.0) + 14, r"$\pi$", 44, color=MID, ha="right")
        c.text(self.PX1, self.PY1 + 70, r"$\sigma$", 44, color=MID, ha="right")
        c.text(self.PX0, 96, f"{d.n_heads} heads, one curve each", 46, weight="bold")
        order = np.argsort(np.abs(d.mean - 0.5))[::-1]
        xs = X(d.sig)
        segs = [np.column_stack([xs, Y(d.Y[k])]) for k in order]
        faint = np.array([rgba(sf.pi_color(d.mean[k]), 0.05) for k in order])
        full = np.array([rgba(sf.pi_color(d.mean[k]), 0.17) for k in order])
        c.lines(segs, faint, lw=1.5, z=2)
        self.lit = c.lines(segs, full, lw=1.5, z=3)
        self.clip = Rectangle((0, 0), 1, self.S, transform=c.ax.transData)
        self.lit.set_clip_path(self.clip)
        c.line(xs, Y(d.median), color=SCA, lw=6.0, z=6)
        self.cur = c.line([0, 0], [self.PY0 - 10, self.PY1 + 14], color=INK, lw=2.2, z=7)
        n = len(d.Y)
        self.dots = c.dots(np.zeros(n), np.zeros(n), 9.0, np.array([rgba(sf.pi_color(d.mean[k]), 0.6)
                                                                     for k in range(n)]), z=8)

    def frame(self, t):
        if self.c is None:
            self.build()
        d = self.d
        u = 0.5 - 0.5 * math.cos(2 * math.pi * t / self.DUR)      # 0 -> 1 -> 0 over one period
        s = lerp(d.sig[0], d.sig[-1], u)
        x = float(self.X(s))
        self.cur.set_data([x, x], [self.PY0 - 10, self.PY1 + 14])
        self.clip.set_bounds(0, 0, x, self.S)
        self.lit.set_clip_path(self.clip)      # a Rectangle becomes a fixed clip box, so set it again
        j = int(np.clip(np.searchsorted(-d.sig, -s), 1, len(d.sig) - 1))
        a = (d.sig[j - 1] - s) / (d.sig[j - 1] - d.sig[j])
        p = (1 - a) * d.Y[:, j - 1] + a * d.Y[:, j]
        self.dots.set_offsets(np.column_stack([np.full(len(p), x), self.Yp(p)]))
        return self.c.grab()


# ----------------------------------------------------------------------------- rendering and encoding
def make_timeline(d):
    return Timeline([SceneProblem(d), SceneNearFar(d), SceneHeads(d), SceneEmulate(d), SceneFit(d),
                     SceneProgressive(d), SceneEnd(d)])


def _worker(args):
    kind, frames, fdir, work, numbers, trace, fps = args
    setup_fonts()
    d = Data(Path(work), numbers, trace, build=False)
    if kind == "explainer":
        tl = make_timeline(d)
        for f in frames:
            Image.fromarray(tl.frame(f / fps, release=True)).save(fdir / f"f{f:05d}.png", compress_level=1)
    else:
        lp = HeadsLoop(d)
        for f in frames:
            Image.fromarray(lp.frame(f / fps)).save(fdir / f"f{f:05d}.png", compress_level=1)
    return len(frames)


def render_frames(kind, n, fdir, a, jobs):
    """Render frames 0..n-1 as PNG into fdir (existing files are overwritten; encode() reads exactly n)."""
    fdir.mkdir(parents=True, exist_ok=True)
    chunks = [list(r) for r in np.array_split(np.arange(n), jobs) if len(r)]
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(len(chunks)) as pool:
        done = 0
        for k in pool.imap_unordered(_worker, [(kind, ch, fdir, a.work, a.numbers, a.trace, a.fps) for ch in chunks]):
            done += k
            print(f"{kind}: {done}/{n} frames, {time.time() - t0:.0f} s", flush=True)
    assert all((fdir / f"f{f:05d}.png").exists() for f in range(n))


def ffmpeg_env():
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{FFMPEG}/lib:" + env.get("LD_LIBRARY_PATH", "")
    return env


def encode(fdir, n, mp4, fps, crf, tune="animation"):
    cmd = [str(FFMPEG / "bin" / "ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-threads", "4",
           "-framerate", str(fps), "-i", str(fdir / "f%05d.png"), "-frames:v", str(n), "-an",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+bicubic,format=yuv420p,"
                  "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-profile:v", "high"]
    if tune:
        cmd += ["-tune", tune]
    cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-movflags", "+faststart", "-threads", "4", str(mp4)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=ffmpeg_env())
    return mp4.stat().st_size


def poster(png, path, width):
    im = Image.open(png).convert("RGB")
    h = round(im.height * width / im.width)
    im.resize((width, h), Image.LANCZOS).save(path, "WEBP", quality=88, method=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--numbers", default=str(NUMBERS))
    ap.add_argument("--out", default=str(REPO / "web" / "assets" / "video" / "method"))
    ap.add_argument("--teaser-out", default=str(REPO / "web" / "assets" / "video" / "teaser"))
    ap.add_argument("--work", default=str(WORK), help="scratch dir for the image cache, frames and previews")
    ap.add_argument("--trace", default=None, help="progressive trace dir for scene 6 (see the docstring)")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--loop-crf", type=int, default=20)
    ap.add_argument("--max-mb", type=float, default=10.0)
    ap.add_argument("--loop-max-mb", type=float, default=1.5)
    ap.add_argument("--jobs", type=int, default=10, help="render processes (each one python, no threads)")
    ap.add_argument("--poster-at", type=float, default=None, help="seconds into the video for the poster")
    ap.add_argument("--loop-poster-at", type=float, default=2.0)
    ap.add_argument("--only", choices=["all", "explainer", "loop"], default="all")
    ap.add_argument("--preview", default="", help="comma-separated seconds; write PNG frames and stop")
    ap.add_argument("--preview-loop", default="", help="comma-separated seconds of the loop; write PNGs and stop")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--reencode", action="store_true", help="encode the frames already in <work>/frames, no render")
    a = ap.parse_args()
    family = setup_fonts()
    work = Path(a.work)
    build_cache(work, a.rebuild_cache)
    d = Data(work, a.numbers, a.trace)
    out, tout = Path(a.out), Path(a.teaser_out)
    tl = make_timeline(d)
    print(f"font {family}; table {d.table_path}; total {tl.total:.2f} s; scene starts "
          + ", ".join(f"{s:.1f}" for s in tl.starts), flush=True)
    if a.preview or a.preview_loop:
        pdir = work / "preview"
        pdir.mkdir(parents=True, exist_ok=True)
        for T in (float(x) for x in a.preview.split(",") if x.strip()):
            p = pdir / f"t{T:06.2f}.png"
            Image.fromarray(tl.frame(T)).save(p)
            print(p, flush=True)
        lp = HeadsLoop(d)
        for T in (float(x) for x in a.preview_loop.split(",") if x.strip()):
            p = pdir / f"loop_t{T:05.2f}.png"
            Image.fromarray(lp.frame(T)).save(p)
            print(p, flush=True)
        return
    if d.trace is not None and d.trace.synthetic:
        for p in (out, tout):
            if REPO / "web" in p.resolve().parents or p.resolve() == REPO / "web":
                sys.exit(f"--trace {a.trace} is marked synthetic; refusing to write into {p}")
    jobs = max(1, a.jobs)
    if a.only in ("all", "explainer"):
        out.mkdir(parents=True, exist_ok=True)
        write_subtitles(tl.cues, out / "sca_explainer.srt", out / "sca_explainer.vtt")
        n = int(round(tl.total * a.fps))
        fdir = work / "frames" / "explainer"
        if not (a.reencode and all((fdir / f"f{f:05d}.png").exists() for f in range(n))):
            render_frames("explainer", n, fdir, a, jobs)
        crf = a.crf
        while True:
            size = encode(fdir, n, out / "sca_explainer.mp4", a.fps, crf)
            print(f"sca_explainer.mp4: {size / 1e6:.2f} MB at CRF {crf}", flush=True)
            if size <= a.max_mb * 1e6 or crf >= 30:
                break
            crf += 1
        pt_ = a.poster_at if a.poster_at is not None else tl.starts[2] + 18.0
        poster(fdir / f"f{int(round(pt_ * a.fps)):05d}.png", out / "sca_explainer_poster.webp", 1280)
        for t0, t1, line in tl.cues:
            print(f"{t0:7.2f} {t1:7.2f}  {line}")
    if a.only in ("all", "loop"):
        tout.mkdir(parents=True, exist_ok=True)
        n = int(round(HeadsLoop.DUR * a.fps))
        fdir = work / "frames" / "loop"
        if not (a.reencode and all((fdir / f"f{f:05d}.png").exists() for f in range(n))):
            render_frames("loop", n, fdir, a, min(jobs, 8))
        crf = a.loop_crf
        while True:
            size = encode(fdir, n, tout / "heads_loop.mp4", a.fps, crf)
            print(f"heads_loop.mp4: {size / 1e6:.2f} MB at CRF {crf}", flush=True)
            if size <= a.loop_max_mb * 1e6 or crf >= 34:
                break
            crf += 1
        poster(fdir / f"f{int(round(a.loop_poster_at * a.fps)):05d}.png", tout / "heads_loop.webp", 960)


if __name__ == "__main__":
    main()
