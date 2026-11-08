#!/usr/bin/env python3
"""Render the per-example "how it works" videos and the two teaser loops of the project page (CPU only).

Subcommands
  process  For every example that has both a progressive SCA trace and an AReA trace (same prompt, seed, size and
           crop window): web/assets/video/process/<sid>.mp4 and <sid>.webp (poster: the last frame, 1280 px), then
           web/assets/js/process.js. 1920x1080, 30 fps, H.264 High yuv420p (BT.709), +faststart, silent, x264
           veryslow at the lowest CRF whose file fits --max-bytes (2.5 MB). Left: progressive SCA's decoded estimate
           of the clean image at each of its steps, under a stage rail (1024^2 layout in grey, 2048^2 and 4096^2
           refinement in green) with "step k / n" of the stage and sigma; below it the per-head mix pi of the step
           (57 blocks x 24 heads, PI_CMAP, heads in one fixed order per block), "full attention at 1024^2" during
           the native stage. Right: AReA's 28 steps at 4096^2 under its phase rail (pooled global keys in blue, then
           shrinking windows in amber), its pooling factor or window side below. Both images carry the same yellow
           window and a native-pixel inset of it (tag "1:1", or "4x upsampled" / "2x upsampled" while the
           progressive run is at 1024^2 / 2048^2). One clock normalised to the video length: 0.5 s lead, 13 s in
           which each run's steps are spread evenly, then a 2 s hold on each run's final.png (its own crop).
  zoom     web/assets/video/teaser/zoom_loop.mp4 + zoom_loop.webp (first frame): a seamless 7.2 s loop at 1080x1080,
           a pure zoom (log-uniform scale, smootherstep ease) from the whole image to a 1080 x 1080 window at 1:1,
           a 1.2 s hold and back, resampled (Lanczos) from the full-size PNG; the hold is the PNG's own pixels.
           Tags "<W> x <H>" and "1:1 native pixels". x264 codes the hold at quantizer --hold-q (20) and the moving
           frames at the lowest CRF that fits 1.5 MB. --src takes any square PNG (e.g. a 5120^2 image), --center the
           native-pixel point to zoom into (default for 4096^2: the bridge parapet, 1792 772).
  stages   web/assets/video/teaser/stages_loop.mp4 + stages_loop.webp: one progressive trace through its stages,
           1280x720: the estimate with its window (left) and the window (right) at the last step of each stage,
           1.8 s each with 0.5 s dissolves, back to the first stage (seamless); stage dots 1024^2, 2048^2, 4096^2.
  standin  Test data only: stand-in traces in the progressive schema under <work>/standin, built from the
           single-pass sca_v2 traces (stages invented), marked "synthetic": true. Renders from them carry a red
           STAND-IN banner, and nothing that reads a synthetic trace may write into web/ (refused before rendering).

Progressive trace schema (scripts/sample_trace.py, ProgTracer and finish; <dir>/trace.json plus the step images)
  top level   prompt, seed (label), derived_seed, box_xys [x, y, side] in final-image pixels, res "HxW" (final,
              square), sid (else the folder name), steps_total (must equal len(steps)), complete (false = refused),
              synthetic, final {"png": "final.png"} (the end hold; the last step is shown if it is missing),
              progressive.stages ["1024x1024", "2048x2048", "4096x4096"]; model optional (default FLUX.1-dev).
  steps[]     one record per step of every stage in execution order (30 + 17 + 10 = 57):
                stage        0, 1, 2 (required, or one of the fallbacks below)
                sigma        the step's noise level (shown with 3 decimals)
                preview      relative path of the decoded x0 estimate (any size, shown at 744 px)
                crop         relative path of the box_xys window of the same decode (any size; the tracer scales
                             the box to the stage and resamples it back to the box side)
                crop_scale   final width / stage width (4, 2, 1): the inset tag; else derived from the stage sizes
                phase        "native" (stage 0, pi null), "global" (SCA stage above the gate: pi null, band reads
                             "global keys only"), "window" or "mixed" (pi present; window shown when not 64)
                pi           [57][24] blocks x heads (or [24][57], or a relative .npy/.npz/.json path) on every
                             stage >= 1 step that is not "global"; a missing pi is an error naming the steps
  fallbacks   stage from per-step stage_res / res, from preview or crop sizes that change with the stage, or from a
              top-level stage_steps [30, 17, 10] (or progressive.base_steps + highres_steps); stage sizes from
              top-level stages / stage_res / stage_shapes, (config.)progressive.stages, per-step stage_res, or the
              changing preview or crop sizes. Stages must start at 0 and never decrease; the last must be res.
  AReA trace  the single-pass schema: steps[] with sigma, preview, crop, phase "coarse" (area_size = pool factor:
              one key per f x f block) then "fine" (area_size = window side in tokens), final.png.
Discovery walks /scratch/09032/saini_2/sca/web_assets (trace/ and trace_showcase/ alike; depth 5; skipping step
and image folders and <work>) for trace.json: progressive when method or label names "prog" or stage fields are
present, AReA when method or label is "area" or "area_*" (area_sc). Pairs are matched by sid and must agree on
prompt, seed, derived seed, size and box. --prog-root / --area-root restrict the search; --list reports each pair
and how its stages were read. A schema problem names the file and the field; with --only it is fatal.

process.js: window.SCA_PROCESS = {<sid>: {sid, title (web/build/showcase_prompts.jsonl), prompt, model, size [w, h],
seed, steps {progressive [30, 17, 10], stages ["1024x1024", ...], area 28}, seconds_per_image {progressive, area}
(numbers.json prog.clean.prog / prog.clean.area), mp4, poster (relative to web/), duration}}, merged with the
entries already present whose files exist. While no video exists the committed placeholder
(window.SCA_PROCESS = null, the page then shows the single-pass videos) is left as it is.

Commands (repository root; CPU only; nice -n 19; ffmpeg is the vendored build and always runs with 4 threads)
  PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
  nice -n 19 $PY web/build/render_process_videos.py process --list               # pairs found and their schema
  nice -n 19 $PY web/build/render_process_videos.py process                      # every pair (2 in parallel)
  nice -n 19 $PY web/build/render_process_videos.py process --only <sid>         # one example
  nice -n 19 $PY web/build/render_process_videos.py process --only <sid> --preview   # PNGs to <work>/preview only
  nice -n 19 $PY web/build/render_process_videos.py process --js-only            # process.js from the videos present
  nice -n 19 $PY web/build/render_process_videos.py zoom                         # the default 4096^2 source
  nice -n 19 $PY web/build/render_process_videos.py zoom --src <square png> --center X Y [--preview]
  nice -n 19 $PY web/build/render_process_videos.py stages --only <sid> [--preview]
  Layout tests on stand-ins (scratch only):
  nice -n 19 $PY web/build/render_process_videos.py standin
  nice -n 19 $PY web/build/render_process_videos.py process --prog-root <work>/standin --out-dir <work>/test \\
      --js <work>/test/process.js [--only <sid>] [--preview]
  nice -n 19 $PY web/build/render_process_videos.py stages --prog-root <work>/standin --only <sid> \\
      --out-dir <work>/test [--preview]
<work> = /scratch/09032/saini_2/sca/web_assets/process (previews, tests, stand-ins; the lossless master of each
video stays in <work>/tmp, overwritten by the next render of the same item, so nothing on scratch is deleted).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WEB = REPO / "web"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "paper/figures/src"))
from render_trace_videos import font, inside, web_path  # noqa: E402  (shared fonts and path helpers)
from scafig import GLOBAL, GLOBAL_LIGHT, INK, LOCAL, MID, PI_CMAP, SCA, ZOOM  # noqa: E402  (paper palette)

ASSETS = Path("/scratch/09032/saini_2/sca/web_assets")
WORK = ASSETS / "process"
TRACE_ROOT = ASSETS / "trace"
OUT_DIR = WEB / "assets/video/process"
TEASER_DIR = WEB / "assets/video/teaser"
JS = WEB / "assets/js/process.js"
NUMBERS = WEB / "build/numbers.json"
FFMPEG = Path("/work/09032/saini_2/vista/library/ffmpeg-7.1-shared")
ZOOM_SRC = ("/scratch/09032/saini_2/sca/out/dupfix_prog/sca_v2_prog/4096x4096/"
            "a4k_eval_4096_minus3112502764021036396_*_seed0.png")
ZOOM_CENTER = (1792, 772)          # bridge parapet, native x 1600..1984, y 580..964

WHITE, GRID, FAINT = "#FFFFFF", "#E6E9EC", "#8A939C"
STAGE0 = "#7D8790"                 # rail colour of the native stage: the model's own full attention
FPS = 30
LEAD_S, RUN_S, HOLD_S = 0.5, 13.0, 2.0
XFADE = 3                          # frames of crossfade after a step change
MAX_BYTES, TEASER_BYTES = 2_500_000, 1_500_000
CRF_MIN, CRF_MAX = 18, 40
SKIP_DIRS = {"steps", "frames", "img", "cache", "preview", "previews", "review", "tmp", "dev", "look"}
BANNED = re.compile("|".join(["calibrat", "curated", "random", chr(0x2014), chr(0xA7)]), re.I)  # wording rules
LANCZOS, BICUBIC, NEAREST = Image.Resampling.LANCZOS, Image.Resampling.BICUBIC, Image.Resampling.NEAREST


class TraceError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- small helpers

def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise TraceError(f"{path}: missing") from None
    except ValueError as err:
        raise TraceError(f"{path}: not valid JSON ({err})") from None


def parse_res(value):
    """'HxW' -> (w, h); also accepts [h, w] or {'res': 'HxW'}."""
    if isinstance(value, dict):
        value = value.get("res")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[1]), int(value[0])
    m = re.fullmatch(r"\s*(\d+)\s*[x×]\s*(\d+)\s*", str(value))
    if not m:
        raise TraceError(f"cannot read a resolution from {value!r} (expected 'HxW')")
    return int(m.group(2)), int(m.group(1))


def res_text(wh):
    return f"{wh[0]}²" if wh[0] == wh[1] else f"{wh[0]} × {wh[1]}"


def model_name(meta):
    m = meta.get("model") or (meta.get("config") or {}).get("model") or "black-forest-labs/FLUX.1-dev"
    return str(m).rsplit("/", 1)[-1]


def runs(values):
    """Ordered distinct values of a sequence and whether each value forms one contiguous run."""
    order, ok = [], True
    for k, v in enumerate(values):
        if v not in order:
            order.append(v)
        elif values[k - 1] != v:
            ok = False
    return order, ok


def check_words(strings, where):
    bad = sorted({s for s in strings if s and BANNED.search(s)})
    if bad:
        raise SystemExit(f"{where}: banned wording in {bad}")


def text(d, xy, s, size, fill=INK, bold=False, anchor="ls"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)
    return d.textlength(s, font=font(size, bold))


def tlen(d, s, size, bold=False):
    return d.textlength(s, font=font(size, bold))


_MASKS = {}


def round_mask(w, h, r):
    """Anti-aliased rounded-rectangle mask (drawn at 4x, reduced)."""
    key = (w, h, r)
    if key not in _MASKS:
        m = Image.new("L", (4 * w, 4 * h), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, 4 * w - 1, 4 * h - 1), radius=4 * r, fill=255)
        _MASKS[key] = m.resize((w, h), LANCZOS)
    return _MASKS[key]


def smooth(x, a=0.0, b=1.0):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * t * (t * (t * 6 - 15) + 10)


def tag(base, xy, label, size=20, alpha=1.0, anchor="lt"):
    """Small dark tag (white text on translucent black) composited onto an RGB PIL image in place."""
    if alpha <= 0:
        return
    f = font(size, True)
    probe = ImageDraw.Draw(base)
    w, h = int(probe.textlength(label, font=f)) + 2 * int(size * 0.55), int(size * 1.55)
    x, y = xy
    x = x - w if anchor[0] == "r" else x
    y = y - h if anchor[1] == "b" else y
    layer = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((0, 0, 2 * w - 1, 2 * h - 1), radius=12, fill=(0, 0, 0, int(140 * alpha)))
    d.text((w, h), label, font=font(2 * size, True), fill=(255, 255, 255, int(255 * alpha)), anchor="mm")
    layer = layer.resize((w, h), LANCZOS)
    region = base.crop((x, y, x + w, y + h)).convert("RGBA")
    base.paste(Image.alpha_composite(region, layer).convert("RGB"), (x, y))


# ----------------------------------------------------------------------------- traces

def stage_res_list(meta):
    for key in ("stages", "stage_res", "stage_shapes"):
        v = meta.get(key)
        if isinstance(v, list) and v:
            return [parse_res(x) for x in v]
    for parent in (meta.get("progressive"), (meta.get("config") or {}).get("progressive")):
        if isinstance(parent, dict) and isinstance(parent.get("stages"), list):
            return [parse_res(x) for x in parent["stages"]]
    return None


def stage_counts(meta):
    if isinstance(meta.get("stage_steps"), list):
        return [int(x) for x in meta["stage_steps"]]
    for parent in (meta.get("progressive"), (meta.get("config") or {}).get("progressive")):
        if isinstance(parent, dict) and "base_steps" in parent and "highres_steps" in parent:
            return [int(parent["base_steps"])] + [int(x) for x in parent["highres_steps"]]
    return None


def image_sizes(root, steps, key):
    out = []
    for s in steps:
        with Image.open(root / s[key]) as im:
            out.append(im.size)
    return out


def detect_stages(meta, steps, root, final, box, where):
    """Stage index of every step, the (w, h) of every stage and how both were found."""
    res = stage_res_list(meta)
    stage, how = None, None
    for key in ("stage", "stage_index", "stage_idx"):
        if all(key in s for s in steps):
            stage, how = [int(s[key]) for s in steps], f"per-step '{key}'"
            break
    for key in ("stage_res", "res"):
        if stage is not None or not all(key in s for s in steps):
            continue
        order, ok = runs([parse_res(s[key]) for s in steps])
        if not ok:
            raise TraceError(f"{where}: per-step '{key}' values do not form one run per stage")
        stage, how = [order.index(parse_res(s[key])) for s in steps], f"per-step '{key}'"
        res = res or order
    if stage is not None and res is None and all("stage_res" in s for s in steps):
        res = [parse_res(steps[stage.index(k)]["stage_res"]) for k in range(max(stage) + 1)]
    sizes = {}
    if stage is None or res is None:
        for key in ("preview", "crop"):
            sizes[key] = image_sizes(root, steps, key)
            order, ok = runs(sizes[key])
            if len(order) > 1 and ok:
                if stage is None:
                    stage, how = [order.index(z) for z in sizes[key]], f"{key} image sizes"
                if res is None:
                    if key == "preview":
                        res = order
                    else:
                        res = [(round(final[0] * z[0] / box[2]), round(final[1] * z[1] / box[2])) for z in order]
                break
    if stage is None:
        counts = stage_counts(meta)
        if counts and sum(counts) == len(steps):
            stage, how = [k for k, n in enumerate(counts) for _ in range(n)], "top-level stage counts"
    if stage is None:
        raise TraceError(
            f"{where}: cannot tell the stage of each step. Give every step 'stage' (0, 1, 2) or 'res' "
            f"('1024x1024'), or a top-level 'stage_steps' [30, 17, 10]. Keys of steps[0]: {sorted(steps[0])}")
    if any(b < a for a, b in zip(stage, stage[1:])) or stage[0] != 0:
        raise TraceError(f"{where}: stage indices must start at 0 and never decrease ({how})")
    n = max(stage) + 1
    if sorted(set(stage)) != list(range(n)):
        raise TraceError(f"{where}: stage indices skip a stage ({sorted(set(stage))})")
    if res is None:
        raise TraceError(f"{where}: no stage sizes. Add a top-level 'stages': ['1024x1024', '2048x2048', "
                         f"'4096x4096'] (or per-step 'res'); stage found from {how}")
    if len(res) != n:
        raise TraceError(f"{where}: {n} stages in the steps but {len(res)} stage sizes {res}")
    if tuple(res[-1]) != tuple(final):
        raise TraceError(f"{where}: last stage {res[-1]} is not the final size {final}")
    return stage, [tuple(r) for r in res], how


def load_array(path):
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            return z[z.files[0]]
    return np.asarray(json.loads(path.read_text()))


def load_pis(meta, steps, stage, root, where):
    extra = {}
    for e in meta.get("pi_trace") or []:
        if isinstance(e, dict) and e.get("pi") is not None and "step" in e:
            extra[(e.get("stage"), int(e["step"]))] = e["pi"]
    out, missing = [], []
    for i, (s, st) in enumerate(zip(steps, stage)):
        if st == 0:
            out.append(None)
            continue
        v = s.get("pi")
        if v is None and "step" in s:
            v = extra.get((st, int(s["step"])), extra.get((None, int(s["step"]))))
        if v is None and s.get("phase") == "global":
            out.append(None)                         # global branch only: pi unused on this step
            continue
        if v is None:
            missing.append(i)
            out.append(None)
            continue
        a = np.asarray(load_array(root / v) if isinstance(v, str) else v, dtype=np.float64)
        if a.shape == (24, 57):
            a = a.T
        if a.shape != (57, 24) or not np.all(np.isfinite(a)):
            raise TraceError(f"{where}: steps[{i}].pi has shape {a.shape} or non-finite values; expected [57, 24]")
        out.append(a)
    if missing:
        raise TraceError(f"{where}: no per-head pi for {len(missing)} steps of stage >= 1 (steps {missing[:4]}"
                         f"{' ...' if len(missing) > 4 else ''}). Record steps[i]['pi'] as the [57, 24] array that "
                         f"step's attention used.")
    return out


class Trace:
    """Common part of both trace kinds: prompt, seed, final size, crop window, step images."""

    def __init__(self, path):
        self.dir = Path(path)
        self.where = str(self.dir / "trace.json")
        m = self.meta = load_json(self.dir / "trace.json")
        if m.get("complete") is False:
            raise TraceError(f"{self.where}: complete is false (still being written?)")
        steps = m.get("steps")
        if not isinstance(steps, list) or not steps:
            raise TraceError(f"{self.where}: no 'steps' list")
        if "steps_total" in m and m["steps_total"] != len(steps):
            raise TraceError(f"{self.where}: {len(steps)} steps recorded, steps_total {m['steps_total']}")
        lacking = [k for k in ("prompt", "seed", "box_xys", "res") if k not in m]
        if lacking:
            raise TraceError(f"{self.where}: missing top-level {lacking}")
        bad = [f"steps[{i}].{k}" for i, s in enumerate(steps) for k in ("sigma", "preview", "crop") if s.get(k) is None]
        if bad:
            raise TraceError(f"{self.where}: missing {bad[:4]}{' ...' if len(bad) > 4 else ''}")
        absent = [str(self.dir / s[k]) for s in steps for k in ("preview", "crop") if not (self.dir / s[k]).exists()]
        if absent:
            raise TraceError(f"{self.where}: {len(absent)} step images missing, first {absent[0]}")
        self.steps = steps
        self.sid = str(m.get("sid") or self.dir.name)
        self.synthetic = bool(m.get("synthetic"))
        self.final = parse_res(m["res"])
        self.box = [int(v) for v in m["box_xys"]]
        if self.final[0] != self.final[1]:
            raise TraceError(f"{self.where}: final size {self.final} is not square; the layouts assume square images")
        x, y, s = self.box
        if s <= 0 or x < 0 or y < 0 or x + s > self.final[0] or y + s > self.final[1]:
            raise TraceError(f"{self.where}: box_xys {self.box} is not inside {self.final}")
        self._img = {}

    def final_path(self):
        rel = (self.meta.get("final") or {}).get("png") or "final.png"
        path = self.dir / rel
        return path if path.exists() else None

    def image(self, kind, i):
        key = (kind, i)
        if key not in self._img:
            if len(self._img) > 24:
                self._img.clear()
            with Image.open(self.dir / self.steps[i][kind]) as im:
                self._img[key] = im.convert("RGB")
        return self._img[key]


class ProgTrace(Trace):
    def __init__(self, path):
        super().__init__(path)
        self.stage, self.stage_res, self.how = detect_stages(self.meta, self.steps, self.dir, self.final, self.box,
                                                             self.where)
        self.n_stages = len(self.stage_res)
        self.counts = [self.stage.count(s) for s in range(self.n_stages)]
        self.first = [self.stage.index(s) for s in range(self.n_stages)]
        self.pi = load_pis(self.meta, self.steps, self.stage, self.dir, self.where)
        pis = [p for p in self.pi if p is not None]
        self.order = np.argsort(-np.mean(pis, axis=0), axis=1, kind="stable") if pis else None

    def describe(self):
        return (f"{self.n_stages} stages {[res_text(r) for r in self.stage_res]}, steps {self.counts} "
                f"(stage from {self.how})")


class AreaTrace(Trace):
    PHASES = ("coarse", "fine")

    def __init__(self, path):
        super().__init__(path)
        phases = [s.get("phase") for s in self.steps]
        if any(p not in self.PHASES for p in phases):
            raise TraceError(f"{self.where}: every AReA step needs phase 'coarse' or 'fine' "
                             f"(found {sorted(set(map(str, phases)))})")
        order, ok = runs(phases)
        if not ok or order[0] != "coarse":
            raise TraceError(f"{self.where}: AReA phases must run coarse then fine")
        self.n_coarse = phases.count("coarse")

    def describe(self):
        return f"{len(self.steps)} steps, {self.n_coarse} with pooled keys"


def check_pair(p, a):
    problems = []
    if " ".join(str(p.meta["prompt"]).split()) != " ".join(str(a.meta["prompt"]).split()):
        problems.append("prompts differ")
    if p.meta["seed"] != a.meta["seed"]:
        problems.append(f"seeds differ ({p.meta['seed']} vs {a.meta['seed']})")
    if "derived_seed" in p.meta and "derived_seed" in a.meta and p.meta["derived_seed"] != a.meta["derived_seed"]:
        problems.append("derived seeds differ")
    if p.final != a.final:
        problems.append(f"final sizes differ ({p.final} vs {a.final})")
    if p.box != a.box:
        problems.append(f"crop windows differ ({p.box} vs {a.box})")
    if problems:
        raise TraceError(f"{p.sid}: {p.where} and {a.where} do not match: " + "; ".join(problems))


# ----------------------------------------------------------------------------- discovery

def trace_dirs(root, exclude):
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        depth = len(here.relative_to(root).parts)
        dirnames[:] = sorted(n for n in dirnames if n not in SKIP_DIRS and depth < 5
                             and not any(inside(here / n, e) for e in exclude))
        if "trace.json" in filenames:
            yield here


def classify(meta, d):
    names = {str(meta.get(k) or "").lower() for k in ("method", "label")} - {""}
    steps = meta.get("steps") if isinstance(meta.get("steps"), list) else []
    if (any("prog" in n for n in names) or any(isinstance(s, dict) and "stage" in s for s in steps[:3])
            or any(k in meta for k in ("stages", "stage_shapes", "stage_steps", "progressive"))):
        return "prog"
    if any(n == "area" or n.startswith("area_") for n in names) or (not names and d.parent.name == "area"):
        return "area"
    return None


def discover(prog_roots, area_roots, allow_synthetic):
    found = {"prog": {}, "area": {}}
    seen = set()
    for kind, roots in (("prog", prog_roots), ("area", area_roots)):
        for root in roots:
            exclude = [] if inside(root, WORK) else [WORK]
            for d in trace_dirs(root, exclude):
                try:
                    meta = json.loads((d / "trace.json").read_text())
                except (OSError, ValueError) as err:
                    if d not in seen:
                        print(f"  skip {d}: unreadable trace.json ({err})", flush=True)
                    seen.add(d)
                    continue
                seen.add(d)
                if classify(meta, d) != kind or (meta.get("synthetic") and not allow_synthetic):
                    continue
                sid = str(meta.get("sid") or d.name)
                rank = (meta.get("complete") is not False, not meta.get("synthetic"), str(meta.get("created", "")),
                        (d / "trace.json").stat().st_mtime)
                if sid not in found[kind] or rank > found[kind][sid][0]:
                    found[kind][sid] = (rank, d)
    return {k: {sid: v[1] for sid, v in found[k].items()} for k in found}


# ----------------------------------------------------------------------------- drawing pieces

def pi_map(values, order, cw, ch):
    """[57 blocks, 24 heads] -> RGB image, blocks along x, heads (fixed order per block) along y."""
    v = np.asarray(values, dtype=np.float64)
    v = np.take_along_axis(v, order, axis=1) if order is not None else v
    rgb = (np.asarray(PI_CMAP(np.clip(v.T, 0, 1)))[..., :3] * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(rgb).resize((v.shape[0] * cw, v.shape[1] * ch), NEAREST)
    ImageDraw.Draw(im).line((19 * cw, 0, 19 * cw, im.height), fill=WHITE, width=2)   # 19 double-stream blocks
    return im


def colorbar(w, h):
    rgb = (np.asarray(PI_CMAP(np.linspace(0, 1, w)))[:, :3] * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(np.repeat(rgb[None], h, axis=0))


def segments(x, w, counts, gap):
    total, out, cur = sum(counts), [], x
    for n in counts:
        sw = (w - gap * (len(counts) - 1)) * n / total
        out.append((cur, cur + sw))
        cur += sw + gap
    return out


def draw_rail(d, x, y, w, counts, pos, labels, colors, h=8, gap=6, size=19):
    """Segmented progress rail. pos: steps elapsed (float, 0..sum(counts)). labels: (bold, regular) per segment;
    the active segment's label is ink, the others faint; the last label is right-aligned."""
    segs = segments(x, w, counts, gap)
    edges = np.cumsum([0] + list(counts))
    active = int(np.searchsorted(edges, pos, side="right") - 1)
    active = min(max(active, 0), len(counts) - 1)
    for k, ((a, b), (bold, reg)) in enumerate(zip(segs, labels)):
        on = k == active
        wb = tlen(d, bold, size, True)
        full = wb + (tlen(d, " " + reg, size) if reg else 0)
        lx = b - full if k == len(segs) - 1 and k > 0 else a
        text(d, (lx, y), bold, size, INK if on else FAINT, bold=True)
        if reg:
            text(d, (lx + wb, y), " " + reg, size, MID if on else FAINT)
    by = y + 14
    for k, (a, b) in enumerate(segs):
        d.rounded_rectangle((a, by, b, by + h), radius=h / 2, fill=GRID)
        frac = min(1.0, max(0.0, (pos - edges[k]) / counts[k]))
        if frac > 0:
            d.rounded_rectangle((a, by, max(a + h, a + (b - a) * frac), by + h), radius=h / 2, fill=colors[k])
    return active


def inset_geometry(box, final, P, size, pad):
    """Crop window in display px, the inset rectangle (corner farthest from the window) and the two connectors."""
    k = P / final[0]
    bx0, by0, bs = box[0] * k, box[1] * k, box[2] * k
    bx1, by1 = bx0 + bs, by0 + bs
    right, below = (bx0 + bx1) / 2 < P / 2, (by0 + by1) / 2 < P / 2
    while True:
        ix0 = P - pad - size if right else pad
        iy0 = P - pad - size if below else pad
        ix1, iy1 = ix0 + size, iy0 + size
        clear = ix0 > bx1 + 8 or ix1 < bx0 - 8 or iy0 > by1 + 8 or iy1 < by0 - 8
        if clear or size <= 200:
            break
        size -= 8
    if right == below:      # inset to the lower right or upper left: hull through the other two corners
        lines = [((bx1, by0), (ix1, iy0)), ((bx0, by1), (ix0, iy1))]
    else:
        lines = [((bx0, by0), (ix0, iy0)), ((bx1, by1), (ix1, iy1))]
    return (bx0, by0, bx1, by1), (ix0, iy0, ix1, iy1), lines


def image_layer(preview, crop, geo, P, label, connect=True):
    """P x P RGB array: the estimate, the yellow window, connectors, the native-pixel inset (yellow frame,
    soft shadow) and its tag."""
    (bx0, by0, bx1, by1), (ix0, iy0, ix1, iy1), lines = geo
    base = preview.resize((P, P), LANCZOS) if preview.size != (P, P) else preview.copy()
    ss = 3
    over = Image.new("RGBA", (P * ss, P * ss), (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    zoom = tuple(int(ZOOM[i:i + 2], 16) for i in (1, 3, 5))
    for (a, b) in (lines if connect else []):
        od.line((a[0] * ss, a[1] * ss, b[0] * ss, b[1] * ss), fill=zoom + (200,), width=2 * ss)
    od.rectangle((bx0 * ss, by0 * ss, bx1 * ss, by1 * ss), outline=zoom + (255,), width=3 * ss)
    over = over.resize((P, P), LANCZOS)
    shadow = Image.new("RGBA", (P, P), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle((ix0 - 2, iy0 + 2, ix1 + 2, iy1 + 8), fill=(0, 0, 0, 90))
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    out = Image.alpha_composite(Image.alpha_composite(base.convert("RGBA"), shadow), over).convert("RGB")
    side = int(round(ix1 - ix0))
    cimg = crop.resize((side, side), LANCZOS if crop.width >= side else BICUBIC)
    d = ImageDraw.Draw(out)
    f = 4
    d.rectangle((ix0 - f, iy0 - f, ix0 + side + f - 1, iy0 + side + f - 1), fill=ZOOM)
    out.paste(cimg, (int(round(ix0)), int(round(iy0))))
    if label:
        tag(out, (int(ix0) + 10, int(iy0) + side - 10), label, size=19, anchor="lb")
    return np.asarray(out)


# ----------------------------------------------------------------------------- the process video

class ProcessVideo:
    P = 744
    X = (156, 1020)
    TITLE_Y, RAIL_Y, PY = 54, 92, 128           # baselines of the titles and rail labels; top of the images
    INSET, PAD = 300, 18
    BAND, FOOT_Y = 890, 1056                    # top of the per-head map; footer baseline
    CELL = (7, 5)
    CONNECT = False                # connector lines from the window to the inset
    FW, FH = 1920, 1080

    def __init__(self, prog, area, secs, lead=LEAD_S, run=RUN_S, hold=HOLD_S):
        self.p, self.a, self.secs = prog, area, secs
        self.lead, self.run, self.hold = lead, run, hold
        self.n = (len(prog.steps), len(area.steps))
        self.geo = inset_geometry(prog.box, prog.final, self.P, self.INSET, self.PAD)
        self.frames = round((lead + run + hold) * FPS)
        self.finals = [t.final_path() for t in (prog, area)]
        for t, f in zip((prog, area), self.finals):
            if f is None:
                print(f"  {t.sid}: no final.png in {t.dir}; the end hold shows the last step", flush=True)
        self._layers = {}
        self._bands = {}
        w, h = prog.final
        self.labels_p = [(res_text(r), "layout" if s == 0 else "refinement") for s, r in enumerate(prog.stage_res)]
        self.labels_a = [("pooled global keys", ""), ("shrinking windows", "")]
        self.settings = f"{model_name(prog.meta)} · {w} × {h} · seed {prog.meta['seed']}"
        self.note = "Steps shown on a common clock; time per image measured on one GH200."
        self.names = ("Progressive SCA", "AReA")
        self.times = (f"{secs[0]} s per image", f"{secs[1]} s per image")
        self.background = self.static()

    # -- what is drawn, for the wording check
    def strings(self):
        out = [self.settings, self.note, *self.names, *self.times, "per-head mix", "global", "local",
               f"full attention at {res_text(self.p.stage_res[0])}", "1:1"]
        out += [a + " " + b for a, b in self.labels_p + self.labels_a]
        out += [self.area_detail(s) for s in self.a.steps] + [self.inset_label(i) for i in range(self.n[0])]
        out += [self.mix_label(i) for i in range(self.n[0])] + ["global keys only"]
        return out

    def static(self):
        im = Image.new("RGB", (self.FW, self.FH), WHITE)
        d = ImageDraw.Draw(im)
        for x, name, when, color in zip(self.X, self.names, self.times, (SCA, INK)):
            w = text(d, (x, self.TITLE_Y), name, 34, color, bold=True)
            text(d, (x + w + 16, self.TITLE_Y), when, 22, MID)
        text(d, (self.X[0], self.FOOT_Y), self.settings, 20, MID)
        text(d, (self.X[1] + self.P, self.FOOT_Y), self.note, 20, MID, anchor="rs")
        if self.p.synthetic or self.a.synthetic:
            text(d, (self.FW / 2, 6), "STAND-IN TEST DATA, NOT FOR THE PAGE", 18, "#BE1E1E", bold=True, anchor="mt")
        return np.asarray(im)

    # -- clock
    def locate(self, t, run):
        n = self.n[run]
        u = (t - self.lead) / (self.run / n)        # steps elapsed
        i = int(min(n - 1, max(0, math.floor(u + 1e-9))))
        since = round((t - self.lead - i * self.run / n) * FPS) if t >= self.lead else 10 ** 6
        return i, since, min(float(n), max(0.0, u))

    # -- layers
    def inset_label(self, i):
        f = self.p.steps[i].get("crop_scale")
        f = float(f) if f is not None else self.p.final[0] / self.p.stage_res[self.p.stage[i]][0]
        return "1:1" if abs(f - 1) < 1e-6 else f"{f:g}× upsampled"

    def mix_label(self, i):
        win = self.p.steps[i].get("window")
        return "per-head mix" if win in (None, 64) else f"per-head mix, window {win}"

    def layer(self, run, i):
        key = (run, i)
        if key not in self._layers:
            if len(self._layers) > 8:
                self._layers.pop(next(iter(self._layers)))
            tr = self.p if run == 0 else self.a
            if i == "final":                         # the sampler's final image: downscaled, and its native window
                with Image.open(self.finals[run]) as im:
                    full = im.convert("RGB")
                if full.size != tuple(tr.final):
                    raise TraceError(f"{self.finals[run]}: {full.size}, expected {tr.final}")
                x, y, b = tr.box
                self._layers[key] = image_layer(full.resize((self.P, self.P), LANCZOS), full.crop((x, y, x + b, y + b)),
                                                self.geo, self.P, "1:1", self.CONNECT)
            else:
                label = self.inset_label(i) if run == 0 else "1:1"
                self._layers[key] = image_layer(tr.image("preview", i), tr.image("crop", i), self.geo, self.P, label,
                                                self.CONNECT)
        return self._layers[key]

    def band(self, i):
        if i not in self._bands:
            cw, ch = self.CELL
            w, h = 57 * cw, 24 * ch
            if self.p.pi[i] is None:
                native = self.p.stage[i] == 0
                im = Image.new("RGB", (w, h), WHITE)
                d = ImageDraw.Draw(im)
                d.rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill=GRID if native else GLOBAL_LIGHT)
                label = f"full attention at {res_text(self.p.stage_res[0])}" if native else "global keys only"
                text(d, (w / 2, h / 2), label, 20, MID if native else GLOBAL, anchor="mm")
            else:
                im = pi_map(self.p.pi[i], self.p.order, cw, ch)
            self._bands[i] = np.asarray(im)
        return self._bands[i]

    @staticmethod
    def blend(prev, cur, since):
        if since >= XFADE:
            return cur
        a = (since + 1) / (XFADE + 1)
        return (prev * (1 - a) + cur * a + 0.5).astype(np.uint8)

    def area_detail(self, st):
        size = st.get("area_size")
        if st["phase"] == "coarse":
            return f"one key per {size} × {size} block" if size else "pooled keys"
        if size == 64:
            return "native window, 64 × 64 tokens"
        return f"window {size} × {size} tokens" if size else "local windows"

    # -- one frame
    def frame(self, f):
        t = f / FPS
        (i, si, ui), (j, sj, uj) = self.locate(t, 0), self.locate(t, 1)
        canvas = self.background.copy()
        P, Y = self.P, self.PY
        held = round((t - self.lead - self.run) * FPS)
        for run, (k, since), x in ((0, (i, si), self.X[0]), (1, (j, sj), self.X[1])):
            if held >= 0 and self.finals[run] is not None:
                img = self.blend(self.layer(run, k), self.layer(run, "final"), held)
            else:
                cur = self.layer(run, k)
                img = self.blend(self.layer(run, k - 1), cur, since) if k > 0 else cur
            region = Image.fromarray(canvas[Y:Y + P, x:x + P])
            region.paste(Image.fromarray(img), (0, 0), round_mask(P, P, 10))
            canvas[Y:Y + P, x:x + P] = np.asarray(region)
        b = self.band(i)
        if i > 0 and si < XFADE and self.p.pi[i] is not None and self.p.pi[i - 1] is not None:
            b = self.blend(self.band(i - 1), b, si)          # map to map only; placeholders switch cleanly
        x0, y0 = self.X[0], self.BAND
        canvas[y0:y0 + b.shape[0], x0:x0 + b.shape[1]] = b
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        tr, st = self.p, self.p.stage[i]
        k = i - tr.first[st] + 1
        self.counter(d, self.X[0], f"step {k} / {tr.counts[st]}", tr.steps[i]["sigma"])
        self.counter(d, self.X[1], f"step {j + 1} / {self.n[1]}", self.a.steps[j]["sigma"])
        draw_rail(d, self.X[0], self.RAIL_Y, P, tr.counts, ui, self.labels_p, [STAGE0] + [SCA] * (tr.n_stages - 1))
        draw_rail(d, self.X[1], self.RAIL_Y, P, [self.a.n_coarse, self.n[1] - self.a.n_coarse], uj, self.labels_a,
                  [GLOBAL, LOCAL])
        bx = x0 + b.shape[1] + 28
        if tr.pi[i] is not None:
            text(d, (bx, y0 + 32), self.mix_label(i), 21, INK, bold=True)
            im.paste(colorbar(150, 10), (int(bx), y0 + 50))
            text(d, (bx, y0 + 84), "global", 19, MID)
            text(d, (bx + 150, y0 + 84), "local", 19, MID, anchor="rs")
        ast = self.a.steps[j]
        text(d, (self.X[1], y0 + 32), self.area_detail(ast), 21, INK, bold=True)
        text(d, (self.X[1], y0 + 62), "the same for every head", 19, MID)
        return np.asarray(im)

    def counter(self, d, x, step, sigma):
        right = x + self.P
        w = text(d, (right, self.TITLE_Y), f"σ {sigma:.3f}", 22, MID, anchor="rs")
        text(d, (right - w - 26, self.TITLE_Y), step, 22, INK, anchor="rs")


# ----------------------------------------------------------------------------- encoding

def ffmpeg_env():
    return dict(os.environ, LD_LIBRARY_PATH=f"{FFMPEG / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")


def write_lossless(frames, size, path):
    """RGB frames -> lossless RGB H.264 (libx264rgb, qp 0), the master for the size-capped encodes."""
    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(FFMPEG / "bin/ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264rgb",
           "-qp", "0", "-preset", "ultrafast", "-threads", "4", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, env=ffmpeg_env())
    last, count = None, 0
    try:
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
            last, count = fr, count + 1
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0 or not path.exists():
        raise RuntimeError(f"ffmpeg failed (rc={rc}) writing {path}")
    return last, count


def encode(master, out, crf, params=None):
    tmp = out.with_name(f".{out.stem}.crf{crf}.mp4")
    cmd = [str(FFMPEG / "bin/ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(master), "-an",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
           "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
           "-c:v", "libx264", "-preset", "veryslow", "-crf", str(crf), "-g", "900",
           *(["-x264-params", params] if params else []), "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", "-threads", "4", str(tmp)]
    rc = subprocess.run(cmd, env=ffmpeg_env()).returncode
    if rc != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed (rc={rc}) for {out}")
    return tmp, tmp.stat().st_size


def encode_capped(master, out, max_bytes, crf0, params=None):
    """Lowest CRF in [CRF_MIN, CRF_MAX] whose file fits max_bytes (size is monotone in CRF and roughly halves per
    +6), found in a few encodes; returns (crf, bytes, {crf: bytes tried}). params: extra x264 options."""
    files, sizes, crf = {}, {}, crf0
    try:
        while True:
            files[crf], sizes[crf] = encode(master, out, crf, params)
            fits = [c for c in sizes if sizes[c] <= max_bytes]
            over = [c for c in sizes if sizes[c] > max_bytes]
            if not fits:
                if crf >= CRF_MAX:
                    raise RuntimeError(f"{out.name}: {sizes[crf] / 1e6:.2f} MB even at CRF {crf}")
                crf = min(CRF_MAX, crf + max(1, math.ceil(6 * math.log2(sizes[crf] / max_bytes))))
                continue
            best = min(fits)
            floor = max([c for c in over if c < best], default=CRF_MIN - 1) + 1
            if best <= floor:
                break
            crf = max(floor, best - max(1, int(6 * math.log2(max_bytes / sizes[best]))))
        os.replace(files.pop(best), out)
        return best, sizes[best], dict(sorted(sizes.items()))
    finally:
        for f in files.values():
            if f.exists():
                f.unlink()


def save_webp(arr, path, width, quality=84):
    im = Image.fromarray(arr)
    if width and im.width != width:
        im = im.resize((width, round(im.height * width / im.width)), LANCZOS)
    tmp = path.with_name(f".{path.stem}.partial.webp")
    im.save(tmp, format="WEBP", quality=quality, method=6)
    os.replace(tmp, path)


def guard(out_paths, traces):
    synth = [t.where for t in traces if t.synthetic]
    inside_web = [str(p) for p in out_paths if inside(p, WEB)]
    if synth and inside_web:
        raise SystemExit(f"refused: synthetic or stand-in traces {synth} may not be rendered into {WEB} "
                         f"({inside_web[0]}); pass --out-dir/--js under {WORK}")


# ----------------------------------------------------------------------------- process.js

def read_js(js):
    if not js.exists():
        return {}
    body = js.read_text()
    m = re.search(r"window\.SCA_PROCESS\s*=\s*(\{.*\})\s*;\s*$", body, re.S)
    return json.loads(m.group(1)) if m else {}


PLACEHOLDER = ("// Written by web/build/render_process_videos.py once the progressive traces exist; until then the page "
               "shows the single-pass videos.\nwindow.SCA_PROCESS = null;\n")


def write_js(js, new, web_root):
    old = read_js(js)
    merged = {}
    for sid, e in {**old, **new}.items():
        paths = [Path(e[k]) if Path(e[k]).is_absolute() else Path(web_root) / e[k] for k in ("mp4", "poster")]
        if all(p.exists() for p in paths):
            merged[sid] = e
    if not merged:                                   # keep the page's null placeholder (legacy videos shown)
        if old:
            js.write_text(PLACEHOLDER)
            print(f"wrote {js}: no videos left, back to the null placeholder", flush=True)
        else:
            print(f"no process videos yet; {js} left as it is", flush=True)
        return
    check_words([json.dumps({k: v for k, v in e.items() if k != "prompt"}) for e in merged.values()], str(js))
    js.parent.mkdir(parents=True, exist_ok=True)
    tmp = js.with_name(f".{js.name}.partial")
    tmp.write_text("// Generated by web/build/render_process_videos.py. Do not edit by hand.\n"
                   "// One entry per example: progressive SCA against AReA, same prompt, seed and size.\n"
                   "window.SCA_PROCESS = " + json.dumps(merged, indent=1, ensure_ascii=False) + ";\n")
    os.replace(tmp, js)
    print(f"wrote {js}: {len(merged)} examples", flush=True)


def titles():
    path = WEB / "build/showcase_prompts.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    return {r["id"]: r.get("title", "") for r in rows if "id" in r}


def js_entry(prog, area, out, poster, seconds, web_root, secs):
    return {"sid": prog.sid, "title": titles().get(prog.sid, ""), "prompt": " ".join(str(prog.meta["prompt"]).split()),
            "model": model_name(prog.meta), "size": list(prog.final), "seed": prog.meta["seed"],
            "steps": {"progressive": prog.counts, "stages": [f"{w}x{h}" for w, h in prog.stage_res],
                      "area": len(area.steps)},
            "seconds_per_image": {"progressive": secs[0], "area": secs[1]},
            "mp4": web_path(out, web_root), "poster": web_path(poster, web_root), "duration": seconds}


def seconds_per_image(path):
    d = json.loads(Path(path).read_text())
    try:
        vals = (d["prog.clean.prog"]["display"], d["prog.clean.area"]["display"])
    except KeyError as err:
        raise SystemExit(f"{path} has no {err}; run web/build/build_numbers.py") from None
    if not all(re.fullmatch(r"\d+(\.\d+)?", v) for v in vals):
        raise SystemExit(f"{path}: prog.clean.prog / prog.clean.area are not landed numbers: {vals}")
    return vals


# ----------------------------------------------------------------------------- process command

def load_pair(sid, found):
    prog, area = ProgTrace(found["prog"][sid]), AreaTrace(found["area"][sid])
    check_pair(prog, area)
    return prog, area


def render_one(job):
    sid, found, args, secs = job
    prog, area = load_pair(sid, found)
    out = Path(args.out_dir) / f"{sid}.mp4"
    poster = out.with_suffix(".webp")
    if not args.preview:
        guard([out, poster, args.js], [prog, area])
    vid = ProcessVideo(prog, area, secs)
    check_words(vid.strings(), sid)
    if args.preview:
        pdir = WORK / "preview"
        pdir.mkdir(parents=True, exist_ok=True)
        for t in args.times:
            f = min(vid.frames - 1, round(t * FPS))
            path = pdir / f"{sid}_t{t:05.2f}.png"
            Image.fromarray(vid.frame(f)).save(path)
            print(f"  wrote {path}", flush=True)
        return sid, None
    newest = max((t.dir / "trace.json").stat().st_mtime for t in (prog, area))
    if not args.force and out.exists() and poster.exists() and out.stat().st_mtime > newest:
        print(f"{sid}: {out.name} is up to date", flush=True)
        return sid, js_entry(prog, area, out, poster, round(vid.frames / FPS, 2), args.web_root, secs)
    out.parent.mkdir(parents=True, exist_ok=True)
    master = WORK / "tmp" / f"{sid}.master.mkv"
    last, count = write_lossless((vid.frame(f) for f in range(vid.frames)), (vid.FW, vid.FH), master)
    crf, size, tried = encode_capped(master, out, args.max_bytes, crf0=27)
    save_webp(last, poster, 1280)
    print(f"{sid}: wrote {out} ({size / 1e6:.2f} MB, CRF {crf}, tried {tried}, {count} frames, "
          f"{count / FPS:.1f} s) and {poster.name}", flush=True)
    return sid, js_entry(prog, area, out, poster, round(count / FPS, 2), args.web_root, secs)


def cmd_process(args):
    secs = seconds_per_image(args.numbers)
    if args.js_only:
        write_js(args.js, {}, args.web_root)
        return 0
    found = discover(args.prog_root, args.area_root, allow_synthetic=bool(args.allow_synthetic))
    both = sorted(set(found["prog"]) & set(found["area"]))
    print(f"found {len(found['prog'])} progressive and {len(found['area'])} AReA traces; {len(both)} with both",
          flush=True)
    for sid in sorted(set(found["prog"]) - set(found["area"])):
        print(f"  {sid}: progressive trace {found['prog'][sid]} has no AReA trace; skipped", flush=True)
    if args.only:
        missing = [s for s in args.only if s not in both]
        if missing:
            for s in missing:
                print(f"  {s}: progressive={found['prog'].get(s)} area={found['area'].get(s)}", flush=True)
            raise SystemExit(f"--only: no progressive + AReA pair for {missing}")
        both = [s for s in both if s in args.only]
    if args.list:
        for sid in both:
            try:
                p, a = load_pair(sid, found)
                print(f"  {sid}: OK  progressive {p.dir} ({p.describe()}); AReA {a.dir} ({a.describe()})", flush=True)
            except TraceError as err:
                print(f"  {sid}: PROBLEM {err}", flush=True)
        return 0
    if not both:
        print("nothing to render", flush=True)
        if not args.preview:
            write_js(args.js, {}, args.web_root)
        return 0
    if not args.preview:                             # refuse stand-ins for web/ before rendering anything
        synth = [str(found[k][sid]) for sid in both for k in ("prog", "area")
                 if load_json(found[k][sid] / "trace.json").get("synthetic")]
        if synth and (inside(args.out_dir, WEB) or inside(args.js, WEB)):
            raise SystemExit(f"refused: synthetic or stand-in traces ({synth[0]}, ...) may not be rendered into "
                             f"{WEB}; pass --out-dir and --js under {WORK}")
    jobs = [(sid, found, args, secs) for sid in both]
    entries, failed = {}, {}
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futs = {sid: pool.submit(render_one, job) for sid, job in zip(both, jobs)}
            for sid, fut in futs.items():
                try:
                    entries.update([fut.result()])
                except TraceError as err:
                    failed[sid] = str(err)
    else:
        for sid, job in zip(both, jobs):
            try:
                entries.update([render_one(job)])
            except TraceError as err:
                failed[sid] = str(err)
    for sid, err in failed.items():
        print(f"FAILED {sid}: {err}", flush=True)
    if not args.preview:
        write_js(args.js, {k: v for k, v in entries.items() if v}, args.web_root)
    if failed and args.only:
        return 1
    return 1 if failed else 0


# ----------------------------------------------------------------------------- zoom loop

def zoom_plan(W, S, center, hold_full, zoom_s, hold_deep):
    """Per frame: (x0, y0, side) of the source window. Pure zoom about a fixed point from the whole image (scale
    S / W) to an S x S window at 1:1 centred on `center`, log-uniform in scale with a smootherstep ease."""
    s0, s1 = S / W, 1.0
    T = np.array([min(max(center[0] - S / 2, 0), W - S), min(max(center[1] - S / 2, 0), W - S)], dtype=np.float64)
    T = np.round(T)                                  # integer window at the deepest point: exact native pixels
    Q = T / (1 / s0 - 1 / s1)                        # screen position of the fixed point
    F = T + Q / s1                                   # fixed point in source pixels
    n_h0 = round(hold_full / 2 * FPS)
    n_z, n_d = round(zoom_s * FPS), round(hold_deep * FPS)
    n_h1 = round(hold_full * FPS) - n_h0
    zs = ([0.0] * n_h0 + [smooth(k / (n_z - 1)) for k in range(n_z)] + [1.0] * n_d
          + [smooth(1 - k / (n_z - 1)) for k in range(n_z)] + [0.0] * n_h1)
    plan = []
    for z in zs:
        s = math.exp(math.log(s0) + z * (math.log(s1) - math.log(s0)))
        side = S / s
        o = F - Q / s
        if z <= 0:
            o, side = np.zeros(2), float(W)
        elif z >= 1:
            o, side = T, float(S)
        o = np.clip(o, 0, W - side)
        plan.append((float(o[0]), float(o[1]), float(side), z))
    return plan


def zoom_frame(im, S, win, labels):
    x0, y0, side, z = win
    if z >= 1 and abs(side - S) < 1e-9 and x0 == int(x0) and y0 == int(y0):
        out = im.crop((int(x0), int(y0), int(x0) + S, int(y0) + S))
    else:
        out = im.resize((S, S), LANCZOS, box=(x0, y0, x0 + side, y0 + side))
    a_full, a_deep = 1 - smooth(z, 0.15, 0.45), smooth(z, 0.8, 0.98)
    tag(out, (28, S - 28), labels[0], size=24, alpha=a_full, anchor="lb")
    tag(out, (28, S - 28), labels[1], size=24, alpha=a_deep, anchor="lb")
    return np.asarray(out)


def resolve_src(pattern):
    hits = sorted(glob.glob(str(pattern)))
    if len(hits) != 1:
        raise SystemExit(f"--src {pattern} matches {len(hits)} files; give exactly one PNG")
    return Path(hits[0])


def cmd_zoom(args):
    src = resolve_src(args.src)
    im = Image.open(src).convert("RGB")
    W, H = im.size
    if W != H:
        raise SystemExit(f"{src}: {W}x{H} is not square; the zoom loop expects a square image")
    S = args.size
    if S > W:
        raise SystemExit(f"--size {S} is larger than the image ({W})")
    center = args.center or (ZOOM_CENTER if W == 4096 else (W // 2, W // 2))
    labels = (args.label_full or f"{W} × {H}", args.label_deep)
    check_words(labels, "zoom labels")
    plan = zoom_plan(W, S, center, args.hold_full, args.zoom_seconds, args.hold_deep)
    out_dir = Path(args.out_dir)
    if args.preview:
        pdir = WORK / "preview"
        pdir.mkdir(parents=True, exist_ok=True)
        deep = max(range(len(plan)), key=lambda k: plan[k][3])
        for f in sorted({0, len(plan) // 6, len(plan) // 4, len(plan) // 3, deep}):
            path = pdir / f"zoom_f{f:03d}.png"
            Image.fromarray(zoom_frame(im, S, plan[f], labels)).save(path)
            print(f"  wrote {path} (window x {plan[f][0]:.0f}, y {plan[f][1]:.0f}, side {plan[f][2]:.0f})")
        native = np.asarray(im.crop((int(plan[deep][0]), int(plan[deep][1]), int(plan[deep][0]) + S,
                                     int(plan[deep][1]) + S)))
        same = np.array_equal(zoom_frame(im, S, plan[deep], ("", ""))[:S - 80], native[:S - 80])
        print(f"  deepest frame equals the native crop (above the tag): {same}")
        return 0
    out = out_dir / "zoom_loop.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    master = WORK / "tmp" / "zoom_loop.master.mkv"
    write_lossless((zoom_frame(im, S, w, labels) for w in plan), (S, S), master)
    deep = [k for k, w in enumerate(plan) if w[3] >= 1]
    zone = f"zones={deep[0]},{deep[-1]},q={args.hold_q}"     # the static 1:1 hold at a fixed low quantizer
    crf, size, tried = encode_capped(master, out, args.max_bytes, crf0=31, params=zone)
    save_webp(zoom_frame(im, S, plan[0], labels), out.with_suffix(".webp"), S, quality=82)
    print(f"wrote {out} ({size / 1e6:.2f} MB, CRF {crf} with {zone}, tried {tried}, {len(plan)} frames, "
          f"{len(plan) / FPS:.2f} s; source {src}, centre {center}) and {out.with_suffix('.webp').name}")
    return 0


# ----------------------------------------------------------------------------- stages loop

def stages_frames(tr, hold_s=1.8, fade_s=0.5):
    """1280x720 frames: the estimate (left) and the native window (right) at the last step of each stage, held
    and cross-dissolved in order, back to the first stage at the end (seamless)."""
    FWs, FHs, S, Y = 1280, 720, 584, 36
    XL, XR = 40, 1280 - 40 - S
    geo = inset_geometry(tr.box, tr.final, S, 10, 0)[0]
    states = []
    for s in range(tr.n_stages):
        i = tr.first[s] + tr.counts[s] - 1
        im = Image.new("RGB", (FWs, FHs), WHITE)
        d = ImageDraw.Draw(im)
        prev = tr.image("preview", i).resize((S, S), LANCZOS)
        pd = ImageDraw.Draw(prev)
        pd.rectangle(geo, outline=ZOOM, width=3)
        im.paste(prev, (XL, Y), round_mask(S, S, 10))
        crop = tr.image("crop", i)
        crop = crop.resize((S, S), LANCZOS if crop.width >= S else BICUBIC)
        d.rectangle((XR - 4, Y - 4, XR + S + 3, Y + S + 3), fill=ZOOM)
        im.paste(crop, (XR, Y))
        f = round(tr.final[0] / tr.stage_res[s][0])
        tag(im, (XR + 12, Y + S - 12), "1:1" if f == 1 else f"{f}× upsampled", size=19, anchor="lb")
        states.append(np.asarray(im))
    labels = [res_text(r) for r in tr.stage_res]
    hold, fade = round(hold_s * FPS), round(fade_s * FPS)

    def chrome(arr, pos):
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        cx = [FWs / 2 + (k - (len(labels) - 1) / 2) * 190 for k in range(len(labels))]
        d.line((cx[0], 680, cx[-1], 680), fill=GRID, width=2)
        for k, (x, lab) in enumerate(zip(cx, labels)):
            w = max(0.0, 1 - abs(pos - k)) if k or pos <= len(labels) - 1 else 0
            if k == 0 and pos > len(labels) - 1:
                w = pos - (len(labels) - 1)
            on = w > 0.5
            col = (SCA if k else STAGE0) if on else FAINT
            r = 6 + 3 * w
            d.ellipse((x - r, 680 - r, x + r, 680 + r), fill=col if on else WHITE, outline=col, width=2)
            text(d, (x, 668), lab, 22, INK if on else FAINT, bold=on, anchor="ms")
        if tr.synthetic:
            text(d, (FWs / 2, 4), "STAND-IN TEST DATA, NOT FOR THE PAGE", 18, "#BE1E1E", bold=True, anchor="mt")
        return np.asarray(im)

    n = len(states)
    for s in range(n):
        for _ in range(hold):
            yield chrome(states[s], float(s))
        nxt = states[(s + 1) % n]
        for k in range(fade):
            a = smooth((k + 1) / (fade + 1))
            yield chrome((states[s] * (1 - a) + nxt * a + 0.5).astype(np.uint8), s + a)


def cmd_stages(args):
    found = discover(args.prog_root, [], allow_synthetic=bool(args.allow_synthetic))["prog"]
    sids = args.only or sorted(found)
    if not sids:
        raise SystemExit("no progressive trace found (see --prog-root); stages_loop needs a real progressive trace")
    sid = sids[0]
    if sid not in found:
        raise SystemExit(f"no progressive trace for {sid} (found: {sorted(found)})")
    tr = ProgTrace(found[sid])
    out = Path(args.out_dir) / "stages_loop.mp4"
    guard([out], [tr])
    check_words([res_text(r) for r in tr.stage_res] + ["1:1", "2× upsampled", "4× upsampled"], "stages labels")
    frames = stages_frames(tr)
    if args.preview:
        pdir = WORK / "preview"
        pdir.mkdir(parents=True, exist_ok=True)
        for k, fr in enumerate(frames):
            if k % 30 == 0:
                path = pdir / f"stages_{sid}_f{k:03d}.png"
                Image.fromarray(fr).save(path)
                print(f"  wrote {path}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    master = WORK / "tmp" / "stages_loop.master.mkv"
    last, count = write_lossless(frames, (1280, 720), master)
    crf, size, tried = encode_capped(master, out, args.max_bytes, crf0=22)
    first = next(stages_frames(tr))
    save_webp(first, out.with_suffix(".webp"), 1280)
    print(f"wrote {out} ({size / 1e6:.2f} MB, CRF {crf}, tried {tried}, {count} frames; trace {tr.dir})")
    return 0


# ----------------------------------------------------------------------------- stand-ins (tests only)

def cmd_standin(args):
    """Stand-ins shaped like scripts/sample_trace.py's sca_v2_prog traces (ProgTracer), built from the single-pass
    sca_v2 traces: stage 0 walks the 28 single-pass steps over 30, stage 1 the last 16 over 17, stage 2 the last 10;
    each crop is reduced to the stage's pixels and resampled back to 512 px (Lanczos) as the tracer does; sigmas
    from the example's progressive sidecar when it exists; pi from the single-pass step. Marked synthetic."""
    src_root, dst_root = TRACE_ROOT / "sca_v2", WORK / "standin"
    sel = {it["sid"]: it for it in json.loads((WEB / "build/compare_selection.json").read_text())["items"]}
    sids = args.only or sorted(p.name for p in src_root.iterdir() if (p / "trace.json").exists())
    plan = [(0, 30, 0, 1024, 0), (1, 17, 12, 2048, 13), (2, 10, 18, 4096, 20)]   # stage, steps, first, res, index
    for sid in sids:
        src = src_root / sid
        meta = load_json(src / "trace.json")
        sig = None
        prog = (sel.get(sid) or {}).get("paths", {}).get("prog")
        if prog and Path(prog).with_suffix(".json").exists():
            sig = load_json(Path(prog).with_suffix(".json")).get("stage_sigmas")
        dst = dst_root / sid
        (dst / "steps").mkdir(parents=True, exist_ok=True)
        steps, n = [], 0
        for stage, count, first, res, index in plan:
            for k in range(count):
                j = int(round(first + k * (27 - first) / (count - 1)))
                s = meta["steps"][j]
                crop = Image.open(src / s["crop"]).convert("RGB")
                side = 512 * res // 4096
                if side != 512:
                    crop = crop.resize((side, side), LANCZOS).resize((512, 512), LANCZOS)
                rel = f"steps/crop_{n:02d}.png"
                crop.save(dst / rel)
                link = dst / f"steps/x0_{n:02d}.webp"
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink(src / s["preview"], link)
                rec = {"index": n, "step": n + 1, "stage": stage, "stage_res": f"{res}x{res}", "stage_step": k,
                       "schedule_index": index + k, "sigma": float(sig[stage][k]) if sig else float(s["sigma"]),
                       "preview": f"steps/x0_{n:02d}.webp", "crop": rel, "crop_scale": 4096 / res,
                       "standin_source_step": j}
                if stage == 0:
                    rec.update(phase="native", pi=None)
                else:
                    pi = s.get("pi") or next(t["pi"] for t in meta["steps"][j:] if t.get("pi") is not None)
                    rec.update(phase="mixed", window=64, pi=pi, pi_mean=float(np.mean(pi)))
                steps.append(rec)
                n += 1
        out = {"schema_version": 1, "complete": True, "synthetic": True, "standin_of": str(src),
               "method": "sca_v2_prog_standin", "label": "sca_v2_prog_standin", "sid": sid,
               "prompt": meta["prompt"], "seed": meta["seed"], "derived_seed": meta.get("derived_seed"),
               "res": meta["res"], "steps_total": n, "box_xys": meta["box_xys"], "preview_px": 1024, "crop_px": 512,
               "progressive": {"stages": ["1024x1024", "2048x2048", "4096x4096"], "base_steps": 30,
                               "highres_steps": [17, 10]}, "steps": steps}
        if (src / "final.png").exists():
            link = dst / "final.png"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(src / "final.png", link)
            out["final"] = {"png": "final.png"}
        (dst / "trace.json").write_text(json.dumps(out))
        print(f"wrote stand-in {dst} ({n} steps)", flush=True)
    return 0


# ----------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, out_dir, max_bytes):
        sp.add_argument("--prog-root", type=Path, action="append", help="where to look for progressive traces")
        sp.add_argument("--only", nargs="+", help="example ids (sid)")
        sp.add_argument("--preview", action="store_true", help=f"write PNG frames to {WORK}/preview, no video")
        sp.add_argument("--out-dir", type=Path, default=out_dir)
        sp.add_argument("--max-bytes", type=int, default=max_bytes)
        sp.add_argument("--allow-synthetic", action="store_true",
                        help="accept stand-in traces (implied by a --prog-root under the scratch work dir)")

    sp = sub.add_parser("process", help="per-example progressive SCA vs AReA videos and process.js")
    common(sp, OUT_DIR, MAX_BYTES)
    sp.add_argument("--area-root", type=Path, action="append", help="where to look for AReA traces")
    sp.add_argument("--js", type=Path, default=JS)
    sp.add_argument("--web-root", type=Path, default=WEB, help="paths in process.js are relative to this")
    sp.add_argument("--numbers", type=Path, default=NUMBERS)
    sp.add_argument("--times", type=float, nargs="+", default=[0.3, 4.0, 8.2, 10.5, 12.6, 15.0],
                    help="seconds of the --preview frames")
    sp.add_argument("--jobs", type=int, default=2, help="examples rendered in parallel (each runs a 4-thread ffmpeg)")
    sp.add_argument("--force", action="store_true", help="re-render videos newer than their traces")
    sp.add_argument("--list", action="store_true", help="report the traces found and their schema, render nothing")
    sp.add_argument("--js-only", action="store_true", help="only rewrite process.js from the videos present")

    sp = sub.add_parser("zoom", help="zoom_loop.mp4 teaser")
    sp.add_argument("--src", default=ZOOM_SRC, help="square PNG (glob allowed, must match one file)")
    sp.add_argument("--center", type=int, nargs=2, metavar=("X", "Y"), help="native-pixel point to zoom into")
    sp.add_argument("--size", type=int, default=1080, help="output side; the deepest frame shows size x size pixels")
    sp.add_argument("--hold-full", type=float, default=0.8, help="seconds on the whole image (split over the seam)")
    sp.add_argument("--zoom-seconds", type=float, default=2.6, help="seconds of each zoom")
    sp.add_argument("--hold-deep", type=float, default=1.2, help="seconds at 1:1")
    sp.add_argument("--label-full", help="tag on the whole image (default: '<W> × <H>')")
    sp.add_argument("--label-deep", default="1:1 native pixels")
    sp.add_argument("--out-dir", type=Path, default=TEASER_DIR)
    sp.add_argument("--max-bytes", type=int, default=TEASER_BYTES)
    sp.add_argument("--hold-q", type=int, default=20, help="x264 quantizer of the 1:1 hold frames")
    sp.add_argument("--preview", action="store_true", help=f"write PNG frames to {WORK}/preview, no video")

    sp = sub.add_parser("stages", help="stages_loop.mp4 teaser from one progressive trace")
    common(sp, TEASER_DIR, TEASER_BYTES)

    sp = sub.add_parser("standin", help="stand-in progressive traces for layout tests (scratch only)")
    sp.add_argument("--only", nargs="+")

    args = p.parse_args(argv)
    if args.cmd in ("process", "stages"):
        args.prog_root = args.prog_root or [ASSETS]
        if any(inside(r, WORK) for r in args.prog_root):
            args.allow_synthetic = True
    if args.cmd == "process":
        args.area_root = args.area_root or [ASSETS]
    return {"process": cmd_process, "zoom": cmd_zoom, "stages": cmd_stages, "standin": cmd_standin}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
