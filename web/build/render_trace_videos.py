#!/usr/bin/env python3
"""Render the denoising-trace videos of the project page from complete traces (CPU only).

Reads /scratch/09032/saini_2/sca/web_assets/trace/<method>/<sid>/ (trace.json, steps/x0_NN.webp,
steps/crop_NN.png, written by scripts/sample_trace.py) and writes to web/assets/video/trace/:

  <sid>_sca.mp4      1920x1080. Left: the decoded x0 prediction, large. Right: the phase indicator ("global only"
                     in blue during the opening steps, "mixed" in green after), a 28-tick sigma timeline, the
                     per-head pi strip of the step (57 blocks x 24 heads, paper colormap PI_CMAP, grey while pi
                     is unused) and the 1:1 crop; a step counter.
  <sid>_vs_area.mp4  SCA and AReA previews side by side, their crops below, each method's phase label.
  <stem>.webp        poster: the last frame, 1280 px wide.

then web/assets/js/traces.js: window.SCA_TRACES = {available: [...], expected: [...]}.

Timing: 0.45 s per step (short crossfade of the images) and a 2 s hold on the final frame, 30 fps. H.264
yuv420p (BT.709) with the vendored ffmpeg, preset slow, CRF 23 raised in steps of 2 until the file is under
3 MB, +faststart, no audio, always 4 encoder threads.

--js-only rewrites traces.js from the videos already present. Traces marked "synthetic": true (renderer
tests) are stamped SYNTHETIC TEST on every frame and are refused for any output inside the repository's web/.
Run with the sca env python under nice -n 19.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WEB = REPO / "web"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "paper/figures/src"))
from make_trace_tasks import SIDS  # noqa: E402  (the eight web prompts, task order)
from scafig import PI_CMAP  # noqa: E402  (paper colormap: 0 global blue, 0.5 neutral, 1 local amber)

TRACE_ROOT = Path("/scratch/09032/saini_2/sca/web_assets/trace")
OUT_DIR = WEB / "assets/video/trace"
JS = WEB / "assets/js/traces.js"
SELECTION = WEB / "build/compare_selection.json"
FFMPEG = Path("/work/09032/saini_2/vista/library/ffmpeg-7.1-shared")
STEPS = 28
FW, FH, FPS = 1920, 1080, 30
STEP_S, HOLD_S, XFADE = 0.45, 2.0, 4
CRF, CRF_MAX, MAX_BYTES = 23, 31, 3_000_000
INK, MID, RULE, GREY, WHITE = "#1F2328", "#5B6570", "#C3C9CF", "#E6E9EC", "#FFFFFF"
GLOBAL, SCA, AREA, AREA_LIGHT, ZOOM = "#2F6DB3", "#13865F", "#7A4EA3", "#E9E0F1", "#FFC21A"
FONTS = {False: ["/usr/share/fonts/urw-base35/NimbusSans-Regular.otf",
                 "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"],
         True: ["/usr/share/fonts/urw-base35/NimbusSans-Bold.otf",
                "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"]}
KINDS = ("sca", "vs_area")
_FONT_CACHE = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        for path in FONTS[bold]:
            if Path(path).exists():
                _FONT_CACHE[key] = ImageFont.truetype(path, size)
                break
        else:
            _FONT_CACHE[key] = ImageFont.load_default(size)
    return _FONT_CACHE[key]


# ----------------------------------------------------------------------------- traces

class Trace:
    def __init__(self, path):
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "trace.json").read_text())
        self.steps = self.meta["steps"]
        self.synthetic = bool(self.meta.get("synthetic"))
        self.box = self.meta["box_xys"]
        self.size = [int(v) for v in self.meta["res"].split("x")][::-1]   # (width, height)
        self._img = {}

    def image(self, kind, i):
        key = (kind, i)
        if key not in self._img:
            rel = self.steps[i]["preview" if kind == "preview" else "crop"]
            with Image.open(self.dir / rel) as im:
                self._img[key] = im.convert("RGB")
        return self._img[key]

    def head_order(self):
        """Fixed head order per block for the whole video: descending mean pi over the mixed steps."""
        pis = [np.asarray(s["pi"], dtype=np.float64) for s in self.steps if s.get("pi") is not None]
        if not pis:
            return None
        return np.argsort(-np.mean(pis, axis=0), axis=1, kind="stable")


def complete(path):
    meta = Path(path) / "trace.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except ValueError:
        return False
    steps = data.get("steps", [])
    return bool(data.get("complete")) and len(steps) == STEPS and all(
        (Path(path) / s[k]).exists() for s in steps for k in ("preview", "crop"))


# ----------------------------------------------------------------------------- drawing helpers

def text(d, xy, s, size, fill=INK, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)
    return d.textlength(s, font=font(size, bold))


def pill(d, x, y, label, size, fill, fg, outline=None, h=None):
    f = font(size, True)
    w = int(d.textlength(label, font=f)) + 2 * int(size * 0.9)
    h = h or int(size * 1.9)
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=fill, outline=outline, width=2 if outline else 0)
    d.text((x + w / 2, y + h / 2), label, font=f, fill=fg, anchor="mm")
    return w


def dashed_v(d, x, y0, y1, fill, dash=6, gap=5, width=2):
    y = y0
    while y < y1:
        d.line((x, y, x, min(y + dash, y1)), fill=fill, width=width)
        y += dash + gap


def paste_image(canvas, d, im, x, y, size, box=None, native=None, label=None):
    """Image at (x, y) scaled to size x size, a thin rule frame, the crop box in ZOOM, an optional label."""
    canvas.paste(im if im.size == (size, size) else im.resize((size, size), Image.Resampling.LANCZOS), (x, y))
    d.rectangle((x - 1, y - 1, x + size, y + size), outline=RULE, width=1)
    if box is not None:
        k = size / native
        bx, by, bs = box[0] * k, box[1] * k, box[2] * k
        d.rectangle((x + bx, y + by, x + bx + bs, y + by + bs), outline=ZOOM, width=3)
    if label:
        f = font(22)
        w = d.textlength(label, font=f)
        d.rounded_rectangle((x + 12, y + 12, x + 12 + w + 24, y + 48), radius=18, fill=WHITE)
        d.text((x + 24, y + 30), label, font=f, fill=INK, anchor="lm")


def synthetic_stamp(d, x, y, w, h):
    size = max(14, min(56, int(w / 9)))
    d.rectangle((x, y + h // 2 - size, x + w, y + h // 2 + size), fill=(255, 255, 255))
    d.text((x + w / 2, y + h / 2), "SYNTHETIC TEST", font=font(size, True), fill=(190, 30, 30), anchor="mm")


def pi_strip(values, order, cw, ch):
    """[57 blocks, 24 heads] -> RGB image, blocks along x, heads (fixed order) along y."""
    v = np.asarray(values, dtype=np.float64)
    v = np.take_along_axis(v, order, axis=1) if order is not None else v
    rgb = (np.asarray(PI_CMAP(np.clip(v.T, 0, 1)))[..., :3] * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(rgb).resize((v.shape[0] * cw, v.shape[1] * ch), Image.Resampling.NEAREST)
    ImageDraw.Draw(im).line((19 * cw, 0, 19 * cw, im.height), fill=WHITE, width=2)   # 19 double-stream blocks
    return im


def colorbar(w, h):
    rgb = (np.asarray(PI_CMAP(np.linspace(0, 1, w)))[:, :3] * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(np.repeat(rgb[None], h, axis=0))


def fmt_sigma(s):
    return f"σ = {s:.3f}"


# ----------------------------------------------------------------------------- layouts

class ScaLayout:
    """<sid>_sca.mp4: preview left, phase, timeline, pi strip and crop right."""
    PREV = (28, 28, 1024)
    X0, X1 = 1096, 1892
    CROP = (1096, 540, 512)
    CELL = (13, 7)

    def __init__(self, tr):
        self.tr, self.order = tr, tr.head_order()
        self.gate = tr.meta.get("gate_sigma", 0.89)
        self.n_open = sum(1 for s in tr.steps if s.get("phase") == "global")

    def regions(self):
        x, y, s = self.PREV
        cx, cy, cs = self.CROP
        return [(x, y, x + s, y + s), (cx, cy, cx + cs, cy + cs)]

    def frame(self, i):
        tr, st = self.tr, self.tr.steps[i]
        im = Image.new("RGB", (FW, FH), WHITE)
        d = ImageDraw.Draw(im)
        x, y, s = self.PREV
        paste_image(im, d, tr.image("preview", i), x, y, s, tr.box, tr.size[0], "predicted clean image")
        X0, X1 = self.X0, self.X1
        text(d, (X0, 28), "SCA at 4096 × 4096", 36, bold=True)
        text(d, (X1, 28), f"step {i + 1} / {len(tr.steps)}", 36, bold=True, anchor="ra")
        mixed = st.get("phase") == "mixed"
        w = pill(d, X0, 96, "global only", 26, WHITE if mixed else GLOBAL, MID if mixed else WHITE,
                 outline=RULE if mixed else None, h=52)
        pill(d, X0 + w + 14, 96, "mixed", 26, SCA if mixed else WHITE, WHITE if mixed else MID,
             outline=None if mixed else RULE, h=52)
        text(d, (X1, 122), fmt_sigma(st["sigma"]), 32, anchor="rm")
        self.timeline(d, i)
        text(d, (X0, 318), "π for 57 blocks × 24 heads", 24, MID)
        cb = colorbar(150, 14)
        im.paste(cb, (X1 - 150 - 64, 324))
        text(d, (X1 - 150 - 72, 331), "global", 20, MID, anchor="rm")
        text(d, (X1 - 56, 331), "local", 20, MID, anchor="lm")
        cw, ch = self.CELL
        hx, hy, hw, hh = X0, 352, 57 * cw, 24 * ch
        if mixed and st.get("pi") is not None:
            im.paste(pi_strip(st["pi"], self.order, cw, ch), (hx, hy))
        else:
            d.rectangle((hx, hy, hx + hw - 1, hy + hh - 1), fill=GREY)
            text(d, (hx + hw / 2, hy + hh / 2), f"unused while σ > {self.gate:g}: global only", 22, MID,
                 anchor="mm")
        cx, cy, cs = self.CROP
        paste_image(im, d, tr.image("crop", i), cx, cy, cs)
        d.rectangle((cx - 2, cy - 2, cx + cs + 1, cy + cs + 1), outline=ZOOM, width=3)
        text(d, (cx + cs + 24, cy + 2), "1:1 crop", 26, bold=True)
        text(d, (cx + cs + 24, cy + 40), "512 px, native", 22, MID)
        text(d, (cx + cs + 24, cy + 70), "yellow box on the left", 22, MID)
        if tr.synthetic:
            synthetic_stamp(d, x, y, s, s)
            synthetic_stamp(d, cx, cy, cs, cs)
        return im

    def timeline(self, d, i):
        X0, X1, n = self.X0, self.X1, len(self.tr.steps)
        a, b, axis = X0 + 10, X1 - 10, 232
        xs = [a + k * (b - a) / (n - 1) for k in range(n)]
        d.line((a, axis, b, axis), fill=RULE, width=2)
        for k, xk in enumerate(xs):
            color = (GLOBAL if self.tr.steps[k].get("phase") == "global" else SCA) if k <= i else RULE
            if k == i:
                d.rectangle((xk - 3, axis - 22, xk + 3, axis + 22), fill=color)
                d.polygon([(xk - 9, axis - 36), (xk + 9, axis - 36), (xk, axis - 26)], fill=INK)
            else:
                d.rectangle((xk - 2, axis - 12, xk + 2, axis + 12), fill=color)
        if 0 < self.n_open < n:
            g = (xs[self.n_open - 1] + xs[self.n_open]) / 2
            dashed_v(d, g, axis - 30, axis + 30, MID)
            text(d, (g, 176), f"gate σ = {self.gate:g}", 22, MID, anchor="ma")
            text(d, ((xs[0] + xs[self.n_open - 1]) / 2, 266), f"{self.n_open} steps", 22, GLOBAL, anchor="ma")
            text(d, ((xs[self.n_open] + xs[-1]) / 2, 266), f"{n - self.n_open} steps", 22, SCA, anchor="ma")
        text(d, (X0, 266), "σ 1", 22, MID)
        text(d, (X1, 266), f"{self.tr.steps[-1]['sigma']:.2f}", 22, MID, anchor="ra")


class VsLayout:
    """<sid>_vs_area.mp4: SCA and AReA previews side by side, crops below, labels in the outer margins."""
    P, C, PY, CY = 640, 312, 92, 746
    XS = (960 - 16 - 640, 976)

    def __init__(self, sca, area):
        self.tr = (sca, area)

    def regions(self):
        out = []
        for x in self.XS:
            cx = x + (self.P - self.C) // 2
            out += [(x, self.PY, x + self.P, self.PY + self.P), (cx, self.CY, cx + self.C, self.CY + self.C)]
        return out

    @staticmethod
    def phase(method, st):
        if method == "sca":
            if st.get("phase") == "mixed":
                mean = st.get("pi_mean")
                return "mixed", SCA, WHITE, None, "" if mean is None else f"mean π {mean:.2f}"
            return "global only", GLOBAL, WHITE, None, "π unused"
        if st.get("phase") == "coarse":
            return "phase I", AREA, WHITE, None, f"pooled grid, factor {st.get('area_size')}"
        if st.get("phase") == "fine":
            return "phase II", AREA_LIGHT, AREA, AREA, f"windows, side {st.get('area_size')}"
        return "plain", WHITE, MID, RULE, ""

    def frame(self, i):
        im = Image.new("RGB", (FW, FH), WHITE)
        d = ImageDraw.Draw(im)
        sca = self.tr[0]
        text(d, (28, 26), "SCA vs AReA at 4096 × 4096, same seed", 34, bold=True)
        w = text(d, (1892, 26), fmt_sigma(sca.steps[i]["sigma"]), 30, anchor="ra")
        text(d, (1892 - w - 36, 26), f"step {i + 1} / {len(sca.steps)}", 34, bold=True, anchor="ra")
        for (method, name, color), x, tr in zip((("sca", "SCA", SCA), ("area", "AReA", AREA)), self.XS, self.tr):
            st = tr.steps[i]
            left = method == "sca"                      # SCA labels in the left margin, AReA in the right one
            lx = x - 24 if left else x + self.P + 24
            text(d, (lx, self.PY), name, 36, color, bold=True, anchor="ra" if left else "la")
            label, fill, fg, outline, detail = self.phase(method, st)
            pw = int(d.textlength(label, font=font(24, True))) + 2 * int(24 * 0.9)
            pill(d, lx - pw if left else lx, self.PY + 56, label, 24, fill, fg, outline=outline, h=48)
            if detail:
                text(d, (lx, self.PY + 124), detail, 22, MID, anchor="ra" if left else "la")
            paste_image(im, d, tr.image("preview", i), x, self.PY, self.P, tr.box, tr.size[0])
            cx = x + (self.P - self.C) // 2
            paste_image(im, d, tr.image("crop", i), cx, self.CY, self.C)
            d.rectangle((cx - 2, self.CY - 2, cx + self.C + 1, self.CY + self.C + 1), outline=ZOOM, width=3)
            if tr.synthetic:
                synthetic_stamp(d, x, self.PY, self.P, self.P)
                synthetic_stamp(d, cx, self.CY, self.C, self.C)
        cx = self.XS[0] + (self.P - self.C) // 2 - 24
        text(d, (cx, self.CY), "1:1 regions, 512 px", 22, MID, anchor="ra")
        text(d, (cx, self.CY + 32), f"shown at {self.C / 512:.2f}×", 22, MID, anchor="ra")
        return im


# ----------------------------------------------------------------------------- video

def frames(layout, n):
    """Step frames with a short crossfade of the image regions, then the hold on the last frame."""
    bounds = [round(k * STEP_S * FPS) for k in range(n + 1)]
    prev = None
    for k in range(n):
        cur = np.asarray(layout.frame(k))
        for j in range(bounds[k + 1] - bounds[k]):
            if prev is not None and j < XFADE:
                a = (j + 1) / (XFADE + 1)
                out = cur.copy()
                for x0, y0, x1, y1 in layout.regions():
                    out[y0:y1, x0:x1] = (prev[y0:y1, x0:x1] * (1 - a) + cur[y0:y1, x0:x1] * a + 0.5).astype(np.uint8)
                yield out
            else:
                yield cur
        prev = cur
    for _ in range(round(HOLD_S * FPS)):
        yield prev


def encode(layout, n, out, crf):
    env = dict(os.environ, LD_LIBRARY_PATH=f"{FFMPEG / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    tmp = out.with_name(f".{out.stem}.partial.mp4")
    cmd = [str(FFMPEG / "bin/ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FW}x{FH}", "-r", str(FPS), "-i", "-",
           "-an", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
           "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", "-threads", "4", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, env=env)
    last, count = None, 0
    try:
        for f in frames(layout, n):
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
            last, count = f, count + 1
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed (rc={rc}) for {out}")
    return tmp, last, count


def render(layout, n, out, args):
    crf = args.crf
    while True:
        tmp, last, count = encode(layout, n, out, crf)
        size = tmp.stat().st_size
        if size <= args.max_bytes or crf >= CRF_MAX:
            break
        print(f"  {out.name}: {size / 1e6:.2f} MB at CRF {crf}, re-encoding at CRF {crf + 2}", flush=True)
        crf += 2
    os.replace(tmp, out)
    poster = out.with_suffix(".webp")
    ptmp = out.with_name(f".{out.stem}.partial.webp")
    Image.fromarray(last).resize((1280, 720), Image.Resampling.LANCZOS).save(ptmp, format="WEBP", quality=85, method=6)
    os.replace(ptmp, poster)
    print(f"  wrote {out} ({size / 1e6:.2f} MB, CRF {crf}, {count} frames, {count / FPS:.1f} s) and {poster.name}",
          flush=True)
    return {"bytes": size, "crf": crf, "frames": count, "seconds": round(count / FPS, 2)}


# ----------------------------------------------------------------------------- traces.js

def web_path(path, web_root):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(web_root).resolve()))
    except ValueError:
        return str(path)


def write_js(js, out_dir, web_root, selection):
    expected, available = [], []
    for sid in SIDS:
        item = selection[sid]
        vids = {k: {"mp4": web_path(out_dir / f"{sid}_{k}.mp4", web_root),
                    "poster": web_path(out_dir / f"{sid}_{k}.webp", web_root)} for k in KINDS}
        expected.append({"sid": sid, "label": item.get("label"), "prompt": item["prompt"], "kinds": list(KINDS),
                         "videos": vids})
        for k in KINDS:
            if (out_dir / f"{sid}_{k}.mp4").exists() and (out_dir / f"{sid}_{k}.webp").exists():
                available.append({"sid": sid, "kind": k, "mp4": vids[k]["mp4"], "poster": vids[k]["poster"],
                                  "prompt": item["prompt"], "label": item.get("label")})
    body = json.dumps({"available": available, "expected": expected}, indent=1, ensure_ascii=False)
    js.parent.mkdir(parents=True, exist_ok=True)
    tmp = js.with_name(f".{js.name}.partial")
    tmp.write_text("// Generated by web/build/render_trace_videos.py. Do not edit by hand.\n"
                   f"window.SCA_TRACES = {body};\n")
    os.replace(tmp, js)
    print(f"wrote {js}: {len(available)} available, {len(expected)} expected", flush=True)


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace-root", type=Path, default=TRACE_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--js", type=Path, default=JS)
    p.add_argument("--web-root", type=Path, default=WEB, help="paths in traces.js are relative to this")
    p.add_argument("--selection", type=Path, default=SELECTION)
    p.add_argument("--sid", nargs="+", default=SIDS)
    p.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    p.add_argument("--crf", type=int, default=CRF)
    p.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    p.add_argument("--force", action="store_true", help="re-render videos that are newer than their traces")
    p.add_argument("--js-only", action="store_true", help="only rewrite traces.js from the videos present")
    args = p.parse_args(argv)
    selection = {it["sid"]: it for it in json.loads(args.selection.read_text())["items"]}
    if not args.js_only:
        for sid in args.sid:
            dirs = {m: args.trace_root / m / sid for m in ("sca", "area")}
            ok = {m: complete(dirs[m]) for m in dirs}
            jobs = []
            if "sca" in args.kinds and ok["sca"]:
                jobs.append("sca")
            if "vs_area" in args.kinds and ok["sca"] and ok["area"]:
                jobs.append("vs_area")
            if not jobs:
                print(f"{sid}: no complete trace yet (sca={ok['sca']}, area={ok['area']})", flush=True)
                continue
            tr = {m: Trace(dirs[m]) for m in dirs if ok[m]}
            if any(t.synthetic for t in tr.values()) and (inside(args.out_dir, WEB) or inside(args.js, WEB)):
                raise SystemExit(f"{sid}: synthetic test traces may not be rendered into {WEB}")
            for kind in jobs:
                out = args.out_dir / f"{sid}_{kind}.mp4"
                newest = max((t.dir / "trace.json").stat().st_mtime for t in tr.values())
                if not args.force and out.exists() and out.with_suffix(".webp").exists() and out.stat().st_mtime > newest:
                    print(f"{sid}: {out.name} is up to date", flush=True)
                    continue
                args.out_dir.mkdir(parents=True, exist_ok=True)
                layout = ScaLayout(tr["sca"]) if kind == "sca" else VsLayout(tr["sca"], tr["area"])
                print(f"{sid}: rendering {out.name}", flush=True)
                render(layout, len(tr["sca"].steps), out, args)
    write_js(args.js, args.out_dir, args.web_root, selection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
