#!/usr/bin/env python3
"""Render the showcase "how it works" videos from progressive SCA and AReA traces (CPU only; HANDOVER Section 32).

Reads /scratch/09032/saini_2/sca/web_assets/trace/{sca_v2_prog,area_sc}/<prompt id>/ (trace.json and the
per-step previews and crops written by scripts/sample_trace.py) and writes to --out-dir (default
/scratch/09032/saini_2/sca/web_assets/video_showcase):

  <id>_prog.mp4          1920x1080. Left: the decoded x0 prediction of every step (57: 30 at 1024^2 with the model's own
                         attention, 17 at 2048^2 and 10 at 4096^2 with SCA). Right: the stage, the phase ("native",
                         "global only" while sigma is above the gate, "window <w>" on the widening-window steps, "mixed"),
                         sigma, a 57-tick timeline grouped by stage, the per-head pi strip of the step (57 blocks x 24
                         heads, paper colormap) and the crop of the 512 px box (native pixels at 4096^2; the same region
                         at 1024^2 and 2048^2, enlarged 4x and 2x).
  <id>_prog_vs_area.mp4  progressive SCA and AReA side by side, aligned by progress (frame k shows SCA step k of 57 and
                         AReA step floor(28 k / 57)), crops below, each with its step counter and phase.
  <id>_*.webp            posters (last frame, 1280 px).

Drawing helpers, encoder settings (H.264 yuv420p BT.709, CRF raised until the file is under --max-bytes) and the
colormap come from web/build/render_trace_videos.py. Run with the sca env python under nice -n 19.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_trace_videos as R  # noqa: E402

TRACES = Path("/scratch/09032/saini_2/sca/web_assets/trace")
OUT = Path("/scratch/09032/saini_2/sca/web_assets/video_showcase")
NATIVE_STAGE = "#8A929B"


class Trace(R.Trace):
    def head_order(self):
        pis = [np.asarray(s["pi"], dtype=np.float64) for s in self.steps if s.get("pi") is not None]
        return np.argsort(-np.mean(pis, axis=0), axis=1, kind="stable") if pis else None


def complete(path):
    meta = Path(path) / "trace.json"
    if not meta.exists():
        return False
    data = json.loads(meta.read_text())
    steps = data.get("steps", [])
    return bool(data.get("complete")) and len(steps) == data.get("steps_total") and all(
        (Path(path) / s[k]).exists() for s in steps for k in ("preview", "crop"))


def phase_style(st):
    """(label, fill, fg, outline) of a progressive-trace step."""
    ph = st.get("phase")
    if ph == "native":
        return "native attention", NATIVE_STAGE, R.WHITE, None
    if ph == "global":
        return "global only", R.GLOBAL, R.WHITE, None
    if ph == "window":
        return f"window {st.get('window')}", R.WIDE, R.WHITE, None
    return "mixed, native window", R.SCA, R.WHITE, None


def tick_color(st):
    return {"native": NATIVE_STAGE, "global": R.GLOBAL, "window": R.WIDE}.get(st.get("phase"), R.SCA)


class ProgLayout:
    PREV = (28, 28, 1024)
    X0, X1 = 1096, 1892
    CROP = (1096, 540, 512)
    CELL = (13, 7)

    def __init__(self, tr):
        self.tr, self.order = tr, tr.head_order()
        self.gate = tr.meta.get("gate_sigma", 0.89)
        stages = tr.meta["progressive"]["stages"]
        self.stage_names = [s.split("x")[0] + "²" for s in stages]
        self.counts = [sum(1 for s in tr.steps if s["stage"] == k) for k in range(len(stages))]

    def regions(self):
        x, y, s = self.PREV
        cx, cy, cs = self.CROP
        return [(x, y, x + s, y + s), (cx, cy, cx + cs, cy + cs)]

    def frame(self, i):
        tr, st = self.tr, self.tr.steps[i]
        im = Image.new("RGB", (R.FW, R.FH), R.WHITE)
        d = ImageDraw.Draw(im)
        x, y, s = self.PREV
        R.paste_image(im, d, tr.image("preview", i), x, y, s, tr.box, tr.size[0], "predicted clean image")
        X0, X1 = self.X0, self.X1
        R.text(d, (X0, 28), "Progressive SCA, 4096 × 4096", 36, bold=True)
        R.text(d, (X1, 28), f"step {i + 1} / {len(tr.steps)}", 36, bold=True, anchor="ra")
        px = X0
        for k, name in enumerate(self.stage_names):
            on = st["stage"] == k
            px += R.pill(d, px, 92, name, 24, R.INK if on else R.WHITE, R.WHITE if on else R.MID,
                         outline=None if on else R.RULE, h=46) + 10
        label, fill, fg, outline = phase_style(st)
        R.pill(d, px + 16, 92, label, 24, fill, fg, outline=outline, h=46)
        R.text(d, (X1, 115), R.fmt_sigma(st["sigma"]), 30, anchor="rm")
        self.timeline(d, i)
        R.text(d, (X0, 318), "π for 57 blocks × 24 heads", 24, R.MID)
        im.paste(R.colorbar(150, 14), (X1 - 150 - 64, 324))
        R.text(d, (X1 - 150 - 72, 331), "global", 20, R.MID, anchor="rm")
        R.text(d, (X1 - 56, 331), "local", 20, R.MID, anchor="lm")
        cw, ch = self.CELL
        hx, hy, hw, hh = X0, 352, 57 * cw, 24 * ch
        if st.get("pi") is not None:
            im.paste(R.pi_strip(st["pi"], self.order, cw, ch), (hx, hy))
        else:
            d.rectangle((hx, hy, hx + hw - 1, hy + hh - 1), fill=R.GREY)
            msg = ("the model's own attention at 1024²" if st.get("phase") == "native"
                   else f"unused while σ > {self.gate:g}: global only")
            R.text(d, (hx + hw / 2, hy + hh / 2), msg, 22, R.MID, anchor="mm")
        cx, cy, cs = self.CROP
        R.paste_image(im, d, tr.image("crop", i), cx, cy, cs)
        d.rectangle((cx - 2, cy - 2, cx + cs + 1, cy + cs + 1), outline=R.ZOOM, width=3)
        scale = st.get("crop_scale", 1)
        R.text(d, (cx + cs + 24, cy + 2), "1:1 crop" if scale == 1 else f"crop, {scale:g}× enlarged", 26, bold=True)
        R.text(d, (cx + cs + 24, cy + 40), f"512 px at 4096²" if scale == 1 else f"same region at {st['stage_res'].split('x')[0]}²", 22, R.MID)
        R.text(d, (cx + cs + 24, cy + 70), "yellow box on the left", 22, R.MID)
        return im

    def timeline(self, d, i):
        X0, X1, n = self.X0, self.X1, len(self.tr.steps)
        gap = 18
        a, b, axis = X0 + 10, X1 - 10, 232
        unit = (b - a - gap * (len(self.counts) - 1)) / max(n - 1, 1)
        xs, k0 = [], 0
        for k, c in enumerate(self.counts):
            xs += [a + (k0 + j) * unit + k * gap for j in range(c)]
            k0 += c
        d.line((a, axis, b, axis), fill=R.RULE, width=2)
        for k, xk in enumerate(xs):
            color = tick_color(self.tr.steps[k]) if k <= i else R.RULE
            if k == i:
                d.rectangle((xk - 3, axis - 22, xk + 3, axis + 22), fill=color)
                d.polygon([(xk - 9, axis - 36), (xk + 9, axis - 36), (xk, axis - 26)], fill=R.INK)
            else:
                d.rectangle((xk - 2, axis - 12, xk + 2, axis + 12), fill=color)
        k0 = 0
        for k, c in enumerate(self.counts):
            mid = (xs[k0] + xs[k0 + c - 1]) / 2
            R.text(d, (mid, 266), f"{self.stage_names[k]}: {c} steps", 22, R.MID, anchor="ma")
            k0 += c


class ProgVsLayout:
    P, C, PY, CY = 640, 312, 92, 746
    XS = (960 - 16 - 640, 976)

    def __init__(self, prog, area):
        self.tr = (prog, area)
        self.n = len(prog.steps)

    def regions(self):
        out = []
        for x in self.XS:
            cx = x + (self.P - self.C) // 2
            out += [(x, self.PY, x + self.P, self.PY + self.P), (cx, self.CY, cx + self.C, self.CY + self.C)]
        return out

    def frame(self, i):
        prog, area = self.tr
        ia = min(len(area.steps) - 1, (i * len(area.steps)) // self.n)
        im = Image.new("RGB", (R.FW, R.FH), R.WHITE)
        d = ImageDraw.Draw(im)
        R.text(d, (28, 26), "Progressive SCA vs AReA at 4096 × 4096, same seed", 34, bold=True)
        R.text(d, (1892, 26), "aligned by progress, not wall time", 24, R.MID, anchor="ra")
        for k, (x, tr, j) in enumerate(zip(self.XS, self.tr, (i, ia))):
            st = tr.steps[j]
            left = k == 0
            lx = x - 24 if left else x + self.P + 24
            name, color = ("SCA", R.SCA) if left else ("AReA", R.AREA)
            R.text(d, (lx, self.PY), name, 36, color, bold=True, anchor="ra" if left else "la")
            R.text(d, (lx, self.PY + 52), f"step {j + 1} / {len(tr.steps)}", 24, R.INK, bold=True,
                   anchor="ra" if left else "la")
            if left:
                label, fill, fg, outline = phase_style(st)
                detail = st["stage_res"].split("x")[0] + "² stage"
            else:
                label, fill, fg, outline, detail = R.VsLayout.phase("area", st)
            pw = int(d.textlength(label, font=R.font(22, True))) + 2 * int(22 * 0.9)
            R.pill(d, lx - pw if left else lx, self.PY + 96, label, 22, fill, fg, outline=outline, h=44)
            if detail:
                R.text(d, (lx, self.PY + 160), detail, 22, R.MID, anchor="ra" if left else "la")
            R.paste_image(im, d, tr.image("preview", j), x, self.PY, self.P, tr.box, tr.size[0])
            cx = x + (self.P - self.C) // 2
            R.paste_image(im, d, tr.image("crop", j), cx, self.CY, self.C)
            d.rectangle((cx - 2, self.CY - 2, cx + self.C + 1, self.CY + self.C + 1), outline=R.ZOOM, width=3)
        return im


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", help="prompt ids (default: every complete progressive trace)")
    p.add_argument("--traces", type=Path, default=TRACES)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--step-s", type=float, default=0.3, help="seconds per step")
    p.add_argument("--crf", type=int, default=R.CRF)
    p.add_argument("--max-bytes", type=int, default=R.MAX_BYTES)
    a = p.parse_args(argv)
    R.STEP_S = a.step_s
    ids = a.ids or sorted(q.name for q in (a.traces / "sca_v2_prog").iterdir() if complete(q))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for pid in ids:
        prog_dir, area_dir = a.traces / "sca_v2_prog" / pid, a.traces / "area_sc" / pid
        if not complete(prog_dir):
            print(f"skip {pid}: no complete progressive trace", flush=True)
            continue
        prog = Trace(prog_dir)
        done[pid] = {"prog": R.render(ProgLayout(prog), len(prog.steps), a.out_dir / f"{pid}_prog.mp4", a)}
        if complete(area_dir):
            area = Trace(area_dir)
            done[pid]["prog_vs_area"] = R.render(ProgVsLayout(prog, area), len(prog.steps),
                                                 a.out_dir / f"{pid}_prog_vs_area.mp4", a)
    (a.out_dir / "videos.json").write_text(json.dumps(done, indent=1) + "\n")


if __name__ == "__main__":
    main()
