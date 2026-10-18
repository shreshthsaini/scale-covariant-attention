"""SCA method diagram (fig:method) as TikZ, compiled to figures/method.pdf.

The schematic is generated as figures/src/method_diagram.tex (paper fonts:
Helvetica text, Computer Modern math, colours from sca_colors.tex) and
compiled with tectonic. Grids are schematic. Two elements carry real data:
the coefficient inset is block 28 (zero-based) of the archived scale-4 table
in original head/bin order, and the schedule strip colours steps 9 to 28 by
the mean table weight over all heads in each step's noise bin. The 28 noise
levels follow the FLUX.1 shifted schedule (mu = 1.15).

Run on compute from paper/figures/src (pymupdf only refreshes the SVG preview):
  UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache ~/.local/bin/uv run --no-project \
    --with numpy --with matplotlib --with pymupdf python fig_method.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

import scafig

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
DATA = HERE / "data" / "k1_1024_pi.npz"
TEX = HERE / "method_diagram.tex"
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")
W = 5.5
BLOCK = 28
SIGMA_G = 0.89
MU = 1.15


def rgb(hexcolor: str) -> str:
    h = hexcolor.lstrip("#")
    return "{rgb,255:red,%d;green,%d;blue,%d}" % (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def schedule(steps: int = 28) -> list[float]:
    t = np.linspace(1.0, 1.0 / steps, steps)
    return [float(math.exp(MU) / (math.exp(MU) + (1.0 / s - 1.0))) for s in t]


class Tikz:
    def __init__(self):
        self.lines: list[str] = []

    def add(self, s: str):
        self.lines.append(s)

    def rect(self, x0, y0, x1, y1, style):
        self.add(r"\path[%s] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (style, x0, -y0, x1, -y1))

    def line(self, pts, style):
        self.add(r"\draw[%s] %s;" % (style, " -- ".join("(%.4f,%.4f)" % (x, -y) for x, y in pts)))

    def text(self, x, y, s, style="lbl", anchor="west"):
        self.add(r"\node[%s,anchor=%s] at (%.4f,%.4f) {%s};" % (style, anchor, x, -y, s))

    def grid(self, x, y, side, n, border="scaMid"):
        self.rect(x, y, x + side, y + side, "fill=white")
        step = side / n
        for k in range(1, n):
            self.line([(x + k * step, y), (x + k * step, y + side)], "gridln")
            self.line([(x, y + k * step), (x + side, y + k * step)], "gridln")
        self.rect(x, y, x + side, y + side, "draw=%s,line width=0.5pt" % border)

    def dots(self, x, y, side, n, k, color, r):
        """Block-centre keys: one dot per (n/k x n/k)-cell block of an n-cell grid."""
        b = side / k
        for i in range(k):
            for j in range(k):
                self.add(r"\fill[%s] (%.4f,%.4f) circle (%.4f);" % (color, x + (i + .5) * b, -(y + (j + .5) * b), r))

    def window(self, x, y, side, n, wx, wy, wn, tx, ty, tn):
        c = side / n
        self.rect(x + wx * c, y + wy * c, x + (wx + wn) * c, y + (wy + wn) * c,
                  "fill=scaLocalLight,draw=scaLocal,line width=0.7pt")
        self.rect(x + tx * c, y + ty * c, x + (tx + tn) * c, y + (ty + tn) * c, "fill=scaInk")

    def tensor(self, x, y, w, h, color, light):
        d = 0.035
        for k in (2, 1):
            self.rect(x + k * d, y - k * d, x + w + k * d, y + h - k * d,
                      "fill=%s,draw=%s,line width=0.45pt" % (light, color))
        self.rect(x, y, x + w, y + h, "fill=white,draw=%s,line width=0.6pt" % color)
        for k in range(1, 4):
            self.line([(x, y + k * h / 4), (x + w, y + k * h / 4)], "draw=%s!45,line width=0.3pt" % color)

    def arrow(self, pts, style="arr"):
        self.line(pts, style)


def main() -> None:
    scafig.write_tikz_colors(str(HERE / "sca_colors.tex"))
    z = np.load(DATA)
    table = z["s4_pi_head"]
    assert table.shape == (57, 24, 14) and np.isfinite(table).all()
    assert np.all((table >= 0) & (table <= 1))
    a, b = z["s4_stats"][:2]
    fitted = np.clip(np.divide(b, a, out=np.ones_like(a), where=a > 0), 0, 1)
    populated = np.flatnonzero(z["populated"])
    nearest = populated[np.abs(np.arange(14)[:, None] - populated[None, :]).argmin(1)]
    np.testing.assert_allclose(table, fitted[..., nearest], rtol=1e-12, atol=1e-12)
    pi = table[BLOCK]                         # 24 heads x 14 bins, original order
    sig = schedule()
    assert sum(s > SIGMA_G for s in sig) == 8
    step_bin = [min(13, int(s * 14)) for s in sig]
    step_mean = [float(table[:, :, k].mean()) for k in step_bin]

    t = Tikz()
    # ---------------- panel a: calibration at native resolution ----------------------
    Ha = 1.20
    t.rect(0, 0, W, Ha, "panel")
    t.text(0.09, 0.14, r"\textbf{a}\hspace{6pt}\textbf{Calibrate once at native resolution}", "ttl")
    t.text(2.62, 0.145, r"frozen model; full attention drives the sampler", "note")
    gx, gy, gs = 0.16, 0.40, 0.56
    t.text(gx + gs / 2, gy - 0.075, r"$1024^2$ image", "small", "center")
    t.grid(gx, gy, gs, 8)
    t.text(gx + gs / 2, gy + gs + 0.09, r"$64\times64$ tokens", "tiny", "center")
    rows = [(0.42, "full", r"$\Ofull$\;\,full attention", "lbl"),
            (0.68, "glob", r"$\Oglob$\;\,$16^2$ centre keys", "lblg"),
            (0.94, "loc", r"$\Oloc$\;\,$16\times16$ window", "lbll")]
    ix, isz = 0.98, 0.19
    for yc, kind, label, style in rows:
        t.arrow([(gx + gs + 0.04, gy + gs / 2), (ix - 0.05, yc)])
        if kind == "full":
            t.rect(ix, yc - isz / 2, ix + isz, yc + isz / 2, "fill=scaMid!30")
            for k in range(1, 4):
                t.line([(ix + k * isz / 4, yc - isz / 2), (ix + k * isz / 4, yc + isz / 2)], "draw=white,line width=0.3pt")
                t.line([(ix, yc - isz / 2 + k * isz / 4), (ix + isz, yc - isz / 2 + k * isz / 4)], "draw=white,line width=0.3pt")
            t.rect(ix, yc - isz / 2, ix + isz, yc + isz / 2, "draw=scaMid,line width=0.5pt")
        elif kind == "glob":
            t.grid(ix, yc - isz / 2, isz, 4)
            t.dots(ix, yc - isz / 2, isz, 4, 4, "scaGlobal", 0.015)
        else:
            t.grid(ix, yc - isz / 2, isz, 8)
            t.window(ix, yc - isz / 2, isz, 8, 2, 2, 4, 3.5, 3.5, 1)
        t.text(ix + isz + 0.07, yc, label, style)
    t.text(ix, 1.12, r"emulated target scale $s_c=4$", "tiny", "west")
    # statistics and closed-form fit
    sx0, sx1, sy0, sy1 = 2.40, 4.24, 0.33, 1.03
    for yc, _, _, _ in rows:
        t.arrow([(2.21, yc), (sx0 - 0.04, yc)])
    t.rect(sx0, sy0, sx1, sy1, "box")
    cx = (sx0 + sx1) / 2
    t.text(cx, sy0 + 0.11, r"sums per block $l$, head $h$, noise bin $\beta$", "tiny", "center")
    t.text(cx, sy0 + 0.27, r"$A=\sum\|\Oloc-\Oglob\|^2$", "maths", "center")
    t.text(cx, sy0 + 0.43, r"$B=\sum\langle\Ofull-\Oglob,\,\Oloc-\Oglob\rangle$", "maths", "center")
    t.text(cx, sy0 + 0.60, r"$\pi^{\star}_{lh\beta}=\operatorname{clip}(B/A,\,0,\,1)$", "mathb", "center")
    # coefficient inset: real data
    hx0, hy0, hw, hh = 4.45, 0.39, 0.90, 0.46
    t.arrow([(sx1 + 0.04, 0.62), (hx0 - 0.05, 0.62)])
    cw, ch = hw / 14, hh / 24
    for h in range(24):
        for k in range(14):
            x = hx0 + (13 - k) * cw            # sigma decreases left to right
            y = hy0 + h * ch
            t.rect(x, y, x + cw + 0.0015, y + ch + 0.0015, "fill=%s" % rgb(scafig.pi_color(pi[h, k])))
    xg = hx0 + (1 - SIGMA_G) * hw
    t.rect(hx0, hy0, xg, hy0 + hh, "fill=white,fill opacity=0.6")
    t.rect(hx0, hy0, hx0 + hw, hy0 + hh, "draw=scaMid,line width=0.4pt")
    t.line([(xg, hy0), (xg, hy0 + hh)], "draw=scaInk,line width=0.45pt,dash pattern=on 1.4pt off 1pt")
    t.text(hx0 + hw / 2, hy0 - 0.075, r"fitted $\pi_{lh}(\sigma)$, block %d" % (BLOCK + 1), "tiny", "center")
    t.text(hx0, hy0 + hh + 0.07, r"$\sigma=1$", "tiny", "west")
    t.text(hx0 + hw, hy0 + hh + 0.07, r"$0$", "tiny", "east")
    ky = hy0 + hh + 0.15
    n = 24
    for k in range(n):
        x = hx0 + k * hw / n
        t.rect(x, ky, x + hw / n + 0.002, ky + 0.04, "fill=%s" % rgb(scafig.pi_color((k + .5) / n)))
    t.text(hx0, ky + 0.105, r"0 global", "tinyg", "west")
    t.text(hx0 + hw, ky + 0.105, r"local 1", "tinyl", "east")

    # ---------------- panel b: deployment at the target -----------------------------
    y0 = Ha + 0.08
    Hb = 1.56
    t.rect(0, y0, W, y0 + Hb, "panel")
    t.text(0.09, y0 + 0.14, r"\textbf{b}\hspace{6pt}\textbf{Apply the table at the target resolution}", "ttl")
    t.text(2.95, y0 + 0.145, r"example $4096^2$: $s=4$, $256\times256$ tokens", "note")
    ts = 0.48
    gyc, lyc = y0 + 0.53, y0 + 1.11
    t.grid(0.16, gyc - ts / 2, ts, 16)
    t.dots(0.16, gyc - ts / 2, ts, 16, 8, "scaGlobal", 0.012)
    t.text(0.74, gyc - 0.15, r"\textbf{global branch}", "lblg")
    t.text(0.74, gyc + 0.005, r"$64^2$ block-centre keys", "lbl")
    t.text(0.74, gyc + 0.15, r"positions $c(p)=\lfloor p/s\rfloor$", "lbl")
    t.grid(0.16, lyc - ts / 2, ts, 16)
    t.window(0.16, lyc - ts / 2, ts, 16, 5, 5, 4, 6.5, 6.5, 1)
    t.text(0.74, lyc - 0.15, r"\textbf{local branch}", "lbll")
    t.text(0.74, lyc + 0.005, r"native $64\times64$ window", "lbl")
    t.text(0.74, lyc + 0.15, r"per $8\times8$ query tile, positions $p$", "lbl")
    t.text(0.16, y0 + Hb - 0.085, r"Both branches also attend to all $T$ text keys; text queries use $\Oglob$.", "tiny", "west")
    ox = 2.62
    t.arrow([(2.42, gyc), (ox - 0.05, gyc)], "arrg")
    t.arrow([(2.42, lyc), (ox - 0.05, lyc)], "arrl")
    t.tensor(ox, gyc - 0.15, 0.17, 0.30, "scaGlobal", "scaGlobalLight")
    t.tensor(ox, lyc - 0.15, 0.17, 0.30, "scaLocal", "scaLocalLight")
    t.text(ox + 0.29, gyc - 0.02, r"$\Oglob_h$", "mathg", "west")
    t.text(ox + 0.29, lyc - 0.02, r"$\Oloc_h$", "mathl", "west")
    mx0, mx1, my0, my1 = 3.40, 5.40, y0 + 0.62, y0 + 0.98
    t.arrow([(ox + 0.60, gyc + 0.03), (mx0 - 0.04, my0 + 0.10)], "arrg")
    t.arrow([(ox + 0.60, lyc - 0.03), (mx0 - 0.04, my1 - 0.10)], "arrl")
    t.rect(mx0, my0, mx1, my1, "mixbox")
    t.text((mx0 + mx1) / 2, (my0 + my1) / 2,
           r"$O_h=\pi_h(\sigma)\,\Oloc_h+\bigl(1-\pi_h(\sigma)\bigr)\,\Oglob_h$", "maths", "center")
    tx = 5.26
    t.arrow([(tx, ky + 0.16), (tx, my0 - 0.04)], "arrt")
    t.text(tx - 0.07, y0 + 0.43, r"transfer $\pi_{lh}(\sigma)$", "tinya", "east")
    sx0s, sw = mx0, mx1 - mx0
    sy = y0 + 1.16
    cw = sw / 28
    for k, (s, m) in enumerate(zip(sig, step_mean)):
        fill = scafig.GLOBAL if s > SIGMA_G else scafig.pi_color(m)
        t.rect(sx0s + k * cw + 0.004, sy, sx0s + (k + 1) * cw - 0.004, sy + 0.085, "fill=%s" % rgb(fill))
    xg8 = sx0s + 8 * cw
    t.line([(xg8, sy - 0.035), (xg8, sy + 0.12)], "draw=scaInk,line width=0.5pt")
    t.text(sx0s, sy - 0.075, r"1 to 8: global only", "tinyg", "west")
    t.text(sx0s + sw, sy - 0.075, r"9 to 28: per-head mix", "tinyl", "east")
    t.text(xg8, sy + 0.165, r"$\sigma_g=%.2f$" % SIGMA_G, "tiny", "center")
    total = y0 + Hb

    doc = r"""%% Generated by fig_method.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\input{sca_colors.tex}
\newcommand{\Oloc}{O^{\mathrm{loc}}}
\newcommand{\Oglob}{O^{\mathrm{glob}}}
\newcommand{\Ofull}{O^{\mathrm{full}}}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  panel/.style={fill=scaGrid!55,rounded corners=3pt},
  box/.style={fill=white,draw=scaRule,line width=0.5pt,rounded corners=2pt},
  mixbox/.style={fill=white,draw=scaAccent,line width=0.9pt,rounded corners=2.5pt},
  gridln/.style={draw=scaRule,line width=0.25pt},
  arr/.style={draw=scaMid,line width=0.55pt,-{Stealth[length=3.4pt,width=2.8pt]}},
  arrg/.style={arr,draw=scaGlobal},
  arrl/.style={arr,draw=scaLocal},
  arrt/.style={arr,draw=scaAccent,line width=0.7pt,dash pattern=on 2.2pt off 1.4pt},
  ttl/.style={font=\sffamily\small,text=scaInk,inner sep=0pt},
  lbl/.style={font=\sffamily\footnotesize,text=scaInk,inner sep=0pt},
  lblg/.style={lbl,text=scaGlobal},
  lbll/.style={lbl,text=scaLocal!85!black},
  note/.style={font=\sffamily\footnotesize,text=scaMid,inner sep=0pt},
  notec/.style={note},
  small/.style={font=\sffamily\footnotesize,text=scaInk,inner sep=0pt},
  tiny/.style={font=\sffamily\scriptsize,text=scaMid,inner sep=0pt},
  tinyg/.style={tiny,text=scaGlobal},
  tinyl/.style={tiny,text=scaLocal!85!black},
  tinym/.style={tiny},
  math/.style={font=\small,text=scaInk,inner sep=0pt},
  maths/.style={font=\footnotesize,text=scaInk,inner sep=0pt},
  tinya/.style={tiny,text=scaAccent},
  mathb/.style={font=\footnotesize,text=scaAccent,inner sep=0pt},
  mathg/.style={font=\small,text=scaGlobal,inner sep=0pt},
  mathl/.style={font=\small,text=scaLocal!85!black,inner sep=0pt}]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (W, -total, "\n".join(t.lines))
    TEX.write_text(doc)
    subprocess.run([TECTONIC, "-X", "compile", TEX.name, "--outdir", str(HERE)], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    built = HERE / "method_diagram.pdf"
    out = FIG / "method.pdf"
    out.write_bytes(built.read_bytes())
    built.unlink()
    try:  # SVG preview; needs pymupdf (uv run --with pymupdf ...)
        import pymupdf
        (FIG / "method.svg").write_text(pymupdf.open(out)[0].get_svg_image(text_as_path=False))
    except ImportError:
        print("pymupdf missing: method.svg not refreshed")
    prov = dict(script="paper/figures/src/fig_method.py", tikz="paper/figures/src/method_diagram.tex",
                data=str(DATA.relative_to(FIG.parent.parent)),
                data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                inset="s4_pi_head[%d] (zero-based block), 24 heads x 14 bins, original order; "
                      "bins drawn with sigma decreasing left to right" % BLOCK,
                schedule=dict(mu=MU, steps=28, sigma=[round(s, 4) for s in sig],
                              steps_above_gate=sum(s > SIGMA_G for s in sig)),
                step_mean_pi=[round(m, 4) for m in step_mean],
                height_in=round(total, 4), width_in=W,
                pdf_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (FIG / "method_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"method.pdf {out.stat().st_size/1e3:.0f} kB, {W} x {total:.3f} in")


if __name__ == "__main__":
    main()
