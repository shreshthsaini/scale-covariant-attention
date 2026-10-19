raise SystemExit("Retired: figures/method.pdf is now built from figures/src/method.tex (TikZ). Do not run.")
"""SCA method diagram (fig:method) as TikZ, compiled to figures/method.pdf.

The figure is generated as figures/src/method_diagram.tex (paper fonts:
Helvetica text, Computer Modern math, colours from sca_colors.tex) and
compiled with tectonic.

Panel a (calibration, schematic icons plus real data): the native 1024^2 run,
the three operators at emulated scale s_c = 4, the closed-form weight and the
fitted table of block 29 (zero-based 28) of the archived scale-4 table: 24
heads in original order by 14 noise bins, sigma decreasing left to right. Bin 0
is unpopulated and holds the nearest populated bin, as in the stored table. The
region sigma > sigma_g, which inference never reads, is veiled.

Panel b (apply): the per-head mix, the 28-step schedule (steps with
sigma > sigma_g use the global branch only; FLUX.1 shifted schedule, mu = 1.15)
and a small target grid drawn at one scale (s = 4, n0 = 8, t = 2, so 32 x 32
tokens): one t x t query tile, its native n0 x n0 window (start a(tau) of
Section 2.2) and the n0 x n0 block-centre keys kappa(b) = s b + floor(s/2).
FLUX.1 itself has n0 = 64 and t = 8.

Run on compute from paper/figures/src (pymupdf refreshes the SVG preview and
checks that no text is smaller than MIN_PT or outside the page):
  UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache ~/.local/bin/uv run --no-project \
    --with numpy --with matplotlib --with pymupdf python fig_method.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
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
BLOCK = 28          # zero-based index; labelled "block 29"
SIGMA_G = 0.89
MU = 1.15
STEPS = 28
S, N0, TQ = 4, 8, 2  # drawn example of panel b: scale, native grid side, query tile side
TILE = (8, 6)        # drawn query tile index (x, y)
MIN_PT = 6.5         # smallest allowed glyph size in PDF points (figure is placed at 100%)


def rgb(hexcolor: str) -> str:
    h = hexcolor.lstrip("#")
    return "{rgb,255:red,%d;green,%d;blue,%d}" % (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def schedule(steps: int = STEPS) -> list[float]:
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

    def segs(self, segs, style):
        if segs:
            self.add(r"\draw[%s] %s;" % (style, " ".join(
                "(%.4f,%.4f) -- (%.4f,%.4f)" % (a, -b, c, -d) for (a, b), (c, d) in segs)))

    def text(self, x, y, s, style="lbl", anchor="west", extra=""):
        self.add(r"\node[%s,anchor=%s%s] at (%.4f,%.4f) {%s};" % (style, anchor, extra, x, -y, s))

    def grid(self, x, y, side, n):
        self.rect(x, y, x + side, y + side, "fill=white")
        step = side / n
        self.segs([((x + k * step, y), (x + k * step, y + side)) for k in range(1, n)]
                  + [((x, y + k * step), (x + side, y + k * step)) for k in range(1, n)], "icoln")
        self.rect(x, y, x + side, y + side, "draw=scaMid,line width=0.5pt")


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
    pi = table[BLOCK]                                             # 24 heads x 14 bins, original order
    sig = schedule()
    gated = [k for k, s in enumerate(sig) if s > SIGMA_G]
    ng = len(gated)
    assert gated == list(range(8)), gated

    # drawn example of panel b, from the definitions in Section 2.2
    nt = S * N0
    def start(tau: int) -> int:                                   # window start a(tau) per axis
        return int(np.clip(TQ * tau + TQ // 2 - N0 // 2, 0, nt - N0))
    kappa = [S * k + S // 2 for k in range(N0)]                   # block-centre token per axis
    wx, wy = start(TILE[0]), start(TILE[1])
    tx, ty = TQ * TILE[0], TQ * TILE[1]
    assert wx <= tx and tx + TQ <= wx + N0 and wy <= ty and ty + TQ <= wy + N0
    assert not any(tx <= i < tx + TQ for i in kappa) or not any(ty <= j < ty + TQ for j in kappa)

    t = Tikz()
    AW, GAP = 2.0, 0.06
    bx0 = AW + GAP
    G, gy = 1.30, 1.15                                            # panel b target grid
    H = gy + G + 0.09

    # ---------------- panel a: calibrate once at the native resolution -------------
    t.rect(0, 0, AW, H, "panel")
    t.text(0.09, 0.145, r"\textbf{a}\hspace{6pt}\textbf{Calibrate once at 1024\textsuperscript{2}}", "ttl")
    nx, ny, ns = 0.12, 0.30, 0.44                                 # native run (schematic)
    t.grid(nx, ny, ns, 8)
    t.text(nx + ns / 2, ny + ns + 0.05, "native run", "lbl", "north")
    ix, isz, rows = 0.86, 0.16, (0.36, 0.55, 0.74)
    for yc in rows:
        t.line([(nx + ns + 0.04, ny + ns / 2), (ix - 0.04, yc)], "arr")
    y = rows[0] - isz / 2                                         # full attention: every key
    t.rect(ix, y, ix + isz, y + isz, "fill=scaMid!30")
    t.segs([((ix + k * isz / 4, y), (ix + k * isz / 4, y + isz)) for k in range(1, 4)]
           + [((ix, y + k * isz / 4), (ix + isz, y + k * isz / 4)) for k in range(1, 4)],
           "draw=white,line width=0.3pt")
    t.rect(ix, y, ix + isz, y + isz, "draw=scaMid,line width=0.5pt")
    y = rows[1] - isz / 2                                         # global: one key per block
    t.grid(ix, y, isz, 4)
    for i in range(4):
        for j in range(4):
            t.add(r"\fill[scaGlobal] (%.4f,%.4f) circle (0.014);" % (ix + (i + .5) * isz / 4, -(y + (j + .5) * isz / 4)))
    y = rows[2] - isz / 2                                         # local: window around a tile
    t.grid(ix, y, isz, 8)
    c8 = isz / 8
    t.rect(ix + 2 * c8, y + 2 * c8, ix + 6 * c8, y + 6 * c8, "fill=scaLocalLight,draw=scaLocal,line width=0.6pt")
    t.rect(ix + 3.5 * c8, y + 3.5 * c8, ix + 4.5 * c8, y + 4.5 * c8, "fill=scaInk")
    sx = ix + isz + 0.06
    for yc, sym, style in zip(rows, (r"$\Ofull$", r"$\Oglob$", r"$\Oloc$"), ("math", "mathg", "mathl")):
        t.text(sx, yc, sym, style)
    t.text(ix + 0.28, rows[2] + isz / 2 + 0.03, r"emulated $s_c=4$", "note", "north")
    fy = 1.085                                                    # closed-form weight
    t.text(AW / 2 - 0.06, fy, r"$\pi^{\star}=\operatorname{clip}(B/A,\,0,\,1)$", "fbox", "center")
    bx = sx + 0.39                                                # brace: all three operators feed the fit
    t.add(r"\draw[brace] (%.4f,%.4f) -- (%.4f,%.4f);" % (bx, -(rows[0] - 0.085), bx, -(rows[2] + 0.085)))
    t.line([(bx + 0.07, rows[1]), (1.82, rows[1]), (1.82, fy), (1.62, fy)], "arr")
    hx0, hx1, hy0, hy1 = 0.38, 1.38, 1.36, 2.22                   # fitted table, real data
    t.line([(0.90, fy + 0.11), (0.90, hy0 - 0.03)], "arr")
    cw, rh = (hx1 - hx0) / 14, (hy1 - hy0) / 24
    for h in range(24):
        for k in range(14):
            x = hx0 + (13 - k) * cw                               # sigma decreases left to right
            x1 = hx1 if k == 0 else x + cw + 0.005
            y1 = hy1 if h == 23 else hy0 + (h + 1) * rh + 0.005
            t.rect(x, hy0 + h * rh, x1, y1, "fill=%s" % rgb(scafig.pi_color(pi[h, k])))
    xg = hx0 + (1 - SIGMA_G) * (hx1 - hx0)
    t.rect(hx0, hy0, xg, hy1, "fill=white,fill opacity=0.55")
    t.rect(hx0, hy0, hx1, hy1, "draw=scaMid,line width=0.4pt")
    t.line([(xg, hy0 - 0.05), (xg, hy1)], "draw=scaInk,line width=0.5pt,dash pattern=on 1.4pt off 1pt")
    t.text(xg, hy0 - 0.06, r"$\sigma_g$", "lbl", "south")
    for v in (1.0, 0.5, 0.0):
        x = hx0 + (1 - v) * (hx1 - hx0)
        t.line([(x, hy1), (x, hy1 + 0.035)], "tick")
        t.text(x, hy1 + 0.055, "%g" % v, "tck", "north")
    t.text((hx0 + hx1) / 2, hy1 + 0.17, r"noise level $\sigma$", "lbl", "north")
    for h in (1, 12, 24):
        yy = hy0 + (h - 0.5) * rh
        t.line([(hx0 - 0.035, yy), (hx0, yy)], "tick")
        t.text(hx0 - 0.055, yy, str(h), "tck", "east")
    t.text(0.13, (hy0 + hy1) / 2, "head, block %d" % (BLOCK + 1), "lbl", "center", ",rotate=90")
    cb0, cb1, n = hx1 + 0.08, hx1 + 0.145, 48
    for k in range(n):
        y1 = hy1 - k * (hy1 - hy0) / n
        y0 = hy0 if k == n - 1 else y1 - (hy1 - hy0) / n - 0.005
        t.rect(cb0, y0, cb1, y1, "fill=%s" % rgb(scafig.pi_color((k + .5) / n)))
    t.rect(cb0, hy0, cb1, hy1, "draw=scaMid,line width=0.4pt")
    for v, lab in ((1.0, r"1 \textcolor{scaLocalText}{local}"), (0.5, "0.5"),
                   (0.0, r"0 \textcolor{scaGlobal}{global}")):
        yy = hy1 - v * (hy1 - hy0)
        t.line([(cb1, yy), (cb1 + 0.03, yy)], "tick")
        t.text(cb1 + 0.045, yy, lab, "tck", "west")
    t.text((cb0 + cb1) / 2, hy0 - 0.06, r"$\pi_h(\sigma)$", "lbl", "south")

    # ---------------- panel b: apply at the target -------------------------------
    t.rect(bx0, 0, W, H, "panel")
    t.text(bx0 + 0.09, 0.145, r"\textbf{b}\hspace{6pt}\textbf{Apply at the target}", "ttl")
    t.text((bx0 + W) / 2, 0.42, r"$O_h=\pi_h(\sigma)\,\textcolor{scaLocalText}{\Oloc_h}"
           r"+\bigl(1-\pi_h(\sigma)\bigr)\,\textcolor{scaGlobal}{\Oglob_h}$", "mixbox", "center")
    sx0, sx1, sy0, sy1 = bx0 + 0.22, W - 0.22, 0.70, 0.84         # 28-step schedule
    sw_ = (sx1 - sx0) / STEPS
    t.rect(sx0, sy0, sx0 + ng * sw_, sy1, "fill=scaGlobal")
    t.rect(sx0 + ng * sw_, sy0, sx1, sy1, "fill=scaAccentLight")
    t.segs([((sx0 + k * sw_, sy0), (sx0 + k * sw_, sy1)) for k in range(ng + 1, STEPS)],
           "draw=white,line width=0.6pt,opacity=0.75")
    t.text(sx0 + ng * sw_ / 2, (sy0 + sy1) / 2, "global only", "lblw", "center")
    t.text(sx0 + (ng + STEPS) * sw_ / 2, (sy0 + sy1) / 2, "per-head mix", "lbla", "center")
    for k in (1, 8, 14, 21, 28):
        x = sx0 + (k - 0.5) * sw_
        t.line([(x, sy1), (x, sy1 + 0.035)], "tick")
        t.text(x, sy1 + 0.055, str(k), "tck", "north")
    t.text((sx0 + sx1) / 2, sy1 + 0.165, "denoising step", "lbl", "north")
    gx = bx0 + 0.12                                               # target grid at one scale
    c = G / nt
    t.rect(gx, gy, gx + G, gy + G, "fill=white")
    t.rect(gx + wx * c, gy + wy * c, gx + (wx + N0) * c, gy + (wy + N0) * c, "fill=scaLocalLight")
    t.segs([((gx + k * c, gy), (gx + k * c, gy + G)) for k in range(1, nt) if k % S]
           + [((gx, gy + k * c), (gx + G, gy + k * c)) for k in range(1, nt) if k % S], "cellln")
    t.segs([((gx + k * c, gy), (gx + k * c, gy + G)) for k in range(S, nt, S)]
           + [((gx, gy + k * c), (gx + G, gy + k * c)) for k in range(S, nt, S)], "blockln")
    for i in kappa:
        for j in kappa:
            t.rect(gx + i * c, gy + j * c, gx + (i + 1) * c, gy + (j + 1) * c, "fill=scaGlobal")
    t.rect(gx + tx * c, gy + ty * c, gx + (tx + TQ) * c, gy + (ty + TQ) * c, "fill=scaInk")
    t.rect(gx + wx * c, gy + wy * c, gx + (wx + N0) * c, gy + (wy + N0) * c,
           "draw=scaLocal,line width=0.8pt")
    t.rect(gx, gy, gx + G, gy + G, "draw=scaMid,line width=0.5pt")
    sw, lx = 0.085, gx + G + 0.14
    lt, ly = lx + sw + 0.06, [gy + 0.08 + k * 0.152 for k in range(5)]
    for yy, style in ((ly[0], "fill=scaInk"), (ly[1], "fill=scaLocalLight,draw=scaLocal,line width=0.8pt"),
                      (ly[3], "fill=scaGlobal")):
        t.rect(lx, yy - sw / 2, lx + sw, yy + sw / 2, style)
    t.text(lt, ly[0], r"query tile, $t\times t$", "lbl")
    t.text(lt, ly[1], r"\textcolor{scaLocalText}{local:} $n_0\times n_0$ window,", "lbl")
    t.text(lt, ly[2], r"query at $p$", "lbl")
    t.text(lt, ly[3], r"\textcolor{scaGlobal}{global:} $n_0\times n_0$ block centres,", "lbl")
    t.text(lt, ly[4], r"query at $\lfloor p/s\rfloor$", "lbl")
    t.text(lx, ly[4] + 0.26, "drawn to scale with", "note")
    t.text(lx, ly[4] + 0.41, r"$s=%d$, $n_0=%d$, $t=%d$" % (S, N0, TQ), "note")

    doc = r"""%% Generated by fig_method.py; edit the script, not this file.
\RequirePackage{fix-cm}
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,decorations.pathreplacing}
\DeclareMathSizes{7}{7}{6.75}{6.75}
\DeclareMathSizes{8}{8}{6.75}{6.75}
\DeclareMathSizes{9}{9}{7}{6.75}
\DeclareMathSizes{10}{10}{7}{6.75}
\input{sca_colors.tex}
\colorlet{scaLocalText}{scaLocal!85!black}
\newcommand{\Oloc}{O^{\mathrm{loc}}}
\newcommand{\Oglob}{O^{\mathrm{glob}}}
\newcommand{\Ofull}{O^{\mathrm{full}}}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  panel/.style={fill=scaGrid!55,rounded corners=3pt},
  mixbox/.style={fill=white,draw=scaAccent,line width=0.9pt,rounded corners=2.5pt,
    inner xsep=6pt,inner ysep=4pt,font=\small,text=scaInk},
  fbox/.style={fill=white,draw=scaRule,line width=0.5pt,rounded corners=2pt,
    inner xsep=5pt,inner ysep=3pt,font=\small,text=scaAccent},
  icoln/.style={draw=scaRule,line width=0.25pt},
  cellln/.style={draw=scaGrid,line width=0.25pt},
  blockln/.style={draw=scaRule,line width=0.45pt},
  tick/.style={draw=scaInk,line width=0.4pt},
  arr/.style={draw=scaMid,line width=0.55pt,-{Stealth[length=3.4pt,width=2.8pt]}},
  brace/.style={draw=scaMid,line width=0.55pt,decorate,decoration={brace,amplitude=3pt}},
  ttl/.style={font=\sffamily\small,text=scaInk,inner sep=0pt},
  lbl/.style={font=\sffamily\footnotesize,text=scaInk,inner sep=0pt},
  lblw/.style={lbl,text=white},
  lbla/.style={lbl,text=scaAccent!80!black},
  note/.style={font=\sffamily\footnotesize,text=scaMid,inner sep=0pt},
  tck/.style={font=\sffamily\scriptsize,text=scaInk,inner sep=0pt},
  math/.style={font=\small,text=scaInk,inner sep=0pt},
  mathg/.style={math,text=scaGlobal},
  mathl/.style={math,text=scaLocalText}]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (W, -H, "\n".join(t.lines))
    TEX.write_text(doc)
    subprocess.run([TECTONIC, "-X", "compile", TEX.name, "--outdir", str(HERE)], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    built = HERE / "method_diagram.pdf"
    out = FIG / "method.pdf"
    out.write_bytes(built.read_bytes())
    built.unlink()
    min_pt = None
    try:  # SVG preview and text checks; need pymupdf (uv run --with pymupdf ...)
        import pymupdf
        page = pymupdf.open(out)[0]
        spans = [sp for blk in page.get_text("dict")["blocks"] for ln in blk.get("lines", [])
                 for sp in ln["spans"] if sp["text"].strip()]
        min_pt = round(min(sp["size"] for sp in spans), 2)
        assert min_pt >= MIN_PT - 1e-3, f"text below {MIN_PT} pt: {min_pt}"
        for sp in spans:
            assert page.rect.contains(pymupdf.Rect(sp["bbox"])), f"text outside the page: {sp['text']!r}"
        (FIG / "method.svg").write_text(page.get_svg_image(text_as_path=False))
    except ImportError:
        print("pymupdf missing: method.svg not refreshed, text sizes not checked")
    prov = dict(
        script="paper/figures/src/fig_method.py", tikz="paper/figures/src/method_diagram.tex",
        data=str(DATA.relative_to(FIG.parent.parent)),
        data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        panel_a=dict(icons="schematic: native run and the three operators at emulated scale s_c = 4",
                     table="s4_pi_head", block_zero_based=BLOCK, block_label=BLOCK + 1,
                     heads="24, original order, head 1 at the top",
                     bins="14 noise bins of width 1/14, sigma decreasing left to right; bin 0 is unpopulated "
                          "and holds the nearest populated bin, as stored; sigma > sigma_g veiled",
                     sigma_centers=[round(float(v), 6) for v in z["sigma_centers"]],
                     pi_head_by_bin=[[round(float(v), 4) for v in row] for row in pi]),
        panel_b=dict(drawn_example=dict(s=S, n0=N0, t=TQ, target_side_tokens=nt),
                     tile_index_xy=list(TILE), tile_cells_xy=[tx, ty], window_start_xy=[wx, wy],
                     block_centre_cells=kappa,
                     rules="window start a(tau)=clip(t*tau+floor(t/2)-floor(n0/2),0,s*n0-n0); "
                           "block centre kappa(b)=s*b+floor(s/2) (Section 2.2)",
                     schedule_strip="steps with sigma > sigma_g global only, the rest per-head mix"),
        schedule=dict(mu=MU, steps=STEPS, sigma=[round(s, 4) for s in sig], sigma_g=SIGMA_G,
                      global_only_steps=[k + 1 for k in gated]),
        height_in=round(H, 4), width_in=W, min_text_pt=min_pt,
        pdf_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    text = json.dumps(prov, indent=2)
    # one line per numeric list keeps the file short
    text = re.sub(r"\[\s*(-?[\d.]+(?:,\s*-?[\d.]+)*)\s*\]",
                  lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text)
    (FIG / "method_provenance.json").write_text(text + "\n")
    print(f"method.pdf {out.stat().st_size/1e3:.0f} kB, {W} x {H:.3f} in, smallest text {min_pt} pt")


if __name__ == "__main__":
    main()
