"""SCA method diagram (fig:method) as TikZ, compiled to figures/method.pdf.

The figure is generated as figures/src/method_diagram.tex (paper fonts:
Helvetica text, Computer Modern math, colours from sca_colors.tex) and
compiled with tectonic.

Panel a draws a small target grid at one scale (s = 4, n0 = 8, t = 2, so 32 x 32
tokens): one t x t query tile, its native n0 x n0 window (start a(tau) of
Section 2.2) and the n0 x n0 block-centre keys kappa(b) = s b + floor(s/2) of
the global branch. FLUX.1 itself has n0 = 64 and t = 8.

Panel b carries the real data: the weight pi_h(sigma) that each of the 24 heads
of block 29 (zero-based 28) of the archived scale-4 table uses at each of the 28
sampling steps. Steps with sigma > sigma_g use the global branch only (weight 0);
later steps interpolate the table linearly in sigma between bin centres, as
sca/attn.py PiTable.__call__ does at inference. The 28 noise levels follow the
FLUX.1 shifted schedule (mu = 1.15). Heads are in original order.

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
S, N0, TQ = 4, 8, 2  # drawn example of panel a: scale, native grid side, query tile side
TILE = (8, 6)        # drawn query tile index (x, y)
MIN_PT = 6.5         # smallest allowed glyph size in PDF points (figure is placed at 100%)


def rgb(hexcolor: str) -> str:
    h = hexcolor.lstrip("#")
    return "{rgb,255:red,%d;green,%d;blue,%d}" % (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def schedule(steps: int = STEPS) -> list[float]:
    t = np.linspace(1.0, 1.0 / steps, steps)
    return [float(math.exp(MU) / (math.exp(MU) + (1.0 / s - 1.0))) for s in t]


def pi_at(table: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Per-head weight at noise level sigma, as sca/attn.py PiTable.__call__."""
    j = int(np.searchsorted(centers, sigma))
    if j <= 0:
        return table[..., 0]
    if j >= len(centers):
        return table[..., -1]
    a = (sigma - centers[j - 1]) / (centers[j] - centers[j - 1])
    return (1 - a) * table[..., j - 1] + a * table[..., j]


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


def main() -> None:
    scafig.write_tikz_colors(str(HERE / "sca_colors.tex"))
    z = np.load(DATA)
    table = z["s4_pi_head"]
    centers = np.asarray(z["sigma_centers"], dtype=np.float64)
    assert table.shape == (57, 24, 14) and np.isfinite(table).all()
    assert np.all((table >= 0) & (table <= 1)) and np.all(np.diff(centers) > 0)
    a, b = z["s4_stats"][:2]
    fitted = np.clip(np.divide(b, a, out=np.ones_like(a), where=a > 0), 0, 1)
    populated = np.flatnonzero(z["populated"])
    nearest = populated[np.abs(np.arange(14)[:, None] - populated[None, :]).argmin(1)]
    np.testing.assert_allclose(table, fitted[..., nearest], rtol=1e-12, atol=1e-12)
    sig = schedule()
    gated = [k for k, s in enumerate(sig) if s > SIGMA_G]
    ng = len(gated)
    assert gated == list(range(8)), gated
    pi = np.stack([np.zeros(24) if k < ng else pi_at(table[BLOCK], centers, s)
                   for k, s in enumerate(sig)], axis=1)          # 24 heads x 28 steps

    # drawn example of panel a, from the definitions in Section 2.2
    nt = S * N0
    def start(tau: int) -> int:                                   # window start a(tau) per axis
        return int(np.clip(TQ * tau + TQ // 2 - N0 // 2, 0, nt - N0))
    kappa = [S * k + S // 2 for k in range(N0)]                   # block-centre token per axis
    wx, wy = start(TILE[0]), start(TILE[1])
    tx, ty = TQ * TILE[0], TQ * TILE[1]
    assert wx <= tx and tx + TQ <= wx + N0 and wy <= ty and ty + TQ <= wy + N0
    assert not any(tx <= i < tx + TQ for i in kappa) or not any(ty <= j < ty + TQ for j in kappa)

    t = Tikz()
    AW, GAP = 2.2, 0.06
    G, gy = 1.40, 0.29
    gx, c = 0.12, G / nt
    ly = [gy + G + 0.135 + k * 0.148 for k in range(4)]
    H = ly[-1] + 0.115

    # ---------------- panel a: one query tile at the target, one scale -------------
    t.rect(0, 0, AW, H, "panel")
    t.text(0.09, 0.145, r"\textbf{a}\hspace{6pt}\textbf{Keys of one query tile, to scale}", "ttl")
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
    sw, lx = 0.085, 0.12
    lt = lx + sw + 0.06
    for y, style in ((ly[0], "fill=scaInk"), (ly[1], "fill=scaLocalLight,draw=scaLocal,line width=0.8pt"),
                     (ly[2], "fill=scaGlobal")):
        t.rect(lx, y - sw / 2, lx + sw, y + sw / 2, style)
    t.text(lt, ly[0], r"query tile, $t\times t$", "lbl")
    px = gx + G + 0.08                                            # drawn parameters, right of the grid
    t.text(px, gy + 0.07, "drawn with", "note")
    for k, s_ in enumerate((r"$s=%d$" % S, r"$n_0=%d$" % N0, r"$t=%d$" % TQ)):
        t.text(px, gy + 0.235 + k * 0.15, s_, "lbl")
    t.text(lt, ly[1], r"\textcolor{scaLocalText}{local:} $n_0\times n_0$ window, query at $p$", "lbl")
    t.text(lt, ly[2], r"\textcolor{scaGlobal}{global:} $n_0\times n_0$ block centres,", "lbl")
    t.text(lt, ly[3], r"query at $\lfloor p/s\rfloor$", "lbl")

    # ---------------- panel b: per-head mix and the 28-step schedule ----------------
    bx0 = AW + GAP
    t.rect(bx0, 0, W, H, "panel")
    t.text(bx0 + 0.09, 0.145, r"\textbf{b}\hspace{6pt}\textbf{Per-head mix and schedule}", "ttl")
    t.text((bx0 + W) / 2, 0.45, r"$O_h=\pi_h(\sigma)\,\textcolor{scaLocalText}{\Oloc_h}"
           r"+\bigl(1-\pi_h(\sigma)\bigr)\,\textcolor{scaGlobal}{\Oglob_h}$", "mixbox", "center")
    hx0, hx1, hy0, hy1 = bx0 + 0.45, W - 0.70, 0.88, H - 0.36
    cw, rh = (hx1 - hx0) / STEPS, (hy1 - hy0) / 24
    for h in range(24):
        for k in range(ng, STEPS):
            x, y = hx0 + k * cw, hy0 + h * rh
            # each cell overlaps its right and lower neighbours, which are drawn later: no seams
            x1 = hx1 if k == STEPS - 1 else x + cw + 0.005
            y1 = hy1 if h == 23 else y + rh + 0.005
            t.rect(x, y, x1, y1, "fill=%s" % rgb(scafig.pi_color(pi[h, k])))
    t.rect(hx0, hy0, hx0 + ng * cw, hy1, "fill=%s" % rgb(scafig.pi_color(0.0)))
    t.rect(hx0, hy0, hx1, hy1, "draw=scaMid,line width=0.4pt")
    xm, ym = hx0 + ng * cw / 2, (hy0 + hy1) / 2
    t.text(xm, ym - 0.07, "global", "lblw", "center")
    t.text(xm, ym + 0.07, "only", "lblw", "center")
    xg = hx0 + ng * cw
    t.line([(xg, hy0 - 0.06), (xg, hy1)], "draw=scaInk,line width=0.6pt")
    t.text(xg, hy0 - 0.085, r"$\sigma_g=%.2f$" % SIGMA_G, "lbl", "south")
    for k in (1, 8, 14, 21, 28):
        x = hx0 + (k - 0.5) * cw
        t.line([(x, hy1), (x, hy1 + 0.035)], "tick")
        t.text(x, hy1 + 0.055, str(k), "tck", "north")
    t.text((hx0 + hx1) / 2, hy1 + 0.2, "denoising step", "lbl", "north")
    for h in (1, 12, 24):
        y = hy0 + (h - 0.5) * rh
        t.line([(hx0 - 0.035, y), (hx0, y)], "tick")
        t.text(hx0 - 0.055, y, str(h), "tck", "east")
    t.text(bx0 + 0.15, ym, "head, block %d" % (BLOCK + 1), "lbl", "center", ",rotate=90")
    cb0, cb1, n = hx1 + 0.13, hx1 + 0.205, 48
    for k in range(n):
        y1 = hy1 - k * (hy1 - hy0) / n
        y0 = hy0 if k == n - 1 else y1 - (hy1 - hy0) / n - 0.005
        t.rect(cb0, y0, cb1, y1, "fill=%s" % rgb(scafig.pi_color((k + .5) / n)))
    t.rect(cb0, hy0, cb1, hy1, "draw=scaMid,line width=0.4pt")
    for v, lab in ((1.0, r"1 \textcolor{scaLocalText}{local}"), (0.5, "0.5"),
                   (0.0, r"0 \textcolor{scaGlobal}{global}")):
        y = hy1 - v * (hy1 - hy0)
        t.line([(cb1, y), (cb1 + 0.03, y)], "tick")
        t.text(cb1 + 0.05, y, lab, "tck", "west")
    t.text((cb0 + cb1) / 2, hy0 - 0.085, r"$\pi_h(\sigma)$", "lbl", "south")

    doc = r"""%% Generated by fig_method.py; edit the script, not this file.
\RequirePackage{fix-cm}
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\DeclareMathSizes{7}{7}{6.75}{6.75}
\DeclareMathSizes{8}{8}{6.75}{6.75}
\DeclareMathSizes{9}{9}{7}{6.75}
\DeclareMathSizes{10}{10}{7}{6.75}
\input{sca_colors.tex}
\colorlet{scaLocalText}{scaLocal!85!black}
\newcommand{\Oloc}{O^{\mathrm{loc}}}
\newcommand{\Oglob}{O^{\mathrm{glob}}}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  panel/.style={fill=scaGrid!55,rounded corners=3pt},
  mixbox/.style={fill=white,draw=scaAccent,line width=0.9pt,rounded corners=2.5pt,
    inner xsep=6pt,inner ysep=4pt,font=\small,text=scaInk},
  cellln/.style={draw=scaGrid,line width=0.25pt},
  blockln/.style={draw=scaRule,line width=0.45pt},
  tick/.style={draw=scaInk,line width=0.4pt},
  ttl/.style={font=\sffamily\small,text=scaInk,inner sep=0pt},
  lbl/.style={font=\sffamily\footnotesize,text=scaInk,inner sep=0pt},
  lblw/.style={lbl,text=white},
  note/.style={font=\sffamily\footnotesize,text=scaMid,inner sep=0pt},
  tck/.style={font=\sffamily\scriptsize,text=scaInk,inner sep=0pt}]
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
        panel_a=dict(drawn_example=dict(s=S, n0=N0, t=TQ, target_side_tokens=nt),
                     tile_index_xy=list(TILE), tile_cells_xy=[tx, ty], window_start_xy=[wx, wy],
                     block_centre_cells=kappa,
                     rules="window start a(tau)=clip(t*tau+floor(t/2)-floor(n0/2),0,s*n0-n0); "
                           "block centre kappa(b)=s*b+floor(s/2) (Section 2.2)"),
        panel_b=dict(table="s4_pi_head", block_zero_based=BLOCK, block_label=BLOCK + 1,
                     heads="24, original order, head 1 at the top",
                     rule="steps with sigma > sigma_g drawn as 0 (global only); other steps interpolate "
                          "the table linearly in sigma between sigma_centers (sca/attn.py PiTable.__call__)",
                     sigma_centers=[round(float(v), 6) for v in centers],
                     pi_head_by_step=[[round(float(v), 4) for v in row] for row in pi]),
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
