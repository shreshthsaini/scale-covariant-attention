"""Page-1 hero (fig:teaser): curated SCA outputs, a six-method strip, timing and metrics.

Top row: two curated SCA outputs from the 191 evaluation prompts (seed 0, 28 steps,
guidance 4.5), a 2048x6144 panorama and a 4096^2 interior. Each has two native-pixel zoom
tiles beside it; every crop window and its tile share one colour, and no leader line
crosses an image. All four tiles have the same printed size.

Bottom: one curated evaluation prompt at 4096^2 for six methods, the full image above the
same native-pixel crop, then a small table: sampling steps, seconds per image labelled
clean or campaign, and the 191-prompt mean ImageReward / HPSv2.1 (x100).

Timing (FLUX.1-dev, 4096^2, one GH200):
  clean    = batch-1 median of three prompts after warm-up on an idle GPU
             (results/latency_curve_summary.csv): full attention, AReA, SCA.
  campaign = mean over the 191-prompt evaluation campaign on shared nodes
             (results/e1_*_base_summary.csv, metric "seconds"): DyPE, SEGA, UltraImage.
Every number shown is checked against those CSVs (and the steps against the source
sidecars) before anything is written; a mismatch stops the build.

Everything is generated: JPEG assets (deterministic PIL encoding), a TikZ source with
vector boxes and labels, its PDF (tectonic), the snippet, and provenance with source
hashes and crop boxes. No metric chose any image.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_hero.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

from PIL import Image

import imgsel
import scafig

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent.parent
RESULTS = REPO / "results"
ASSETS = FIG / "teaser"
TEX = HERE / "teaser.tex"
OUT_PDF = FIG / "teaser.pdf"
SNIPPET = FIG / "snippets" / "teaser_six.tex"
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")

W = 5.5          # text width, inches
G = 0.045        # gutter inside a mosaic group (image and its tiles, stacked tiles)
G2 = 0.10        # gutter between the panorama group and the square group
M = 0.02         # margin of badges from image edges
DPI = 600        # pixel budget for downsampled full images
ZOOM_B = "38C6F4"  # second zoom colour (cyan); the first is the house scaZoom yellow

# Mosaic. Zoom windows are (x, y, side) in native pixels plus the colour of the window and
# its tile ("a" = house yellow, "b" = cyan); tiles stack top to bottom in list order.
# Panorama: the stone bridge with its crowd and the town houses, two of the prompt's subjects.
# Square: one of the near chandeliers and the rose window.
PANO = dict(method="sca_nb", res="2048x6144", pid="minus2742856837225126754",
            zooms=[(4064, 832, 384, "b"),     # houses with flower boxes
                   (2304, 1088, 384, "a")])   # crowd on the bridge and an arch
SQUARE = dict(method="sca_nb", res="4096x4096", pid="minus8737279395568207187",
              zooms=[(3420, 1520, 448, "b"),  # right chandelier
                     (1792, 1536, 448, "a")]) # rose window

# Six-method strip: one curated prompt; the crop window is shared by all methods.
# Chosen from the 191 evaluation prompts after viewing all six methods side by side (contact
# sheets, then candidates at full size) for a visible layout difference: full attention is a
# flat band, DyPE and SEGA lose the peaks and the hills under speckle, AReA's lower half
# dissolves into a streaked haze, and SCA and UltraImage keep peaks above rolling hills. The
# window sits on the main peak, where the methods differ most in snow and rock detail.
STRIP_PID = "1269964769307095670"
STRIP_BOX = (1664, 1408, 768)                 # x, y, side in native pixels
STRIP = [("full", "Full attention"), ("dype", "DyPE"), ("sega_official", "SEGA"),
         ("area", "AReA"), ("ultraimage", "UltraImage"), ("sca_nb", "SCA (ours)")]

# Numbers printed in the figure (exactly as specified for the paper); check_numbers()
# re-derives each from the result CSVs and the sidecars.
SECONDS = {"full": (186.5, "clean"), "dype": (196.1, "campaign"),
           "sega_official": (190.5, "campaign"), "area": (202.4, "clean"),
           "ultraimage": (308.5, "campaign"), "sca_nb": (71.1, "clean")}
STEPS = {"full": "28", "dype": "28", "sega_official": "28", "area": "28",
         "ultraimage": "30+30", "sca_nb": "28"}
METRICS = {"full": (-2.18, 7.61), "dype": (-0.11, 20.52), "sega_official": (0.55, 24.37),
           "area": (0.92, 27.20), "ultraimage": (1.28, 30.80), "sca_nb": (1.23, 29.71)}
METRIC_FILES = {"full": "final4k_full_extra_summary.csv",
                "dype": "e1_baselines_extra_summary.csv",
                "sega_official": "e1_baselines_extra_summary.csv",
                "area": "e1_baselines_extra_summary.csv",
                "ultraimage": "e1_ultraimage_extra_summary.csv",
                "sca_nb": "e1_baselines_extra_summary.csv"}
CAMPAIGN_FILES = {"dype": "e1_baselines_base_summary.csv",
                  "sega_official": "e1_baselines_base_summary.csv",
                  "ultraimage": "e1_ultraimage_base_summary.csv"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(method: str, res: str, pid: str) -> tuple[Path, dict]:
    root = Path(imgsel.OUT) / "main" / method / res
    hits = sorted(root.glob(f"a4k_eval_4096_{pid}_*_seed0.png"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{method} {res} {pid}: {hits}")
    png = hits[0]
    meta = json.loads(png.with_suffix(".json").read_text())
    if meta.get("complete") is not True or meta.get("png_sha256") != sha(png):
        raise ValueError(f"incomplete or hash-mismatched source {png}")
    return png, meta


def save_jpeg(img: Image.Image, path: Path, quality: int, subsampling: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=quality, optimize=True, subsampling=subsampling)
    return path


def preview(img: Image.Image, width_in: float) -> Image.Image:
    w, h = img.size
    tw = min(w, round(DPI * width_in))
    if tw == w:
        return img
    return img.resize((tw, round(h * tw / w)), Image.Resampling.LANCZOS)


def rows(name: str) -> list[dict]:
    with open(RESULTS / name) as fh:
        return [r for r in csv.DictReader(fh) if r.get("res", "4096x4096") == "4096x4096"]


def check_numbers(metas: dict[str, dict]) -> dict:
    """Re-derive every printed number from the result CSVs and the sidecars."""
    rec = {}
    lat = {r["method"]: r for r in rows("latency_curve_summary.csv")}
    for m, (secs, kind) in SECONDS.items():
        if kind == "clean":
            raw = float(lat[m]["median_seconds"])
            n, src = int(lat[m]["n"]), "latency_curve_summary.csv (median_seconds)"
        else:
            hit = [r for r in rows(CAMPAIGN_FILES[m]) if r["method"] == m and r["metric"] == "seconds"]
            assert len(hit) == 1, (m, hit)
            raw, n = float(hit[0]["value"]), int(hit[0]["n_prompts"])
            src = CAMPAIGN_FILES[m] + " (metric seconds, value)"
            assert n == 191, (m, n)
        assert round(raw, 1) == secs, (m, raw, secs)
        rec.setdefault(m, {})["seconds"] = dict(shown=secs, kind=kind, raw=raw, n=n,
                                                source="results/" + src)
    for m, (ir, hps) in METRICS.items():
        got = {}
        for r in rows(METRIC_FILES[m]):
            if r["method"] == m and r["metric"] in ("ir", "hpsv21"):
                assert int(r["n_prompts"]) == 191, (m, r)
                got[r["metric"]] = float(r["value"])
        assert round(got["ir"], 2) == ir and round(100 * got["hpsv21"], 2) == hps, (m, got)
        rec[m]["metrics"] = dict(imagereward=got["ir"], hpsv21=got["hpsv21"], n_prompts=191,
                                 source="results/" + METRIC_FILES[m])
    for m, steps in STEPS.items():
        c = metas[m]["config"]
        if m == "ultraimage":
            call = c["ultraimage"]["call"]
            got = "%d+%d" % (call["num_inference_steps1"], call["num_inference_steps2"])
        else:
            got = str(c["steps"])
        assert got == steps, (m, got, steps)
        rec[m]["steps"] = steps
    return rec


def latex_escape(s: str) -> str:
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
             "^": r"\textasciicircum{}"}
    return "".join(table.get(ch, ch) for ch in s)


def num(v: float, digits: int) -> str:
    s = "%.*f" % (digits, abs(v))
    return (r"\textminus " + s) if v < 0 else s


def main() -> None:
    scafig.write_tikz_colors(str(HERE / "sca_colors.tex"))
    records, tex = [], []
    # Geometry, inches; origin at the top-left corner, y grows downward (negated in TikZ).
    h = (W - G - G2) / 5               # mosaic row: pano 3h, tiles, square h, tiles
    zt = (h - G) / 2                   # zoom tile side (all four tiles)
    pano_w = 3 * h
    zx1 = pano_w + G                   # panorama tile column
    sq_x = zx1 + zt + G2               # square
    zx2 = sq_x + h + G                 # square tile column
    col = 0.695                        # method column width (thumbnail = crop side)
    gs = 0.05                          # gutter between method columns
    lw = W - 6 * col - 5 * gs          # label column on the left of the strip
    y0 = h + 0.165                     # top of the strip thumbnails
    gv = 0.03                          # gap between a thumbnail and its crop
    y_tab = y0 + 2 * col + gv + 0.028  # top rule of the table
    pitch = 0.108                      # table row pitch
    base = [y_tab + 0.09 + k * pitch for k in range(3)]
    y_rule2 = base[-1] + 0.04
    y_leg = y_rule2 + 0.093
    total_h = y_leg + 0.025

    def node(x, y, w, hh, name):
        tex.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                   r"{\includegraphics[width=%.4fin,height=%.4fin]{%s}};" % (x, -y, w, hh, name))

    def box(x0, y0_, x1, y1, color="scaZoom"):
        tex.append(r"\draw[halo] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (x0, -y0_, x1, -y1))
        tex.append(r"\draw[zoombox,draw=%s] (%.4f,%.4f) rectangle (%.4f,%.4f);"
                   % (color, x0, -y0_, x1, -y1))

    def frame(x0, y0_, x1, y1, style, color="scaZoom"):
        tex.append(r"\draw[%s,draw=%s] (%.4f,%.4f) rectangle (%.4f,%.4f);"
                   % (style, color, x0, -y0_, x1, -y1))

    def text(x, y, body, style, anchor="base"):
        tex.append(r"\node[%s,anchor=%s] at (%.4f,%.4f) {%s};" % (style, anchor, x, -y, body))

    color_of = {"a": "scaZoom", "b": "scaZoomB"}

    def mosaic(spec, x_img, w_img, h_img, x_tiles, stem, size, role, badge_text):
        png, meta = source(spec["method"], spec["res"], spec["pid"])
        img = Image.open(png).convert("RGB")
        assert img.size == size, img.size
        full = save_jpeg(preview(img, w_img), ASSETS / f"{stem}.jpg", 88, 2)
        node(x_img, 0, w_img, h_img, full.name)
        sx = w_img / img.width
        tiles = []
        for k, (x, y, s, ck) in enumerate(spec["zooms"]):
            p = save_jpeg(img.crop((x, y, x + s, y + s)), ASSETS / f"{stem}_zoom{k + 1}.jpg", 90, 0)
            ty = k * (zt + G)
            node(x_tiles, ty, zt, zt, p.name)
            box(x_img + x * sx, y * sx, x_img + (x + s) * sx, (y + s) * sx, color_of[ck])
            frame(x_tiles, ty, x_tiles + zt, ty + zt, "zoomframe", color_of[ck])
            tiles.append(dict(file=p.name, box_xys=[x, y, s], color=color_of[ck],
                              tile_xy_in=[round(x_tiles, 4), round(ty, 4)], tile_in=round(zt, 4)))
        text(x_img + M, M, badge_text, "badge", "north west")
        records.append(dict(role=role, source=str(png), png_sha256=sha(png),
                            prompt=meta["config"]["prompt"], config=meta["config"],
                            preview=full.name, preview_px=list(Image.open(full).size),
                            zooms=tiles))

    # --- top row: panorama and square, each with two tiles beside it ------------------
    mosaic(PANO, 0, pano_w, h, zx1, "pano", (6144, 2048), "mosaic panorama",
           r"2048\,$\times$\,6144")           # height x width, as in the text
    mosaic(SQUARE, sq_x, h, h, zx2, "square", (4096, 4096), "mosaic square",
           r"4096\,$\times$\,4096")

    # --- bottom: six-method strip with a table underneath --------------------------------
    metas = {}
    bx, by, bs = STRIP_BOX
    for k, (method, label) in enumerate(STRIP):
        png, meta = source(method, "4096x4096", STRIP_PID)
        metas[method] = meta
        img = Image.open(png).convert("RGB")
        assert img.size == (4096, 4096)
        x0 = lw + k * (col + gs)
        fullp = save_jpeg(preview(img, col), ASSETS / f"strip_{method}_full.jpg", 88, 2)
        cropp = save_jpeg(img.crop((bx, by, bx + bs, by + bs)), ASSETS / f"strip_{method}_crop.jpg", 90, 0)
        node(x0, y0, col, col, fullp.name)
        node(x0, y0 + col + gv, col, col, cropp.name)
        sx = col / 4096
        box(x0 + bx * sx, y0 + by * sx, x0 + (bx + bs) * sx, y0 + (by + bs) * sx)
        frame(x0, y0 + col + gv, x0 + col, y0 + 2 * col + gv, "cropframe")
        ours = method == "sca_nb"
        xc = x0 + col / 2
        text(xc, y0 - 0.05, label, "headours" if ours else "head")
        secs, kind = SECONDS[method]
        ir, hps = METRICS[method]
        best_ir = ir == max(v[0] for v in METRICS.values())
        best_hps = hps == max(v[1] for v in METRICS.values())
        fast = secs == min(v[0] for v in SECONDS.values())
        text(xc, base[0], STEPS[method], "val")
        t = r"%.1f" % secs
        t = (r"\textbf{\color{scaAccent}%s}" % t) if fast else t
        text(xc, base[1], t + r"\,\,{\tagfont %s}" % kind, "val")
        a, b = num(ir, 2), num(hps, 2)
        a = r"\textbf{%s}" % a if best_ir else a
        b = r"\textbf{%s}" % b if best_hps else b
        text(xc, base[2], a + r"\,/\," + b, "val")
        records.append(dict(role="strip " + method, source=str(png), png_sha256=sha(png),
                            prompt=meta["config"]["prompt"], config=meta["config"],
                            crop_xys=list(STRIP_BOX), steps=STEPS[method], seconds=secs,
                            seconds_kind=kind, imagereward=ir, hpsv21_x100=hps,
                            full=fullp.name, crop=cropp.name))
    numbers = check_numbers(metas)

    # Label column: the prompt beside the full images, then right-aligned row labels.
    prompt = metas["sca_nb"]["config"]["prompt"]
    assert all(metas[m]["config"]["prompt"] == prompt for m in metas)
    text(0, y0 - 0.05, "Prompt", "labhead", "base west")
    tex.append(r"\node[prompt,text width=%.4fin,anchor=north west] at (0,%.4f) {\hyphenpenalty=10000 ``%s''};"
               % (lw - 0.1, -(y0 - 0.005), latex_escape(prompt)))
    xr = lw - 0.06
    text(xr, y0 + 1.5 * col + gv, r"%d$^2$ crop, 1:1" % bs, "lab", "east")
    for y, s in zip(base, ["Steps", "Time (s)", "ImageReward / HPSv2.1"]):
        text(xr, y, s, "lab", "base east")
    tex.append(r"\draw[tabrule] (0,%.4f) -- (%.4f,%.4f);" % (-y_tab, W, -y_tab))
    tex.append(r"\draw[tabrule] (0,%.4f) -- (%.4f,%.4f);" % (-y_rule2, W, -y_rule2))
    text(0, y_leg, r"\textbf{clean}: batch-1 median of three after warm-up, idle GPU;\enspace "
                   r"\textbf{campaign}: mean over the 191-prompt run, shared nodes.\enspace "
                   r"HPSv2.1 is $\times$100.", "legend", "base west")

    doc = r"""%% Generated by fig_hero.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{textcomp}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{tikz}
\graphicspath{{../teaser/}}
\input{sca_colors.tex}
\definecolor{scaZoomB}{HTML}{%s}
\newcommand{\tagfont}{\fontsize{6.5}{7.8}\selectfont\color{scaMid}}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  halo/.style={draw=black,draw opacity=0.35,line width=1.15pt},
  zoombox/.style={line width=0.55pt},
  zoomframe/.style={line width=0.8pt},
  cropframe/.style={line width=0.5pt},
  tabrule/.style={draw=scaRule,line width=0.4pt},
  badge/.style={fill=black,fill opacity=0.58,text opacity=1,text=white,rounded corners=1.2pt,
                inner xsep=1.8pt,inner ysep=1.15pt,font=\sffamily\fontsize{6.5}{7}\selectfont},
  head/.style={inner sep=0pt,font=\sffamily\fontsize{8}{9.6}\selectfont,text=scaInk},
  headours/.style={inner sep=0pt,font=\sffamily\bfseries\fontsize{8}{9.6}\selectfont,text=scaAccent},
  lab/.style={inner sep=0pt,font=\sffamily\fontsize{6.5}{7.8}\selectfont,text=scaMid},
  labhead/.style={inner sep=0pt,font=\sffamily\fontsize{8}{9.6}\selectfont,text=scaInk},
  prompt/.style={inner sep=0pt,align=left,font=\sffamily\itshape\fontsize{6.5}{7.7}\selectfont,text=scaMid},
  val/.style={inner sep=0pt,font=\sffamily\fontsize{7}{8.4}\selectfont,text=scaInk},
  legend/.style={inner sep=0pt,font=\sffamily\fontsize{6.5}{7.8}\selectfont,text=scaMid},
  ]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (ZOOM_B, W, -total_h, "\n".join(tex))
    TEX.write_text(doc)
    subprocess.run([TECTONIC, "-X", "compile", TEX.name, "--outdir", str(FIG)], cwd=HERE,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    produced = FIG / "teaser.pdf"
    assert produced == OUT_PDF and produced.exists()

    caption = (
        r"\textbf{\sca{} keeps layout and fine detail at $4096^2$ and $2048\times6144$ without "
        r"training and is the fastest method shown.} "
        r"Top: curated \sca{} outputs; tiles show same-colour boxes at native pixels. "
        r"Bottom: one curated prompt at $4096^2$, seed 0; UltraImage uses its own two-stage "
        r"recipe. Times are clean batch-1 medians or campaign means (one GH200); metrics are "
        r"191-prompt means.")
    SNIPPET.write_text("\n".join([
        "% Generated by src/fig_hero.py (TikZ source src/teaser.tex, assets figures/teaser/).",
        "% Intended placement: top of page 2, full text width; see DESIGN_CHANGES.md.",
        r"\begin{figure*}[t]",
        r"  \centering",
        r"  \includegraphics[width=\linewidth]{figures/teaser.pdf}",
        r"  \caption{" + caption + "}",
        r"  \label{fig:teaser}",
        r"\end{figure*}", ""]))
    (ASSETS / "provenance.json").write_text(json.dumps(dict(
        selection="Curated by visual inspection; no metric ranking. Mosaic: SCA outputs. "
                  "Strip: chosen from the 191 evaluation prompts at 4096x4096 where all six "
                  "methods have a complete seed-0 output, for a visible layout difference.",
        strip_selection=dict(
            procedure="Six-method contact sheets of all 191 prompts (256 px thumbnails) were "
                      "screened for layout failures (duplication, stripes, broken structure) in "
                      "the baselines with a coherent SCA output; shortlisted prompts were then "
                      "viewed at 512 px per method and SCA at full size. Prompts where SCA itself "
                      "duplicated subjects or broke structure were rejected.",
            shortlist=["1269964769307095670", "6978695891801939003", "minus3112502764021036396",
                       "5467301199144503406", "minus836402013429605880"],
            chosen=STRIP_PID,
            reason="Full attention is a flat band; DyPE and SEGA lose the peaks and the hills "
                   "under speckle; AReA keeps the peaks but its lower half dissolves into a "
                   "streaked haze; UltraImage and SCA keep peaks above rolling hills. The "
                   "alternatives either showed small repeated structures in SCA (waterfall "
                   "bridge) or weaker AReA failures.",
            crop="Shared native window on the main peak, where snow and rock detail differ "
                 "most between methods (x, y, side in pixels)."),
        strip_prompt_id="a4k_eval_4096_" + STRIP_PID, strip_crop_xys=list(STRIP_BOX),
        timing_definitions=dict(
            clean="batch-1 median of three prompts after warm-up on an idle GPU "
                  "(results/latency_curve_summary.csv, 4096x4096)",
            campaign="mean over the 191-prompt evaluation campaign on shared nodes "
                     "(results/e1_*_base_summary.csv, metric seconds)"),
        metrics_definition="ImageReward and HPSv2.1 (x100), means over the 191 evaluation "
                           "prompts at 4096x4096",
        numbers=numbers, width_in=W, height_in=round(total_h, 4),
        pdf=str(OUT_PDF.relative_to(REPO)), records=records,
        asset_sha256={p.name: sha(p) for p in sorted(ASSETS.glob("*.jpg"))}),
        indent=2) + "\n")
    print(f"teaser.pdf {OUT_PDF.stat().st_size/1e6:.2f} MB, height {total_h:.3f} in, "
          f"label column {lw:.3f} in, assets "
          f"{sum(p.stat().st_size for p in ASSETS.glob('*.jpg'))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
