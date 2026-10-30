"""Page-1 hero (fig:teaser): pictures only, in the reference style (REFERENCE_STYLE.md).

Top: a mosaic of three curated SCA outputs from the 195 evaluation prompts (seed 0, 28 steps,
guidance 4.5) in one row: a 2048x6144 panorama and two 4096^2 squares, one size tag each. The
panorama and the cathedral carry one native-pixel inset each, framed in the one zoom colour
together with its source window.

Bottom: one curated evaluation prompt at 4096^2 for six methods, names once with SCA (ours)
last, the full image above the same native-pixel (1:1) crop. No prompt, timing, metric or
footnote text: the caption carries the claim and Table 1 the numbers.

SCA is the final operator (sca_v2, out/dupfix/sca_v2) for the same prompts and seed as the published
operator (sca_nb) on which every curated choice below was made; imgsel.final_twin checks each image
against its published twin (same prompt, prompt id and derived seed) and the sca_v2 flags. The final
images were viewed whole and at native pixels on 2026-10-30 (FINAL_VIEW, in the provenance).

Final-operator audit (2026-10-30, .agent/notes/figure_curation_20261030.md): the panorama's inset window showed
garbled lettering on two facades, so the panorama prompt was replaced by the 2026-10-20 rule run on the final
operator's 2048x6144 per-image CSVs (eval/a4k195_2048x6144_sca_v2): eligible prompts (not shown elsewhere, not
development), AReA's ImageReward and HPSv2.1 at or above its own 25th percentile, ranked by SCA's ImageReward and
MUSIQ_c percentiles, overview pre-screen, then every native tile audited. The first prompt that passed is a turquoise
river below snow peaks (7095623346664983157); its window is on the snow ridges. The cathedral, the portrait and the
strip image have flaws only at 1:1 and stay (AUDIT_20261030).

Timing and metrics are re-derived from the sources of Table 1 (FLUX.1-dev, 4096^2, 195 prompts) and
stored in the provenance only; neither the figure nor its snippet caption prints a time or a score.
Every time is the campaign mean of Table 1 (camp. column: mean over the 195-prompt evaluation on
shared nodes, results/<run>/base_summary.csv, metric "seconds"), because the final operator's clean
batch-1 latency is pending; ImageReward and HPSv2.1 come from results/<run>/extra_summary.csv. The
runs are a4k195_main for the baselines and a4k195_main_sca_v2 for SCA. A mismatch with those CSVs or
the source sidecars stops the build.

Everything is generated: JPEG assets (deterministic PIL encoding), a TikZ source with
vector boxes and labels, its PDF (tectonic), the snippet, and provenance with source
hashes and crop boxes.

Selection audit (2026-10-20): every SCA image was viewed whole and at native pixels. The
panorama and the cathedral are clean and stay. The previous portrait (3657338173664010496)
had a second, miniature pipe on the headband and a stray stem at the far mouth corner, so it
was replaced. The previous strip prompt (1269964769307095670) was chosen for a visible layout
difference and its baselines sat far below their own per-image medians (AReA ImageReward at its
1st percentile, SEGA 2nd, DyPE 8th), which flatters SCA; it was replaced by a prompt on which
every baseline scores near its median. Candidates were shortlisted on the per-image CSVs of the
195-prompt evaluation (eval/a4k195_main), SCA's ImageReward and MUSIQ_c percentiles high, then
viewed at full resolution; prompts shown elsewhere in the paper and the development prompts were
excluded. provenance.json records the shortlist, the rejections and the values used.

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

W = 5.5            # text width, inches
G = 1.5 / 72       # white gutter between images (house rule: 1 to 2 pt)
M = 0.025          # margin of size tags and insets from image edges
DPI = 600          # pixel budget for downsampled full images
FRAME = 0.6 / 72   # line width of the one frame colour (scaZoom), inches

# Mosaic: three curated SCA outputs in one row of equal height (panorama 3h, squares h). A zoom
# window is (x, y, side) in native pixels; its native-pixel inset sits inside the image at the
# bottom-right corner, `inset` inches wide. Panorama (since 2026-10-30): the snow ridges, inset on the
# gravel of the right bank. Cathedral: the rose window, inset on the pews. Portrait (a bearded man with a hat and
# an acoustic guitar): no inset, since any corner large enough would cover the guitar or the
# subject.
MOSAIC = [
    (dict(method="sca_v2", res="2048x6144", pid="7095623346664983157",
          zoom=dict(box=(2100, 610, 384), inset=0.50)), "pano"),
    (dict(method="sca_v2", res="4096x4096", pid="minus8737279395568207187",
          zoom=dict(box=(1792, 1536, 448), inset=0.44)), "square"),
    (dict(method="sca_v2", res="4096x4096", pid="5849277588805396896"), "portrait"),
]
OURS = "sca_v2"  # the final operator; every curated choice here was made on the published operator (sca_nb)
FINAL_VIEW = {  # the final operator's images, viewed whole and at native pixels (2026-10-30)
    "pano": ("7095623346664983157 (since 2026-10-30): one river and one range of snow peaks; every native tile audited; "
             "flaws only at 1:1: blocky 4 to 8 px texture on the right-hand snow ridge (x 4096-5300, y 420-850), "
             "gritty speckle on the right gravel, a small pale lump at (4526, 1488) and a 6 px dark fragment above "
             "the ridge at (5049, 434); the prompt's starry sky shows only as 1 to 3 px dots. The window "
             "(2100, 610, 384) on the snow ridges is clean"),
    "square": ("same layout; the rose window is inside the zoom window; a few small seated people and a wall "
               "sign at the end of the side aisles (the published image also has a few seated people)"),
    "portrait": "same man, hat and guitar; the headstock still carries extra tuners, as in the published image",
    "strip": ("one church and the flock, as in the published image; the faint distant spires of the published "
              "image are gone; the strip window still lies on the church tower"),
}
AUDIT_20261030 = dict(  # .agent/notes/figure_curation_20261030.md; percentiles are floor(100 x share <= value)
    replaced=dict(
        pano=dict(
            old="minus2742856837225126754", new="7095623346664983157",
            why=("final operator: garbled two-line pseudo-lettering on a yellow facade (x 4124-4202, y 1046-1077) and "
                 "partial pseudo-signs inside the inset window (4064, 832, 384), the brightest detail of the inset. A "
                 "text-free window on the same houses existed (3420, 740, 384), but the prompt also fails the "
                 "fairness rule at this size (AReA HPSv2.1 p17), so it was replaced, not re-windowed"),
            rule=("eligible (not shown in any figure, not development; 101 of 195), AReA ImageReward and HPSv2.1 >= its "
                  "own p25 at 2048x6144 (58 left), ranked by SCA ImageReward + MUSIQ_c percentiles on "
                  "eval/a4k195_2048x6144_sca_v2; overview pre-screen of the top ten, then every native tile"),
            percentiles=("ImageReward/MUSIQ_c/HPSv2.1 at 2048x6144, final operator: new SCA 66/87/35 (ImageReward "
                         "1.17, MUSIQ_c 71.1), AReA 56/96/47; old SCA 70/91/87, AReA 27/98/17"),
            rejected={"1638867997226496674": "two pairs of children for one boy and one girl",
                      "1118979843958125111": "smeared, striped forearm and a floating hand on the sofa back",
                      "m9219623328452982854": "a second face behind the climber",
                      "m3880872769316591135": "extra cloaked figures and extra trees for a lone figure and tree",
                      "m293370523718655394": "a second, thinner waterfall for one"},
            also_accepted={"minus1017950324987649681": "ranked second (SCA 74/76/46); clean at print, kept as backup"})),
    kept={"cathedral": ("flaws only at 1:1: two chandeliers hang from caps without visible chains, confetti-like "
                        "lancet glass, a tiny pseudo-text plaque; the rose-window inset is clean"),
          "portrait": ("flaws only at 1:1: crumpled tuner keys and red striping on the headstock, a bright rim "
                       "around dark silhouettes; hands sound"),
          "strip": ("flaws only at 1:1: a loose dark rod on the snowy nave roof (x 2458-2593, y 2173-2388), a "
                    "1.2 grey-level step in the sky at x 2112; one church, about 40 sheep; the belfry window is "
                    "clean and shows no lattice")})

# Six-method strip: one curated prompt; the crop window is shared by all methods.
# Chosen from the 195 evaluation prompts so that SCA's output is clean at native pixels and has
# high ImageReward and MUSIQ_c, while every baseline scores near its own per-image median
# (ImageReward and HPSv2.1 percentiles 33 to 69), so no baseline is shown on an unusually bad
# prompt. Sheep in snow before a stone church: full attention is a flat field, DyPE and SEGA
# cover the scene in a fine lattice, and AReA, UltraImage and SCA keep the church and the flock.
# The window is placed by hand on SCA's church tower, at a spot where AReA and UltraImage also
# show church walls and windows, so the crops compare like with like (stone, glass, snow).
STRIP_PID = "minus8343371271909676658"
STRIP_BOX = (1650, 1300, 768)                 # x, y, side in native pixels
STRIP = [("full", "Full attention"), ("dype", "DyPE"), ("sega_official", "SEGA"),
         ("area", "AReA"), ("ultraimage", "UltraImage"), (OURS, "SCA (ours)")]

# Numbers kept in the provenance (not printed in the figure or its caption); check_numbers()
# re-derives each from the sources of Table 1 and the sidecars. Seconds: the camp. column of
# Table 1 for every method (the final operator's clean latency is pending).
N_TABLE1 = 195
RUNS = {"full": "a4k195_main", "dype": "a4k195_main", "sega_official": "a4k195_main", "area": "a4k195_main",
        "ultraimage": "a4k195_main", OURS: "a4k195_main_sca_v2"}
SECONDS = {"full": 196.2, "dype": 195.9, "sega_official": 190.4, "area": 264.4, "ultraimage": 308.5, OURS: 78.4}
STEPS = {"full": "28", "dype": "28", "sega_official": "28", "area": "28",
         "ultraimage": "30+30", OURS: "28"}
METRICS = {"full": (-2.18, 7.60), "dype": (-0.10, 20.55), "sega_official": (0.56, 24.34),
           "area": (0.92, 27.16), "ultraimage": (1.28, 30.77), OURS: (1.25, 29.73)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(method: str, res: str, pid: str) -> tuple[Path, dict]:
    if method == OURS:  # the published image's twin, checked by imgsel.final_twin
        png = Path(imgsel.final_png(res, pid)[0])
        return png, json.loads(png.with_suffix(".json").read_text())
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
    """Re-derive every recorded number from the sources of Table 1 and the sidecars."""
    rec = {}
    for m, secs in SECONDS.items():
        name = f"{RUNS[m]}/base_summary.csv"
        hit = [r for r in rows(name) if r["method"] == m and r["metric"] == "seconds"]
        assert len(hit) == 1, (m, hit)
        raw, n = float(hit[0]["value"]), int(hit[0]["n_prompts"])
        assert n == N_TABLE1 and round(raw, 1) == secs, (m, raw, n, secs)
        rec.setdefault(m, {})["seconds"] = dict(shown=secs, kind="campaign", raw=raw, n=n,
                                                source=f"results/{name} (metric seconds, value)")
    for m, (ir, hps) in METRICS.items():
        name, got = f"{RUNS[m]}/extra_summary.csv", {}
        for r in rows(name):
            if r["method"] == m and r["metric"] in ("ir", "hpsv21"):
                assert int(r["n_prompts"]) == N_TABLE1, (m, r)
                got[r["metric"]] = float(r["value"])
        assert round(got["ir"], 2) == ir and round(100 * got["hpsv21"], 2) == hps, (m, got)
        rec[m]["metrics"] = dict(imagereward=got["ir"], hpsv21=got["hpsv21"], n_prompts=N_TABLE1,
                                 source=f"results/{name}")
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
    records, tex, written = [], [], []
    # Geometry, inches; origin at the top-left corner, y grows downward (negated in TikZ).
    h = (W - 2 * G) / 5                # mosaic row: panorama 3h and two squares h
    col = (W - 5 * G) / 6              # strip column: full image and crop side
    y_name = h + 0.135                 # baseline of the method names
    y0 = y_name + 0.04                 # top of the strip
    total_h = y0 + 2 * col + G

    def node(x, y, w, hh, name):
        tex.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                   r"{\includegraphics[width=%.4fin,height=%.4fin]{%s}};" % (x, -y, w, hh, name))

    def box(x0, y0_, x1, y1, halo=True):
        """The one frame colour, drawn just inside the rectangle; a dark halo on images."""
        d = FRAME / 2
        r = (x0 + d, -(y0_ + d), x1 - d, -(y1 - d))
        if halo:
            tex.append(r"\draw[halo] (%.4f,%.4f) rectangle (%.4f,%.4f);" % r)
        tex.append(r"\draw[frame] (%.4f,%.4f) rectangle (%.4f,%.4f);" % r)

    def text(x, y, body, style, anchor="base"):
        tex.append(r"\node[%s,anchor=%s] at (%.4f,%.4f) {%s};" % (style, anchor, x, -y, body))

    def jpeg(img, name, quality, subsampling):
        p = save_jpeg(img, ASSETS / name, quality, subsampling)
        written.append(p)
        return p

    # --- mosaic: three curated SCA outputs, one size tag each ----------------------------
    x = 0.0
    for spec, stem in MOSAIC:
        png, meta = source(spec["method"], spec["res"], spec["pid"])
        img = Image.open(png).convert("RGB")
        hpx, wpx = map(int, spec["res"].split("x"))          # res is height x width
        assert img.size == (wpx, hpx), img.size
        w_img = h * img.width / img.height
        full = jpeg(preview(img, w_img), f"{stem}.jpg", 88, 2)
        node(x, 0, w_img, h, full.name)
        zooms = []
        z = spec.get("zoom")
        if z:
            bx, by, bs = z["box"]
            q = z["inset"]
            crop = jpeg(img.crop((bx, by, bx + bs, by + bs)), f"{stem}_zoom.jpg", 90, 0)
            s = w_img / img.width
            b0, b1 = (x + bx * s, by * s), (x + (bx + bs) * s, (by + bs) * s)
            i0 = (x + w_img - M - q, h - M - q)               # bottom-right corner
            i1 = (i0[0] + q, i0[1] + q)
            assert not (i0[0] < b1[0] and b0[0] < i1[0] and i0[1] < b1[1] and b0[1] < i1[1]), stem
            node(i0[0], i0[1], q, q, crop.name)
            box(*b0, *b1)
            box(*i0, *i1)
            zooms.append(dict(file=crop.name, box_xys=[bx, by, bs], inset_in=q,
                              inset_xy_in=[round(i0[0] - x, 4), round(i0[1], 4)],
                              magnification=round((q / bs) / s, 2)))
        text(x + M, M, r"%d\,\texttimes\,%d" % (img.height, img.width), "badge", "north west")
        records.append(dict(role="mosaic " + stem, source=str(png), png_sha256=sha(png),
                            prompt=meta["config"]["prompt"], config=meta["config"],
                            preview=full.name, preview_px=list(Image.open(full).size),
                            zooms=zooms))
        x += w_img + G
    assert abs(x - G - W) < 1e-6, x

    # --- six-method strip: names once, each full image above the same 1:1 crop -----------
    metas = {}
    bx, by, bs = STRIP_BOX
    for k, (method, label) in enumerate(STRIP):
        png, meta = source(method, "4096x4096", STRIP_PID)
        metas[method] = meta
        img = Image.open(png).convert("RGB")
        assert img.size == (4096, 4096)
        x0 = k * (col + G)
        fullp = jpeg(preview(img, col), f"strip_{method}_full.jpg", 88, 2)
        cropp = jpeg(img.crop((bx, by, bx + bs, by + bs)), f"strip_{method}_crop.jpg", 90, 0)
        node(x0, y0, col, col, fullp.name)
        node(x0, y0 + col + G, col, col, cropp.name)
        sx = col / 4096
        box(x0 + bx * sx, y0 + by * sx, x0 + (bx + bs) * sx, y0 + (by + bs) * sx)
        box(x0, y0 + col + G, x0 + col, y0 + 2 * col + G, halo=False)
        text(x0 + col / 2, y_name, label, "headours" if method == OURS else "head")
        records.append(dict(role="strip " + method, source=str(png), png_sha256=sha(png),
                            prompt=meta["config"]["prompt"], config=meta["config"],
                            crop_xys=list(STRIP_BOX), full=fullp.name, crop=cropp.name))
    numbers = check_numbers(metas)
    prompt = metas[OURS]["config"]["prompt"]
    assert all(metas[m]["config"]["prompt"] == prompt for m in metas)
    seeds = {metas[m]["config"]["derived_seed"] for m in metas}
    assert len(seeds) == 1, seeds  # same prompt and initial noise in every column
    for stale in sorted(ASSETS.glob("strip_*.jpg")):  # strip tiles of a method no longer shown
        if stale not in written:
            stale.unlink()

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
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  halo/.style={draw=black,draw opacity=0.35,line width=1.15pt},
  frame/.style={draw=scaZoom,line width=0.6pt},
  badge/.style={fill=black,fill opacity=0.55,text opacity=1,text=white,rounded corners=1pt,
                inner xsep=1.6pt,inner ysep=1.0pt,font=\sffamily\fontsize{6.1}{7}\selectfont},
  head/.style={inner sep=0pt,font=\sffamily\fontsize{7.5}{9}\selectfont,text=scaInk},
  headours/.style={inner sep=0pt,font=\sffamily\bfseries\fontsize{7.5}{9}\selectfont,text=scaAccent},
  ]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (W, -total_h, "\n".join(tex))
    TEX.write_text(doc)
    subprocess.run([TECTONIC, "-X", "compile", TEX.name, "--outdir", str(FIG)], cwd=HERE,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=dict(os.environ, SOURCE_DATE_EPOCH="0"))  # byte-identical PDF per run
    produced = FIG / "teaser.pdf"
    assert produced == OUT_PDF and produced.exists()

    caption = (
        r"\textbf{\sca{} keeps layout and fine detail at $4096^2$ and $2048\times6144$ without "
        r"training.} "
        + r"Top: curated \sca{} outputs with native-pixel insets. Bottom: one curated prompt at "
        r"$4096^2$, each full image above the same 1:1 crop; UltraImage uses its own two-stage "
        r"recipe.")
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
        layout="Pictures only (REFERENCE_STYLE.md house rule 1): one size tag per mosaic image, "
               "method names once, one frame colour; numbers moved to the caption and Table 1.",
        final_operator=dict(
            method=OURS, images="out/dupfix/sca_v2 (imgsel.final_twin of each published image)",
            switched="2026-10-30: every SCA image (mosaic and strip) now shows the final operator, same "
                     "prompts and seed; the crops and zoom windows are unchanged except the panorama, whose "
                     "prompt and window were replaced after the final-operator audit (audit_20261030); every "
                     "other curated choice and audit below was made on the published operator (sca_nb)",
            viewed=FINAL_VIEW),
        audit_20261030=AUDIT_20261030,
        selection="Curated. Mosaic: SCA outputs chosen by visual inspection; the portrait was "
                  "re-chosen on 2026-10-20 from a shortlist ranked by SCA's per-image ImageReward "
                  "and MUSIQ_c percentiles (eval/a4k195_main) and viewed at native pixels. Strip: "
                  "chosen from the 195 evaluation prompts at 4096x4096 where all six methods have a "
                  "complete seed-0 output, with SCA clean and high-scoring and every baseline near "
                  "its own per-image median.",
        audit_20261020=dict(
            kept=dict(pano="minus2742856837225126754: clean whole and at native pixels (castle, "
                           "bridge crowd, inset facades, island church)",
                      cathedral="minus8737279395568207187: clean (rose window, chandeliers, pews); "
                                "ImageReward is low for every method on this prompt"),
            replaced=dict(
                portrait=dict(old="3657338173664010496", new="5849277588805396896",
                              why="old: a second, miniature pipe sits on the headband and a stray "
                                  "stem sticks out of the far mouth corner (duplicated object)",
                              new_values="SCA ImageReward 1.91 (p92 of SCA's 195), MUSIQ_c 68.7 "
                                         "(p73), HPSv2.1 p90; one man, plausible hands; the "
                                         "headstock carries extra tuners (a generic FLUX object "
                                         "error, not a duplication)"),
                strip=dict(old="1269964769307095670", new=STRIP_PID,
                           why="old: chosen for a layout difference; AReA ImageReward p1 and "
                               "HPSv2.1 p0 of its own 195, SEGA p2, DyPE p8, so the baselines "
                               "were shown on an unusually bad prompt")),
            portrait_rejected={"minus7393293802729135153": "third hand under the veil",
                               "minus3108444262580777290": "two extra small men on the cliff",
                               "7109958488611222679": "clean; used in Figure zoom4k instead"}),
        strip_selection=dict(
            procedure="Per-image CSVs of the 195-prompt evaluation (eval/a4k195_main base and "
                      "extra) ranked the prompts not shown elsewhere in the paper and not used in "
                      "development by SCA's ImageReward + MUSIQ_c + HPSv2.1 percentiles, keeping "
                      "prompts where each of DyPE, SEGA, AReA and UltraImage has ImageReward and "
                      "HPSv2.1 at or above its own 25th percentile. Shortlisted prompts were "
                      "viewed as six-method contact sheets, then SCA whole and at native pixels; "
                      "prompts where SCA repeated or added subjects were rejected.",
            shortlist=["minus8343371271909676658", "minus3268147817845827695",
                       "4820545723762191815", "3826098298648979216", "minus2670470656524755471",
                       "minus3413373481783671760", "minus7713773572301072741",
                       "minus6127054465263660958"],
            rejected={"4820545723762191815": "SCA: five moons and two extra small bears",
                      "3826098298648979216": "SCA: three waterfalls for one",
                      "minus2670470656524755471": "SCA: spray speckle around the umbrellas, "
                                                  "extra small figures",
                      "minus7713773572301072741": "SCA: a second small tree",
                      "minus6127054465263660958": "SCA: three people for two",
                      "minus3268147817845827695": "its prompt asks for a lone figure and SCA adds "
                                                  "about three small pedestrians; the sheep prompt "
                                                  "was preferred (it was briefly in Figure zoom4k "
                                                  "and was replaced there for the same reason)",
                      "minus3413373481783671760": "clean, but SCA's own scores are near its "
                                                  "median (ImageReward p42)"},
            chosen=STRIP_PID,
            values=("at selection (published operator, sca_nb), percentiles within each method's own "
                    "195 images, ImageReward/MUSIQ_c/HPSv2.1: SCA 57/79/89 (ImageReward 1.52, MUSIQ_c 70.9); DyPE 37/54/53; "
                    "SEGA 47/90/33; AReA 36/49/51; UltraImage 41/53/69; full attention's "
                    "ImageReward sits at its usual floor (-2.28)"),
            reason="Sheep in snow before a stone church. Full attention is a flat field; DyPE "
                   "and SEGA cover the scene in a fine lattice; AReA, UltraImage and SCA keep "
                   "the church and the flock. SCA's image is clean at native pixels (stonework, "
                   "windows, the flock); a distant town church is a separate building, not a "
                   "copy of the main one.",
            crop="Shared native window placed by hand on SCA's church tower (x, y, side in "
                 "pixels); AReA and UltraImage also show church walls and windows there, so the "
                 "crops compare stone, glass and snow detail like for like."),
        previous_strip=dict(prompt_id="1269964769307095670", crop_xys=[1664, 1408, 768],
                            dropped_because="baselines far below their own medians"),
        strip_prompt_id="a4k_eval_4096_" + STRIP_PID, strip_crop_xys=list(STRIP_BOX),
        timing_definitions=dict(
            campaign="mean over the 195-prompt evaluation on shared nodes, the camp. column of Table 1 "
                     "(results/<run>/base_summary.csv, metric seconds)",
            clean="not recorded: the final operator's clean batch-1 latency is pending, so no clean time "
                  "or speedup appears in the figure, its caption or this record"),
        metrics_definition="ImageReward and HPSv2.1 (x100), means over the 195 evaluation "
                           "prompts at 4096x4096, as in Table 1",
        numbers=numbers,
        width_in=W, height_in=round(total_h, 4),
        pdf=str(OUT_PDF.relative_to(REPO)), records=records,
        asset_sha256={p.name: sha(p) for p in sorted(set(written))}),
        indent=2) + "\n")
    print(f"teaser.pdf {OUT_PDF.stat().st_size/1e6:.2f} MB, height {total_h:.3f} in, assets "
          f"{sum(p.stat().st_size for p in set(written))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
