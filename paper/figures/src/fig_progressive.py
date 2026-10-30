"""Progressive SCA against HRDiT and single-pass SCA (fig:progressive) as a TikZ-composed PDF.

Two prompts at 4096^2, seed 0, one prompt per row:
  minus5023570400257826529 (ringed planet), then minus2679953061215100202 (dancer).
Column groups, left to right: HRDiT | progressive SCA (ours) | single-pass SCA (ours). Each group is
the full image plus the same native 512x512 crop. Progressive SCA runs HRDiT's stage schedule
(1024^2, 2048^2, 4096^2; 30 + 17 + 10 steps) with SCA attention; single-pass SCA runs 28 steps at
the target size. Both SCA columns show the final operator (since 2026-10-30): sca_v2_prog from the
eval root of prog195_sca_v2 (out/dupfix_prog/sca_v2_prog) and sca_v2 (out/dupfix/sca_v2), each the
twin of the published image (sca_prog, sca_nb) of the same prompt and seed, with the window schedule
in its sidecar; HRDiT is unchanged.

Selection (recorded in figures/progressive/provenance.json and stated in the caption):
* Prompts are random, not curated: random.Random(17).sample(sorted(ids), 2) over the 46 prompt ids
  of the progressive panel manifest (PROMPTS_JSONL), with the a4k_eval_4096_ prefix stripped. The
  script recomputes the draw and stops unless it gives PROMPTS in this order. The paper's numbers now
  cover all 195 Aesthetic-4K prompts; the draw stays on the original 46-prompt panel, as the caption says.
* Crop windows are automatic, never placed by hand: fig_qual_main.auto_box with a 512x512 window,
  i.e. the window with the most edge energy (PIL FIND_EDGES response of the 8x box-downsampled
  greyscale image) averaged over the three methods of the row, at least 256 px from every border.

Every source PNG must have a complete sidecar whose png_sha256 matches the file, config res 4096x4096,
seed 0 and the expected prompt id. Header times (not drawn; the caption points to Table tab:panel46) are
the campaign mean seconds per image that the table prints over all 195 prompts, from
results/<run>/base_summary.csv (TIMES; HRDiT cross-checked against the second run that holds it).

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_progressive.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

import csv

import imgsel
import scafig
from fig_qual_main import auto_box, sha

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent.parent
ASSETS = FIG / "progressive"
TEX = HERE / "progressive.tex"
OUT_PDF = FIG / "progressive.pdf"
SNIPPET = FIG / "snippets" / "progressive.tex"
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")

CAMPAIGN = Path("/scratch/09032/saini_2/sca/eval/prog46_v1")  # the original panel: its prompt list fixes the draw
PROMPTS_JSONL = CAMPAIGN / "prompts.jsonl"
ROOT195 = Path("/scratch/09032/saini_2/sca/eval/prog195_sca_v2/root")  # <method>/4096x4096/ links to the PNGs
PUBLISHED_PROG = Path(imgsel.OUT) / "prog" / "sca_prog" / "4096x4096"  # the published progressive operator
RESULTS = REPO / "results"
N_TIMES = 195
# run whose results/<run>/base_summary.csv gives each method's campaign time as Table tab:panel46 prints it, and a
# second run holding the same method's time, which must agree
TIMES = {"hrdit_dev_author": "a4k195_panel", "sca_v2_prog": "prog195_sca_v2", "sca_v2": "a4k195_main_sca_v2"}
CHECK = {"hrdit_dev_author": "prog195_sca_v2"}
PREFIX = "a4k_eval_4096_"
SEED_DRAW, N_DRAW = 17, 2
PROMPTS = ["minus5023570400257826529", "minus2679953061215100202"]  # ringed planet, dancer
METHODS = ["hrdit_dev_author", "sca_v2_prog", "sca_v2"]
# header text, TikZ style, bold: both SCA variants are ours (accent); "SCA (ours)" is last and the only
# bold label. Times are not drawn (the caption points to the table); campaign_seconds() still validates them.
HEADERS = {"hrdit_dev_author": ("HRDiT", "head", False),
           "sca_v2_prog": ("Progressive SCA (ours)", "headours", False),
           "sca_v2": ("SCA (ours)", "headours", True)}
PUBLISHED = {"sca_v2_prog": "sca_prog", "sca_v2": "sca_nb"}  # the published operator each SCA column replaced
SCHEDULE = {k: v for k, v in imgsel.FINAL_FLAGS.items() if k not in ("window", "pi_variant", "gate_sigma")}
RES, CROP = 4096, 512

W = 5.5
G_IN, G_OUT = 1 / 72, 2 / 72  # white gutters (in): 1 pt inside a group, 2 pt between groups (as fig_qual_main)
HEAD, GV = 0.14, 2 / 72     # header band and row gap (in)
DPI = 600                   # downscaled full images: at most 600 dpi of printed width
FRAME = "A6ADB4"            # the one neutral frame colour (crop windows and crop tiles)


def check_selection() -> tuple[list[str], dict, int]:
    rows = [json.loads(line) for line in PROMPTS_JSONL.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or not all(i.startswith(PREFIX) for i in ids):
        raise ValueError("prompt manifest ids are not unique or lack the a4k_eval_4096_ prefix")
    short = sorted(i[len(PREFIX):] for i in ids)
    drawn = random.Random(SEED_DRAW).sample(short, N_DRAW)
    if drawn != PROMPTS:
        raise ValueError(f"selection rule gives {drawn}, not {PROMPTS}")
    return drawn, {r["id"][len(PREFIX):]: r["prompt"] for r in rows}, len(ids)


def seconds_row(run: str, method: str) -> dict:
    with open(RESULTS / run / "base_summary.csv", newline="") as fh:
        hits = [r for r in csv.DictReader(fh) if r["method"] == method and r["metric"] == "seconds"
                and r["res"] == f"{RES}x{RES}"]
    if len(hits) != 1:
        raise ValueError(f"results/{run}: {len(hits)} seconds rows for {method}")
    return hits[0]


def campaign_seconds() -> dict:
    """Campaign mean seconds per image as Table tab:panel46 prints them (results/<run>/base_summary.csv, TIMES),
    each over the same 195 prompts at RES, cross-checked against a second run where one holds the method."""
    out = {}
    for m in METHODS:
        row = seconds_row(TIMES[m], m)
        mean, n = float(row["value"]), int(float(row["n_prompts"]))
        if m in CHECK and abs(float(seconds_row(CHECK[m], m)["value"]) - mean) > 0.01:
            raise ValueError(f"{m}: results/{TIMES[m]} and results/{CHECK[m]} disagree")
        out[m] = dict(mean_seconds=mean, n=n, run=TIMES[m], label=f"{mean:.1f}",
                      source=f"results/{TIMES[m]}/base_summary.csv")
    if {v["n"] for v in out.values()} != {N_TIMES}:
        raise ValueError(f"campaign means must cover the same {N_TIMES} prompts")
    return out


def check_final(method: str, png: Path, cfg: dict) -> None:
    """An SCA column must be the final operator: the twin of the published image, with the window schedule."""
    if method == "sca_v2":
        twin, _ = imgsel.final_png(f"{RES}x{RES}", cfg["prompt_id"][len(PREFIX):])
        if Path(twin).resolve() != png:
            raise ValueError(f"{png} is not imgsel.final_twin of the published image")
        return
    pub = json.loads((PUBLISHED_PROG / png.name).with_suffix(".json").read_text())["config"]
    same = ("prompt_id", "prompt", "res", "seed", "derived_seed", "steps", "guidance", "model")
    diff = {k: (pub.get(k), cfg.get(k)) for k in same if pub.get(k) != cfg.get(k)}
    stages = cfg["sca"]["stages"]
    flags = [{k: st.get(k) for k in SCHEDULE} for st in stages]
    if cfg.get("method") != method or diff or any(f != SCHEDULE for f in flags) or \
            any(st.get("gate_sigma") != 0.89 or st.get("window") != 64 for st in stages):
        raise ValueError(f"{png}: not the final progressive operator ({cfg.get('method')}, {diff}, {flags})")


def source(method: str, pid: str) -> tuple[Path, Path, dict, str]:
    """The sample symlink (or, for single-pass SCA, the image itself), its resolved PNG, the sidecar and the PNG
    hash, all checked."""
    if method == "sca_v2":
        hits = [Path(imgsel.final_png(f"{RES}x{RES}", pid)[0])]
    else:
        hits = sorted((ROOT195 / method / f"{RES}x{RES}").glob(f"{PREFIX}{pid}_*_seed0.png"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{method} {pid}: {hits}")
    png = hits[0].resolve()
    side = hits[0].with_suffix(".json").resolve()
    meta = json.loads(side.read_text())
    digest = sha(png)
    cfg = meta.get("config", {})
    if meta.get("complete") is not True or meta.get("png_sha256") != digest:
        raise ValueError(f"incomplete or hash-mismatched source {png}")
    if cfg.get("res") != f"{RES}x{RES}" or cfg.get("seed") != 0 or cfg.get("prompt_id") != PREFIX + pid:
        raise ValueError(f"unexpected config in {side}: {cfg.get('res')} {cfg.get('seed')} {cfg.get('prompt_id')}")
    if method in PUBLISHED:
        check_final(method, png, cfg)
    return hits[0], png, meta, digest


def check_colors() -> None:
    """The TikZ colours must be the current scafig palette; this script never rewrites them."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(scafig.write_tikz_colors(os.path.join(tmp, "c.tex"))).read_text()
    if (HERE / "sca_colors.tex").read_text() != fresh:
        raise ValueError("src/sca_colors.tex is stale: run scafig.py first")


def layout():
    """Six square tiles per row: (full, crop) for each of the three methods."""
    tile = (W - 3 * G_IN - 2 * G_OUT) / 6
    xs = {}
    for g, m in enumerate(METHODS):
        xs[m] = g * (2 * tile + G_IN + G_OUT)
        xs[m + "_crop"] = xs[m] + tile + G_IN
    assert abs(xs[METHODS[-1] + "_crop"] + tile - W) < 1e-9
    return xs, tile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="trial run: write the PDF, snippet and assets here instead of the paper tree")
    a = ap.parse_args()
    assets, out_pdf, snippet, tex = ASSETS, OUT_PDF, SNIPPET, TEX
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        assets = a.out_dir / "progressive"
        out_pdf, snippet, tex = a.out_dir / "progressive.pdf", a.out_dir / "progressive_snippet.tex", \
            a.out_dir / "progressive.tex"
    assets.mkdir(parents=True, exist_ok=True)
    check_colors()
    drawn, prompt_text, n_panel = check_selection()
    secs = campaign_seconds()
    xs, tile = layout()
    sx = tile / RES
    tex_lines, records, boxes = [], [], {}

    def node(x, y, w, h, name):
        tex_lines.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                         r"{\includegraphics[width=%.4fin,height=%.4fin]{%s}};" % (x, -y, w, h, name))

    def rect(x0, y0, x1, y1, style):
        tex_lines.append(r"\draw[%s] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (style, x0, -y0, x1, -y1))

    for r, pid in enumerate(PROMPTS):
        y = HEAD + r * (tile + GV)
        imgs, src = {}, {}
        for m in METHODS:
            src[m] = source(m, pid)
            imgs[m] = Image.open(src[m][1]).convert("RGB")
            assert imgs[m].size == (RES, RES), (m, pid, imgs[m].size)
        bx, by, bw, bh = auto_box([imgs[m] for m in METHODS], CROP, CROP)
        assert min(bx, by) >= RES // 16 and max(bx + bw, by + bh) <= RES - RES // 16, (pid, bx, by)
        boxes[pid] = [bx, by, bw, bh]
        for m in METHODS:
            link, png, meta, digest = src[m]
            small = imgs[m].resize((round(DPI * tile),) * 2, Image.Resampling.LANCZOS)
            fp = assets / f"cmp_{pid}_{m}.jpg"
            small.save(fp, "JPEG", quality=88, optimize=True, subsampling=2)
            node(xs[m], y, tile, tile, fp.name)
            rect(xs[m] + bx * sx, y + by * sx, xs[m] + (bx + bw) * sx, y + (by + bh) * sx, "zoombox")
            cp = assets / f"cmp_{pid}_{m}_crop.jpg"
            imgs[m].crop((bx, by, bx + bw, by + bh)).save(cp, "JPEG", quality=90, optimize=True, subsampling=0)
            node(xs[m + "_crop"], y, tile, tile, cp.name)
            i = 0.3 / 72  # half the line width: the frame stays inside the tile
            rect(xs[m + "_crop"] + i, y + i, xs[m + "_crop"] + tile - i, y + tile - i, "cropframe")
            records.append(dict(prompt_id=pid, method=m, crop_xywh=[bx, by, bw, bh], sample_link=str(link),
                                source=str(png), png_sha256=digest,
                                sidecar=str(link.with_suffix(".json").resolve()), config=meta["config"],
                                image_seconds=meta.get("seconds"), full=fp.name, full_px=small.width,
                                crop=cp.name))
    written = {Path(r[k]).name for r in records for k in ("full", "crop")}
    for stale in sorted(assets.glob("cmp_*.jpg")):  # tiles of a method no longer shown
        if stale.name not in written:
            stale.unlink()
    for m in METHODS:
        name, style, bold = HEADERS[m]
        label = r"\textbf{%s}" % name if bold else name
        center = xs[m] + tile + G_IN / 2
        tex_lines.append(r"\node[%s,anchor=base] at (%.4f,%.4f) {%s};" % (style, center, -(HEAD - 0.04), label))
    total = HEAD + 2 * tile + GV
    gpath = os.path.relpath(assets, tex.parent) if not a.out_dir else str(assets)
    doc = r"""%% Generated by fig_progressive.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{graphicx}
\usepackage{tikz}
\graphicspath{{%s/}}
\input{%s}
\definecolor{scaFrame}{HTML}{%s}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in,
  zoombox/.style={draw=scaFrame,line width=0.6pt},
  cropframe/.style={draw=scaFrame,line width=0.6pt},
  head/.style={font=\sffamily\fontsize{8}{9}\selectfont,text=scaInk},
  headours/.style={font=\sffamily\fontsize{8}{9}\selectfont,text=scaAccent}]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (gpath, "sca_colors.tex" if not a.out_dir else str(HERE / "sca_colors.tex"), FRAME, W, -total,
       "\n".join(tex_lines))
    tex.write_text(doc)
    subprocess.run([TECTONIC, "-X", "compile", tex.name, "--outdir", str(out_pdf.parent)], cwd=tex.parent,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env={**os.environ, "SOURCE_DATE_EPOCH": "0"})  # byte-identical PDF on rerun
    n_camp = secs["sca_v2_prog"]["n"]
    caption = (
        r"\textbf{Progressive \sca{} keeps one large subject, as HRDiT does; single-pass \sca{} can shrink or "
        r"repeat it.} Two prompts drawn at random from the original %d-prompt panel; same 1:1 ${%d\times%d}$ "
        r"crop per row; times in Table~\ref{tab:panel46}."
        % (n_panel, CROP, CROP))
    # the table's campaign means cover all 195 prompts, of which the original panel is a part
    assert n_panel < n_camp == N_TIMES, (n_panel, n_camp)
    snippet.write_text("\n".join([
        "% Generated by src/fig_progressive.py (TikZ source src/progressive.tex, assets "
        "figures/progressive/cmp_*).",
        r"\begin{figure}[t]", r"  \centering",
        r"  \includegraphics[width=\linewidth]{figures/progressive.pdf}",
        r"  \caption{" + caption + "}", r"  \label{fig:progressive}", r"\end{figure}", ""]))
    (assets / "provenance.json").write_text(json.dumps(dict(
        figure="fig:progressive", script="paper/figures/src/fig_progressive.py",
        outputs=dict(pdf="paper/figures/progressive.pdf", tikz="paper/figures/src/progressive.tex",
                     snippet="paper/figures/snippets/progressive.tex"),
        selection=dict(
            rule=(f"random.Random({SEED_DRAW}).sample(sorted(ids), {N_DRAW}) over the {n_panel} prompt ids of "
                  f"{PROMPTS_JSONL} with the {PREFIX} prefix stripped; rows in the drawn order. Random, not "
                  f"curated: no metric or visual judgment entered the choice."),
            prompts_jsonl=str(PROMPTS_JSONL), prompts_jsonl_sha256=sha(PROMPTS_JSONL), n_prompts=n_panel,
            drawn=drawn, reproduced=drawn == PROMPTS,
            prompts={pid: prompt_text[pid] for pid in PROMPTS}),
        crop_rule=(f"Automatic, not hand-placed: fig_qual_main.auto_box over {', '.join(METHODS)} with a "
                   f"{CROP}x{CROP} window, i.e. the window with the largest sum of the mean PIL FIND_EDGES "
                   f"response of the 8x box-downsampled greyscale images, at least {RES // 16} px from every "
                   f"border; the same window is cropped from every method of the row at native pixels."),
        crop_windows_xywh=boxes,
        headers=dict(sources={f"results/{run}/base_summary.csv": sha(RESULTS / run / "base_summary.csv")
                              for run in sorted(set(TIMES.values()) | set(CHECK.values()))},
                     cross_check={m: f"results/{run}/base_summary.csv" for m, run in CHECK.items()},
                     statistic=f"campaign mean seconds per image on shared GH200 nodes over all {N_TIMES} "
                               "prompts (Table tab:panel46), one decimal; not drawn in the figure",
                     methods=secs),
        protocols={"hrdit_dev_author": "HRDiT (FLUX.1-dev), stages 1024^2, 2048^2, 4096^2 with 30, 17, 10 steps",
                   "sca_v2_prog": "progressive SCA, final operator: HRDiT's stage schedule with SCA attention and "
                                  "the window schedule in the 2048^2 and 4096^2 stages, 57 steps in all",
                   "sca_v2": "single-pass SCA, final operator, 28 steps at 4096^2, guidance 4.5"},
        final_operator=dict(
            switched="2026-10-30: both SCA columns show the final operator for the same drawn prompts and seed",
            replaced=PUBLISHED, images=dict(sca_v2_prog=str(ROOT195 / "sca_v2_prog" / "4096x4096"),
                                            sca_v2="out/dupfix/sca_v2/4096x4096 (imgsel.final_twin)"),
            hrdit_images=str(ROOT195 / "hrdit_dev_author" / "4096x4096")),
        layout_in=dict(width=W, height=round(total, 4), tile=round(tile, 4),
                       gutters=dict(inside_group=round(G_IN, 4), between_groups=round(G_OUT, 4),
                                    rows=round(GV, 4)), header=HEAD, frame_hex=FRAME, frame_pt=0.6,
                       header_times_drawn=False),
        jpeg=dict(full="Lanczos to 600 dpi of printed width, quality 88, 4:2:0",
                  crop="native pixels, quality 90, 4:4:4"),
        records=records, height_in=round(total, 4),
        asset_sha256={p.name: sha(p) for p in sorted(assets.glob("cmp_*.jpg"))}), indent=2) + "\n")
    size = sum(p.stat().st_size for p in assets.glob("cmp_*.jpg"))
    print(f"{out_pdf} {out_pdf.stat().st_size / 1e6:.2f} MB, height {total:.3f} in, tile {tile:.3f} in, "
          f"assets {size / 1e6:.2f} MB, windows {boxes}, seconds "
          f"{ {m: v['label'] for m, v in secs.items()} }")


if __name__ == "__main__":
    main()
