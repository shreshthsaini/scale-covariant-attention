"""Main-text comparison grid at 4096^2 (fig:qual_prog): progressive SCA against AReA, UltraImage and HRDiT, TikZ PDF.

Columns, left to right: AReA, UltraImage, HRDiT and progressive SCA (ours, last, bold accent header). Rows: ROWS
curated prompts. Each cell is the method's full image (Lanczos to THUMB_DPI of its printed size) and, beside it, the
native 1:1 crop of one window shared by the whole row; the window is outlined on every full image by a 1 pt frame in
the zoom colour (scaZoom) drawn just outside it, over a faint dark halo. Method names appear once; no other text, no
numbers. Same prompt and seed (0) in every row, asserted from the sidecars (same prompt text and derived seed). The
images are those scored by the 195-prompt per-image CSVs: AReA and UltraImage eval/a4k195_main, HRDiT eval/a4k195_panel,
progressive SCA eval/prog195_sca_v2 (sca_v2_prog: HRDiT's released three-stage schedule, 1024^2 for 30 steps with
stock attention, then the last 17 steps at 2048^2 and the last 10 at 4096^2 with SCA attention; asserted).

Curated (recorded in figures/qual_prog/provenance.json; the caption must say so):
* Pool: the 195 Aesthetic-4K evaluation prompts minus the development prompts, minus every prompt of a curated
  figure (teaser, zoom4k, showcase, duplication examples, heads_view, method, ablation, failures, hero_prog) and the
  draw of the uncurated gallery fig:gallery_prog; 154 prompts.
* Fairness: each baseline's image must have ImageReward at or above that method's own 10th percentile over its 195
  images (FAIR_PCT, asserted), so that no baseline is shown on one of its rare catastrophic failures.
* Ranking: the mean, over the three baselines and over MUSIQ_c and ImageReward, of the rank (0 to 1, over the 195
  prompts) of progressive SCA's per-image advantage over that baseline; the top 35 eligible prompts were audited.
* Audit: progressive SCA's image viewed whole and as sixteen 1024 px tiles at 1:1 and rejected for any visible flaw,
  extra or repeated subject or pseudo-text; the baselines viewed whole and rejected when a failure is atypical for
  the method. For every kept prompt several 320 x 320 px windows were compared at 1:1 across the four methods.
* Window: placed by hand (curated) on the region of progressive SCA's image where the 1:1 crops show its advantage
  most clearly (crisp detail against the baselines' blur, smear, stripes, speckle or echo outlines), free of any
  flaw of the progressive image; one window per row, shared by the four methods. UltraImage shares progressive
  SCA's layout (same first stage and seed), so its crop shows the same content; AReA and HRDiT have their own layouts.
* Rows: the prompts with the clearest advantage after this audit, with varied subjects (SELECTED, with reasons;
  REJECTED lists every audited prompt not shown and why).

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_qual_prog.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

import fig_hero_prog as hp
import imgsel
import scafig

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent.parent
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")
EVAL = Path("/scratch/09032/saini_2/sca/eval")
RES, SEED = "4096x4096", "0"
# key, header (TeX), per-image CSV (base), extra CSV, method name in the CSVs
METHODS = [
    ("area", "AReA", EVAL / "a4k195_main" / "base" / "per_image.csv",
     EVAL / "a4k195_main" / "extra" / "extra_per_image.csv", "area"),
    ("ultra", "UltraImage", EVAL / "a4k195_main" / "base" / "per_image.csv",
     EVAL / "a4k195_main" / "extra" / "extra_per_image.csv", "ultraimage"),
    ("hrdit", "HRDiT", EVAL / "a4k195_panel" / "base" / "per_image.csv",
     EVAL / "a4k195_panel" / "extra" / "extra_per_image.csv", "hrdit_dev_author"),
    ("prog", r"\textbf{Progressive SCA}", EVAL / "prog195_sca_v2" / "base" / "per_image.csv",
     EVAL / "prog195_sca_v2" / "extra" / "extra_per_image.csv", "sca_v2_prog")]
OURS = "prog"
PROG = dict(method="sca_v2_prog", steps=57, attention="stock@native+sca@highres",
            stages=["1024x1024", "2048x2048", "4096x4096"], base_steps=30, highres_steps=[17, 10])
FAIR_PCT = 10          # baselines: ImageReward at or above their own 10th percentile on every shown prompt
WIN = 320              # window side (px)
POOL_N = 154           # eligible prompts (pool(), asserted)

SELECTED = [{'pid': '1046736963520652161',
  'window': (600, 1384),
  'why': 'mountain lake: clarity 4 of 5 in the audit; all four crops comparable',
  'crops': {'area': 'its own fir slope, mushy trees, diagonal hatching lattice on the snowfield',
            'ultra': 'the same cliff, blurred; rock and trees smeared together',
            'hrdit': 'dark fir slope under a snowy ridge, soft, little detail',
            'prog': 'crisp rock faces and crevices with individually resolved firs'}},
 {'pid': '8899498425633254014',
  'window': (1632, 960),
  'why': 'solitary tree at sunset: clarity 4 of 5 in the audit; AReA has a hard vertical seam at x 3068 '
         '(upper-right sky), outside the window',
  'crops': {'area': 'near-black twig silhouette in haze with a fine lattice, no needles',
            'ultra': 'the same crown, soft smeared sprigs with a halo',
            'hrdit': 'its crown as a field of repeated bead-like knobs, no needles',
            'prog': 'crisp individual pine-needle sprays against a clean sunset sky'}},
 {'pid': '58350587024203491',
  'window': (2040, 664),
  'why': 'armoured figure, skeletons and a creature: clarity 4 of 5 in the audit; all four crops comparable',
  'crops': {'area': 'hazy blurred creature chin',
            'ultra': 'the same creature face, soft, teeth smeared',
            'hrdit': 'sharp but incoherent wormy spiky texture',
            'prog': 'crisp creature face with glowing eyes, nostrils and clean fangs'}},
 {'pid': 'm2729538226706404616',
  'window': (2536, 256),
  'why': "woman's profile in clouds: clarity 4 of 5 in the audit; all four crops comparable",
  'crops': {'area': 'soft hair strands behind water droplets and haze',
            'ultra': 'the same hair, soft, no curl definition',
            'hrdit': 'sky with thin cloud wisps and a few loose strands',
            'prog': 'glossy, sharply resolved curls with single strands and golden highlights'}}]
REJECTED = {'11799097571596124': 'progressive image rejected: legible pseudo-text on the chest plate, belt tag and shoulder '
                      'patch; 8 px dot lattice in the defocused background at 4x',
 'm2978915343052374306': 'passed the audit (clarity 3) but not chosen: advantage clear against UltraImage only; '
                         'HRDiT crisp throughout',
 'm5691444384207183173': 'excluded: AReA atypical: a 16 px lattice of about 10 grey levels, 4th strongest of '
                         "AReA's 195 images (median 0.28)",
 '6648013380537069782': 'progressive image rejected: three sets of toes (an extra foot) at the bottom; UltraImage '
                        'shares the layout',
 '7414267858891695058': 'progressive image rejected: legible pseudo-text on the foreground book spines (readable '
                        "in the overview); AReA's image is also atypical (emptied, extremely blurred scene)",
 '880657115260655727': "progressive image rejected: legible pseudo-text sign on the central facade ('KLERATTANKS & "
                       "ATHERLARE'), readable in the overview, plus a second sign",
 'm8314017227778408466': 'excluded: advantage weak (clarity 2, over UltraImage only); a glow halo along the whole '
                         'silhouette is faintly visible at 512 px',
 '7961019668960160206': 'progressive image rejected: extra moons (one on the ring in front of the planet visible '
                        'at print, a small dark moon, a grey crescent) for the one the prompt asks for; an echo '
                        'circle on the dark side',
 '8065862701789418887': 'excluded: an 8 px diagonal checker of 2-6 grey levels over the whole snowy massif, '
                        'clearly visible at 1:1',
 '9112592444131713102': 'progressive image rejected: the bottom-left snow mass reads at print size as an animal '
                        'head in profile (an implied extra subject)',
 '2100482621711410240': "passed the audit (clarity 4) but not chosen: AReA's echo contours are extreme here; "
                        'pseudo-text on the map at 1:1; HRDiT as crisp as ours',
 '4673559490456019105': 'excluded: weak advantage (clarity 3, AReA not worse; shallow depth of field leaves much '
                        'of our image soft); illegible pseudo-text lines on the book pages',
 'm4460699761507361961': 'progressive image rejected: legible pseudo-text on the shirts and the bucket, visible in '
                         'the overview',
 'm4462838239161450556': 'excluded: a diagonal hairline hatch over the whole sky, clearly visible at 1:1 and '
                         'faintly in the overview; a rotated pseudo-text watermark on the right edge',
 '2022937813048109690': "excluded: an 8 px lattice and hatch in the frosted trees clearly visible at 1:1; HRDiT's "
                        'full image carries a garbled caption line across its top-left corner',
 '6384782444147762726': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 '5941097972571555184': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 '175243114551627459': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 '5826245701465116957': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 'm7320315357894569493.Jpg': 'progressive image rejected in the page-1 hero audit '
                             '(figures/hero_prog/provenance.json)',
 '1195301905298426486': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 'm2742856837225126754': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 'm9039328566106297566': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 'm2712631331773394876': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 '7543519915112968451': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 'm3268147817845827695': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 'm2670470656524755471': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 'm7946139341219184985': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 '1139740195494332405': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 '4820545723762191815': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 'm5795767365495583614': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 '3943607367684487957': 'progressive image rejected in the page-1 hero audit (figures/hero_prog/provenance.json)',
 '7241348597486365303': 'not audited: Emmys logos and pseudo-text',
 '3191725509755028570': 'not audited: numbered silhouettes (text, counts)',
 'm4469448794852185540': 'not audited: numbered silhouettes (text, counts)',
 '3411143122470771726': 'not audited: anime',
 '3305730548209556640': 'not audited: flat graphic, little detail'}
AUDIT_NOTES = {'1046736963520652161': 'MINOR: 1:1 only: very faint 8 px checker in smooth water; fine grain on hazy mountain '
                        'faces',
 '8899498425633254014': 'MINOR: 1:1 only: fine diagonal hatch in sky gaps of the lower canopy; drawn-on bright '
                        'cloud rims near the sun; canvas-like grain on rocks; faint x~2115 vertical line in flat '
                        'areas',
 '58350587024203491': "MINOR: 1:1 only: about seven claw toes on the front skeleton's foot; the creature grips a "
                      'separate pale pole behind the blade; a faint fog-shrouded silhouette at top-left reads as '
                      'dead branches at print; overlapping background skulls',
 'm2729538226706404616': 'MINOR: 1:1 only: faint 4/8 px lattice (0.2-0.4 levels) in flat out-of-focus clouds, '
                         'visible at 3-4x; a tiny orange glint in the hair; thin cyan halo on the upper lip'}

W = 397.5                    # text width (TeX pt)
G_PAIR, G_METHOD, G_ROW = 1.0, 2.0, 2.0  # gutters (pt): image to crop, between methods, between rows
HEAD = 10.0                  # header band (pt)
THUMB_DPI = 700              # full images: Lanczos to this many px per printed inch
FRAME_LW, HALO_LW = 1.0, 1.7


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def short(pid: str) -> str:
    return pid.rsplit("_", 1)[-1].replace("minus", "m")


_rows: dict = {}


def rows(path: Path) -> list[dict]:
    if path not in _rows:
        with open(path, newline="") as fh:
            _rows[path] = [r for r in csv.DictReader(fh) if r["seed"] == SEED and r["res"] == RES]
    return _rows[path]


def per_image(key: str, base: Path, extra: Path, name: str) -> dict[str, dict]:
    out = {short(r["prompt_id"]): dict(csv_prompt_id=r["prompt_id"], path=r["path"], musiq_c=float(r["musiq_crop1024"]))
           for r in rows(base) if r["method"] == name}
    for r in rows(extra):
        if r["method"] == name:
            e = out[short(r["prompt_id"])]
            assert e["path"] == r["path"], (key, r["prompt_id"])
            e.update(ir=float(r["ir"]), hpsv21=float(r["hpsv21"]))
    assert len(out) == 195, (key, len(out))
    vals = sorted(e["ir"] for e in out.values())
    for e in out.values():
        e["ir_pct"] = 100 * sum(v <= e["ir"] for v in vals) // len(vals)
    return out


def pool(all195: list[str]) -> dict:
    """195 minus development prompts, every prompt of a curated figure (hp.CURATED plus hero_prog) and the draw of
    fig:gallery_prog; read from the figures' provenance files."""
    dev = imgsel.dev4_ids()
    curated = {}
    for name, (path, fields) in hp.CURATED.items():
        curated[name] = sorted(hp.named(json.loads((FIG / path).read_text()), fields))
    curated["hero_prog (fig:prog_hero)"] = sorted(
        short(s) for s in json.loads((FIG / "hero_prog" / "provenance.json").read_text())["shown"])
    gallery = sorted(json.loads((FIG / "gallery_prog" / "provenance.json").read_text())["draw"])
    excluded = set(dev).union(*map(set, curated.values()), gallery)
    eligible = sorted(set(all195) - excluded)
    return dict(development=sorted(dev), curated=curated, gallery_prog_draw=gallery, eligible_count=len(eligible),
                eligible_ids=eligible)


def layout() -> tuple[float, list[float]]:
    """Tile side t and the x of each method's full image (its crop follows at x + t + G_PAIR)."""
    t = (W - 4 * G_PAIR - 3 * G_METHOD) / 8
    xs = [i * (2 * t + G_PAIR + G_METHOD) for i in range(4)]
    assert abs(xs[-1] + 2 * t + G_PAIR - W) < 1e-9
    return t, xs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None, help="trial run: write everything here")
    a = ap.parse_args()
    base = a.out_dir if a.out_dir else FIG
    assets, out_pdf = base / "qual_prog", base / "qual_prog.pdf"
    assets.mkdir(parents=True, exist_ok=True)
    for old in assets.glob("*.jpg"):  # this folder holds only this script's output
        old.unlink()
    scafig.write_tikz_colors(str(HERE / "sca_colors.tex"))
    data = {k: per_image(k, b, x, n) for k, _, b, x, n in METHODS}
    elig = pool(sorted(data[OURS]))
    assert elig["eligible_count"] == POOL_N, elig["eligible_count"]
    assert all(sel["pid"] in elig["eligible_ids"] for sel in SELECTED), [sel["pid"] for sel in SELECTED]
    t, xs = layout()
    total = HEAD + len(SELECTED) * t + (len(SELECTED) - 1) * G_ROW
    tex, records, written = [], [], []

    def node(x, y, side, name):
        tex.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                   r"{\includegraphics[width=%.4fpt,height=%.4fpt]{%s}};" % (x, -y, side, side, name))

    for r, sel in enumerate(SELECTED):
        pid, (bx, by) = sel["pid"], sel["window"]
        assert bx % 8 == 0 and by % 8 == 0 and 256 <= bx <= 4096 - 256 - WIN and 256 <= by <= 4096 - 256 - WIN, pid
        assert pid not in REJECTED, pid
        top = HEAD + r * (t + G_ROW)
        metas, recs = {}, {}
        for i, (key, _, _, _, name) in enumerate(METHODS):
            e = data[key][pid]
            if key != OURS:
                assert e["ir_pct"] >= FAIR_PCT, (pid, key, e["ir_pct"])  # fairness rule
            png = Path(e["path"])
            meta = json.loads(png.with_suffix(".json").read_text())
            if meta.get("complete") is not True or meta.get("png_sha256") != sha(png):
                raise ValueError(f"incomplete or hash-mismatched source {png}")
            metas[key] = meta
            img = Image.open(png).convert("RGB")
            assert img.size == (4096, 4096), (png, img.size)
            px = round(THUMB_DPI * t / 72.27)
            full, crop = assets / f"{pid}_{key}.jpg", assets / f"{pid}_{key}_crop.jpg"
            img.resize((px, px), Image.Resampling.LANCZOS).save(full, "JPEG", quality=90, optimize=True, subsampling=0)
            img.crop((bx, by, bx + WIN, by + WIN)).save(crop, "JPEG", quality=95, optimize=True, subsampling=0)
            written += [full, crop]
            x = xs[i]
            node(x, top, t, full.name)
            node(x + t + G_PAIR, top, t, crop.name)
            s, d = t / 4096, FRAME_LW / 2
            box = (x + bx * s - d, -(top + by * s - d), x + (bx + WIN) * s + d, -(top + (by + WIN) * s + d))
            tex.append(r"\draw[halo] (%.4f,%.4f) rectangle (%.4f,%.4f);" % box)
            tex.append(r"\draw[frame] (%.4f,%.4f) rectangle (%.4f,%.4f);" % box)
            recs[key] = dict(csv_method=name, source=str(png), png_sha256=meta["png_sha256"], full=full.name,
                             full_px=px, crop=crop.name, metrics={k: e[k] for k in ("ir", "ir_pct", "musiq_c",
                                                                                     "hpsv21")},
                             crop_shows=sel["crops"][key])
        cfg = {k: m["config"] for k, m in metas.items()}
        assert len({c["prompt"] for c in cfg.values()}) == 1, pid             # same prompt
        assert len({str(c["derived_seed"]) for c in cfg.values()}) == 1, pid  # same initial noise
        pc = cfg[OURS]
        assert dict(method=pc["method"], steps=pc["steps"], attention=pc["attention"],
                    stages=pc["progressive"]["stages"], base_steps=pc["progressive"]["base_steps"],
                    highres_steps=pc["progressive"]["highres_steps"]) == PROG, pid
        records.append(dict(row=r, prompt_id=pid, csv_prompt_id=data[OURS][pid]["csv_prompt_id"],
                            prompt=pc["prompt"], derived_seed=pc["derived_seed"], window_xys=[bx, by, WIN],
                            why=sel["why"], progressive_audit=AUDIT_NOTES.get(pid), methods=recs))
    for i, (key, head, *_rest) in enumerate(METHODS):
        tex.append(r"\node[%s,anchor=base] at (%.4f,%.4f) {%s};" % ("headours" if key == OURS else "head",
                                                                    xs[i] + t + G_PAIR / 2, -(HEAD - 2.6), head))
    doc = r"""%% Generated by paper/figures/src/fig_qual_prog.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{graphicx}
\usepackage{tikz}
\input{sca_colors.tex}
\begin{document}
\begin{tikzpicture}[x=1pt,y=1pt,
  halo/.style={draw=black,draw opacity=0.35,line width=%.2fpt},
  frame/.style={draw=scaZoom,line width=%.2fpt},
  head/.style={inner sep=0pt,font=\sffamily\fontsize{8}{9}\selectfont,text=scaInk},
  headours/.style={inner sep=0pt,font=\sffamily\fontsize{8}{9}\selectfont,text=scaAccent}]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (HALO_LW, FRAME_LW, W, -total, "\n".join(tex))
    (assets / "qual_prog.tex").write_text(doc)
    shutil.copy(HERE / "sca_colors.tex", assets / "sca_colors.tex")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([TECTONIC, "-X", "compile", "qual_prog.tex", "--outdir", tmp], cwd=assets, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env=dict(os.environ, SOURCE_DATE_EPOCH="0"))
        shutil.copy(Path(tmp) / "qual_prog.pdf", out_pdf)
    size = sum(p.stat().st_size for p in written)
    prov = dict(
        figure="qual_prog (fig:qual_prog)", script="paper/figures/src/fig_qual_prog.py", pdf=str(out_pdf),
        pdf_sha256=sha(out_pdf), curated=True,
        selection=dict(pool="195 minus development prompts, curated figures (teaser, zoom4k, showcase, dup, heads_view, "
                            "method, ablation, failures, hero_prog) and fig:gallery_prog's draw: 154 prompts",
                       fairness=f"every baseline's ImageReward at or above its own p{FAIR_PCT} (asserted)",
                       ranking=("mean over baselines (AReA, UltraImage, HRDiT) and metrics (MUSIQ_c, ImageReward) of "
                                "the rank over the 195 prompts of progressive SCA's per-image advantage; top 35 "
                                "eligible prompts audited"),
                       window_rule=("curated: placed by hand where the 1:1 crops show progressive SCA's advantage "
                                    "most clearly, free of any flaw of the progressive image; shared by the row"),
                       rejected=REJECTED, eligible=elig),
        layout_pt=dict(width=W, height=round(total, 3), tile=round(t, 3), header=HEAD,
                       gutters=dict(pair=G_PAIR, method=G_METHOD, row=G_ROW), window_px=WIN,
                       crop_ppi=round(WIN / (t / 72.27), 1), frame=dict(colour="scaZoom", lw=FRAME_LW, halo=HALO_LW),
                       text_pt=8),
        jpeg=dict(full=f"Lanczos to {THUMB_DPI} px per printed inch, quality 90, 4:4:4",
                  crop="native 1:1, quality 95, 4:4:4", total_bytes=size),
        rows=records, asset_sha256={p.name: sha(p) for p in sorted(written)})
    (assets / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"{out_pdf} {W} x {total:.1f} pt, tile {t:.2f} pt, crop {WIN / (t / 72.27):.0f} ppi, JPEGs {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
