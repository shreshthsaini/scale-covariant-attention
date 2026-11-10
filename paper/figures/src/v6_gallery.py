"""Appendix sample pages of v6: one family with Figure 5 (same grid, labels, red 1 pt boxes, native crops, crop rule).

Every figure is written as paper/figures/v6/<stem>.pdf with a float snippet paper/figures/v6/snippets/<stem>.tex
(\\includegraphics at natural width, caption, the v5.1 \\label), and a provenance file
paper/figures/v6/<stem>/provenance.json (sources with SHA256, windows, the selection rule and its draw).

Selection. Figures whose v5.1 prompts came from a seeded draw keep that frozen draw (read from the v5.1 provenance and
shown in draw order). Figures whose v5.1 prompts were picked by eye get a seeded draw here (FIGS[...]["rule"]); no
figure is chosen by eye. Crop rule everywhere: the window of fixed size with the highest mean edge energy averaged over
the methods shown (v6_grid); the aspect-ratio figure, one image per size, takes each image's own window.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_gallery.py [stem ...] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402

import imgsel  # noqa: E402
import v6_common as vc  # noqa: E402
import v6_grid as vg  # noqa: E402
import v6_hero  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
FIG = vc.FIG
SNIP = vc.V6 / "snippets"
W, G, GG = vc.TEXT_PT, vc.G, vc.GG
LW = 19.0                  # rotated label column (two lines of 8 pt)
OUT = "/scratch/09032/saini_2/sca/out"
EVAL = vc.EVAL

LAB = {"area": "AReA", "ultraimage": "UltraImage", "hrdit_dev_author": "HRDiT", "dype": "DyPE", "sega_official": "SEGA",
       "full": r"Full\\attention", "sca_v2": r"Single-pass\\SCA", "sca_v2_prog": r"Progressive\\SCA",
       "sca": r"Single-pass\\SCA", "sca_nb": r"SCA w/o\\schedule", "flux2_full": r"Full\\attention",
       "flux2_sca": r"SCA w/o\\schedule", "krea_area": "AReA", "krea_dype": "DyPE", "krea_sega": "SEGA",
       "krea_sca": r"Single-pass\\SCA", "abl_v0": r"No\\opening", "abl_local": r"Local\\only",
       "abl_uniform": r"Shared\\$\pi(\sigma)$", "abl_global": r"Global\\only", "abl_mean": r"Mean\\pooling",
       "abl_sca": r"SCA w/o\\schedule", "prog_headmean": r"Shared\\$\pi(\sigma)$",
       "prog_layermean": r"Per-layer\\$\pi_\ell(\sigma)$", "prog_win": r"Window\\only",
       "f1_prog": r"FLUX.1-dev\\progressive", "f2_sca": r"FLUX.2-dev\\single-pass", "qwen_sca": r"Qwen-Image\\single-pass"}
OURS_KEYS = {"sca_v2", "sca_v2_prog", "sca", "sca_nb", "flux2_sca", "krea_sca", "abl_sca", "f1_prog", "f2_sca",
             "qwen_sca"}


def ours(methods):
    return {m: vc.SCA for m in methods if m in OURS_KEYS}


# ------------------------------------------------------------------------------------------------ sources
def frozen(gallery: str, rename: dict | None = None) -> tuple[list, dict]:
    """(ids in draw order, {id: {method: png}}) from a v5.1 gallery provenance; methods renamed by `rename`."""
    man = json.loads((FIG / gallery / "provenance.json").read_text())
    rename = rename or {}
    src = {}
    for p in man["prompts"]:
        sid = p.get("short_id") or imgsel.short_id(p["id"])
        src[sid] = {rename.get(m, m): s["png"] for m, s in p["sources"].items()}
    order = (man.get("display") or {}).get("draw_order") or man.get("draw") or [
        p.get("short_id") or imgsel.short_id(p["id"]) for p in man["prompts"]]
    return order, src


def eval_paths(run: str, method: str, res: str) -> dict:
    """{short id: png} for one method of an evaluation root's per-image CSV at seed 0."""
    out = {}
    with open(EVAL / run / "base" / "per_image.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["method"] == method and r["seed"] == "0" and r["res"] == res:
                out[imgsel.short_id(r["prompt_id"])] = r["path"]
    return out


def eligible195(pool) -> list:
    return sorted(set(pool["sca_v2_prog"]) - imgsel.dev4_ids())


# ------------------------------------------------------------------------------------------------ output
def snippet(stem: str, floats: list[tuple[str, str, str]]) -> Path:
    """floats: (pdf stem, caption, label). One file holding every float of the figure."""
    SNIP.mkdir(parents=True, exist_ok=True)
    body = ["% Written by paper/figures/src/v6_gallery.py; edit the script, not this file."]
    for pdf, cap, lab in floats:
        body += [r"\begin{figure}[p]", r"  \centering", r"  \includegraphics{figures/v6/%s.pdf}" % pdf,
                 r"  \caption{%s}" % cap, r"  \label{%s}" % lab, r"\end{figure}"]
    p = SNIP / f"{stem}.tex"
    p.write_text("\n".join(body) + "\n")
    return p


def rows_geo(n: int, full_h: float, aspect: float = 1.0) -> float:
    """crop width that fills the text width with n examples per band at row height full_h."""
    per = (W - LW - G - (n - 1) * GG) / n
    return per - G - full_h * aspect


def run_grid(stem, exs, methods, full_h, n_per_band, blocks, out_dir, crop_ppi=250, aspect=1.0, check_noise=True,
             labels=None):
    crop_w = rows_geo(n_per_band, full_h, aspect)
    assert crop_w >= 72.0, (stem, crop_w)
    crop_px = (round(crop_ppi * crop_w / 72.27) // 8 * 8, round(crop_ppi * full_h / 72.27) // 8 * 8)
    vg.windows(exs, crop_px, check_noise=check_noise)
    assets = (out_dir or vc.V6) / stem
    pdf = (out_dir or vc.V6) / f"{stem}.pdf"
    rec = vg.build(exs, methods, labels or LAB, "rows", assets, pdf, stem, full_h=full_h, crop_w=crop_w,
                   label_w=LW, ours=ours(methods), blocks=blocks)
    rec["crop_px"] = list(crop_px)
    return rec, assets


def examples(ids, src: dict, methods: list) -> list:
    return [vg.Example(key=s, paths={m: src[s][m] for m in methods}) for s in ids]


# ------------------------------------------------------------------------------------------------ figures
def fig_prog16(out_dir):
    """Figure 5 expanded: 18 more evaluation prompts, 9 per page."""
    pool = vc.pool195()
    texts = vc.prompts195(pool)
    import v6_qual_main as qm
    shown5 = set(vc.draw_by_category(texts, qm.CATS[:2], qm.SEED, exclude=imgsel.dev4_ids()).values())
    frozen6 = set(json.loads((FIG / "gallery_prog" / "provenance.json").read_text())["draw"])
    elig = sorted(set(eligible195(pool)) - shown5 - frozen6)
    ids = vc.seeded_draw(elig, 18, 16)
    methods = ["area", "ultraimage", "hrdit_dev_author", "sca_v2_prog"]
    src = {s: {m: pool[m][s]["path"] for m in methods} for s in ids}
    recs = []
    for k, stem in enumerate(("gallery_prog16", "gallery_prog16b")):
        rec, assets = run_grid(stem, examples(ids[9 * k: 9 * k + 9], src, methods), methods, 45.5, 3, 3, out_dir)
        recs.append((stem, rec, assets))
    rule = ("random.Random(16).sample(sorted(eligible), 18); eligible = the 195 evaluation prompts minus the four "
            "development prompts, the two prompts of Figure 5 and the six of fig:gallery_prog")
    cap = (r"\textbf{Progressive \sca{} against AReA, UltraImage and HRDiT at $4096^2$, prompts %d to %d of 18.} "
           r"Eighteen evaluation prompts outside Figures~\ref{fig:qual_prog} and~\ref{fig:gallery_prog}, chosen by a "
           r"fixed seed (seed 16), seed 0 "
           r"for every image. Each image is followed by the native 1:1 crop of its red box, placed at the highest "
           r"edge energy averaged over the four methods.")
    floats = [("gallery_prog16", cap % (1, 9) + r" Figure~\ref{fig:gallery_prog16b} continues.", "fig:gallery_prog16"),
              ("gallery_prog16b", cap % (10, 18), "fig:gallery_prog16b")]
    return recs, floats, dict(rule=rule, seed=16, draw=ids)


def fig_prog(out_dir):
    """The frozen six-prompt draw of fig:gallery_prog with every 4096^2 method of the comparison."""
    pool = vc.pool195()
    man = json.loads((FIG / "gallery_prog" / "provenance.json").read_text())
    ids = man["draw"]
    methods = ["area", "ultraimage", "hrdit_dev_author", "sca_v2", "sca_v2_prog"]
    src = {s: {m: pool[m][s]["path"] for m in methods} for s in ids}
    rec, assets = run_grid("gallery_prog", examples(ids, src, methods), methods, 45.5, 3, 2, out_dir)
    cap = (r"\textbf{Single-pass \sca{} keeps more native detail; progressive \sca{} scores higher on preference.} "
           r"Six evaluation prompts chosen by a fixed seed (seed 0, frozen before any image was viewed), "
           r"seed 0 for every image, at $4096^2$. Crops are native 1:1 windows at the highest edge energy averaged "
           r"over the five methods.")
    return [("gallery_prog", rec, assets)], [("gallery_prog", cap, "fig:gallery_prog")], dict(
        rule=man["selection"] if isinstance(man.get("selection"), str) else "frozen v5.1 draw (gallery_prog)",
        seed=man.get("selection_seed"), draw=ids)


def fig_zoom(out_dir):
    """fig:qual_main of the appendix: three more categories of Figure 5's rule, single-pass SCA added."""
    pool = vc.pool195()
    texts = vc.prompts195(pool)
    cats = ["architecture", "animal", "still life"]
    picks = vc.draw_by_category(texts, cats, 0, exclude=imgsel.dev4_ids())
    ids = [picks[c] for c in cats]
    methods = ["area", "ultraimage", "hrdit_dev_author", "sca_v2", "sca_v2_prog"]
    src = {s: {m: pool[m][s]["path"] for m in methods} for s in ids}
    rec, assets = run_grid("zoom4k", examples(ids, src, methods), methods, 45.5, 3, 1, out_dir)
    cap = (r"\textbf{Crop by crop at $4096^2$.} One evaluation prompt from each of the architecture, animal and "
           r"still-life categories, drawn with seed 0 by the rule of Figure~\ref{fig:qual_prog}; seed 0 for every "
           r"image. Crops are native 1:1 windows at the highest edge energy averaged over the five methods.")
    return [("zoom4k", rec, assets)], [("zoom4k", cap, "fig:qual_main")], dict(
        rule="v6_common.draw_by_category over architecture, animal, still life with seed 0 (Figure 5's rule)",
        seed=0, draw=picks)


def fig_aspect(out_dir):
    """fig:hero: single-pass SCA at three aspect ratios on the same prompts."""
    sizes = ["4096x4096", "2048x4096", "2048x6144"]
    runs = {"4096x4096": "a4k195_main_sca_v2", "2048x4096": "a4k195_2048x4096_sca_v2",
            "2048x6144": "a4k195_2048x6144_sca_v2"}
    paths = {r: eval_paths(runs[r], "sca_v2", r) for r in sizes}
    elig = sorted((set(paths["4096x4096"]) & set(paths["2048x4096"]) & set(paths["2048x6144"])) - imgsel.dev4_ids())
    ids = vc.seeded_draw(elig, 4, 3)
    h = (W - 2 * G) / 6               # square h, 2:1 2h, 3:1 3h
    ch = 52.0                          # crop strip height
    H = len(ids) * (h + G + ch) + (len(ids) - 1) * GG
    out = out_dir or vc.V6
    assets, pdf = out / "aspect", out / "aspect.pdf"
    cv = vc.Canvas(W, H, assets)
    rec = dict(examples=[])
    from multiprocessing import Pool
    jobs = [(paths[r][s], r, s) for s in ids for r in sizes]
    with Pool(len(jobs)) as pool:
        maps = pool.map(_edge, [j[0] for j in jobs])
    k = 0
    for i, s in enumerate(ids):
        y = i * (h + G + ch + GG)
        x = 0.0
        vc.same_prompt_and_noise({r: paths[r][s] for r in sizes})  # same prompt and initial noise at every size
        for j, r in enumerate(sizes):
            a, (iw, ih) = maps[k]
            k += 1
            fw = h * iw / ih
            cw = fw
            wpx, hpx = round(250 * cw / 72.27) // 8 * 8, round(250 * ch / 72.27) // 8 * 8
            win = vc.best_windows([a], wpx, hpx, k=1)[0]
            im = Image.open(paths[r][s]).convert("RGB")
            tn, cn = f"{s}_{r}_full.jpg", f"{s}_{r}_crop_{win[0]}_{win[1]}.jpg"
            vc.save_jpeg(vc.thumb(im, fw, h), assets / tn, vc.THUMB_Q)
            vc.save_jpeg(im.crop((win[0], win[1], win[0] + win[2], win[1] + win[3])), assets / cn, vc.CROP_Q)
            cv.image(x, y, fw, h, tn)
            sc = fw / iw
            cv.box(x + win[0] * sc, y + win[1] * sc, x + (win[0] + win[2]) * sc, y + (win[1] + win[3]) * sc,
                   inside=False)
            cv.image(x, y + h + G, cw, ch, cn)
            cv.box(x, y + h + G, x + cw, y + h + G + ch, inside=True)
            if i == 0:
                hh, ww = r.split("x")
                cv.text(x + 3, y + 3, r"%s\,\texttimes\,%s" % (ww, hh), size=vc.FS_TAG, chip=True)
            rec["examples"].append(dict(key=s, res=r, window_px=list(win), source=paths[r][s],
                                        crop_ppi=round(win[2] / (cw / 72.27), 1)))
            x += fw + G
    cv.compile(pdf, "aspect")
    rec.update(pdf=str(pdf), pdf_bytes=pdf.stat().st_size, size_in=[round(W / 72.27, 3), round(H / 72.27, 3)])
    cap = (r"\textbf{One reach table serves every aspect ratio.} Single-pass \sca{} on FLUX.1-dev at $4096^2$, "
           r"$2048\times4096$ and $2048\times6144$ for four evaluation prompts chosen by a fixed seed (seed 3), seed 0 "
           r"for every image. Each crop shows native pixels of the red box above it, at the highest edge energy of "
           r"that image.")
    return [("aspect", rec, assets)], [("aspect", cap, "fig:hero")], dict(
        rule="random.Random(3).sample(sorted(prompts with single-pass SCA at all three sizes, minus development), 4)",
        seed=3, draw=ids)


def _edge(p):
    im = Image.open(p).convert("RGB")
    return vc.detail_map(im), im.size


def fig_4096(out_dir):
    order, src = frozen("gallery_4096", {"sca": "sca_v2"})
    methods = ["area", "sca_v2"]
    out = []
    for stem, part in (("gallery_4096", order[:8]), ("gallery_4096_p2", order[8:16])):
        rec, assets = run_grid(stem, examples(part, src, methods), methods, 63.5, 2, 4, out_dir)
        out.append((stem, rec, assets))
    cap = (r"\textbf{Single-pass \sca{} against AReA at $4096^2$, prompts %d to %d of the frozen draw.} Evaluation "
           r"prompts chosen by a fixed seed (seed 0) before any image was viewed, in draw order, seed 0 for every image. "
           r"Crops are native 1:1 windows at the highest edge energy averaged over both methods.")
    floats = [("gallery_4096", cap % (1, 8), "fig:gallery_4096"), ("gallery_4096_p2", cap % (9, 16),
                                                                   "fig:gallery_4096_p2")]
    return out, floats, dict(rule="frozen v5.1 draw random.Random(0).sample(sorted(eligible), 32), first 16 in "
                                  "draw order", seed=0, draw=order[:16])


def fig_wide(out_dir, res):
    g = "gallery_4096x2048" if res == "2048x4096" else "gallery_6144x2048"
    order, src = frozen(g, {"sca": "sca_v2"})
    methods = ["area", "sca_v2"]
    aspect = 2.0 if res == "2048x4096" else 3.0
    rec, assets = run_grid(g, examples(order[:4], src, methods), methods, 64.0, 1, 4, out_dir, aspect=aspect)
    hh, ww = res.split("x")
    seed = 33 if res == "2048x4096" else 39
    cap = (r"\textbf{Single-pass \sca{} against AReA at $%s\times%s$.} The first four prompts of a draw with a fixed "
           r"seed (seed %d) frozen before any image was viewed, seed 0 for every image. Crops are native 1:1 windows at "
           r"the highest edge energy averaged over both methods.") % (hh, ww, seed)
    label = "fig:gallery_4096x2048" if res == "2048x4096" else "fig:gallery_6144x2048"
    return [(g, rec, assets)], [(g, cap, label)], dict(rule=f"frozen v5.1 draw (seed {seed}), first four",
                                                        seed=seed, draw=order[:4])


def fig_flux2(out_dir):
    order, src = frozen("gallery_flux2")
    methods = ["flux2_full", "flux2_sca"]
    rec, assets = run_grid("gallery_flux2", examples(order[:4], src, methods), methods, 63.5, 2, 2, out_dir)
    cap = (r"\textbf{FLUX.2-dev at $4096^2$: full attention blurs, \sca{} without the schedule is sharp but distorts "
           r"the portrait.} Four of 50 evaluation prompts chosen by a fixed seed (seed 17), seed 0 for every image. "
           r"Crops are native 1:1 windows at the highest edge energy averaged over both methods.")
    return [("gallery_flux2", rec, assets)], [("gallery_flux2", cap, "fig:gallery_flux2")], dict(
        rule="frozen v5.1 draw (seed 17)", seed=17, draw=order[:4])


def fig_ablation(out_dir):
    roots = {"abl_v0": "abl/abl_v0", "abl_local": "abl/abl_localonly_nb", "abl_uniform": "abl/abl_uniform_nb",
             "abl_global": "abl/abl_globalonly", "abl_mean": "abl/abl_mean_nb", "abl_sca": "main/sca_nb"}
    res = "4096x4096"
    idx = {m: imgsel.index(f"{OUT}/{r}/{{res}}", res) for m, r in roots.items()}
    common = set.intersection(*[set(imgsel.short_id(p) for p in d) for d in idx.values()])
    elig = sorted(common - imgsel.dev4_ids())
    assert len(elig) == 46, len(elig)
    ids = vc.seeded_draw(elig, 2, 46)
    src = {s: {m: next(p for pid, p in idx[m].items() if imgsel.short_id(pid) == s) for m in roots} for s in ids}
    methods = list(roots)
    rec, assets = run_grid("ablation", examples(ids, src, methods), methods, 85.0, 2, 1, out_dir)
    cap = (r"\textbf{Both views and the opening are needed.} Two of the \nabl{} ablation prompts chosen by a fixed seed "
           r"(seed 46) at $4096^2$, seed 0 for every image; every row is \sca{} without the schedule with one part "
           r"changed (Table~\ref{tab:ablation}). Crops are native 1:1 windows at the highest edge energy averaged over "
           r"the six variants.")
    return [("ablation", rec, assets)], [("ablation", cap, "fig:ablation_grid")], dict(
        rule="random.Random(46).sample(sorted(46 ablation prompts), 2)", seed=46, draw=ids)


def fig_showcase_models(out_dir):
    picks = v6_hero.draw()
    ids = [picks[c] for c in v6_hero.ORDER]
    roots = {"f1_prog": "showcase/flux1_sca_v2_prog", "f2_sca": "showcase/flux2_sca_v2", "qwen_sca": "showcase/qwen_sca"}
    src = {}
    for s in ids:
        src[s] = {}
        for m, r in roots.items():
            pngs = sorted(glob.glob(f"{OUT}/{r}/4096x4096/{s}_*_seed0.png"))
            assert len(pngs) == 1, (m, s, pngs)
            src[s][m] = pngs[0]
    methods = list(roots)
    rec, assets = run_grid("showcase_models", examples(ids, src, methods), methods, 60.0, 2, 3, out_dir,
                           check_noise=False)
    for e in rec["examples"]:
        pr = {vc.sidecar(p)["config"]["prompt"] for p in e["sources"].values()}
        assert len(pr) == 1, e["key"]
    cap = (r"\textbf{The same six prompts on three models at $4096^2$.} Progressive \sca{} on FLUX.1-dev (as in "
           r"Figure~\ref{fig:prog_hero}) and single-pass \sca{} on FLUX.2-dev and Qwen-Image, one showcase prompt per "
           r"category drawn with seed 0, seed 0 for every image. Crops are native 1:1 windows at the highest edge "
           r"energy averaged over the three models.")
    return [("showcase_models", rec, assets)], [("showcase_models", cap, "fig:showcase_models")], dict(
        rule="v6_hero.draw(): one showcase prompt per category, seed 0", seed=0, draw=picks)


def fig_all_methods(out_dir):
    pool = vc.pool195()
    elig = eligible195(pool)
    ids = vc.seeded_draw(elig, 2, 8)
    methods = ["full", "dype", "sega_official", "area", "ultraimage", "hrdit_dev_author", "sca_v2", "sca_v2_prog"]
    src = {s: {m: pool[m][s]["path"] for m in methods} for s in ids}
    rec, assets = run_grid("all_methods", examples(ids, src, methods), methods, 66.0, 2, 1, out_dir)
    cap = (r"\textbf{Every $4096^2$ method on the same two prompts.} Evaluation prompts chosen by a fixed seed "
           r"(seed 8), seed 0 for every image. Crops are native 1:1 windows at the highest edge energy averaged over "
           r"the eight methods.")
    return [("all_methods", rec, assets)], [("all_methods", cap, "fig:teaser")], dict(
        rule="random.Random(8).sample(sorted(191 non-development evaluation prompts), 2)", seed=8, draw=ids)


def fig_baselines(out_dir):
    order, src = frozen("gallery_baselines", {"sca": "sca_v2"})
    methods = ["area", "dype", "sega_official", "ultraimage", "sca_v2"]
    rec, assets = run_grid("gallery_baselines", examples(order[:3], src, methods), methods, 45.5, 3, 1, out_dir)
    cap = (r"\textbf{Single-pass \sca{} against the other single-pass baselines at $4096^2$.} Three evaluation "
           r"prompts chosen by a fixed seed (seed 17), seed 0 for every image; UltraImage runs its two-stage defaults. "
           r"Crops are native 1:1 windows at the highest edge energy averaged over the five methods.")
    return [("gallery_baselines", rec, assets)], [("gallery_baselines", cap, "fig:gallery_baselines")], dict(
        rule="frozen v5.1 draw (seed 17)", seed=17, draw=order[:3])


def fig_krea(out_dir):
    order, src = frozen("gallery_krea_complete", {"krea_sca": "krea_sca"})
    methods = ["krea_area", "krea_dype", "krea_sega", "krea_sca"]
    rec, assets = run_grid("gallery_krea_complete", examples(order[:2], src, methods), methods, 63.5, 2, 1, out_dir)
    cap = (r"\textbf{FLUX.1-Krea-dev at $4096^2$.} Two evaluation prompts chosen by a fixed seed (seed 17), seed 0 "
           r"for every image, with a reach table fitted on FLUX.1-Krea-dev. Crops are native 1:1 windows at the highest "
           r"edge energy averaged over the four methods.")
    return [("gallery_krea_complete", rec, assets)], [("gallery_krea_complete", cap, "fig:gallery_krea_complete")], \
        dict(rule="frozen v5.1 draw (seed 17)", seed=17, draw=order[:2])


def fig_big(out_dir, side):
    g = f"gallery_{side}_complete"
    order, src = frozen(g, {"sca": "sca_nb"})
    methods = ["area", "sca_nb"]
    rec, assets = run_grid(g, examples(order[:4], src, methods), methods, 63.5, 2, 2, out_dir)
    cap = (r"\textbf{Exploratory $%d^2$ generation.} AReA and \sca{} without the schedule on four evaluation prompts "
           r"chosen by a fixed seed (seed 17), seed 0 for every image. Crops are native 1:1 windows at the highest edge "
           r"energy averaged over both methods.") % side
    return [(g, rec, assets)], [(g, cap, f"fig:{g}")], dict(rule="frozen v5.1 draw (seed 17)", seed=17,
                                                             draw=order[:4])


def fig_progressive(out_dir):
    man = json.loads((FIG / "progressive" / "provenance.json").read_text())
    ids = [imgsel.short_id(p) for p in man["selection"]["drawn"]]
    key = {"hrdit_dev_author": "hrdit_dev_author", "sca_v2_prog": "sca_v2_prog", "sca_v2": "sca_v2"}
    pool = vc.pool195()
    methods = ["hrdit_dev_author", "sca_v2_prog", "sca_v2"]
    src = {s: {m: pool[key[m]][s]["path"] for m in methods} for s in ids}
    rec, assets = run_grid("progressive", examples(ids, src, methods), methods, 63.5, 2, 1, out_dir)
    cap = (r"\textbf{Progressive \sca{} keeps HRDiT's large subject; single-pass \sca{} adds native detail.} Two "
           r"prompts of the 46-prompt panel chosen by a fixed seed (seed 17), seed 0 for every image. Crops are native "
           r"1:1 windows at the highest edge energy averaged over the three methods.")
    return [("progressive", rec, assets)], [("progressive", cap, "fig:progressive")], dict(
        rule="frozen v5.1 draw random.Random(17).sample(sorted(46 panel prompts), 2)", seed=17, draw=ids)


FIGS = {"gallery_prog16": fig_prog16, "gallery_prog": fig_prog, "zoom4k": fig_zoom, "aspect": fig_aspect,
        "gallery_4096": fig_4096, "gallery_4096x2048": lambda o: fig_wide(o, "2048x4096"),
        "gallery_6144x2048": lambda o: fig_wide(o, "2048x6144"), "gallery_flux2": fig_flux2,
        "ablation": fig_ablation, "showcase_models": fig_showcase_models, "all_methods": fig_all_methods,
        "gallery_baselines": fig_baselines, "gallery_krea_complete": fig_krea,
        "gallery_6144_complete": lambda o: fig_big(o, 6144), "gallery_8192_complete": lambda o: fig_big(o, 8192),
        "progressive": fig_progressive}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    for stem in a.stems or list(FIGS):
        parts, floats, sel = FIGS[stem](a.out_dir)
        for name, rec, assets in parts:
            rec.update(selection=sel, script="paper/figures/src/v6_gallery.py",
                       crop_rule="fixed-size window at the highest mean edge energy averaged over the methods shown")
            (assets / "provenance.json").write_text(json.dumps(rec, indent=1) + "\n")
            print(f"{name}: {rec['pdf_bytes'] / 1e6:.2f} MB, {rec['size_in'][0]:.2f} x {rec['size_in'][1]:.2f} in, "
                  f"crop ppi {sorted({e.get('crop_ppi') for e in rec['examples']})}")
        if a.out_dir is None:
            snippet(stem, floats)
        else:
            (a.out_dir / "captions.json").open("a").write(json.dumps({stem: floats}) + "\n")


if __name__ == "__main__":
    main()
