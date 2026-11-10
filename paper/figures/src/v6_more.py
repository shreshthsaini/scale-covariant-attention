"""More appendix sample pages of v6 (round "more visual results"): whole-image comparisons by percentile of the paired
ImageReward gain over AReA, and a mosaic of progressive SCA samples.

  deciles   fig:deciles_a, fig:deciles_b   the 10th, 20th, ..., 90th percentile (nearest rank) of the paired ImageReward
            difference, progressive SCA minus AReA, over the 195 evaluation prompts at seed 0; AReA, UltraImage, HRDiT,
            progressive SCA (all nine deciles, the median also in Figure 5).
  seeds12   fig:seeds12                    the 10th, 25th, 50th, 75th and 90th percentile of the same difference at seed 1
            and at seed 2 (eval/prog195_s1, prog195_s2); AReA and progressive SCA. HRDiT has no seed-1 or seed-2 images.
  hrgen     fig:hrgen_samples              the 10th, 25th, 50th, 75th and 90th percentile of the same difference over the
            HRGen-1K prompts that name no person (eval/prog_hrgen1k, seed 0); AReA, single-pass SCA, progressive SCA.
  mosaic    fig:mosaic                     16 progressive SCA images at 4096^2 of the showcase prompts outside Figure 1:
            categories in the order portrait, animal, landscape, architecture, dense, product, taken in turn, each
            category's ids shuffled with random.Random("1:<category>"); insets on the four images whose highest-edge-
            energy window has the most edge energy.
Per-image ImageReward comes from the files the paired bootstraps read; each set's mean difference is checked against the
archived paired file (results/.../sca_v2_prog_vs_area.csv). Same prompt and initial noise per row (asserted).

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_more.py [deciles seeds12 hrgen mosaic] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402

import fig_speed  # noqa: E402
import imgsel  # noqa: E402
import v6_common as vc  # noqa: E402
import v6_hero  # noqa: E402
import v6_wholegrid as wg  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
RESULTS = vc.REPO / "results"
SNIP = vc.V6 / "snippets"


def per_image(run: str, seed: str, methods: list, res: str = "4096x4096") -> dict:
    """{method: {short id: (ImageReward, png)}} from an evaluation root's extra per-image CSV."""
    out = {m: {} for m in methods}
    with open(vc.EVAL / run / "extra" / "extra_per_image.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["method"] in out and r["seed"] == seed and r["res"] == res:
                out[r["method"]][imgsel.short_id(r["prompt_id"])] = (float(r["ir"]), r["path"])
    return out


def archived_mean(run: str) -> float:
    with open(RESULTS / run / "sca_v2_prog_vs_area.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["metric"] == "ir" and r["method"] == "sca_v2_prog" and r["reference"] == "area":
                return float(r["delta"])
    raise KeyError(run)


def nearest_rank(diff: dict, pcts) -> list:
    srt = sorted(diff, key=lambda s: (diff[s], s))
    out = []
    for p in pcts:
        k = math.ceil(p * len(srt) / 100)
        out.append((p, srt[k - 1], diff[srt[k - 1]], k))
    return out


def plabel(p):
    return "median" if p == 50 else f"{p}th pct."


def secs_line():
    d = fig_speed.load()
    r1 = fig_speed.r1
    return {"area": "%.1f\\,s" % r1(d["clean"]["area"]), "hrdit_dev_author": "%.1f\\,s" % r1(d["clean"]["hrdit_dev_author"]),
            "sca_v2_prog": "%.1f\\,s" % r1(d["clean"]["sca_v2_prog"]), "sca_v2": "%.1f\\,s" % r1(d["clean"]["sca_v2"]),
            "ultraimage": "%.1f\\,s\\textdaggerdbl" % r1(d["camp"]["ultraimage"])}


def write_snippet(name, floats):
    SNIP.mkdir(parents=True, exist_ok=True)
    body = ["% Written by paper/figures/src/v6_more.py; edit the script, not this file."]
    for pdf, cap, lab in floats:
        body += [r"\begin{figure}[p]", r"  \centering", r"  \includegraphics{figures/v6/%s.pdf}" % pdf,
                 r"  \caption{%s}" % cap, r"  \label{%s}" % lab, r"\end{figure}"]
    (SNIP / f"{name}.tex").write_text("\n".join(body) + "\n")


def out_paths(out_dir, stem):
    assets = (out_dir or vc.ASSETS) / stem
    return (out_dir or vc.V6) / f"{stem}.pdf", assets, (out_dir or vc.V6) / f"{stem}_provenance.json"


def check_rows(rows):
    for r in rows:
        vc.same_prompt_and_noise({k: c["path"] for k, c in r["cells"].items()})


# ------------------------------------------------------------------------------------------------ deciles
def fig_deciles(out_dir):
    M = ["area", "ultraimage", "hrdit_dev_author", "sca_v2_prog"]
    src = {"area": "a4k195_main", "ultraimage": "a4k195_main", "hrdit_dev_author": "prog195_sca_v2",
           "sca_v2_prog": "prog195_sca_v2"}
    data = {}
    for m in M:
        data.update({m: per_image(src[m], "0", [m])[m]})
    diff = {s: data["sca_v2_prog"][s][0] - data["area"][s][0] for s in data["area"]}
    assert len(diff) == 195 and abs(sum(diff.values()) / 195 - archived_mean("prog195_vs_baselines")) < 5e-6
    picks = nearest_rank(diff, range(10, 100, 10))
    cols = [("area", "AReA", False), ("ultraimage", "UltraImage", False), ("hrdit_dev_author", "HRDiT", False),
            ("sca_v2_prog", "Progressive SCA", True)]
    secs = secs_line()
    recs, floats = [], []
    for stem, part, lab in (("deciles_a", picks[:5], "fig:deciles_a"), ("deciles_b", picks[5:], "fig:deciles_b")):
        rows = [dict(label=(plabel(p), "%+.2f" % d), meta=dict(percentile=p, nearest_rank=k, key=s, difference=d),
                     cells={m: dict(path=data[m][s][1], value=data[m][s][0]) for m in M}) for p, s, d, k in part]
        check_rows(rows)
        pdf, assets, prov = out_paths(out_dir, stem)
        rec = wg.build(rows, cols, pdf, assets, stem, secs=secs)
        rec.update(selection=dict(rule="nearest-rank deciles of ImageReward(progressive SCA) - ImageReward(AReA), "
                                       "195 evaluation prompts, seed 0", n=195, archived_mean=sum(diff.values()) / 195))
        prov.write_text(json.dumps(rec, indent=1) + "\n")
        recs.append((stem, rec))
    cap = (r"\textbf{Whole-image comparison at every decile of progressive \sca{}'s ImageReward gain over AReA (%s).} "
           r"Rows: the evaluation prompts at the 10th, 20th, \ldots, 90th percentile (nearest rank) of the paired "
           r"ImageReward difference, progressive \sca{} minus AReA, over the 195 prompts at seed~0; the label gives the "
           r"difference. Under each image its ImageReward (best bold); insets show each image's highest-detail "
           r"$256^2$ window at 1:1; times are clean seconds per $4096^2$ image (\textdaggerdbl{} campaign mean).")
    floats = [("deciles_a", cap % r"10th to 50th percentile" + r" Figure~\ref{fig:deciles_b} continues.",
               "fig:deciles_a"),
              ("deciles_b", cap % r"60th to 90th percentile", "fig:deciles_b")]
    if out_dir is None:
        write_snippet("deciles", floats)
    return recs


# ------------------------------------------------------------------------------------------------ seeds 1 and 2
def fig_seeds(out_dir):
    pcts = [10, 25, 50, 75, 90]
    seeds = {}
    for seed, run in (("1", "prog195_s1"), ("2", "prog195_s2")):
        d = per_image(run, seed, ["area", "sca_v2_prog"])
        full = {s: d["sca_v2_prog"][s][0] - d["area"][s][0] for s in d["area"]}
        assert len(full) == 195 and abs(sum(full.values()) / 195 - archived_mean(run)) < 5e-6, (run, sum(full.values()) / 195)
        texts = {s: json.loads(Path(d["sca_v2_prog"][s][1]).with_suffix(".json").read_text())["config"]["prompt"]
                 for s in full}
        # Person-free fallback (lead's rule): the all-prompt draw put our seed-1 median image on a prompt where a second
        # tiny figure repeats the one asked for, so both seeds draw from the prompts that name no person.
        diff = {s: v for s, v in full.items() if not vc.names_person(texts[s])}
        seeds[seed] = (d, nearest_rank(diff, pcts), sum(full.values()) / 195, len(diff))
    cols = [("a1", "AReA", False), ("p1", "Progressive SCA", True), ("a2", "AReA", False), ("p2", "Progressive SCA", True)]
    groups = [("Seed 1", ["a1", "p1"]), ("Seed 2", ["a2", "p2"])]
    sl = secs_line()
    secs = {"a1": sl["area"], "a2": sl["area"], "p1": sl["sca_v2_prog"], "p2": sl["sca_v2_prog"]}
    rows = []
    for i, p in enumerate(pcts):
        d1, pk1, _, _ = seeds["1"]
        d2, pk2, _, _ = seeds["2"]
        (_, s1, df1, k1), (_, s2, df2, k2) = pk1[i], pk2[i]
        rows.append(dict(label=(plabel(p), "%+.2f, %+.2f" % (df1, df2)),
                         meta=dict(percentile=p, seed1=dict(key=s1, difference=df1, nearest_rank=k1),
                                   seed2=dict(key=s2, difference=df2, nearest_rank=k2)),
                         cells={"a1": dict(path=d1["area"][s1][1], value=d1["area"][s1][0]),
                                "p1": dict(path=d1["sca_v2_prog"][s1][1], value=d1["sca_v2_prog"][s1][0]),
                                "a2": dict(path=d2["area"][s2][1], value=d2["area"][s2][0]),
                                "p2": dict(path=d2["sca_v2_prog"][s2][1], value=d2["sca_v2_prog"][s2][0])}))
        vc.same_prompt_and_noise({"a": d1["area"][s1][1], "p": d1["sca_v2_prog"][s1][1]})
        vc.same_prompt_and_noise({"a": d2["area"][s2][1], "p": d2["sca_v2_prog"][s2][1]})
    pdf, assets, prov = out_paths(out_dir, "seeds12")
    rec = wg.build(rows, cols, pdf, assets, "seeds12", groups=groups, secs=secs, label_w=56.0,
                   bold_sets=[["a1", "p1"], ["a2", "p2"]])   # best bold within each seed
    rec.update(selection=dict(rule="per seed, nearest-rank 10th, 25th, 50th, 75th, 90th percentile of ImageReward"
                                   "(progressive SCA) - ImageReward(AReA) over the evaluation prompts that name no person",
                              n_person_free={s: v[3] for s, v in seeds.items()},
                              archived_mean_all={s: v[2] for s, v in seeds.items()},
                              history=("the draw over all 195 prompts put the seed-1 median on m5795767365495583614, "
                                       "whose progressive image has a second tiny figure for the one the prompt asks "
                                       "for; the person-free fallback was then applied to every row of both seeds")),
               note="HRDiT has no seed-1 or seed-2 images on disk; its column is absent")
    prov.write_text(json.dumps(rec, indent=1) + "\n")
    cap = (r"\textbf{Seeds 1 and 2: progressive \sca{} against AReA across the distribution of its ImageReward gain.} "
           r"Per seed, the evaluation prompts at the 10th, 25th, 50th, 75th and 90th percentile (nearest rank) of the "
           r"paired ImageReward difference, progressive \sca{} minus AReA, among the prompts that name no person "
           r"(keyword rule); the label gives the "
           r"difference at seed~1, then seed~2. HRDiT has no images at these seeds. Under each image its ImageReward "
           r"(better of each pair bold); "
           r"insets show each image's highest-detail $256^2$ window at 1:1.")
    if out_dir is None:
        write_snippet("seeds12", [("seeds12", cap, "fig:seeds12")])
    return [("seeds12", rec)]


# ------------------------------------------------------------------------------------------------ HRGen-1K
def fig_hrgen(out_dir):
    pcts = [10, 25, 50, 75, 90]
    d = per_image("prog_hrgen1k", "0", ["area", "sca_v2", "sca_v2_prog"])
    full = {s: d["sca_v2_prog"][s][0] - d["area"][s][0] for s in d["area"]}
    assert len(full) == 1000 and abs(sum(full.values()) / 1000 - archived_mean("prog_hrgen1k")) < 5e-6
    texts = {s: json.loads(Path(d["sca_v2_prog"][s][1]).with_suffix(".json").read_text())["config"]["prompt"]
             for s in full}
    diff = {s: v for s, v in full.items() if not vc.names_person(texts[s])}
    picks = nearest_rank(diff, pcts)
    M = ["area", "sca_v2", "sca_v2_prog"]
    cols = [("area", "AReA", False), ("sca_v2", "Single-pass SCA", True), ("sca_v2_prog", "Progressive SCA", True)]
    rows = [dict(label=(plabel(p), "%+.2f" % df), meta=dict(percentile=p, nearest_rank=k, key=s, difference=df,
                                                             prompt=texts[s]),
                 cells={m: dict(path=d[m][s][1], value=d[m][s][0]) for m in M}) for p, s, df, k in picks]
    check_rows(rows)
    pdf, assets, prov = out_paths(out_dir, "hrgen_samples")
    rec = wg.build(rows, cols, pdf, assets, "hrgen_samples", secs={m: secs_line()[m] for m in M},
                   cell=1.45 * 72.27)
    rec.update(selection=dict(rule="nearest-rank 10th, 25th, 50th, 75th, 90th percentile of ImageReward(progressive "
                                   "SCA) - ImageReward(AReA) over the HRGen-1K prompts that name no person",
                              n_person_free=len(diff), n_all=len(full), archived_mean_all=sum(full.values()) / 1000))
    prov.write_text(json.dumps(rec, indent=1) + "\n")
    cap = (r"\textbf{HRGen-1K: progressive \sca{} and single-pass \sca{} against AReA across the distribution of the "
           r"ImageReward gain.} Rows: the HRGen-1K prompts at the 10th, 25th, 50th, 75th and 90th percentile (nearest "
           r"rank) of the paired ImageReward difference, progressive \sca{} minus AReA, among the %d of 1000 prompts that "
           r"name no person (keyword rule), seed~0; the label gives the difference. Under each image its ImageReward "
           r"(best bold); insets show each image's highest-detail $256^2$ window at 1:1.") % len(diff)
    if out_dir is None:
        write_snippet("hrgen_samples", [("hrgen_samples", cap, "fig:hrgen_samples")])
    return [("hrgen_samples", rec)]


# ------------------------------------------------------------------------------------------------ mosaic
MOSAIC_ORDER = ["portrait", "animal", "landscape", "architecture", "dense", "product"]


MW = 256   # mosaic inset window (native px)


def _detail(p):
    a = vc.detail_map(Image.open(p).convert("RGB"))
    win = vc.best_windows([a], MW, MW, k=1)[0]
    x, y = win[0] // 8, win[1] // 8
    return float(a[y:y + MW // 8, x:x + MW // 8].mean()), win


def fig_mosaic(out_dir, n=16, ncol=4):
    sc = v6_hero.showcase()
    hero = set(v6_hero.draw().values())
    by = {}
    for c in MOSAIC_ORDER:
        ids = sorted(i for i, r in sc.items() if r["category"] == c and i not in hero)
        random.Random(f"1:{c}").shuffle(ids)
        by[c] = ids
    order = []
    while len(order) < n and any(by.values()):
        for c in MOSAIC_ORDER:
            if by[c] and len(order) < n:
                order.append(by[c].pop(0))
    srcs = {s: v6_hero.source(s) for s in order}            # asserts the progressive configuration
    with Pool(len(order)) as mp:
        det = dict(zip(order, mp.map(_detail, [str(srcs[s][0]) for s in order])))
    insets = set(sorted(order, key=lambda s: (-det[s][0], s))[:4])
    G = vc.G
    cell = (vc.TEXT_PT - (ncol - 1) * G) / ncol
    nrow = math.ceil(n / ncol)
    H = nrow * cell + (nrow - 1) * G
    pdf, assets, prov = out_paths(out_dir, "mosaic")
    cv = vc.Canvas(vc.TEXT_PT, H, assets)
    rec = dict(images=[])
    INSET, M = 40.0, 2.5
    for i, s in enumerate(order):
        r_, c_ = divmod(i, ncol)
        x, y = c_ * (cell + G), r_ * (cell + G)
        png = str(srcs[s][0])
        im = Image.open(png).convert("RGB")
        tn = f"{s}_full.jpg"
        vc.save_jpeg(vc.thumb(im, cell, cell, dpi=250), assets / tn, vc.THUMB_Q)
        cv.image(x, y, cell, cell, tn)
        e = dict(id=s, category=sc[s]["category"], source=png, png_sha256=srcs[s][1]["png_sha256"],
                 prompt=sc[s]["prompt"], window_edge_energy=round(det[s][0], 3))
        if s in insets:
            win = det[s][1]
            k = cell / im.width
            b0, b1 = (x + win[0] * k, y + win[1] * k), (x + (win[0] + MW) * k, y + (win[1] + MW) * k)
            for corner in ("br", "bl", "tr", "tl"):
                ix = x + M if corner[1] == "l" else x + cell - M - INSET
                iy = y + M if corner[0] == "t" else y + cell - M - INSET
                if not (ix < b1[0] + 1 and b0[0] - 1 < ix + INSET and iy < b1[1] + 1 and b0[1] - 1 < iy + INSET):
                    break
            cn = f"{s}_inset_{win[0]}_{win[1]}.jpg"
            vc.save_jpeg(im.crop((win[0], win[1], win[0] + MW, win[1] + MW)), assets / cn, 85, subsampling=2)
            cv.box(*b0, *b1, inside=False)
            cv.image(ix, iy, INSET, INSET, cn)
            cv.box(ix, iy, ix + INSET, iy + INSET, inside=True)
            e.update(window_px=list(win), inset_corner=corner)
        rec["images"].append(e)
    cv.compile(pdf, "mosaic")
    rec.update(pdf=str(pdf), pdf_bytes=pdf.stat().st_size, size_in=[round(vc.TEXT_PT / 72.27, 3), round(H / 72.27, 3)],
               selection=dict(rule=("showcase prompts outside Figure 1; categories taken in turn in the order "
                                    + ", ".join(MOSAIC_ORDER) + ", each category shuffled with random.Random('1:<category>')"),
                              order=order, insets=sorted(insets),
                              inset_rule="the four images whose highest-edge-energy 256 x 256 window has the most energy"))
    prov.write_text(json.dumps(rec, indent=1) + "\n")
    cap = (r"\textbf{Progressive \sca{} at $4096^2$ on 16 more showcase prompts.} Prompts outside "
           r"Figure~\ref{fig:prog_hero}, the six categories taken in turn, each category in an order fixed by seed~1; "
           r"insets show 1:1 crops of the four images with the most detailed window.")
    if out_dir is None:
        write_snippet("mosaic", [("mosaic", cap, "fig:mosaic")])
    return [("mosaic", rec)]


FIGS = {"deciles": fig_deciles, "seeds12": fig_seeds, "hrgen": fig_hrgen, "mosaic": fig_mosaic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    for f in a.figs or list(FIGS):
        for stem, rec in FIGS[f](a.out_dir):
            print(f"{stem}: {rec['pdf_bytes'] / 1e6:.2f} MB, {rec['size_in'][0]:.2f} x {rec['size_in'][1]:.2f} in")


if __name__ == "__main__":
    main()
