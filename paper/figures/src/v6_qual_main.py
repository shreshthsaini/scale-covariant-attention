"""Figure 5 of v6 (paper/figures/v6/qual_main.pdf): whole 4096^2 images of AReA, UltraImage, HRDiT and progressive SCA
for a small, a typical and a large ImageReward gain of progressive SCA over AReA; crops are secondary.

Rows (one clause in the caption): the evaluation prompts at the 25th, 50th and 75th percentile (nearest rank: rank
ceil(p n / 100) of the ascending differences, n = 195) of the paired ImageReward difference, progressive SCA minus
AReA, at seed 0. Per-image ImageReward comes from the files the paired bootstrap read (eval/a4k195_main for AReA and
UltraImage, eval/prog195_sca_v2 for HRDiT and progressive SCA; results/prog195_vs_baselines/README.md) and is checked
against eval/prog195_vs_baselines and against the archived mean difference. A row label gives the percentile and the
difference. Fallback (not used; stated in visuals.md): if one of our percentile images showed repeated figures, the
same nearest-rank rule would run within the prompts that name no person.

Columns: AReA, UltraImage, HRDiT, progressive SCA, with a grey line of seconds per 4096^2 image under each head (clean
batch-1 medians from fig_speed.load, UltraImage's campaign mean marked with Table 1's double dagger). Each cell is the
method's full image; that image's own 512 x 512 window of highest edge energy is boxed in red and shown at native
pixels as an inset in the bottom-right corner (another corner, in the order bottom-left, top-right, top-left, when the
inset would cover the box). Under each image: its ImageReward, best of the row bold.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_qual_main.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402

import fig_speed  # noqa: E402
import imgsel  # noqa: E402
import v6_common as vc  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
METHODS = ["area", "ultraimage", "hrdit_dev_author", "sca_v2_prog"]
HEAD = {"area": "AReA", "ultraimage": "UltraImage", "hrdit_dev_author": "HRDiT", "sca_v2_prog": "Progressive SCA"}
SOURCE = {"area": "a4k195_main", "ultraimage": "a4k195_main", "hrdit_dev_author": "prog195_sca_v2",
          "sca_v2_prog": "prog195_sca_v2"}
PCTS = [(25, "25th pct."), (50, "median"), (75, "75th pct.")]
WIN = 512
ARCHIVED_MEAN = 0.399078   # results/prog195_vs_baselines/sca_v2_prog_vs_area.csv, ir delta (checked, not plotted)


def archived_ir() -> dict:
    """{method: {short id: ImageReward}} at seed 0, 4096^2, from the per-image files the paired bootstrap read."""
    out = {}
    for m, run in SOURCE.items():
        with open(vc.EVAL / run / "extra" / "extra_per_image.csv", newline="") as fh:
            out[m] = {imgsel.short_id(r["prompt_id"]): float(r["ir"]) for r in csv.DictReader(fh)
                      if r["method"] == m and r["seed"] == "0" and r["res"] == "4096x4096"}
        assert len(out[m]) == 195, (m, len(out[m]))
    return out


def rows_from(pool, texts, person_free: bool = False):
    """[(percentile, label, short id, difference)] by nearest rank of progressive SCA minus AReA in ImageReward."""
    ir = archived_ir()
    for m in METHODS:
        assert all(abs(ir[m][s] - pool[m][s]["ir"]) < 1e-9 for s in ir[m]), m
    d = {s: ir["sca_v2_prog"][s] - ir["area"][s] for s in ir["area"]}
    assert abs(sum(d.values()) / len(d) - ARCHIVED_MEAN) < 5e-6, sum(d.values()) / len(d)
    ids = [s for s in d if not (person_free and vc.names_person(texts[s]))]
    srt = sorted(ids, key=lambda s: (d[s], s))
    out = []
    for p, lab in PCTS:
        k = math.ceil(p * len(srt) / 100)
        s = srt[k - 1]
        out.append((p, lab, s, d[s], k))
    return out, ir


def _edge(p):
    return vc.detail_map(Image.open(p).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    pool = vc.pool195()
    texts = vc.prompts195(pool)
    rows, ir = rows_from(pool, texts)
    paths = {s: {m: pool[m][s]["path"] for m in METHODS} for _, _, s, _, _ in rows}
    meta = {s: vc.same_prompt_and_noise(paths[s]) for s in paths}          # verifies every sidecar hash
    sha = {s: {m: json.loads(Path(p).with_suffix(".json").read_text())["png_sha256"] for m, p in paths[s].items()}
           for s in paths}
    jobs = [(s, m) for _, _, s, _, _ in rows for m in METHODS]
    with Pool(len(jobs)) as mp:
        maps = dict(zip(jobs, mp.map(_edge, [paths[s][m] for s, m in jobs])))
    wins = {j: vc.best_windows([maps[j]], WIN, WIN, k=1)[0] for j in jobs}  # each image's own window

    d = fig_speed.load()
    r1 = fig_speed.r1
    secs = {"area": "%.1f\\,s" % r1(d["clean"]["area"]), "hrdit_dev_author": "%.1f\\,s" % r1(d["clean"]["hrdit_dev_author"]),
            "sca_v2_prog": "%.1f\\,s" % r1(d["clean"]["sca_v2_prog"]),
            "ultraimage": "%.1f\\,s\\textdaggerdbl" % r1(d["camp"]["ultraimage"])}

    out = a.out_dir or vc.V6
    assets = (a.out_dir or vc.ASSETS) / "qual_main"
    pdf = out / "qual_main.pdf"
    W, G = vc.TEXT_PT, vc.G
    LW = 45.0                                   # row labels: percentile and difference, two horizontal lines
    s_ = (W - LW - G - 3 * G) / 4               # square cells across the rest of the width
    head_h, ir_h, gap = 18.0, 7.5, 2.5
    H = head_h + len(rows) * (s_ + ir_h) + (len(rows) - 1) * gap
    INSET, M = 34.0, 2.5                        # inset side (0.47 in) and margin (pt)
    cv = vc.Canvas(W, H, assets)
    xs = [LW + G + k * (s_ + G) for k in range(4)]
    for k, m in enumerate(METHODS):
        cv.text(xs[k] + s_ / 2, head_h - 10.5, HEAD[m], anchor="base",
                color=vc.SCA if m == "sca_v2_prog" else vc.INK, bold=(m == "sca_v2_prog"))
        cv.text(xs[k] + s_ / 2, head_h - 3.0, secs[m], size=6.5, anchor="base", color=vc.MID)
    rec = dict(rows=[])
    corners = ["br", "bl", "tr", "tl"]
    for i, (p, lab, sid, diff, rank) in enumerate(rows):
        y = head_h + i * (s_ + ir_h + gap)
        cv.text(LW / 2, y + s_ / 2 - 5.0, lab, anchor="base")
        cv.text(LW / 2, y + s_ / 2 + 5.5, "%+.2f" % diff, anchor="base")
        best = max(METHODS, key=lambda m: ir[m][sid])
        cells = {}
        for k, m in enumerate(METHODS):
            x = xs[k]
            im = Image.open(paths[sid][m]).convert("RGB")
            tn, bx = f"{sid}_{m}_full.jpg", wins[(sid, m)]
            vc.save_jpeg(vc.thumb(im, s_, s_), assets / tn, vc.THUMB_Q)
            cv.image(x, y, s_, s_, tn)
            sc = s_ / 4096
            b0, b1 = (x + bx[0] * sc, y + bx[1] * sc), (x + (bx[0] + bx[2]) * sc, y + (bx[1] + bx[3]) * sc)
            for c in corners:                    # the first corner whose inset does not cover the box
                ix = x + M if c[1] == "l" else x + s_ - M - INSET
                iy = y + M if c[0] == "t" else y + s_ - M - INSET
                if not (ix < b1[0] + 1 and b0[0] - 1 < ix + INSET and iy < b1[1] + 1 and b0[1] - 1 < iy + INSET):
                    break
            cn = f"{sid}_{m}_inset_{bx[0]}_{bx[1]}.jpg"
            vc.save_jpeg(im.crop((bx[0], bx[1], bx[0] + WIN, bx[1] + WIN)), assets / cn, vc.CROP_Q)
            cv.box(*b0, *b1, inside=False)
            cv.image(ix, iy, INSET, INSET, cn)
            cv.box(ix, iy, ix + INSET, iy + INSET, inside=True)
            cv.text(x + s_ / 2, y + s_ + ir_h - 1.6, "%.2f" % ir[m][sid], size=6.5, anchor="base", color=vc.MID,
                    bold=(m == best))
            cells[m] = dict(window_px=list(bx), inset_corner=c, ir=ir[m][sid])
        rec["rows"].append(dict(percentile=p, nearest_rank=rank, label=lab, key=sid, ir_difference=round(diff, 4),
                                category=vc.category(texts[sid]), prompt=meta[sid]["prompt"],
                                prompt_id=meta[sid]["prompt_id"], derived_seed=meta[sid]["derived_seed"],
                                best_ir=best, cells=cells, sources=paths[sid], sources_sha256=sha[sid]))
    cv.compile(pdf, "qual_main")
    rec.update(figure="Figure 5 (v6 qual_main, whole images)", pdf=str(pdf), pdf_bytes=pdf.stat().st_size,
               size_in=[round(W / 72.27, 3), round(H / 72.27, 3)], cell_pt=round(s_, 2), inset_pt=INSET,
               assets=str(assets),
               selection=dict(rule=("the evaluation prompts at the 25th, 50th and 75th percentile (nearest rank) of the "
                                    "paired ImageReward difference, progressive SCA minus AReA, over the 195 prompts at "
                                    "seed 0"), person_free_fallback_used=False,
                              ir_files={m: f"eval/{SOURCE[m]}/extra/extra_per_image.csv" for m in METHODS},
                              archived_mean_difference=ARCHIVED_MEAN),
               window_rule="each image's own 512 x 512 window of highest edge energy, shown at native pixels as an inset",
               seconds=dict(clean={m: d["clean"][m] for m in ("area", "hrdit_dev_author", "sca_v2_prog")},
                            campaign_ultraimage=d["camp"]["ultraimage"]))
    prov = (a.out_dir / "qual_main_provenance.json") if a.out_dir else vc.V6 / "qual_main_provenance.json"
    prov.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{pdf}: {pdf.stat().st_size / 1e6:.2f} MB, {W / 72.27:.2f} x {H / 72.27:.2f} in, cells {s_ / 72.27:.2f} in")
    for r in rec["rows"]:
        print(r["label"], r["key"], r["ir_difference"], {m: round(c["ir"], 2) for m, c in r["cells"].items()},
              {m: c["inset_corner"] for m, c in r["cells"].items()})


if __name__ == "__main__":
    main()
