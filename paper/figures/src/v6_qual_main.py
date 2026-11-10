"""Figure 5 of v6 (paper/figures/v6/qual_main.pdf): AReA, UltraImage, HRDiT and progressive SCA at 4096^2 on the same
prompts and seeds; each example shows the full image with a red 1 pt box and the native 1:1 crop of that box.

Selection rule (one clause in the caption): one evaluation prompt from each of the portrait, landscape and
architecture categories (keyword rule on the prompt text, v6_common.CATEGORIES), drawn with seed 0
(random.Random("0:<category>").choice) from the 191 Aesthetic-4K evaluation prompts outside the four development
prompts. History: a plain seeded draw of three prompts (random.Random(0).sample) gave three portraits, one of them with
an obvious defect at overview; stratifying by category (rule above) was fixed before any image of its draw was viewed
(.agent/notes/paper_v6/visuals.md).

Crop rule: the window of fixed size with the highest mean edge energy averaged over the four methods (v6_grid).

Images: the PNGs scored by eval/prog195_vs_baselines (seed 0, 4096^2): AReA out/main/area, UltraImage and HRDiT their
baseline runs, progressive SCA out/dupfix_prog/sca_v2_prog; complete sidecars with matching hashes; same prompt and
derived seed per example (asserted).

Layouts: "rows" (methods as rows, prompts as columns: full image and crop side by side) and "cols" (methods as
columns, prompts as rows: crop beneath the full image). The paper uses the one recorded in visuals.md.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_qual_main.py [--layout rows|cols] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgsel  # noqa: E402
import v6_common as vc  # noqa: E402
import v6_grid as vg  # noqa: E402

SEED = 0
CATS = ["portrait", "landscape", "architecture"]
METHODS = ["area", "ultraimage", "hrdit_dev_author", "sca_v2_prog"]
LABELS = {"area": "AReA", "ultraimage": "UltraImage", "hrdit_dev_author": "HRDiT",
          "sca_v2_prog": r"Progressive\\SCA"}
OURS = {"sca_v2_prog": vc.SCA}


def selection():
    pool = vc.pool195()
    texts = vc.prompts195(pool)
    dev = imgsel.dev4_ids()
    picks = vc.draw_by_category(texts, CATS, SEED, exclude=dev)
    return pool, texts, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="rows", choices=["rows", "cols"])
    ap.add_argument("--n", type=int, default=2, help="examples (rows layout): the first n categories of CATS")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    pool, texts, picks = selection()
    order = CATS[:a.n] if a.layout == "rows" else CATS[:2]
    exs = [vg.Example(key=picks[c], paths={m: pool[m][picks[c]]["path"] for m in METHODS}) for c in order]
    if a.out_dir:
        tag = f"{a.layout}{a.n}" if a.layout == "rows" else a.layout
        assets, pdf = a.out_dir / f"qual_main_{tag}", a.out_dir / f"qual_main_{tag}.pdf"
    else:
        assets, pdf = vc.V6 / "qual_main", vc.V6 / "qual_main.pdf"
    W, G, GG, LW = vc.TEXT_PT, vc.G, vc.GG, 19.0
    if a.layout == "rows":
        n = len(exs)
        per = (W - LW - G - (n - 1) * GG) / n          # width of one example: full + G + crop
        if n == 2:                                      # rows 0.88 in tall: 3.6 in for four methods
            full_h = (260.2 - 3 * G) / 4
            crop_w = per - G - full_h
            crop_px = (448, round(448 * full_h / crop_w))
        else:                                           # crops exactly 1.0 in wide
            crop_w = 72.27
            full_h = per - G - crop_w
            crop_px = (256, round(256 * full_h / crop_w))
        vg.windows(exs, crop_px)
        rec = vg.build(exs, METHODS, LABELS, "rows", assets, pdf, "qual_main", full_h=full_h, crop_w=crop_w,
                       label_w=LW, ours=OURS)
    else:
        colw = (W - 3 * G) / 4
        crop_h = 60.0
        crop_px = (320, round(320 * crop_h / colw))
        vg.windows(exs, crop_px)
        rec = vg.build(exs, METHODS, LABELS, "cols", assets, pdf, "qual_main", full_h=colw, crop_w=colw,
                       crop_h=crop_h, ours=OURS)
    rec.update(figure="Figure 5 (v6 qual_main)", layout=a.layout, crop_px=list(crop_px),
               selection=dict(rule=("one evaluation prompt per category (portrait, landscape, architecture; keyword "
                                    "rule v6_common.CATEGORIES), random.Random('0:<category>').choice over the "
                                    "category's sorted ids, development prompts excluded"),
                              seed=SEED, draw={c: picks[c] for c in order},
                              categories={picks[c]: vc.category(texts[picks[c]]) for c in order}),
               crop_rule="fixed-size window at the highest mean edge energy averaged over the four methods",
               per_image={e.key: {m: dict(ir=pool[m][e.key]["ir"], musiq_c=pool[m][e.key]["musiq_c"])
                                  for m in METHODS} for e in exs})
    (assets / "provenance.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{pdf}: {pdf.stat().st_size / 1e6:.2f} MB, {rec['size_in'][0]:.2f} x {rec['size_in'][1]:.2f} in; "
          f"crop {crop_px} px; examples {[e['key'] for e in rec['examples']]}; crop ppi "
          f"{[e['crop_ppi'] for e in rec['examples']]}")


if __name__ == "__main__":
    main()
