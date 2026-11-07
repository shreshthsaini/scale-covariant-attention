"""Figure fig:paired_area: progressive SCA and single-pass SCA minus AReA at 4096^2, 14 metrics, 95% paired intervals.

x = 100 * sign * delta / mean_AReA, sign = -1 for the lower-is-better metrics (FID, KID and their patch versions), so
right always means better than AReA. Dividing by a positive constant keeps each interval on the same side of zero,
so the significance calls are exactly those of the paired files. Inception score is left out: it has no paired
interval for progressive SCA. No metric is computed here; every value is read from results/:
  per-image, 2000 prompt draws (scripts/paired_bootstrap.py, seed 0):
    results/prog195_vs_baselines/sca_v2_prog_vs_area.csv (progressive SCA minus AReA)
    results/prog195_pairs/sca_v2_vs_area.csv (single-pass SCA minus AReA)
  set-level, corrected (168 reference identities), 1000 draws: results/prog195_pairs/cfid_vs_area/paired_deltas.csv
    (methods sca_v2_prog and sca_v2, reference area) and its summary.csv (AReA's means);
  AReA's per-image means: results/a4k195_main/{base,extra}_summary.csv (method area).
Run with the sca env python from anywhere: python fig_paired_prog.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = HERE.parents[2]
R = ROOT / "results"
OUT = ROOT / "paper" / "figures" / "paired_area"
W, H = 2.53, 2.55
UNRESOLVED = "#8A939C"
GROUPS = [
    [("ir", "ImageReward"), ("aesthetic", "Aesthetic"), ("clip", "CLIP score"), ("pickscore", "PickScore"),
     ("hpsv2", "HPSv2"), ("hpsv21", "HPSv2.1")],
    [("musiq", "MUSIQ"), ("musiq_crop1024", r"MUSIQ$_\mathsf{c}$"), ("clipiqa", "CLIP-IQA"), ("arniqa", "ARNIQA")],
    [("fid", "FID"), ("kid", "KID"), ("patch_fid", r"FID$_\mathsf{p}$"), ("patch_kid", r"KID$_\mathsf{p}$")],
]
LOWER = {"fid", "kid", "patch_fid", "patch_kid"}
SERIES = [("sca_v2_prog", "progressive", "o", -0.17), ("sca_v2", "single pass", "D", 0.17)]


def read(rel):
    return list(csv.DictReader(open(R / rel)))


def main():
    area = {}
    for rel in ("a4k195_main/base_summary.csv", "a4k195_main/extra_summary.csv"):
        for r in read(rel):
            if r["method"] == "area":
                area[r["metric"]] = float(r["value"])
    for r in read("prog195_pairs/cfid_vs_area/summary.csv"):
        if r["method"] == "area":
            area[r["metric"]] = float(r["value"])  # corrected set-level means
    d = {m: {} for m, *_ in SERIES}
    for m, rel in (("sca_v2_prog", "prog195_vs_baselines/sca_v2_prog_vs_area.csv"),
                   ("sca_v2", "prog195_pairs/sca_v2_vs_area.csv")):
        for r in read(rel):
            assert (r["method"], r["reference"], r["n_prompts"]) == (m, "area", "195"), r
            d[m][r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    for r in read("prog195_pairs/cfid_vs_area/paired_deltas.csv"):
        if r["reference_method"] == "area" and r["method"] in d and r["metric"] in LOWER:
            assert r["bootstrap_valid"] == "1000" and r["n_reference_prompts"] == "168", r
            d[r["method"]][r["metric"]] = (float(r["value"]), float(r["ci_low"]), float(r["ci_high"]))

    sf.use_style()
    fig, ax = plt.subplots(figsize=(W, H))
    y, ticks, labels, counts = 0.0, [], [], {m: [0, 0] for m, *_ in SERIES}
    for g, group in enumerate(GROUPS):
        if g:
            y += 0.55
        for key, label in group:
            sign = -1.0 if key in LOWER else 1.0
            for m, _, marker, off in SERIES:
                v, lo, hi = d[m][key]
                x = [100 * sign * t / area[key] for t in (v, lo, hi)]
                xl, xh = min(x[1], x[2]), max(x[1], x[2])
                win, loss = xl > 0, xh < 0
                colour = sf.SCA if win else (sf.LOSS if loss else UNRESOLVED)
                counts[m][0] += win
                counts[m][1] += loss
                ax.plot([xl, xh], [y + off] * 2, color=colour, lw=1.1, solid_capstyle="butt", zorder=2)
                ax.plot(x[0], y + off, marker=marker, ms=3.6 if marker == "o" else 3.0, zorder=3,
                        mfc=colour if (win or loss) else "white", mec=colour, mew=0.9, ls="none")
            ticks.append(y)
            labels.append(label)
            y += 1.0
    ax.axvline(0, color=sf.INK, lw=0.8, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.4, -0.75)
    ax.set_xlabel("gain over AReA (% of AReA)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}".replace("-", "\u2212") if v else "0"))
    sf.value_grid(ax, "x")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker=mk, ls="none", mfc=sf.SCA, mec=sf.SCA, ms=3.6 if mk == "o" else 3.0,
                      label=name) for _, name, mk, _ in SERIES]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False,
              fontsize=sf.FS_SMALL, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(pad=0.2)
    print(sf.save(fig, str(OUT), formats=("pdf",)))
    for m, name, *_ in SERIES:
        print(f"{name}: significant wins {counts[m][0]}, losses {counts[m][1]} of 14")


if __name__ == "__main__":
    main()
