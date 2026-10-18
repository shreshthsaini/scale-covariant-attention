"""Paired scalar quality differences, reconstructed from archived per-image scores.

Run with the csd-figs uv environment. No generation or scoring is performed.
Whiskers quantify prompt sampling at one generation seed, not seed variability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


import scafig as sf

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "results/evidence/eval"
METHODS = [("full", "Full attention", "final4k_full"),
           ("area", "AReA", "final4k_nb"),
           ("dype", "DyPE", "e1_baselines"),
           ("sega_official", "SEGA", "e1_baselines"),
           ("ultraimage", "UltraImage†", "e1_ultraimage")]
METRICS = [("ir", "ImageReward", 1), ("pickscore", "PickScore", 1),
           ("hpsv21", "HPSv2.1 × 100", 100),
           ("musiq_crop1024", "Crop MUSIQ", 1)]
# Preserve the RNG advancement of scripts/paired_bootstrap.py with all ten metrics.
BOOTSTRAP_ORDER = ["aesthetic", "arniqa", "clip", "clipiqa", "hpsv2", "hpsv21",
                   "ir", "musiq", "musiq_crop1024", "pickscore"]


def load_group(name, sources):
    merged = {}
    for kind, filename in [("base", "per_image.csv"), ("extra", "extra_per_image.csv")]:
        path = EVAL / f"{name}_{kind}" / filename
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        seen = set()
        for row in rows:
            key = row["method"], row["prompt_id"]
            if key in seen:
                raise ValueError(f"duplicate key {key} in {path}")
            seen.add(key)
            if row["res"] != "4096x4096" or row["seed"] != "0":
                raise ValueError(f"unexpected resolution/seed in {path}")
            target = merged.setdefault(key, {"path": row["path"]})
            if target["path"] != row["path"]:
                raise ValueError(f"base/extra image path mismatch for {key}")
            for metric, _, _ in METRICS:
                if row.get(metric, ""):
                    value = float(row[metric])
                    if not np.isfinite(value) or metric in target:
                        raise ValueError(f"nonfinite or repeated metric {metric}: {key}")
                    target[metric] = value
    for key, row in merged.items():
        if any(m not in row for m, _, _ in METRICS):
            raise ValueError(f"incomplete metric coverage for {key}")
    return merged


def paired_statistics():
    sources = {}
    groups = {name: load_group(name, sources) for name in sorted({g for _, _, g in METHODS})}
    canonical = groups["final4k_nb"]
    ids = sorted(p for method, p in canonical if method == "sca_nb")
    if len(ids) != 191:
        raise ValueError(f"expected 191 paired prompts, got {len(ids)}")
    rng = np.random.default_rng(0)
    draw_banks = {metric: rng.integers(0, len(ids), (2000, len(ids)))
                  for metric in BOOTSTRAP_ORDER}
    records = []
    for method, label, group in METHODS:
        data = groups[group]
        archived = None
        if method != "full":
            path = ROOT / f"results/e1_{method}_vs_sca.csv"
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as handle:
                archived = {r["metric"]: r for r in csv.DictReader(handle)}
        if {p for m, p in data if m == method} != set(ids):
            raise ValueError(f"prompt set mismatch for {method}")
        # Every repeated SCA row must refer to the same image and scores.
        for p in ids:
            if ("sca_nb", p) in data:
                ref = data["sca_nb", p]
                main = canonical["sca_nb", p]
                if ref["path"] != main["path"] or any(
                    not np.isclose(ref[m], main[m], rtol=0, atol=1e-5) for m, _, _ in METRICS
                ):
                    raise ValueError(f"SCA scoring/image drift for {group}: {p}")
        for metric, _, scale in METRICS:
            sca = np.array([canonical["sca_nb", p][metric] for p in ids]) * scale
            base = np.array([data[method, p][metric] for p in ids]) * scale
            diff = sca - base
            lo, hi = np.quantile(diff[draw_banks[metric]].mean(axis=1), [.025, .975])
            if archived is not None:
                old = archived[metric]
                if old["method"] != method or old["reference"] != "sca_nb" or int(old["n_prompts"]) != 191:
                    raise ValueError(f"canonical interval protocol mismatch: {method}/{metric}")
                target = [-float(old["delta"]) * scale, -float(old["ci_high"]) * scale,
                          -float(old["ci_low"]) * scale]
                if not np.allclose([diff.mean(), lo, hi], target, rtol=6e-6, atol=1e-6 * scale):
                    raise ValueError(f"canonical interval cannot be reproduced: {method}/{metric}")
                # Keep the paper's canonical interval rounding, after independent reproduction.
                lo, hi = target[1:]
            records.append(dict(baseline=method, baseline_label=label, metric=metric,
                                scale=scale, n_pairs=len(ids), sca_mean=float(sca.mean()),
                                baseline_mean=float(base.mean()), delta=float(diff.mean()),
                                ci_low=float(lo), ci_high=float(hi),
                                protocol="two_stage_30_steps_per_stage" if method == "ultraimage"
                                         else "shared_28_steps_guidance4.5"))
    provenance = dict(sources=sources, difference="SCA minus named baseline", resolution="4096x4096",
                      generation_seed_label=0, prompt_ids=ids, bootstrap_seed=0, bootstrap_draws=2000,
                      bootstrap_unit="matched prompt id", interval="two-sided percentile 95%",
                      multiplicity_adjusted=False, seed_variability_estimated=False,
                      bootstrap_metric_order=BOOTSTRAP_ORDER,
                      canonical_intervals="results/e1_<baseline>_vs_sca.csv, sign reversed; full newly computed",
                      bootstrap_index_sha256={m: hashlib.sha256(d.astype("<i8").tobytes()).hexdigest()
                                              for m, d in draw_banks.items()},
                      caveats=["One generation seed; intervals measure prompt sampling uncertainty only.",
                               "UltraImage uses a two-stage recipe, not matched compute or latent trajectories.",
                               "DyPE and SEGA use FLUX.1-dev, not their preferred Krea checkpoint."])
    return records, provenance


# Display names and label precision for the plot only (the CSV keeps METHODS/METRICS labels).
SHOW = {"ir": ("ImageReward", 2, 2), "pickscore": ("PickScore", 2, 4),  # name, label digits, tick step
        "hpsv21": (r"HPSv2.1 $(\times 100)$", 2, 20), "musiq_crop1024": ("crop MUSIQ", 1, 20)}
ROW = {"full": 4.4, "area": 3.4, "dype": 2.4, "sega_official": 1.4, "ultraimage": 0.0}
NAME = {"full": "Full attention", "area": "AReA", "dype": "DyPE", "sega_official": "SEGA",
        "ultraimage": "UltraImage\u2020"}
UNRESOLVED = "#8A939C"


def plot_deltas(records):
    """One row of four panels sharing the baseline axis; dot = mean paired difference, whisker = 95%."""
    W, H = sf.TEXT, 1.88
    left, gap, right, bottom, top = 0.72, 0.15, 0.03, 0.53, 0.20
    pw = (W - left - right - 3 * gap) / 4
    fig = plt.figure(figsize=(W, H))
    axes = []
    for i in range(4):
        rect = [(left + i * (pw + gap)) / W, bottom / H, pw / W, (H - bottom - top) / H]
        axes.append(fig.add_axes(rect, sharey=axes[0] if axes else None))
    for i, (ax, (metric, _, _)) in enumerate(zip(axes, METRICS)):
        name, digits, step = SHOW[metric]
        rows = [r for r in records if r["metric"] == metric]
        ax.axvline(0, color=sf.INK, lw=0.6, zorder=1)
        ax.axhline(0.7, color=sf.RULE, lw=0.5, zorder=1)
        labels = []
        for r in rows:
            yy, d, lo, hi = ROW[r["baseline"]], r["delta"], r["ci_low"], r["ci_high"]
            col = sf.SCA if lo > 0 else sf.LOSS if hi < 0 else UNRESOLVED
            ax.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt="o", ms=4.2, mew=0.9, mec=col,
                        mfc="white" if lo <= 0 <= hi else col, ecolor=col, elinewidth=1.0,
                        capsize=1.8, capthick=0.8, zorder=3)
            text = f"{d:+.{digits}f}".replace("-", "\u2212")
            ink = sf.MID if col == UNRESOLVED else col  # grey text needs more contrast than the marker
            # Labels sit outside the whisker and never cross the zero line.
            if hi < 0:
                labels.append(ax.annotate(text, (lo, yy), xytext=(-3, 0), textcoords="offset points",
                                          ha="right", va="center", fontsize=sf.FS_SMALL, color=ink))
            else:
                labels.append(ax.annotate(text, (hi, yy), xytext=(3, 0), textcoords="offset points",
                                          ha="left", va="center", fontsize=sf.FS_SMALL, color=ink))
        lo_x = min(0.0, min(r["ci_low"] for r in rows))
        hi_x = max(r["ci_high"] for r in rows)
        pad = 0.04 * (hi_x - lo_x)
        ax.set_xlim(lo_x - pad, hi_x + pad)
        for _ in range(4):  # widen the limits until every value label sits inside the panel
            fig.canvas.draw()
            inv = ax.transData.inverted()
            x0 = min(inv.transform(t.get_window_extent())[0, 0] for t in labels)
            x1 = max(inv.transform(t.get_window_extent())[1, 0] for t in labels)
            cur = ax.get_xlim()
            ax.set_xlim(min(cur[0], x0 - pad), max(cur[1], x1 + pad))
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(rf"$\Delta$ {name}")
        if i == 0:
            ax.set_yticks(list(ROW.values()), [NAME[k] for k in ROW])
            ax.tick_params(axis="y", length=0, pad=3)
            # The method column gets a header in the panel-letter row instead of a rotated label.
            ax.set_ylabel("baseline", rotation=0, ha="left", va="baseline", color=sf.MID,
                          fontsize=sf.FS_TICK)
            ax.yaxis.set_label_coords(-(left - 0.03) / pw, 1 + 0.035 / (H - bottom - top))
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.set_ylim(-0.6, 5.0)
        sf.panel_label(fig, ax, "abcd"[i], dy=0.035 / H)
    x_mid = (left + (W - right)) / 2 / W
    fig.text(x_mid, 0.03 / H, "SCA minus baseline; positive values favor SCA", ha="center",
             va="bottom", fontsize=sf.FS_TICK, color=sf.MID)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "paper/figures/baseline_deltas")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    records, provenance = paired_statistics()
    result = ROOT / "results/baseline_quality_deltas.csv"
    with result.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    sf.use_style()
    plt.rcParams["svg.hashsalt"] = "sca-baseline-deltas"
    fig = plot_deltas(records)
    sf.save(fig, str(args.out))
    if args.preview:
        fig.savefig(args.preview, dpi=150, facecolor="white")
    provenance["statistics_csv"] = str(result.relative_to(ROOT))
    provenance["statistics_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    args.out.with_name(args.out.name + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for row in records:
        print(f'{row["baseline"]:14s} {row["metric"]:18s} {row["delta"]:+.6f} '
              f'[{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}]')


if __name__ == "__main__":
    main()
