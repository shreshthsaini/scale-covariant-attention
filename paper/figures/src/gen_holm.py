"""Holm-adjusted significance over the ten per-image metrics, from the archived per-image evaluation files.

For every paired comparison the paper reports (method minus reference, same prompts and seeds), this module re-runs
the paired bootstrap of scripts/paired_bootstrap.py on the per-image files the archived paired CSV was built from:
prompts resampled with replacement, 2000 draws, one numpy generator seeded 0 and shared by the metrics in sorted
order, exactly as that script does. Every reproduced difference and 95% percentile interval must equal the archived
value under results/ to the six significant digits it stores; a mismatch fails the run. From the same 2000 draws:

  bootstrap p-value   p = min(1, 2 min(1 + #{b <= 0}, 1 + #{b >= 0}) / (B + 1)), two-sided. It is the p-value
                      that inverts the percentile interval: p < 0.05 when the 95% interval excludes zero (up to one
                      draw of interpolation), so unadjusted calls equal the paper's colour convention.
  sign-flip p-value   p = (1 + #{|mean(s * d)| >= |mean(d)|}) / (B + 1) over 2000 random sign vectors s (a second
                      numpy generator seeded 0, metrics in the same order): the paired randomization test, a check.

Holm's step-down procedure then controls the family-wise error rate at 5% over the ten per-image metrics of each pair
(adjusted p_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)); significant when the adjusted p is at most 0.05). The
direction is the sign of the difference. Set-level metrics (FID, KID and their patch versions) have no per-image
values and are not part of the family.

gen_main_tables.py imports this module: it writes app_tables/holm.tex (Table tab:holm) and adds the \\holm... macros
to headline.tex. Run alone for a report: <sca env python> paper/figures/src/gen_holm.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "results"
EVAL = Path("/scratch/09032/saini_2/sca/eval")
DRAWS, SEED, ALPHA = 2000, 0, 0.05
SKIP = {"method", "res", "prompt_id", "seed", "path", "seconds"}  # as scripts/paired_bootstrap.py
# The ten per-image metrics, in the paper's reading order (preference, then image quality).
METRICS = [("ir", "ImageReward"), ("aesthetic", "Aesthetic"), ("clip", "CLIP score"), ("pickscore", "PickScore"),
           ("hpsv2", "HPSv2"), ("hpsv21", "HPSv2.1"), ("musiq", "MUSIQ"), ("musiq_crop1024", r"MUSIQ$_\text{c}$"),
           ("clipiqa", "CLIP-IQA"), ("arniqa", "ARNIQA")]
TEN = {k for k, _ in METRICS}


def _files(d):
    """The two per-image files of an evaluation root: eval.py's base/per_image.csv and eval_extra.py's extra file."""
    d = EVAL / d
    found = [d / f for f in ("base/per_image.csv", "extra/extra_per_image.csv", "per_image.csv",
                             "extra_per_image.csv") if (d / f).exists()]
    assert found, d
    return found


# (key, group, row label, method, its evaluation roots, reference, its roots, archived paired CSV under results/)
PAIRS = [
    ("prog_area", "head", r"Progressive \sca{} $-$ AReA", "sca_v2_prog", ["prog195_sca_v2"], "area",
     ["a4k195_main"], "prog195_vs_baselines/sca_v2_prog_vs_area.csv"),
    ("prog_hrdit", "head", r"Progressive \sca{} $-$ HRDiT", "sca_v2_prog", ["prog195_sca_v2"], "hrdit_dev_author",
     ["prog195_sca_v2"], "prog195_vs_baselines/sca_v2_prog_vs_hrdit_dev_author.csv"),
    ("sca_area", "head", r"Single-pass \sca{} $-$ AReA", "sca_v2", ["a4k195_main_sca_v2"], "area",
     ["a4k195_main_sca_v2"], "prog195_pairs/sca_v2_vs_area.csv"),
    ("prog_ultra", "t1", r"Progressive $-$ UltraImage", "sca_v2_prog", ["prog195_sca_v2"], "ultraimage",
     ["a4k195_main"], "prog195_vs_baselines/sca_v2_prog_vs_ultraimage.csv"),
    ("prog_dype", "t1", r"Progressive $-$ DyPE", "sca_v2_prog", ["prog195_sca_v2"], "dype",
     ["a4k195_main"], "prog195_vs_baselines/sca_v2_prog_vs_dype.csv"),
    ("prog_sega", "t1", r"Progressive $-$ SEGA", "sca_v2_prog", ["prog195_sca_v2"], "sega_official",
     ["a4k195_main"], "prog195_vs_baselines/sca_v2_prog_vs_sega_official.csv"),
    ("prog_full", "t1", r"Progressive $-$ full attention", "sca_v2_prog", ["prog195_sca_v2"], "full",
     ["a4k195_main"], "prog195_vs_baselines/sca_v2_prog_vs_full.csv"),
    ("prog_sca", "t1", r"Progressive $-$ single-pass \sca{}", "sca_v2_prog", ["prog195_sca_v2"], "sca_v2",
     ["a4k195_main_sca_v2"], "prog195_vs_baselines/sca_v2_prog_vs_sca_v2.csv"),
    ("prog_area_s1", "gen", r"Progressive $-$ AReA, seed 1", "sca_v2_prog", ["prog195_s1"], "area",
     ["prog195_s1"], "prog195_s1/sca_v2_prog_vs_area.csv"),
    ("prog_area_s2", "gen", r"Progressive $-$ AReA, seed 2", "sca_v2_prog", ["prog195_s2"], "area",
     ["prog195_s2"], "prog195_s2/sca_v2_prog_vs_area.csv"),
    ("prog_area_hrgen", "gen", r"Progressive $-$ AReA, HRGen-1K", "sca_v2_prog", ["prog_hrgen1k"], "area",
     ["prog_hrgen1k"], "prog_hrgen1k/sca_v2_prog_vs_area.csv"),
    ("sca_area_s1", "gen", r"Single-pass $-$ AReA, seed 1", "sca_v2", ["a4k195_seed1_sca_v2"], "area",
     ["a4k195_seed1_sca_v2"], "a4k195_seed1_sca_v2/sca_v2_vs_area.csv"),
    ("sca_area_s2", "gen", r"Single-pass $-$ AReA, seed 2", "sca_v2", ["a4k195_seed2_sca_v2"], "area",
     ["a4k195_seed2_sca_v2"], "a4k195_seed2_sca_v2/sca_v2_vs_area.csv"),
    ("sca_area_hrgen", "gen", r"Single-pass $-$ AReA, HRGen-1K", "sca_v2", ["hrgen1k_dev_sca_v2"], "area",
     ["hrgen1k_dev_sca_v2"], "hrgen1k_dev_sca_v2/sca_v2_vs_area.csv"),
    ("sca_area_krea", "gen", r"Single-pass $-$ AReA, FLUX.1-Krea", "sca_v2_krea", ["krea195_sca_v2"], "area",
     ["krea195_sca_v2"], "krea195_sca_v2/sca_v2_krea_vs_area.csv"),
    ("flux2", "gen", r"w/o schedule $-$ full attn., FLUX.2-dev", "flux2_sca", ["flux2_4k_base", "flux2_4k_extra"],
     "flux2_full", ["flux2_4k_base", "flux2_4k_extra"], "e8_flux2_sca_vs_full.csv"),
]
# Ablations (Table 2): each variant minus its per-head reference; 46 prompts without the schedule, 195 progressive.
ABL_SINGLE = [("abl_uniform_nb", r"shared $\pi(\sigma)$"), ("abl_layer_nb", r"per-layer $\pi_\ell(\sigma)$"),
              ("abl_v0", "no opening (v0)"), ("abl_localonly_nb", "local only"), ("abl_globalonly", "global only"),
              ("abl_mean_nb", "mean pooling")]
ABL_PROG = [("prog_headmean", r"shared $\pi(\sigma)$"), ("prog_layermean", r"per-layer $\pi_\ell(\sigma)$"),
            ("prog_win", "window only")]
PAIRS += [(m, "abl", rf"{lab} $-$ per-head, single pass", m, ["ablnb_base", "ablnb_extra"], "sca_nb",
           ["ablnb_base", "ablnb_extra"], f"ablnb_{m}_deltas.csv") for m, lab in ABL_SINGLE]
PAIRS += [(m, "ablprog", rf"{lab} $-$ per-head, progressive", m, ["prog195_ablation"], "sca_v2_prog",
           ["prog195_ablation"], f"prog195_ablation/{m}_vs_sca_v2_prog.csv") for m, lab in ABL_PROG]


def _load(roots, method):
    """{(res, seed): {metric: {prompt_id: value}}} of one method, read as scripts/paired_bootstrap.py reads."""
    out = {}
    for root in roots:
        for f in _files(root):
            for r in csv.DictReader(open(f)):
                if r["method"] != method:
                    continue
                for k, v in r.items():
                    if k in SKIP or k.startswith("eval_seconds") or v in ("", None):
                        continue
                    try:
                        value = float(v)
                    except ValueError:
                        continue
                    if not math.isfinite(value):
                        raise ValueError(f"nonfinite {k} in {f}: {r['prompt_id']}")
                    dest = out.setdefault((r.get("res", ""), r.get("seed", "")), {}).setdefault(k, {})
                    if r["prompt_id"] in dest:
                        raise ValueError(f"duplicate {k} for {method}/{r['prompt_id']} in {roots}")
                    dest[r["prompt_id"]] = value
    return out


def _g6(x):
    return float(f"{x:.6g}")


def _holm(p):
    """Holm-adjusted p-values, same order as p."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def pair(spec):
    """Reproduce one archived paired CSV and add the p-values and Holm calls.

    Returns {"key", "label", "n", "src", "metrics": {metric: {delta, lo, hi, p_boot, p_flip, holm_boot,
    holm_flip, sig_raw, sig_holm, sig_flip}}, "up", "dn", "up_raw", "dn_raw", "up_flip", "dn_flip"}; sig_* are
    +1 (method significantly better), -1 (worse) or 0. Every metric of the ten is higher-is-better."""
    key, group, label, method, mroots, ref, rroots, archive = spec
    X, Y = _load(mroots, method), _load(rroots, ref)
    assert set(X) == set(Y) and len(X) == 1, (key, set(X), set(Y))
    grp = next(iter(X))
    x, y = X[grp], Y[grp]
    rng, flip = np.random.default_rng(SEED), np.random.default_rng(SEED)
    got, n_all = {}, set()
    for m in sorted(set(x) | set(y)):  # paired_bootstrap.py: metrics in sorted order share one generator
        a, b = x.get(m, {}), y.get(m, {})
        if set(a) != set(b):
            raise ValueError(f"{key}: unpaired {m}")
        ids = sorted(a)
        if len(ids) < 2:
            continue
        d = np.array([a[i] - b[i] for i in ids])
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(DRAWS)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        t = abs(d.mean())
        flips = np.array([(flip.choice([-1.0, 1.0], len(d)) * d).mean() for _ in range(DRAWS)])
        p_boot = min(1.0, 2 * min(1 + int((boot <= 0).sum()), 1 + int((boot >= 0).sum())) / (DRAWS + 1))
        p_flip = (1 + int((np.abs(flips) >= t - 1e-15).sum())) / (DRAWS + 1)
        got[m] = dict(delta=float(d.mean()), lo=float(lo), hi=float(hi), p_boot=p_boot, p_flip=p_flip)
        n_all.add(len(ids))
    assert TEN <= set(got), (key, sorted(got))
    # the archived paired CSV must be reproduced to the digits it stores
    arch = {r["metric"]: r for r in csv.DictReader(open(RES / archive)) if r["method"] == method and
            r["reference"] == ref}
    for m in TEN:
        r = arch[m]
        for f in ("delta", "lo", "hi"):
            col = {"lo": "ci_low", "hi": "ci_high"}.get(f, f)
            if _g6(got[m][f]) != float(r[col]):
                raise AssertionError(f"{key} {m} {f}: reproduced {got[m][f]:.6g}, archived {r[col]} ({archive})")
        assert int(r["n_prompts"]) in n_all, (key, m, r["n_prompts"], n_all)
    ten = [k for k, _ in METRICS]
    hb = _holm([got[m]["p_boot"] for m in ten])
    hf = _holm([got[m]["p_flip"] for m in ten])
    out = {}
    for m, adj_b, adj_f in zip(ten, hb, hf):
        g = got[m]
        s = 1 if g["delta"] > 0 else -1
        g.update(holm_boot=adj_b, holm_flip=adj_f,
                 sig_raw=s if (g["lo"] > 0 or g["hi"] < 0) else 0,
                 sig_holm=s if adj_b <= ALPHA else 0, sig_flip=s if adj_f <= ALPHA else 0)
        # the unadjusted bootstrap p and the interval agree (up to one draw of interpolation)
        if (g["p_boot"] < ALPHA) != bool(g["sig_raw"]) and abs(g["p_boot"] - ALPHA) > 2.0 / (DRAWS + 1):
            raise AssertionError(f"{key} {m}: p {g['p_boot']:.4f} disagrees with the interval [{g['lo']}, {g['hi']}]")
        out[m] = g
    n = n_all.pop() if len(n_all) == 1 else max(n_all)
    res = dict(key=key, group=group, label=label, n=n, src=f"results/{archive}", metrics=out)
    for tag, f in (("", "sig_holm"), ("_raw", "sig_raw"), ("_flip", "sig_flip")):
        res["up" + tag] = sum(out[m][f] > 0 for m in ten)
        res["dn" + tag] = sum(out[m][f] < 0 for m in ten)
    return res


def run():
    return [pair(s) for s in PAIRS]


if __name__ == "__main__":
    for r in run():
        print(f"{r['key']:16s} n={r['n']:5d}  unadjusted +{r['up_raw']}/-{r['dn_raw']}  Holm (bootstrap) "
              f"+{r['up']}/-{r['dn']}  Holm (sign-flip) +{r['up_flip']}/-{r['dn_flip']}")
        if r["group"] == "head" or r["up"] != r["up_raw"] or r["dn"] != r["dn_raw"]:
            for m, name in METRICS:
                g = r["metrics"][m]
                print(f"    {m:15s} {g['delta']:+.4g} [{g['lo']:.4g}, {g['hi']:.4g}]  p_boot {g['p_boot']:.4f} "
                      f"holm {g['holm_boot']:.4f}  p_flip {g['p_flip']:.4f} holm {g['holm_flip']:.4f}  "
                      f"raw {g['sig_raw']:+d} holm {g['sig_holm']:+d} flip {g['sig_flip']:+d}")
