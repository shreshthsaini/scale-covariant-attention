#!/usr/bin/env python3
"""Group module of gen_app_tables.py: the appendix tables of app:phase2 and app:budget that move to sca_v2.

tables() returns one dict per table in gen_app_tables.py's format:
  krea_6144         float    tab:krea_6144         replaces \\input{figures/snippets/krea_6144.tex}
  phase2_baselines  tabular  tab:phase2_baselines
  phase2_general    tabular  tab:phase2_general
  panel46_deltas    tabular  tab:panel46_deltas
  prog_deltas       float    tab:prog_deltas       replaces \\input{figures/snippets/prog_deltas.tex}

Sources, published (pending) and final (landed). Per-image intervals are scripts/paired_bootstrap.py files (2000
prompt draws, seed 0) unless stated; "derived" means v2swap.derived_pairs, which runs the same script on the per-image
CSVs of two eval runs after this module checks that both methods cover the same prompt ids, resolution and seed label.
  krea_6144         6144^2 column: results/e2_6144, sca_nb minus area (100 prompts); never switches, "w/o sched.".
                    FLUX.1-Krea-dev SCA column: results/krea195 (gen_late_tables.load_deltas: eval.py intervals,
                    1000 draws, set-level from corrected_fidp_ci) -> krea195_sca_v2, sca_v2_krea minus area from
                    v2swap.load (eval.py intervals of the same run, 1000 draws, negated; FID and KID keep eval.py's
                    prompt-only interval, the patch versions none, since the z98x runs have no corrected_fidp_ci).
                    DyPE and SEGA (results/e9_krea, minus area) never involve SCA and stay.
  phase2_baselines  baseline minus sca_nb: DyPE, SEGA, UltraImage from results/a4k195_main, HRDiT from
                    results/a4k195_panel (195 prompts; not results/hrdit191, which has 191) -> derived baseline
                    (same published run) minus sca_v2 of a4k195_main_sca_v2.
  phase2_general    sca_nb minus area of hrgen1k_dev, a4k195_seed1, a4k195_seed2 -> sca_v2 minus area of the
                    matching sca_v2 run (within-run file); FLUX.2-dev (results/e8_flux2_sca_vs_full.csv) never
                    switches, "w/o sched.".
  panel46_deltas    per-image rows: baseline minus sca_nb (a4k195_main: full, dype, sega_official, area;
                    a4k195_panel: sigma_dev_matched, ultraimage, hrdit_dev_author) -> derived baseline minus sca_v2
                    of a4k195_main_sca_v2, except AReA: the negated within-run sca_v2_vs_area file of that run.
                    FID, KID, patch versions (corrected_fidp_ci cluster bootstrap) and time (eval.py) never switch:
                    set-level intervals across runs do not exist; a group row marks them "w/o schedule".
  prog_deltas       progressive vs. HRDiT: a4k195_panel sca_prog minus hrdit_dev_author -> prog195_sca_v2
                    sca_v2_prog minus hrdit_dev_author (within-run file). Progressive vs. SCA (single pass, 28
                    steps): sca_prog minus sca_nb -> derived sca_v2_prog (prog195_sca_v2) minus sca_v2
                    (a4k195_main_sca_v2), so the column keeps its meaning (progressive against single pass; the run's
                    sca_v2_prog_vs_sca_prog file is the schedule ablation inside progressive, a different question).
                    The 42-step columns never switch; a header row marks them "w/o schedule".
Not generated, because no cell can ever switch honestly:
  tab:phase2_distribution  FID, KID and patch versions of DyPE, SEGA, UltraImage minus SCA with cluster-bootstrap
                           intervals; the sca_v2 runs have no corrected_fidp_ci and these baselines are in another
                           run, so only interval-free cross-run points would exist. Its caption needs "without the
                           schedule" when the paper switches.
  tab:phase2_sensitivity   reach-emulation budget and gate variants exist only for published SCA (46 prompts).
  tab:phase2_budget        56-step and guidance-7 runs exist only for published SCA.
Warnings are computed only for columns whose run has landed.

Paper v6 layout (2026-11-10; numbers, intervals and colours unchanged): 7 pt, the shared macros of
main_tables/tabstyle.tex, no \\resizebox. krea_6144, phase2_baselines, phase2_general and panel46_deltas put the
comparisons in rows (ours shaded) and stack one panel per metric family, each estimate above its grey interval;
prog_deltas keeps one row per metric with each difference beside its interval. The float captions (krea_6144,
prog_deltas) follow the v6 order: lead, what they expand, setup, marks, one key-result sentence computed from the
cells. The tabular captions live in appendix.tex; the checks here cover the claims proposed for them.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data195  # noqa: E402
import gen_late_tables as late  # noqa: E402  (import has no side effects; only helpers and constants are used)
import gen_prog_tables as prog  # noqa: E402  (same)
import v2swap  # noqa: E402

MAIN_V2 = "a4k195_main_sca_v2"
WOS = "without the schedule"  # group rows have room for the full name (render review 2026-11-07)
WOS_SHORT = "w/o sched."

# (csv metric, row label, scale, decimals) for the 4096^2 delta tables typed after gen_tables.py's MET
PER_IMAGE = [("ir", "ImageReward", 1, 3), ("aesthetic", "Aesthetic", 1, 2), ("clip", "CLIP score", 1, 2),
             ("pickscore", "PickScore", 1, 2), ("hpsv2", r"HPSv2 ($\times100$)", 100, 2),
             ("hpsv21", r"HPSv2.1 ($\times100$)", 100, 2), ("musiq", "MUSIQ", 1, 1),
             ("musiq_crop1024", r"MUSIQ$_\text{c}$", 1, 1), ("clipiqa", r"CLIP-IQA ($\times100$)", 100, 1),
             ("arniqa", r"ARNIQA ($\times100$)", 100, 1)]
PREFERENCE = ("ir", "aesthetic", "pickscore", "hpsv2", "hpsv21")
# the stacked panels of the v6 tables with comparisons as rows: one per metric family
PANELS = [("(a) Preference", ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21"]),
          ("(b) Image quality", ["musiq", "musiq_crop1024", "clipiqa", "arniqa"])]
HEADS = {k: (k, label, scale, d, True) for k, label, scale, d in PER_IMAGE}  # column heads keep their x100 units
assert sorted(k for _, keys in PANELS for k in keys) == sorted(HEADS)
NAMES = {"ir": "ImageReward", "aesthetic": "Aesthetic", "clip": "CLIP score", "pickscore": "PickScore",
         "hpsv2": "HPSv2", "hpsv21": "HPSv2.1", "musiq": "MUSIQ", "musiq_crop": r"MUSIQ$_\text{c}$",
         "musiq_crop1024": r"MUSIQ$_\text{c}$", "clipiqa": "CLIP-IQA", "arniqa": "ARNIQA", "fid": "FID",
         "kid": "KID", "patch_fid": "patch FID", "patch_kid": "patch KID", "is": "Inception score",
         "seconds": "time"}
WORDS = "no one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split()


# ---------------------------------------------------------------- formatting ----------------------------------------
# Paper v6 table style (2026-11-10): the shared macros of main_tables/tabstyle.tex (written by gen_main_tables.py).
# \scaFitTab{colspec}{n}{rows} sets a booktabs tabular at \linewidth; \scaRowOurs shades our rows; colour scaCtx
# greys intervals. Every table is \scriptsize (7 pt) without \resizebox. Tables with many comparisons put the
# comparisons in rows and stack one panel per metric family, each estimate above its interval (as gen_main_tables.py's
# app_tables/gen_all.tex and prog_all.tex); tables with four comparisons keep one row per metric and set each estimate
# and its interval side by side in two columns 3 pt apart (PAIR).
STYLE = r"\ifdefined\scaFitTab\else\input{figures/snippets/main_tables/tabstyle.tex}\fi"
OURS = r"\scaRowOurs "
PAIR = r"r@{\hspace{3pt}}l"
PAIR_GAP = r"r@{\hspace{3pt}}l<{\hspace{6pt}}"  # a pair that ends a column group: 6 pt more room after it


def signed(x, d):
    """gen_prog_tables.signed: explicit sign, LaTeX {-} for minus, a rounded zero keeps the sign of x."""
    return prog.signed(x, d)


def bound(x, d):
    """An interval bound as the v6 tables print it: signed() without the plus sign (same digits)."""
    s = signed(x, d)
    return s[1:] if s.startswith("+") else s


def cell_parts(v, scale, d, higher=True):
    """(estimate, interval) of a delta: the estimate coloured \\up / \\dn when its 95% interval excludes zero in
    favour of / against the method, the interval in grey. No interval (lo or hi None): the point alone, unmarked,
    and an empty interval."""
    delta, lo, hi = v
    if lo is None or hi is None:
        return f"${signed(delta * scale, d)}$", ""
    delta, lo, hi = delta * scale, lo * scale, hi * scale
    body = f"${signed(delta, d)}$"
    good, bad = (lo > 0, hi < 0) if higher else (hi < 0, lo > 0)
    if good:
        body = r"\up{" + body + "}"
    elif bad:
        body = r"\dn{" + body + "}"
    return body, r"{\color{scaCtx}$[" + bound(lo, d) + ", " + bound(hi, d) + r"]$}"


def panel(title, heads, groups, label_head, shade=()):
    """One stacked panel of a table with comparisons as rows: an italic title, then one column per metric, each
    estimate with its grey interval on the line below (gen_all.tex's layout). heads: [(key, head, scale, decimals,
    higher)]; groups: [(group head or None, [(row label, {key: (delta, lo, hi)}, pending)])]."""
    n = len(heads) + 1
    out = [rf"\textit{{{title}}}\par\smallskip", rf"\scaFitTab{{l*{{{len(heads)}}}{{r}}}}{{{n}}}{{%", r"  \toprule",
           "  " + label_head + " & " + " & ".join(h[1] for h in heads) + r" \\", r"  \midrule"]
    for i, (head, rows) in enumerate(groups):
        if i:
            out.append(r"  \midrule")
        if head:
            out.append(rf"  \multicolumn{{{n}}}{{l}}{{\emph{{{head}}}}} \\")
        for label, vals, pend in rows:
            est, low = [], []
            for key, _, scale, d, higher in heads:
                e, iv = cell_parts(vals[key], scale, d, higher)
                est.append(v2swap.pend(e, pend))
                low.append(v2swap.pend(iv, pend))
            pre = OURS if label in shade else ""
            out += ["  " + pre + label + " & " + " & ".join(est) + r" \\",
                    "  " + pre + " & " + " & ".join(low) + r" \\[1pt]"]
    return out + [r"  \bottomrule", "}"]


def stack(*panels):
    """Panels one above the other, \\medskip apart."""
    out = []
    for i, p in enumerate(panels):
        out += ([r"\par\medskip"] if i else []) + p
    return out


# Metric families for the tables with one row per metric (krea_6144, prog_deltas); the arrow of a family head holds
# for its rows, the set-level rows carry their own.
ROW_FAMILIES = [(r"Preference $\uparrow$", ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21"]),
                (r"Image quality $\uparrow$", ["musiq", "musiq_crop", "musiq_crop1024", "clipiqa", "arniqa"]),
                ("Set level and time", ["fid", "kid", "patch_fid", "patch_kid", "is", "seconds"])]
ROW_LABEL = {"ir": "ImageReward", "aesthetic": "Aesthetic", "clip": "CLIP score", "pickscore": "PickScore",
             "hpsv2": r"HPSv2 ($\times100$)", "hpsv21": r"HPSv2.1 ($\times100$)", "musiq": "MUSIQ",
             "musiq_crop": r"MUSIQ$_\text{c}$", "musiq_crop1024": r"MUSIQ$_\text{c}$",
             "clipiqa": r"CLIP-IQA ($\times100$)", "arniqa": r"ARNIQA ($\times100$)", "fid": r"FID $\downarrow$",
             "kid": r"KID ($\times\smash{10^3}$) $\downarrow$", "patch_fid": r"FID$_\text{p}$ $\downarrow$",
             "patch_kid": r"KID$_\text{p}$ ($\times\smash{10^3}$) $\downarrow$", "is": r"Inception score $\uparrow$",
             "seconds": r"Time (s) $\downarrow$"}


def metric_rows(keys, ncols, cells_of):
    """Body of a table with one row per metric, in the order of `keys`, under an italic head per family with
    \\midrule between families; cells_of(key) gives the row's cells after its label."""
    out, first = [], True
    for head, fam in ROW_FAMILIES:
        rows = [k for k in keys if k in fam]
        if not rows:
            continue
        if not first:
            out.append(r"\midrule")
        first = False
        out.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{head}}}}} \\")
        out += [ROW_LABEL[k] + " & " + " & ".join(cells_of(k)) + r" \\" for k in rows]
    if sorted(k for _, fam in ROW_FAMILIES for k in fam if k in keys) != sorted(keys):
        raise ValueError(f"metric_rows: keys outside the families: {keys}")
    return out


def sig(v, higher=True):
    """+1 significantly favourable, -1 significantly unfavourable, 0 interval spans zero, None no interval."""
    _, lo, hi = v
    if lo is None or hi is None:
        return None
    if (lo > 0) if higher else (hi < 0):
        return 1
    if (hi < 0) if higher else (lo > 0):
        return -1
    return 0


def neg(v):
    """a minus b from b minus a under the same prompt draws: the percentile interval negates exactly."""
    d, lo, hi = v
    return (-d, None if hi is None else -hi, None if lo is None else -lo)


def join(items):
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def label_list(keys):
    return join(NAMES[k] for k in keys)


# ---------------------------------------------------------------- inputs --------------------------------------------
def _rows(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def pair_file(run, a, b, n=None, seed=None, res="4096x4096"):
    """results/<run>/<a>_vs_<b>.csv (published or sca_v2 run) as {metric: (delta, lo, hi)}, checking its prompt count,
    resolution and seed label."""
    root = v2swap.V2_RESULTS if run in v2swap.RUNS else v2swap.RESULTS
    path = root / run / f"{a}_vs_{b}.csv"
    rows = _rows(path)
    if not rows:
        raise FileNotFoundError(path)
    for r in rows:
        got = (r["method"], r["reference"], r["n_prompts"], r.get("res"), r.get("seed"))
        want = (a, b, str(n) if n is not None else r["n_prompts"], res, str(seed) if seed is not None else r.get("seed"))
        if got != want:
            raise ValueError(f"{path}: row {got}, expected {want}")
    out = v2swap.pairs(run, a, b)
    missing = [k for k, *_ in PER_IMAGE if k not in out]
    if missing:
        raise ValueError(f"{path}: no {missing}")
    return out


def check_paired(run_a, a, run_b, b):
    """Both methods must cover the same (prompt id, resolution, seed label) set in their runs' per-image CSVs."""
    sets = []
    for run, m in ((run_a, a), (run_b, b)):
        keys = set()
        for p in v2swap._per_image_files(run):
            keys |= {(r["prompt_id"], r.get("res", ""), r.get("seed", "")) for r in csv.DictReader(open(p))
                     if r["method"] == m}
        if not keys:
            raise FileNotFoundError(f"no per-image rows for {m} in {run}")
        sets.append(keys)
    if sets[0] != sets[1]:
        raise ValueError(f"{run_a}/{a} and {run_b}/{b} are not paired: {len(sets[0] - sets[1])} keys only in the "
                         f"first, {len(sets[1] - sets[0])} only in the second")
    return len({k[0] for k in sets[0]}), sorted({k[1:] for k in sets[0]})


def derived(run_a, a, run_b, b):
    """a (run_a) minus b (run_b), per-image metrics, via v2swap.derived_pairs after the pairing check."""
    n, groups = check_paired(run_a, a, run_b, b)
    if n != 195 or groups != [("4096x4096", "0")]:
        raise ValueError(f"{run_a}/{a} vs {run_b}/{b}: {n} prompts in {groups}, expected 195 at 4096x4096, seed 0")
    out = v2swap.derived_pairs(run_a, a, run_b, b)
    missing = [k for k, *_ in PER_IMAGE if k not in out]
    if missing:
        raise ValueError(f"derived {run_a}/{a} vs {run_b}/{b}: no {missing}")
    return out


def header(sources):
    return ("% Generated by src/gen_app_tables.py (module app_tab_phase2.py) from " + ", ".join(sources)
            + "; do not edit by hand.")


def pair_cells(cols, key, scale, d, pend_flags, higher=True):
    """The (estimate, interval) cells of one metric row across comparison columns; pend_flags[i] wraps column i in
    \\pend."""
    out = []
    for c, p in zip(cols, pend_flags):
        e, iv = cell_parts(c[key], scale, d, higher)
        out += [v2swap.pend(e, p), v2swap.pend(iv, p)]
    return out


def sig_counts(v, spec):
    """(better, worse) metric keys of one comparison {key: (delta, lo, hi)} over spec [(key, higher)]."""
    return ([k for k, h in spec if sig(v[k], h) == 1], [k for k, h in spec if sig(v[k], h) == -1])


# ---------------------------------------------------------------- krea_6144 -----------------------------------------
KREA_RUN = "krea195_sca_v2"
# column heads of the stacked panels (gen_late_tables keys: musiq_crop is MUSIQ on the 1024^2 crops)
PANEL_HEAD = {"ir": "ImageReward", "aesthetic": "Aesthetic", "clip": "CLIP score", "pickscore": "PickScore",
              "hpsv2": r"HPSv2 ($\times100$)", "hpsv21": r"HPSv2.1 ($\times100$)", "musiq": "MUSIQ",
              "musiq_crop": r"MUSIQ$_\text{c}$", "clipiqa": r"CLIP-IQA ($\times100$)", "arniqa": r"ARNIQA ($\times100$)",
              "fid": r"FID $\downarrow$", "kid": r"KID ($\times\smash{10^3}$) $\downarrow$",
              "patch_fid": r"FID$_\text{p}$ $\downarrow$", "patch_kid": r"KID$_\text{p}$ ($\times\smash{10^3}$) $\downarrow$",
              "is": r"IS $\uparrow$", "seconds": r"Time (s) $\downarrow$"}
KREA_PANELS = [("(a) Preference", ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21"]),
               ("(b) Image quality", ["musiq", "musiq_crop", "clipiqa", "arniqa"]),
               ("(c) Set level and time", ["fid", "kid", "patch_fid", "patch_kid", "is", "seconds"])]


def krea_v2():
    """sca_v2_krea minus area from the krea195_sca_v2 run, keyed like gen_late_tables.load_deltas, and its seconds."""
    S, D, ref, corrected_ci = v2swap.load(KREA_RUN)
    if ref != "sca_v2_krea":
        raise ValueError(f"{KREA_RUN}: reference {ref}")
    out = {}
    for (m, metric), v in D.items():
        if m == "area":
            out[late.norm(metric)] = neg(v)
    keys = [k for k, *_ in late.PER_IMAGE + late.DIST]
    missing = [k for k in keys if k not in out]
    if missing:
        raise ValueError(f"{KREA_RUN}: no area deltas for {missing}")
    n = data195.n_prompts(KREA_RUN, v2swap.V2_RESULTS)
    secs = S[("sca_v2_krea", "seconds")] / S[("area", "seconds")]
    return out, corrected_ci, n, secs


def krea_claim(v, secs):
    """'on FLUX.1-Krea-dev \\sca{} wins six metrics and loses ImageReward and patch FID, in a third of the time' from
    the data: wins and losses over the 15 quality metrics with an interval, time as a fraction of AReA's."""
    spec = [(k, h) for k, _, _, _, h in late.PER_IMAGE + late.DIST if k != "seconds"]
    wins = [k for k, h in spec if sig(v[k], h) == 1]
    losses = [k for k, h in spec if sig(v[k], h) == -1]
    untested = [k for k, h in spec if sig(v[k], h) is None]
    n = len(wins)
    text = rf"on FLUX.1-Krea-dev single-pass \sca{{}} wins {WORDS[n] if n < len(WORDS) else n} metric{'' if n == 1 else 's'}"
    text += f" and loses {label_list(losses)}" if losses else " and loses none"
    if 0.28 <= secs < 0.40:
        text += ", in a third of the time"
    elif 0.45 <= secs < 0.55:
        text += ", in half the time"
    else:
        text += rf", in {round(100 * secs)}\% of the time"
    return text, untested


def krea_6144():
    landed = v2swap.landed(KREA_RUN)
    cols = [("e2_6144", "sca_nb", r"\sca{} w/o schedule"), ("e9_krea", "sca_krea", r"Single-pass \sca{}"),
            ("e9_krea", "dype", "DyPE"), ("e9_krea", "sega_official", "SEGA")]
    data = [late.load_deltas(f, m) for f, m, _ in cols]
    sources = ["results/e2_6144", "results/e9_krea"]
    if late.KREA195:  # as gen_late_tables: SCA on all 195 Krea prompts; DyPE and SEGA keep the 100 they ran
        cols[1] = ("krea195", "sca_krea", r"Single-pass \sca{}")
        data[1] = late.load_deltas("krea195", "sca_krea", n="195", setlevel="corrected_fidp_ci/paired_deltas.csv")
        sources.append("results/krea195")
    summary = late.load_summary("krea195" if late.KREA195 else "e9_krea")
    secs = summary[("sca_krea", "seconds")] / summary[("area", "seconds")]
    warnings, corrected_ci = [], True
    claim_today, _ = krea_claim(data[1], secs)
    if landed:
        data[1], corrected_ci, n, secs = krea_v2()
        sources = [s for s in sources if s != "results/krea195"] + [f"results/{KREA_RUN} (sca_v2_krea)"]
        if n != {195}:
            warnings.append(f"{KREA_RUN} records {sorted(n)} prompts; the caption says all 195")
    claim, untested = krea_claim(data[1], secs)
    if landed:
        if claim != claim_today:
            warnings.append(f"caption claim regenerated from {KREA_RUN}: '{claim}' (published: '{claim_today}')")
        if untested:
            warnings.append(f"{label_list(untested)} of sca_v2_krea minus AReA have no interval (no "
                            "corrected_fidp_ci in the z98x run), so the caption's win / loss counts leave them out")
    spec = [(k, h) for k, _, _, _, h in late.PER_IMAGE + late.DIST if k != "seconds"]
    wins6144 = sum(sig(data[0][k], h) == 1 for k, h in spec)
    note = "pendAppKreaSix"
    intervals = (r"Per-image intervals use 1000 prompt draws; FID, KID, and their patch versions resample prompts and "
                 r"reference-image clusters")
    if landed and not corrected_ci:
        intervals += (r", except for \sca{} on FLUX.1-Krea-dev, whose FID and KID intervals resample prompts only and "
                      r"whose patch versions have no interval")
    # Key result (v6): what the two Krea baselines do against AReA under the same protocol, from the colours
    others = []
    for i, ((_, _, name), x) in enumerate(zip(cols[2:], data[2:])):
        better, worse = sig_counts(x, spec)
        of = f" of the {len(spec)} metrics" if i == 0 else ""
        others.append(f"{name} wins {WORDS[len(better)]}{of} and loses {WORDS[len(worse)]}")
    key = "against AReA on FLUX.1-Krea-dev, " + (", and ".join(others) if len(others) == 2 else join(others))
    pending_key = False  # DyPE and SEGA never switch
    caption = (rf"\caption{{\textbf{{At $6144^2$ \sca{{}} without the schedule beats AReA on {wins6144} of {len(spec)} "
               rf"metrics; {claim}.}} Expands the FLUX.1-Krea-dev row of Table~\ref{{tab:generality}} and the $6144^2$ "
               r"result of Section~\ref{sec:sizes}. Each cell is the method minus AReA with its 95\% paired "
               r"percentile-bootstrap interval below it in grey. First row: FLUX.1-dev at $6144^2$, 100 prompts. Other "
               r"rows: FLUX.1-Krea-dev at $4096^2$ with a reach table fitted on that checkpoint, "
               + (r"on all 195 prompts for single-pass \sca{} and on the 100 that DyPE and SEGA ran. " if late.KREA195
                  else r"on 100 prompts each. ")
               + intervals + r". IS: Inception score; time: campaign mean on shared GH200 nodes. \up{Blue} / \dn{orange}: "
               r"significantly better / worse than AReA; for FID, KID, their patch versions, and time, lower is better. "
               r"Key result: " + v2swap.pend(key, pending_key) + "." + "\\" + note + "}")
    pend_flags = [False, not landed, False, False]
    # v6: the four comparisons as rows (ours shaded), one panel per metric family, each estimate above its interval;
    # side by side with their intervals the four columns are 23 pt wider than the text at 7 pt
    heads = {k: (k, PANEL_HEAD[k], scale, d, higher) for k, _, scale, d, higher in late.PER_IMAGE + late.DIST}
    rows = [(c[2], x, p) for c, x, p in zip(cols, data, pend_flags)]
    groups = [(r"FLUX.1-dev, $6144^2$", rows[:1]), (r"FLUX.1-Krea-dev, $4096^2$", rows[1:])]
    lines = [header(sources), STYLE, v2swap.PEND_FALLBACK, r"\begin{table}[!htbp]", caption, r"\label{tab:krea_6144}",
             r"\centering\scriptsize"]
    lines += stack(*[panel(title, [heads[k] for k in keys], groups, r"Method $-$ AReA", shade=(cols[0][2], cols[1][2]))
                     for title, keys in KREA_PANELS])
    lines += [r"\end{table}"]
    if sorted(k for _, keys in KREA_PANELS for k in keys) != sorted(heads):
        raise ValueError("krea_6144: the panels do not cover every metric")
    subs = [(r"\sca{} w/o schedule", r"\sca{}"),
            (r"At $6144^2$ \sca{} without the schedule beats", r"At $6144^2$ \sca{} beats"),
            (r"on FLUX.1-Krea-dev \sca{} wins", r"on FLUX.1-Krea-dev it wins")]
    return {"name": "krea_6144", "kind": "float", "label": "tab:krea_6144", "old": "krea_6144.tex", "runs": [KREA_RUN],
            "note": note, "tex": "\n".join(lines) + "\n", "subs": subs, "warnings": warnings}


# ---------------------------------------------------------------- phase2_baselines ----------------------------------
BASELINE_COLS = [("DyPE", "a4k195_main", "dype"), ("SEGA", "a4k195_main", "sega_official"),
                 ("UltraImage", "a4k195_main", "ultraimage"), ("HRDiT", "a4k195_panel", "hrdit_dev_author")]


def phase2_baselines():
    landed = v2swap.landed(MAIN_V2)
    if landed:
        cols = [derived(run, m, MAIN_V2, "sca_v2") for _, run, m in BASELINE_COLS]
    else:
        cols = [pair_file(run, m, "sca_nb", n=195, seed=0) for _, run, m in BASELINE_COLS]
    # v6: baselines as rows, grouped like Table tab:main; one panel per metric family
    rows = {name: (name, c, not landed) for (name, _, _), c in zip(BASELINE_COLS, cols)}
    groups = [("Single pass, shared sampler", [rows["DyPE"], rows["SEGA"]]),
              ("Multi-stage, own recipe", [rows["UltraImage"], rows["HRDiT"]])]
    assert sum(len(g[1]) for g in groups) == len(BASELINE_COLS)
    lines = [header(["results/a4k195_main", "results/a4k195_panel"]
                    + ([f"per-image CSVs of {MAIN_V2} (sca_v2)"] if landed else [])),
             STYLE, v2swap.PEND_FALLBACK, r"\centering\scriptsize"]
    lines += stack(*[panel(f"{title}: baseline $-$ single-pass \\sca{{}}", [HEADS[k] for k in keys], groups,
                           "Baseline") for title, keys in PANELS])
    warnings = []
    if landed:
        keys = [k for k, *_ in PER_IMAGE]
        for (name, _, _), c in zip(BASELINE_COLS, cols):
            s = [sig(c[k]) for k in keys]
            if name in ("DyPE", "SEGA") and any(x != -1 for x in s):
                ok = [NAMES[k] for k, x in zip(keys, s) if x != -1]
                warnings.append(f"caption 'DyPE and SEGA trail \\sca{{}} on every per-image metric' fails: {name} "
                                f"is not significantly worse on {join(ok)}")
            if name in ("UltraImage", "HRDiT") and not (1 in s and -1 in s):
                warnings.append(f"caption 'UltraImage and HRDiT are mixed' fails for {name}: "
                                f"{s.count(1)} significantly better, {s.count(-1)} significantly worse")
        ultra = dict(zip(keys, (sig(cols[2][k]) for k in keys)))
        claim = {"hpsv21": 1, "pickscore": 1, "clip": -1, "musiq": -1, "musiq_crop1024": -1}
        bad = [NAMES[k] for k, want in claim.items() if ultra[k] != want]
        if bad:
            warnings.append("main.tex paragraph 'Every shared-sampler baseline trails' says UltraImage is significantly "
                            "better on HPSv2.1 and PickScore and worse on CLIP score and both MUSIQ scores; it no "
                            f"longer holds for {join(bad)}")
    return {"name": "phase2_baselines", "kind": "tabular", "label": "tab:phase2_baselines", "runs": [MAIN_V2],
            "note": "pendAppPhaseBaselines", "tex": "\n".join(lines) + "\n", "subs": [], "warnings": warnings}


# ---------------------------------------------------------------- phase2_general ------------------------------------
GENERAL_COLS = [("HRGen-1K", "hrgen1k_dev", "hrgen1k_dev_sca_v2", 1000, 0), ("Seed 1", "a4k195_seed1",
                 "a4k195_seed1_sca_v2", 195, 1), ("Seed 2", "a4k195_seed2", "a4k195_seed2_sca_v2", 195, 2)]
FLUX2 = v2swap.RESULTS / "e8_flux2_sca_vs_full.csv"


def flux2():
    out = {}
    for r in _rows(FLUX2):
        if (r["method"], r["reference"], r["n_prompts"]) != ("flux2_sca", "flux2_full", "50"):
            raise ValueError(f"{FLUX2}: row {r}")
        out[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    return out


def phase2_general():
    cols, flags, warnings, sources = [], [], [], []
    for name, pub, run, n, seed in GENERAL_COLS:
        if v2swap.landed(run):
            c = pair_file(run, "sca_v2", "area", n=n, seed=seed)
            sources.append(f"results/{run}")
            s = [sig(c[k]) for k, *_ in PER_IMAGE]
            if any(x != 1 for x in s):
                warnings.append(f"caption 'The gains hold on HRGen-1K, two more seeds, and FLUX.2-dev' fails for "
                                f"{name}: sca_v2 is not significantly better than AReA on "
                                + join(NAMES[k] for (k, *_), x in zip(PER_IMAGE, s) if x != 1)
                                + (" (and significantly worse somewhere, which the caption's colour key omits)"
                                   if -1 in s else ""))
        else:
            c = pair_file(pub, "sca_nb", "area", n=n, seed=seed)
            sources.append(f"results/{pub}")
        cols.append(c)
        flags.append(not v2swap.landed(run))
    cols.append(flux2())
    flags.append(False)
    sources.append("results/e8_flux2_sca_vs_full.csv")
    open_ = [k for k, *_ in PER_IMAGE if sig(cols[-1][k]) != 1]
    if open_ != ["clip"]:
        warnings.append("caption key result 'only FLUX.2-dev's CLIP score interval includes zero' fails: FLUX.2-dev "
                        f"is not significantly better on {label_list(open_) or 'none'}")
    # v6: settings as rows, as in gen_main_tables.py's app_tables/gen_all.tex; one panel per metric family
    rows = [(name, c, f) for (name, *_), c, f in zip(GENERAL_COLS, cols, flags)]
    groups = [(r"Single-pass \sca{} $-$ AReA", rows), (rf"\sca{{}} {WOS} $-$ full attention",
                                                       [("FLUX.2-dev", cols[-1], flags[-1])])]
    lines = [header(sources), STYLE, v2swap.PEND_FALLBACK, r"\centering\scriptsize"]
    lines += stack(*[panel(title, [HEADS[k] for k in keys], groups, "Setting") for title, keys in PANELS])
    return {"name": "phase2_general", "kind": "tabular", "label": "tab:phase2_general",
            "runs": [c[2] for c in GENERAL_COLS], "note": "pendAppPhaseGeneral", "tex": "\n".join(lines) + "\n",
            "subs": [], "warnings": warnings}


# ---------------------------------------------------------------- panel46_deltas ------------------------------------
PANEL_LABELS = {"hpsv2": "HPSv2", "hpsv21": "HPSv2.1", "clipiqa": "CLIP-IQA", "arniqa": "ARNIQA"}
PANEL_PER_IMAGE = [(k, PANEL_LABELS.get(k, lab), s, d) for k, lab, s, d in PER_IMAGE]
PANEL_SET = [("fid", r"FID $\downarrow$", 1, 1), ("kid", r"KID $\downarrow$", 1000, 1),
             ("patch_fid", r"FID$_\text{p}$ $\downarrow$", 1, 1), ("patch_kid", r"KID$_\text{p}$ $\downarrow$", 1000, 1),
             ("seconds", r"Time (s) $\downarrow$", 1, 1)]
SINGLE = [("Full attn.", "a4k195_main", "full"), ("DyPE", "a4k195_main", "dype"),
          ("SigMa", "a4k195_panel", "sigma_dev_matched"), ("SEGA", "a4k195_main", "sega_official"),
          ("AReA", "a4k195_main", "area")]
MULTI = [("UltraImage", "a4k195_panel", "ultraimage"), ("HRDiT", "a4k195_panel", "hrdit_dev_author")]


def panel46_deltas():
    landed = v2swap.landed(MAIN_V2)
    loaded = {run: data195.load(run) for run in ("a4k195_main", "a4k195_panel")}
    for run, (_, _, ref, corrected_ci) in loaded.items():
        if ref != "sca_nb" or not corrected_ci:
            raise ValueError(f"results/{run}: reference {ref}, corrected_fidp_ci {corrected_ci}")

    def per_image(run, m):
        if not landed:
            return pair_file(run, m, "sca_nb", n=195, seed=0)
        if m == "area":  # AReA was re-evaluated in the sca_v2 run itself: its within-run file, negated
            return {k: neg(v) for k, v in pair_file(MAIN_V2, "sca_v2", "area", n=195, seed=0).items()}
        return derived(run, m, MAIN_V2, "sca_v2")

    def setlevel(run, m):
        D = loaded[run][1]
        return {k: D[(m, k)] for k, *_ in PANEL_SET}

    lines = [header(["results/a4k195_main", "results/a4k195_panel"]
                    + ([f"results/{MAIN_V2} and per-image CSVs of {MAIN_V2} (sca_v2)"] if landed else [])),
             STYLE, v2swap.PEND_FALLBACK, r"\centering\scriptsize"]
    warnings = []
    # v6: baselines as rows in the two groups of Table tab:main, one panel per metric family; the set-level and time
    # panel pairs against SCA without the schedule (set-level intervals across runs do not exist), and says so
    groups_pi, groups_sl = [], []
    for group, title in ((SINGLE, "Single pass, shared sampler"), (MULTI, "Multi-stage, own recipe")):
        pi = [per_image(run, m) for _, run, m in group]
        sl = [setlevel(run, m) for _, run, m in group]
        groups_pi.append((title, [(name, c, not landed) for (name, _, _), c in zip(group, pi)]))
        groups_sl.append((title, [(name, c, False) for (name, _, _), c in zip(group, sl)]))
        if landed:
            keys = [k for k, *_ in PER_IMAGE]
            for (name, _, _), c in zip(group, pi):
                s_ = [sig(c[k]) for k in keys]
                if group is SINGLE and any(x != -1 for x in s_):
                    warnings.append("caption 'every shared-protocol baseline trails \\sca{} on all ten per-image "
                                    f"metrics' fails: {name} is not significantly worse on "
                                    + join(NAMES[k] for k, x in zip(keys, s_) if x != -1))
                if group is MULTI:
                    lead = [NAMES[k] for k in PREFERENCE if sig(c[k]) == 1]
                    if len(lead) < 2:
                        warnings.append(f"caption 'HRDiT and UltraImage lead on several preference metrics' fails "
                                        f"for {name}: significantly better only on {join(lead) or 'none'}")
    # proposed v6 caption, key result: "both multi-stage recipes are significantly better on Aesthetic, PickScore, and
    # HPSv2.1 and worse on both MUSIQ scores, and both have higher patch FID and patch KID than SCA without the schedule"
    for name, pi_c, sl_c in zip([g[0] for g in MULTI], [r[1] for r in groups_pi[1][1]], [r[1] for r in groups_sl[1][1]]):
        wrong = [NAMES[k] for k in ("aesthetic", "pickscore", "hpsv21") if sig(pi_c[k]) != 1] + \
            [NAMES[k] for k in ("musiq", "musiq_crop1024") if sig(pi_c[k]) != -1] + \
            [NAMES[k] for k in ("patch_fid", "patch_kid") if sig(sl_c[k], higher=False) != -1]
        if wrong:
            warnings.append(f"caption key result 'both multi-stage recipes ...' fails for {name} on {join(wrong)}")
    heads_pi = {k: (k, label, scale, d, True) for k, label, scale, d in PANEL_PER_IMAGE}
    heads_sl = [(k, label, scale, d, False) for k, label, scale, d in PANEL_SET]
    set_title = r"(c) Set level and time: baseline $-$ \sca{} " + WOS
    lines += stack(*[panel(f"{title}: baseline $-$ single-pass \\sca{{}}", [heads_pi[k] for k in keys], groups_pi,
                           "Baseline") for title, keys in PANELS],
                   panel(set_title, heads_sl, groups_sl, "Baseline"))
    # the set-level panel title marks the cells that stay against SCA without the schedule; --check removes it
    text = "\n".join(lines) + "\n"
    subs = [(set_title, "")]
    if text.count(set_title) != 1:
        raise ValueError(f"set-level panel title occurs {text.count(set_title)} times")
    return {"name": "panel46_deltas", "kind": "tabular", "label": "tab:panel46_deltas", "runs": [MAIN_V2],
            "note": "pendAppPanelDeltas", "tex": text, "subs": subs, "warnings": warnings}


# ---------------------------------------------------------------- prog_deltas ---------------------------------------
PROG_RUN = "prog195_sca_v2"


def prog_deltas():
    prog_landed = v2swap.landed(PROG_RUN)
    single_landed = prog_landed and v2swap.landed(MAIN_V2)
    pub = [pair_file("a4k195_panel", a, b, n=prog.N_PROMPTS, seed=0, res=prog.RES) for a, b, _ in prog.PAIRS]
    data = list(pub)
    sources = ["results/a4k195_panel"]
    if prog_landed:
        data[0] = pair_file(PROG_RUN, "sca_v2_prog", "hrdit_dev_author", n=prog.N_PROMPTS, seed=0, res=prog.RES)
        sources.append(f"results/{PROG_RUN}")
    if single_landed:
        data[1] = derived(PROG_RUN, "sca_v2_prog", MAIN_V2, "sca_v2")
        sources.append(f"per-image CSVs of {PROG_RUN} and {MAIN_V2}")
    flags = [not prog_landed, not single_landed, False, False]
    note = "pendAppProgDeltas"
    worse_now = [NAMES[k] for k, *_ in prog.METRICS if prog_landed and data[0][k][2] < 0]
    lead = (rf"Per image, progressive \sca{{}} is significantly worse than HRDiT only on {join(worse_now)}."
            if worse_now else r"Against HRDiT, progressive \sca{} is not significantly worse on any per-image metric; "
            r"single-pass \sca{} without the schedule at 42 steps is.")
    # Key result (v6): progressive against HRDiT and against single-pass SCA, from the colours of the first two columns
    keys = [k for k, *_ in prog.METRICS]
    better_h, _ = sig_counts(data[0], [(k, True) for k in keys])
    better_s, worse_s = sig_counts(data[1], [(k, True) for k in keys])
    key = (r"progressive \sca{} is significantly better than HRDiT on "
           + (label_list(better_h) if better_h else "none of the ten metrics")
           + rf" and better than single-pass \sca{{}} on {WORDS[len(better_s)]} of {WORDS[len(keys)]}"
           + (f", but worse on {label_list(worse_s)}" if worse_s else ", and worse on none"))
    caption = (r"\caption{\textbf{" + lead + "}"
               r" Expands the progressive rows of Table~\ref{tab:main}. FLUX.1-dev at $4096^2$, "
               rf"{prog.N_PROMPTS} prompts, seed label 0. Each cell is the first-named setting minus the second, with its "
               rf"95\% paired percentile-bootstrap interval over {prog.DRAWS} prompt draws in grey. \up{{Blue}} / "
               r"\dn{orange}: the interval excludes zero in favour of / against the first-named setting. Progressive "
               r"\sca{} runs HRDiT's schedule with \sca{} attention (Section~\ref{sec:progressive}), and single-pass "
               r"\sca{} takes 28 steps. The last two columns use \sca{} without the schedule, at 42 steps sized to "
               r"HRDiT's campaign time and at 28. Intervals that include zero do not establish equivalence; "
               r"Table~\ref{tab:holm} gives the Holm-adjusted calls. Key result: "
               + v2swap.pend(key, flags[0] or flags[1]) + "." + "\\" + note + "}")
    head = [r"\toprule",
            r" & \multicolumn{4}{c}{Progressive \sca{} $-$} & \multicolumn{4}{c}{\sca{} " + WOS + r"} \\",
            r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
            r"Metric & \multicolumn{2}{c}{HRDiT} & \multicolumn{2}{c}{single-pass \sca{}} & "
            r"\multicolumn{2}{c}{42 $-$ 28 steps} & \multicolumn{2}{c}{HRDiT $-$ 42 steps} \\",
            r"\midrule"]
    fmt = {k: (scale, d) for k, _, scale, d in prog.METRICS}
    body = metric_rows(keys, 9, lambda k: pair_cells(data, k, *fmt[k], flags))
    lines = [header(sources), STYLE, v2swap.PEND_FALLBACK, r"\begin{table}[!htbp]", caption, r"\label{tab:prog_deltas}",
             r"\centering\scriptsize", rf"\scaFitTab{{l{PAIR}{PAIR_GAP}{PAIR}{PAIR}}}{{5}}{{%"]
    lines += ["  " + x for x in head + body + [r"\bottomrule"]] + ["}", r"\end{table}"]
    warnings = []
    if prog_landed:
        worse = [NAMES[k] for k, *_ in prog.METRICS if data[0][k][2] < 0]
        if worse:  # the caption lead names them (worse_now); main.tex must say so too
            warnings.append(f"note: progressive \\sca{{}} is significantly worse than HRDiT on {join(worse)}; the "
                            "caption lead names them; check main.tex paragraph 'Progressive \\sca{} matches HRDiT'")
    if single_landed and data[1]["musiq_crop1024"][2] >= 0:
        warnings.append("main.tex paragraph 'Progressive \\sca{} matches HRDiT' says progressive has lower "
                        "MUSIQ$_\\text{c}$ than single-pass \\sca{}; the sca_v2 interval no longer excludes zero")
    subs = [(r"single-pass \sca{} without the schedule at 42 steps is.", r"single-pass \sca{} with 42 steps is.")]
    return {"name": "prog_deltas", "kind": "float", "label": "tab:prog_deltas", "old": "prog_deltas.tex",
            "runs": [PROG_RUN, MAIN_V2], "note": note, "tex": "\n".join(lines) + "\n", "subs": subs,
            "warnings": warnings}


def tables():
    return [krea_6144(), phase2_baselines(), phase2_general(), panel46_deltas(), prog_deltas()]
