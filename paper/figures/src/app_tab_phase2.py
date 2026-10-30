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
  tab:phase2_sensitivity   calibration-budget and gate variants exist only for published SCA (46 prompts).
  tab:phase2_budget        56-step and guidance-7 runs exist only for published SCA.
Warnings are computed only for columns whose run has landed.
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
WOS = "w/o schedule"
WOS_SHORT = "w/o sched."

# (csv metric, row label, scale, decimals) for the 4096^2 delta tables typed after gen_tables.py's MET
PER_IMAGE = [("ir", "ImageReward", 1, 3), ("aesthetic", "Aesthetic", 1, 2), ("clip", "CLIP score", 1, 2),
             ("pickscore", "PickScore", 1, 2), ("hpsv2", r"HPSv2 ($\times100$)", 100, 2),
             ("hpsv21", r"HPSv2.1 ($\times100$)", 100, 2), ("musiq", "MUSIQ", 1, 1),
             ("musiq_crop1024", r"MUSIQ$_\text{c}$", 1, 1), ("clipiqa", r"CLIP-IQA ($\times100$)", 100, 1),
             ("arniqa", r"ARNIQA ($\times100$)", 100, 1)]
PREFERENCE = ("ir", "aesthetic", "pickscore", "hpsv2", "hpsv21")
NAMES = {"ir": "ImageReward", "aesthetic": "Aesthetic", "clip": "CLIP score", "pickscore": "PickScore",
         "hpsv2": "HPSv2", "hpsv21": "HPSv2.1", "musiq": "MUSIQ", "musiq_crop": r"MUSIQ$_\text{c}$",
         "musiq_crop1024": r"MUSIQ$_\text{c}$", "clipiqa": "CLIP-IQA", "arniqa": "ARNIQA", "fid": "FID",
         "kid": "KID", "patch_fid": "patch FID", "patch_kid": "patch KID", "is": "Inception score",
         "seconds": "time"}
WORDS = "no one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split()


# ---------------------------------------------------------------- formatting ----------------------------------------
def signed(x, d):
    """gen_prog_tables.signed: explicit sign, LaTeX {-} for minus, a rounded zero keeps the sign of x."""
    return prog.signed(x, d)


def cell(v, scale, d, higher=True):
    """Delta with its 95% interval; \\up / \\dn when the interval excludes zero in favour of / against the method.
    No interval (lo or hi None): the point alone, unmarked."""
    delta, lo, hi = v
    if lo is None or hi is None:
        return f"${signed(delta * scale, d)}$"
    delta, lo, hi = delta * scale, lo * scale, hi * scale
    body = f"${signed(delta, d)}$"
    good, bad = (lo > 0, hi < 0) if higher else (hi < 0, lo > 0)
    if good:
        body = r"\up{" + body + "}"
    elif bad:
        body = r"\dn{" + body + "}"
    return body + r" {\tiny[$" + signed(lo, d) + "$, $" + signed(hi, d) + "$]}"


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


def rows_for(cols, metrics, pend_flags, higher=True):
    """Body rows: metric label, then one cell per column; pend_flags[i] wraps column i in \\pend."""
    out = []
    for key, label, scale, d in metrics:
        cells = [v2swap.pend(cell(c[key], scale, d, higher), p) for c, p in zip(cols, pend_flags)]
        out.append(label + " & " + " & ".join(cells) + r" \\")
    return out


# ---------------------------------------------------------------- krea_6144 -----------------------------------------
KREA_RUN = "krea195_sca_v2"


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
    text = rf"on FLUX.1-Krea-dev \sca{{}} wins {WORDS[n] if n < len(WORDS) else n} metric{'' if n == 1 else 's'}"
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
    cols = [("e2_6144", "sca_nb", rf"\sca{{}} {WOS_SHORT}"), ("e9_krea", "sca_krea", r"\sca{}"),
            ("e9_krea", "dype", "DyPE"), ("e9_krea", "sega_official", "SEGA")]
    data = [late.load_deltas(f, m) for f, m, _ in cols]
    sources = ["results/e2_6144", "results/e9_krea"]
    if late.KREA195:  # as gen_late_tables: SCA on all 195 Krea prompts; DyPE and SEGA keep the 100 they ran
        cols[1] = ("krea195", "sca_krea", r"\sca{}")
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
    caption = (rf"\caption{{\textbf{{At $6144^2$ \sca{{}} without the schedule beats AReA on {wins6144} of {len(spec)} "
               rf"metrics; {claim}.}} Method minus AReA with 95\% paired intervals over "
               + (r"100 prompts at $6144^2$ and for DyPE and SEGA, which ran only those, and all 195 for \sca{} on "
                  r"FLUX.1-Krea-dev; first column " if late.KREA195 else "100 prompts each; first column ")
               + r"FLUX.1-dev at $6144^2$, others FLUX.1-Krea-dev at $4096^2$ with a Krea-calibrated \sca{} table. "
               + intervals + r". Lower is better for FID, KID, their patch versions, and time. \up{Blue} / \dn{orange}: "
               r"significantly better / worse than AReA. Time is the campaign mean on shared GH200 nodes." + "\\" + note
               + "}")
    pend_flags = [False, not landed, False, False]
    lines = [v2swap.PEND_FALLBACK, header(sources), r"\begin{table}[!htbp]", caption, r"\label{tab:krea_6144}",
             r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}\let\tiny\scriptsize",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r" & $6144^2$ & \multicolumn{3}{c}{FLUX.1-Krea-dev, $4096^2$} \\", r"\cmidrule(lr){2-2}\cmidrule(l){3-5}",
             "Metric & " + " & ".join(c[2] for c in cols) + r" \\", r"\midrule"]
    for key, label, scale, d, higher in late.PER_IMAGE + late.DIST:
        cells = [v2swap.pend(cell(x[key], scale, d, higher), p) for x, p in zip(data, pend_flags)]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
        if key == "arniqa":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    subs = [(rf"\sca{{}} {WOS_SHORT}", r"\sca{}"),
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
    lines = [v2swap.PEND_FALLBACK,
             header(["results/a4k195_main", "results/a4k195_panel"]
                    + ([f"per-image CSVs of {MAIN_V2} (sca_v2)"] if landed else [])),
             r"\centering\scriptsize\setlength{\tabcolsep}{3pt}\let\tiny\scriptsize",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
             r"\begin{tabular}{lcccc}", r"\toprule",
             "Metric & " + " & ".join(c[0] for c in BASELINE_COLS) + r" \\", r"\midrule"]
    lines += rows_for(cols, PER_IMAGE, [not landed] * len(cols))
    lines += [r"\bottomrule", r"\end{tabular}}"]
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
    lines = [v2swap.PEND_FALLBACK, header(sources),
             r"\centering\scriptsize\setlength{\tabcolsep}{2pt}\let\tiny\scriptsize",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
             r"\begin{tabular}{lcccc}", r"\toprule",
             "Metric & " + " & ".join(c[0] for c in GENERAL_COLS) + rf" & FLUX.2-dev {WOS_SHORT} \\", r"\midrule"]
    lines += rows_for(cols, PER_IMAGE, flags)
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return {"name": "phase2_general", "kind": "tabular", "label": "tab:phase2_general",
            "runs": [c[2] for c in GENERAL_COLS], "note": "pendAppPhaseGeneral", "tex": "\n".join(lines) + "\n",
            "subs": [(f"FLUX.2-dev {WOS_SHORT}", "FLUX.2-dev")], "warnings": warnings}


# ---------------------------------------------------------------- panel46_deltas ------------------------------------
PANEL_LABELS = {"hpsv2": "HPSv2", "hpsv21": "HPSv2.1", "clipiqa": "CLIP-IQA", "arniqa": "ARNIQA"}
PANEL_PER_IMAGE = [(k, PANEL_LABELS.get(k, lab), s, d) for k, lab, s, d in PER_IMAGE]
PANEL_SET = [("fid", r"FID$\downarrow$", 1, 1), ("kid", r"KID$\downarrow$", 1000, 1),
             ("patch_fid", r"FID$_\text{p}\downarrow$", 1, 1), ("patch_kid", r"KID$_\text{p}\downarrow$", 1000, 1),
             ("seconds", r"Time (s)$\downarrow$", 1, 1)]
SINGLE = [("Full attn.", "a4k195_main", "full"), ("DyPE", "a4k195_main", "dype"),
          ("SigMa", "a4k195_panel", "sigma_dev_matched"), ("SEGA", "a4k195_main", "sega_official"),
          ("AReA", "a4k195_main", "area")]
MULTI = [("UltraImage", "a4k195_panel", "ultraimage"), ("HRDiT", "a4k195_panel", "hrdit_dev_author")]


def panel_group_row(ncols):
    """Marks the rows that stay against published SCA: set-level intervals across runs do not exist."""
    return r"\multicolumn{" + str(ncols) + r"}{l}{\emph{Baseline minus \sca{} " + WOS + r"}} \\"


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

    lines = [v2swap.PEND_FALLBACK,
             header(["results/a4k195_main", "results/a4k195_panel"]
                    + ([f"results/{MAIN_V2} and per-image CSVs of {MAIN_V2} (sca_v2)"] if landed else [])),
             r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}\let\tiny\scriptsize"]
    blocks, warnings, subs = [], [], []
    for group, cols, spec, title in ((SINGLE, 5, r"l*{5}{Y{62pt}}", r"Single pass, shared sampler"),
                                     (MULTI, 2, "lcc", r"Multi-stage, own recipe")):
        pi = [per_image(run, m) for _, run, m in group]
        sl = [setlevel(run, m) for _, run, m in group]
        block = [r"\begin{tabular}{" + spec + "}", r"\toprule",
                 rf" & \multicolumn{{{cols}}}{{c}}{{\emph{{{title}}}}} \\", rf"\cmidrule(l){{2-{cols + 1}}}",
                 "Metric & " + " & ".join(c[0] for c in group) + r" \\", r"\midrule"]
        block += rows_for(pi, PANEL_PER_IMAGE, [not landed] * cols)
        marker = r"\midrule" + "\n" + panel_group_row(cols + 1)
        block.append(marker)
        subs.append((marker, ""))
        block += rows_for(sl, PANEL_SET, [False] * cols, higher=False)
        block += [r"\bottomrule", r"\end{tabular}"]
        blocks.append(block)
        if landed:
            keys = [k for k, *_ in PER_IMAGE]
            for (name, _, _), c in zip(group, pi):
                s = [sig(c[k]) for k in keys]
                if group is SINGLE and any(x != -1 for x in s):
                    warnings.append("caption 'every shared-protocol baseline trails \\sca{} on all ten per-image "
                                    f"metrics' fails: {name} is not significantly worse on "
                                    + join(NAMES[k] for k, x in zip(keys, s) if x != -1))
                if group is MULTI:
                    lead = [NAMES[k] for k in PREFERENCE if sig(c[k]) == 1]
                    if len(lead) < 2:
                        warnings.append(f"caption 'HRDiT and UltraImage lead on several preference metrics' fails "
                                        f"for {name}: significantly better only on {join(lead) or 'none'}")
    lines += blocks[0] + ["", r"\medskip"] + blocks[1]
    # a uniqueness guard for the two marker substitutions: --check undoes them with a plain replace
    text = "\n".join(lines) + "\n"
    for new, _ in subs:
        if text.count(new) != 1:
            raise ValueError(f"marker row occurs {text.count(new)} times")
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
    group = r" & & & \multicolumn{2}{c}{\emph{\sca{} " + WOS + r"}} \\" + "\n" + r"\cmidrule(l){4-5}"
    caption = (r"\caption{\textbf{Against HRDiT, progressive \sca{} is not significantly worse on any per-image metric; "
               rf"single-pass \sca{{}} without the schedule at 42 steps is.}} $4096^2$, {prog.N_PROMPTS} prompts, seed "
               r"label 0. Per-image differences (first-named setting minus "
               rf"second) with 95\% paired intervals over {prog.DRAWS} prompt draws; \up{{blue}} / \dn{{orange}}: the "
               r"interval excludes zero in favour of / against the first-named setting. Progressive: HRDiT's schedule "
               r"with \sca{} attention (Section~\ref{sec:progressive}); 42 steps: single-pass \sca{} without the "
               r"schedule, sized to HRDiT's campaign time. Intervals that include zero do not establish equivalence."
               + "\\" + note + "}")
    lines = [v2swap.PEND_FALLBACK, header(sources), r"\begin{table}[!htbp]", caption, r"\label{tab:prog_deltas}",
             r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}\let\tiny\scriptsize",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
             r"\begin{tabular}{l" + "c" * len(prog.PAIRS) + "}", r"\toprule", group,
             "Metric & " + " & ".join(p[2] for p in prog.PAIRS) + r" \\", r"\midrule"]
    for key, label, scale, d in prog.METRICS:
        lines.append(label + " & " + " & ".join(v2swap.pend(prog.cell(x[key], scale, d), f)
                                                 for x, f in zip(data, flags)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    warnings = []
    if prog_landed:
        worse = [NAMES[k] for k, *_ in prog.METRICS if data[0][k][2] < 0]
        if worse:
            warnings.append("caption 'Against HRDiT, progressive \\sca{} is not significantly worse on any per-image "
                            f"metric' fails: significantly worse on {join(worse)} (also main.tex, paragraph "
                            "'Progressive \\sca{} matches HRDiT')")
    if single_landed and data[1]["musiq_crop1024"][2] >= 0:
        warnings.append("main.tex paragraph 'Progressive \\sca{} matches HRDiT' says progressive has lower "
                        "MUSIQ$_\\text{c}$ than single-pass \\sca{}; the sca_v2 interval no longer excludes zero")
    subs = [(group, ""),
            (r"single-pass \sca{} without the schedule at 42 steps is.", r"single-pass \sca{} with 42 steps is."),
            (r"42 steps: single-pass \sca{} without the schedule, sized to", r"42 steps: single-pass \sca{} sized to")]
    return {"name": "prog_deltas", "kind": "float", "label": "tab:prog_deltas", "old": "prog_deltas.tex",
            "runs": [PROG_RUN, MAIN_V2], "note": note, "tex": "\n".join(lines) + "\n", "subs": subs,
            "warnings": warnings}


def tables():
    return [krea_6144(), phase2_baselines(), phase2_general(), panel46_deltas(), prog_deltas()]
