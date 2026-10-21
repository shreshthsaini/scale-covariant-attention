"""Build the published-number comparison (CSV + LaTeX) for the shared Aesthetic-4K / HRGen-1K benchmarks.

Published values come from published.csv (transcribed from the cited tables and checked against the PDF text).
Our values come from the evaluation summaries under results/. The 195-prompt evaluations (results/a4k195_*,
results/krea195, results/hrgen1k_*) are used when present; otherwise the builder falls back to the older subsets
(191 prompts on FLUX.1-dev, 100 on FLUX.1-Krea-dev, 200 HRGen prompts) and marks every such row with its prompt count.
Run: python3 -I results/published_compare/build.py   (writes ours.csv, compare.csv, and
paper/figures/snippets/published_compare.tex)
"""
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
PAPER = RES.parent / "paper" / "figures" / "snippets" / "published_compare.tex"
PAPER_HRGEN = RES.parent / "paper" / "figures" / "snippets" / "published_hrgen.tex"  # appendix: all of AReA's Table 2

# metric key -> (column label, higher is better)
METRICS = {"ir": ("ImageReward", True), "aesthetic": ("Aesthetic", True), "clip": ("CLIP", True),
           "pickscore": ("PickScore", True), "hpsv2": ("HPS", True), "musiq": ("MUSIQ", True),
           "clipiqa": ("CLIP-IQA", True), "arniqa": ("ARNIQA", True), "is": ("IS", True), "fid": ("FID", False),
           "fidp": ("FIDp", False)}
OUR_NAME = {"sca_nb": "SCA (ours)", "sca_krea": "SCA (ours)", "area": "AReA", "full": "Base (full attention)",
            "dype": "DyPE", "sega_official": "SEGA", "ultraimage": "UltraImage", "hrdit_dev_author": "HRDiT",
            "sigma_dev_matched": "SigMa"}


def read_summary(path):
    """Return {(method, metric): (value, n_prompts)} from an eval.py or eval_extra.py summary CSV."""
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        try:
            v = float(r["value"])
        except (KeyError, ValueError):
            continue
        out[(r["method"], r["metric"])] = (v, int(float(r.get("n_prompts") or 0)))
    return out


def gather(sources):
    """sources: list of (directory, corrected_patch_csv or None); later entries never override earlier ones."""
    vals, origin = {}, {}
    for d, corrected in sources:
        d = RES / d
        files = [d / "base_summary.csv", d / "extra_summary.csv"] if d.is_dir() else []
        if not d.is_dir():  # flat historical layout: results/<prefix>_{base,extra}_summary.csv
            files = [RES / f"{d.name}_base_summary.csv", RES / f"{d.name}_extra_summary.csv"]
        for f in files:
            for k, v in read_summary(f).items():
                if k not in vals:
                    vals[k], origin[k] = v, str(f.relative_to(RES))
        if corrected:
            for (m, metric), v in read_summary(RES / corrected).items():
                if metric == "patch_fid" and (m, "fidp") not in vals:
                    vals[(m, "fidp")], origin[(m, "fidp")] = v, corrected
    return vals, origin


BLOCKS = [
    # (block id, label, model, benchmark, res, methods, source list in priority order)
    ("krea", "FLUX.1-Krea-dev", "FLUX.1-Krea-dev, 28 steps, guidance 4.5", "Aesthetic-4K", "4096x4096",
     ["area", "dype", "sega_official", "sca_krea"],
     [("krea195", "krea195/corrected_fidp/summary.csv"), ("e9_krea", None)]),
    ("dev", "FLUX.1-dev", "FLUX.1-dev, shared sampler (28 steps, guidance 4.5)", "Aesthetic-4K", "4096x4096",
     ["full", "dype", "sega_official", "ultraimage", "sigma_dev_matched", "hrdit_dev_author", "area", "sca_nb"],
     [("a4k195_main", "a4k195_main/corrected_fidp/summary.csv"),
      ("a4k195_panel", "a4k195_panel/corrected_fidp/summary.csv"),
      ("hrdit191", "hrdit191/corrected_fidp/hrdit191_corrected/summary.csv"),
      ("final4k_nb", None), ("e1_baselines", None), ("e1_ultraimage", None), ("final4k_full", None),
      ("baseline46", None)]),
    ("hrgen_krea", "FLUX.1-Krea-dev", "FLUX.1-Krea-dev, 28 steps, guidance 4.5", "HRGen-1K", "4096x4096",
     ["sca_krea"], [("hrgen1k_krea", None)]),
    ("hrgen_dev", "FLUX.1-dev", "FLUX.1-dev, shared sampler", "HRGen-1K", "4096x4096",
     ["area", "sca_nb"], [("hrgen1k_dev", None), ("e4_hrgen", None)]),
]
FULL_N = {"Aesthetic-4K": 195, "HRGen-1K": 1000}


def main():
    pub = list(csv.DictReader(open(HERE / "published.csv")))
    ours_rows = []
    ours = {}
    for bid, label, model, bench, res, methods, sources in BLOCKS:
        vals, origin = gather(sources)
        for m in methods:
            for metric in METRICS:
                key = (m, "fidp" if metric == "fidp" else metric)
                if key not in vals:
                    continue
                v, n = vals[key]
                ours[(bid, m, metric)] = (v, n)
                ours_rows.append(dict(block=bid, model=model, benchmark=bench, res=res, method=OUR_NAME[m], method_key=m,
                                      metric=metric, value=f"{v:.6g}", n_prompts=n, full_benchmark=n == FULL_N[bench],
                                      source=origin[key]))
    with open(HERE / "ours.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ours_rows[0].keys())); w.writeheader(); w.writerows(ours_rows)

    # compare.csv: published value next to our value for the same method, benchmark and size.
    cmp_rows = []
    for p in pub:
        if p["res"] != "4096x4096":
            continue
        for bid, label, model, bench, res, methods, _ in BLOCKS:
            if bench != p["benchmark"]:
                continue
            for m in methods:
                if OUR_NAME[m] != p["method"] and not (p["method"].startswith("Base") and m == "full"):
                    continue
                if (bid, m, p["metric"]) not in ours:
                    continue
                v, n = ours[(bid, m, p["metric"])]
                cmp_rows.append(dict(benchmark=bench, res=res, method=p["method"], metric=p["metric"],
                                     published=p["value"], published_source=f"{p['source']} {p['arxiv']} {p['table']}",
                                     ours=f"{v:.6g}", ours_model=label, ours_n=n, ours_minus_published=f"{v - float(p['value']):.4g}",
                                     comparable="no: crop protocol not stated" if p["metric"] == "fidp" else
                                     ("checkpoint differs" if label == "FLUX.1-dev" else "yes")))
    with open(HERE / "compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys())); w.writeheader(); w.writerows(cmp_rows)
    write_tex(pub, ours)
    print(f"ours.csv {len(ours_rows)} rows, compare.csv {len(cmp_rows)} rows, wrote {PAPER}")


COLS = ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "musiq", "fid"]
HEAD = ["Image-", "Aes-", "", "Pick-", "", "", ""]
HEAD2 = ["Reward$\\uparrow$", "thetic$\\uparrow$", "CLIP$\\uparrow$", "Score$\\uparrow$", "HPS$\\uparrow$",
         "MUSIQ$\\uparrow$", "FID$\\downarrow$"]
FMT = {"ir": "{:.2f}", "aesthetic": "{:.2f}", "clip": "{:.2f}", "pickscore": "{:.2f}", "hpsv2": "{:.2f}",
       "musiq": "{:.1f}", "fid": "{:.1f}", "is": "{:.1f}"}


def fmt(metric, v):
    return FMT[metric].format(100 * v if metric == "hpsv2" else v)


def write_tex(pub, ours):
    pubv, prec1 = {}, set()
    for p in pub:
        if p["source"] == "AReA" and p["benchmark"] == "Aesthetic-4K" and p["res"] == "4096x4096":
            # Table 1 rows come first in published.csv; Table 6 only fills metrics Table 1 lacks (HPS, MUSIQ, ...).
            pubv.setdefault(p["method"], {}).setdefault(p["metric"], float(p["value"]))
            if p["table"].startswith("Table 6") and p["metric"] == "clip":
                prec1.add(p["method"])  # Table 6 prints CLIP with one decimal
    rows = []  # (group, label, values, n, rankable, ours_flag)
    for name in ["Base (full attention)", "UltraImage", "DyPE", "SigMa", "SEGA", "AReA"]:
        rows.append(("pub", name.replace("Base (full attention)", "Full attention"), pubv[name], 195, True, False))
    # AReA's multi-stage rows (its Table 6: HiFlow, I-Max, ScaleDiff) stay in published.csv and compare.csv only;
    # the main-text table keeps the single-pass methods.
    for bid in ("krea", "dev"):
        ms = ["area", "dype", "sega_official", "sca_krea"] if bid == "krea" else ["area", "sca_nb"]
        for m in ms:
            vals = {k: ours[(bid, m, k)][0] for k in COLS if (bid, m, k) in ours}
            ns = {ours[(bid, m, k)][1] for k in COLS if (bid, m, k) in ours}
            if not vals:
                continue
            n = min(ns)
            rows.append((bid, OUR_NAME[m], vals, n, bid == "krea" and n == 195, m.startswith("sca")))
    best = {}
    for c in COLS:
        hi = METRICS[c][1]
        cand = sorted({round(r[2][c], 6) for r in rows if r[4] and c in r[2]}, reverse=hi)
        best[c] = cand[:2]
    # HRGen-1K (AReA's Table 2): AReA's published row, then our runs; ranked within this block only, among the
    # published row and our FLUX.1-Krea-dev row on all 1000 prompts (the FLUX.1-dev rows use another checkpoint).
    hpub = hrgen_published(pub)
    hrows = [("hpub", "AReA, published (FLUX.1)", hpub["AReA"], 1000, True, False)] + \
        [r for r in hrgen_ours(ours) if r[0] == "hrgen_krea"]
    hrows = [(g, lab, {c: v for c, v in vals.items() if c in COLS}, n, rk, o) for g, lab, vals, n, rk, o in hrows]
    hbest = {}
    for c in COLS:
        cand = sorted({round(r[2][c], 6) for r in hrows if r[4] and c in r[2]}, reverse=METRICS[c][1])
        hbest[c] = cand[:1] if len(cand) > 1 else []  # bold the better of two ranked values, no underline
    lines = [
        "% Generated by results/published_compare/build.py; do not edit by hand.",
        "\\begin{table}[t]",
        "  \\caption{\\textbf{Published numbers at $4096^2$ next to our runs.} Aesthetic-4K, 195 prompts: AReA's "
        "Table~1 (HPS and MUSIQ from its Table~6; ``FLUX.1''), our runs with FLUX.1-Krea-dev, "
        "the checkpoint AReA's released code uses (28 steps, guidance 4.5), and our FLUX.1-dev protocol. Last block: "
        "HRGen-1K (all rows in Table~\\ref{tab:published_hrgen}). HPS: HPSv2 $\\times100$; MUSIQ on the full image; "
        "FID at $299^2$ against the 195 reference rows; patch FID is omitted because AReA does not state its crops. "
        "$^\\dagger$Fewer than all prompts; not ranked. \\textbf{Bold} / \\underline{underline}: best / second "
        "best among ranked rows (HRGen-1K: bold only); ours shaded. Key result: " + key_result(rows, hrows) + "}",
        "  \\label{tab:published_compare}",
        "  \\centering\\footnotesize",
        "  \\setlength{\\tabcolsep}{3.2pt}",
        "  \\begin{tabular}{@{}lr*{7}{r}@{}}",
        "    \\toprule",
        "     & & " + " & ".join(HEAD) + " \\\\",
        "    Method & $n$ & " + " & ".join(HEAD2) + " \\\\",
        "    \\midrule",
    ]
    titles = {"pub": "Published by AReA, single pass (FLUX.1)", "pubms": "Published by AReA, multi-stage (FLUX.1)",
              "krea": "Our runs, FLUX.1-Krea-dev, same prompts and metrics", "dev": "Our runs, FLUX.1-dev, shared sampler"}
    group = None
    for g, label, vals, n, rankable, is_ours in rows:
        if g != group:
            if group is not None:
                lines.append("    \\midrule")
            lines.append(f"    \\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{titles[g]}}}}} \\\\")
            group = g
        cells = []
        for c in COLS:
            if c not in vals:
                cells.append("--"); continue
            s = f"{vals[c]:.1f}" if (c == "clip" and g == "pubms" and label in prec1) else fmt(c, vals[c])
            if s.startswith("-"):
                s = "$-$" + s[1:]
            if rankable and best[c] and round(vals[c], 6) == best[c][0]:
                s = f"\\textbf{{{s}}}"
            elif rankable and len(best[c]) > 1 and round(vals[c], 6) == best[c][1]:
                s = f"\\underline{{{s}}}"
            cells.append(s)
        name = "\\textbf{\\sca{} (ours)}" if is_ours else label
        ntxt = f"{n}" + ("" if n == 195 else "$^\\dagger$")
        prefix = "\\rowcolor[HTML]{EAF4F1}" if is_ours else ""
        lines.append(f"    {prefix}{name} & {ntxt} & " + " & ".join(cells) + " \\\\")
    lines.append("    \\midrule")
    lines.append("    \\multicolumn{9}{@{}l}{\\emph{HRGen-1K, 1000 prompts without references: AReA's Table~2 and our FLUX.1-Krea-dev run}} \\\\")
    for g, label, vals, n, rankable, is_ours in hrows:
        lines.append("    " + row_tex(label, vals, n, rankable, is_ours, hbest, COLS, 1000))
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    PAPER.write_text("\n".join(lines))
    write_hrgen_appendix(pub, ours)


def hrgen_published(pub):
    out = {}
    for p in pub:
        if p["source"] == "AReA" and p["benchmark"] == "HRGen-1K" and p["res"] == "4096x4096":
            out.setdefault(p["method"], {}).setdefault(p["metric"], float(p["value"]))
    return out


def hrgen_ours(ours):
    """Our HRGen-1K rows: (group, label, values, n, rankable, ours_flag)."""
    rows = []
    for bid, m, label in (("hrgen_krea", "sca_krea", "\\sca{} (ours), FLUX.1-Krea-dev"),
                          ("hrgen_dev", "area", "AReA, our run, FLUX.1-dev"),
                          ("hrgen_dev", "sca_nb", "\\sca{} (ours), FLUX.1-dev")):
        vals = {k: ours[(bid, m, k)][0] for k in METRICS if (bid, m, k) in ours}
        if vals:
            n = min(ours[(bid, m, k)][1] for k in vals)
            rows.append((bid, label, vals, n, bid == "hrgen_krea" and n == 1000, m.startswith("sca")))
    return rows


def row_tex(label, vals, n, rankable, is_ours, best, cols, full_n):
    cells = []
    for c in cols:
        if c not in vals:
            cells.append("--"); continue
        s = fmt(c, vals[c])
        if s.startswith("-"):
            s = "$-$" + s[1:]
        if rankable and best[c] and round(vals[c], 6) == best[c][0]:
            s = f"\\textbf{{{s}}}"
        elif rankable and len(best[c]) > 1 and round(vals[c], 6) == best[c][1]:
            s = f"\\underline{{{s}}}"
        cells.append(s)
    name = f"\\textbf{{{label}}}" if is_ours else label
    ntxt = f"{n}" + ("" if n == full_n else "$^\\dagger$")
    prefix = "\\rowcolor[HTML]{EAF4F1}" if is_ours else ""
    return f"{prefix}{name} & {ntxt} & " + " & ".join(cells) + " \\\\"


HCOLS = ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "is", "musiq"]
HHEAD2 = ["Reward$\\uparrow$", "thetic$\\uparrow$", "CLIP$\\uparrow$", "Score$\\uparrow$", "HPS$\\uparrow$",
          "IS$\\uparrow$", "MUSIQ$\\uparrow$"]


def write_hrgen_appendix(pub, ours):
    """Appendix table: every row of AReA's HRGen-1K Table 2 next to our three runs, ranked among the published rows
    and our FLUX.1-Krea-dev row when it covers all 1000 prompts."""
    hpub = hrgen_published(pub)
    rows = [("hpub", n.replace("Base (full attention)", "Full attention"), hpub[n], 1000, True, False)
            for n in ["Base (full attention)", "UltraImage", "DyPE", "SigMa", "SEGA", "AReA"]] + hrgen_ours(ours)
    best = {}
    for c in HCOLS:
        cand = sorted({round(r[2][c], 6) for r in rows if r[4] and c in r[2]}, reverse=METRICS[c][1])
        best[c] = cand[:2] if len(cand) > 1 else []
    krea = next((r for r in rows if r[0] == "hrgen_krea"), None)
    area = hpub["AReA"]
    hi = [METRICS[c][0] for c in HCOLS if krea and c in krea[2] and c in area and (krea[2][c] >= area[c]) == METRICS[c][1]]
    lo = [METRICS[c][0] for c in HCOLS if krea and c in krea[2] and c in area and METRICS[c][0] not in hi]
    key = ("\\sca{} on FLUX.1-Krea-dev " + (f"exceeds AReA's published {join_words(hi)}" if hi else
           "does not exceed AReA's published numbers") + (f" and trails it on {join_words(lo)}" if lo else "")
           + (f" ({krea[3]} of 1000 prompts)" if krea and krea[3] != 1000 else "") + ".") if krea else "pending."
    lines = [
        "% Generated by results/published_compare/build.py; do not edit by hand.",
        "\\begin{table}[!htbp]",
        "  \\caption{\\textbf{HRGen-1K: AReA's published Table~2 next to our runs.} All 1000 prompts at $4096^2$, no "
        "reference images. Top: AReA's Table~2 (``FLUX.1'', which its released code runs as FLUX.1-Krea-dev). Bottom: "
        "\\sca{} with a Krea-calibrated table on FLUX.1-Krea-dev (28 steps, guidance 4.5), and our paired FLUX.1-dev "
        "runs of AReA and \\sca{}. HPS: HPSv2 $\\times100$; IS: Inception score; MUSIQ on the full image; "
        "$^\\dagger$fewer than 1000 prompts. \\textbf{Bold} / \\underline{underline}: best / second best among the "
        "published rows and our FLUX.1-Krea-dev row. Table~\\ref{tab:published_compare} holds the Aesthetic-4K "
        "comparison. Key result: " + key + "}",
        "  \\label{tab:published_hrgen}",
        "  \\centering\\footnotesize",
        "  \\setlength{\\tabcolsep}{3.2pt}",
        "  \\begin{tabular}{@{}lr*{7}{r}@{}}",
        "    \\toprule",
        "     & & Image- & Aes- &  & Pick- &  &  &  \\\\",
        "    Method & $n$ & " + " & ".join(HHEAD2) + " \\\\",
        "    \\midrule",
        "    \\multicolumn{9}{@{}l}{\\emph{Published by AReA (FLUX.1)}} \\\\",
    ]
    group = "hpub"
    for g, label, vals, n, rankable, is_ours in rows:
        if g != group and group == "hpub":
            lines += ["    \\midrule", "    \\multicolumn{9}{@{}l}{\\emph{Our runs}} \\\\"]
        group = g
        lines.append("    " + row_tex(label, vals, n, rankable, is_ours, best, HCOLS, 1000))
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    PAPER_HRGEN.write_text("\n".join(lines))


def join_words(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + (", and " if len(xs) > 2 else " and ") + xs[-1]


def key_result(rows, hrows=()):
    """Generated key result. The published numbers check the protocol; SCA is compared with our AReA run on the same
    checkpoint and prompts (paired intervals in Table tab:krea_6144); HRGen-1K has no AReA rerun on FLUX.1-Krea-dev,
    so there SCA is compared with AReA's published row."""
    pubarea = next(r for r in rows if r[0] == "pub" and r[1] == "AReA")[2]
    ours_area = next((r for r in rows if r[0] == "krea" and r[1] == "AReA"), None)
    ours_sca = next((r for r in rows if r[0] == "krea" and r[5]), None)
    shown = lambda c, v: float(fmt(c, v))
    unit = lambda c: 0.1 if FMT[c] == "{:.1f}" else 0.01
    def join(xs):
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + (", and " if len(xs) > 2 else " and ") + xs[-1]
    parts = []
    if ours_area:
        diff = {c: abs(shown(c, ours_area[2][c]) - shown(c, pubarea[c])) for c in COLS
                if c != "fid" and c in ours_area[2] and c in pubarea}
        close = [METRICS[c][0] for c in diff if diff[c] <= 0.1 + 1e-9]
        rest = [c for c in diff if diff[c] > 0.1 + 1e-9]
        if close:
            txt = ("our run of AReA's code on FLUX.1-Krea-dev reproduces its published " + join(close) + " within 0.1")
            if rest:
                txt += f" and its {join([METRICS[c][0] for c in rest])} within {max(diff[c] for c in rest):.2f}"
            parts.append(txt)
    if ours_sca and ours_area and ours_sca[3] == ours_area[3]:
        a, s_ = ours_area[2], ours_sca[2]
        cols = [c for c in COLS if c in a and c in s_]
        delta = {c: (shown(c, s_[c]) - shown(c, a[c])) * (1 if METRICS[c][1] else -1) for c in cols}
        hi = [METRICS[c][0] for c in cols if delta[c] > 0.1 + 1e-9]
        near = [METRICS[c][0] for c in cols if abs(delta[c]) <= 0.1 + 1e-9]
        lo = [METRICS[c][0] for c in cols if delta[c] < -0.1 - 1e-9]
        txt = "against that run \\sca{} is higher on " + join(hi) if hi else "against that run \\sca{}"
        if near:
            txt += ("," if hi else " is") + " within 0.1 on " + join(near)
        if lo:
            txt += f", and worse on {join(lo)}"
        parts.append(txt)
    hk = next((r for r in hrows if r[0] == "hrgen_krea"), None)
    ha = next((r for r in hrows if r[0] == "hpub"), None)
    if hk and ha:
        cols = [c for c in COLS if c in hk[2] and c in ha[2]]
        hi = [METRICS[c][0] for c in cols if (hk[2][c] >= ha[2][c]) == METRICS[c][1]]
        lo = [METRICS[c][0] for c in cols if METRICS[c][0] not in hi]
        txt = ("on HRGen-1K, where we did not rerun AReA on FLUX.1-Krea-dev, \\sca{} on that checkpoint "
               + ("exceeds every published AReA score" if hi and not lo else
                  f"exceeds AReA's published {join(hi)}" if hi else "does not exceed AReA's published numbers"))
        if lo:
            txt += f" and trails it on {join(lo)}"
        if hk[3] != 1000:
            txt += f" ({hk[3]} of 1000 prompts)"
        parts.append(txt)
    s = "; ".join(parts) + "."
    return s[0].upper() + s[1:]


if __name__ == "__main__":
    main()
