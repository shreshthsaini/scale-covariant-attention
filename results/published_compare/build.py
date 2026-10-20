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
       "musiq": "{:.1f}", "fid": "{:.1f}"}


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
    for name in ["HiFlow", "I-Max", "ScaleDiff"]:
        rows.append(("pubms", name, pubv[name], 195, True, False))
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
    lines = [
        "% Generated by results/published_compare/build.py; do not edit by hand.",
        "\\begin{table}[t]",
        "  \\caption{\\textbf{Published Aesthetic-4K numbers at $4096^2$ next to our runs.} Top: AReA's Table~1 (HPS, "
        "MUSIQ, and the multi-stage rows from its Table~6), all 195 prompts, ``FLUX.1''. Middle: our runs on the same "
        "prompts and references with FLUX.1-Krea-dev, the checkpoint AReA's released code uses (28 steps, guidance 4.5). "
        "Bottom: our FLUX.1-dev protocol. HPS: HPSv2 $\\times100$; MUSIQ on the full image; FID at $299^2$ against the "
        "195 reference rows; AReA does not state its patch crops, so patch FID is omitted. $^\\dagger$Fewer than 195 "
        "prompts; not ranked. \\textbf{Bold} / \\underline{underline}: best / second best among ranked rows; ours "
        "shaded. Key result: " + key_result(rows) + "}",
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
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    PAPER.write_text("\n".join(lines))


def key_result(rows):
    pubarea = next(r for r in rows if r[0] == "pub" and r[1] == "AReA")[2]
    ours_area = next((r for r in rows if r[0] == "krea" and r[1] == "AReA"), None)
    ours_sca = next((r for r in rows if r[0] == "krea" and r[5]), None)
    shown = lambda c, v: float(fmt(c, v))
    def join(xs):
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + (", and " if len(xs) > 2 else " and ") + xs[-1]
    parts = []
    if ours_area:
        close = [METRICS[c][0] for c in COLS if c != "fid" and c in ours_area[2] and c in pubarea
                 and abs(shown(c, ours_area[2][c]) - shown(c, pubarea[c])) <= 0.1 + 1e-9]
        if close:
            parts.append("on FLUX.1-Krea-dev our run of AReA's released code reproduces its published " + join(close)
                         + " within 0.1, so the protocol matches")
    if ours_sca:
        cols = [c for c in COLS if c in ours_sca[2] and c in pubarea and (c != "fid" or ours_sca[3] == 195)]
        hi = [METRICS[c][0] for c in cols if (ours_sca[2][c] >= pubarea[c]) == METRICS[c][1]]
        lo = [METRICS[c][0] for c in cols if METRICS[c][0] not in hi]
        txt = "\\sca{} on that checkpoint " + (f"exceeds AReA's published {join(hi)}" if hi else "does not exceed AReA's published numbers")
        if lo:
            txt += f" and trails it on {join(lo)}"
        if ours_sca[3] != 195:
            txt += f" ({ours_sca[3]} of 195 prompts; no FID)"
        parts.append(txt)
    s = "; ".join(parts) + "."
    return s[0].upper() + s[1:]


if __name__ == "__main__":
    main()
