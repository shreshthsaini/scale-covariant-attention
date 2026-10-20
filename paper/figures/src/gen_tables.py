"""Generate appendix tables from archived CSVs and corrected distribution checks.

Read-only on results. Corrected patch points never inherit historical intervals.
Image-identity cluster intervals supersede row intervals where archived; missing
corrected patch intervals are omitted, and missing corrected points appear as n/a.
"""
import csv
import sys
from pathlib import Path

R = Path(sys.argv[1])
OUT = Path(sys.argv[2])

# metric key -> (label, scale, decimals, lower_is_better)
MET = [
    ("ir", "ImageReward", 1, 3, False),
    ("aesthetic", "Aesthetic", 1, 2, False),
    ("clip", "CLIP score", 1, 2, False),
    ("pickscore", "PickScore", 1, 2, False),
    ("hpsv2", "HPSv2 ($\\times100$)", 100, 2, False),
    ("hpsv21", "HPSv2.1 ($\\times100$)", 100, 2, False),
    ("musiq", "MUSIQ", 1, 1, False),
    ("musiq_crop1024", "MUSIQ$_\\text{c}$", 1, 1, False),
    ("clipiqa", "CLIP-IQA ($\\times100$)", 100, 1, False),
    ("arniqa", "ARNIQA ($\\times100$)", 100, 1, False),
    ("is", "Inception score", 1, 2, False),
    ("fid", "FID$\\downarrow$", 1, 1, True),
    ("kid", "KID$\\downarrow$ ($\\times10^3$)", 1000, 2, True),
    ("patch_fid", "FID$_\\text{p}\\downarrow$", 1, 1, True),
    ("patch_kid", "KID$_\\text{p}\\downarrow$ ($\\times10^3$)", 1000, 2, True),
    ("seconds", "Seconds per image$^\\ddagger\\downarrow$", 1, 1, True),
]
METD = {m[0]: m for m in MET}


def rows(name):
    p = R / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def summary(prefix):
    """{(method, metric): value} from <prefix>_base_summary.csv and _extra_summary.csv."""
    d = {}
    for r in rows(prefix + "_base_summary.csv") + rows(prefix + "_extra_summary.csv"):
        d[(r["method"], r["metric"])] = float(r["value"])
    return d


def deltas(files):
    """{(method, metric): (delta, lo, hi)} from any paired-delta CSV layout."""
    d = {}
    for f in files:
        for r in rows(f):
            v = r.get("value", r.get("delta"))
            d[(r["method"], r["metric"])] = (float(v), float(r["ci_low"]), float(r["ci_high"]))
    return d


def corrected_distributions(sd, dd, groups, reference):
    """Overlay corrected points and available image-cluster CIs for one comparison."""
    distribution_metrics = ("fid", "kid", "patch_fid", "patch_kid")
    # Never silently reuse the historical patch bank or its confidence intervals.
    for d in (sd, dd):
        for key in list(d):
            if key[1] in ("patch_fid", "patch_kid"):
                del d[key]
    corrected = {}
    clustered = []
    for group in groups:
        for metric in distribution_metrics:
            stem = "evidence/distributions/checks/" + group
            for row in rows(stem + "_rows_rows_" + metric + "_0.csv"):
                corrected[(row["method"], metric)] = float(row["value"])
            for path in sorted(R.glob(stem + "_rows_images_" + metric + "_1000*.csv")):
                clustered.extend(rows(str(path.relative_to(R))))
    sd.update(corrected)
    for (method, metric), value in corrected.items():
        if method == reference or (reference, metric) not in sd:
            continue
        delta = value - sd[(reference, metric)]
        if metric.startswith("patch_"):
            dd[(method, metric)] = (delta, None, None)
        elif (method, metric) in dd:
            _, low, high = dd[(method, metric)]
            dd[(method, metric)] = (delta, low, high)
    for row in clustered:
        if row["reference_method"] != reference or not row["ci_low"]:
            continue
        dd[(row["method"], row["metric"])] = tuple(
            float(row[key]) for key in ("paired_delta", "ci_low", "ci_high"))


def f(x, m, signed=False):
    _, _, sc, dec, _ = METD[m]
    s = f"{x * sc:+.{dec}f}" if signed else f"{x * sc:.{dec}f}"
    return "$" + s.replace("-", "{-}") + "$" if signed or x < 0 else s


def dcell(t, m):
    if t is None:
        return "n/a"
    v, lo, hi = t
    if lo is None or hi is None:
        return f(v, m, True)
    sig = lo > 0 or hi < 0
    lib = METD[m][4]
    better = (v < 0) if lib else (v > 0)
    core = f(v, m, True)
    if sig:
        core = "\\up{" + core + "}" if better else "\\dn{" + core + "}"
    return core + " {\\tiny[" + f(lo, m, True) + ", " + f(hi, m, True) + "]}"


def vcell(d, k, m):
    return f(d[k], m) if k in d else "n/a"


out = []
W = out.append


def table(caption, label, groups, metrics, absolute=True):
    """groups: list of (header, method, sumdict, deltadict, refname)."""
    ncol = 1 + sum((2 if absolute else 1) for _ in groups)
    W("\\begin{table}[h]")
    W("  \\caption{" + caption + "}")
    W("  \\label{" + label + "}")
    W("  \\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}")
    W("  \\resizebox{\\linewidth}{!}{%")
    W("  \\begin{tabular}{l" + "c" * (ncol - 1) + "}")
    W("    \\toprule")
    hdr = ["Metric"]
    for h, *_ in groups:
        hdr += ([h + " value", "$\\Delta$ [95\\% CI]"] if absolute else [h])
    W("    " + " & ".join(hdr) + " \\\\")
    W("    \\midrule")
    for m in metrics:
        cells = [METD[m][1]]
        for h, meth, sd, dd, ref in groups:
            if absolute:
                cells.append(vcell(sd, (meth, m), m))
            cells.append(dcell(dd.get((meth, m)), m) if meth != ref else "reference")
        W("    " + " & ".join(cells) + " \\\\")
    W("    \\bottomrule")
    W("  \\end{tabular}}")
    W("\\end{table}")
    W("")


ALL = [m[0] for m in MET]

# ---- 4096^2: AReA reference; SCA final, SCA + band, full attention
s_nb, s_b, s_f = summary("final4k_nb"), summary("final4k"), summary("final4k_full")
d_nb = deltas(["final4k_nb_base_paired_deltas.csv", "final4k_nb_extra_paired_deltas.csv"])
d_b = deltas(["final4k_base_paired_deltas.csv", "final4k_extra_paired_deltas.csv"])
d_f = deltas(["final4k_full_base_paired_deltas.csv", "final4k_full_extra_paired_deltas.csv"])
corrected_distributions(s_nb, d_nb, ["final4k_nb_base"], "area")
corrected_distributions(s_b, d_b, ["final4k_base"], "area")
corrected_distributions(s_f, d_f, ["final4k_full_base"], "area")
area4 = {("area", m): v for (meth, m), v in s_nb.items() if meth == "area"}
W("\\begin{table}[h]")
W("  \\caption{\\textbf{All metrics at $4096^2$}, 191 prompts, paired against AReA (released code). $\\Delta$ is method minus AReA with its 95\\% paired bootstrap interval (1000 draws); \\up{blue} / \\dn{orange}: significantly better / worse than AReA. Set-level metrics (FID, KID and their patch versions, Inception score) are computed on the whole set; the final SCA FID/KID and patch-FID/KID intervals use reference-image clusters; other FID/KID intervals retain the historical reference-row bootstrap unless a corrected image-cluster interval is available. Inception-score intervals resample only generated images. Corrected patch values without recomputed intervals have no significance marks. $^\\ddagger$Mean campaign wall-clock on shared nodes, not a clean latency measurement (Table~\\ref{tab:latency} gives clean latency). Sources: \\texttt{results/final4k\\_nb\\_*}, \\texttt{final4k\\_*}, \\texttt{final4k\\_full\\_*}.}")
W("  \\label{tab:app_4k}")
W("  \\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}")
W("  \\resizebox{\\linewidth}{!}{%")
W("  \\begin{tabular}{lccccccc}")
W("    \\toprule")
W("    Metric & AReA & \\sca{} final & $\\Delta$ [95\\% CI] & \\sca{} + band & $\\Delta$ [95\\% CI] & Full attention & $\\Delta$ [95\\% CI] \\\\")
W("    \\midrule")
for m in ALL:
    c = [METD[m][1], vcell(area4, ("area", m), m),
         vcell(s_nb, ("sca_nb", m), m), dcell(d_nb.get(("sca_nb", m)), m),
         vcell(s_b, ("sca", m), m), dcell(d_b.get(("sca", m)), m),
         vcell(s_f, ("full", m), m), dcell(d_f.get(("full", m)), m)]
    W("    " + " & ".join(c) + " \\\\")
W("    \\bottomrule")
W("  \\end{tabular}}")
W("\\end{table}")
W("")

# ---- non-square
W("\\begin{table}[h]")
W("  \\caption{\\textbf{All metrics at the non-square sizes}, 191 prompts each, \\sca{} final paired against AReA (released code). Conventions as in Table~\\ref{tab:app_4k}. Corrected patch intervals use reference-image clusters where available; absent intervals are omitted. Sources: \\texttt{results/final\\_2048x4096\\_*}, \\texttt{final\\_2048x6144\\_*}.}")
W("  \\label{tab:app_wide}")
W("  \\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}")
W("  \\resizebox{\\linewidth}{!}{%")
W("  \\begin{tabular}{lcccccc}")
W("    \\toprule")
W("    & \\multicolumn{3}{c}{$2048\\times4096$, factors (2, 4)} & \\multicolumn{3}{c}{$2048\\times6144$, factors (2, 6)} \\\\")
W("    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
W("    Metric & AReA & \\sca{} & $\\Delta$ [95\\% CI] & AReA & \\sca{} & $\\Delta$ [95\\% CI] \\\\")
W("    \\midrule")
sw = {r: summary(f"final_{r}") for r in ("2048x4096", "2048x6144")}
dw = {r: deltas([f"final_{r}_base_paired_deltas.csv", f"final_{r}_extra_paired_deltas.csv"]) for r in sw}
for res in sw:
    corrected_distributions(sw[res], dw[res], [f"final_{res}_base"], "area")
for m in ALL:
    c = [METD[m][1]]
    for r in ("2048x4096", "2048x6144"):
        c += [vcell(sw[r], ("area", m), m), vcell(sw[r], ("sca_nb", m), m), dcell(dw[r].get(("sca_nb", m)), m)]
    W("    " + " & ".join(c) + " \\\\")
W("    \\bottomrule")
W("  \\end{tabular}}")
W("\\end{table}")
W("")

# ---- ps2 at 2048x4096 (46 prompts), reference sca_nb with the scale-4 table
dps = deltas(["ps2_abl_ps2_vs_scanb.csv", "ps2_area_vs_scanb.csv"])
W("\\begin{table}[h]")
W("  \\caption{\\textbf{Calibration table at $2048\\times4096$}, 46 prompts, paired against \\sca{} final with the scale-4 table. Sources: \\texttt{results/ps2\\_*}.}")
W("  \\label{tab:app_ps2}")
W("  \\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}")
W("  \\begin{tabular}{lcc}")
W("    \\toprule")
W("    Metric & \\sca{}, scale-2 table & AReA (released code) \\\\")
W("    \\midrule")
for m in PM:
    W("    " + " & ".join([METD[m][1], dcell(dps.get(("abl_ps2", m)), m), dcell(dps.get(("area", m)), m)]) + " \\\\")
W("    \\bottomrule")
W("  \\end{tabular}")
W("\\end{table}")
W("")

# ---- ablations, final (no band) setting
PM = ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21", "musiq", "musiq_crop1024", "clipiqa", "arniqa"]
snb = summary("ablnb")
dnb = deltas([f"ablnb_{x}_deltas.csv" for x in ("abl_localonly_nb", "abl_uniform_nb", "abl_layer_nb", "abl_globalonly", "abl_mean_nb", "abl_v0")])
corrected_distributions(snb, dnb, ["ablnb_base", "localnb_base"], "sca_nb")
var = [("abl_localonly_nb", "$\\pi=1$"), ("abl_uniform_nb", "shared $\\pi(\\sigma)$"), ("abl_layer_nb", "per-layer"),
       ("abl_globalonly", "$\\pi=0$"), ("abl_mean_nb", "mean pool"), ("abl_v0", "v0, no gate")]
W("\\begin{table}[h]")
W("  \\caption{\\textbf{Ablation intervals for Table~\\ref{tab:ablation}}, 46 prompts at $4096^2$, paired against \\sca{} final (2000 bootstrap draws). The first column gives \\sca{} final's absolute value; the last row gives patch FID per variant (set-level, no paired interval). Sources: \\texttt{results/ablnb\\_*}.}")
W("  \\label{tab:app_abl}")
W("  \\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}")
W("  \\resizebox{\\linewidth}{!}{%")
W("  \\begin{tabular}{lc" + "c" * len(var) + "}")
W("    \\toprule")
W("    Metric & \\sca{} final & " + " & ".join(v for _, v in var) + " \\\\")
W("    \\midrule")
for m in PM:
    c = [METD[m][1], vcell(snb, ("sca_nb", m), m)] + [dcell(dnb.get((k, m)), m) for k, _ in var]
    W("    " + " & ".join(c) + " \\\\")
W("    \\midrule")
W("    " + " & ".join([METD["patch_fid"][1], vcell(snb, ("sca_nb", "patch_fid"), "patch_fid")] + [vcell(snb, (k, "patch_fid"), "patch_fid") for k, _ in var]) + " \\\\")
W("    \\bottomrule")
W("  \\end{tabular}}")
W("\\end{table}")
W("")

# ---- ablations around the band guard (reference: SCA + band)
sab = summary("abl")
dab = deltas(["abl_base_paired_deltas.csv", "abl_extra_paired_deltas.csv", "route_base_paired_deltas.csv", "route_extra_paired_deltas.csv"])
corrected_distributions(sab, dab, ["ablfinal_base", "route_base"], "sca")
varb = [("abl_noband", "no band"), ("abl_nogate", "no gate"), ("abl_uniform", "shared $\\pi(\\sigma)$"), ("abl_layer", "per-layer"),
        ("abl_localonly", "$\\pi=1$"), ("abl_route50", "route 0.5"), ("abl_route30", "route 0.3"), ("abl_mean", "mean pool"),
        ("abl_globalonly", "$\\pi=0$"), ("abl_v0", "v0")]
varb = [v for v in varb if any((v[0], m) in dab for m in PM)]
for part, chunk in enumerate((varb[:5], varb[5:])):
    if not chunk:
        continue
    W("\\begin{table}[h]")
    cap = ("\\textbf{Intervals for the band-guard ablations} (Table~\\ref{tab:ablation_band} and further variants), 46 prompts at $4096^2$, paired against \\sca{} + band guard, whose absolute values form the first column. Corrected patch FID is shown without intervals. Sources: \\texttt{results/abl\\_*}, \\texttt{route\\_*}."
           if part == 0 else "\\textbf{Band-guard ablations, continued.} Conventions as in Table~\\ref{tab:app_abl_band}.")
    W("  \\caption{" + cap + "}")
    W("  \\label{tab:app_abl_band" + ("" if part == 0 else "2") + "}")
    W("  \\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}")
    W("  \\resizebox{\\linewidth}{!}{%")
    W("  \\begin{tabular}{lc" + "c" * len(chunk) + "}")
    W("    \\toprule")
    W("    Metric & Reference & " + " & ".join(v for _, v in chunk) + " \\\\")
    W("    \\midrule")
    for m in PM + ["patch_fid"]:
        c = [METD[m][1], vcell(sab, ("sca", m), m)] + [dcell(dab.get((k, m)), m) for k, _ in chunk]
        W("    " + " & ".join(c) + " \\\\")
    W("    \\bottomrule")
    W("  \\end{tabular}}")
    W("\\end{table}")
    W("")

OUT.write_text("\n".join(out) + "\n")
print("wrote", OUT, len(out), "lines")
