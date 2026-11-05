"""Compact, exact summary of one results/review1_<name>/ directory (read-only; stdlib only).
Prints paired per-image deltas (2000-draw paired bootstrap over prompts), set-level metrics with intervals,
set-level paired deltas, and duplication rates with Wilson intervals and exact McNemar p."""
import csv, glob, os, sys
LOWER = {"fid", "kid", "patch_fid", "patch_kid", "latency"}
def rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))
def g(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v:.{nd}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:.1f}"
def sig(lo, hi, better_lower):
    lo, hi = float(lo), float(hi)
    if lo > 0: return "worse" if better_lower else "BETTER"
    if hi < 0: return "BETTER" if better_lower else "worse"
    return "n.s."
d = sys.argv[1]
print("#", d)
for p in sorted(glob.glob(f"{d}/*_vs_*.csv")):
    r = rows(p)
    if not r: continue
    print(f"\n## paired {r[0]['method']} - {r[0]['reference']} (n={r[0]['n_prompts']}, {r[0]['res']})")
    for x in r:
        m = x["metric"]
        print(f"  {m:10s} {g(x['delta']):>10s} [{g(x['ci_low'])}, {g(x['ci_high'])}]  {sig(x['ci_low'], x['ci_high'], m in LOWER)}")
for name in ("base_summary.csv", "extra_summary.csv", "base/summary.csv", "corrected_fidp/summary.csv", "corrected_fidp_ci/summary.csv"):
    p = f"{d}/{name}"
    if not os.path.exists(p): continue
    print(f"\n## {name}")
    by = {}
    for x in rows(p):
        by.setdefault(x["metric"], []).append(x)
    for m, xs in by.items():
        s = "; ".join(f"{x['method']} {g(x['value'])}" + (f" [{g(x['ci_low'])}, {g(x['ci_high'])}]" if x.get("ci_low") else "") for x in xs)
        print(f"  {m:10s} {s}")
for name in ("corrected_fidp_ci/paired_deltas.csv", "base/paired_deltas.csv", "corrected_fidp/paired_deltas.csv", "base_paired_deltas.csv", "extra_paired_deltas.csv"):
    p = f"{d}/{name}"
    if not os.path.exists(p): continue
    r = rows(p)
    if not r or "reference_method" not in r[0]:
        print(f"\n## {name}: columns {list(r[0].keys()) if r else []}"); continue
    print(f"\n## {name} (method minus reference)")
    for x in r:
        ci = f" [{g(x['ci_low'])}, {g(x['ci_high'])}] {sig(x['ci_low'], x['ci_high'], x['metric'] in LOWER)}" if x.get("ci_low") else " (no interval)"
        print(f"  {x['metric']:10s} {x['method']} - {x['reference_method']}: {g(x['value'])}{ci}")
for p in sorted(glob.glob(f"{d}/dup_rates_*.csv")):
    print(f"\n## {os.path.basename(p)}")
    for x in rows(p):
        extra = (f"  diff vs ref {g(x['diff_vs_ref'])} [{g(x['diff_lo95'])}, {g(x['diff_hi95'])}], McNemar p={x['mcnemar_p']} (n_paired={x['n_paired']})"
                 if x.get("diff_vs_ref") not in (None, "") else "  (reference)")
        print(f"  {x['method']:14s} {x['dup']}/{x['n']} = {g(x['rate'])} [{g(x['lo95'])}, {g(x['hi95'])}]{extra}")
