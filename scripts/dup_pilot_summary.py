#!/usr/bin/env python3
"""One-line-per-variant summary of a duplication pilot or 46-prompt run: detector rates (people/animals and any
subject), paired metric deltas against sca_nb (extra and base metric sets, 95% bootstrap intervals; * marks an
interval excluding 0) and median seconds per image (first image of a task includes compilation, so the median
is used). Usage: dup_pilot_summary.py EVAL_DIR [OUT_ROOT]"""

import csv
import glob
import json
from pathlib import Path
import statistics
import sys

E = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/scratch/09032/saini_2/sca/out/dup")
MET = ["ir", "hpsv21", "pickscore", "aesthetic", "hpsv2", "clip", "clipiqa", "arniqa", "musiq", "patch_fid"]


def rates(f):
    out = {}
    if f.exists():
        for r in csv.DictReader(open(f)):
            out[r["method"]] = f"{int(r['dup'])}/{r['n']}"
    return out


people, anyr = rates(E / "dup_rates_people.csv"), rates(E / "dup_rates.csv")
deltas = {}
for f in (E / "extra/extra_paired_deltas.csv", E / "base/paired_deltas.csv"):
    if f.exists():
        for r in csv.DictReader(open(f)):
            v, lo, hi = float(r["value"]), r.get("ci_low"), r.get("ci_high")
            sig = lo not in (None, "") and hi not in (None, "") and (float(lo) > 0 or float(hi) < 0)
            deltas.setdefault(r["method"], {})[r["metric"]] = f"{v:+.3g}{'*' if sig else ''}"
times = {}
for d in sorted(glob.glob(str(OUT / "*/4096x4096"))) + ["/scratch/09032/saini_2/sca/out/main/sca_nb/4096x4096"]:
    t = [json.load(open(j)).get("seconds") for j in glob.glob(d + "/*.json")]
    t = [x for x in t if x]
    if t:
        times[d.split("/")[-2]] = statistics.median(t)
base_t = times.get("dup_base") or times.get("sca_nb")
methods = sorted(set(people) | set(anyr) | set(deltas))
print("method".ljust(20), "people any   sec(+%)  " + " ".join(m[:9].ljust(9) for m in MET))
for m in methods:
    t = times.get(m)
    ts = f"{t:6.1f}({(t / base_t - 1) * 100:+.0f}%)" if t and base_t else "   n/a      "
    print(m.ljust(20), people.get(m, "-").ljust(6), anyr.get(m, "-").ljust(6), ts,
          " ".join(deltas.get(m, {}).get(k, "-").ljust(9) for k in MET))
