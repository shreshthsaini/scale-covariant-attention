#!/usr/bin/env python3
"""Paired per-image deltas against AReA on the 46 ablation prompts for published SCA and each duplication variant
(variant images scored by eval/dup46_<label>, AReA and SCA by eval/a4k195_main; per-image metrics are deterministic
per image, checked on SCA's shared images). 95% bootstrap intervals over prompts (4000 draws); '+' marks a win whose
interval excludes 0. Usage: dup_vs_area.py LABEL [LABEL ...]"""

import csv
import json
from pathlib import Path
import random
import sys

S = Path("/scratch/09032/saini_2/sca")
MET = ["ir", "aesthetic", "hpsv2", "hpsv21", "pickscore", "clip", "clipiqa", "arniqa", "musiq", "musiq_crop1024"]
ids = [json.loads(l)["id"] for l in open(S / "abl50.jsonl")]
dev = {json.loads(l)["id"] for l in open(S / "dev4.jsonl")}
ids46 = [i for i in ids if i not in dev]


def load(edir):
    out = {}
    for f in (edir / "extra/extra_per_image.csv", edir / "base/per_image.csv"):
        if f.exists():
            for r in csv.DictReader(open(f)):
                for m in MET:
                    if r.get(m) not in (None, ""):
                        out[(r["method"], r["prompt_id"], m)] = float(r[m])
    return out


def ci(d, draws=4000):
    rng = random.Random(0)
    b = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(draws))
    return sum(d) / len(d), b[int(0.025 * draws)], b[int(0.975 * draws) - 1]


main = load(S / "eval/a4k195_main")
rows = [("sca_nb", main, "sca_nb")]
for label in sys.argv[1:]:
    v = load(S / f"eval/dup46_{label}")
    chk = [abs(v[("sca_nb", i, "ir")] - main[("sca_nb", i, "ir")]) for i in ids46 if ("sca_nb", i, "ir") in v and ("sca_nb", i, "ir") in main]
    if chk and max(chk) > 1e-3:
        print(f"WARNING {label}: SCA per-image IR differs between evaluations by up to {max(chk):.4f}")
    rows.append((label, v, label))
print("vs AReA, 46 prompts:", "  ".join(MET))
for name, src, m_name in rows:
    parts = []
    for m in MET:
        d = [src[(m_name, i, m)] - main[("area", i, m)] for i in ids46 if (m_name, i, m) in src and ("area", i, m) in main]
        if not d:
            parts.append(f"{m} n/a")
            continue
        mu, lo, hi = ci(d)
        parts.append(f"{m} {mu:+.3g}[{lo:+.3g},{hi:+.3g}]{'+' if lo > 0 else ('-' if hi < 0 else '')}")
    print(f"{name}: " + "; ".join(parts))
