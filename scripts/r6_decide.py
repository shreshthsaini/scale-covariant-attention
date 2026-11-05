#!/usr/bin/env python3
"""Round-6 acceptance check for 46-prompt candidates (eval/r6_46_<label>, slurm/round6_tasks.py).

The bar (lead, 2026-11-05): a fix passes only if (1) its human-corrected people/animal duplication rate is at or below
AReA's on the same prompts, (2) it has no significant paired loss versus sca_v2 on any quality metric (95% bootstrap
interval entirely on the losing side), (3) every metric on which sca_v2 significantly beats AReA stays significantly
better than AReA, and (4) its median time is at most 1.25 x sca_v2's.

Human-corrected rate on the P eligible prompts: TP (human-confirmed detector flags) + U x miss, U the unflagged prompts,
miss the method's unflagged duplicate rate from Step 1 (results/dup_audit_20261027/human_rates_95.csv; candidates take
sca_v2's). sca_v2 and AReA use their Step 1 labels where an image was labeled. Candidate flags are confirmed from
results/dup_audit_20261027/human_labels_r6.csv (label, prompt_id, human_final); without it the detector flag stands.
Usage: r6_decide.py LABEL [LABEL ...]
"""

import csv
import json
from pathlib import Path
import random
import sys

S = Path("/scratch/09032/saini_2/sca")
ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/dup_audit_20261027"
LOWER = {"patch_fid", "patch_kid", "fid", "kid"}


def read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def step1():
    lab = {(r["method"], r["prompt_id"]): r["human_final"] for r in read(A / "human_labels_95.csv")}
    miss = {r["method"]: float(r["miss_rate"]) for r in read(A / "human_rates_95.csv") if r["label"] == "primary"}
    return lab, miss


def human_rate(flags, method, labels, miss):
    """flags {prompt_id: 0/1} -> (TP, flagged, unflagged, labeled unflagged dups, estimate)."""
    tp = sum(1 for p, f in flags.items() if f and labels.get((method, p), "yes") == "yes")
    unl = [p for p, f in flags.items() if not f]
    known = [p for p in unl if (method, p) in labels]
    k_known = sum(labels[(method, p)] == "yes" for p in known)
    est = tp + k_known + (len(unl) - len(known)) * miss
    return tp, sum(flags.values()), len(unl), k_known, est / max(len(flags), 1)


def boot_ci(d, draws=4000, seed=0):
    rng = random.Random(seed)
    if len(d) < 2:
        return float("nan"), float("nan")
    m = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(draws))
    return m[int(0.025 * draws)], m[int(0.975 * draws) - 1]


def deltas(E, a, b):
    out = {r["metric"]: r for r in read(E / f"paired/{a}_vs_{b}.csv")}
    for r in read(E / "base/paired_deltas.csv"):
        if r.get("method") == a and r.get("reference_method") == b and r["metric"] in LOWER:
            out[r["metric"]] = {"delta": r["value"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
    return out


def sig(r, metric):
    """+1 significantly better, -1 significantly worse, 0 unresolved (lower is better for FID/KID)."""
    try:
        lo, hi = float(r["ci_low"]), float(r["ci_high"])
    except (TypeError, ValueError, KeyError):
        return 0
    if metric in LOWER:
        lo, hi = -hi, -lo
    return 1 if lo > 0 else -1 if hi < 0 else 0


def main():
    labels, miss = step1()
    cand = {(r["label"], r["prompt_id"]): r["human_final"] for r in read(A / "human_labels_r6.csv")}
    for label in sys.argv[1:]:
        E = S / f"eval/r6_46_{label}"
        if not (E / "DONE").exists():
            print(f"{label}: not finished ({E}/DONE missing)")
            continue
        fl = {}
        for r in read(E / "dup_per_image_people.csv"):
            fl.setdefault(r["method"], {})[r["prompt_id"]] = int(r["dup"])
        lab_all = dict(labels)
        lab_all.update({(label, p): v for (l, p), v in cand.items() if l == label})
        res = {}
        for m in ("sca_v2", label, "area"):
            mm = "sca_v2" if m == label else m
            res[m] = human_rate(fl.get(m, {}), m, lab_all, miss.get(mm, 0.0))
        checked = all((label, p) in cand for p, f in fl.get(label, {}).items() if f)
        d = deltas(E, label, "sca_v2")
        losses = [f"{m} {float(r['delta']):+.4g} [{float(r['ci_low']):+.4g},{float(r['ci_high']):+.4g}]"
                  for m, r in sorted(d.items()) if sig(r, m) < 0]
        gains = [m for m, r in sorted(d.items()) if sig(r, m) > 0]
        base = deltas(E, "sca_v2", "area")
        mine = deltas(E, label, "area")
        wins = [m for m, r in base.items() if sig(r, m) > 0]
        lost = [m for m in wins if sig(mine.get(m, {}), m) <= 0]
        t = {ln.split()[0]: float(ln.split()[-1]) for ln in open(E / "timing.txt") if ln.split()[-1] != "None"}
        ratio = t.get(label, float("nan")) / t.get("sca_v2", float("nan"))
        hr, ha, hv = res[label][4], res["area"][4], res["sca_v2"][4]
        ok = (hr <= ha + 1e-9) and not losses and not lost and ratio <= 1.25
        P = len(fl.get(label, {}))
        print(f"{label}: {'PASS' if ok else 'FAIL'} | people/animal on {P} prompts, detector flags {res[label][1]} "
              f"(sca_v2 {res['sca_v2'][1]}, AReA {res['area'][1]}); human-confirmed {res[label][0]}"
              f"{'' if checked else ' (flags not all human-checked yet)'}; human-corrected {hr:.1%} vs AReA {ha:.1%} "
              f"vs sca_v2 {hv:.1%} | significant losses vs sca_v2: {losses or 'none'}; gains: {gains or 'none'} | wins over "
              f"AReA kept {len(wins) - len(lost)}/{len(wins)} (lost: {lost or 'none'}) | time {t.get(label)} s vs "
              f"{t.get('sca_v2')} s = x{ratio:.3f}")
        # softening on prompts both methods leave clean (detector, any subject)
        anyf = {}
        for r in read(E / "dup_per_image_any.csv"):
            anyf.setdefault(r["method"], {})[r["prompt_id"]] = int(r["dup"])
        per = {}
        for f in (E / "extra/extra_per_image.csv", E / "base/per_image.csv"):
            for r in read(f):
                for k, v in r.items():
                    try:
                        per[(r["method"], r["prompt_id"], k)] = float(v)
                    except (TypeError, ValueError):
                        pass
        clean = [p for p, f in anyf.get(label, {}).items() if not f and not anyf.get("sca_v2", {}).get(p, 1)]
        soft = []
        for m in ("ir", "aesthetic", "hpsv2", "hpsv21", "pickscore", "clip", "clipiqa", "arniqa", "musiq", "musiq_crop1024"):
            dd = [per[(label, p, m)] - per[("sca_v2", p, m)] for p in clean if (label, p, m) in per and ("sca_v2", p, m) in per]
            if dd:
                lo, hi = boot_ci(dd)
                if hi < 0:
                    soft.append(f"{m} {sum(dd) / len(dd):+.4g} [{lo:+.4g},{hi:+.4g}]")
        print(f"  {label}: softening on {len(clean)} prompts clean in both (any subject): {soft or 'none'}")


if __name__ == "__main__":
    main()
