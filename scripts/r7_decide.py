#!/usr/bin/env python3
"""Round-7 acceptance check: the round-6 bar (scripts/r6_decide.py) on any evaluation directory, plus the fixed / clean
split of the paired metric deltas (.agent/notes/duplication_fix_round7.md).

Usage: r7_decide.py [--frontier OUT.csv] [--split OUT.csv] EVAL_DIR:LABEL [EVAL_DIR:LABEL ...]
  EVAL_DIR is eval/r7_46_<label>, eval/r6_46_<label> or eval/r7_195_<label> (DONE present); the directory holds
  dup_per_image_{people,any}.csv, extra/extra_per_image.csv, base/per_image.csv, paired/<a>_vs_<b>.csv and timing.txt.
Bar: human-corrected people/animal rate <= AReA's on the same prompts, no significant paired loss versus sca_v2 (95%
interval entirely on the losing side; set-level FID/KID/patch FID/patch KID from corrected_fidp_ci when present, else
base/paired_deltas.csv), every metric where sca_v2 significantly beats AReA still significantly better, median time
<= 1.25 x sca_v2's. Human labels of candidate flags: results/dup_audit_20261027/human_labels_r6.csv and
human_labels_r7.csv (label, prompt_id, human_final); an unlabeled flag counts as a duplicate.
Split (per-image metrics, LABEL - sca_v2, bootstrap 95% interval over prompts): fixed = people/animal prompts sca_v2
flags and LABEL does not; clean = prompts neither flags for any subject; new = prompts LABEL flags (people/animal) and
sca_v2 does not.
"""

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r6_decide as r6  # noqa: E402

A = r6.A
PER_IMAGE = ("ir", "aesthetic", "hpsv2", "hpsv21", "pickscore", "clip", "clipiqa", "arniqa", "musiq", "musiq_crop1024")


def human_labels():
    out = {}
    for name in ("human_labels_r6.csv", "human_labels_r7.csv"):
        for r in r6.read(A / name):
            out[(r["label"], r["prompt_id"])] = r["human_final"]
    return out


def set_deltas(E, a, b):
    """Paired per-image deltas plus set-level deltas (method a minus reference b) from the best available file."""
    out = {r["metric"]: r for r in r6.read(E / f"paired/{a}_vs_{b}.csv")}
    src = E / "corrected_fidp_ci/paired_deltas.csv"
    rows = r6.read(src) if src.exists() else r6.read(E / "base/paired_deltas.csv")
    for r in rows:
        if r.get("method") == a and r.get("reference_method") == b and r["metric"] in r6.LOWER:
            out[r["metric"]] = {"delta": r["value"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
    return out


def flags(E, kind):
    fl = {}
    for r in r6.read(E / f"dup_per_image_{kind}.csv"):
        fl.setdefault(r["method"], {})[r["prompt_id"]] = int(r["dup"])
    return fl


def per_image(E):
    per = {}
    for f in (E / "extra/extra_per_image.csv", E / "base/per_image.csv"):
        for r in r6.read(f):
            for k, v in r.items():
                try:
                    per[(r["method"], r["prompt_id"], k)] = float(v)
                except (TypeError, ValueError):
                    pass
    return per


def split(E, label, people, anyf, per):
    lab, ref = people.get(label, {}), people.get("sca_v2", {})
    la, ra = anyf.get(label, {}), anyf.get("sca_v2", {})
    sets = {"fixed": [p for p, f in ref.items() if f and not lab.get(p, 1)],
            "clean": [p for p, f in la.items() if not f and not ra.get(p, 1)],
            "new": [p for p, f in lab.items() if f and not ref.get(p, 1)]}
    rows = []
    for name, ids in sets.items():
        for m in PER_IMAGE:
            d = [per[(label, p, m)] - per[("sca_v2", p, m)] for p in ids if (label, p, m) in per and ("sca_v2", p, m) in per]
            if not d:
                continue
            lo, hi = r6.boot_ci(d)
            rows.append({"label": label, "subset": name, "n": len(d), "metric": m, "delta": sum(d) / len(d),
                         "ci_low": lo, "ci_high": hi, "sig": "loss" if hi < 0 else "gain" if lo > 0 else ""})
    return sets, rows


def decide(E, label, labels, miss, cand):
    people, anyf = flags(E, "people"), flags(E, "any")
    lab_all = dict(labels)
    lab_all.update({(l, p): v for (l, p), v in cand.items() if l == label})
    res = {}
    for m in ("sca_v2", label, "area"):
        res[m] = r6.human_rate(people.get(m, {}), m, lab_all, miss.get("sca_v2" if m == label else m, 0.0))
    checked = all((label, p) in cand for p, f in people.get(label, {}).items() if f)
    d = set_deltas(E, label, "sca_v2")
    losses = [f"{m} {float(r['delta']):+.4g} [{float(r['ci_low']):+.4g},{float(r['ci_high']):+.4g}]"
              for m, r in sorted(d.items()) if r6.sig(r, m) < 0]
    gains = [m for m, r in sorted(d.items()) if r6.sig(r, m) > 0]
    base, mine = set_deltas(E, "sca_v2", "area"), set_deltas(E, label, "area")
    wins = [m for m, r in base.items() if r6.sig(r, m) > 0]
    lost = [m for m in wins if r6.sig(mine.get(m, {}), m) <= 0]
    t = {}
    for ln in open(E / "timing.txt"):
        x = ln.split()
        if x and x[-1] != "None":
            t[x[0]] = float(x[-1])
    ratio = t.get(label, float("nan")) / t.get("sca_v2", float("nan"))
    hr, ha, hv = res[label][4], res["area"][4], res["sca_v2"][4]
    ok = (hr <= ha + 1e-9) and not losses and not lost and ratio <= 1.25
    sets, rows = split(E, label, people, anyf, per_image(E))
    P = len(people.get(label, {}))
    flagged = sorted(p for p, f in people.get(label, {}).items() if f)
    print(f"{label} [{E.name}]: {'PASS' if ok else 'FAIL'} | people/animal on {P} prompts: detector flags {res[label][1]} "
          f"(sca_v2 {res['sca_v2'][1]}, AReA {res['area'][1]}); human-confirmed {res[label][0]}"
          f"{'' if checked else ' (flags not all human-checked yet)'}; human-corrected {hr:.1%} vs AReA {ha:.1%} vs "
          f"sca_v2 {hv:.1%} | significant losses vs sca_v2 ({len(losses)}): {losses or 'none'}; gains: {gains or 'none'} "
          f"| wins over AReA kept {len(wins) - len(lost)}/{len(wins)} (lost: {lost or 'none'}) | time {t.get(label)} s "
          f"vs {t.get('sca_v2')} s = x{ratio:.3f}")
    print(f"  flagged prompts: {flagged}")
    print(f"  split: fixed {len(sets['fixed'])}, clean {len(sets['clean'])}, new {len(sets['new'])} prompts; "
          + "; ".join(f"{r['subset']} {r['metric']} {r['delta']:+.4g} [{r['ci_low']:+.4g},{r['ci_high']:+.4g}]"
                      for r in rows if r["sig"] == "loss"))
    row = {"label": label, "eval": E.name, "prompts": P, "detector_flags": res[label][1], "human_confirmed": res[label][0],
           "flags_checked": int(checked), "human_rate": round(hr, 4), "area_rate": round(ha, 4), "sca_v2_rate": round(hv, 4),
           "n_losses": len(losses), "losses": "; ".join(losses), "gains": "; ".join(gains),
           "wins_kept": f"{len(wins) - len(lost)}/{len(wins)}", "n_fixed": len(sets["fixed"]), "n_clean": len(sets["clean"]),
           "n_new": len(sets["new"]),
           "clean_losses": "; ".join(f"{r['metric']} {r['delta']:+.4g}" for r in rows if r["subset"] == "clean" and r["sig"] == "loss"),
           "fixed_losses": "; ".join(f"{r['metric']} {r['delta']:+.4g}" for r in rows if r["subset"] == "fixed" and r["sig"] == "loss"),
           "time_ratio": round(ratio, 3), "pass": int(ok)}
    return row, rows


def main():
    args = sys.argv[1:]
    frontier = split_out = None
    while args and args[0] in ("--frontier", "--split"):
        if args[0] == "--frontier":
            frontier = Path(args[1])
        else:
            split_out = Path(args[1])
        args = args[2:]
    labels, miss = r6.step1()
    cand = human_labels()
    table, srows = [], []
    for item in args:
        d, label = item.rsplit(":", 1)
        E = Path(d) if Path(d).is_absolute() else r6.S / d
        if not (E / "DONE").exists():
            print(f"{label}: not finished ({E}/DONE missing)")
            continue
        row, rows = decide(E, label, labels, miss, cand)
        table.append(row)
        srows += rows
    for path, data in ((frontier, table), (split_out, srows)):
        if path and data:
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
            print("->", path)


if __name__ == "__main__":
    main()
