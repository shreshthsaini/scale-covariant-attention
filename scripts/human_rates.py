#!/usr/bin/env python3
"""Human-validated duplication rates (round 6, Step 1) from the blind labels, the unblinding key and the adjudication.

Inputs (results/dup_audit_20261027): human_labels_95_sample_key.json (img -> method, prompt, detector flag, png),
human_labels_95_blind.jsonl (one blind label per image), human_labels_95_adjudication.json (final label where it
differs from the blind one, with a reason). Writes human_labels_95.csv (one row per labeled image) and
human_rates_95.csv / .txt.

Rate per method over its population of P prompts (95 people/animal prompts), F flagged by the detector (all labeled)
and U = P - F unflagged (n of them labeled at random): rate = (TP_F + U * k / n) / P, TP_F = human-confirmed flags,
k = human duplicates among the n sampled unflagged. Interval: the 95% Wilson interval of k / n carried through the
same formula (the flagged stratum is a census, so its count has no sampling error). Primary label: yes only;
sensitivity: yes or borderline. Images whose subject is missing entirely (failed generations) are counted as
not duplicated and reported separately.
"""

import csv
import json
import math
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results/dup_audit_20261027"
METHODS = ("area", "hrdit", "full", "sca_v2")
FAILED = ("failed", "degenerate", "collapsed", "blank", "broken", "no subject", "missing entirely", "is missing",
          "no person at all", "gradient bands", "featureless")


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def main():
    key = {r["img"]: r for r in json.loads((D / "human_labels_95_sample_key.json").read_text())}
    blind = {}
    for line in (D / "human_labels_95_blind.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            blind[d["img"]] = d
    adj = {k: v for k, v in json.loads((D / "human_labels_95_adjudication.json").read_text()).items() if k.startswith("img")}
    assert set(blind) == set(key), (sorted(set(key) - set(blind)), sorted(set(blind) - set(key)))
    rows = []
    for img in sorted(key):
        r, b = key[img], blind[img]
        final, note = adj.get(img, (b["dup"], ""))
        reason = b.get("reason", "")
        failed = any(s in reason.lower() for s in FAILED) and all(v == 0 for v in (b.get("counts") or {}).values())
        rows.append({"img": img, "method": r["method"], "prompt_id": r["prompt_id"], "png": r["png"],
                     "sample": "flagged" if r["detector_flag"] else "unflagged_random", "detector_flag": r["detector_flag"],
                     "detector_counts": r["detector_counts"], "human_blind": b["dup"], "human_final": final,
                     "human_counts": json.dumps(b.get("counts"), sort_keys=True),
                     "implied": json.dumps({s["query"]: s["implied"] for s in r["subjects"]}, sort_keys=True),
                     "subject_missing": int(failed), "confidence": b.get("confidence", ""),
                     "reason": reason, "adjudication": note})
    with open(D / "human_labels_95.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pop = {m: {"F": 0, "U": 0} for m in METHODS}
    for r in csv.DictReader(open(D / "per_image_a4k195_people.csv")):
        if r["method"] in pop:
            pop[r["method"]]["F" if r["dup"] == "1" else "U"] += 1
    sv = json.loads((D / "human_labels_95_sample_key.json").read_text())
    if pop["sca_v2"]["F"] + pop["sca_v2"]["U"] == 0:   # sca_v2 is not in the a4k195 table: count from the key's flags
        flagged = sum(1 for r in sv if r["method"] == "sca_v2" and r["detector_flag"] == 1)
        pop["sca_v2"] = {"F": flagged, "U": 95 - flagged}
    out, lines = [], []
    for m in METHODS:
        mine = [r for r in rows if r["method"] == m]
        F, U = pop[m]["F"], pop[m]["U"]
        fl = [r for r in mine if r["sample"] == "flagged"]
        un = [r for r in mine if r["sample"] == "unflagged_random"]
        assert len(fl) == F, (m, len(fl), F)
        for label, pos in (("primary", ("yes",)), ("with_borderline", ("yes", "borderline"))):
            tp = sum(r["human_final"] in pos for r in fl)
            k, n = sum(r["human_final"] in pos for r in un), len(un)
            lo, hi = wilson(k, n)
            P = F + U
            rate, rlo, rhi = (tp + U * k / n) / P, (tp + U * lo) / P, (tp + U * hi) / P
            det = F / P
            rec = {"method": m, "label": label, "prompts": P, "flagged": F, "unflagged": U, "tp_flagged": tp,
                   "precision": round(tp / F, 3) if F else "", "sampled_unflagged": n, "dup_in_sample": k,
                   "miss_rate": round(k / n, 3), "human_rate": round(rate, 4), "lo95": round(rlo, 4), "hi95": round(rhi, 4),
                   "detector_rate": round(det, 4),
                   "subject_missing_in_sample": sum(int(r["subject_missing"]) for r in mine)}
            out.append(rec)
            lines.append(f"{m:7s} {label:15s} human {100 * rate:5.1f}% [{100 * rlo:4.1f}, {100 * rhi:4.1f}] | detector "
                         f"{100 * det:4.1f}% ({F}/{P}) precision {tp}/{F} | unflagged sample {k}/{n} | subject missing in "
                         f"{rec['subject_missing_in_sample']} of {len(mine)} labeled")
    with open(D / "human_rates_95.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (D / "human_rates_95.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
