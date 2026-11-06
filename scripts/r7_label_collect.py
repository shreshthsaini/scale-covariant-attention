#!/usr/bin/env python3
"""Collect blind labels of a round-7 packet (scripts/r7_label_prep.py) into results/dup_audit_20261027/human_labels_r7.csv.

Reads PACKET/key.json and PACKET/out_*.jsonl (one JSON line per image, protocol human_labels_95_protocol.md), joins the
detector flag of each image from the per-image people CSV(s) given, and appends one row per image (label, prompt_id,
img, detector_flag, detector_counts, human_blind, human_final, reason, adjudication). human_final = human_blind unless
--adjudicate img=final:reason is given. Rows already present for (label, prompt_id) are replaced.
Usage: r7_label_collect.py --packet DIR --flags per_image_people.csv [...] [--adjudicate img001=no:"reflection" ...]
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/dup_audit_20261027/human_labels_r7.csv"
FIELDS = ["label", "prompt_id", "img", "detector_flag", "detector_counts", "human_blind", "human_final", "reason",
          "adjudication", "packet"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--flags", nargs="+", default=[])
    p.add_argument("--adjudicate", nargs="*", default=[])
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    key = json.loads((a.packet / "key.json").read_text())
    blind = {}
    for f in sorted(a.packet.glob("out_*.jsonl")):
        for ln in f.read_text().splitlines():
            try:
                x = json.loads(ln)
            except json.JSONDecodeError:
                continue
            blind[x["img"]] = x
    det = {}
    for f in a.flags:
        for r in csv.DictReader(open(f)):
            det[(r["method"], r["prompt_id"])] = (r["dup"], r.get("counts", ""))
    adj = {}
    for item in a.adjudicate:
        img, rest = item.split("=", 1)
        final, _, why = rest.partition(":")
        adj[img] = (final, why)
    rows = list(csv.DictReader(open(a.out))) if a.out.exists() else []
    new = []
    missing = []
    for img, k in sorted(key.items()):
        if img not in blind:
            missing.append(img)
            continue
        b = blind[img]
        d = det.get((k["label"], k["prompt_id"]), ("", ""))
        final, why = adj.get(img, (b["dup"], ""))
        new.append({"label": k["label"], "prompt_id": k["prompt_id"], "img": img, "detector_flag": d[0],
                    "detector_counts": d[1], "human_blind": b["dup"], "human_final": final, "reason": b.get("reason", ""),
                    "adjudication": why, "packet": a.packet.name})
    done = {(r["label"], r["prompt_id"]) for r in new}
    rows = [r for r in rows if (r["label"], r["prompt_id"]) not in done] + new
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(new)} rows from {a.packet.name} -> {a.out}; missing blind labels: {missing or 'none'}")
    for r in new:
        print(r["img"], r["label"], r["prompt_id"][-6:], "det", r["detector_flag"], "blind", r["human_blind"],
              "final", r["human_final"])


if __name__ == "__main__":
    main()
