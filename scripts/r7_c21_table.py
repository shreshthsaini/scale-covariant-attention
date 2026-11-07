#!/usr/bin/env python3
"""Hand counts on copyprone21 (the 21 person/animal prompts outside the 46-subset where sca_v2 seed 0 has a detector
flag; /scratch/09032/saini_2/sca/r7/copyprone21.jsonl) for every label with at least 15 blind-labeled images, from
results/dup_audit_20261027/human_labels_95.csv (Step 1, column method) and human_labels_r7.csv (round 7, column
label). Prints label, n labeled, yes, borderline, sorted by yes. Usage: r7_c21_table.py [--out FILE]
"""
import argparse
import csv
import json
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "results/dup_audit_20261027"
C21 = "/scratch/09032/saini_2/sca/r7/copyprone21.jsonl"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    ids = [json.loads(ln)["id"] for ln in open(C21) if ln.strip()]
    lab = {}
    for r in csv.DictReader(open(A / "human_labels_95.csv")):
        lab.setdefault(r["method"], {})[r["prompt_id"]] = r["human_final"]
    for r in csv.DictReader(open(A / "human_labels_r7.csv")):
        lab.setdefault(r["label"], {})[r["prompt_id"]] = r["human_final"]
    rows = []
    for m, d in lab.items():
        v = [d[i] for i in ids if i in d]
        if len(v) >= 15:
            rows.append((sum(x == "yes" for x in v), m, len(v), sum(x == "borderline" for x in v)))
    text = "".join(f"{m:16s} n={n:2d} yes={y:2d} borderline={b}\n" for y, m, n, b in sorted(rows))
    print(text, end="")
    if a.out:
        a.out.write_text(text)


if __name__ == "__main__":
    main()
