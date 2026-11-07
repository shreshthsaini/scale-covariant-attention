#!/usr/bin/env python3
"""Census human labels of sca_v2_prog, AReA and HRDiT on the 95 person/animal prompts at 4096^2.

Every image of the three methods on the 95 prompts has one blind label (protocol
results/dup_audit_20261027/human_labels_95_protocol.md). Packet pP95 (scripts/r7_label_prep.py, seed 20261107, 4 images
per Opus labeler, methods shuffled together) holds sca_v2_prog's 95 images and the 55 HRDiT and 32 AReA images no
earlier round labeled. The other 40 HRDiT and 63 AReA images keep their earlier final labels (human_labels_95.csv,
round 6; human_labels_r7.csv, packets pR1 and pL1). Adjudication of pP95 (human_labels_prog95_adjudication.json, same
rule as round 6): every image whose blind label disagrees with the detector flag or is borderline, plus yes labels of
medium or low confidence and no labels of low confidence, decided from full-resolution crops.

Rate = final yes / 95 with the 95% Wilson interval; the sensitivity label counts borderline as yes. Images whose subject
is missing entirely count as not duplicated and are reported. Paired exact McNemar test (two-sided binomial on the
discordant prompts) of sca_v2_prog against each baseline.

Usage: human_rates_prog95.py [--from-csv]   (--from-csv recomputes the rates file from the archived CSV alone)
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dup_rate as dr  # noqa: E402
from human_rates import FAILED, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/dup_audit_20261027"
S = Path("/scratch/09032/saini_2/sca")
PACKET = S / "r7/label/pP95"
PROG_DETS = S / "eval/prog195_sca_v2/det"
METHODS = ("sca_v2_prog", "area", "hrdit")
FIELDS = ["method", "prompt_id", "source", "img", "png", "detector_flag", "detector_counts", "human_blind", "human_final",
          "human_counts", "implied", "subject_missing", "confidence", "extra", "reason", "adjudication"]


def read_jsonl(paths):
    out = {}
    for f in paths:
        for ln in Path(f).read_text().splitlines():
            try:
                x = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out[x["img"]] = x
    return out


def missing(reason, counts):
    return int(any(s in reason.lower() for s in FAILED) and all(v == 0 for v in (counts or {}).values()))


def collect():
    subj = {x["id"]: x for x in json.loads((D / "subjects_a4k195.json").read_text())}
    ppl = [json.loads(ln)["id"] for ln in (S / "r7/people95.jsonl").read_text().splitlines() if ln.strip()]
    det = {(r["method"], r["prompt_id"]): (r["dup"], r["counts"]) for r in csv.DictReader(open(D / "per_image_a4k195_people.csv"))}
    args = SimpleNamespace(sides="800,1333", thr=0.35, nms=0.5, contain=0.9, contain_area=0.3, max_frac=0.9,
                           kinds={"person", "animal"})
    for (m, pid), (f, c) in dr.flags(dr.load([PROG_DETS]), subj, args).items():
        det[(m, pid)] = (str(f), c)

    def implied(pid):
        return json.dumps({s["query"]: s["implied"] for s in subj[pid]["subjects"] if s.get("kind") in ("person", "animal")},
                          sort_keys=True)

    rows = {}
    # round 6 labels (AReA and HRDiT rows only)
    b95 = read_jsonl([D / "human_labels_95_blind.jsonl"])
    for r in csv.DictReader(open(D / "human_labels_95.csv")):
        if r["method"] not in ("area", "hrdit"):
            continue
        rows[(r["method"], r["prompt_id"])] = {
            "method": r["method"], "prompt_id": r["prompt_id"], "source": "human_labels_95", "img": r["img"], "png": r["png"],
            "detector_flag": r["detector_flag"], "detector_counts": r["detector_counts"], "human_blind": r["human_blind"],
            "human_final": r["human_final"], "human_counts": r["human_counts"], "implied": implied(r["prompt_id"]),
            "subject_missing": r["subject_missing"], "confidence": r["confidence"],
            "extra": json.dumps(b95[r["img"]].get("extra", [])), "reason": r["reason"], "adjudication": r["adjudication"]}
    # round 7 labels (AReA rows of packets pR1 and pL1)
    for r in csv.DictReader(open(D / "human_labels_r7.csv")):
        if r["label"] != "area":
            continue
        pk = S / "r7/label" / r["packet"]
        b = read_jsonl(sorted(pk.glob("out_*.jsonl")))[r["img"]]
        key = json.loads((pk / "key.json").read_text())[r["img"]]
        assert key["prompt_id"] == r["prompt_id"] and (r["label"], r["prompt_id"]) not in rows
        rows[("area", r["prompt_id"])] = {
            "method": "area", "prompt_id": r["prompt_id"], "source": f"human_labels_r7:{r['packet']}", "img": r["img"],
            "png": key["png"], "detector_flag": r["detector_flag"], "detector_counts": r["detector_counts"],
            "human_blind": r["human_blind"], "human_final": r["human_final"],
            "human_counts": json.dumps(b.get("counts"), sort_keys=True), "implied": implied(r["prompt_id"]),
            "subject_missing": missing(b.get("reason", ""), b.get("counts")), "confidence": b.get("confidence", ""),
            "extra": json.dumps(b.get("extra", [])), "reason": r["reason"], "adjudication": r["adjudication"]}
    # packet pP95
    key = json.loads((PACKET / "key.json").read_text())
    blind = read_jsonl(sorted(PACKET.glob("out_*.jsonl")))
    adj = {k: v for k, v in json.loads((D / "human_labels_prog95_adjudication.json").read_text()).items()
           if k.startswith("img")}
    assert set(blind) == set(key), ("blind labels missing", sorted(set(key) - set(blind)))
    assert set(adj) <= set(key)
    for img, k in sorted(key.items()):
        m, pid, b = k["label"], k["prompt_id"], blind[img]
        assert (m, pid) not in rows, (m, pid)
        final, note = adj.get(img, (b["dup"], ""))
        f, c = det[(m, pid)]
        rows[(m, pid)] = {
            "method": m, "prompt_id": pid, "source": "pP95", "img": img, "png": k["png"], "detector_flag": f,
            "detector_counts": c, "human_blind": b["dup"], "human_final": final,
            "human_counts": json.dumps(b.get("counts"), sort_keys=True), "implied": implied(pid),
            "subject_missing": missing(b.get("reason", ""), b.get("counts")), "confidence": b.get("confidence", ""),
            "extra": json.dumps(b.get("extra", [])), "reason": b.get("reason", ""), "adjudication": note}
    for m in METHODS:   # census: every method has every prompt; detector fields from the archived run
        assert sorted(pid for mm, pid in rows if mm == m) == sorted(ppl), m
        for pid in ppl:
            r = rows[(m, pid)]
            if str(r["detector_flag"]) != "":   # earlier rows carry the flag of the same run (pL1 rows left it blank)
                assert str(r["detector_flag"]) == det[(m, pid)][0], (m, pid, r["detector_flag"], det[(m, pid)])
            r["detector_flag"], r["detector_counts"] = det[(m, pid)]
    out = [rows[(m, pid)] for m in METHODS for pid in sorted(ppl)]
    with open(D / "human_labels_prog95.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    return out


def rates(rows):
    by = {m: {r["prompt_id"]: r for r in rows if r["method"] == m} for m in METHODS}
    P = len(by[METHODS[0]])
    L = [f"Census human labels, {P} person/animal prompts at 4096^2, every image of each method labeled blind "
         "(protocol human_labels_95_protocol.md; scripts/human_rates_prog95.py; per-image rows human_labels_prog95.csv).",
         "Rate = final yes / prompts, 95% Wilson interval. with_borderline counts borderline as yes. Detector: "
         "dup_rate.py, thr 0.35, person/animal queries.", ""]
    for label, pos in (("primary", ("yes",)), ("with_borderline", ("yes", "borderline"))):
        for m in METHODS:
            rs = list(by[m].values())
            k = sum(r["human_final"] in pos for r in rs)
            lo, hi = wilson(k, P)
            F = sum(r["detector_flag"] == "1" for r in rs)
            tp = sum(r["detector_flag"] == "1" and r["human_final"] in pos for r in rs)
            miss = sum(r["detector_flag"] == "0" and r["human_final"] in pos for r in rs)
            sm = sum(int(r["subject_missing"]) for r in rs)
            L.append(f"{m:11s} {label:15s} human {k:2d}/{P} = {100 * k / P:5.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}] | "
                     f"detector {F:2d}/{P} = {100 * F / P:4.1f}%, flags confirmed {tp}/{F}, detector misses {miss} | "
                     f"subject missing {sm}")
        L.append("")
    L.append("Paired exact McNemar test over the 95 prompts (two-sided binomial on discordant prompts; b = sca_v2_prog yes "
             "and baseline not, c = the reverse):")
    for label, pos in (("primary", ("yes",)), ("with_borderline", ("yes", "borderline"))):
        for base in METHODS[1:]:
            b = sum(by["sca_v2_prog"][p]["human_final"] in pos and by[base][p]["human_final"] not in pos for p in by[base])
            c = sum(by["sca_v2_prog"][p]["human_final"] not in pos and by[base][p]["human_final"] in pos for p in by[base])
            kp = sum(r["human_final"] in pos for r in by["sca_v2_prog"].values())
            kb = sum(r["human_final"] in pos for r in by[base].values())
            L.append(f"  sca_v2_prog vs {base:5s} {label:15s} {kp}/{P} vs {kb}/{P}, difference {100 * (kp - kb) / P:+5.1f} "
                     f"points, b = {b}, c = {c}, p = {dr.mcnemar(b, c):.4f}")
    for m in METHODS:
        L += ["", f"Copies found, {m} (final yes; extra figures as reported by the blind labeler, xy in full-image pixels; "
              "the adjudication note, when present, is the decision):"]
        for lab in ("yes", "borderline"):
            if lab == "borderline":
                L.append(f"  borderline, {m}:")
            for pid, r in sorted(by[m].items()):
                if r["human_final"] != lab:
                    continue
                L.append(f"  {pid} [{r['source']} {r['img']}] counts {r['human_counts']} implied {r['implied']} "
                         f"detector {r['detector_counts']}")
                for e in json.loads(r["extra"]):
                    xy = e.get("xy") or ["?", "?"]
                    L.append(f"      {e.get('subject', '?')} at ({xy[0]}, {xy[1]}), {e.get('height_px', '?')} px, "
                             f"copy_like {e.get('copy_like')}: {e.get('what', '')}")
                if r["adjudication"]:
                    L.append(f"      adjudicated {r['human_final']}: {r['adjudication']}")
    (D / "human_rates_prog95.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12] + L[13:20]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-csv", action="store_true", help="recompute the rates from human_labels_prog95.csv")
    a = p.parse_args()
    rows = list(csv.DictReader(open(D / "human_labels_prog95.csv"))) if a.from_csv else collect()
    rates(rows)


if __name__ == "__main__":
    main()
