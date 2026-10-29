#!/usr/bin/env python3
"""Apply the lead's acceptance rule (2026-10-28) to finished 46-prompt duplication runs (eval/dup46_<label>).

Rule: (1) people/animal duplication rate <= 10% on the 46 prompts with a Wilson interval overlapping AReA's;
(2) no significant loss vs published SCA (sca_nb) on ImageReward, HPSv2.1 or PickScore (paired 95% interval
entirely below 0); (3) CLIP-IQA and ARNIQA losses do not flip SCA's lead over AReA
(195-prompt lead + 46-prompt delta of the variant vs sca_nb > 0; the same-prompt 46 lead is printed too); (4) median time at most +10% over sca_nb on the same prompts.
Patch FID and the any-subject rate are reported, not ruled. Usage: dup_decide.py LABEL [LABEL ...]"""

import csv
import glob
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dup_rate as D  # noqa: E402

S = Path("/scratch/09032/saini_2/sca")
ROOT = Path(__file__).resolve().parents[1]
ids = [json.loads(l)["id"] for l in open(S / "abl50.jsonl")]
dev = {json.loads(l)["id"] for l in open(S / "dev4.jsonl")}
ids46 = [i for i in ids if i not in dev]
subj = {r["id"]: r for r in json.load(open(ROOT / "results/dup_audit_20261027/subjects_a4k195.json")) if r["id"] in ids46}


def rate(rows, method, kinds):
    a = SimpleNamespace(sides="800,1333", thr=0.35, nms=0.5, contain=0.9, contain_area=0.3, max_frac=0.9, kinds=kinds)
    fl = {k: v for k, v in D.flags(rows, subj, a).items() if k[0] == method}
    k, n = sum(f for f, _ in fl.values()), len(fl)
    return k, n, D.wilson(k, n)


per = list(csv.DictReader(open(S / "eval/a4k195_main/extra/extra_per_image.csv")))
lead = {}
for m in ("clipiqa", "arniqa"):
    v = {(r["method"], r["prompt_id"]): float(r[m]) for r in per if r["prompt_id"] in ids46 and r.get(m)}
    d = [v[("sca_nb", i)] - v[("area", i)] for i in ids46 if ("sca_nb", i) in v and ("area", i) in v]
    lead[m] = sum(d) / len(d)


def med_seconds(d, keep):
    t = [json.load(open(j)).get("seconds") for j in glob.glob(str(d) + "/*.json")
         if json.load(open(j))["config"]["prompt_id"] in keep]
    return statistics.median([x for x in t if x]) if t else None


lead195 = {r["metric"]: float(r["delta"]) for r in csv.DictReader(open(ROOT / "results/a4k195_main/sca_nb_vs_area.csv"))
           if r["method"] == "sca_nb" and r["reference"] == "area"}
base_t = med_seconds(S / "out/main/sca_nb/4096x4096", set(ids))
for label in sys.argv[1:]:
    E = S / f"eval/dup46_{label}"
    if not (E / "DONE").exists():
        print(f"{label}: not finished ({E}/DONE missing)")
        continue
    rows = D.load([S / "eval/dup_detect_20261027", E / "det"])
    kp, np_, ci = rate(rows, label, {"person", "animal"})
    ka, na, cia = rate(rows, "area", {"person", "animal"})
    kb, nb, _ = rate(rows, "sca_nb", {"person", "animal"})
    kany, nany, _ = rate(rows, label, None)
    r1 = kp / np_ <= 0.10 and ci[0] <= cia[1]
    deltas = {r["metric"]: r for r in csv.DictReader(open(E / f"paired/{label}_vs_sca_nb.csv"))}
    loss = {m: float(deltas[m]["ci_high"]) < 0 for m in ("ir", "hpsv21", "pickscore") if m in deltas}
    r2 = not any(loss.values()) and len(loss) == 3
    # primary: the paper's 195-prompt lead plus the variant's paired delta; also shown on the same 46 prompts
    keep = {m: round(lead195[m] + float(deltas[m]["delta"]), 4) for m in ("clipiqa", "arniqa") if m in deltas}
    keep46 = {m: round(lead[m] + float(deltas[m]["delta"]), 4) for m in ("clipiqa", "arniqa") if m in deltas}
    r3 = all(v > 0 for v in keep.values()) and len(keep) == 2
    t = med_seconds(S / f"out/dup46/{label}/4096x4096", set(ids))
    r4 = t is not None and base_t and t / base_t <= 1.10
    pf = deltas.get("patch_fid", {}).get("delta", "n/a")
    # strict rule (user directive 2026-10-28): no significant paired loss on ANY metric vs published SCA
    lower_better = {"patch_fid", "patch_kid", "fid", "kid"}
    allm = dict(deltas)
    bp = E / "base/paired_deltas.csv"
    if bp.exists():
        for r in csv.DictReader(open(bp)):
            if r.get("method") == label and r.get("reference_method", "sca_nb") == "sca_nb" and r["metric"] in lower_better:
                allm[r["metric"]] = {"delta": r["value"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
    strict_loss = []
    for m, r in allm.items():
        try:
            lo, hi = float(r["ci_low"]), float(r["ci_high"])
        except (TypeError, ValueError, KeyError):
            continue
        if (m in lower_better and lo > 0) or (m not in lower_better and hi < 0):
            strict_loss.append(f"{m} {float(r['delta']):+.4g} [{lo:+.4g},{hi:+.4g}]")
    strict = r1 and r4 and not strict_loss
    print(f"{label}: STRICT {'PASS' if strict else 'FAIL'} (metrics checked {len(allm)}; significant losses: "
          f"{strict_loss or 'none'}); all deltas: " + "; ".join(
              f"{m} {float(r['delta']):+.4g} [{float(r['ci_low']):+.4g},{float(r['ci_high']):+.4g}]" for m, r in sorted(allm.items())
              if r.get("ci_low") not in (None, "") and r.get("ci_high") not in (None, "")))
    fmt = lambda m: f"{m} {float(deltas[m]['delta']):+.3f} [{float(deltas[m]['ci_low']):+.3f},{float(deltas[m]['ci_high']):+.3f}]"
    print(f"{label}: {'PASS' if (r1 and r2 and r3 and r4) else 'FAIL'} | people {kp}/{np_} = {kp / np_:.1%} "
          f"[{ci[0]:.1%},{ci[1]:.1%}] (sca_nb {kb}/{nb}, AReA {ka}/{na} [{cia[0]:.1%},{cia[1]:.1%}]) r1={r1} | any "
          f"{kany}/{nany} | {'; '.join(fmt(m) for m in ('ir', 'hpsv21', 'pickscore', 'clipiqa', 'arniqa') if m in deltas)} "
          f"r2={r2} | lead over AReA after (195-prompt lead) {keep}, (46-prompt lead) {keep46} r3={r3} | time {t} vs {base_t} r4={r4} | patch_fid {pf}")
