"""Table tab:dupfix: every late-duplication variant run on the 46-prompt subset, one row per run.

Sources (read only):
* results/dup_audit_20261027/rates_46_all_people.csv: people/animal duplication k of 26 per run, AReA and
  published SCA (sca_nb) included.
* results/dup_audit_20261027/dup46_20261029/decision*.txt: scripts/dup_decide.py output, one file per round.
  Per run: "STRICT" line (significant losses against published SCA over 12 metrics: paired 95% interval
  entirely on the losing side), main line (people k/n and median seconds per image over the 46 generated
  images, "time X vs Y" with Y published SCA), and the stratified line (real softening = the interval on
  prompts the detector finds clean in both runs lies below zero). A later round overrides an earlier one.
* results/dup_audit_20261027/dup46_20261029/vs_area_46.txt: published SCA against AReA on the 46 prompts.
A run whose decision lines lack a field (round one predates the stratified lines) or that is not in the results
yet (the queued dup_i_160, dup_i_bridge32 and dup_i_ext75, once eval/dup46_<label>/DONE exists) is filled by
running scripts/dup_decide.py on its eval directory, the same code that wrote the decision files. The table
prints only what these sources say; descriptions of the schedule runs are built from their image sidecars.

Writes paper/figures/snippets/dupfix_variants.tex.
Regenerate: <sca env python> paper/figures/src/gen_dupfix_table.py
"""
from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = ROOT / "results/dup_audit_20261027"
RATES46 = AUDIT / "rates_46_all_people.csv"
DEC = AUDIT / "dup46_20261029"
ROUNDS = ["decision.txt", "decision_c.txt", "decision_h.txt", "decision_round2.txt",
          "decision_bridge_win128_stratified.txt", "decision_round3.txt", "decision_round4.txt"]  # oldest first
VS_AREA = DEC / "vs_area_46.txt"
DECIDE = ROOT / "scripts/dup_decide.py"
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
EVAL = Path("/scratch/09032/saini_2/sca/eval")
OUTIMG = Path("/scratch/09032/saini_2/sca/out/dup46")
OUT = ROOT / "paper/figures/snippets/dupfix_variants.tex"
QUEUED = ["dup_i_160", "dup_i_bridge32", "dup_i_ext75"]  # rows appear once their eval directory has DONE
OURS = "dup_i_shrink"

# (group heading, [(label, description of at most six words)]); descriptions follow each run's sidecar flags.
GROUPS = [
    ("Global bridge ($8\\times8$ cells; 16 keys per cell unless noted)", [
        ("dup_bridge", r"Bridge, 16 keys per cell"),
        ("dup_bridge32", r"Bridge, 32 keys per cell"),
        ("dup_bridge_k64mid", r"Bridge, 64 keys while $\sigma>0.6$"),
        ("dup_bridge_mid", r"Bridge only while $\sigma>0.5$"),
    ]),
    ("Softer opening after the gate", [
        ("dup_ramp70", r"Linear $\pi$ ramp to $\sigma=0.7$"),
        ("dup_bridge_ramp70", r"Linear $\pi$ ramp, bridge"),
        ("dup_head40", r"Per-head opening by $\sigma=0.4$"),
        ("dup_bridge_head50", r"Per-head opening by $0.5$, bridge"),
    ]),
    ("Low-band anchor from the opening", [
        ("dup_anchor_lin", r"Anchor, linear fade to $\sigma=0.3$"),
        ("dup_anchor_cos", r"Anchor, cosine fade to $\sigma=0.3$"),
        ("dup_anchor_lin_r4", r"Linear anchor, refreshed every 4 steps"),
        ("dup_anchor_lin_bridge", r"Linear anchor, bridge"),
        ("dup_anchor_lin_bridge_win128", r"Linear anchor, bridge, 128 windows"),
    ]),
    ("Wider windows after the gate, all with the bridge", [
        ("dup_bridge_win96", r"96 windows ($\sigma>0.8$), scale-4 $\pi$"),
        ("dup_bridge_win128", r"128 windows ($\sigma>0.8$), scale-4 $\pi$"),
        ("dup_bridge_win128s", r"128 windows ($\sigma>0.85$), scale-4 $\pi$"),
        ("dup_e_win128pi2", r"128 windows ($\sigma>0.8$), matched $\pi$"),
        ("dup_j_win128s_pi2", r"128 windows ($\sigma>0.85$), matched $\pi$"),
        ("dup_f_headwin50", r"128 windows for heads $\pi<0.5$"),
        ("dup_f_headwin70", r"128 windows for heads $\pi<0.7$"),
        ("dup_g_dil2", r"Stride-2 dilated windows ($\sigma>0.8$)"),
    ]),
    ("Shrinking window schedule with matched $\\pi$ and the bridge", [(OURS, None)]),
]

SHORT = {"ir": "IR", "aesthetic": "Aes.", "clip": "CLIP", "pickscore": "Pick", "hpsv2": "HPSv2",
         "hpsv21": "HPSv2.1", "musiq": "MUSIQ", "musiq_crop1024": r"MUSIQ$_\mathrm{c}$", "clipiqa": "CLIP-IQA",
         "arniqa": "ARNIQA", "patch_fid": r"FID$_\mathrm{p}$", "patch_kid": r"KID$_\mathrm{p}$"}
ORDER = list(SHORT)

STRICT = re.compile(r"^(\w+): STRICT (?:PASS|FAIL) \(metrics checked (\d+); significant losses: (\[.*?\])\)")
MAIN = re.compile(r"^(\w+): (?:PASS|FAIL) \| people (\d+)/(\d+) .*?\| time ([\d.]+) vs ([\d.]+)")
SOFT = re.compile(r"^\s+(\w+): real softening on clean-in-both prompts: (\[.*?\]|none)\s*$")


def parse(lines, info: dict, source: str) -> None:
    for line in lines:
        m = STRICT.match(line)
        if m:
            losses = [x.split()[0] for x in ast.literal_eval(m.group(3))]
            info.setdefault(m.group(1), {})["strict"] = dict(n_metrics=int(m.group(2)), losses=losses, source=source)
        m = MAIN.match(line)
        if m:
            info.setdefault(m.group(1), {})["main"] = dict(k=int(m.group(2)), n=int(m.group(3)),
                                                           seconds=float(m.group(4)), base_seconds=float(m.group(5)),
                                                           source=source)
        m = SOFT.match(line)
        if m:
            soft = [] if m.group(2) == "none" else list(ast.literal_eval(m.group(2)))
            info.setdefault(m.group(1), {})["soft"] = dict(metrics=soft, source=source)


def schedule_desc(label: str) -> str:
    """Windows of the schedule runs, read from one image sidecar of the run."""
    js = sorted((OUTIMG / label / "4096x4096").glob("*.json"))
    c = json.loads(js[0].read_text())["config"]["sca"]
    sizes = "/".join(str(w) for _, w in c["win_sched"]) + f"/{c['window']}"
    extra = "" if c.get("bridge_k") == 16 else f", {c['bridge_k']} bridge keys"
    return f"Windows {sizes}{extra}"


def names(metrics: list[str]) -> str:
    unknown = [m for m in metrics if m not in SHORT]
    assert not unknown, f"no short name for {unknown}"
    return ", ".join(SHORT[m] for m in sorted(metrics, key=ORDER.index)) if metrics else "none"


def prose(metrics: list[str]) -> str:
    parts = names(metrics).split(", ")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def main():
    rates = {r["method"]: r for r in csv.DictReader(open(RATES46))}
    info: dict = {}
    for f in ROUNDS:
        parse(open(DEC / f), info, f"results/dup_audit_20261027/dup46_20261029/{f}")
    labels = [lab for _, rows in GROUPS for lab, _ in rows]
    queued = [q for q in QUEUED if (EVAL / f"dup46_{q}" / "DONE").exists()]
    labels += queued
    missing = [lab for lab in labels if not {"strict", "main", "soft"} <= set(info.get(lab, {}))]
    ran = []
    if missing:
        runnable = [lab for lab in missing if (EVAL / f"dup46_{lab}" / "DONE").exists()]
        out = subprocess.run([PY, str(DECIDE), *runnable], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        fresh: dict = {}
        parse(out.splitlines(), fresh, f"scripts/dup_decide.py on {EVAL}/dup46_<label> (run by this script)")
        for lab in runnable:
            for key, val in fresh.get(lab, {}).items():
                if key not in info.get(lab, {}):
                    info.setdefault(lab, {})[key] = val
                    ran.append((lab, key))
    rows = []
    base_t = {round(v["main"]["base_seconds"], 6) for v in info.values() if "main" in v}
    assert len(base_t) == 1, f"published SCA time differs between rounds: {base_t}"
    base_t = base_t.pop()
    for lab in labels:
        d = info[lab]
        assert {"strict", "main", "soft"} <= set(d), f"{lab}: incomplete ({sorted(d)})"
        k, n = d["main"]["k"], d["main"]["n"]
        if lab in rates:  # the results table is the authority for k/26; the decision line must agree
            assert (int(rates[lab]["dup"]), int(rates[lab]["n"])) == (k, n), f"{lab}: rates table and decision disagree"
        assert n == 26 and d["strict"]["n_metrics"] == 12
        rows.append(dict(label=lab, k=k, n=n, seconds=d["main"]["seconds"], losses=d["strict"]["losses"],
                         soft=d["soft"]["metrics"], sources=sorted({d[x]["source"] for x in ("strict", "main", "soft")})))
    by = {r["label"]: r for r in rows}

    # Published SCA vs AReA on the 46 prompts: metrics where published SCA leads significantly.
    area_line = next(ln for ln in open(VS_AREA) if ln.startswith("sca_nb:"))
    cells = [c.strip() for c in area_line.split(":", 1)[1].split(";") if c.strip()]
    area_worse = [c.split()[0] for c in cells if c.endswith("+")]
    assert len(cells) == 10
    nb, ar = rates["sca_nb"], rates["area"]
    assert int(nb["n"]) == 26 and int(ar["n"]) == 26

    ours = by[OURS]
    pct = 100 * (ours["seconds"] / base_t - 1)
    area_k = int(ar["dup"])
    reach = [r for r in rows if r["k"] <= area_k]
    assert reach and all(r["soft"] for r in reach), "the key-result claim about runs that reach AReA does not hold"
    key = (rf"Key result: the schedule cuts duplication from {nb['dup']}/26 to {ours['k']}/26 at {pct:.0f}\% more "
           rf"time per image and loses only {prose(ours['losses'])}; no run reaches AReA's {area_k}/26 without "
           r"softening clean prompts.")

    def line(desc, r, shade=False):
        pre = r"\rowcolor[HTML]{EAF4F1}[0pt][0pt]" if shade else ""
        return (pre + f"{desc} & {r['k']}/{r['n']} & {r['seconds']:.1f} & {names(r['losses'])} & "
                f"{names(r['soft'])} \\\\")

    L = ["% Generated by paper/figures/src/gen_dupfix_table.py; do not edit by hand.",
         r"\begin{table}[!htbp]",
         r"\centering\scriptsize",
         r"\caption{\textbf{Every schedule variant on the 46-prompt subset.} Each row is one run at $4096^2$, seed 0: "
         r"Dup.\ is the number of the 26 people and animal prompts with a duplicated subject (open-vocabulary detector), s/img "
         r"is the median time per image over the 46 generated images, Worse lists the metrics (of 12) whose paired "
         r"95\% interval against published \sca{} lies on the losing side, and Softens repeats that test for the ten "
         r"per-image metrics on prompts with no duplicated subject in either run. "
         + key + "}",
         r"\label{tab:dupfix}",
         r"\vspace{2pt}",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}l r r >{\raggedright\arraybackslash}p{0.30\linewidth} "
         r">{\raggedright\arraybackslash}p{0.22\linewidth}@{}}",
         r"\toprule",
         r"Variant & Dup.\,$\downarrow$ & s/img\,$\downarrow$ & Worse than published \sca{} & Softens clean prompts \\",
         r"\midrule",
         rf"Without the schedule (published) & {nb['dup']}/{nb['n']} & {base_t:.1f} & reference & reference \\",
         rf"\textcolor{{gray}}{{AReA}} & \textcolor{{gray}}{{{ar['dup']}/{ar['n']}}} & \textcolor{{gray}}{{n/a}} & "
         rf"\textcolor{{gray}}{{{'all 10 per-image' if len(area_worse) == 10 else names(area_worse)}}} & "
         r"\textcolor{gray}{n/a} \\"]
    for head, items in GROUPS:
        L += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{head}}}}} \\"]
        for lab, desc in items:
            if lab == OURS:
                L.append(line(r"\textbf{\sca{} (ours)}, " + schedule_desc(lab).replace("Windows", "windows"),
                              by[lab], shade=True))
                for q in queued:
                    L.append(line(schedule_desc(q), by[q]))
            else:
                L.append(line(desc, by[lab]))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(L) + "\n"
    assert "---" not in tex and "—" not in tex and "§" not in tex
    for lab, desc in [x for _, items in GROUPS for x in items if x[1]]:
        assert len(re.sub(r"\$[^$]*\$", "x", desc).split()) <= 6, f"{lab}: description over six words"
    OUT.write_text(tex)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} runs + published SCA + AReA; queued included: {queued}")
    print(f"fields filled by running dup_decide.py: {ran}")
    for r in rows:
        print(f"{r['label']:30s} {r['k']:2d}/26 {r['seconds']:6.1f}s  worse: {names(r['losses'])}  "
              f"soft: {names(r['soft'])}")


if __name__ == "__main__":
    main()
