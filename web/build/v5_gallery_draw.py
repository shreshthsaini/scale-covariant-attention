#!/usr/bin/env python3
"""The eight-method comparison set of gallery.html: a seeded draw of 15 evaluation prompts (rule below).

Rule: the 195 Aesthetic-4K evaluation prompts (results/evidence/prompts/aesthetic4k_eval.jsonl), minus the prompts
the main page already shows in a figure (the nine decile prompts of the gain explorer, assets/data/percentiles/data.js,
and the sheep-and-church prompt of "How it works"), sorted by prompt id; then Python's random.Random(0).sample(pool, 15).
Every drawn prompt must have all eight methods' 4096^2 images at seed 0: progressive and single-pass SCA, AReA,
UltraImage, DyPE, SEGA and full attention from the run folders build_zoom.py names, HRDiT from the images the paper's
Table 1 scored (eval/prog195_sca_v2/base/per_image.csv, method hrdit_dev_author). A missing image stops the script;
it never redraws.

Writes web/build/gallery_draw.json (rule, seed, pool, exclusions, picks with their image paths), which
build_zoom.py compare reads.
Usage: /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_gallery_draw.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "web"
PROMPTS = REPO / "results/evidence/prompts/aesthetic4k_eval.jsonl"
PERCENTILES = WEB / "assets/data/percentiles/data.js"
OUT = WEB / "build/gallery_draw.json"
O = "/scratch/09032/saini_2/sca/out"
DIRS = {"prog": f"{O}/dupfix_prog/sca_v2_prog/4096x4096", "sca": f"{O}/dupfix/sca_v2/4096x4096",
        "area": f"{O}/main/area/4096x4096", "ultra": f"{O}/main/ultraimage/4096x4096",
        "dype": f"{O}/main/dype/4096x4096", "sega": f"{O}/main/sega_official/4096x4096",
        "full": f"{O}/main/full/4096x4096"}
HRDIT_CSV = "/scratch/09032/saini_2/sca/eval/prog195_sca_v2/base/per_image.csv"
SEED, N = 0, 15
SHEEP = "a4k_eval_4096_minus8343371271909676658"        # the image of "How it works" and the views component


def sid_of(pid: str) -> str:
    s = pid.replace("a4k_eval_4096_", "")
    return ("m" + s[5:]) if s.startswith("minus") else s


def pid_of(sid: str) -> str:
    return "a4k_eval_4096_" + (("minus" + sid[1:]) if sid.startswith("m") else sid)


def main() -> None:
    rows = [json.loads(l) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    prompt = {r["id"]: r["prompt"] for r in rows}
    t = PERCENTILES.read_text(encoding="utf-8")
    i = t.index("=", t.index("window.SCA_DATA.percentiles")) + 1
    deciles = [pid_of(st["key"]) for st in json.loads(t[i:t.rindex(";")])["stops"]]
    excluded = sorted(set(deciles) | {SHEEP})
    missing = [e for e in excluded if e not in prompt]
    if missing:
        raise SystemExit(f"excluded ids not in the evaluation set: {missing}")
    pool = sorted(p for p in prompt if p not in excluded)
    picks = random.Random(SEED).sample(pool, N)
    hrdit = {}
    with open(HRDIT_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("method") == "hrdit_dev_author":
                hrdit[os.path.basename(r["path"]).rsplit("_", 2)[0]] = r["path"]
    out = []
    for pid in picks:
        hits = sorted(glob.glob(f"{DIRS['area']}/{pid}_*_seed0.png"))
        if len(hits) != 1:
            raise SystemExit(f"{pid}: AReA image not unique: {hits}")
        stem = os.path.basename(hits[0])[:-4]
        paths = {m: f"{d}/{stem}.png" for m, d in DIRS.items()}
        paths["hrdit"] = hrdit.get(pid)
        bad = [m for m, p in paths.items() if not p or not os.path.exists(p)]
        if bad:
            raise SystemExit(f"{pid}: missing {bad}")
        out.append({"sid": sid_of(pid), "id": pid, "prompt": prompt[pid], "stem": stem, "paths": paths})
    OUT.write_text(json.dumps({
        "rule": "random.Random(0).sample(sorted(pool), 15); pool = the 195 Aesthetic-4K evaluation prompt ids minus the "
                "main page's figure prompts (the nine gain-explorer deciles and the How-it-works sheep prompt)",
        "seed": SEED, "n": N, "pool_size": len(pool), "excluded": excluded, "picks": out}, indent=1) + "\n", encoding="utf-8")
    print(f"pool {len(pool)} (195 minus {len(excluded)}); picks: " + ", ".join(p["sid"] for p in out))


if __name__ == "__main__":
    main()
