#!/usr/bin/env python3
"""Every number the SCA project page prints, computed from the archived results.

Configurations (paper rewrite of 2026-11-07). The headline is progressive SCA (method sca_v2_prog, run
results/prog195_sca_v2): HRDiT's released three-stage schedule (30 steps at 1024^2 with the model's own attention,
then the last 17 steps at 2048^2 and the last 10 at 4096^2) with SCA attention at the two upper stages. Single-pass
SCA (sca_v2: 28 steps, global opening, shrinking windows, 1,024 bridge tokens) is the fast mode. Rows that exist only
for the operator without the schedule (sca_nb: 6144^2, FLUX.2-dev) keep sca_nb and the page labels them. HRGen-1K
reads the single-pass run (results/hrgen1k_dev_sca_v2). Progressive SCA's pairs against every baseline come from
results/prog195_vs_baselines (per-image, 2000 prompt draws) and results/prog195_pairs/cfid_vs_{area,sca_v2}
(set-level, 1000 draws, 168 reference identities); its clean latency from results/prog_latency (sweep 3, timed back
to back with HRDiT and AReA). Results the paper lists as pending (seeds 1 and 2, HRGen-1K, set-level intervals
against the other baselines) are emitted as "pending" until their files exist. The head-versus-noise variance split
comes from results/heads_view/variance_split.csv.

Inputs outside results/, as in the paper's own generators (paper/figures/src/scav2_runs.py, gen_main_tables.py):
the eval.py paired deltas of the sca_v2 runs (1000 prompt draws,
/scratch/09032/saini_2/sca/eval/<run>/{base/paired_deltas,extra/extra_paired_deltas}.csv; the evaluation tasks do
not copy them), the sca_v2 generation sidecars (out/dupfix/sca_v2/4096x4096) for its configuration constants, the
progressive sidecars (out/dupfix_prog/sca_v2_prog/4096x4096) for its stage schedule, and the HRDiT sidecars of the
images eval/prog195_sca_v2 scored (paths from its base/per_image.csv) for HRDiT's stage steps.

Inputs are otherwise files under results/ only: the evaluation CSVs, the calibration JSONs (plus the calibration
table's .npz, read for its block x head shape and its cell fractions), the compute summary, and the generation
sidecars (results/evidence/generation_sidecars.jsonl.gz) for configuration constants. No number on the page is
typed by hand.

Outputs (repository-relative):
  web/build/numbers.json    {key: {"value", "display", "src", "how"}}
  web/assets/js/numbers.js  window.SCA_NUMBERS = {key: display};
A key that ends in ".cls" holds CSS classes for the element of the key without that suffix (Table 1: "best" and
"second" rank a cell within its group, "sig-up" and "sig-dn" mark a row significantly better or worse than
progressive SCA); --fill writes them into that element's class attribute.

Usage, with the sca env python (a few CSV reads; light enough for any node):
  build_numbers.py                      build both outputs and print the check table against the paper
  build_numbers.py --fill web/index.html
                                        also rewrite the text of every <tag ... data-num="key" ...>TEXT</tag>
                                        in that file with the key's display string (repeatable); an unknown
                                        key or a data-num element with nested tags is an error
  build_numbers.py --quiet              print only the flagged rows and the summary of the check
  build_numbers.py --strict             exit 1 on any mismatch with the paper that EXPLAINED does not cover

PAPER_QUOTES holds the strings paper/main.tex prints. It feeds only the check table; no output is read
from it. PAPER_ONLY lists numbers the paper prints that no results file holds; they are not emitted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(os.path.abspath(__file__)).parents[2]
RES = REPO / "results"
OUT_JSON = REPO / "web" / "build" / "numbers.json"
OUT_JS = REPO / "web" / "assets" / "js" / "numbers.js"

MINUS = "−"  # U+2212 MINUS SIGN in every negative display (page request); "-" gives ASCII
TIMES = "×"   # multiplication sign printed after speed ratios
WORDS = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen"]

# short key -> (CSV metric, label in metric lists, scale, decimals of the paper's tables, lower is better);
# Table app_4k order
METRICS = {
    "ir": ("ir", "ImageReward", 1, 3, False),
    "aesthetic": ("aesthetic", "aesthetic score", 1, 2, False),
    "clip": ("clip", "CLIP score", 1, 2, False),
    "pick": ("pickscore", "PickScore", 1, 2, False),
    "hps2": ("hpsv2", "HPSv2", 100, 2, False),
    "hps21": ("hpsv21", "HPSv2.1", 100, 2, False),
    "musiq": ("musiq", "MUSIQ", 1, 1, False),
    "musiqc": ("musiq_crop1024", "MUSIQ (crops)", 1, 1, False),
    "clipiqa": ("clipiqa", "CLIP-IQA", 100, 1, False),
    "arniqa": ("arniqa", "ARNIQA", 100, 1, False),
    "is": ("is", "Inception score", 1, 2, False),
    "fid": ("fid", "FID", 1, 1, True),
    "kid": ("kid", "KID", 1000, 2, True),
    "fidp": ("patch_fid", "patch FID", 1, 1, True),
    "kidp": ("patch_kid", "patch KID", 1000, 2, True),
}
PER_IMAGE = list(METRICS)[:10]
FIFTEEN = list(METRICS)  # the 15 quality metrics of Table app_4k and Figure paired_area; seconds excluded
DIST = {"fid", "kid", "patch_fid", "patch_kid"}
UNITS = {1: "", 100: ", x100", 1000: ", x1000"}

# Table tab:main: (method key, CSV method, run); metrics with the table's decimals
# SCA (ours) is the final operator, sca_v2 (window schedule after the opening and 1,024 bridge tokens)
MAIN_V2 = "a4k195_main_sca_v2"
TABLE1 = [("full", "full", "a4k195_main"), ("dype", "dype", "a4k195_main"),
          ("sega", "sega_official", "a4k195_main"), ("area", "area", "a4k195_main"),
          ("sca", "sca_v2", MAIN_V2), ("ultraimage", "ultraimage", "a4k195_main"),
          ("hrdit", "hrdit_dev_author", "a4k195_panel"), ("prog", "sca_v2_prog", "prog195_sca_v2")]
# Table 1 groups as the paper ranks them (bold best, underline second within a group); campaign time is not ranked
TABLE1_GROUPS = [["full", "dype", "sega", "area", "sca"], ["ultraimage", "hrdit", "prog"]]
# the final operator's 195-prompt runs; their eval.py paired deltas (1000 prompt draws) are not copied to results/
# by the evaluation tasks, so they are read from the run's evaluation directory on scratch, exactly as
# paper/figures/src/scav2_runs.py does
V2_RUNS = ["a4k195_main_sca_v2", "a4k195_2048x4096_sca_v2", "a4k195_2048x6144_sca_v2", "krea195_sca_v2",
           "prog195_sca_v2"]
EVAL = Path("/scratch/09032/saini_2/sca/eval")
V2_SIDECARS = Path("/scratch/09032/saini_2/sca/out/dupfix/sca_v2/4096x4096")  # generation sidecars of sca_v2
LAT_V2_SAMPLES = RES / "latency_curve_v2" / "samples.csv"   # second clean sweep: sca_v2 and re-timed sca_nb
LAT_V2_SUMMARY = RES / "latency_curve_v2" / "summary.csv"
TABLE1_METRICS = [("ir", 2), ("hps21", 2), ("pick", 2), ("clip", 2), ("musiqc", 1), ("fid", 1), ("fidp", 1)]

# Progressive SCA (the headline): HRDiT's three-stage schedule with SCA attention at 2048^2 and 4096^2
PROG = "sca_v2_prog"
PROG_RUN = "prog195_sca_v2"
PROG_SIDECARS = Path("/scratch/09032/saini_2/sca/out/dupfix_prog/sca_v2_prog/4096x4096")
PROG_EVAL_PER_IMAGE = EVAL / PROG_RUN / "base" / "per_image.csv"   # image paths of the HRDiT run scored beside it
PROG_LAT = RES / "prog_latency" / "latency_summary.csv"   # sweep 3: progressive SCA, HRDiT and AReA back to back
PROG_PAIRS = RES / "prog195_vs_baselines"                  # sca_v2_prog_vs_<baseline>.csv, per-image, 2000 draws
PROG_CFID = {"area": RES / "prog195_pairs" / "cfid_vs_area" / "paired_deltas.csv",
             "sca_v2": RES / "prog195_pairs" / "cfid_vs_sca_v2" / "paired_deltas.csv"}
PROG_CFID_ALL = PROG_PAIRS / "cfid" / "paired_deltas.csv"  # every baseline, reference sca_v2_prog (pending)
# (page key, CSV method) of every baseline progressive SCA is paired with
PROG_VS = [("area", "area"), ("dype", "dype"), ("sega", "sega_official"), ("full", "full"),
           ("ultraimage", "ultraimage"), ("hrdit", "hrdit_dev_author"), ("sca", "sca_v2")]
# results the paper lists as pending, with the file names paper/figures/src/gen_main_tables.py accepts
PROG_LATER = {"seed1": (195, ["prog195_s1/sca_v2_prog_vs_area.csv", "prog195_seed1/sca_v2_prog_vs_area.csv"]),
              "seed2": (195, ["prog195_s2/sca_v2_prog_vs_area.csv", "prog195_seed2/sca_v2_prog_vs_area.csv"]),
              "hrgen": (1000, ["prog_hrgen1k/sca_v2_prog_vs_area.csv", "hrgen1k_prog/sca_v2_prog_vs_area.csv"])}
# decimals of paired differences in the paper's running text (scale as in METRICS)
TEXT_DEC = {"ir": 2, "aesthetic": 2, "clip": 2, "pick": 2, "hps2": 2, "hps21": 2, "musiq": 1, "musiqc": 1,
            "clipiqa": 1, "arniqa": 1, "is": 2, "fid": 1, "kid": 2, "fidp": 1, "kidp": 2}
SETLEVEL = ["fid", "kid", "fidp", "kidp"]
# Figure paired_area (paper/figures/src/fig_paired_prog.py): both configurations minus AReA, 14 metrics (no IS)
CHART_PA = [k for k in METRICS if k != "is"]
CHART_PA_SERIES = [("prog", PROG, PROG_PAIRS / "sca_v2_prog_vs_area.csv"),
                   ("single", "sca_v2", RES / "prog195_pairs" / "sca_v2_vs_area.csv")]
CHART_PA_SET = RES / "prog195_pairs" / "cfid_vs_area"
HEADS_VAR = RES / "heads_view" / "variance_split.csv"

LAT_CURVE = RES / "latency_curve.csv"            # audited clean batch-1 sweep, full precision, no warm-up rows
LAT_BATCH1 = RES / "latency_curve_batch1.csv"    # same sweep at 0.01 s with warm-up rows, plus full at 6144^2, HRDiT
LAT_SUMMARY = RES / "latency_curve_summary.csv"  # archived medians, used as a cross-check
WARMUP = "sca_latency_warmup"
SIDES = [2048, 3072, 4096, 6144, 8192]
LAT_METHODS = [("sca", "sca_nb"), ("area", "area"), ("full", "full")]
SIDECARS = RES / "evidence" / "generation_sidecars.jsonl.gz"
CALIB = RES / "evidence" / "calib"

# ---------------------------------------------------------------------------------------------------------
# Strings paper/main.tex prints (LaTeX signs and \times written out; metric lists in this script's labels).
# Used ONLY by the check table.
PAPER_QUOTES = {
    # Table tab:main
    **{f"main.{m}.{k}": v for m, row in {
        "full": ("28", "186.5", "196.2", "-2.18", "7.60", "17.94", "15.37", "33.1", "312.4", "252.2"),
        "dype": ("28", "n/m", "195.9", "-0.10", "20.55", "20.51", "24.92", "50.3", "205.5", "171.9"),
        "sega": ("28", "n/m", "190.4", "0.56", "24.34", "21.64", "26.95", "50.3", "164.9", "125.8"),
        "area": ("28", "202.4", "264.4", "0.92", "27.16", "22.36", "27.69", "50.4", "145.8", "102.9"),
        "sca": ("28", "71.1", "71.5", "1.23", "29.67", "23.09", "29.14", "61.2", "138.1", "99.1"),
        "ultraimage": ("30+30", "n/m", "308.5", "1.28", "30.77", "23.33", "28.84", "41.7", "142.0", "109.5"),
        "hrdit": (None, "108.6", "110.4", "1.34", "30.92", "23.46", "29.12", "53.5", "138.4", "103.0"),
    }.items() for k, v in zip(("steps", "time_clean", "time_camp", "ir", "hps21", "pick", "clip", "musiqc",
                                "fid", "fidp"), row) if v is not None},
    "main.n_prompts": "195",
    # Section 4.1 sentence, Figure paired_area caption, Table app_4k (SCA final vs AReA)
    "paired.n_better": "11", "paired.n_unresolved": "4", "paired.n_total": "15", "paired.n_worse": "0",
    "paired.ir.headline": "+0.31 [0.23, 0.38]", "paired.fid.headline": "-7.7 [-13.1, -2.6]",
    "paired.ir.delta2": "+0.31", "paired.ir.ci2": "[0.23, 0.38]",
    "paired.fid.delta1": "-7.7", "paired.fid.ci1": "[-13.1, -2.6]",
    **{f"paired.{k}.ci": v for k, v in {
        "ir": "+0.308 [+0.227, +0.384]", "aesthetic": "+0.17 [+0.12, +0.22]", "clip": "+1.45 [+1.12, +1.80]",
        "pick": "+0.74 [+0.61, +0.86]", "hps2": "+1.13 [+0.95, +1.31]", "hps21": "+2.51 [+2.12, +2.90]",
        "musiq": "+6.0 [+4.2, +7.8]", "musiqc": "+10.8 [+9.4, +12.2]", "clipiqa": "+5.3 [+3.9, +6.7]",
        "arniqa": "+3.9 [+3.1, +4.9]", "is": "-0.05 [-0.48, +0.62]", "fid": "-7.7 [-13.1, -2.6]",
        "kid": "-0.98 [-3.23, +1.47]", "fidp": "-3.8 [-7.5, +1.3]", "kidp": "-2.29 [-4.60, +0.05]"}.items()},
    **{f"paired.{k}.{side}": v for k, (a, s) in {
        "ir": ("0.921", "1.228"), "aesthetic": ("6.27", "6.44"), "clip": ("27.69", "29.14"),
        "pick": ("22.36", "23.09"), "hps2": ("28.62", "29.75"), "hps21": ("27.16", "29.67"),
        "musiq": ("39.4", "45.4"), "musiqc": ("50.4", "61.2"), "clipiqa": ("67.7", "73.0"),
        "arniqa": ("67.4", "71.3"), "is": ("6.66", "6.62"), "fid": ("145.8", "138.1"), "kid": ("7.17", "6.19"),
        "fidp": ("102.9", "99.1"), "kidp": ("11.40", "9.11")}.items() for side, v in (("area", a), ("sca", s))},
    # abstract, Table tab:main caption, Section 4.1 and Limitations
    "speed.sca_s": "71.1", "speed.area_s": "202.4", "speed.full_s": "186.5", "speed.hrdit_s": "108.6",
    "speed.ratio": "2.85×", "speed.hrdit_ratio": "1.5×",
    # Table tab:latency, Section 4.3, Appendix latency paragraph, Figure latency caption
    "latency.full.2048": "22.9", "latency.area.2048": "41.0", "latency.sca.2048": "18.2",
    "latency.full.3072": "73.9", "latency.area.3072": "114.3", "latency.sca.3072": "40.2",
    "latency.full.4096": "186.5", "latency.area.4096": "202.4", "latency.sca.4096": "71.1",
    "latency.full.6144": "762.7", "latency.area.6144": "482.3", "latency.sca.6144": "161.6",
    "latency.full.8192": "not run", "latency.area.8192": "952.0", "latency.sca.8192": "287.6",
    "latency.tokens.2048": "16,384", "latency.tokens.3072": "36,864", "latency.tokens.4096": "65,536",
    "latency.tokens.6144": "147,456", "latency.tokens.8192": "262,144",
    "latency.exp.sca": "1.00", "latency.exp.area": "1.12",
    "latency.at6144.vs_full": "4.7×", "latency.at6144.vs_area": "3.0×", "latency.at8192.vs_area": "3.31×",
    "latency.speedup_area.8192": "3.31×",
    "latency.reduction_area.6144": "66.5%", "latency.reduction_area.8192": "69.8%",
    "latency.growth.sca": "15.8×", "latency.growth.pixels": "16×",
    # Section 4.3 Coverage, Table krea_6144 title, Table app_wide, Limitations
    "sizes.6144.n_better": "14", "sizes.6144.n_total": "15", "sizes.6144.n": "100",
    "sizes.2048x6144.n_better": "9", "sizes.2048x6144.n_worse": "2",
    "sizes.2048x6144.worse_metrics": "aesthetic score, patch FID",
    "sizes.2048x4096.n_worse": "4", "sizes.2048x4096.worse_metrics": "aesthetic score, CLIP-IQA, patch FID, patch KID",
    "sizes.2048x4096.time_sca": "38.1", "sizes.2048x4096.time_area": "112.4",
    "sizes.2048x6144.time_sca": "55.8", "sizes.2048x6144.time_area": "150.5",
    # Section 4.5, Table tab:generality, Table tab:phase2_general, Table krea_6144, Section 4.2
    "general.hrgen.n": "1000", "general.hrgen.n_better": "10", "general.hrgen.n_total": "10",
    "general.hrgen.ir": "+0.818 [0.768, 0.870]", "general.hrgen.ir_delta": "+0.818",
    "general.hrgen.hps21": "+4.93", "general.hrgen.musiqc": "+13.89",
    "general.seed1.n_better": "10", "general.seed1.ir": "+0.274 [+0.204, +0.349]",
    "general.seed1.ir_delta": "+0.274", "general.seed1.hps21": "+2.28", "general.seed1.musiqc": "+11.51",
    "general.seed2.n_better": "10", "general.seed2.ir": "+0.250 [+0.183, +0.320]",
    "general.seed2.ir_delta": "+0.250", "general.seed2.hps21": "+2.50", "general.seed2.musiqc": "+12.06",
    "general.flux2.n": "50", "general.flux2.n_better": "9", "general.flux2.n_total": "10",
    "general.flux2.ir_full": "0.75", "general.flux2.ir_sca": "1.20", "general.flux2.ir": "+0.456 [+0.320, +0.598]",
    "general.flux2.ir_delta": "+0.456", "general.flux2.hps21": "+1.84", "general.flux2.musiqc": "+21.44",
    "general.krea.n": "195", "general.krea.n_better": "6", "general.krea.n_total": "10",
    "general.krea.ir": "-0.052 [-0.105, +0.000]", "general.krea.ir_delta": "-0.052",
    "general.krea.hps21": "+0.63", "general.krea.musiqc": "+4.23",
    "general.krea.ir_1000": "-0.052 [-0.103, -0.001]", "general.krea.fidp": "+8.1 [5.5, 12.8]",
    "general.krea.fid": "+1.4 [-3.0, 7.0]", "general.krea.speedup": "3.3×",
    "general.krea_table.n_better": "6", "general.krea_table.n_worse": "2",
    "general.krea_table.worse_metrics": "ImageReward, patch FID",
    "general.prog.time_prog": "49.5", "general.prog.time_hrdit": "110.4",
    "general.prog.ir": "-0.02 [-0.09, 0.03]", "general.prog.hps21": "+0.08 [-0.17, 0.32]",
    "general.prog.pick": "-0.04 [-0.13, 0.05]", "general.prog.n_worse": "0", "general.prog.n_better": "3",
    "general.prog.better_metrics": "HPSv2, MUSIQ, ARNIQA",
    # Section 3 (analysis), Figure pi_heads, Table ablation (Err.), Appendix C
    "calib.n_heads": "1368", "calib.n_prompts": "32", "calib.res": "1024", "calib.n_prompts_1280": "16",
    "calib.split_half.s2": "0.989", "calib.split_half.s4": "0.986", "calib.split_half.r1280": "0.974",
    "calib.cross_half": "0.0576", "calib.in_sample": "0.0574",
    "calib.err.s2.per_head": "0.057", "calib.err.s2.switch": "0.092", "calib.err.s2.per_layer": "0.071",
    "calib.err.s2.shared": "0.077", "calib.err.s4.per_head": "0.200", "calib.err.s4.switch": "0.261",
    "calib.err.s4.per_layer": "0.223", "calib.err.s4.shared": "0.230", "calib.err.s4.global": "0.627",
    "calib.err.s4.local": "0.261",
    "calib.transfer.s2_to_s4": "0.205", "calib.transfer.s4_own": "0.200",
    "calib.transfer.r1024_to_1280": "0.0589", "calib.transfer.r1280_own": "0.0585",
    "calib.transfer.max_penalty": "5.45%", "calib.per_layer_penalty": "12 to 24%",
    "calib.pi.std_range": "0.24 to 0.26", "calib.pi.mean_range": "0.64 to 0.74",
    "calib.pi.frac_gt95": "11.7%", "calib.pi.frac_lt05": "2.9%",
    # Limitations and Appendix failures
    "failures.dup_n": "5", "failures.dup_total": "11",
    # Section 4 Models and hyperparameters, Table hparams, Section 2
    "config.steps": "28", "config.opening_steps": "8", "config.gate_sigma": "0.89", "config.window": "64",
    "config.tile": "8", "config.guidance": "4.5", "config.shift": "1.15", "config.pi_scale": "4",
    "config.blocks": "57", "config.heads_per_block": "24", "config.sigma_bins": "14",
    "compute.gh200_hours": "123.3",
}
# 2026-10-30: the paper now reports the final operator (sca_v2); quotes from paper/main.tex and its generated
# snippets (figures/snippets/main_tables, app_tables) of delivery delivery_32xjiemn. Older quotes of the operator
# without the schedule are dropped where the paper no longer prints them.
for _k in [k for k in PAPER_QUOTES if k.startswith(("general.hrgen.", "general.seed", "general.krea", "general.prog.",
                                                     "latency.growth"))]:
    PAPER_QUOTES.pop(_k)   # 2026-10-31: the paper no longer prints the HRGen-1K values without the schedule
PAPER_QUOTES.update({
    **{f"main.sca.{k}": v for k, v in zip(("time_clean", "time_camp", "ir", "hps21", "pick", "clip", "musiqc", "fid",
                                           "fidp"), ("77.1", "78.4", "1.25", "29.73", "23.12", "29.14", "60.1", "140.2",
                                                     "98.4"))},
    **{f"main.{m}.dup": v for m, v in (("full", "7"), ("dype", "17"), ("sega", "12"), ("area", "4"), ("sca", "26"),
                                       ("ultraimage", "5"), ("hrdit", "11"))},
    "paired.n_better": "12", "paired.n_unresolved": "3", "paired.n_total": "15", "paired.n_worse": "0",
    "paired.ir.headline": "+0.33 [0.25, 0.41]", "paired.ir.delta2": "+0.33", "paired.ir.ci2": "[0.25, 0.41]",
    "paired.fid.headline": "-5.6 [-10.6, -0.4]", "paired.fid.delta1": "-5.6", "paired.fid.ci1": "[-10.6, -0.4]",
    **{f"paired.{k}.ci": v for k, v in {
        "ir": "+0.327 [+0.247, +0.405]", "aesthetic": "+0.15 [+0.10, +0.20]", "clip": "+1.45 [+1.13, +1.82]",
        "pick": "+0.76 [+0.64, +0.88]", "hps2": "+1.15 [+0.96, +1.34]", "hps21": "+2.56 [+2.20, +2.97]",
        "musiq": "+5.6 [+3.7, +7.4]", "musiqc": "+9.7 [+8.2, +11.1]", "clipiqa": "+4.6 [+3.1, +6.1]",
        "arniqa": "+3.4 [+2.5, +4.4]", "is": "-0.22 [-0.65, +0.39]", "fid": "-5.6 [-10.6, -0.4]",
        "kid": "-1.35 [-3.17, +0.49]", "fidp": "-4.4 [-7.9, +0.4]", "kidp": "-2.53 [-4.57, -0.60]"}.items()},
    **{f"paired.{k}.sca": v for k, v in {
        "ir": "1.248", "aesthetic": "6.42", "clip": "29.14", "pick": "23.12", "hps2": "29.77", "hps21": "29.73",
        "musiq": "45.0", "musiqc": "60.1", "clipiqa": "72.3", "arniqa": "70.7", "is": "6.44", "fid": "140.2",
        "kid": "5.82", "fidp": "98.4", "kidp": "8.87"}.items()},
    "speed.sca_s": "77.1", "speed.ratio": "2.63×", "speed.hrdit_ratio": "1.4×", "speed.overhead": "8%",
    "latency.sca.2048": "19.8", "latency.sca.3072": "44.3", "latency.sca.4096": "77.1", "latency.sca.6144": "174.7",
    "latency.sca.8192": "309.7", "latency.exp.sca": "0.99", "latency.reduction_area.6144": "63.8%",
    "latency.reduction_area.8192": "67.5%", "latency.speedup_area.8192": "3.07×", "latency.at8192.vs_area": "3.07×",
    "latency.at6144.vs_full": "4.4×", "latency.sca_nb.4096": "71.1",
    "sizes.2048x6144.n_better": "7", "sizes.2048x6144.n_worse": "2",
    "sizes.2048x6144.worse_metrics": "aesthetic score, patch FID", "sizes.2048x6144.time_sca": "60.5",
    "sizes.2048x4096.n_better": "3", "sizes.2048x4096.n_worse": "3",
    "sizes.2048x4096.worse_metrics": "aesthetic score, CLIP score, CLIP-IQA", "sizes.2048x4096.time_sca": "41.1",
    "general.hrgen.n": "1000", "general.hrgen.n_better": "10", "general.hrgen.n_total": "10",
    "general.hrgen.ir": "+0.764 [+0.714, +0.816]", "general.hrgen.ir_delta": "+0.764", "general.hrgen.hps21": "+4.60",
    "general.hrgen.musiqc": "+12.56",
    "general.seed1.ir_delta": "+0.306", "general.seed1.hps21": "+2.49", "general.seed1.musiqc": "+10.27",
    "general.seed2.ir_delta": "+0.264", "general.seed2.hps21": "+2.53", "general.seed2.musiqc": "+10.94",
    "general.krea.n": "195", "general.krea.ir_delta": "+0.020", "general.krea.hps21": "+0.91",
    "general.krea.musiqc": "+2.69", "general.krea.n_better": "7", "general.krea.ir_1000": "+0.020 [-0.021, 0.060]",
    "general.krea.fidp": "+3.4 [1.6, 7.1]", "general.krea.fid": "+1.0 [-3.6, 6.0]",
    "general.krea_table.n_better": "8", "general.krea_table.n_worse": "1", "general.krea_table.worse_metrics": "patch FID",
    "general.prog.time_prog": "50.7", "general.prog.time_hrdit": "110.4", "general.prog.n_worse": "1",
    "general.prog.worse_metrics": "CLIP-IQA", "general.prog.better_metrics": "HPSv2, MUSIQ, ARNIQA",
    "dup.sca_v2.pct": "26%", "dup.sca_v2.k": "25", "dup.sca_v2.n": "95", "dup.sca_nb.pct": "57%", "dup.area.pct": "4%",
    "dup.sigma.pct": "27%", "dup.sca_v2.lo": "18.5", "dup.sca_v2.hi": "36.0", "dup.sca_v2.seed_range": "26 to 28%",
    "dup.sca_nb.seed_range": "49 to 57%", "dup.sca_v2_2048x4096.pct": "24%", "dup.sca_v2_2048x6144.pct": "33%",
    "dup.sca_nb_2048x4096.pct": "43%", "dup.sca_nb_2048x6144.pct": "54%", "dup.area_2048x4096.pct": "18%",
    "dup.area_2048x6144.pct": "6%", "dup.sca_v2_prog.pct": "6%", "dup.sca_prog.pct": "8%",
    "dup.a46_sca_g93.k": "20", "dup.a46_sca_nb.k": "15", "dup.a46_sca_g85.k": "7", "dup.a46_abl_globalonly.k": "1",
    "dup.a46_dup_i_shrink.k": "4", "dup.a46_sca_nb.n": "26", "dup.a46_dup_i_shrink.pct": "15%",
    "config.bridge_tokens": "1,024", "config.win_sched": "128, 112, 96, 80", "config.wide_steps": "4",
    "config.native_steps": "16", "config.bridge_cells": "8", "config.bridge_k": "16",
})
PAPER_QUOTES.pop("latency.at6144.vs_area")  # the rewritten paper no longer prints the 6144^2 AReA ratio
# 2026-11-07: the paper's headline is progressive SCA (paper/main.tex and figures/snippets/main_tables of that day,
# clean latency of sweep 3 included). The page no longer shows duplication rates, so their quotes are dropped, as are
# the quotes of the earlier progressive-versus-HRDiT keys (general.prog.*), now prog.vs.hrdit.*.
for _k in [k for k in PAPER_QUOTES if k.startswith(("dup.", "failures.", "general.prog.")) or k.endswith(".dup")]:
    PAPER_QUOTES.pop(_k)
PAPER_QUOTES.update({
    "main.hrdit.steps": "30+17+10", "main.hrdit.time_clean": "107.0", "speed.hrdit_s": "107.0",
    **{f"main.prog.{k}": v for k, v in zip(("steps", "time_clean", "time_camp", "ir", "hps21", "pick", "clip", "musiqc",
                                            "fid", "fidp"), ("30+17+10", "49.1", "50.7", "1.32", "31.02", "23.41",
                                                             "29.39", "54.1", "141.9", "107.7"))},
    # abstract, introduction, Section 4.1 and 4.3 (headline.tex macros hlProg*, hlSecProg, hlSpeedProg*)
    "prog.clean.prog": "49.1", "prog.clean.hrdit": "107.0", "prog.clean.area": "202.4", "prog.clean.area_x": "4.1×",
    "prog.clean.hrdit_x": "2.2×", "prog.clean.hrdit_pct": "46%",
    "prog.camp.prog": "50.7", "prog.camp.hrdit": "110.4", "prog.camp.area": "264.4", "prog.camp.area_x": "5.2×",
    "prog.camp.hrdit_pct": "46%",
    **{f"prog.vs.area.{k}.ci": v for k, v in {
        "ir": "+0.40 [0.32, 0.49]", "pick": "+1.05 [0.93, 1.19]", "hps21": "+3.86 [3.41, 4.30]",
        "musiqc": "+3.7 [2.0, 5.4]", "musiq": "+2.3 [0.5, 3.9]", "clip": "+1.70 [1.33, 2.07]"}.items()},
    "prog.set.area.fid.ci": "-3.9 [-8.5, 3.1]", "prog.set.area.fidp.ci": "+4.8 [0.1, 10.6]",
    **{f"prog.vs.hrdit.{k}.ci": v for k, v in {
        "ir": "-0.02 [-0.09, 0.03]", "hps21": "+0.10 [-0.14, 0.35]", "pick": "-0.05 [-0.14, 0.04]",
        "hps2": "+0.38 [0.24, 0.51]", "arniqa": "+2.9 [2.3, 3.4]", "musiq": "+1.3 [0.2, 2.4]",
        "clipiqa": "-0.9 [-1.8, -0.1]", "musiqc": "+0.6 [-0.7, 1.8]"}.items()},
    **{f"prog.vs.sca.{k}.ci": v for k, v in {
        "ir": "+0.07 [0.01, 0.13]", "pick": "+0.29 [0.20, 0.38]", "hps21": "+1.29 [0.99, 1.59]",
        "musiq": "-3.4 [-5.1, -1.6]", "musiqc": "-6.0 [-7.3, -4.6]"}.items()},
    "prog.set.sca.fid.ci": "+1.6 [-2.3, 7.3]", "prog.set.sca.fidp.ci": "+9.2 [5.1, 13.2]",
    # 16:38 regeneration: set-level intervals against every baseline (prog195_vs_baselines/cfid)
    "prog.set.hrdit.fid.ci": "+3.4 [-1.8, 7.2]", "prog.set.hrdit.fidp.ci": "+4.6 [0.6, 7.7]",
    # 16:48 regeneration: progressive SCA on seed label 1 (Table generality)
    "prog.later.seed1.ir_ci2": "+0.37 [0.29, 0.46]", "prog.later.seed1.hps21_ci": "+3.67 [3.26, 4.06]",
    "prog.later.seed1.pick_ci": "+1.00 [0.88, 1.12]",
    **{f"prog.vs.{k}.n_better": v for k, v in (("area", "10"), ("dype", "10"), ("sega", "10"), ("full", "10"),
                                                ("ultraimage", "9"))},
    "prog.vs.ultraimage.unresolved_metrics": "aesthetic",
    "prog.vs.hrdit.better_metrics": "HPSv2, MUSIQ, ARNIQA", "prog.vs.hrdit.worse_metrics": "CLIP-IQA",
    "prog.cfg.base_steps": "30", "prog.cfg.steps_2048": "17", "prog.cfg.steps_4096": "10",
    "prog.cfg.sigma_2048": "0.81", "prog.cfg.sigma_4096": "0.61",
    # Section 3 and Table variance_split (hlVar*)
    "heads.var.head": "97.5", "heads.var.noise": "2.5", "heads.var.avg": "0.7", "heads.var.krea_head": "97.6",
    "heads.var.flux2_head": "99.1", "heads.var.curve_lo": "0.57", "heads.var.curve_hi": "0.62",
    "heads.var.min_head": "96.9",
    "prog.vs.area.all_word": "all ten", "config.cost_x_4096": "256", "config.first_wide_step": "9",
    "config.last_wide_step": "12", "config.grid_side_4096": "256", "calib.key_ratio": "16", "abl.n_prompts": "46",
    "abl.v0.ir_drop": "1.33", "calib.n_prompts_krea": "16", "calib.n_prompts_flux2": "16",
})
# key -> reason, for quotes that legitimately differ from the display (rounding source etc.)
EXPLAINED: dict[str, str] = {
    "prog.vs.ultraimage.unresolved_metrics": "same metric: the page names it 'aesthetic score', the paper's sentence "
                                             "'aesthetic'",
}
# numbers the paper prints that no results file holds; reported, never emitted
PAPER_ONLY: dict[str, tuple] = {}

# ---------------------------------------------------------------------------------------------------------
NUM: dict[str, dict] = {}


def rel(path) -> str:
    return Path(os.path.relpath(os.path.abspath(path), REPO)).as_posix()


def put(key: str, value, display: str, src, how: str) -> None:
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{key}: non-finite value")
    srcs = list(src) if isinstance(src, (list, tuple)) else [src]
    NUM[key] = {"value": value, "display": display, "src": " + ".join(dict.fromkeys(rel(s) for s in srcs)),
                "how": how}


def num(x: float, d: int) -> str:
    return f"{x:.{d}f}".replace("-", MINUS)


def sgn(x: float, d: int) -> str:
    return f"{x:+.{d}f}".replace("-", MINUS)


def ci_text(v: float, lo: float, hi: float, d: int) -> str:
    """'+0.31 [0.23, 0.38]': signed point and plain bounds, as the paper's running text prints intervals."""
    return f"{sgn(v, d)} [{num(lo, d)}, {num(hi, d)}]"


def times(x: float, d: int) -> str:
    return f"{x:.{d}f}{TIMES}"


def dp(d: int) -> str:
    return f"{d} decimal" + ("" if d == 1 else "s")


def rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _int(x):
    return int(float(x)) if x not in (None, "") else None


def _float(x):
    return float(x) if x not in (None, "") else None


class Run:
    """results/<run>: base/extra summaries and eval.py paired deltas (1000 prompt draws, method minus the run's
    reference). As paper/figures/src/data195.py does, FID, KID and their patch versions come from the corrected
    crop bank: corrected_fidp_ci (prompt and reference-cluster bootstrap) when archived, else corrected_fidp
    points without intervals. A run with neither (e2_6144) keeps eval.py's values, as gen_late_tables.py does."""

    def __init__(self, run: str):
        self.name, self.dir = run, RES / run
        self.S: dict[tuple, dict] = {}
        self.D: dict[tuple, dict] = {}
        for name in ("base_summary.csv", "extra_summary.csv"):
            p = self.dir / name
            for r in rows(p) if p.exists() else []:
                self.S[(r["method"], r["metric"])] = dict(value=float(r["value"]), n=_int(r.get("n_prompts")),
                                                          res=r.get("res"), src=p)
        refs = set()
        self.delta_paths = []
        for name, sub in (("base_paired_deltas.csv", "base/paired_deltas.csv"),
                          ("extra_paired_deltas.csv", "extra/extra_paired_deltas.csv")):
            p = self.dir / name
            if not p.exists() and run in V2_RUNS:
                p = EVAL / run / sub
                if not p.exists():
                    raise FileNotFoundError(f"{run}: eval.py paired deltas missing ({p})")
            if p.exists():
                self.delta_paths.append(p)
            for r in rows(p) if p.exists() else []:
                assert r["delta_definition"] == "method_minus_reference", (p, r)
                refs.add(r["reference_method"])
                self.D[(r["method"], r["metric"])] = dict(v=float(r["value"]), lo=_float(r["ci_low"]),
                                                          hi=_float(r["ci_high"]), ref=r["reference_method"],
                                                          n=_int(r["n_prompts"]), res=r["res"], src=[p])
        if len(refs) > 1:
            raise ValueError(f"{run}: several references {refs}")
        self.ref = refs.pop() if refs else None
        self.corrected = None
        for sub in ("corrected_fidp_ci", "corrected_fidp"):
            if (self.dir / sub / "summary.csv").exists():
                self.corrected = self.dir / sub
                break
        if self.corrected is None:
            return
        with_ci = self.corrected.name == "corrected_fidp_ci"
        for r in rows(self.corrected / "summary.csv"):
            if r["metric"] in DIST:
                self.S[(r["method"], r["metric"])] = dict(value=float(r["value"]), n=_int(r.get("n_prompts")),
                                                          res=r.get("res"), src=self.corrected / "summary.csv")
        pd_path = self.corrected / "paired_deltas.csv"
        for r in rows(pd_path) if pd_path.exists() else []:
            if r["metric"] not in DIST:
                continue
            if r["reference_method"] != self.ref:
                raise ValueError(f"{run}: corrected reference {r['reference_method']} != {self.ref}")
            key = (r["method"], r["metric"])
            base = dict(v=float(r["value"]), ref=self.ref, n=_int(r["n_prompts"]), res=r["res"], src=[pd_path])
            if r["ci_low"] and r["ci_high"]:
                self.D[key] = dict(base, lo=float(r["ci_low"]), hi=float(r["ci_high"]))
            elif r["metric"] in ("fid", "kid") and key in self.D:  # historical eval.py interval, corrected point
                self.D[key] = dict(base, lo=self.D[key]["lo"], hi=self.D[key]["hi"], src=[pd_path] + self.D[key]["src"])
            else:
                self.D[key] = dict(base, lo=None, hi=None)
        if not with_ci:
            for key in [k for k in self.D if k[1] in ("patch_fid", "patch_kid")]:
                self.D[key] = dict(self.D[key], lo=None, hi=None)

    def delta_files(self) -> list:
        extra = [self.corrected / "paired_deltas.csv"] if self.corrected is not None else []
        return self.delta_paths + extra

    def delta(self, method: str, metric: str, against: str) -> dict:
        """method minus `against` from this run's deltas; negates exactly when the stored delta is the reverse."""
        if self.ref == against and (method, metric) in self.D:
            return self.D[(method, metric)]
        if self.ref == method and (against, metric) in self.D:
            d = self.D[(against, metric)]
            lo, hi = d["lo"], d["hi"]
            return dict(d, v=-d["v"], lo=None if hi is None else -hi, hi=None if lo is None else -lo, negated=True)
        raise KeyError(f"{self.name}: no delta {method} - {against} for {metric}")


def pair_file(path: Path, a: str, b: str) -> dict:
    """{metric: dict(v, lo, hi, n, src)} from a paired_bootstrap.py CSV (a minus b, 2000 prompt draws)."""
    out = {}
    for r in rows(path):
        if r["method"] != a or r["reference"] != b:
            raise ValueError(f"{path} holds {r['method']} vs {r['reference']}, expected {a} vs {b}")
        out[r["metric"]] = dict(v=float(r["delta"]), lo=float(r["ci_low"]), hi=float(r["ci_high"]),
                                n=_int(r["n_prompts"]), seed=r.get("seed"), src=path)
    return out


def verdict(v, lo, hi, lower_better: bool) -> str:
    if lo is None or hi is None:
        return "no interval"
    if lo > 0 or hi < 0:
        return "better" if ((v < 0) if lower_better else (v > 0)) else "worse"
    return "unresolved"


def tally(prefix: str, verdicts: dict, src, how: str) -> None:
    """n_better, n_worse, n_unresolved, n_total and the metric lists for one comparison."""
    better = [k for k, x in verdicts.items() if x == "better"]
    worse = [k for k, x in verdicts.items() if x == "worse"]
    unres = [k for k, x in verdicts.items() if x != "better" and x != "worse"]
    rule = "; significant when the 95% paired interval excludes zero, better when it favours SCA"
    for name, ks in (("n_better", better), ("n_worse", worse), ("n_unresolved", unres)):
        put(f"{prefix}.{name}", len(ks), str(len(ks)), src, f"{how}: count {name[2:]}{rule}")
    put(f"{prefix}.n_total", len(verdicts), str(len(verdicts)), src, f"{how}: number of metrics compared")
    for name, ks in (("better_metrics", better), ("worse_metrics", worse), ("unresolved_metrics", unres)):
        label = ", ".join(METRICS[k][1] for k in ks) or "none"
        put(f"{prefix}.{name}", label, label, src, f"{how}: metrics {name.split('_')[0]}, Table app_4k order")


# ---------------------------------------------------------------------------------------------------------
def load_latency() -> dict:
    """(res, method) -> dict(vals, src): timed images without warm-ups; a cell comes from the first file that
    holds it and must agree image for image with any later file to that file's 0.01 s rounding."""
    cells: dict = {}
    for path in (LAT_CURVE, LAT_BATCH1):
        rs = rows(path)
        idc = "prompt_id" if "prompt_id" in rs[0] else "image"
        cur = defaultdict(list)
        for r in rs:
            if "warmup_verified" in r and r["warmup_verified"].strip().lower() != "true":
                raise ValueError(f"{path}: unverified warm-up ordering")
            if r[idc].startswith(WARMUP):
                continue
            cur[(r["res"], r["method"])].append(float(r["seconds"]))
        for k, v in cur.items():
            if k not in cells:
                cells[k] = dict(vals=v, src=path)
                continue
            a, b = sorted(cells[k]["vals"]), sorted(v)
            if len(a) != len(b) or any(abs(x - y) > 0.005 + 1e-6 for x, y in zip(a, b)):
                raise ValueError(f"latency cell {k} differs between {cells[k]['src'].name} and {path.name}")
    for r in rows(LAT_SUMMARY):
        k = (r["res"], r["method"])
        if abs(statistics.median(cells[k]["vals"]) - float(r["median_seconds"])) > 1e-9:
            raise ValueError(f"latency median {k} disagrees with {LAT_SUMMARY.name}")
    return cells


def load_latency_v2() -> dict:
    """(res, method) -> dict(vals, src) from the second clean sweep (sca_v2 and the re-timed sca_nb), warm-ups
    dropped; every median must equal results/latency_curve_v2/summary.csv."""
    cells = defaultdict(list)
    for r in rows(LAT_V2_SAMPLES):
        if r["warmup_verified"].strip().lower() != "true":
            raise ValueError(f"{LAT_V2_SAMPLES}: unverified warm-up ordering")
        if r["prompt_id"].startswith(WARMUP):
            continue
        cells[(r["res"], r["method"])].append(float(r["seconds"]))
    out = {k: dict(vals=v, src=LAT_V2_SAMPLES) for k, v in cells.items()}
    for r in rows(LAT_V2_SUMMARY):
        k = (r["res"], r["method"])
        if abs(statistics.median(out[k]["vals"]) - float(r["median_seconds"])) > 1e-9 or int(r["n"]) != len(out[k]["vals"]):
            raise ValueError(f"latency median {k} disagrees with {LAT_V2_SUMMARY.name}")
    return out


def load_v2_sidecars() -> list:
    """(config, sigmas) of every complete sca_v2 4096x4096 generation sidecar (195 images)."""
    out = []
    for path in sorted(V2_SIDECARS.glob("*.json")):
        sc = json.loads(path.read_text())
        if sc.get("complete") and sc["config"].get("method") == "sca_v2":
            out.append((sc["config"], sc.get("sigmas")))
    if len(out) != 195:
        raise ValueError(f"{V2_SIDECARS}: {len(out)} complete sca_v2 sidecars, expected 195")
    return out


def load_prog_sidecars() -> list:
    """config of every complete sca_v2_prog 4096x4096 generation sidecar (195 images, out/dupfix_prog), with the
    first sigma of each stage and the number of steps on which the gate was active."""
    out = []
    for path in sorted(PROG_SIDECARS.glob("*.json")):
        sc = json.loads(path.read_text())
        if sc.get("complete") and sc["config"].get("method") == PROG:
            out.append(dict(sc["config"], _stage_sigma0=[s[0] for s in sc["stage_sigmas"]],
                            _stage_n=[len(s) for s in sc["stage_sigmas"]], _gate_active=sc["gate_active_steps"]))
    if len(out) != 195:
        raise ValueError(f"{PROG_SIDECARS}: {len(out)} complete {PROG} sidecars, expected 195")
    return out


def load_hrdit_steps() -> tuple:
    """(stage steps, stage guidance, n images) from the sidecars of the HRDiT images eval/prog195_sca_v2 scored."""
    pngs = sorted({r["path"] for r in rows(PROG_EVAL_PER_IMAGE) if r["method"] == "hrdit_dev_author"})
    if len(pngs) != 195:
        raise ValueError(f"{PROG_EVAL_PER_IMAGE}: {len(pngs)} HRDiT images, expected 195")
    prof = [json.loads(Path(p[:-4] + ".json").read_text())["config"]["profile"] for p in pngs]
    steps = unique((tuple(p["steps"]) for p in prof), "HRDiT stage steps")
    guid = unique((tuple(p["guidance"]) for p in prof), "HRDiT stage guidance")
    return steps, guid, len(pngs)


def load_prog_latency() -> dict:
    """method -> (median seconds, n timed) of the 4096x4096 rows of results/prog_latency (sweep 3), or {}."""
    if not PROG_LAT.exists():
        return {}
    out = {}
    for r in rows(PROG_LAT):
        if r["res"] == "4096x4096":
            out[r["method"]] = (float(r["median_s"]), int(r["n_timed"]))
    return out


def prog_setlevel(base: str):
    """({metric: dict(v, lo, hi)} of progressive SCA minus `base` for FID, KID and their patch versions, sources) from
    the corrected bank with 1000 draws over prompts and 168 reference identities, exactly as gen_main_tables.py's
    prog_setlevel; (None, reason) while no such file holds the pair."""
    out = {}
    if PROG_CFID_ALL.exists():
        for r in rows(PROG_CFID_ALL):
            if r["method"] == base and r["reference_method"] == PROG and r["metric"] in DIST:
                assert r["bootstrap_valid"] == "1000" and r["n_reference_prompts"] == "168", r
                v, lo, hi = float(r["value"]), float(r["ci_low"]), float(r["ci_high"])
                out[r["metric"]] = dict(v=-v, lo=-hi, hi=-lo)
        if out:
            return out, [PROG_CFID_ALL]
    if base in PROG_CFID:
        for r in rows(PROG_CFID[base]):
            if r["method"] == PROG and r["reference_method"] == base and r["metric"] in DIST:
                assert r["bootstrap_valid"] == "1000" and r["n_reference_prompts"] == "168", r
                out[r["metric"]] = dict(v=float(r["value"]), lo=float(r["ci_low"]), hi=float(r["ci_high"]))
        return out, [PROG_CFID[base]]
    return None, [PROG_CFID_ALL]


def load_sidecars() -> dict:
    """(method, res) -> list of (config, sigmas) from the archived generation sidecars."""
    out = defaultdict(list)
    with gzip.open(SIDECARS, "rt") as fh:
        for line in fh:
            sc = json.loads(line)["sidecar"]
            cfg = sc["config"]
            out[(cfg.get("method"), cfg.get("res"))].append((cfg, sc.get("sigmas")))
    return out


def unique(values, what: str):
    vals = set(values)
    if len(vals) != 1:
        raise ValueError(f"{what}: expected one value across sidecars, found {sorted(map(str, vals))}")
    return vals.pop()


# ---------------------------------------------------------------------------------------------------------
def build() -> None:
    runs = {name: Run(name) for name in ("a4k195_main", "a4k195_panel", "e2_6144", "a4k195_2048x4096",
                                         "a4k195_2048x6144", "krea195") + tuple(V2_RUNS)}
    lat = load_latency()
    lat2 = load_latency_v2()
    sidecars = load_sidecars()
    v2cars = load_v2_sidecars()
    main, main2 = runs["a4k195_main"], runs[MAIN_V2]
    sc_src = SIDECARS

    # 1. Table tab:main (SCA = the final operator, sca_v2, from results/a4k195_main_sca_v2) ----------------
    ns = {e["n"] for (m, _), e in main.S.items()} | {e["n"] for (m, _), e in main2.S.items()}
    put("main.n_prompts", unique(ns, "a4k195_main n_prompts"), str(unique(ns, "n")),
        [main.dir / "base_summary.csv", main2.dir / "base_summary.csv"],
        "n_prompts of every row of results/a4k195_main and results/a4k195_main_sca_v2 summaries (4096x4096)")
    steps_src = {"full": "full", "dype": "dype", "sega": "sega_official", "area": "area"}
    for key, method in steps_src.items():
        s = unique((c["steps"] for c, _ in sidecars[(method, "4096x4096")]), f"{method} steps")
        put(f"main.{key}.steps", s, str(s), sc_src,
            f"config.steps of every method={method}, res=4096x4096 generation sidecar")
    s = unique((c["steps"] for c, _ in v2cars), "sca_v2 steps")
    put("main.sca.steps", s, str(s), V2_SIDECARS, f"config.steps of all {len(v2cars)} sca_v2 4096x4096 sidecars")
    ui = [c["ultraimage"]["call"] for c, _ in sidecars[("ultraimage", "4096x4096")]]
    s1 = unique((c["num_inference_steps1"] for c in ui), "ultraimage stage 1 steps")
    s2 = unique((c["num_inference_steps2"] for c in ui), "ultraimage stage 2 steps")
    put("main.ultraimage.steps", f"{s1}+{s2}", f"{s1}+{s2}", sc_src,
        "config.ultraimage.call.num_inference_steps1 + num_inference_steps2 of every method=ultraimage, "
        "res=4096x4096 sidecar")
    hsteps, hguid, n_h = load_hrdit_steps()
    txt = "+".join(str(x) for x in hsteps)
    put("main.hrdit.steps", txt, txt, PROG_EVAL_PER_IMAGE,
        f"config.profile.steps of the sidecars of the {n_h} HRDiT images listed in eval/{PROG_RUN}/base/per_image.csv "
        "(identical), joined with +")
    pcars = load_prog_sidecars()
    pst = unique((tuple([c["progressive"]["base_steps"]] + list(c["progressive"]["highres_steps"])) for c in pcars),
                 "progressive stage steps")
    if unique((tuple(c["_stage_n"]) for c in pcars), "progressive stage sigmas") != pst:
        raise ValueError("progressive sidecars: stage_sigmas lengths differ from base_steps + highres_steps")
    txt = "+".join(str(x) for x in pst)
    put("main.prog.steps", txt, txt, PROG_SIDECARS,
        f"config.progressive.base_steps + highres_steps of all {len(pcars)} {PROG} 4096x4096 sidecars (identical; "
        "equal to the lengths of stage_sigmas)")
    if pst != hsteps:
        raise ValueError(f"progressive SCA stage steps {pst} differ from HRDiT's {hsteps}")
    plat = load_prog_latency()
    for key, method, run_name in TABLE1:
        run = runs[run_name]
        for mk, dec in TABLE1_METRICS:
            metric, _, scale, _, _ = METRICS[mk]
            e = run.S[(method, metric)]
            assert e["n"] == 195 and e["res"] == "4096x4096", (run_name, method, metric, e)
            v = e["value"] * scale
            put(f"main.{key}.{mk}", v, num(v, dec), e["src"],
                f"value of method={method}, metric={metric}, res=4096x4096, n_prompts=195{UNITS[scale]}; "
                f"{dp(dec)}")
        e = run.S[(method, "seconds")]
        put(f"main.{key}.time_camp", e["value"], num(e["value"], 1), e["src"],
            f"value of method={method}, metric=seconds (campaign mean per image on shared nodes), "
            f"res=4096x4096; 1 decimal")
        cell = (lat2 if method == "sca_v2" else lat).get(("4096x4096", method))
        if method in (PROG, "hrdit_dev_author") and method in plat:
            # Table 1 takes HRDiT and progressive SCA from sweep 3, timed back to back (paper, 2026-11-07)
            med, n_t = plat[method]
            put(f"main.{key}.time_clean", med, num(med, 1), PROG_LAT,
                f"median_s of res=4096x4096, method={method} in results/prog_latency (sweep 3, {n_t} timed prompts "
                "after a warm-up); 1 decimal")
        elif method == PROG:
            put(f"main.{key}.time_clean", "pending", "pending", PROG_LAT,
                "results/prog_latency/latency_summary.csv absent: progressive SCA's clean latency has not landed")
        elif cell:
            med = statistics.median(cell["vals"])
            put(f"main.{key}.time_clean", med, num(med, 1), cell["src"],
                f"median of the {len(cell['vals'])} timed rows res=4096x4096, method={method} (warm-up rows "
                f"dropped); 1 decimal")
        else:
            put(f"main.{key}.time_clean", "n/m", "n/m", [LAT_CURVE, LAT_BATCH1],
                f"no clean batch-1 row for method={method} at 4096x4096 in either latency CSV: not measured")

    # 2. SCA minus AReA at 4096^2, the 15 metrics of Table app_4k (final operator) ---------------------------
    verdicts = {}
    for mk in FIFTEEN:
        metric, label, scale, dec, lower = METRICS[mk]
        d = main2.delta("sca_v2", metric, "area")
        assert d["n"] == 195 and d["lo"] is not None, (mk, d)
        a, s = main2.S[("area", metric)], main2.S[("sca_v2", metric)]
        v, lo, hi = (x * scale for x in (d["v"], d["lo"], d["hi"]))
        src = d["src"]
        how = (f"{metric}: SCA (sca_v2) minus AReA, exact negation of the area minus sca_v2 row "
               f"(reference_method=sca_v2, 1000 prompt draws{', corrected crop bank with reference-cluster resampling' if metric in DIST else ''}){UNITS[scale]}")
        put(f"paired.{mk}.delta", v, sgn(v, dec), src, f"{how}; point; {dp(dec)}")
        put(f"paired.{mk}.lo", lo, num(lo, dec), src, f"{how}; 2.5th percentile; {dp(dec)}")
        put(f"paired.{mk}.hi", hi, num(hi, dec), src, f"{how}; 97.5th percentile; {dp(dec)}")
        put(f"paired.{mk}.ci", ci_text(v, lo, hi, dec), ci_text(v, lo, hi, dec), src, f"{how}; point [lo, hi]")
        sign = -1.0 if lower else 1.0
        p = [100.0 * sign * x / a["value"] for x in (d["v"], d["lo"], d["hi"])]
        pct_how = (f"{how}; 100 * sign * delta / AReA mean ({a['value']:.6g}), sign -1 for lower-is-better, "
                   f"as fig_paired_area.py")
        psrc = src + [a["src"]]
        put(f"paired.{mk}.pct", p[0], sgn(p[0], 1) + "%", psrc, f"{pct_how}; point; 1 decimal")
        put(f"paired.{mk}.pct_lo", min(p[1:]), num(min(p[1:]), 1) + "%", psrc, f"{pct_how}; lower end; 1 decimal")
        put(f"paired.{mk}.pct_hi", max(p[1:]), num(max(p[1:]), 1) + "%", psrc, f"{pct_how}; upper end; 1 decimal")
        verdicts[mk] = verdict(d["v"], d["lo"], d["hi"], lower)
        sig = verdicts[mk] in ("better", "worse")
        put(f"paired.{mk}.sig", "yes" if sig else "no", "yes" if sig else "no", src,
            f"{how}; yes when the 95% interval excludes zero")
        put(f"paired.{mk}.verdict", verdicts[mk], verdicts[mk], src, f"{how}; better / worse / unresolved for SCA")
        for side, e, m in (("area", a, "area"), ("sca", s, "sca_v2")):
            x = e["value"] * scale
            put(f"paired.{mk}.{side}", x, num(x, dec), e["src"],
                f"value of method={m}, metric={metric}, res=4096x4096{UNITS[scale]}; {dp(dec)} (Table app_4k)")
    # the Section 4.1 quotes, at the decimals the running text prints (2 for ImageReward, 1 for FID)
    for mk, dec, txt in (("ir", 2, "ImageReward"), ("fid", 1, "FID (corrected bank, prompt and reference-cluster "
                                                               "bootstrap)")):
        metric, _, scale, _, _ = METRICS[mk]
        d = main2.delta("sca_v2", metric, "area")
        v, lo, hi = (x * scale for x in (d["v"], d["lo"], d["hi"]))
        how = f"{txt} SCA minus AReA at 4096x4096 with its 95% paired interval; {dp(dec)} as Section 4.1 prints it"
        put(f"paired.{mk}.headline", ci_text(v, lo, hi, dec), ci_text(v, lo, hi, dec), d["src"], how)
        put(f"paired.{mk}.delta{dec}", v, sgn(v, dec), d["src"], f"{how}; point")
        bounds = f"[{num(lo, dec)}, {num(hi, dec)}]"
        put(f"paired.{mk}.ci{dec}", bounds, bounds, d["src"], f"{how}; [lo, hi]")
    tally("paired", verdicts, main2.delta_files(),
          "SCA (sca_v2) minus AReA at 4096x4096 over the 15 metrics of Table app_4k")

    # 3. headline speed (clean batch-1 medians at 4096^2; SCA from the second sweep) ---------------------------
    med = {(r, m): statistics.median(c["vals"]) for (r, m), c in lat.items()}
    srcs = {(r, m): c["src"] for (r, m), c in lat.items()}
    for (r, m), c in lat2.items():   # sca_v2 and the re-timed sca_nb of results/latency_curve_v2
        med[(r, m + "@v2")], srcs[(r, m + "@v2")] = statistics.median(c["vals"]), c["src"]
    k4 = "4096x4096"
    SCA = "sca_v2@v2"
    for key, m in (("sca_s", SCA), ("sca_nb_s", "sca_nb@v2"), ("area_s", "area"), ("full_s", "full"),
                   ("hrdit_s", "hrdit_dev_author")):
        if m == "hrdit_dev_author" and m in plat:   # the paper's HRDiT clean time is sweep 3's once it exists
            put(f"speed.{key}", plat[m][0], num(plat[m][0], 1), PROG_LAT,
                f"median_s of res=4096x4096, method={m} in results/prog_latency (sweep 3); 1 decimal")
            continue
        put(f"speed.{key}", med[(k4, m)], num(med[(k4, m)], 1), srcs[(k4, m)],
            f"median clean batch-1 seconds, res=4096x4096, method={m.replace('@v2', ' (second sweep)')}; 1 decimal")
    r = med[(k4, "area")] / med[(k4, SCA)]
    put("speed.ratio", r, times(r, 2), [srcs[(k4, "area")], srcs[(k4, SCA)]],
        "AReA median (first sweep) / SCA (sca_v2, second sweep) median clean seconds at 4096x4096; 2 decimals")
    r = med[(k4, "full")] / med[(k4, SCA)]
    put("speed.ratio_full", r, times(r, 1), [srcs[(k4, "full")], srcs[(k4, SCA)]],
        "full-attention median / SCA (sca_v2) median clean seconds at 4096x4096; 1 decimal")
    o = 100.0 * (med[(k4, SCA)] / med[(k4, "sca_nb@v2")] - 1.0)
    put("speed.overhead", o, f"{o:.0f}%", srcs[(k4, SCA)],
        "100 * (sca_v2 / sca_nb - 1), medians of the second sweep at 4096x4096 (timed back to back); 0 decimals")
    camp = {"sca_v2": main2.S[("sca_v2", "seconds")]["value"], "ultraimage": main.S[("ultraimage", "seconds")]["value"],
            "hrdit_dev_author": runs["a4k195_panel"].S[("hrdit_dev_author", "seconds")]["value"]}
    for key, m, src in (("hrdit_ratio", "hrdit_dev_author", runs["a4k195_panel"].dir / "base_summary.csv"),
                        ("ultraimage_ratio", "ultraimage", main.dir / "base_summary.csv")):
        r = camp[m] / camp["sca_v2"]
        put(f"speed.{key}", r, times(r, 1), [src, main2.dir / "base_summary.csv"],
            f"campaign mean seconds of method={m} / sca_v2 at 4096x4096 (Table tab:main camp. column); 1 decimal")

    # 4. latency sweep (SCA = sca_v2 of the second sweep; AReA and full attention from the first) ----------------
    grid = unique((tuple(c["grid"]) for c, _ in v2cars), "sca_v2 4096 grid")
    patch = unique((4096 // grid[0], 4096 // grid[1]), "pixels per token")  # 16: VAE factor 8, 2x2 packing
    lat_methods = [("sca", SCA), ("sca_nb", "sca_nb@v2"), ("area", "area"), ("full", "full")]
    for side in SIDES:
        res = f"{side}x{side}"
        tok = (side // patch) ** 2
        put(f"latency.tokens.{side}", tok, f"{tok:,}", [LAT_V2_SAMPLES, V2_SIDECARS],
            f"N = (W/p)(H/p) image tokens for res={res}, p = 4096 / config.grid of the sca_v2 4096x4096 "
            f"sidecars = {patch} pixels per token")
        for key, m in lat_methods:
            if (res, m) in med:
                n_rows = len((lat2 if m.endswith("@v2") else lat)[(res, m.replace("@v2", ""))]["vals"])
                put(f"latency.{key}.{side}", med[(res, m)], num(med[(res, m)], 1), srcs[(res, m)],
                    f"median of the {n_rows} timed rows res={res}, method={m.replace('@v2', ' (second sweep)')} "
                    f"(warm-up rows dropped); 1 decimal")
            else:
                put(f"latency.{key}.{side}", "not run", "not run", [LAT_CURVE, LAT_BATCH1, LAT_V2_SAMPLES],
                    f"no row res={res}, method={m} in the latency CSVs")
        if (res, "area") in med:
            r = med[(res, "area")] / med[(res, SCA)]
            put(f"latency.speedup_area.{side}", r, times(r, 2), [srcs[(res, "area")], srcs[(res, SCA)]],
                f"AReA median / SCA (sca_v2) median at {res}; 2 decimals")
            red = 100.0 * (1.0 - med[(res, SCA)] / med[(res, "area")])
            put(f"latency.reduction_area.{side}", red, num(red, 1) + "%", [srcs[(res, "area")], srcs[(res, SCA)]],
                f"100 * (1 - SCA median / AReA median) at {res}; 1 decimal")
    for key, m in lat_methods:
        rs = [f"{s}x{s}" for s in SIDES if (f"{s}x{s}", m) in med]
        x = np.log([(int(r.split("x")[0]) // patch) * (int(r.split("x")[1]) // patch) for r in rs])
        y = np.log([med[(r, m)] for r in rs])
        b = float(np.polyfit(x, y, 1)[0])
        put(f"latency.exp.{key}", b, num(b, 2), sorted({srcs[(r, m)] for r in rs}, key=str),
            f"least-squares slope of log median seconds on log N over {len(rs)} sizes "
            f"({', '.join(r.split('x')[0] for r in rs)}), method={m}; 2 decimals (fig_latency.py fit)")
    for side, other, key, dec in ((6144, "full", "at6144.vs_full", 1), (6144, "area", "at6144.vs_area", 1),
                                  (8192, "area", "at8192.vs_area", 2)):
        res = f"{side}x{side}"
        r = med[(res, other)] / med[(res, SCA)]
        put(f"latency.{key}", r, times(r, dec), [srcs[(res, other)], srcs[(res, SCA)]],
            f"{other} median / SCA (sca_v2) median clean seconds at {res}; {dp(dec)} as Section 4.3 prints it")
    g = med[("8192x8192", SCA)] / med[("2048x2048", SCA)]
    put("latency.growth.sca", g, times(g, 1), srcs[("8192x8192", SCA)],
        "SCA (sca_v2) median at 8192x8192 / its median at 2048x2048; 1 decimal")
    px = (8192 * 8192) / (2048 * 2048)
    put("latency.growth.pixels", px, times(px, 0), LAT_CURVE, "pixel ratio of the 8192x8192 and 2048x2048 cells")

    # 5. other sizes, SCA vs AReA (sca_v2 where its run exists; 6144^2 has only the operator without the schedule)
    for size, run_name, m_sca in (("6144", "e2_6144", "sca_nb"), ("2048x6144", "a4k195_2048x6144_sca_v2", "sca_v2"),
                                  ("2048x4096", "a4k195_2048x4096_sca_v2", "sca_v2")):
        run = runs[run_name]
        v_ = {}
        files = set()
        for mk in FIFTEEN:
            metric, _, _, _, lower = METRICS[mk]
            d = run.delta(m_sca, metric, "area")
            files.update(d["src"])
            v_[mk] = verdict(d["v"], d["lo"], d["hi"], lower)
        n = unique((run.delta(m_sca, METRICS[mk][0], "area")["n"] for mk in FIFTEEN), f"{run_name} n")
        src = sorted(files, key=str)
        put(f"sizes.{size}.n", n, str(n), src, f"n_prompts of the {m_sca} minus area rows in results/{run_name}")
        put(f"sizes.{size}.operator", m_sca, "final operator" if m_sca == "sca_v2" else "without the schedule", src,
            f"SCA label of results/{run_name}: sca_v2 = final operator, sca_nb = SCA without the schedule")
        tally(f"sizes.{size}", v_, src, f"SCA ({m_sca}) minus AReA at {size} over the 15 metrics ("
              + ("eval.py intervals, 1000 prompt draws" if run.corrected is None else
                 "eval.py intervals; FID/KID and patch versions from corrected_fidp_ci") + ")")
        for key, m in (("time_sca", m_sca), ("time_area", "area")):
            e = run.S[(m, "seconds")]
            put(f"sizes.{size}.{key}", e["value"], num(e["value"], 1), e["src"],
                f"campaign mean seconds per image, method={m}, results/{run_name}; 1 decimal")
        r = run.S[("area", "seconds")]["value"] / run.S[(m_sca, "seconds")]["value"]
        put(f"sizes.{size}.speedup", r, times(r, 1), run.dir / "base_summary.csv",
            f"AReA / SCA ({m_sca}) campaign mean seconds; 1 decimal")

    # 6. generality ------------------------------------------------------------------------------------------
    def per_image_block(prefix, P, n_expected, decs=(("ir_delta", "ir", 3), ("hps21", "hps21", 2),
                                                       ("musiqc", "musiqc", 2)), ci_dec=3):
        src = next(iter(P.values()))["src"]
        assert {P[METRICS[k][0]]["n"] for k in PER_IMAGE} == {n_expected}, (prefix, src)
        put(f"{prefix}.n", n_expected, str(n_expected), src, "n_prompts of the paired rows")
        tally(prefix, {k: verdict(P[METRICS[k][0]]["v"], P[METRICS[k][0]]["lo"], P[METRICS[k][0]]["hi"],
                                  METRICS[k][4]) for k in PER_IMAGE}, src,
              f"{rel(src)}: the ten per-image metrics, 2000 prompt draws")
        ir = P["ir"]
        put(f"{prefix}.ir", ci_text(ir["v"], ir["lo"], ir["hi"], ci_dec), ci_text(ir["v"], ir["lo"], ir["hi"], ci_dec),
            src, f"metric=ir delta [ci_low, ci_high]; {dp(ci_dec)}")
        for key, mk, dec in decs:
            metric, _, scale, _, _ = METRICS[mk]
            v = P[metric]["v"] * scale
            put(f"{prefix}.{key}", v, sgn(v, dec), src, f"metric={metric} delta{UNITS[scale]}; {dp(dec)} "
                                                          f"(Table tab:generality)")

    hr2 = RES / "hrgen1k_dev_sca_v2" / "sca_v2_vs_area.csv"
    if hr2.exists():
        per_image_block("general.hrgen", pair_file(hr2, "sca_v2", "area"), 1000)
        put("general.hrgen.status", "landed", "final operator", hr2, "results/hrgen1k_dev_sca_v2 exists")
    else:
        put("general.hrgen.status", "pending", "pending", [RES / "hrgen1k_dev_sca_v2"],
            "results/hrgen1k_dev_sca_v2/sca_v2_vs_area.csv absent: the final operator's HRGen-1K run has not landed")
    per_image_block("general.hrgen_prev", pair_file(RES / "hrgen1k_dev" / "sca_nb_vs_area.csv", "sca_nb", "area"),
                    1000)
    for seed in (1, 2):
        P = pair_file(RES / f"a4k195_seed{seed}_sca_v2" / "sca_v2_vs_area.csv", "sca_v2", "area")
        assert {p["seed"] for p in P.values()} == {str(seed)}
        per_image_block(f"general.seed{seed}", P, 195)
    f2 = pair_file(RES / "e8_flux2_sca_vs_full.csv", "flux2_sca", "flux2_full")
    per_image_block("general.flux2", f2, 50)
    fx = {(r["method"], r["metric"]): float(r["value"]) for p in ("flux2_4k_base_summary.csv",
                                                                   "flux2_4k_extra_summary.csv")
          for r in rows(RES / p)}
    for key, m in (("ir_full", "flux2_full"), ("ir_sca", "flux2_sca")):
        put(f"general.flux2.{key}", fx[(m, "ir")], num(fx[(m, "ir")], 2), RES / "flux2_4k_extra_summary.csv",
            f"value of method={m}, metric=ir, 4096x4096, 50 prompts; 2 decimals")
    for key, m in (("time_sca", "flux2_sca"), ("time_full", "flux2_full")):
        put(f"general.flux2.{key}", fx[(m, "seconds")], num(fx[(m, "seconds")], 1), RES / "flux2_4k_base_summary.csv",
            f"value of method={m}, metric=seconds (mean per image); 1 decimal")
    fr = fx[("flux2_sca", "seconds")] / fx[("flux2_full", "seconds")]
    put("general.flux2.time_frac", fr, num(fr, 2), RES / "flux2_4k_base_summary.csv",
        "SCA / full-attention mean seconds per image; 2 decimals")
    put("general.flux2.speedup", 1 / fr, times(1 / fr, 1), RES / "flux2_4k_base_summary.csv",
        "full-attention / SCA mean seconds per image; 1 decimal")

    krea = runs["krea195_sca_v2"]
    KM = "sca_v2_krea"
    per_image_block("general.krea", pair_file(krea.dir / "sca_v2_krea_vs_area.csv", KM, "area"), 195)
    d = krea.delta(KM, "ir", "area")
    put("general.krea.ir_1000", ci_text(d["v"], d["lo"], d["hi"], 3), ci_text(d["v"], d["lo"], d["hi"], 3), d["src"],
        f"metric=ir, {KM} minus area, eval.py 1000 prompt draws (Table krea_6144); 3 decimals")
    for key, metric in (("fidp", "patch_fid"), ("fid", "fid")):
        d = krea.delta(KM, metric, "area")
        put(f"general.krea.{key}", ci_text(d["v"], d["lo"], d["hi"], 1), ci_text(d["v"], d["lo"], d["hi"], 1),
            d["src"], f"metric={metric}, {KM} minus area, corrected bank, prompt and reference-cluster "
                      f"bootstrap (1000 draws); 1 decimal")
    kv = {}
    for mk in FIFTEEN:
        d = krea.delta(KM, METRICS[mk][0], "area")
        assert d["n"] == 195, (mk, d)
        kv[mk] = verdict(d["v"], d["lo"], d["hi"], METRICS[mk][4])
    tally("general.krea_table", kv, krea.delta_files(),
          f"SCA ({KM}) minus AReA on FLUX.1-Krea-dev, 195 prompts, the 15 metrics as Table krea_6144 marks them "
          "(eval.py 1000 prompt draws; FID/KID and patch versions from corrected_fidp_ci)")
    for key, m in (("time_sca", KM), ("time_area", "area")):
        e = krea.S[(m, "seconds")]
        put(f"general.krea.{key}", e["value"], num(e["value"], 1), e["src"],
            f"campaign mean seconds, method={m}, results/krea195_sca_v2; 1 decimal")
    r = krea.S[("area", "seconds")]["value"] / krea.S[(KM, "seconds")]["value"]
    put("general.krea.speedup", r, times(r, 1), krea.dir / "base_summary.csv",
        "AReA / SCA campaign mean seconds, FLUX.1-Krea-dev, 195 prompts; 1 decimal")

    # 6b. progressive SCA, the headline configuration (pairs, set level, times, schedule, pending results) -------
    build_prog(runs, pcars, hguid, plat, per_image_block)
    # 6c. Figure paired_area: both configurations minus AReA, the paper's construction --------------------------
    build_chart_pa(main)
    # 6d. reach is set by the head: variance of the attention share near the query, head against noise level -----
    build_heads()
    # 6e. Table 1 cell classes: rank within each group, significance against progressive SCA ------------------
    build_table1_classes()

    # 7. calibration -----------------------------------------------------------------------------------------
    j = json.loads((CALIB / "k1_1024.json").read_text())
    j1280 = json.loads((CALIB / "k1_1280b.json").read_text())
    audit_path = RES / "evidence" / "calibration_split_half_audit.json"
    audit = json.loads(audit_path.read_text())
    npz_path = CALIB / "k1_1024.npz"
    z = np.load(npz_path, allow_pickle=False)
    jp, j1280p = CALIB / "k1_1024.json", CALIB / "k1_1280b.json"
    blocks, heads, bins = z["s4_pi_head"].shape
    put("config.blocks", blocks, str(blocks), npz_path, "axis 0 of s4_pi_head (blocks x heads x sigma bins)")
    put("config.heads_per_block", heads, str(heads), npz_path, "axis 1 of s4_pi_head")
    put("config.sigma_bins", bins, str(bins), npz_path, "axis 2 of s4_pi_head")
    put("calib.n_heads", blocks * heads, str(blocks * heads), npz_path, "blocks x heads of the s4_pi_head table")
    put("calib.n_prompts", j["n_prompts"], str(j["n_prompts"]), jp, "n_prompts")
    side = int(j["res"].split("x")[0])
    put("calib.res", side, str(side), jp, f"res = {j['res']}; side length")
    put("calib.n_prompts_1280", j1280["n_prompts"], str(j1280["n_prompts"]), j1280p, f"n_prompts at {j1280['res']}")
    sh = {v["scale"]: v["correlation"] for v in audit["values"]}
    for s in (2, 4):
        assert abs(sh[s] - j["scales"][str(s)]["split_half_corr"]["per_head"]) < 1e-12
        put(f"calib.split_half.s{s}", sh[s], num(sh[s], 3), audit_path,
            f"values[scale={s}].correlation (per-head tables of disjoint prompt halves); 3 decimals")
    v = j1280["scales"]["2"]["split_half_corr"]["per_head"]
    put("calib.split_half.r1280", v, num(v, 3), j1280p, "scales.2.split_half_corr.per_head at 1280x1280; 3 decimals")
    v = j["scales"]["2"]["cross_half_rel_err_per_head"]
    put("calib.cross_half", v, num(v, 4), jp, "scales.2.cross_half_rel_err_per_head; 4 decimals")
    v = j["scales"]["2"]["rel_err"]["per_head"]
    put("calib.in_sample", v, num(v, 4), jp, "scales.2.rel_err.per_head; 4 decimals")
    names = {"per_head": "per_head", "per_layer": "per_layer", "shared": "uniform", "switch": "switch",
             "global": "glob", "local": "loc"}
    for s in (2, 4):
        for key, field in names.items():
            v = j["scales"][str(s)]["rel_err"][field]
            put(f"calib.err.s{s}.{key}", v, num(v, 3), jp, f"scales.{s}.rel_err.{field}; 3 decimals")
    t = j["transfer"]["1024x1024/s2 -> 1024x1024/s4"]
    put("calib.transfer.s2_to_s4", t["rel_err"], num(t["rel_err"], 3), jp,
        "transfer['1024x1024/s2 -> 1024x1024/s4'].rel_err; 3 decimals")
    put("calib.transfer.s4_own", t["own"], num(t["own"], 3), jp,
        "transfer['1024x1024/s2 -> 1024x1024/s4'].own (the scale-4 table's own error); 3 decimals")
    t = j1280["transfer"]["1024x1024/s2 -> 1280x1280/s2"]
    put("calib.transfer.r1024_to_1280", t["rel_err"], num(t["rel_err"], 4), j1280p,
        "transfer['1024x1024/s2 -> 1280x1280/s2'].rel_err; 4 decimals")
    put("calib.transfer.r1280_own", t["own"], num(t["own"], 4), j1280p,
        "transfer['1024x1024/s2 -> 1280x1280/s2'].own; 4 decimals")
    pen = max(100.0 * (t_["rel_err"] / t_["own"] - 1.0) for jj in (j, j1280) for t_ in jj["transfer"].values())
    put("calib.transfer.max_penalty", pen, num(pen, 2) + "%", [jp, j1280p],
        "max over every transfer pair in both files of 100 * (rel_err / own - 1); 2 decimals")
    lay = [100.0 * (jj["scales"][s]["rel_err"]["per_layer"] / jj["scales"][s]["rel_err"]["per_head"] - 1.0)
           for jj in (j, j1280) for s in jj["scales"]]
    txt = f"{min(lay):.0f} to {max(lay):.0f}%"
    put("calib.per_layer_penalty", txt, txt, [jp, j1280p],
        "range over calibration targets (1024 s2, 1024 s4, 1280 s2) of 100 * (per_layer / per_head rel_err - 1)")
    s2 = j["scales"]["2"]
    pop = [i for i, p in enumerate(j["populated_bins"]) if p]
    for key, field in (("std_range", "pi_head_std_over_heads_per_bin"), ("mean_range", "pi_head_mean_per_bin")):
        vals = [s2[field][i] for i in pop]
        txt = f"{min(vals):.2f} to {max(vals):.2f}"
        put(f"calib.pi.{key}", txt, txt, jp, f"min to max of scales.2.{field} over populated sigma bins; 2 decimals")
    put("calib.pi.frac_mid", 100 * s2["frac_pi_mid"], num(100 * s2["frac_pi_mid"], 0) + "%", jp,
        "100 * scales.2.frac_pi_mid (cells with pi in [0.2, 0.8]); 0 decimals")
    cells = z["s2_pi_head"][:, :, np.asarray(z["populated"], bool)]
    for key, frac, rule in (("frac_gt95", float((cells > 0.95).mean()), "> 0.95"),
                            ("frac_lt05", float((cells < 0.05).mean()), "< 0.05")):
        put(f"calib.pi.{key}", 100 * frac, num(100 * frac, 1) + "%", npz_path,
            f"percent of s2_pi_head cells {rule} over populated sigma bins; 1 decimal")

    # configuration constants from the sca_v2 4096^2 generation sidecars (out/dupfix/sca_v2) ----------------------
    for key, get, dec in (("steps", lambda c, s: c["steps"], None), ("guidance", lambda c, s: c["guidance"], 1),
                          ("gate_sigma", lambda c, s: c["sca"]["gate_sigma"], 2),
                          ("window", lambda c, s: c["sca"]["window"], None),
                          ("tile", lambda c, s: c["sca"]["tile"], None),
                          ("pi_scale", lambda c, s: c["sca"]["pi_scale"], None),
                          ("bridge_cells", lambda c, s: c["sca"]["bridge_cells"], None),
                          ("bridge_k", lambda c, s: c["sca"]["bridge_k"], None)):
        v = unique((get(c, s) for c, s in v2cars), f"sca_v2 {key}")
        put(f"config.{key}", v, str(v) if dec is None else num(v, dec), V2_SIDECARS,
            f"config{'' if key in ('steps', 'guidance') else '.sca'}.{key} of all {len(v2cars)} method=sca_v2, "
            f"res=4096x4096 sidecars (identical)")
    b = unique((c["sca"]["bridge_cells"] ** 2 * c["sca"]["bridge_k"] for c, _ in v2cars), "bridge tokens")
    put("config.bridge_tokens", b, f"{b:,}", V2_SIDECARS, "bridge_cells^2 * bridge_k of the sca_v2 sidecars")
    sched = unique((tuple(tuple(x) for x in c["sca"]["win_sched"]) for c, _ in v2cars), "win_sched")
    txt = ", ".join(str(w) for _, w in sched)
    put("config.win_sched", txt, txt, V2_SIDECARS, "window sides of config.sca.win_sched, in step order")
    put("config.win_sched_n", len(sched), str(len(sched)), V2_SIDECARS, "number of win_sched entries")
    put("config.win_first", sched[0][1], str(sched[0][1]), V2_SIDECARS, "first (widest) win_sched window side")
    put("config.win_last", sched[-1][1], str(sched[-1][1]), V2_SIDECARS, "last win_sched window side")
    side = max(grid)
    rat = [f"{side / w:.2f}".rstrip("0").rstrip(".") for _, w in sched]
    txt = ", ".join(rat[:-1]) + " and " + rat[-1]
    put("config.win_ratios", txt, txt, V2_SIDECARS,
        f"image-to-window ratios max(grid) / w = {side} / w of the win_sched windows at 4096x4096; 2 decimals, "
        "trailing zeros dropped")
    nat = f"{side / unique((c['sca']['window'] for c, _ in v2cars), 'w'):.0f}"
    put("config.native_ratio", int(nat), nat, V2_SIDECARS, f"{side} / native window side")
    v = unique((sum(1 for x in s[: c["steps"]] if x > c["sca"]["gate_sigma"]) for c, s in v2cars if s),
               "opening steps")
    put("config.opening_steps", v, str(v), V2_SIDECARS,
        "number of the sidecar's step sigmas above config.sca.gate_sigma (global-only opening phase), identical "
        "in every sca_v2 4096x4096 sidecar")
    lo = min(x for x, _ in sched)
    v = unique((sum(1 for x in s[: c["steps"]] if x <= lo) for c, s in v2cars if s), "native steps")
    put("config.native_steps", v, str(v), V2_SIDECARS,
        f"number of step sigmas at or below the last win_sched bound ({lo}): native {unique((c['sca']['window'] for c, _ in v2cars), 'w')}-token windows")
    wide = unique((sum(1 for x in s[: c["steps"]] if lo < x <= c["sca"]["gate_sigma"]) for c, s in v2cars if s),
                  "wide steps")
    put("config.wide_steps", wide, str(wide), V2_SIDECARS, "number of step sigmas in (last win_sched bound, gate]")
    op = NUM["config.opening_steps"]["value"]
    put("config.first_wide_step", op + 1, str(op + 1), V2_SIDECARS, "config.opening_steps + 1 (1-based step number)")
    put("config.last_wide_step", op + wide, str(op + wide), V2_SIDECARS,
        "config.opening_steps + config.wide_steps (1-based step number)")
    put("config.grid_side_4096", side, str(side), V2_SIDECARS, "max(config.grid) of the sca_v2 4096x4096 sidecars")
    n_nat = unique((c["sca"]["window"] for c, _ in v2cars), "native window") ** 2
    cx = (side * side / n_nat) ** 2
    put("config.cost_x_4096", cx, f"{cx:.0f}", V2_SIDECARS,
        f"(N / N0)^2: image tokens at 4096x4096 ({side}^2) over the native grid's (window^2 = {n_nat}), squared, the "
        "growth of full attention's query-key pairs")
    kr = unique((c["sca"]["pi_scale"] for c, _ in v2cars), "pi_scale") ** 2
    put("calib.key_ratio", kr, str(kr), V2_SIDECARS,
        "pi_scale^2: at emulated ratio 4 each branch sees r^2 times fewer keys than at the target")
    for key, name in (("krea", "krea_1024.json"), ("flux2", "flux2_1024.json")):
        jj = json.loads((CALIB / name).read_text())
        put(f"calib.n_prompts_{key}", jj["n_prompts"], str(jj["n_prompts"]), CALIB / name, f"n_prompts at {jj['res']}")
    e8 = RES / "e2_8192_base_summary.csv"
    n8 = unique((int(r["n_prompts"]) for r in rows(e8)), "8192^2 prompts")
    put("sizes.8192.n", n8, str(n8), e8, "n_prompts of every row of the 8192x8192 summary (exploratory, sca_nb and area)")
    ab = RES / "abl_base_summary.csv"
    n46 = unique((int(r["n_prompts"]) for r in rows(ab)), "ablation subset prompts")
    put("abl.n_prompts", n46, str(n46), ab, "n_prompts of every row of the ablation summary (the fixed subset)")
    v0 = pair_file(RES / "ablnb_abl_v0_deltas.csv", "abl_v0", "sca_nb")["ir"]
    assert v0["n"] == n46, v0
    put("abl.v0.ir", ci_text(v0["v"], v0["lo"], v0["hi"], 2), ci_text(v0["v"], v0["lo"], v0["hi"], 2), v0["src"],
        "metric=ir, abl_v0 (per-head mix from step 1, no opening) minus sca_nb; 2 decimals")
    put("abl.v0.ir_drop", -v0["v"], num(-v0["v"], 2), v0["src"], "minus the ir delta of abl_v0 minus sca_nb; 2 decimals")
    nb, nt = NUM["prog.vs.area.n_better"]["value"], NUM["prog.vs.area.n_total"]["value"]
    txt = f"all {WORDS[nt]}" if nb == nt else f"{WORDS[nb]} of {WORDS[nt]}"
    put("prog.vs.area.all_word", txt, txt, NUM["prog.vs.area.n_better"]["src"].split(" + ")[0],
        "prog.vs.area.n_better of n_total as words ('all ten' when equal), as the abstract prints it")
    shifts = [c.get("scheduler", {}).get("max_shift") for c, _ in sidecars[("full", "4096x4096")]]
    v = unique(shifts, "max_shift")
    put("config.shift", v, num(v, 2), sc_src,
        "config.scheduler.max_shift of every method=full, res=4096x4096 sidecar (shared sampler, mu_mode=max_shift)")

    # compute ------------------------------------------------------------------------------------------------
    cs = RES / "compute" / "summary.json"
    v = json.loads(cs.read_text())["attributable_resource_hours_lower_bound"]
    put("compute.gh200_hours", v, num(v, 1), cs, "attributable_resource_hours_lower_bound; 1 decimal")


# ---------------------------------------------------------------------------------------------------------
def prog_pairs(base: str) -> tuple:
    """({CSV metric: dict(v, lo, hi, n, seed, src)}, source) of progressive SCA minus `base`, per-image, 2000 draws."""
    src = PROG_PAIRS / f"{PROG}_vs_{base}.csv"
    P = pair_file(src, PROG, base)
    if {P[METRICS[k][0]]["n"] for k in PER_IMAGE} != {195}:
        raise ValueError(f"{src}: expected 195 prompts on every per-image metric")
    return P, src


def build_prog(runs: dict, pcars: list, hguid: tuple, plat: dict, per_image_block) -> None:
    """Progressive SCA: paired differences against every baseline, set-level intervals where they exist, campaign
    and clean times, the stage schedule from its sidecars, and the results the paper still lists as pending."""
    prog_run, panel, main = runs[PROG_RUN], runs["a4k195_panel"], runs["a4k195_main"]
    # per-image pairs (results/prog195_vs_baselines/sca_v2_prog_vs_<baseline>.csv)
    for key, base in PROG_VS:
        P, src = prog_pairs(base)
        verdicts = {}
        for mk in PER_IMAGE:
            metric, _, scale, _, lower = METRICS[mk]
            p = P[metric]
            v, lo, hi = (x * scale for x in (p["v"], p["lo"], p["hi"]))
            d = TEXT_DEC[mk]
            how = (f"{metric}: progressive SCA ({PROG}) minus {base}, per-image paired bootstrap over prompts (2000 "
                   f"draws, seed {p['seed']}){UNITS[scale]}")
            put(f"prog.vs.{key}.{mk}.delta", v, sgn(v, d), src, f"{how}; point; {dp(d)}")
            put(f"prog.vs.{key}.{mk}.ci", ci_text(v, lo, hi, d), ci_text(v, lo, hi, d), src,
                f"{how}; point [lo, hi]; {dp(d)}")
            verdicts[mk] = verdict(p["v"], p["lo"], p["hi"], lower)
            put(f"prog.vs.{key}.{mk}.verdict", verdicts[mk], verdicts[mk], src,
                f"{how}; better / worse / unresolved for progressive SCA")
        tally(f"prog.vs.{key}", verdicts, src,
              f"progressive SCA minus {base} over the ten per-image metrics (2000 prompt draws)")
    # set-level pairs: corrected FID, KID and their patch versions, 1000 draws, 168 reference identities
    for key, base in PROG_VS:
        sl, srcs = prog_setlevel(base)
        if sl is None:
            put(f"prog.set.{key}.status", "pending", "pending", srcs,
                f"no corrected set-level interval of {PROG} against {base} yet: {rel(PROG_CFID_ALL)} has not landed")
            continue
        put(f"prog.set.{key}.status", "landed", "landed", srcs, f"corrected set-level pairs of {PROG} against {base}")
        verdicts = {}
        for mk in SETLEVEL:
            metric, _, scale, _, lower = METRICS[mk]
            r = sl[metric]
            v, lo, hi = (x * scale for x in (r["v"], r["lo"], r["hi"]))
            d = TEXT_DEC[mk]
            how = (f"{metric}: progressive SCA minus {base}, corrected bank, 1000 draws over prompts and the 168 "
                   f"reference identities{UNITS[scale]}")
            put(f"prog.set.{key}.{mk}.delta", v, sgn(v, d), srcs, f"{how}; point; {dp(d)}")
            put(f"prog.set.{key}.{mk}.ci", ci_text(v, lo, hi, d), ci_text(v, lo, hi, d), srcs,
                f"{how}; point [lo, hi]; {dp(d)}")
            verdicts[mk] = verdict(r["v"], r["lo"], r["hi"], lower)
            put(f"prog.set.{key}.{mk}.verdict", verdicts[mk], verdicts[mk], srcs,
                f"{how}; better / worse / unresolved for progressive SCA (lower is better)")
        tally(f"prog.set.{key}", verdicts, srcs, f"progressive SCA minus {base} over FID, KID, patch FID, patch KID")
    # campaign means on shared nodes (Table 1 camp. column)
    camp = {"prog": prog_run.S[(PROG, "seconds")], "hrdit": panel.S[("hrdit_dev_author", "seconds")],
            "area": main.S[("area", "seconds")]}
    for k, m, run_ in (("prog", PROG, prog_run), ("hrdit", "hrdit_dev_author", panel), ("area", "area", main)):
        e = camp[k]
        put(f"prog.camp.{k}", e["value"], num(e["value"], 1), e["src"],
            f"campaign mean seconds per image, method={m}, results/{run_.name}; 1 decimal")
    x = camp["area"]["value"] / camp["prog"]["value"]
    put("prog.camp.area_x", x, times(x, 1), [camp["area"]["src"], camp["prog"]["src"]],
        "AReA / progressive SCA campaign mean seconds at 4096x4096; 1 decimal")
    x = 100.0 * camp["prog"]["value"] / camp["hrdit"]["value"]
    put("prog.camp.hrdit_pct", x, f"{x:.0f}%", [camp["prog"]["src"], camp["hrdit"]["src"]],
        "100 * progressive SCA / HRDiT campaign mean seconds at 4096x4096; 0 decimals")
    # clean batch-1 latency, sweep 3 (progressive SCA, HRDiT and AReA timed back to back on one GPU)
    names = (("prog", PROG), ("hrdit", "hrdit_dev_author"), ("area", "area"))
    if all(m in plat for _, m in names):
        gpus = {r["gpu"].split(",")[1].strip() for r in rows(PROG_LAT) if r["res"] == "4096x4096"}
        if len(gpus) != 1 or "GH200" not in rows(PROG_LAT)[0]["gpu"]:
            raise ValueError(f"{PROG_LAT}: expected one GH200 for all rows, found {gpus}")
        for k, m in names:
            med, n_t = plat[m]
            put(f"prog.clean.{k}", med, num(med, 1), PROG_LAT,
                f"median_s of res=4096x4096, method={m} (sweep 3: {n_t} timed prompts after a warm-up, one GH200, "
                "all three methods back to back); 1 decimal")
        put("prog.clean.n_timed", unique((plat[m][1] for _, m in names), "n_timed"),
            str(unique((plat[m][1] for _, m in names), "n_timed")), PROG_LAT, "n_timed of every 4096x4096 row")
        P_, H_, A_ = (plat[m][0] for _, m in names)
        put("prog.clean.area_x", A_ / P_, times(A_ / P_, 1), PROG_LAT, "AReA / progressive SCA median_s, sweep 3; 1 decimal")
        put("prog.clean.hrdit_x", H_ / P_, times(H_ / P_, 1), PROG_LAT, "HRDiT / progressive SCA median_s, sweep 3; 1 decimal")
        put("prog.clean.hrdit_pct", 100 * P_ / H_, f"{100 * P_ / H_:.0f}%", PROG_LAT,
            "100 * progressive SCA / HRDiT median_s, sweep 3; 0 decimals")
        put("prog.clean.status", "landed", "landed", PROG_LAT, "results/prog_latency/latency_summary.csv exists")
    else:
        for k in ("prog", "hrdit", "area", "n_timed", "area_x", "hrdit_x", "hrdit_pct", "status"):
            put(f"prog.clean.{k}", "pending", "pending", PROG_LAT,
                "results/prog_latency/latency_summary.csv absent: the clean latency of progressive SCA has not landed")
    # the stage schedule (HRDiT's, kept as released) from the 195 progressive sidecars
    pg = lambda f, what: unique((f(c) for c in pcars), what)  # noqa: E731
    base_steps = pg(lambda c: c["progressive"]["base_steps"], "base_steps")
    hi_steps = pg(lambda c: tuple(c["progressive"]["highres_steps"]), "highres_steps")
    stages = pg(lambda c: tuple(c["progressive"]["stages"]), "stages")
    guid = pg(lambda c: tuple(c["progressive"]["guidance"]), "guidance")
    if guid != hguid:
        raise ValueError(f"progressive guidance {guid} differs from HRDiT's {hguid}")
    ratios = pg(lambda c: tuple(s["pi_scale"] for s in c["sca"]["stages"]), "pi_scale per stage")
    sca_res = pg(lambda c: tuple(s["res"] for s in c["sca"]["stages"]), "SCA stage sizes")
    sig0 = pg(lambda c: tuple(round(x, 2) for x in c["_stage_sigma0"]), "first sigma of each stage")
    gate = pg(lambda c: tuple(c["_gate_active"]), "gate-active steps per SCA stage")
    win = pg(lambda c: tuple(s["window"] for s in c["sca"]["stages"]), "SCA windows")
    bridge = pg(lambda c: tuple(s["bridge_cells"] ** 2 * s["bridge_k"] for s in c["sca"]["stages"]), "bridge tokens")
    attn = pg(lambda c: c["attention"], "attention")
    total = pg(lambda c: c["steps"], "steps")
    if stages != ("1024x1024", "2048x2048", "4096x4096") or sca_res != stages[1:] or attn != "stock@native+sca@highres":
        raise ValueError(f"unexpected progressive stages {stages}, SCA stages {sca_res}, attention {attn}")
    how = f"of all {len(pcars)} complete {PROG} 4096x4096 sidecars (identical), out/dupfix_prog"
    put("prog.cfg.base_steps", base_steps, str(base_steps), PROG_SIDECARS, f"config.progressive.base_steps {how}")
    for k, n in (("steps_2048", hi_steps[0]), ("steps_4096", hi_steps[1])):
        put(f"prog.cfg.{k}", n, str(n), PROG_SIDECARS, f"config.progressive.highres_steps {how}")
    put("prog.cfg.total_steps", total, str(total), PROG_SIDECARS, f"config.steps {how}")
    for k, g in zip(("g1024", "g2048", "g4096"), guid):
        put(f"prog.cfg.{k}", g, num(g, 1), PROG_SIDECARS, f"config.progressive.guidance {how}; 1 decimal")
    txt = ", ".join(num(g, 1) for g in guid[:-1]) + " and " + num(guid[-1], 1)
    put("prog.cfg.guidance", txt, txt, PROG_SIDECARS, f"config.progressive.guidance {how}")
    for k, r_ in zip(("ratio_2048", "ratio_4096"), ratios):
        put(f"prog.cfg.{k}", r_, str(r_), PROG_SIDECARS, f"config.sca.stages[].pi_scale {how} (calibrated ratio read)")
    for k, s_ in zip(("sigma_2048", "sigma_4096"), sig0[1:]):
        put(f"prog.cfg.{k}", s_, num(s_, 2), PROG_SIDECARS, f"first entry of stage_sigmas of that stage {how}; 2 decimals")
    put("prog.cfg.gate_steps", sum(gate), str(sum(gate)), PROG_SIDECARS,
        f"sum of gate_active_steps over the two SCA stages {how}: steps run on the global view alone")
    put("prog.cfg.window", unique(win, "window"), str(unique(win, "window")), PROG_SIDECARS,
        f"config.sca.stages[].window {how}")
    b = unique(bridge, "bridge")
    put("prog.cfg.bridge_tokens", b, f"{b:,}", PROG_SIDECARS, f"bridge_cells^2 * bridge_k of config.sca.stages[] {how}")
    # results the paper lists as pending: progressive SCA on seeds 1 and 2 and on HRGen-1K
    for k, (n_exp, cands) in PROG_LATER.items():
        hit = next((RES / c for c in cands if (RES / c).exists()), None)
        if hit is None:
            put(f"prog.later.{k}.status", "pending", "pending", [RES / c for c in cands],
                "none of these files exists yet: the run has not landed")
            continue
        put(f"prog.later.{k}.status", "landed", "landed", hit, f"{rel(hit)} exists")
        P = pair_file(hit, PROG, "area")
        if k.startswith("seed") and {p["seed"] for p in P.values()} != {k[-1]}:
            raise ValueError(f"{hit}: seed column {sorted({p['seed'] for p in P.values()})}, expected {k[-1]}")
        per_image_block(f"prog.later.{k}", P, n_exp)
        for key, mk in (("ir_ci2", "ir"), ("hps21_ci", "hps21"), ("pick_ci", "pick")):   # Table generality's cells
            metric, _, scale, _, _ = METRICS[mk]
            p = P[metric]
            txt = ci_text(p["v"] * scale, p["lo"] * scale, p["hi"] * scale, 2)
            put(f"prog.later.{k}.{key}", txt, txt, hit, f"metric={metric} delta [ci_low, ci_high]{UNITS[scale]}; "
                                                        "2 decimals, as Table generality prints it")


def build_chart_pa(main) -> None:
    """Figure paired_area as paper/figures/src/fig_paired_prog.py draws it: progressive and single-pass SCA minus
    AReA, x = 100 * sign * delta / AReA's mean (sign -1 when lower is better), 14 metrics without IS."""
    area = {}
    for name in ("base_summary.csv", "extra_summary.csv"):
        for r in rows(RES / "a4k195_main" / name):
            if r["method"] == "area":
                area[r["metric"]] = (float(r["value"]), RES / "a4k195_main" / name)
    for r in rows(CHART_PA_SET / "summary.csv"):
        if r["method"] == "area":   # corrected set-level means
            area[r["metric"]] = (float(r["value"]), CHART_PA_SET / "summary.csv")
    for s, method, path in CHART_PA_SERIES:
        P = pair_file(path, method, "area")
        D = {m: (d["v"], d["lo"], d["hi"], path) for m, d in P.items()}
        for r in rows(CHART_PA_SET / "paired_deltas.csv"):
            if r["reference_method"] == "area" and r["method"] == method and r["metric"] in DIST:
                assert r["bootstrap_valid"] == "1000" and r["n_reference_prompts"] == "168", r
                D[r["metric"]] = (float(r["value"]), float(r["ci_low"]), float(r["ci_high"]),
                                  CHART_PA_SET / "paired_deltas.csv")
        verdicts, files = {}, set()
        for mk in CHART_PA:
            metric, label, scale, dec, lower = METRICS[mk]
            v, lo, hi, src = D[metric]
            a, asrc = area[metric]
            files.update((src, asrc))
            sign = -1.0 if lower else 1.0
            x = [100.0 * sign * t / a for t in (v, lo, hi)]
            xl, xh = min(x[1:]), max(x[1:])
            vd = "better" if xl > 0 else ("worse" if xh < 0 else "unresolved")
            how = (f"{metric}: {method} minus AReA as percent of AReA's mean ({a:.6g}), sign -1 when lower is better "
                   "(fig_paired_prog.py)")
            put(f"chart.pa.{s}.{mk}.pct", x[0], sgn(x[0], 1) + "%", [src, asrc], f"{how}; point; 1 decimal")
            put(f"chart.pa.{s}.{mk}.pct_lo", xl, num(xl, 1) + "%", [src, asrc], f"{how}; lower end; 1 decimal")
            put(f"chart.pa.{s}.{mk}.pct_hi", xh, num(xh, 1) + "%", [src, asrc], f"{how}; upper end; 1 decimal")
            put(f"chart.pa.{s}.{mk}.verdict", vd, vd, [src, asrc], f"{how}; better / worse / unresolved")
            verdicts[mk] = vd
        tally(f"chart.pa.{s}", verdicts, sorted(files, key=str),
              f"{method} minus AReA over the 14 metrics of Figure paired_area")


def build_heads() -> None:
    """Reach is set by the head: variance of the share of each head's image attention within r tokens of the query
    over (block, head, noise bin) cells, split into head and noise level (results/heads_view/variance_split.csv)."""
    R = {(r["checkpoint"], r["res"], r["sigma_range"], r["quantity"], r["radius"]): r for r in rows(HEADS_VAR)}

    def row(ckpt, res="1024x1024", radius="16"):
        return R[(ckpt, res, "post_gate", "image_share", radius)]

    sel = "res=1024x1024, sigma_range=post_gate (after the first 8 steps), quantity=image_share, radius=16"
    r = row("flux1_dev")
    for key, val, d, what in (
            ("head", 100 * (1 - float(r["noise_within_head"])), 1, "100 * (1 - noise_within_head): the head's share"),
            ("noise", 100 * float(r["noise_within_head"]), 1, "100 * noise_within_head: the noise level's share"),
            ("avg", 100 * float(r["head_avg_curve"]), 1, "100 * head_avg_curve: the head-averaged curve's share"),
            ("within_block", 100 * float(r["head_within_layer"]), 1, "100 * head_within_layer"),
            ("between_blocks", 100 * float(r["layer"]), 1, "100 * layer"),
            ("curve_lo", float(r["curve_min"]), 2, "curve_min of the head-averaged curve"),
            ("curve_hi", float(r["curve_max"]), 2, "curve_max of the head-averaged curve")):
        put(f"heads.var.{key}", val, num(val, d), HEADS_VAR, f"checkpoint=flux1_dev, {sel}: {what}; {dp(d)}")
    put("heads.var.radius", int(r["radius"]), r["radius"], HEADS_VAR, f"checkpoint=flux1_dev, {sel}: radius")
    put("heads.var.n_heads", int(r["n_heads"]), r["n_heads"], HEADS_VAR, f"checkpoint=flux1_dev, {sel}: n_heads")
    for key, ckpt in (("krea_head", "flux1_krea_dev"), ("flux2_head", "flux2_dev")):
        v = 100 * (1 - float(row(ckpt)["noise_within_head"]))
        put(f"heads.var.{key}", v, num(v, 1), HEADS_VAR, f"checkpoint={ckpt}, {sel}: 100 * (1 - noise_within_head)")
    put("heads.var.flux2_n_heads", int(row("flux2_dev")["n_heads"]), row("flux2_dev")["n_heads"], HEADS_VAR,
        f"checkpoint=flux2_dev, {sel}: n_heads")
    cells = [(c, s, rad) for c, s in (("flux1_dev", "1024x1024"), ("flux1_dev", "1280x1280"),
                                      ("flux1_krea_dev", "1024x1024"), ("flux2_dev", "1024x1024")) for rad in ("8", "16")]
    v = min(100 * (1 - float(row(c, s, rad)["noise_within_head"])) for c, s, rad in cells)
    put("heads.var.min_head", v, num(math.floor(v * 10) / 10, 1), HEADS_VAR,
        "minimum over the rows of Table variance_split (FLUX.1-dev at 1024^2 and 1280^2, FLUX.1-Krea-dev, FLUX.2-dev; "
        "radius 8 and 16; post_gate) of 100 * (1 - noise_within_head); rounded down to 1 decimal, since the text "
        "reads 'at least'")


def build_table1_classes() -> None:
    """main.<row>.<column>.cls: "best" / "second" within the row's group (Table 1's bold and underline) and "sig-up" /
    "sig-dn" where the row is significantly better / worse than progressive SCA (its 95% paired interval)."""
    cols = [("time_clean", True, True), ("time_camp", True, False), ("ir", False, True), ("hps21", False, True),
            ("pick", False, True), ("clip", False, True), ("musiqc", False, True), ("fid", True, True),
            ("fidp", True, True)]
    rank = {}
    for group in TABLE1_GROUPS:
        for col, lower, ranked in cols:
            if not ranked:
                continue
            vals = [(NUM[f"main.{r}.{col}"]["value"], r) for r in group
                    if isinstance(NUM[f"main.{r}.{col}"]["value"], (int, float))]
            vals.sort(key=lambda t: t[0] if lower else -t[0])
            for (_, r), name in zip(vals, ("best", "second")):
                rank[(r, col)] = name
    for key, method, _ in TABLE1:
        for col, lower, ranked in cols:
            cls, why = [], []
            src = NUM[f"main.{key}.{col}"]["src"].split(" + ")
            if (key, col) in rank:
                cls.append(rank[(key, col)])
                why.append(f"{rank[(key, col)]} of its group on main.*.{col}")
            vk = None
            if key != "prog" and col in ("ir", "hps21", "pick", "clip", "musiqc"):
                vk = f"prog.vs.{key}.{col}.verdict"
            elif key != "prog" and col in ("fid", "fidp") and NUM[f"prog.set.{key}.status"]["value"] == "landed":
                vk = f"prog.set.{key}.{col}.verdict"
            vd = NUM[vk]["value"] if vk else None
            if vk:
                src += NUM[vk]["src"].split(" + ")
            if vd == "better":
                cls.append("sig-dn")
                why.append(f"progressive SCA is significantly better than this row ({vk})")
            elif vd == "worse":
                cls.append("sig-up")
                why.append(f"this row is significantly better than progressive SCA ({vk})")
            txt = " ".join(cls)
            put(f"main.{key}.{col}.cls", txt, txt, [REPO / s for s in src],
                "; ".join(why) or "no rank and no significant difference from progressive SCA")


# ---------------------------------------------------------------------------------------------------------
def norm(s: str) -> str:
    s = s.replace("−", "-").replace("×", "x").replace("\\times", "x").replace("{-}", "-")
    return re.sub(r"[\s$+]", "", s)


def check(quiet: bool) -> int:
    flagged, ok, explained = [], 0, 0
    lines = []
    for key, e in NUM.items():
        q = PAPER_QUOTES.get(key)
        if q is None:
            status = ""
        elif norm(q) == norm(e["display"]):
            status, ok = "ok", ok + 1
        elif key in EXPLAINED:
            status, explained = f"explained: {EXPLAINED[key]}", explained + 1
        else:
            status = "MISMATCH"
            flagged.append(key)
        cut = lambda s: s if len(s) <= 44 else s[:41] + "..."  # noqa: E731  (numbers.json holds the full text)
        lines.append((key, cut(e["display"]), cut(q) if q else "-", status))
    orphan = sorted(set(PAPER_QUOTES) - set(NUM))
    w = (max(len(x[0]) for x in lines), max(len(x[1]) for x in lines), max(len(x[2]) for x in lines))
    print(f"{'key':<{w[0]}}  {'display':<{w[1]}}  {'paper':<{w[2]}}  check")
    for key, disp, q, status in lines:
        if not quiet or status.startswith(("MISMATCH", "explained")):
            print(f"{key:<{w[0]}}  {disp:<{w[1]}}  {q:<{w[2]}}  {status}")
    print(f"\n{len(NUM)} keys; {sum(1 for x in lines if x[2] != '-')} quoted by the paper: {ok} ok, {explained} explained, "
          f"{len(flagged)} MISMATCH")
    for key in flagged:
        print(f"  MISMATCH {key}: display {NUM[key]['display']!r}, paper {PAPER_QUOTES[key]!r}")
    for key in orphan:
        print(f"  QUOTE WITHOUT KEY {key}: paper {PAPER_QUOTES[key]!r} (not emitted)")
    for key, (q, why) in PAPER_ONLY.items():
        print(f"  PAPER ONLY {key}: paper {q!r}; not emitted ({why})")
    return len(flagged) + len(orphan)


def write_outputs() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(NUM, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    disp = {k: e["display"] for k, e in NUM.items()}
    OUT_JS.write_text("// Generated by web/build/build_numbers.py from results/; do not edit. Key -> display string.\n"
                      "window.SCA_NUMBERS = " + json.dumps(disp, indent=1, ensure_ascii=True) + ";\n",
                      encoding="utf-8")
    print(f"wrote {rel(OUT_JSON)} and {rel(OUT_JS)} ({len(NUM)} keys)")


ATTR = r"""\sdata-num\s*=\s*(?P<q>["'])(?P<key>[^"']*)(?P=q)"""
TAG = re.compile(r"<(?P<tag>[A-Za-z][\w:-]*)(?P<attrs>[^<>]*?" + ATTR + r"[^<>]*)>(?P<text>[^<]*)</(?P=tag)\s*>")
ANY = re.compile(ATTR)
CLASS = re.compile(r"""\sclass\s*=\s*(["'])(.*?)\1""")
MANAGED = {"best", "second", "sig-up", "sig-dn"}   # classes owned by the "<key>.cls" entries


def with_classes(attrs: str, key: str) -> str:
    """attrs with the managed classes replaced by NUM[key + '.cls'] (other classes kept); unchanged without one."""
    if key + ".cls" not in NUM:
        return attrs
    want = NUM[key + ".cls"]["display"].split()
    m = CLASS.search(attrs)
    keep = [c for c in (m.group(2).split() if m else []) if c not in MANAGED]
    new = " ".join(keep + want)
    rest = attrs[:m.start()] + attrs[m.end():] if m else attrs
    return rest + (f' class="{new}"' if new else "")


def fill(path: str) -> None:
    """Rewrite the text of every inline <tag ... data-num="key" ...>TEXT</tag> with NUM[key]['display'], and its
    managed classes with NUM[key + '.cls'] where that key exists."""
    p = Path(path)
    with open(p, encoding="utf-8", newline="") as fh:
        text = fh.read()
    keys = [m["key"] for m in ANY.finditer(text)]
    unknown = sorted({k for k in keys if k not in NUM})
    if unknown:
        raise SystemExit(f"{path}: unknown data-num key(s): {', '.join(unknown)}")
    count = 0

    def sub(m):
        nonlocal count
        count += 1
        attrs = with_classes(m["attrs"], m["key"])
        return f"<{m['tag']}{attrs}>{html.escape(NUM[m['key']]['display'], quote=False)}</{m['tag']}>"

    new = TAG.sub(sub, text)
    if count != len(keys):
        raise SystemExit(f"{path}: {len(keys) - count} data-num element(s) hold nested tags or are not closed; "
                         f"only inline text content can be filled")
    if new != text:
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    print(f"filled {count} data-num element(s) in {path}" + ("" if new != text else " (already current)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fill", action="append", default=[], metavar="PATH",
                    help="HTML file whose data-num elements get their display strings (repeatable)")
    ap.add_argument("--quiet", action="store_true", help="print only flagged rows of the check table")
    ap.add_argument("--strict", action="store_true", help="exit 1 on an unexplained mismatch with the paper")
    args = ap.parse_args()
    build()
    write_outputs()
    bad = check(args.quiet)
    for path in args.fill:
        fill(path)
    if args.strict and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
