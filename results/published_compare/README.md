# Published-number comparison on the shared benchmark

Files
- `published.csv`: numbers copied from the papers. One row per (paper, table, size, method, metric).
  - AReA (arXiv 2610.17227v1): Table 1 (FLUX.1 columns, four sizes, p. 8), Table 2 (HRGen-1K, p. 8), and Table 6
    (multi-stage methods, App. F, p. 25). Every value was checked against the PDF text programmatically, with 0
    mismatches.
  - SEGA (arXiv 2605.22668v1): Table 1, FLUX block at 4096x4096 (p. 8). Its prompt count is not stated.
- `ours.csv`: our values, with the evaluation summary each one came from and its prompt count.
- `compare.csv`: each published value next to ours for the same method and metric, with the difference and a
  comparability flag.
- `build.py`: regenerates `ours.csv`, `compare.csv`, and `paper/figures/snippets/published_compare.tex`.
  Run `python3 -I results/published_compare/build.py`.

Protocol facts behind the table (details in `.agent/notes/eval_protocol_survey.md`)
- AReA evaluates all 195 Aesthetic-4K prompts against their 195 reference images (Sec. 4). It reports HRGen-1K on all
  1000 prompts (Table 2).
- AReA's paper says only "FLUX.1". Its released code defaults to FLUX.1-Krea-dev with 28 steps and guidance 4.5.
  SEGA, from the same group, states FLUX.1-Krea-dev (App. C).
  - Rerunning AReA's released code on Krea-dev reproduces its published CLIP, PickScore, aesthetic score, and MUSIQ
    within 0.1. On FLUX.1-dev it does not: ImageReward drops from 1.33 to 0.92.
  - The Krea-dev block is therefore the protocol-matched comparison. Our FLUX.1-dev block is the paper's own paired
    protocol.
- FID uses Inception features at 299^2 against all 195 reference rows. Those rows hold 168 distinct images, 27 of them
  duplicates, and we keep the official rows.
- Patch FID is not compared across papers. AReA and SEGA never define their FIDp crops, and our crop rule gives about
  115 for AReA where AReA reports 58.2.
- Latency is not compared across papers: AReA used an H100, we use a GH200.

Current coverage and what refreshes it
- Krea-dev rows cover 100 of 195 prompts (rows 4-103, results/e9_krea) until `taskq/hold_krea195` (38 generation
  tasks) and `hold_eval/z966_eval_krea195.sh` run.
- FLUX.1-dev rows use the 191-prompt summaries until `hold_eval/z960` and `z961` (195 prompts) finish.
- HRGen-1K rows use 200 prompts until `hold_hrgen1k_krea` and `z967`, or `hold_hrgen1k_dev` and `z968`, finish.
- Rows with fewer than the full prompt count carry a dagger in the LaTeX table and are not ranked. After each
  evaluation lands, rerun `build.py`.
