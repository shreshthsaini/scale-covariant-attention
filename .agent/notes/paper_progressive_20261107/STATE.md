# Paper owner state: progressive-SCA rewrite (2026-11-07, paper/ and web/ owner)

## Goal
Rewrite paper/ and web/ with progressive SCA (HRDiT's three-stage schedule + sca_v2 attention) as the headline
configuration and single-pass SCA as the fast mode; framing per positioning_20261105/recommendation.md; no
duplication content anywhere except the one limitations sentence the user dictated. Two render-review cycles,
<= 10 main pages, 0 overfull, 0 undefined, delivery build with fresh unzip, latest_delivery.md, HANDOVER Section 31,
web rebuilt, commits after each stage.

## Status and next action
- Done: read HANDOVER 24-30, recommendation, meta-review, results; split paper/main.tex -> main.tex + appendix.tex
  (builds 54 pp, 0 overfull, 0 undefined).
- In progress: stage 0 data (variance_split.csv, concept curves), Table 1 redesign in gen_main_tables.py, main text.
- Delegated: appendix agent (appendix.tex + appendix generators), figure agent (hero, prog gallery, zoom figure),
  web agent (after abstract is fixed).

## Results status (fold in as they land; watcher script in scratchpad watch_results.sh)
- LANDED: results/prog195_vs_baselines/sca_v2_prog_vs_{area,dype,full,hrdit_dev_author,sca_v2,sega_official,
  ultraimage}.csv (per-image, 2000 draws); results/prog195_pairs/cfid_vs_{area,sca_v2} (set-level, 1000 draws).
- PENDING: prog195_vs_baselines/cfid/ (set-level vs all), results/prog195_s1, prog195_s2 (seeds), HRGen-1K prog,
  results/prog_latency (clean latency). Krea: not supported by the progressive sampler (single-pass row only).

## Decisions
- Table 1: all methods, groups single-pass / multi-stage, ranks within group, Prog. SCA headline row (shaded); colours
  of every other row against Prog. SCA (blue = row significantly better, orange = worse). No Dup. anywhere.
- Old Table 2 (SigMa, budgets) moves to the appendix (tab:panel46 kept as label, multi-stage rows dropped).
- Old teaser (six methods, single-pass) moves to the appendix; page-1 hero becomes progressive SCA mosaic.
- Speed claims: use matched clean latency when results/prog_latency lands; until then campaign means, said so.

## Map
| file | what |
|---|---|
| paper/main.tex | preamble + main text (mine) |
| paper/appendix.tex | appendix (appendix agent) |
| paper/figures/src/gen_main_tables.py | Tables 1-4 + headline.tex (mine) |
