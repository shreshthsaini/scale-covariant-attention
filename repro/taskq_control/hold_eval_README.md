# hold_eval: 195-prompt protocol evaluations (2026-10-20)

Protocol: all 195 Aesthetic-4K evaluation prompts at 4096x4096, the 4 development prompts included, as in AReA
(arXiv 2610.17227, Sec. 4, Table 1). HRGen-1K uses all 1000 prompts (AReA Table 2). Survey and gap analysis:
`/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/.agent/notes/eval_protocol_survey.md`.

`superseded_191/` holds the old 191-prompt tasks z910-z950. Do not release them.

## How to release
Move a task into pending: `mv hold_eval/<task>.sh pending/`. Workers take tasks in sorted order, so z96x tasks run
after any generation that is still pending.

## Gate behavior
Each task first checks that every method has one PNG and a complete JSON sidecar for every prompt id. If anything is
missing, it prints the per-method counts, parks a copy of itself back here, and exits 3 at once. It never sleeps on a
GPU. The worker then files the running copy under failed/. A later successful run moves that copy to
recovered/a195_gate/. A CUDA OOM requeues the task to pending and exits 75.

## Release order (GPU-h at about 4 s per image per method)
| Order | Task | Methods | Release when | GPU-h |
|---|---|---|---|---|
| 1 | z960_eval_a4k195_main | SCA, AReA, full, DyPE, SEGA, UltraImage | now (195/195 each, gate checked 05:05 CT) | 1.3 |
| 1 | z964_eval_a4k195_2048x4096 | SCA, AReA | now | 0.4 |
| 1 | z965_eval_a4k195_2048x6144 | SCA, AReA | now | 0.5 |
| 1 | z961_eval_a4k195_panel | HRDiT, SigMa, prog, g7, 42, 56 steps, SCA, UltraImage | now: a190_dev4_1/2 finished 05:50 CT and the gate shows 195/195 for all 8 methods | 1.8 |
| 3 | z962_eval_a4k195_seed1 | SCA, AReA, seed 1 | every a191_6_* and a191_7_* chunk in done/ | 0.45 |
| 3 | z963_eval_a4k195_seed2 | SCA, AReA, seed 2 | every a191_8_* and a191_9_* chunk in done/ | 0.45 |
| 4 | z966_eval_krea195 | SCA-Krea, AReA-Krea | all 38 tasks of ../hold_krea195 in done/ | 0.45 |
| 5 | z967_eval_hrgen1k_krea | SCA-Krea, 1000 HRGen-1K | all 100 tasks of ../hold_hrgen1k_krea in done/ | 0.8 |
| 6 | z968_eval_hrgen1k_dev | SCA, AReA, 1000 HRGen-1K | all 240 tasks of ../hold_hrgen1k_dev in done/ | 1.6 |

## Generation tasks that are held, not released
| Folder | Tasks | Purpose | GPU-h |
|---|---|---|---|
| ../hold_krea195 | 38 (19 SCA-Krea, 19 AReA-Krea; 5 prompts each) | AReA's own protocol (FLUX.1-Krea-dev, 28 steps, guidance 4.5) on the 95 prompts not yet run (rows 0-3, 104-194). Needed for the published-number table. | 8.6 |
| ../hold_hrgen1k_krea | 100 (SCA-Krea, 10 prompts each) | HRGen-1K, all 1000 prompts, against AReA's published Table 2. No baseline reruns needed. Recommended. | 22.7 |
| ../hold_hrgen1k_dev | 240 (80 SCA, 160 AReA) | HRGen-1K rows 200-999 on FLUX.1-dev, so the paired SCA vs AReA HRGen table covers all 1000 prompts. Optional: most of the cost is AReA. | 70 |

Finished (05:50 CT, done/): `a190_dev4_1_hrdit_000_004.sh` and `a190_dev4_2_sigma_000_004.sh` (HRDiT and SigMa on
Aesthetic-4K rows 0-3, about 0.4 GPU-h; outputs in out/baseline_expansion_20261017_dev4). Their bodies and wrappers are in ../hold_195/.

## CPU finalize (never through the GPU spool)
After a z96x task writes `eval/<group>/DONE`, run `../hold_cpu_finalize/<group>_bootstrap.sh` on a CPU node. It
computes the set-level FID/KID/patch intervals with 1000 reference-image-cluster draws. Scripts exist for these groups:
`a4k195_main`, `a4k195_panel`, `a4k195_2048x4096`, `a4k195_2048x6144`, `krea195`.

Then rebuild the published comparison:
`python3 -I /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/results/published_compare/build.py`.

Generators: `/scratch/09032/saini_2/sca/code/eval195_20261020/make_{eval,gen}_tasks.py`.
