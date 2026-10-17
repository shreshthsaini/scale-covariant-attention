# STATE (rewritten 2026-10-16 12:10)
## Goal
Phase 2: bulletproof experiments + presentation upgrade of the SCA paper (HANDOVER.md Section 12).
## Done
- Code d800de4 (FLUX.2 port, --model dev|krea, UltraImage wrapper, link_common, slurm/phase2_tasks.py);
  frozen as /scratch/09032/saini_2/sca/code/PHASE2 and code/FLUX2. 30 CPU tests pass.
- Figures committed (paper/figures/snippets/*.tex, README.md has rerun commands).
- Manifests: .agent/notes/phase2_experiments.md (E1-E7, E9; ~121 GPU-h), .agent/notes/flux2_port.md (E8).
## In flight
- Spool: 407 pending (d Krea calib, e E6/E7, f E5 seeds + f000_flux2_smoke, g E1 DyPE/SEGA/UltraImage,
  h E4 HRGen, i E2 6144/8192, j E9 Krea, y E3 latency). FLUX.2 tasks f010.. remain in taskq/hold_flux2
  until f000_flux2_smoke passes (check logs/task_f000_flux2_smoke.log: seconds, peak GB).
- Workers: guests on MF9 nodes c609-092 c638-142 c610-052 c610-011 c638-141 c609-151 (~18 h);
  fleets SCA6 (1075813, 4n 24h) and SCA7 (1075814, 4n 12h) pending.
- Writing agent still editing paper/main.tex (not committed yet).
## Next
- Requeue orphans (requeue_missing.py), add guests on new free-idle nodes, release hold_flux2 after smoke.
- As evals finish: archive eval CSVs to results/, rerun figure scripts (snippets README), fill \todo{E#}.
## Decisions
- Main protocol unchanged; Krea, FLUX.2, UltraImage (own defaults) reported as extra sections.
## 12:35 E7 gate sensitivity done (46 prompts; results/e7_*): sigma*=0.85 worse on aes -0.19*, HPS2.1 -0.009*,
CLIP-IQA -0.032*, ARNIQA -0.040*, MUSIQ -1.9* (IR n.s.); sigma*=0.93 ~ same (IR n.s., CLIP -0.64*, Pick -0.22*,
ARNIQA +0.018*). => 0.89 near-optimal, robust. E6 eval running; E5 started; FLUX.2 smoke running.
## 12:45 FLUX.2 smoke OK: 4096^2 SCA 226 s vs full 433 s, peak 82 GB; SCA visibly sharper/coherent, full
hazy. Released all hold_flux2 tasks (calib f010-f013, merge f200, full f100-f145, SCA f300-f345, eval f900).
## 13:00 E6 calibration budget done (46 prompts; results/e6_*): tables from 4/8/16 prompts vs 32-prompt table:
all metrics n.s. (|dIR| <= 0.012), except cal16 MUSIQ +0.13*. => 4 calibration prompts suffice (~10 GPU-min).
Running: E5 seeds, FLUX.2 calibration (f010-f013). Fleets SCA6/SCA7 still pending; 6 MF9 guests active.
## 15:35 UltraImage bug: wrapper did not pass height/width -> 1024 output (g000,g005 failed). Fixed
(scripts/run_ultraimage.py: call height/width; block-avg cache dir *_v2 since v1 caches were built at
1024; not deleted per scratch rule). Frozen code/PHASE2U. Tasks parked in taskq/hold_ultra (38, now
pointing to PHASE2U); smoke = pending/a000_ultraimage_smoke_4096x4096.sh. If its PNGs are 4096x4096,
move hold_ultra/* to pending.
## 16:05 UltraImage smoke OK (4096x4096, ~303 s/img, coherent). Released hold_ultra -> pending.
## 16:15 E5 seeds (46 prompts, results/e5_*): SCA-AReA seed1 IR +0.31*, HPS2.1 +0.022*, Pick +0.71*, CLIP +1.51*,
MUSIQ-crop +11.6*; seed2 IR +0.36*, HPS2.1 +0.028*, Pick +0.86*, CLIP +1.69*, crop +11.0* (consistent with seed 0).
E8 FLUX.2-dev 4096^2 (50 prompts, results/e8_flux2_sca_vs_full.csv): SCA-full IR +0.46*, aes +0.31*, HPS2.1
+0.018*, Pick +0.56*, CLIP-IQA +0.066*, ARNIQA +0.071*, MUSIQ +16.0*, crop +21.4*, CLIP n.s., -224 s/img (~2x).
## 22:00 E1 BASELINES (191 prompts, 4096^2; results/e1_*): IR/HPS2.1/Pick/CLIP/MUSIQc/FID/s:
SCA 1.229/.297/23.09/29.11/61.2/138.4/71; UltraImage 1.276/.308/23.32/28.80/41.8/142.7/309; AReA .917/.272/
22.35/27.67/50.4/146.0/262; SEGA .552/.244/21.64/26.93/50.4/165.4/190; DyPE -.105/.205/20.49/24.84/50.4/206.2/196.
UltraImage-SCA: HPS2.1 +.011*, Pick +.24*, aes +.055*, ARNIQA +.011*, IR +.047 n.s.; SCA better CLIP .31*,
MUSIQ 11.7*, crop 19.4*, FID, patch FID; 4.3x faster. UltraImage = 2-stage, 2x30 steps, guidance 3.5->7.
Budget checks queued (b*_sca_s56 = 56 steps, b*_sca_g7 = guidance 7; abl50; eval b999_eval_budget.sh -> eval/budget_*).
