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
