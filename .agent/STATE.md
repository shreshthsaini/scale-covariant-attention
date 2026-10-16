# STATE (rewritten 2026-10-16 11:05)
## Goal
Training-free high-res method paper beating AReA (arXiv 2610.17227) on FLUX.1-dev. Full detail: HANDOVER.md.
## Done
- Method final: global gate (sigma>0.89) + per-head calibrated mix of native-grid pooled branch and exact
  64x64 FlexAttention window (code 2d11740; pi table /scratch/09032/saini_2/sca/calib/k1_1024.npz).
- All campaigns finished (497 tasks, 0 failed); results archived in results/; paper/main.pdf (9 pp main,
  0 todo) and paper/sca_overleaf.zip sent to the user. Headline 4K: IR 1.23 vs 0.92, FID 138.4 vs 146.0,
  71 s vs 212.6 s (AReA) vs 184.6 s (full).
## In flight
- Nothing. No Slurm jobs; spool empty; taskq/STOP set.
## Next
- Ask user: create private GitHub repo + push? Add second model / RoPE baselines? (HANDOVER Section 11)
## Decisions
- AReA compared via released code under shared protocol (user choice); band guard is appendix-only;
  scale-4 pi table default. Old chronological log: .agent/notes/state_log_2026-10-15_16.md.
