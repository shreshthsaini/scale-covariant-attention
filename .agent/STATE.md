# SCA live state, 2026-10-17 02:14 CDT


## Goal
Finish Phase 2 experiments, verified evaluation, figures, paper and reproducibility package.
User explicitly transferred full project ownership to Codex; continue autonomously.

## Status and next action
- Root owns campaign, delivery and documentation. Paper agent reconciles corrected metric tables;
  evidence agent checks distribution CIs; figures agent tests memory-safe native8K MUSIQ.
- Complete: Phase1, E1/E4/E5/E6/E7/E8 and budget controls. PDF35pages and standalone ZIP verified.
- Audit:740 scalar means,13 calibration fits exact;152 legacy distribution points reproduced exactly.
- Confirmed official reference195rows/168images. Historical patch-feature assembly aliased rows.
  Corrected pointvalues archived; repro/evaluate_corrected.py fixes assembly,16tests pass.
- Main corrected patchFID99.8156 vs103.4111, patchKID.00916929 vs.01143955. Both imagecluster
  paired95%CIs crosszero. WholeFID benefit remains significant; KID remains nonsignificant.
- SCA6 1075813 RUNNING2nodes c638-152,c640-021, ends03:07CDT. E3 2K/3K latency active.
- SCA7 1075814 PENDING4nodes/4h with2hminimum. No new job submitted. Budget79.47%burned,
  8212.919/40000SUsavailable, verified login usage.map02:13CDT.
- MF9 guests yielding to host workload. Do not restart them; last2SCA tasks finishing on c638-141/142.
- E2 8K12pairs complete. CPU extra+CLIP/PickScore complete; fullMUSIQ killed137 at8K.
  Process88759 reaped. Failed task i999_eval_e2_8192.sh retained; don't blindly relaunch.
- Pending E2 6K and E9 Krea generation remains. Evaluation scripts i998/j999 in hold_cpu;
  root must dispatch once generation completes, with corrected evaluator and strict100coverage.
- E3 long6K/8K bundles in taskq/hold_long, need2hfit. Other3tasks active/pending.
  Summary in hold_cpu/y999_summary_latency.sh. First sca_latency_warmup excluded; validate
  first_call_in_process=false for all timed images and n=3 before reporting.
- NEXT: finish metric corrections and memory-safe8K scoring; publish corrected PDF/ZIP;
  keep GPUs busy and finish E2/E3/E9. Do not claim project complete while these remain.
- No git remote. Current session on CPUcompute i618-051. Two account SSH sessions already held;
  guard before SSH, at most3account-wide. No login compute or poller.
- RA6 c620-061 is reserved by a live dependency-waiting host worker; do not borrow it.

## Map
| Note | Read it when |
| --- | --- |
| HANDOVER.md | Method, historical results, paths and Phase 2 plan |
| .agent/notes/state_20261016_checkpoint.md | Exact inherited chronology and prior results |
| .agent/notes/phase2_experiments.md | E1-E7/E9 task design and protocol |
| .agent/notes/flux2_port.md | FLUX.2 implementation, scheduler and memory decisions |
| .agent/notes/area_code.md | AReA release implementation details |
| paper/figures/snippets/README.md | Figure generation and snippets |
| .agent/notes/primary_source_audit.md | Verified UltraImage, ResDiT and HRDiT overlap |
| results/evidence/README.md | Numeric archive and audit scope; refresh commands |
| results/compute/ | Attributable GPU-hour lower bound and missing-log limitations |

## Decisions
- Preserve frozen campaign code and shared FLUX.1-dev protocol for comparability.
- UltraImage own two-stage defaults and FLUX.2 own scheduler/guidance must be explicit.
- UltraImage wins some preference metrics; SCA wins detail metrics and speed. No universal superiority claim.
- Four-prompt calibration shows no detected loss on 46 prompts; absence of significance is not equivalence.
- Novelty is fitted and transferred per-head/noise output mixing; head roles, two branches and a global
  opening already have prior art. Exact-coordinate range theorem does not eliminate bf16 drift above4K.
- Preserve negative results and protocol limitations; paired CIs do not establish novelty or universal efficacy.
- No source checkout changes in another project. No scratch deletion. Archive compact paper crux on work.
- Existing guest authorization applies; inspect resource ownership and foreign processes before adding workers.

## Open questions
- Need final E2/E3/E9 outputs, coverage checks and paper integration before claiming completion.
- Need final clean build, print-size review, source ZIP rebuild and independent ZIP build.
- No Git remote; keep local commits and defer repository creation/publication to user instruction.
