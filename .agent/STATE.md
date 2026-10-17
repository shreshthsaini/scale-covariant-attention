# SCA live state, 2026-10-17 03:50 CDT

## Goal
Finish experiments, verified evaluation, paper and reproducibility package.
The user transferred full project ownership to Codex. Continue without another approval.

## Status and next action
- User requested unattended continuation. CPU controller1076947 submitted; frozen
  dispatch/20261017_continuation. It queues its successor before starting agent work.
  Live startup/successor verification is pending. Auth smoke and16 offline tests passed.
  Read slurm/CONTINUATION.md and scratch sca/continuation/status.json for current ownership.
- Complete: E1/E4/E5/E6/E7/E8, budget controls, and E2 8K metrics and full gallery.
- Latest verified paper: 37 pages. PDF and extracted Overleaf ZIP build independently;
  no overfull warnings. Final source ZIP rebuild passed; all37 pages render identically
  to the reviewed PDF at72dpi.
  Delivery hashes and validation are in paper/delivery_receipt.json.
- 8K: 12 paired prompts, all 10 metrics complete. Six resolved gains for SCA:
  IR, CLIP, PickScore, HPSv2/2.1, ARNIQA. Four differences remain unresolved.
  MUSIQ, crop MUSIQ and CLIP-IQA means favor AReA; visible stripe/grid failures remain.
  All 8K scores use CPU; HPS float32 differs from the main GPU mixed-precision protocol.
  Native MUSIQ uses query chunks256 with all keys/patches retained. All24scores archived.
  Original GPU OOM task is preserved in taskq/recovered with a separate recovery receipt.
- SCA6 job1075813 timed out03:07:36; its workers are gone and interrupted tasks requeued.
- SCA7 job1075814: PENDING, four nodes, four hours, minimum two hours.
  Scheduler estimate16:02CDT is uncertain. Root activated the frozen v2 compatibility
  bridge while pending at03:35:54CDT. Only this JobID uses the bridge; priority/topology retained.
  Bridge: dispatch/20261017_sca7_1075814_bridge. Four nodes, one task/node are required.
  Actual deadline comes from Slurm env or one rank-zero query; telemetry is reaped.
- SCA9 job1076923: PENDING, two to eight nodes, four hours, minimum one hour.
  Frozen dispatcher: dispatch/20261017_v2. No SCA GPU is currently allocated.
  SCA8 job1076908 was cancelled while pending. MF9 guests yielded; do not restart them.
- Last budget check before submission: 8212.919/40000 SUs available,79.47% used.
- Generation: 6K AReA26/100, SCA24/100; Krea SCA5/100, AReA0/100.
- SCA latency complete:2K18.20s,3K40.16s,4K71.06s,6K161.56s,8K287.64s.
  Matched4K AReA202.44s/full186.54s. Remaining:6K AReA/full and8K AReA.
  Retries preserve interrupted outputs and require fresh warm-up plus3 timed images.
- Strict evaluation promotion is deployed to all77 remaining generation tasks.
  Frozen helper: dispatch/20261017_eval_promotion;30 tests pass. It verifies exact100
  prompts, sidecars, PNG hashes/dimensions and publishes without overwriting task states.
- Krea baseline audit recovered exact source for193/200 images. Relevant baseline
  algorithms match across recoverable revisions. Seven SEGA images need regeneration.
  Task a_krea_sega_source_repair_7.sh is QUEUED,3000s minimum; originals remain intact.
  Overlay: /scratch/09032/saini_2/sca/out/krea_baseline_overlay_20261017.
  All39 Krea generation footers, repair footer and heldE9 evaluation require this overlay.
  Independent deployment audit passed; readiness correctly reports SEGA7 missing.
  New complete Krea gallery requires the same overlay and hashes repair provenance.
- GPU evaluation scripts i998/j999 remain in hold_cpu until generation is complete.
  They compute scores/features with bootstrap0 and create READY_FOR_CPU_BOOTSTRAP.
  Then run taskq/hold_cpu_finalize/e2_6144_bootstrap.sh and e9_krea_bootstrap.sh on CPU.
  These perform1000 reference-image-cluster draws. They are prepared, NOT auto-running.
  Archive rejects the unfinished GPU-only stage. Reuse the unchanged managed sca env.
- Fresh generation environment:67 CPU tests,8 CLI checks,2 supervisor tests passed.
  a_000_repro_gpu_smoke.sh is queued; no learned weights, <=5min, declared tolerances.
  Full fresh metric environments and score-equivalence tests are NOT yet validated.
  Three metric locks resolve; seven Torch asset hashes and historical HF revisions checked.
  Strict split-report merger passes18 tests and an end-to-end synthetic CLI fixture.
- Complete-gallery readiness tests and SCA7 bridge tests:22 passed. No incomplete new
  gallery was rendered. Historical14-prompt6K and partial8K artifacts stay preserved.
- All three agents and root finished and reaped their subprocesses. No owned poller remains.
  Latest live scheduler check: both SCA7/SCA9 pending;82 tasks pending,0 running.
- NEXT: controller finishes metric environment CPU validation while awaiting GPUs; then generation,
  GPU evaluation, CPU bootstraps, latency audit, complete galleries and manuscript.
  Run archive_evidence.py and gen_final_tables.py only after full validated coverage.

## Map
- HANDOVER.md: current corrections override the historical claims below them.
- slurm/CONTINUATION.md: unattended worker, successor, stop file and runtime evidence.
- repro/README.md, repro/environment/, repro/metrics_environment/: restoration and validation.
- results/evidence/krea_baseline_provenance/: launch receipts, source audit, seven repair prompts.
- results/evidence/dispatch/: task definitions, deployment receipts and current snapshot hashes.
- repro/dispatch_migration/sca7_1075814/activation.json: live bridge activation evidence.
- paper/figures/src/fig_phase2.py: strict6144_complete/krea_complete gallery modes.
- /scratch/09032/saini_2/sca/taskq/: live spool, held CPU finalizers and receipts.

## Decisions
- Preserve frozen scientific code/protocols and all original outputs; negative results stay.
- Corrected references retain195 row weights and168 image identities. Main patchFID/KID
  differences are unresolved; whole-image FID remains significantly better.
- Novelty is fitted/transferred per-head/noise output mixing, with prior-art boundaries.
  Exact-coordinate theorem does not remove bf16 drift above4K. Full calibration cost unknown;
  unsupported1.5GPUh claim removed. Distinct2101 main image hashes verified.
- GPU feature extraction, then CPU bootstrap; GPU workers do not wait for dependencies.
- CPU hosti618-051 control job1073260 ends12:45CDT; queued GPU jobs may outlive this session.
  Two unrelated SSH sessions exist. Ours are reaped; at most3 account-wide. login2 works.
- No git remote. Commit locally with explicit paths; publication is not authorized here.

## Open questions
- The project is incomplete until6K, remaining latency groups, Krea repair/generation,
  evaluations, final paper reconciliation and independent source build finish.
- GPU start times and granted walltimes remain scheduler-dependent; more fleet time may
  be needed if both pending jobs receive their minimum allocations.
- Fresh GPU validation is queued. Metric recipe installs and learned-score parity remain unrun.
