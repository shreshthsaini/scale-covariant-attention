# SCA live state, 2026-10-17 17:48 UTC

## Goal
Finish experiments, verified evaluation, the paper and reproducibility delivery.
Codex owns the whole project. Preserve unfavorable results and novelty boundaries.

## Status and next action
- Unit1077042 completed baseline promotion/evaluation hardening. All28 combined CPU
  tests passed:15 promotion and13 evaluation. The separate helper prepares reviewed
  12/46 candidates, sizing missing prompts from twice the maximum observed time
  plus explicit startup headroom. It never enqueues or submits jobs; it retains the
  frozen runner, manifest, output root and WandB. Partial/corrupt outputs refuse.
- Evaluator now preserves all4 GPU reports, including paired deltas; binds reference
  image bytes across195 rows/168 images; checks PNG CRC; explicitly uses4 CLI threads.
  Verified39 frozen files,46 identities for each historical control,6 sampled PNGs.
  New baseline GPU generation/scoring and CPU intervals remain unrun. Smokes absent.
- NEXT: existing GPU event. At17:47:50 UTC, SCA10/1077447 and SCA11/1077473 remain
  PENDING. No SCA GPU allocation active;60 pending,0 running,0 failed,894 done.
  All60 admission headers fit fresh2h windows with120s cleanup. This is not runtime
  proof. No submission, task mutation or controller change. Do not duplicate retries.
- CPU1077042 runs on i618-043; deadline16:41:12 America/Chicago(21:41:12 UTC).
  Successor1077572 is PENDING on dependency. Frozen controller hashes unchanged.
  Currently actionable CPU work is done; return waiting so the controller parks on
  existing GPU events. Do not change successor/dependency or launch another watcher.
- Generation:6K AReA56/100,SCA64/100; Krea AReA0/100,SCA20/100. Strict readiness
  remains closed. Neither E2/E9 CPU READY nor CPU COMPLETE marker exists. When READY
  appears, run its hold_cpu_finalize script once on CPU,1000 reference-image-cluster
  draws with bounded threads. Do not rerun a completed bootstrap.
- After the GCC fresh GPU smoke, review its distinct receipt and promote to canonical
  environment/fresh_gpu_smoke.json, preserving the original failure. Full6K gridfix
  timing remains pending,4800s+cleanup.13 clean groups audited; only full6K pending.
  AReA6K482.2844s/8K951.9874s; SCA161.5585s/287.6398s;2.9852x/3.3097x.
  All7 timing logs identify GH200120GB.
- Baseline next: inspect smoke images/profile fidelity, write a hash-bound review
  with the new helper, prepare12 candidates sized to the fleet, inspect live task
  ownership, then publish. After12 review, prepare46 and evaluate exact8-method
  pairs. Consider affordable191 only after measured cost/protocol review and a
  separate frozen manifest. Promotion must never depend on SCA winning.
- Krea overlay:100DyPE+100SEGA,193 originals+7 replacements. All7 repaired PNGs
  are byte-identical to originals;200 original sidecars unchanged;207 PNGs audited.
- Paper revamp4011238:38 pages,473 source files, independent ZIP build,38 identical
  72dpi renders, changed pages inspected at144dpi, no overfull boxes. Method diagram
  PDF/SVG/source; authentic6-method portrait; paired intervals; readable tables.
  Main30 cells independently verified;42 ablation cells preserved. Paper unchanged
  by this CPU unit. PDF/ZIP hashes remain those in latest_delivery.md.
- After GPU evidence: archive validated E2/E9 results; exact tables; complete6K/Krea
  galleries; reconcile abstract/main interpretation/appendix/limitations. Rebuild
  PDF+ZIP; independently build extracted sources; review appearance and numbers.
- Complete:E1/E4/E5/E6/E7/E8,budget controls, exploratory12-prompt8K metrics/gallery.
  Three fresh metric envs pass bounded CPU parity exactly:39 assets,12.08GB,
  0 downloads,42 focused/71 combined tests. GPU/native4K MUSIQ/full reruns untested.

## Map
- HANDOVER.md first85 lines: current authoritative summary before historical material.
- results/evidence/continuation/unit_validation_1077042.json: latest live checks,
  pending task hashes, frozen hashes and28 tests; read before task actions.
- .agent/notes/unit_1077042.md: completed unit and exact next commands.
- repro/baseline_promotion_1077042/README.md: after smokes, exact12/46 commands.
- .agent/notes/baseline_eval_audit_1077042.md: evaluator fixes and planner review.
- repro/baseline_evaluation_20261017/audit_validation_1077042.json: control audit.
- repro/baseline_evaluation_20261017/README.md: exact8-method GPU/CPU handoff.
- .agent/notes/baseline_audit_20261017.md: primary sources/protocol/novelty review.
- repro/baseline_expansion_20261017/README.md: original frozen generation profiles.
- repro/runtime_repairs_20261017/README.md: repairs already released; no duplicates.
- results/evidence/metric_environment/validation_summary_1076947.json: CPU parity.
- .agent/notes/latest_delivery.md: current PDF/ZIP hashes and appearance proof.
- .agent/notes/paper_numeric_audit_20261017.md: result insertion map; old line counts.
- slurm/CONTINUATION.md: frozen controller; scratch continuation/status.json is live.

## Decisions
- Keep originals, negative evidence, frozen packages and all scratch artifacts.
- No remote; commit explicit paths locally. No posting or external messages.
- Before any submission use one sequential login2 SSH; check balance+reservations,
  retain1600SU and respect96% burn cap. Historical balance7908/40000SU is stale.
- No submissions in this unit. Preserve foreign SSH/jobs; guard before fan-out.
- Promotion uses fidelity, finite valid output and measured cost, never quality wins.
- HRDiT57 progressive steps are a separate budget. SigMa paper dev/g3.5 differs from
  released Krea/g4.5; released mu=max_shift. Identical aliases are not extra controls.
- Reference195 rows/168 images: main patchFID/KID gain unresolved.8K has6 resolved
  gains/4 unresolved; MUSIQ means favor AReA; stripe/grid failures retained.
  8K CPU float32 HPS differs from main GPU precision. Calibration runtime unknown.
- Native emulation+fitted/transferred head/noise mixing is bounded novelty; ingredients
  established. Exact coordinates do not repair bf16 drift;0.015625 is observed error.

## Open questions
- Waiting for existing SCA fleets; new baseline GPU fidelity/cost unknown until smokes.
- E2/E9,full6K timing,fresh GPU,final evaluation,broader baselines,paper remain unfinished.
- No official ResDiT code verified. Do not claim an invented reproduction.
