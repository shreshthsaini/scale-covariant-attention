# SCA live state, 2026-10-17 10:26 PDT

## Goal
Finish experiments, verified evaluation, full paper revamp and reproducibility delivery.
User transferred full ownership to Codex and explicitly requested broader baselines,
professional diagrams, readable tables/plots and continued unattended execution.

## Status and next action
- Paper revamp committed4011238:38pages,473source files, independent ZIP build,
  all38pixel-identical72dpi renders, changed pages144dpi inspected, no overfull boxes.
  Method PDF/SVG/source/provenance; six-method authentic portrait/common crops;
  five-baseline paired intervals; compact main/ablation tables; main latency plot.
  Main30cells and headline numbers independently audited;42ablation cells preserved.
- New clean AReA medians6K482.2844s/8K951.9874s, SCA161.5585s/287.6398s.
  Speedups2.9852x/3.3097x,66.5%/69.8%time reduction.13clean timinggroups verified.
  All7timinglogs identifyGH200120GB. Only full6K clean timing remains pending.
- Baseline audit/source-pinned HRDiT+SigMa package prepared and CPU checked.
  Both one-image smoke tasks frozen/enqueued, GPU generation unrun.18tests passed;
  receipt:evidence/continuation/baseline_smoke_deployment_20261017.json. No duplicates.
  HRDiT blur global RNG reset per prompt; environment versions in resume identity.
  Follow finite/fidelity/runtime checks, then12/46, then affordable191 comparison.
  Do not promote based on SCA winning. HRDiT progressive57steps is separate budget.
- Durable controller1076947 completed first work unit. Successor1077042 PENDING,
  dependency satisfied, estimated10:42PDT; do not duplicate or alter frozen controller.
  Root allocation1073260 ends10:45PDT; read active-team note for ownership transfer.
- GPU fleets SCA10/1077447 (2-8nodes,4h,min2h) and SCA11/1077473
  (2-4nodes,2h,gh-dev) PENDING. No SCA GPU allocation active. Queue60pending,
  0running,0failed,894done after baseline smoke addition. Last SCA11 estimate11:44PDT.
- Owner-gated repair release completed after SCA7/steps ended; original failures kept.
  GCC freshGPU smoke and full6K gridfix pending.18release tests passed.
  Full6K needs4800s pluscleanup. Promote only successful reviewed GCC smoke receipt.
- Generation6K AReA56/100,SCA64/100;Krea AReA0/100,SCA20/100.
  SEGA7repairs complete, overlay100/100; final strict hash audit still required.
  Overlay preserves193originals+7repairs. Generation footers gate exact100coverage.
- Neither E2/E9 CPU bootstrap READY marker exists. After GPU features ready,
  run corresponding hold_cpu_finalize script onCPU,1000reference-image-cluster draws.
  Archive, regenerate exacttables/completegalleries, reconcile wholemanuscript,
  and rebuild/inspect standalone delivery. New baselines must also be integrated.
- Complete:E1/E4/E5/E6/E7/E8,budget controls,12paired8K metrics/gallery.
  Fresh3metricenvs pass boundedCPUparity exactly,39assets12.08GB,0downloads;
  42focused/71combined tests. Does not establish GPU/native4KMUSIQ/fullrerun parity.

## Map
- HANDOVER.md first85lines: authoritative current status; historical claims below.
- .agent/notes/latest_delivery.md: current PDF/ZIP hashes and validation links.
- results/evidence/paper_revamp_20261017.json: delivery and numerical evidence.
- results/evidence/presentation_review_20261017.json:38-page render/layout audit.
- .agent/notes/baseline_audit_20261017.md: primary-paper coverage/protocol/source audit.
- repro/baseline_expansion_20261017/README.md: immutable smoke/pilot procedure.
- .agent/notes/active_team_20261017.md: live ownership; avoid overlapping writes.
- repro/runtime_repairs_20261017/README.md: repairs already released, do not duplicate.
- results/evidence/metric_environment/validation_summary_1076947.json: CPUparity scope.
- .agent/notes/paper_numeric_audit_20261017.md: final result insertion map.
- slurm/CONTINUATION.md: frozen durable controller. /scratch/09032/saini_2/sca/taskq/:spool.

## Decisions
- Preserve original/negative outputs and old freezes. No scratch deletion.
- No remote; commit explicit paths locally, no publication or external messages.
- Last submission balance7908/40000SUs; before ANY new nodes checklogin2,
  reservation headroom and96%burn cap, retain1600SU. Pack tasks into existingfleets.
- Preserve MF9 hostwork/foreignSSH; no guests restarted. At most3SSH, login guardfirst.
- Reference correction195rows/168identities; mainpatchFID/KIDgain unresolved.
- 8K sixresolvedgains/fourunresolved, MUSIQ meansfavorAReA;stripe/gridfailures retained.
  8K CPUfloat32 HPS differs from mainGPUprecision. Fullcalibrationruntime unknown.
- Native emulation and fitted/transferred per-head/noise mixture is boundednovelty.
  Two branches, headroles and globalopening are established. Exact-coordinate
  theorem does not repair bf16drift;0.015625 is observedabsoluteerror,notULPbound.
- SigMa paperdev/g3.5 vs releasedKrea/g4.5 are separateprotocols; releasedmu=max_shift.
  Optional native/matchedKrea aliases are identicalschedules, not independentcontrols.
- Baseline scale-up depends on fidelity/validity/cost, never favorablemetrics.

## Open questions
- Wholeproject remains incomplete:6K/Krea,full6Ktiming,freshGPU smoke,finaleval,
  broaderbaselineexperiments,completegalleries/manuscriptintegration anddelivery.
- NewHRDiT/SigMaGPUcompatibility and measuredcost unknown untilsmokes finish.
- No official ResDiT implementation verified; no invented officialbaseline claim.
- GPU/CPU queue starts remain estimates. Successor is durable but currentlypending.
