# SCA live state, 2026-10-17 03:18 CDT

## Goal
Finish experiments, verified evaluation, paper and reproducibility package.
User transferred full project ownership to Codex; continue autonomously.

## Status and next action
- Complete: E1/E4/E5/E6/E7/E8, budget controls, E2 8K12paired metrics and gallery.
- New paper+ZIP independently build; PDFshad980bda6ab17f51fca104fbc393b86cc0cda59dcbd5e86f399b69cc99a845949.
  Figures agent verified37pages,all11latencygroupsandall50displayed8Kvalues.
- 8K:6 resolved gains (IR/CLIP/Pick/HPSv2/HPSv21/ARNIQA),4 unresolved differences.
  MUSIQ/cropMUSIQ/CLIPIQA means favorAReA; visible stripes remain. All8KmetricsCPU.
  Querychunk256 preservesnativeMUSIQallkeys/patches. All24scorescomplete,72basecacheentries.
  OriginalfailedGPUOOMtask movedto recovered withreceipt; originalfailurepreserved.
- SCA6 1075813 TIMEOUT03:07:36,allstepsgone; threeheldlatencytasksreleased03:12:55.
- SCA7 1075814 PENDING4nodes4h,min2h; latestestimate16:02CDT.
- SCA9 1076923 PENDING2-8nodes4h,min1h,deadline-awarev2frozen dispatcher.
  SCA8 1076908 cancelledwhilePENDING. Lastbudget8212.919/40000SUsavailable79.47%burned.
  MF9guestsfullyyielded; noforeignworkinterference. NoSCAactiveGPUnow.
- E3 SCA2K18.20/3K40.16/4K71.06/6K161.56/8K287.64scomplete.
  Matched4KAReA202.44/full186.54s. Missing6KAReA/full,8KAReA.
  Retrypreflightpreservesoldimagesandfreshwarmup+3. V2fitsruntimeheaders/deadlines.
- 6KAReA26/100,SCA24/100;KreaSCA5/100,AReA0/100. Interrupted6Ktasksrequeued.
- E2/E9GPUevaltasksheldinhold_cpu: strict100coverage,6Klabelsca_nbfixed.
  GPUbasebootstrap0 plusREADY_FOR_CPU_BOOTSTRAP; CPUfinalizersinhold_cpu_finalize
  perform1000drawsimageidentityclusterbootstrap. RootmustlaunchonCPUwhenready.
  ArchiveguardrejectsGPU-onlybootstrap0stage. Paperagentpreparesstrictpromotionhelper.
- KreaDyPE/SEGAtransformerprovenanceverifiedlaunchscripts/env+successfulfleetlogs;
  originalsidecarsmislabelbasepipeline. Paperagentarchivesreceipt.
- Freshgenerationenvsca-repro-20261017:67tests,8CLIchecks,2supervisortestspass.
  Queuea_000_repro_gpu_smoke.sh testsCUDA/compiledSCAwith5minbound,no learnedweights.
  Sourcehashesverifiedattaskstart. Evidenceagentpreparesmetricenvrestorationrecipe.
- RootinheritedcombinedCPU94tests pass; frozenbaseline33tests pass.
- Corrected195row/168imageidentityclusterreferencepolicy retained; mainpatchFID/KIDunresolved.
  Noveltybounded; bf16coordinatecaveat; unsupported1.5GPUhclaimremoved.
- Paperagent:promoter+Kreaprovenance. Figuresagent:printQA. Evidenceagent:metricenvrecipe.
- CPUcomputei618-051. TwoforeignSSHsessionsheld;oursallreaped. login2works;login1stalled.
  Atmost3SSHaccountwide,guardbeforeSSH. No gitremote. Localcommitsexplicitpaths.
- NEXT: review/freezehelper,patchonlypendinggenerationfooters;deliver8KPDFZIP;
  finishCPUrepro/validation,thenE2/E3/E9whenallocationsstart.

## Map
- HANDOVER.md: current corrections override historical claims below them.
- .agent/notes/phase2_experiments.md: original E1-E9 design, some eval implementation superseded.
- repro/README.md, repro/environment/: portable snapshots, data/model restoration, environments.
- results/evidence/: immutable source evidence, corrected metrics, coverage, latency, tracking.
- paper/delivery_receipt.json: PDF/ZIP hashes and standalone source build validation.
- /scratch/09032/saini_2/sca/taskq/: pending/running/done/held tasks and receipts.

## Decisions
- Preserve frozen campaign protocol and outputs. Negative results stay in paper.
- Reference-image-cluster bootstrap retains195rowweights/168identities.
- GPUfeatures thenCPUbootstrap; no GPUworker dependency waits.
- Native8Kmetrics exploratory; no12promptFID/KID or universalqualityclaim.
- Cleanlatencyneedswarmup+3inoneprocess; restartedtasksstartfresh,preserveoldbytes.
- No new repository publication authorized; no remote exists.

## Open questions
- E2 6K, E3 remaining3latencygroups and E9 stillneedGPUallocation/results.
- FreshGPUenvvalidationqueued; metricenvrecipeunderconstruction.
- Finalpaperdeliverymustfollowallresults,provenancechecks,printQAandindependentZIPbuild.
