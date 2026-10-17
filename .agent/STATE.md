# SCA live state, 2026-10-17 03:03 CDT


## Goal
Finish Phase 2 experiments, verified evaluation, figures, paper and reproducibility package.
User explicitly transferred full project ownership to Codex; continue autonomously.

## Status and next action
- Root owns campaign, delivery and documentation. Paper agent monitors SCA6 until03:10 and then
  prepares strict evaluation-promotion automation. Evidence agent tests a fresh isolated uv env;
  figures agent owns/reaps the CPU shard sessions.
- Complete: Phase1, E1/E4/E5/E6/E7/E8 and budget controls. PDF36pages and extracted ZIP verified,
  corrected patch metrics, full8K gallery, readable generality table and2K/3K timings integrated.
- Audit:740 scalar means,13 calibration fits exact;152 legacy distribution points reproduced exactly.
  All targeted corrected distribution CIs complete. Main patchFID/KID unresolved; wholeFID significant.
- Corrected evaluator: --reference-clusters images retains195rowweights/168imageidentityclusters.
  Native8K MUSIQ fixed with opt-in --musiq-query-chunk256; bounded pretrained scores matchexactly.
  Actual8K benchmark succeeded702sCPU/21.76GiBpeak. Snapshot evaluator unchanged.
- SCA6 1075813 RUNNING2nodes c638-152,c640-021, ends03:07CDT. 4K sweep andallSCA latency sizes complete.
- E3 8KSCA completed02:58:46,median287.640s;6KSCA161.56s.
  4Kmatched: SCA71.06s,AReA202.44s,full186.54s. Three longtasks remainheld.
  c640-021 nowruns a_j000_sca_krea_4096x4096.sh,firstimage79.576s.
  c638-152 runs i024_area_6144x6144.sh;generation mayresumeafterwalltime.
- SCA7 1075814 PENDING4nodes/4h min2h; SCA9 1076923 PENDING2-8nodes/4h min1h.
  V2 dispatcher frozen under /scratch/.../sca/dispatch/20261017_v2;23 focusedtests pass.
  SCA8 1076908 cancelledwhilePENDING afterreplacementaccepted. Budget79.47%burned,
  8212.919/40000SUsavailable, verifiedlogin2usage.map beforeSCA9submission.
  V2latestlogs use task_NAME.latest.log; auditor selectsnewestactuallegacy/V2log.
  Heldlongtasks preserveprioroutputinlatency_history andrerunwarmup+3freshonretry.
  Release3hold_longtasks onlyafteroldSCA6workersstop;v2honors runtimeheaders.
- MF9 guests fullyyielded. All12paired8K generated;6K24/100each;Krea0/100each.
- FourCPU8KMUSIQ shards FIGURES-agent-owned (root toolregistry cannotpoll): sessions45166,91706,77575,42657;
  Only45166/PID973089 stillactive5/6scores;others3sessions exited0andreaped.
  Total23/24cached. Firstbenchmarkdone/reaped.
  Outputs/logs: /scratch/.../sca/eval/e2_8192_musiq_shards_20261017/.
  Reapall, use repro/merge_evaluation_caches.py (rejectsconflictingkeys), expect72basecacheentries,
  then finalcorrected evaluator --metrics clip,pickscore,musiq --device cpu --musiq-query-chunk256.
- Failed i999_eval_e2_8192.sh is originalnativeMUSIQOOM; preservehistory, don'tblindlyrelaunch.
- E2/E9 eval scripts i998/j999 inhold_cpu, strict100coverage, correctedimageclusterbootstrap,
  onlineWandB;6Kalsoquerychunk256. Dispatchonlyaftergenerationcomplete.
- E3 scripts/audit_latency.py verifieslogwarmupthen3timedimages, archiveslogs/sidecars andCSVs.
  SCA sidecarslackfirst_callflag; don'trequireit, verifyuninterruptedlogsequenceinstead.
- OnlineWandB snapshotrun sca-phase2-20261017; refresh scripts/track_campaign.py atmilestones.
- NEXT: finish8Kscoremerge/report/paper;releaseheldlongtasksafterSCA6terminal;
  freezeandapplytestedstrictready-evalpromotion;finishE2/E3/E9whenfleetsstart.
  Root94-testCPU suitepassed31.42s;frozenbaseline33-test suitealsopassed.
  Reproductionpackageportablehelpers,baselinefreezeandretrospectivemodelpins committed3d7deb7.
  LastdeliveredPDF36pagesbuilt02:49;sourceincludes8Klatencyand5.45%transferboundawaitingnextbuild.
- No gitremote. CPUcomputei618-051. TwootherprojectSSHsessionsheld; oursallreaped. Login1SSHstalledandwasreaped;login2works11processes.
  Atmost3SSHaccount-wide, guardbeforeSSH. RA6c620-061 reserved; do notborrow.

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
