# SCA live state, 2026-10-17 06:44 CDT

## Goal
Finish experiments, verified evaluation, paper and reproducibility delivery.
The user transferred full project ownership to Codex; continue without new approval.

## Status and next action
- CPU controller1076947 on i615-074 ends2026-10-17 10:19:14CDT.
  Successor1077042 is pending dependency. Frozen controller remains unchanged.
  Never launch another watcher, signal its parent, or change/cancel its successor.
- Live06:41CDT: SCA7=1075814 RUNNING4nodes until08:29:50CDT; SCA9=1076923 PENDING.
  SCA7 frozen bridge and SCA9 frozen v2 dispatcher are unchanged.
  Queue76pending,4running,2failed,2review-held repairs,870done,0quarantine.
  Runtime snapshot: results/evidence/continuation/runtime_1076947_end.json.
- Two terminalrc1 failures have tested replacements in taskq/hold_review:
  a_000_repro_gpu_smoke_gcc_1076947.sh: explicit probed GNU compilers;420s minimum.
  a_y003_latency_6144x6144_full_gridfix_1076947.sh: exact grid metadata;4800s minimum.
  Keep held until owning job1075814 AND its steps are gone. Original failures stay.
  Release with repro/runtime_repairs_20261017/release.py, manifest and README.
  18 release tests pass; live dry run correctly refused RUNNING without queue changes.
  A1hour SCA9 grant cannot fit full6K; inspect actual deadlines before replacement fleet.
- Fresh base/extra/ImageReward envs pass bounded CPU parity exactly.125/123/80packages;
 39 isolated assets12.08GB,0downloads. Real split scoring+merge pass;42 focused tests.
  Combined runtime/metric checks71pass. GPU retries remain unrun; no tolerance relaxed.
  Proof: results/evidence/metric_environment/validation_summary_1076947.json.
- Numeric audit:195hashes,762means,13fits,82appendixrows,593othercells,11latencygroups.
  Paper energy/provenance prose corrected. Current37-page PDF and456sourceZIP verify
  independently; all37renders exact. Changedpage27 inspected144dpi, no overfull boxes.
  Proof: results/evidence/delivery_validation_1076947.json; latest_delivery.md has links.
- Complete: E1/E4/E5/E6/E7/E8,budget controls, exploratory12paired8K metrics/gallery.
  8K six resolved gains;4 unresolved. MUSIQ/cropMUSIQ/CLIP-IQA means favor AReA.
  Stripe/grid failures remain;8K CPUfloat32 HPS differs from main GPU precision.
- Generation6K AReA26/100,SCA24/100;Krea AReA0/100,SCA5/100,SEGA5repairs missing.
  Krea overlay: out/krea_baseline_overlay_20261017;193originals plus7replacements.
  Strict generation footers will promote heldGPUeval only at exact100prompt readiness.
- Latency SCA2K18.20s,3K40.16s,4K71.06s,6K161.56s,8K287.64s.
  AReA4K202.44s/full4K186.54s. Missing6K AReA/full and8K AReA.
  Warmup plus3 same-process timed images required; interrupted outputs preserved.
- Neither final CPU bootstrap READY marker exists. After GPUeval writes READY, run
  taskq/hold_cpu_finalize/e2_6144_bootstrap.sh and e9_krea_bootstrap.sh onCPU.
  Require1000reference-image-cluster draws; never rerun completedCPUbootstraps.
- NEXT: after SCA7 owner exits, apply gated repair release; inspect SCA9 deadline and
  remaining task sizes. Review/promote successful GCC smoke receipt to canonical.
  When GPUeval READY, finalizeCPU,archive,generate exact tables/complete galleries,
  integrate all paper sections, rebuild/review independent delivery. No final claim yet.

## Map
- HANDOVER.md: read first85lines for authoritative current status/science boundaries.
- repro/runtime_repairs_20261017/README.md: when owner1075814 ends, release tested retries.
- .agent/notes/gpu_smoke_compiler_repair_1076947.md: compiler cause, originalfailure, tests.
- .agent/notes/latency_full6k_failure_1076947.md: bf16 failure, minimal freeze, clean retry.
- .agent/notes/metric_extra_restore_20261017.md: metric runtime repairs and actual parity.
- .agent/notes/paper_numeric_audit_20261017.md: source checks and exact final insertion map.
- .agent/notes/latest_delivery.md: current verified PDF/ZIP links and limitations.
- slurm/CONTINUATION.md: frozen controller/successor mechanics; do not alter deployment.
- /scratch/09032/saini_2/sca/taskq/: live spool, holds and preserved failures.

## Decisions
- Preserve frozen protocols and all original/negative outputs; new fixes get new freezes.
- Full grid fix supplies metadata only: bf16 coordinates and SDPA remain unchanged.
- CPU parity is bounded, not full benchmark/native4KMUSIQ/GPUmetric validation.
- Corrected references retain195row weights/168identities; patchFID/KID unresolved.
- Novelty is native emulation and fitted/transferred per-head/noise mixing; prior art retained.
- Exact-coordinate theorem does not repair bf16 drift.0.015625 is observed, not ULP bound.
- Full calibration runtime is unknown. No unsupported1.5GPUh or energy-saving claim.
- Preserve MF9 host work and foreign SSH sessions; no guests restarted/no new submissions.
- Last live budget check7947/40000SUs available; check login2 before ANY new submission,
  include reservations and retain1600SUs. Existing4GPUs productive; no live cancellation.
- No remote. Commit explicit changed paths locally; no publication/external messages.

## Open questions
- Project incomplete until remaining6K/Krea,clean latency,freshGPU smoke,final eval,
  galleries/manuscript reconciliation and delivery are verified.
- SCA9 grant/start unknown. Minimum1hour will not fit repaired full6K task4800s.
- GPU metric precision parity remains outside bounded CPU restoration tests.
