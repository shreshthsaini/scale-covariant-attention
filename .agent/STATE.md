# SCA live state, 2026-10-17 11:51 CDT

## Goal
Finish experiments, verified evaluation, paper and reproducibility delivery.
The user transferred full project ownership to Codex; continue without new approval.

## Status and next action
- Controller1076947 completed first unit. Successor1077042 PENDING for CPU; its
  dependency is satisfied. Do not duplicate the watcher or alter its ownership.
- Root live status11:49-11:51CDT: SCA7/SCA9 completed;24additional tasks done.
  Queue58pending,0running,0failed,894done. No SCA GPU currently held.
- Root released both tested repairs via owner-gated helper after all owner steps ended.
  Originals archived; GCC smoke and full6K gridfix replacements are now PENDING.
  No duplicate release needed.18 release tests and live application passed.
- SCA10 job1077447 PENDING,2-8nodes,4hours,min2hours to fit4800s full6K retry.
  Existing frozen v2 dispatcher unchanged. Submission balance7908/40000SUs available.
  Root receipts: results/evidence/continuation/root_status_20261017_1151.json.
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
- Generation6K AReA56/100,SCA64/100;Krea AReA0/100,SCA20/100.
  All7SEGA repairs now have complete sidecars/PNGs; overlay100/100. Final hash audit remains.
  Krea overlay: out/krea_baseline_overlay_20261017;193originals plus7replacements.
  Strict generation footers will promote heldGPUeval only at exact100prompt readiness.
- Latency SCA2K18.20s,3K40.16s,4K71.06s,6K161.56s,8K287.64s.
  AReA4K202.44s/full4K186.54s.6K/8K AReA timing tasks newly done; harvest/audit logs.
  Repaired full6K timing is pending on SCA10.
  Warmup plus3 same-process timed images required; interrupted outputs preserved.
- Neither final CPU bootstrap READY marker exists. After GPUeval writes READY, run
  taskq/hold_cpu_finalize/e2_6144_bootstrap.sh and e9_krea_bootstrap.sh onCPU.
  Require1000reference-image-cluster draws; never rerun completedCPUbootstraps.
- NEXT: harvest/audit new6K/8K AReA timing logs, monitor SCA10 when granted,
  and review/promote its successful GCC smoke receipt to canonical.
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
- Last live budget check7908/40000SUs available; check login2 before ANY new submission,
  include reservations and retain1600SUs. SCA10 queued; no live allocation cancelled.
- No remote. Commit explicit changed paths locally; no publication/external messages.

## Open questions
- Project incomplete until remaining6K/Krea,clean latency,freshGPU smoke,final eval,
  galleries/manuscript reconciliation and delivery are verified.
- SCA10 start/node grant unknown; minimum2hours requested to fit full6K4800s.
- GPU metric precision parity remains outside bounded CPU restoration tests.
