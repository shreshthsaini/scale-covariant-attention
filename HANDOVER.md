# Scale-Covariant Attention (SCA): handover

Goal: strong training-free method paper beating AReA (arXiv 2610.17227) on FLUX.1-dev
high-resolution generation (4096^2, 2048x4096, 6144x2048), faster than full attention.
Spec: PROPOSAL.md (approved 2026-10-15) plus the calibration refinement below.

## Method (as built)
Per head h, per sigma: image queries get o = pi_h(sigma) * O_loc + (1 - pi_h(sigma)) * O_glob.
- O_loc: blocked neighborhood attention at full resolution, native-size window
  (n0 x n0 tokens = 64x64 for FLUX), text keys included, original RoPE.
- O_glob: K/V pooled by s onto the native grid (block center), query and key RoPE
  coordinates on that grid (AReA Phase-I geometry with f = s fixed), text keys included.
- Text queries: global branch only.
Calibration (output matching): run FLUX at in-distribution resolution n_T (1024, 1280)
with full attention; emulate scale s (window n_T/s, pooling s); least-squares
pi = argmin ||O_full - (pi O_loc + (1-pi) O_glob)||^2 per (layer, head, sigma bin).

## Layout
code: this dir (sca/, scripts/, slurm/, tests/). Scratch: /scratch/09032/saini_2/sca
(taskq spool, logs, telemetry, calib, out). Fleet: slurm/fleet.sbatch (job SCA1).

## State
See .agent/STATE.md.

## Code (implemented 2026-10-15, CPU-tested only)
- sca/attn.py: SCAProcessor on every FluxAttention (modes full | sca | calib), SCAContext,
  PiTable, install() with a transformer pre-hook feeding grid + scheduler sigma,
  attention_flops() cost model (`python -m sca.attn`).
- sca/calib.py: CalibStats (A, B, E_fg, E_fl, F2, count per scale/layer/head/sigma bin;
  text and Chebyshev-radius attention mass on 256 random queries), merge/report, transfer.
- sca/runtime.py: loader, seeding, stems, sidecars copied from constancy_rope conventions.
- scripts/calibrate.py, scripts/sample.py (sidecars readable by constancy_rope eval.py),
  scripts/calib_prompts.jsonl (48 prompts). Tests: tests/test_sca.py (10 pass on CPU).
- --res is HxW (constancy_rope convention). Scheduler: --mu-mode max_shift (default).
- Known cost: the local branch gathers K/V windows per tile; at t=8 the copy traffic
  dominates (est. 4096^2 attention 91 s vs 214 s full; t=16 gives 45 s).
