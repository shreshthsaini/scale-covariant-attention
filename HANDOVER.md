# Scale-Covariant Attention (SCA): handover

Goal: strong training-free method paper beating AReA (arXiv 2610.17227) on FLUX.1-dev
high-resolution generation (4096^2, 2048x4096, 6144x2048), faster than full attention.
Spec: PROPOSAL.md (approved 2026-10-15) plus the calibration refinement below.

## Status (2026-10-16 09:40): campaign complete, paper drafted
Final method (SCA, code 2d11740): sigma > 0.89 global-only (K/V pooled onto native grid, rescaled RoPE);
afterwards per head pi_h(sigma) * O_loc + (1 - pi_h(sigma)) * O_glob, O_loc = exact 64x64 FlexAttention
sliding window, pi table calib/k1_1024.npz (output-matching calibration at 1024^2). Band guard optional.
Results (191 Aesthetic-4K prompts, FLUX.1-dev, guidance 4.5, vs AReA released code): results/final4k_nb_*,
final_2048x4096_*, final_2048x6144_*, final4k_full_*; ablations results/ablnb_*, localnb_*; latency
results/latency_4096_batch1.csv. Paper: paper/main.pdf, paper/sca_overleaf.zip.
Open: no git remote yet; a second model (Qwen-Image/FLUX.2) and DyPE/SEGA/HRDiT baselines not run.

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
  scripts/calib_prompts.jsonl (48 prompts). Tests: tests/test_sca.py (16 pass on CPU).
- --res is HxW (constancy_rope convention). Scheduler: --mu-mode max_shift (default).
- Local branch has two implementations with identical keys (SCAContext.local_impl,
  `--local-impl` in sample.py and calibrate.py). `flex` (default since 2026-10-15) runs
  FlexAttention on a cached BlockMask: image tokens in tile-major order (8x16 query blocks,
  8x8 key blocks shifted so unclamped windows start on a block edge), text keys first,
  mask_mod = the gather-window membership. In sca mode the whole layer (RoPE, pooling, both
  branches, pi mix) is one torch.compile graph (`fused=True`). `gather` is the original
  K/V window copy path. Verified on GH200: flex vs gather max |diff| 1 bf16 ulp (0.0156) and
  <2.4e-5 in fp32 at 64^2..256^2 and 128x256 grids, both equally far from an fp32 reference;
  CPU fp64 tests agree to 1e-15. `local_impl` is written to the sidecar, not fingerprinted.
- Per-layer attention, GH200, min of interleaved runs (scripts/bench_attn.py), t=8:
  4096^2 full 87 ms, SCA flex 18.7 ms, SCA gather 65.6 ms; 2048^2 7.1 / 5.2 / 16.9 ms;
  6144x2048 49.6 / 14.0 / 49.2 ms. Per step (scripts/bench_step.py, GPU shared with other
  jobs, so absolute numbers are inflated): 4096^2 full 11.8 s, flex 4.7 s, gather 9.8 s.
- Env caveat: the `sca` uv env lost its zero-byte files to the scratch purge (92 in torch,
  5 in triton), which breaks torch.compile. sca/envfix.py serves them from memory at
  runtime; reinstalling torch and triton into the env is the real fix.
