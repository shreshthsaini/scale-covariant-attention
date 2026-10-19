# Progressive SCA (label sca_prog), prepared 2026-10-18, not run on GPU

Question: HRDiT beats single-pass SCA on preference metrics at 4096^2 (IR 1.373 vs 1.124) while SCA wins on
detail, FID and patch FID. Is HRDiT's edge its progressive schedule rather than its attention? sca_prog is a
pure operator swap: HRDiT's released schedule unchanged, only the attention at the 2K and 4K stages replaced
by SCA.

## HRDiT recipe as run here (profile hrdit_dev_author)

Sources: frozen runner `/scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1/run.py` (generate
branch, kwargs for method hrdit) and vendored `vendor/hrdit/hrdit/{pipeline,attention,hap,spa,transformer}.py`
(pipeline.py `__call__` lines ~715-1045, `flowmatch_step` ~1080-1180).

- Stages: 1024^2 -> 2048^2 -> 4096^2 (`height=1024`, `target_heights=[2048,4096]`).
- Shared schedule: one 30-step schedule, `sigmas=linspace(1, 1/30, 30)`, mu from the 1K sequence length
  (4096 tokens) = 1.15. `pipe.scheduler.use_dynamic_shifting = False` in run.py/inference.py sets an
  instance attribute only; `set_timesteps` reads `config.use_dynamic_shifting` (True), so the effective
  schedule is the dynamic exponential shift with mu 1.15. Verified: the recorded HRDiT sidecar sigmas
  (1.0, 0.9892, ...) match our recomputation to 1e-6 (test_schedule_reproduces_recorded_hrdit_sigmas).
- Steps: 1K runs all 30. Stage s re-runs the last N_s timesteps of the same schedule (`dlfg_timesteps =
  timesteps[-N:]`): 2K N=17 (indices 13-29), 4K N=10 (indices 20-29). Total 57 transformer calls.
- Re-entry noise: 2K enters at sigma 0.8051, 4K at sigma 0.6123.
- Transition (pipeline.py ~909-933): VAE-decode the stage latent, bicubic upsample in pixel space to the next
  resolution, unsharp mask `2*img - GaussianBlur(k=3, sigma~U(0.1,2.0))(img)` (sigma from the global Torch RNG,
  seeded per prompt with the derived seed), VAE-encode (`latent_dist.mode()`), then
  `scale_noise`: x = sigma_k*noise + (1-sigma_k)*x with noise from the same per-prompt CPU generator.
- Guidance (distilled guidance embedding): 3.5 at 1K, 4.5 at 2K, 6.0 at 4K.
- Structure guidance inside every high-res step (`flowmatch_step`, flow_choice v_theta, upsampling latent,
  structure fft): pred_x0 low band (Butterworth order 4, ratio 0.2, on the 16-channel latent) is pulled toward
  the bicubic-upsampled previous-stage pred_x0 at the same timestep with alpha_i = a_s(N-i)/N, a = (1.0, 0.25);
  velocity momentum with beta_i = 0.5(N-i)/N, reset at each stage start. Native stage: plain Euler.
- Attention: 1K is dense SDPA with native RoPE (HRDiT's proportional scale is exactly 1/sqrt(d) at 4608
  tokens; HAP and SPA are off). At 2K and 4K HRDiT changes attention in four ways: NTK RoPE (theta x4 at 2K,
  x10 at 4K), proportional attention scale sqrt(log_4608(N)/d), HAP head pruning on all high-res steps
  (raster band of +-31 64-token blocks from the shipped scope plan, every 32nd block global, text global), and
  SPA position alignment (bundle-id RoPE averaging, group 80) on 2K steps 0-2. fp16 weights.

## sca_prog design

`scripts/sample_progressive.py` reproduces everything above with the stock diffusers FluxPipeline components
and swaps only the operator: stock FLUX.1-dev processors at 1K (not installed over), SCA (sca.attn) at 2K and
4K. All four HRDiT attention changes (NTK, proportional scale, HAP, SPA) are removed at those stages; SCA
uses native RoPE plus its pooled global branch. Pi from `/scratch/09032/saini_2/sca/calib/k1_1024.npz`, head
variant: scale-2 entry at 2K (pool factor 2), scale-4 entry at 4K (pool factor 4); n0 64, tile 8, center
pool, flex local branch, band 0, route 0, exactly the sca_nb operator settings.

Gate: `--gate-sigma 0.89` (the paper rule) for both stages, fixed a priori; no selection was made on any
prompt. It is inactive by construction because the stages enter at sigma 0.805 and 0.612, so every
high-res step mixes both branches (sidecar `gate_active_steps` = [0, 0]). The layout role the gate plays in
single-pass SCA is taken by the 1K stage. Any future per-stage gate must be chosen on dev4 only
(`/scratch/09032/saini_2/sca/dev4.jsonl`, abl50 indices 0-3) and frozen before the 46-prompt evaluation.

Deliberate deviations from the HRDiT runner (recorded in the sidecar): bf16 weights (SCA's validated
precision; HRDiT ran fp16), no per-step finite sync (finite check at each stage end only), vectorized but
arithmetic-identical Butterworth mask, pred_x0 reference keyed by step index rather than timestep value.
Seeding matches sample.py (CPU generator from image_seed) plus the HRDiT runner's global seed for the blur.

Outputs follow sample.py: `<stem>.png/.json` with `config.method=sca_prog`, `res`, `prompt_id`, `prompt`,
`seed`, `config_fingerprint`, `png_sha256`, `seconds` (CUDA-synced: text encoding, all stages, transitions,
final decode; excludes loading and PNG I/O). Extra fields: `stage_seconds` (denoise per stage),
`transition_seconds`, `text_seconds`, `decode_seconds`, `stage_sigmas`, `stage_shapes`, `blur_sigmas`,
`gate_active_steps`, `pi_trace` (with stage), `mu`, `peak_memory_bytes`. `--check` verifies a slice without
loading the model.

## Files

- `scripts/sample_progressive.py` (new). No shared file edited. Adding it changes
  `sca.runtime.implementation_hash()` for future sample.py runs from a new freeze (frozen copies unaffected).
- `tests/test_progressive.py` (9 CPU tests, tiny random FLUX + tiny VAE): release plan values, schedule vs
  recorded HRDiT sigmas, Butterworth vs upstream loop, guided step reduces to Euler, stage hand-off shapes and
  per-stage processors (stock 6 steps then SCA), pi steps/sigmas per stage, determinism and seed dependence,
  CLI sidecar fields, resume skip, --check barrier, fingerprint refusal, pi scale 2/4 from the real table.
  Full suite: 140 passed (`sca` env, `python -m pytest -q tests`).
- `repro/progressive_sca/prepare_eval.py` (new): prepare/verify/timing for the evaluation workspace. Reused
  rows are re-hashed against `baseline_expansion46_v1/inputs.json` (dry run passed for all 138 reused pairs).

## Spool tasks (held: /scratch/09032/saini_2/sca/taskq/hold_prog, code expected at /scratch/09032/saini_2/sca/code/PROG)

1. `p_prog_a_smoke_000_002.sh`: abl50 0-1 (dev prompts), then `--check` and a timing/peak-memory printout.
2. `p_prog_b_gen_000_005.sh` ... `p_prog_b_gen_045_050.sh`: all 50 abl50 prompts in chunks of 5, same output
   root `/scratch/09032/saini_2/sca/out/prog/sca_prog/4096x4096` (chunk 0 resumes past the smoke), WandB group
   progressive-sca.
3. `p_prog_z_eval46.sh`: gate (fails if generation tasks are pending; waits at most 1800 s while they run),
   `--check` on all 50, prepares `/scratch/09032/saini_2/sca/eval/prog46_v1` (sca_prog, hrdit_dev_author,
   sca_s42 from out/budget/sca_s42, sca_nb, ultraimage; 46 prompts; reference method sca_nb; frozen baseline46
   evaluator code), runs `evaluate_corrected.py` and `eval_extra_portable.py` (GPU, bootstrap 0), the score
   barrier, `paired_bootstrap.py` for sca_prog vs hrdit_dev_author (also vs sca_nb, vs sca_s42, and
   hrdit vs sca_s42), and `timing.csv/json` with per-stage means. CPU FID/KID intervals afterwards:
   `bash /scratch/09032/saini_2/sca/eval/prog46_v1/intervals_cpu.sh` on a CPU node.

Order: freeze code to PROG, publish smoke, inspect images and timing, then publish the ten chunks; publish the
eval after the chunks and the sca_s42 tasks are done.

## Expected GPU time (GH200, one GPU per task; estimates, not measurements)

Per image: 1K 30 steps ~6 s, 2K 17 SCA steps ~8-12 s, 4K 10 SCA steps ~23 s (single-pass SCA ~2.3 s/step),
transitions plus final decode ~15 s; about 55-70 s, roughly a third of HRDiT's ~180 s. Smoke ~10 min (load
and two flex compiles). Generation: 10 chunks x (load+compile ~3-4 min + 5 x ~1 min) ~ 1.5 GPU-hours.
Evaluation of 230 images ~1 GPU-hour. Total about 2.5-3 GPU-hours; headers reserve 1200/1500/6600 s.
