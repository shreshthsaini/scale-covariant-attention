# AReA official code: read-through (2026-10-15)

Checkout: /scratch/09032/saini_2/sca/third_party/AReA, github.com/KooroshRH/AReA,
commit 9e38ec2 (Apache-2.0). Paper: arXiv 2610.17227 (HTML text checked for setup).
Released backbone: FLUX.1 only. Qwen-Image, FLUX.2, Ideogram 4 are "coming soon"; the
Qwen-Image rows of their tables cannot be reproduced from this release.
No evaluation or timing code is included in the repo.

## Files
- generate.py: CLI. MODELS = {flux1: FluxPipeline, checkpoint FLUX.1-Krea-dev, steps 28,
  guidance 4.5}. `--checkpoint` swaps the weights but keeps the family steps/guidance.
  Loads bf16, `pipe.vae.enable_tiling()` always, `pipe.to("cuda")` unless GPU < 60 GB
  (then model CPU offload). Generator: `torch.Generator(cuda).manual_seed(seed)`, the SAME
  seed for every prompt. AReA seed (jitter, Halo draws, random pooling) = generation seed.
  Output name `{i:04d}_{slug}_{W}x{H}_seed{s}.png`; no metadata, no timing.
- area/__init__.py: `AReA(pipe, **AReAConfig)`; `AReAConfig(phase1_steps=None,
  pooling="subsample", halo=8, bridge_ratio=0.5, seed=0)`; `.remove()`; `.bootstrap()`.
- area/backbones/flux.py: FluxBackbone. NATIVE_PATCHES = 64 (1024 px), PHASE1_STEPS = 8,
  BOOTSTRAP_SIDE = 1024. Replaces every block's attention processor (19 dual + 38
  single), adds a transformer forward pre-hook (step, grid, schedule, RoPE tables) and an
  x_embedder forward hook (DWT detail map and Phase II layout from the embedded tokens).
  Replaces the scheduler with `from_config(..., base_shift=max_shift)`.
- area/schedule.py: f0, pooling ramp, w_start, window sizes (Eqs. 2 to 11).
- area/attention.py: Phase I K/V pooling, Phase II grouped windowed attention, cost fitting.
- area/tiling.py: jittered base tiling, 3-level Haar DWT map, quadrant splits, jagged Halo,
  raised-cosine merge weights.
- area/bridge.py: global-KV bridge (token score = squared distance of unrotated key from
  canvas-mean key, 8x8 cell quotas).
- HRGen-1K/prompts.jsonl: 1000 prompts, ids hrgen_0000..hrgen_0999.

## Defaults for FLUX.1 (code = paper unless noted)
- Steps 28, guidance 4.5 (code family default, also used if `--checkpoint FLUX.1-dev`).
  The paper never states guidance, checkpoint (dev vs Krea-dev) or seeds.
- Scheduler: FlowMatchEuler, base sigmas linspace(1, 1/28, 28), mu = 1.15 at every
  resolution (base_shift forced to max_shift). Same sigmas as constancy_rope (which sets
  max_image_seq_len = seq len). sigma[0..9] = 1.0, .9884, .9762, .9634, .9499, .9356,
  .9205, .9045, .8876, .8696; last 0.1047.
- sigma* = 0.89 (paper App. A.3). Code implements it as a step count: Phase I = steps 0..7,
  Phase II = steps 8..27 (sigma[8] = 0.8876). Step located by argmin over
  `scheduler.timesteps` vs `pipe._current_timestep`.
- f0 = smallest common grid divisor >= 2 with max(h,w)/f0 <= 64; fmin = 2; linear ramp
  snapped to non-increasing common divisors. Pooling = block centre (`off = f//2`).
  Phase I RoPE: query coords = floor(pos/f), pooled key coords on (h/f)x(w/f) grid.
  Text keys are never pooled.
- Phase II: w_start = largest w with N(w)(w^2+T)^2/(hw+T)^2 < 1 and N(w) > 1, searched
  from max(h,w) down to max(64, extent/f_last); then geometric shrink to w_min = 64, each
  rung cost-capped by the first. T = 512 text tokens (priced explicitly).
- delta_max (Halo) = 8 patches, jagged per row/column, widths drawn per (seed, step).
- rho (bridge) = 0.5: R = round(0.5 w)^2 bridge tokens per window, cut first (then the Halo)
  so a Phase II step never exceeds full attention cost (commit 6008454).
- DWT splits: 3 Haar levels, gate alpha ramps on tile-energy CV in [0.05, 0.25], split
  candidates in lowest half and below mean, budget = base-tiling cost, dedupe at 0.8 cover.
- Text output in Phase II = mean over windows. Image output = weighted merge / coverage.
- VAE: tiled decode always (`enable_tiling`), default diffusers tile settings.

Schedules from the code (H x W px, 28 steps, T = 512):
| res | grid | factors (Phase I) | windows (Phase II, 20 steps) |
|---|---|---|---|
| 4096x4096 | 256x256 | 4,4,4,4,2,2,2,2 | 180,170,161,153,145,137,130,123,116,110,104,99,94,89,84,80,75,71,68,64 |
| 2048x4096 | 128x256 | 4,4,4,4,2,2,2,2 | 151,144,138,132,113,112,111,110,105,101,96,92,88,84,80,77,73,70,67,64 |
| 6144x2048 | 384x128 | 8,8,8,4,4,4,2,2 | 192,173,171,161,152,144,136,128,121,114,108,102,96,91,85,81,76,72,68,64 |
| 5120x5120 | 320x320 | 5,5,4,4,4,2,2,2 | 225,...,64 |
| 6144x6144 | 384x384 | 6,6,4,4,4,3,3,2 | 271,...,64 |
Transposed aspect ratios give identical schedules. Base-tiling cost per Phase II step in
units of one full-attention step (T = 512 counted):
4096^2: .99 .79 .64 .52 .43 .34 .28 .50 .40 .33 .26 .22 .18 .15 .21 .18 .14 .11 .10 .08
2048x4096: .98 .82 .69 .58 .96 .92 .89 .86 .72 .62 .51 .44 .37 .41 .35 .30 .25 .21 .18 .15
6144x2048: 1.13 1.13 1.08 .85 .68 .55 .44 .35 .74 .59 .48 .39 .31 .31 .24 .20 .19 .16 .13 .10
At 6144x2048 no window is cheaper than full attention once the 512 text tokens are
counted, so w_start falls back to 192 and the first three Phase II steps cost more than
full attention; fit_extra_keys then drops both the bridge and the Halo on those steps
(before DWT splits, which can only lower cost). Steps that are not monotone (e.g. .28 to
.50) are window-count jumps when w crosses a divisor of the grid.

## Paper protocol (Section 4, from arXiv HTML)
- Benchmarks: Aesthetic-4K eval (195 prompts with 4K references) and HRGen-1K (1000).
  Resolutions 6144x2048, 2048x4096, 4096x4096, 5120x5120 (written WxH or HxW, not stated).
- Metrics: FID, FIDp (patch FID on local crops, crop protocol not stated), IS, Aesthetic
  Score (LAION), MUSIQ, CLIP-IQA, ARNIQA, CLIP score, ImageReward, PickScore, HPSv2.
  FID resizes to 299. Appendix G uses FID against the same model at native resolution.
- Latency: "average generation time per image in seconds", NVIDIA H100. No warmup, sync or
  scope stated. FLUX.1 Table 1 latency (s), full-attention Base vs AReA:
  2048x4096 97 vs 66; 6144x2048 185 vs 107; 4096^2 314 vs 151; 5120^2 685 vs 269.
- FLUX.1 Table 1 AReA (IR / Aesthetic / CLIP / PickScore / FID / FIDp):
  2048x4096 1.088/6.34/29.23/22.95/151.9/62.5; 6144x2048 1.045/6.26/29.02/22.74/161.0/62.0;
  4096^2 1.325/6.37/29.71/23.31/148.5/58.2; 5120^2 1.232/6.29/29.34/22.94/152.1/63.3.
  SEGA 4096^2: 1.270/6.31/29.16/23.16/150.6/71.8. DyPE 4096^2 IR 1.107.
- All baselines from official implementations. Ablations (Table 3) at 4096^2 on FLUX.1.

## Differences between code and paper, and things to watch
1. Default checkpoint is FLUX.1-Krea-dev with guidance 4.5; the paper only says "FLUX.1".
   Our wrapper uses FLUX.1-dev and keeps 4.5 by default. FLUX.1-dev's usual guidance (and
   constancy_rope sample.py) is 3.5. Decide once and use the same value for every method.
2. sigma* is a step count (8 of 28), not a sigma test. Changing steps changes sigma*.
3. AReA seeds the noise with one CUDA generator seed for all prompts; we use a CPU generator
   with the constancy_rope per-prompt derived seed. Image-level results differ from theirs;
   the distribution is the same.
4. The bridge/Halo cost cap (fit_extra_keys, commit 6008454) and the jagged Halo edge
   (commit ad74a28) are separate commits after the initial code drop; the paper text
   describes a ring whose width is drawn per window and step in [0, delta_max] and does
   not mention the cap. Whether Table 1 used them is not stated.
5. The bootstrap variant (App. F) is FLUX.1 only and not used for Table 1.
6. Canvases with max(h,w) <= 64 patches run plain attention.

## Wrapper
scripts/run_area.py: constancy_rope sample.py CLI and sidecar format, method "area",
AReA defaults above, local FLUX.1-dev snapshot, CUDA-synced latency (prompt encode +
denoise + VAE decode), peak memory, resumable by config fingerprint. `--dry-run` works on CPU.

## Env `sca` (/work/09032/saini_2/vista/library/uv-envs/sca)
Python 3.11.13. Exact copy of crope's `uv pip freeze` (176 pkgs, installed `--no-deps`
because crope's set does not resolve: huggingface-hub 2.2.0 vs diffusers 0.39.0 pin <2.0)
plus torch-fidelity 0.3.0 (eval Inception backend, absent from crope). torch
2.11.0+cu128 (CUDA 12.8, cuDNN 9.19), diffusers 0.39.0, transformers 5.19.0. AReA needs
nothing extra. Tiny random FluxTransformer2DModel with AReA attached runs on CPU.

## Eval (constancy_rope/scripts/eval.py) from `sca`
Runs (smoke-tested on CPU with tiling/spectrum). Reads `config.method`, `config.res` (HxW),
`config.prompt_id`, `config.seed`, PNG size must match. Set TORCH_HOME=/scratch/09032/
saini_2/tmp/torch (eval.py defaults to tmp/cache, which lacks the weights).
Cached: CLIP L/14, PickScore + CLIP-H processor, ImageReward ckpt, HPSv3 + Qwen2-VL-7B (HF);
Inception FID weights and MUSIQ koniq (TORCH_HOME). Not usable: imagereward (package import
fails on transformers 5.x, same in crope), hpsv3 (package not installed), clipiqa (weights
not cached). Paper metrics missing from this harness: IS, LAION Aesthetic, ARNIQA, HPSv2.
References: /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_{eval,refs}.jsonl,
195 ids, all 195 reference JPGs present under HF_HOME datasets--zhang0jhon--Aesthetic-4K.

## Other baselines (no setup done)
- HRDiT: github.com/zylwithxy/HRDiT (ECCV 2026). FLUX.1-dev out of the box, but multi-stage
  (1024 -> 2048 -> 4096, ntk_factor [4, 10], 17 + 10 high-res steps), square `--resolutions`
  only, fp16 default, FlexAttention (torch >= 2.7), pins transformers==4.48.0 and needs
  pytorch-wavelets: wants its own env. Diffusers port: huggingface.co/remyxai/hrdit-flux-modular.
- DyPE: github.com/guyyariv/DyPE (436fe61); SEGA: github.com/rajabi2001/sega (ddaab08);
  UltraImage: github.com/thu-ml/DiT-Extrapolation branch ultra-flux. Checkouts in
  /scratch/09032/saini_2/constancy_rope/third_party; ported as crope methods dype, sega_official.
