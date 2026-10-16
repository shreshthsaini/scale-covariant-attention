# FLUX.2-dev port of SCA (experiment E8)

Written 2026-10-16 by the FLUX.2 port agent. Code is uncommitted; the lead freezes it to
`/scratch/09032/saini_2/sca/code/FLUX2` and commits.

## Architecture facts (verified from the snapshot configs and diffusers 0.39 source)

- Snapshot `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c` under `HF_HOME=/scratch/09032/saini_2/tmp/hf`.
  Before this port only `model_index.json` was cached. The diffusers-format weights (transformer 61 GiB,
  text encoder 45 GiB, VAE, tokenizer; 106 GiB total) were downloaded on 2026-10-16 from a gg node
  (gated access works with the cached token). The single-file `flux2-dev.safetensors` and `ae.safetensors`
  were not downloaded.
- diffusers 0.39 ships `Flux2Pipeline` and `Flux2Transformer2DModel`; no newer diffusers is needed.
- Transformer: 8 double-stream blocks (`Flux2Attention`) + 48 single-stream blocks
  (`Flux2ParallelSelfAttention`) = 56 attention layers; 48 heads x 128 = 6144 width; mlp_ratio 3
  (SwiGLU); patch_size 1, in_channels 128; no biases anywhere; ~32B parameters.
- Double blocks: separate text projections (`add_q/k/v_proj`, `norm_added_q/k`, `to_add_out`), joint
  attention over `[text | image]`, same structure as FLUX.1.
- Single blocks: one fused `to_qkv_mlp_proj` produces QKV plus the MLP input; attention output and
  SwiGLU output are concatenated and go through one fused `to_out`. Block modulation is shared across
  layers (`Flux2Modulation`), outside the attention module, so SCA does not touch it.
- QK RMSNorm per head before RoPE, as in FLUX.1. RoPE uses the same `apply_rotary_emb`.
- RoPE: 4 axes of 32 dims each (t, h, w, l), theta 2000. Image ids `(0, row, col, 0)` (int64, so no bf16
  rounding problem as in FLUX.1); text ids `(0, 0, 0, j)` for j < 512. Text rotates only on axis l,
  image only on axes h, w.
- VAE `AutoencoderKLFlux2`: 8x spatial downsampling, 32 latent channels, then a 2x2 patchify into 128
  channels. One token = 16 x 16 pixels, as in FLUX.1. Supports `enable_tiling`.
- Text encoder: Mistral-Small-3.1 style `Mistral3ForConditionalGeneration` (24B: 40 layers, width 5120,
  plus a Pixtral vision tower). Prompt features = hidden states of layers 10, 20, 30 stacked to 15360
  dims, padded to 512 tokens, with a fixed system prompt. No pooled embedding.
- Guidance-distilled: `guidance_scale` is embedded (default 4.0); no CFG batch.
- Scheduler: FlowMatchEuler with exponential time shift; the pipeline sets
  `mu = compute_empirical_mu(seq_len, steps)` (ignores max_shift). At 1024^2 and 28 steps mu = 2.1514.
  At 4096^2 the linear extrapolation gives mu = 11.55, which makes every sigma about 1.0 (degenerate).
- Native grid n0 = 64 (1024^2): the pipeline's default size (`default_sample_size` 128 x 8 px) and
  the anchor of the mu fit (seq 4096). BFL advertises outputs up to 4 MP, so 2048^2 may also be
  in-distribution (see risks).

## Design decisions

- **n0 = 64, target 4096x4096** (4x per side, pool factor 4, local window 64, tile 8): identical
  geometry and evaluation resolution to the FLUX.1 main result. 8192^2 with n0 = 128 would be 4x per
  side of 2048^2, but full attention there costs about 16x the 4096^2 attention (hours per image), so
  it is not a feasible baseline.
- **mu_mode native**: mu = compute_empirical_mu(min(seq, 4096), steps) = 2.1514 at every
  resolution. This is the FLUX.2 analogue of FLUX.1's mu = max_shift protocol (mu of the native
  sequence length) and avoids the degenerate 11.55. Implemented by patching the pipeline module's
  `compute_empirical_mu` (looked up at call time).
- **Steps 28, guidance 4.0, seed 0**: 28 steps to keep the "first 8 of 28 steps global-only" rule;
  4.0 is the FLUX.2 pipeline default (there is no AReA comparison on FLUX.2 to share a guidance with).
- **Gate sigma_star = 0.959**: sigmas at 28 steps, mu 2.1514: step 7 = 0.9627, step 8 = 0.9555. The
  midpoint 0.959 makes exactly steps 0..7 global-only (tested). For FLUX.1, 0.89 sits between 0.9045
  and 0.8876 by the same rule.
- **Same method as FLUX.1**: `sca/flux2.py` subclasses `SCAContext` (only `coarse_rope` changes, for
  4-D ids: row/col replaced by floor(id / f) for queries and block indices for pooled keys; t and l
  untouched) and `SCAProcessor` (only `__call__` changes; `sca`, `calibrate`, `branches`, the flex and
  gather local branches, the fused compiled path, gate and pi mix are reused unchanged from
  `sca/attn.py`). Text keys always included; text queries take O_glob.
- **Layer index for pi tables**: double blocks 0..7, then single blocks 8..55 (as in FLUX.1). Tables
  are [56, 48, 14].
- **Memory**: the text encoder (~48 GB) and transformer (~64 GB) do not fit together. Prompts are
  encoded first, saved per prompt to `/scratch/09032/saini_2/sca/flux2_embeds/<sha256>.pt`
  (15.7 MB each), the text encoder is freed, then the transformer is loaded with `device_map=cuda`
  (no 64 GB CPU copy). The pipeline runs with `text_encoder=None` and `prompt_embeds`. The smoke task
  pre-encodes all 66 campaign prompts, so later tasks never load the text encoder. VAE tiling is on
  for areas >= 2048x4096. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in every task.
- Not supported: reference-image editing with KV cache (the pre-hook raises), attention masks, and the
  band guard (dropped from the final method, so no `--band` flag).

## Files (new only; no shared file was edited)

- `sca/flux2.py`: `Flux2SCAContext`, `Flux2SCAProcessor` (double), `Flux2SCASingleProcessor`,
  `install`, `empirical_mu`, `set_mu`, `sigma_schedule`, `gate_from_steps`, `encode_prompts`
  (cached), `load_pipeline`, `flops`.
- `scripts/sample_flux2.py`: same CLI and sidecar format as `scripts/sample.py` (methods
  full|sca|sca_uniform|sca_layer|sca_switch, `--label`, `--gate-sigma`, `--pi-table`, ...), plus
  `--mu-mode native|default`, `--encode-only`. Sidecars also record `mu`, `snapshot`, `text_encoder`.
  `seconds` covers denoising + VAE decode (prompt embeddings precomputed).
- `scripts/calibrate_flux2.py`: same statistics and merge as `scripts/calibrate.py` (`sca/calib.py`
  reused), plus `--start/--end/--no-merge` to split prompts across tasks and a missing-part check
  before merging.
- `tests/test_flux2.py`: 12 CPU tests on a tiny random Flux2 transformer.
- `/scratch/09032/saini_2/sca/prompts/flux2_eval50.jsonl`: first 50 Aesthetic-4K eval prompts excluding
  dev4 (dev4 are source rows 0..3, so this is rows 4..53).

## Tests (all pass: `pytest tests` gives 30 passed, 18 old + 12 new)

- full mode equals the stock processors (whole model, square and non-square grids; module outputs)
  and uninstall restores the stock processors;
- local branch with a window covering the grid equals full attention (flex and gather, two tiles,
  double and single blocks);
- global branch with f = 1 and pi = 0 (and pi = 0.37) equals full attention; with pi = 0 the window
  size is irrelevant;
- 4-D coarse RoPE and local window against brute-force references;
- gate: sigma above threshold equals pi = 0, below equals the plain mix; text outputs independent of
  pi; the double-block image output is exactly 0.3 O_glob + 0.7 O_loc for pi = 0.7; wrong pi table
  shapes are rejected;
- flex (fused and unfused) equals gather;
- calib mode leaves sampling unchanged and gives pi = 1 when the emulated window is exact;
- schedule: native mu 2.1514, default 11.55 at 4096^2, gate 0.959 leaves exactly 8 global steps;
- script plumbing with a tiny `Flux2Pipeline` (patched loaders): sample_flux2 writes sidecars,
  resumes without re-encoding, full method works; calibrate_flux2 splits, refuses to merge with
  missing parts, merges and loads as a PiTable.
- Also checked on CPU: the real FLUX.2 tokenizer chat template gives 512 ids under transformers 5.19.
  The real 32B model was not run (no GPU runs by this agent).

## Task scripts (`/scratch/09032/saini_2/sca/taskq/hold_flux2/`, all `cd /scratch/09032/saini_2/sca/code/FLUX2`)

Release `f000` alone first and check its log before moving the rest to `pending`.

| Task | What | Output |
|---|---|---|
| f000_flux2_smoke | encode 16 calib + 50 eval prompts; calibrate 1 prompt at 1024^2; full 1024^2, SCA 4096^2 (smoke table), full 4096^2 on 1 prompt; prints seconds and peak GB | `calib/flux2_smoke.npz`, `out/flux2_smoke/` |
| f010..f013_flux2_calib_p0..3 | 4 prompts each of the first 16 `scripts/calib_prompts.jsonl`, 1024^2, scales 2,4 | `calib/flux2_1024_parts/` |
| f100..f145_flux2_full_4096x4096 | full attention, 5 prompts each, label `flux2_full` | `out/flux2/flux2_full/4096x4096` |
| f200_flux2_calib_merge | waits for f010..f013, merges (fails if a part is missing) | `calib/flux2_1024.{npz,json}` |
| f300..f345_flux2_sca_4096x4096 | wait for f200, SCA with the scale-4 table, gate 0.959, label `flux2_sca` | `out/flux2/flux2_sca/4096x4096` |
| f900_flux2_eval | waits for f1xx and f3xx; links common ids; eval_extra (ir, aesthetic, hpsv2, hpsv21, clipiqa, arniqa, is, latency) and constancy_rope eval.py (clip, pickscore, musiq, fid, kid, patch fid/kid), reference `flux2_full` | `eval/flux2_4k_{extra,base}` |

All waits target tasks that sort earlier, so the spool cannot deadlock with any number of workers.
The SCA and eval tasks hold a GPU while waiting (as b999 did).

## Expected cost on one GH200 (estimates scaled from measured FLUX.1 numbers, not measured)

- Weights: transformer 64.4 GB bf16 on GPU; text encoder ~48 GB during the encode phase only.
- Peak memory: ~68 GB at 1024^2; ~80 to 88 GB at 4096^2 (single-block fused projection output is
  7.3 GB at 66k tokens). GH200 has ~95 GiB usable, so 4096^2 should fit; the smoke task measures it.
- Time per 4096^2 image: full ~6.5 min (attention 2x FLUX.1 per layer for 48 heads, linear layers
  2.7x); SCA ~3 min. 1024^2 full: ~25 s. Calibration ~7 min per prompt. Model load ~1 to 2 min per
  task.
- Campaign: smoke ~0.5 h, calibration ~2 h, full 50 images ~5.5 h, SCA ~2.5 h, eval ~0.5 h:
  about 11 GPU-hours.

## Open risks

1. **OOM at 4096^2** is the main risk (no GPU test). Fallbacks: offload the VAE until decode, or
   `pipe.transformer.enable_group_offload`, or drop to 3072^2 (3x per side).
2. **Full attention may not collapse on FLUX.2 at 4096^2** as it did on FLUX.1 (FLUX.2 advertises up
   to 4 MP, so 4096^2 is only 2x per side beyond its range). Then SCA vs full is a closer contest
   than on FLUX.1. A variant with n0 = 128 (window 128, pool 2; calibrate at 2048^2) is possible with
   `--n0 128` but costs more and is untested.
3. Text-encoder load through transformers 5.19 with `dtype=` and `device_map=cuda` is untested on
   GPU; the tokenizer path is verified. Mistral prints a deprecation notice about processor kwargs.
4. torch.compile of the fused flex path with 48 heads is untested on GPU (same code as FLUX.1, only
   the head count differs). `--local-impl gather` or the unfused path are fallbacks.
5. FID on 50 images is weak; patch metrics and paired preference metrics carry the comparison.
6. Guidance 4.0 and 28 steps differ from the FLUX.2 model card (50 steps); chosen to keep the gate
   rule and the step budget of the FLUX.1 protocol.
