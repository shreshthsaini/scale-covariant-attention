# UltraImage protocol audit

Paper: "UltraImage: Rethinking Resolution Extrapolation in Image Diffusion Transformers", Zhao, Yan, Yang, Zhu, Zhang, Liu, Li, Zhu. arXiv 2512.04504v1 (only version; submitted 4 Dec 2025).
Local copy: papers/2512.04504/paper.pdf, paper.txt (18 pages: 9 main + 9 supplementary; the supplementary restarts its own page numbering, so "Supp. p.2" is PDF page 12).
Code: github.com/thu-ml/DiT-Extrapolation, branch `ultra-flux`. Local read-only checkout /scratch/09032/saini_2/constancy_rope/third_party/UltraImage at commit 586ae882 ("Fixed bugs for flux.", 2026-06-01). Its file list matches the remote `ultra-flux` tree from the GitHub API. Branch `ultra-qwenimage` listed via the API; qwen/base.yaml and qwen/inference_rope.py read as text. Nothing executed.

## a. Base models and checkpoint

- "Extensive experiments on Flux [22] and Qwen-Image [39]" (Sec. 1, p.2). "In this work, we adopt Flux.1.0-dev [22] as our text-to-image backbone" (Supp. Sec. 8.1, PDF p.12).
- Flux code README: "You can download the model [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)". So FLUX.1-dev, not FLUX.1-Krea-dev (Krea never mentioned).
- Qwen-Image training resolution is printed as "1328p" (Fig. 1 caption p.2; Supp. Sec. 10, Fig. 11) but as "1368p" in the Table 1 caption (p.8). Flux training resolution printed as "ranging from 256^2 to 2048^2" (Table 1 caption).

## b. Prompt sets

- "we randomly select 1K high-quality captions from LAION-5B covering diverse scenarios, following the general setup of prior work [14]" (Sec. 6.1 Evaluation, p.7). [14] = ScaleCrafter (He et al.).
- So: LAION-5B, 1,000 captions, random selection plus an unspecified "high-quality" filter. Prompt list not released: `base.yaml` has `prompts_low: /` and `prompts_high: /` placeholders; no prompt files in either branch.
- Fig. 1 caption says "All prompts used in this paper are provided in the Appendix" (p.2), but the extracted appendix contains no prompt list. NOT FOUND.
- Aesthetic-4K: NOT USED. HRGen-1K, DrawBench, GenEval, DPG, COCO: NOT USED.
- Images per prompt: NOT STATED. Seeds: NOT STATED in the paper. Code: config `seed: 42` (global `torch.random.manual_seed`), and every pipeline call uses `generator=torch.Generator("cuda").manual_seed(0)` (infer_unified.py lines 93, 117).

## c. Resolutions

- Table 1 caption (p.8): "All methods are evaluated at 4096 x 4096 on Qwen-Image ... and 3600 x 3600 on Flux".
- Three scenarios (Sec. 6.1, p.7): (1) direct resolution extrapolation, "4096 x 4096 for Qwen, 3600 x 3600 for Flux"; (2) guided resolution extrapolation on Flux, 1024x1024 generated, upsampled to 3600x3600 and used as guidance, "follows prior work [5, 11]" (HiFlow, I-Max); (3) guided view extrapolation on Flux, a 1024x1024 image placed at the center of a 3600x3600 canvas "following the SDEdit [26]".
- There are NO FLUX numbers at 4096x4096. The only 4096x4096 numbers are Qwen-Image.
- Qualitative only: Qwen-Image up to 6K x 6K (Fig. 1) and 4096 x 4096 (Fig. 11).

## d. Sampler settings

Paper:
- Hyperparameters: "The dominant frequency k = 9 for Flux and k = 8, 9 for Qwen-Image. lambda_min = 1.0, lambda_max = 1.3, p = 2 for all models" (Sec. 6.1, p.7).
- Head pattern types are fixed "from a single diffusion step (requiring only ~2% of total inference cost for 50 steps)" (Sec. 5, p.6).
- Steps, guidance scale, scheduler shift, precision for Flux: NOT STATED. Baseline settings (PE, PI, NTK, YaRN, Entropy, HiFlow, I-Max): NOT STATED.
- Guided resolution: "building upon I-max, we first generate a 1024 x 1024 image and then upsample it ... using the upsampled result as guidance" (Supp. Sec. 9, PDF p.12). Guided view: replace the central region of the high-res x0-prediction with forward-diffused copies of the low-res image (Supp. Sec. 9).

Released Flux code (flux/base.yaml, flux/infer_unified.py, flux/pipeline_flux_unified.py; this is the guided `super_imax` path; a direct-extrapolation config is not shipped):
- Stage 1 at `train_resolution: [1024, 1024]`: `num_inference_steps1=30`, `guidance_scale1=3.5`, `time_shift_1=3`.
- Stage 2 at `resolution: [3600, 3600]`: `num_inference_steps2=30`, `guidance_scale2=7`, `time_shift_2: 6`, `guidance_schedule: super_imax`. The stage-1 latent is decoded, bicubic-upsampled to 3600x3600, re-encoded, and used as an I-Max-style low-frequency (DWT) projected-flow guidance with cosine decay. Stage 2 starts from fresh noise (`prepare_latents(..., latents=None)`), pipeline_flux_unified.py lines 777 to 818 and 940 to 949.
- `use_dynamic_shifting = False`, static shift written into the scheduler config; bf16 (`torch_dtype=torch.bfloat16`); `proportional_attention: true`; `swin_pachify=True`.
- RoPE config: `name_height/width: riflex`, `freq_id: [9]`, `riflex_scale: 0.85`, `linear_factor = resolution/2048`.
- Attention: `attention_decay`, `lambda_max: 1.3`, `lambda_min: 1.1` (paper says 1.0), `gamma: 2.0`, `h_min: 0.44`, `h_max: 6.0`. A 2-step pass records per-head block averages before the real run.
- Qwen-Image code (ultra-qwenimage qwen/base.yaml, inference_rope.py): 4096x4096, `guidance_scale: 4.0` (true CFG), `num_steps` default 50, bf16, `seed: 42`, `lambda_min: 1.0`, `lambda_max: 1.3`, `freq_id: [8, 9]`.

## e. Metrics and implementation

- Metrics: CLIP score [31], FID [15], KID [3] (Sec. 6.1, p.7). "The generated images are compared against 10K high-quality real images from LAION-5B [34]."
- FID implementation: NOT STATED. Resize to 299 and interpolation: NOT STATED. How the 3600/4096 images and the 10K references are resized: NOT STATED. How the "high-quality" references were picked: NOT STATED. KID subset size: NOT STATED.
- Patch or crop FID: NOT USED (none reported).
- CLIP score backbone: NOT STATED. Printed on a 0 to 1 scale (for example 0.3083), unlike the 0 to 100 scale in HRDiT.
- No IS, ImageReward, HPS, PickScore, MUSIQ, MANIQA, CLIP-IQA, ARNIQA, Q-Align, aesthetic score, GLCM, compression ratio, or user study.
- No evaluation code in either branch. Metric code: UNVERIFIED (not public).

## f. Latency hardware

- No latency is reported and no GPU is named. NOT STATED.

## g. Tables (as printed)

### Table 1 (p.8): "Quantitative comparison across three generation scenarios. extra. denote extrapolation. All methods are evaluated at 4096 x 4096 on Qwen-Image (training resolution 1368p) and 3600 x 3600 on Flux (training resolution ranging from 256^2 to 2048^2)."

Checked against the rendered page (papers/2512.04504/table1.png). The left block is Flux and the right block is Qwen-Image, consistent with the text's "113.41 over NTK and 41.31 over PI on Flux".

Direct extra. on Flux (3600 x 3600):

| Method | FID | KID | CLIP Score |
|---|---|---|---|
| PE | 206.2 | 0.1133 | 0.2280 |
| PI [8] | 124.5 | 0.0391 | 0.2789 |
| NTK [4] | 196.6 | 0.1055 | 0.2345 |
| YaRN [28] | 186.2 | 0.0947 | 0.2391 |
| Entropy [19] | 117.3 | 0.0334 | 0.2708 |
| Ours | 83.19 | 0.0114 | 0.3083 |

Direct extra. on Qwen-Image (4096 x 4096):

| Method | FID | KID | CLIP Score |
|---|---|---|---|
| PE | 86.93 | 0.0162 | 0.3257 |
| PI [8] | 94.03 | 0.0217 | 0.3310 |
| NTK [4] | 86.94 | 0.0144 | 0.3246 |
| YaRN [28] | 96.47 | 0.0174 | 0.3304 |
| Entropy [19] | 89.92 | 0.0164 | 0.3399 |
| Ours | 78.15 | 0.0086 | 0.3337 |

Guided resolution extra. on Flux (3600 x 3600):

| Method | FID | KID | CLIP Score |
|---|---|---|---|
| HiFlow [5] | 73.13 | 0.0085 | 0.3375 |
| I-Max [11] | 72.00 | 0.0078 | 0.3392 |
| Ours | 68.98 | 0.0076 | 0.3417 |

Guided view extra. on Flux (3600 x 3600):

| Method | FID | KID | CLIP Score |
|---|---|---|---|
| NTK [4] | 118.6 | 0.0399 | 0.2291 |
| YaRN [28] | 111.7 | 0.0352 | 0.2339 |
| Ours | 104.7 | 0.0276 | 0.2842 |

Text claim (Sec. 6.2, p.7): "FID improvements of 113.41 over NTK and 41.31 over PI on Flux" (196.6 - 83.19 = 113.41; 124.5 - 83.19 = 41.31, consistent with Table 1).

### Table 2 (Supp., PDF p.12): "Quantitative ablation studies." (setting not printed; the baseline row equals the Flux direct-extrapolation PE row, so this is Flux at 3600 x 3600)

| Method | FID | KID | CLIP |
|---|---|---|---|
| baseline | 206.2 | 0.1133 | 0.2280 |
| +RDFC | 107.81 | 0.0257 | 0.2829 |
| full version | 83.19 | 0.0114 | 0.3083 |

## Notes for the SCA comparison

- UltraImage reports no FLUX number at 4096 x 4096 and nothing on Aesthetic-4K. Its FLUX numbers are at 3600 x 3600 against 10K LAION-5B references.
- Paper lambda_min = 1.0 but the shipped Flux config has `lambda_min: 1.1`.
