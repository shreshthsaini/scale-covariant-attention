# ScaleDiff protocol audit

Paper: "ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion", Koh, Cha, Oh, Lee, Kim (Hanyang University). arXiv 2510.25818v1 (only version; submitted 29 Oct 2025; arXiv comment "NeurIPS 2025. Code: ...").
Local copy: papers/2510.25818/paper.pdf, paper.txt (19 pages). Tables 2 and 3 were also rendered to PNG (papers/2510.25818/table2.png, table3.png) and checked by eye.
Code: github.com/KSH00906/ScaleDiff (branch `main`, last push 2025-10-31). Files read as text: README.md, FLUX/run_scalediff_flux.py, FLUX/run_scalediff_flux_schnell.py, FLUX/pipeline_scalediff_flux.py. The repo has no evaluation code and no prompt list. Nothing executed.

## a. Base models and checkpoint

- "We evaluate our proposed method, ScaleDiff, on both FLUX [21] and SDXL [30]" (Sec. 4.1, p.7). Reference [21]: "B. F. Labs. Flux.1: An advanced generative ai model. https://flux1.io/, 2024."
- The paper never says dev vs schnell. The sampler settings it gives (30 steps, guidance 3.5) match FLUX.1-dev, and the code README section is titled "FLUX.1-dev" with `ckpt = "black-forest-labs/FLUX.1-dev"`. The repo also ships `run_scalediff_flux_schnell.py` (FLUX.1-schnell, guidance 0.0, 4 steps, restart_ratio 0.5), which the paper does not report.
- FLUX.1-Krea-dev: not used.
- SDXL: "stabilityai/stable-diffusion-xl-base-1.0" (README).

## b. Prompt sets

- "we randomly sample 1,000 image-text pairs from the LAION-5B [38] dataset and generate one image per prompt using each method" (Sec. 4.1 Evaluation, p.8).
- So: LAION-5B, 1,000 random image-text pairs, 1 image per prompt. Sampling seed or filter: NOT STATED. Prompt list not released.
- Aesthetic-4K: NOT USED. HRGen-1K, DrawBench, GenEval, DPG, COCO: NOT USED.
- Seeds: NOT STATED in the paper. Code examples use `manual_seed(77)`.
- Prompt suffix: the FLUX example script sends `prompt + ", highly detailed, 4k resolution, best quality"`. Whether the evaluation used this suffix: NOT STATED.
- No user study.

## c. Resolutions

- 2048^2 and 4096^2 (Table 2, p.8), produced "within an iterative 1024^2 -> 2048^2 -> 4096^2 generation pipeline" (Sec. 4.1, p.7). Square only in the tables. Aspect-ratio handling: "we ensure that the longer side of the initial image matches the model's trained resolution" (Supp. A, p.14). Panorama 1024 x 4096 is qualitative only (Fig. 10).

## d. Sampler settings

- FLUX: "we use a noise timestep tau = 600, and a structure guidance strength of gamma_t = t. This setup uses 30 denoising steps with a guidance scale of 3.5" (Sec. 4.1, p.7).
- SDXL: tau = 400, gamma_t = 1 - alpha_bar_t, 50 steps, CFG 7.5 (Sec. 4.1, p.7).
- Frequency decomposition downsampling ratio "4 for FLUX and 8 for SDXL" (Supp. A, p.14).
- Query Window Random Shifting: "Note that this technique was not used during the evaluation in this paper" (Supp. B.1, p.14). The shipped FLUX example sets `query_random_jitter = True`.
- MultiDiffusion timing: 50% overlap, batch 16 on SDXL and 1 on FLUX (Supp. A, p.14).
- FLUX baselines are "the base model (natively supports resolutions up to 2048^2) and SR methods" (BSRGAN, OSEDiff) (p.8). The U-Net training-free baselines run on SDXL only. Baseline sampler settings: NOT STATED.
- Precision: NOT STATED in the paper. Code: `torch_dtype=torch.bfloat16` for FLUX, float16 for SDXL; `pipe.vae.enable_tiling()`; `max_sequence_length=256`.
- Code schedule (read, not executed): one 30-step sigma schedule with dynamic-shift `mu` computed at `image_seq_len = 4096` (the 1024^2 token count) is reused for every stage. Each upsampling stage bicubic-upsamples the latent and the decoded RGB image, mixes them (LFM, `scale_factor=0.25`), re-noises to `timesteps[restart_step]` with `restart_step = int(30 * (1 - 0.6)) = 12`, and denoises the remaining 18 steps with NPA. `upsample_stage = 2` gives 1024 -> 2048 -> 4096 (pipeline_scalediff_flux.py lines 252 to 340).

## e. Metrics and implementation

- Metrics: FID [13], KID [4], IS [37] "between generated images and real images"; patch FIDp, KIDp, ISp; CLIP Score [32]; time in seconds (Sec. 4.1, p.8; Table 2).
- Resize: "these metrics typically require resizing images to 299^2 pixels, thereby limiting the evaluation of fine-grained details" (p.8). Resize interpolation: NOT STATED.
- FID reference set: "real images". Which and how many are NOT STATED explicitly; the sampling of 1,000 LAION-5B image-text pairs implies the 1,000 paired real images, but the paper does not say so.
- Patch metrics: "we extract multiple patches from each image and calculate patch-level FIDp, KIDp, and ISp following [8]" (p.8). [8] = DemoFusion. Crop size, crops per image, crop positions, resizing, reference crops: NOT STATED in ScaleDiff (inherited from DemoFusion by citation only).
- FID implementation library: NOT STATED. KID subset size: NOT STATED. CLIP backbone: NOT STATED (0 to 100 scale, for example 31.14).
- No ImageReward, HPS, PickScore, MUSIQ, MANIQA, CLIP-IQA, ARNIQA, Q-Align, aesthetic score, GLCM, compression ratio, or user study.
- Metric code: UNVERIFIED (not in the repository).

## f. Latency hardware

- "All experiments are conducted on a single NVIDIA A6000 GPU" (Sec. 4.1, p.7). Table 2 times in seconds.

## g. Tables (as printed)

### Table 2 (p.8): "Quantitative comparison results. The best results are shown in bold, and the second best results are underlined. All time measurements are expressed in seconds."

Columns: Model | Resolution | Method | FID down | KID down | IS up | FIDp down | KIDp down | ISp up | CLIP up | Time down

SDXL, 2048^2:
| Method | FID | KID | IS | FIDp | KIDp | ISp | CLIP | Time |
|---|---|---|---|---|---|---|---|---|
| SDXL Direct [30] | 88.56 | 0.0124 | 13.25 | 58.73 | 0.0137 | 20.79 | 31.57 | 47 |
| SDXL + BSRGAN [48] | 64.60 | 0.0041 | 18.40 | 41.40 | 0.0092 | 23.19 | 33.03 | 13 |
| SDXL + OSEDiff [46] | 64.79 | 0.0046 | 18.89 | 41.76 | 0.0094 | 23.58 | 32.79 | 29 |
| UltraPixel [33] | 64.61 | 0.0056 | 18.58 | 42.44 | 0.0093 | 25.15 | 32.61 | 71 |
| ScaleCrafter [12] | 68.68 | 0.0033 | 16.56 | 43.46 | 0.0064 | 23.52 | 32.07 | 64 |
| HiDiffusion [49] | 69.52 | 0.0040 | 18.22 | 42.92 | 0.0067 | 24.01 | 31.50 | 33 |
| DiffuseHigh [20] | 63.27 | 0.0033 | 19.10 | 38.15 | 0.0062 | 24.95 | 32.77 | 45 |
| FreeScale [31] | 63.50 | 0.0031 | 19.06 | 38.27 | 0.0062 | 24.25 | 32.62 | 69 |
| AccDiffusion v2 [24] | 64.86 | 0.0039 | 18.37 | 38.24 | 0.0068 | 25.66 | 32.62 | 199 |
| Demofusion [8] | 63.36 | 0.0032 | 19.15 | 35.98 | 0.0050 | 26.42 | 32.72 | 125 |
| ScaleDiff (Ours) | 62.98 | 0.0032 | 19.54 | 38.03 | 0.0067 | 25.70 | 33.11 | 31 |

SDXL, 4096^2:
| Method | FID | KID | IS | FIDp | KIDp | ISp | CLIP | Time |
|---|---|---|---|---|---|---|---|---|
| SDXL Direct [30] | 182.05 | 0.0717 | 7.99 | 80.80 | 0.0250 | 17.68 | 27.82 | 328 |
| SDXL + BSRGAN [48] | 64.88 | 0.0044 | 18.16 | 48.97 | 0.0160 | 17.04 | 33.02 | 14 |
| SDXL + OSEDiff [46] | 65.35 | 0.0045 | 18.69 | 45.67 | 0.0118 | 17.61 | 32.88 | 122 |
| UltraPixel [33] | 65.39 | 0.0055 | 19.08 | 47.09 | 0.0112 | 20.64 | 32.33 | 386 |
| ScaleCrafter [12] | 86.66 | 0.0110 | 15.14 | 79.39 | 0.0217 | 14.47 | 30.25 | 932 |
| HiDiffusion [49] | 105.37 | 0.0216 | 13.87 | 112.30 | 0.0494 | 12.22 | 27.21 | 124 |
| DiffuseHigh [20] | 63.91 | 0.0034 | 18.99 | 42.30 | 0.0079 | 19.54 | 32.68 | 325 |
| FreeScale [31] | 64.33 | 0.0036 | 19.18 | 39.56 | 0.0079 | 18.91 | 32.56 | 517 |
| AccDiffusion v2 [24] | 64.64 | 0.0037 | 18.56 | 40.92 | 0.0083 | 18.42 | 32.34 | 1599 |
| Demofusion [8] | 65.06 | 0.0041 | 19.13 | 41.29 | 0.0079 | 19.59 | 32.61 | 1005 |
| ScaleDiff (Ours) | 61.87 | 0.0025 | 19.56 | 38.89 | 0.0080 | 20.41 | 33.04 | 113 |

FLUX, 2048^2:
| Method | FID | KID | IS | FIDp | KIDp | ISp | CLIP | Time |
|---|---|---|---|---|---|---|---|---|
| FLUX Direct [21] | 68.78 | 0.0069 | 18.57 | 42.84 | 0.0086 | 22.46 | 30.79 | 150 |
| FLUX + BSRGAN [48] | 64.65 | 0.0052 | 19.07 | 42.01 | 0.0081 | 22.98 | 31.21 | 33 |
| FLUX + OSEDiff [46] | 65.10 | 0.0056 | 18.46 | 41.88 | 0.0078 | 23.25 | 31.03 | 46 |
| ScaleDiff (Ours) | 64.31 | 0.0047 | 18.51 | 40.03 | 0.0073 | 23.38 | 31.22 | 103 |

FLUX, 4096^2:
| Method | FID | KID | IS | FIDp | KIDp | ISp | CLIP | Time |
|---|---|---|---|---|---|---|---|---|
| **FLUX Direct [21]** | **459.07** | **0.2775** | **1.61** | **367.47** | **0.2642** | **1.22** | **18.03** | **1251** |
| **FLUX + BSRGAN [48]** | **64.76** | **0.0051** | **18.84** | **49.30** | **0.0125** | **16.92** | **31.19** | **34** |
| **FLUX + OSEDiff [46]** | **64.22** | **0.0052** | **19.16** | **48.37** | **0.0112** | **16.99** | **31.13** | **136** |
| **ScaleDiff (Ours)** | **64.06** | **0.0044** | **18.36** | **44.29** | **0.0098** | **17.41** | **31.14** | **407** |

(Bold in this file marks the FLUX 4096^2 block, not the paper's best-result bolding. In the PDF, ScaleDiff's FLUX 4096^2 FID 64.06, KID 0.0044, FIDp 44.29, KIDp 0.0098, ISp 17.41 are bold; BSRGAN's CLIP 31.19 and Time 34 are bold.)

### Table 3 (p.9): "Quantitative results of ablation study." Its rows match the SDXL 4096^2 entries of Table 2 (NPA + LFM + SG = 61.87 / 0.0025 / 19.56 / 38.89 / 0.0080 / 20.41 / 33.04 / 113), so this is SDXL at 4096^2. Check-mark columns confirmed from the rendered page.

| Attention | LFM | SG | FID | KID | IS | FIDp | KIDp | ISp | CLIP | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | v | v | 61.91 | 0.0028 | 19.47 | 39.94 | 0.0082 | 20.09 | 33.01 | 185 |
| MultiDiffusion | v | v | 61.71 | 0.0021 | 19.71 | 38.08 | 0.0069 | 20.85 | 33.04 | 239 |
| NPA | v | v | 61.87 | 0.0025 | 19.56 | 38.89 | 0.0080 | 20.41 | 33.04 | 113 |
| NPA | | | 64.17 | 0.0036 | 19.49 | 41.55 | 0.0092 | 19.41 | 33.02 | 113 |
| NPA | v | | 62.34 | 0.0028 | 19.19 | 39.49 | 0.0085 | 20.16 | 33.01 | 113 |
| NPA | | v | 64.12 | 0.0035 | 18.86 | 41.50 | 0.0091 | 19.71 | 33.04 | 113 |
| NPA | v | v | 61.87 | 0.0025 | 19.56 | 38.89 | 0.0080 | 20.41 | 33.04 | 113 |

The ablation text quotes FLUX times not in any table: MultiDiffusion "1148s on FLUX", NPA "407s on FLUX, 2.8x speedup" (Sec. 4.4, p.9).

### Table 4 (p.10): "Ablating noise timestep tau." Resolution NOT STATED (the FLUX tau = 600 row 64.45 does not equal either FLUX row of Table 2).

| Model | tau | FID | KID | IS | FIDp | KIDp | ISp | CLIP |
|---|---|---|---|---|---|---|---|---|
| SDXL | 700 | 63.61 | 0.0036 | 18.93 | 37.65 | 0.0064 | 25.02 | 33.01 |
| SDXL | 600 | 63.60 | 0.0034 | 19.20 | 37.94 | 0.0067 | 25.15 | 33.06 |
| SDXL | 500 | 63.30 | 0.0033 | 19.43 | 38.77 | 0.0073 | 24.42 | 33.09 |
| SDXL | 400 | 63.12 | 0.0032 | 19.36 | 38.33 | 0.0070 | 24.56 | 33.07 |
| SDXL | 300 | 63.22 | 0.0032 | 19.56 | 39.38 | 0.0077 | 23.85 | 33.07 |
| FLUX | 700 | 64.44 | 0.0049 | 18.22 | 40.37 | 0.0073 | 23.62 | 31.27 |
| FLUX | 600 | 64.45 | 0.0049 | 18.52 | 40.76 | 0.0076 | 23.16 | 31.29 |
| FLUX | 500 | 64.13 | 0.0049 | 18.36 | 41.72 | 0.0081 | 22.78 | 31.29 |
| FLUX | 400 | 63.59 | 0.0047 | 18.45 | 41.99 | 0.0083 | 22.90 | 31.25 |
| FLUX | 300 | 63.67 | 0.0048 | 18.30 | 42.07 | 0.0084 | 23.24 | 31.33 |

## Notes for the SCA comparison

- ScaleDiff's FLUX 4096^2 numbers come from a 3-stage cascade (1024 -> 2048 -> 4096, re-noise to tau = 600), scored against LAION-5B real images on 1,000 LAION prompts. Not Aesthetic-4K.
- Its "FLUX Direct" 4096^2 row (FID 459.07) is single-pass native 4096^2 FLUX, the closest analogue to an un-modified single-pass baseline.
