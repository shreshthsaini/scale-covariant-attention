# HRDiT protocol audit

Paper: "HRDiT: Training-Free High-Resolution Image Generation with Off-the-Shelf Diffusion Transformer Models", Xue, Qu, Li, Xu, Yin, See, Rahmani, Liu. arXiv 2608.07003v1 (only version; submitted 7 Aug 2026; arXiv comment "Accepted by ECCV 2026").
Local copy: papers/2608.07003/paper.pdf, paper.txt (36 pages). Page numbers below are PDF pages.
Code: github.com/zylwithxy/HRDiT (default branch `master`, last push 2026-09-01). Files read as text via raw.githubusercontent.com, nothing executed: README.md, inference.py, configs/pipeline.json, hrdit/pipeline.py. Companion repo github.com/zylwithxy/HRDiT-HAP (plan search) tree listed.

## a. Base models and checkpoint

- "we apply it to several popular off-the-shelf DiT models, including Stable Diffusion 3 (SD3) [8] and FLUX [22]" (Sec. 4.1, p.13).
- "we use the FLUX.1 dev model for FLUX [22] and the Stable Diffusion 3 Medium model for Stable Diffusion 3 [8]" (Supp. B, p.20).
- U-Net reference baseline: SDXL + FreeScale (Sec. 4.2, p.14).
- FLUX.1-Krea-dev: not used (no mention anywhere in the text).
- Code README: `huggingface-cli download black-forest-labs/FLUX.1-dev`.

## b. Prompt sets

- Evaluation prompts: "following [6,20], we randomly sample 1,000 captions from the LAION-5B dataset [36]" (Sec. 4.1, p.13 to 14). [6] = DemoFusion, [20] = DiffuseHigh.
- So: LAION-5B, 1,000 prompts, random sample. The prompt list is not released (no prompt file in either repo). Sampling seed or filter: NOT STATED.
- Aesthetic-4K: NOT USED (no occurrence of "Aesthetic" in the paper). HRGen-1K, DrawBench, GenEval, DPG, COCO: NOT USED.
- Images per prompt: NOT STATED (one image per prompt is implied by "1,000 captions" but never written).
- Seeds: NOT STATED in the paper. Released `inference.py` default `--seed 3407` (one fixed generator seed, re-used for every prompt).
- HAP calibration prompts: 30 further LAION-5B prompts, "explicitly excluding any prompts used during evaluation" (Supp. B, p.20). Ablation Tab. 14 swaps these for ChatGPT prompts (p.27).
- User study: 100 LAION-5B prompts, 4K, FLUX, 5 participants, two 1 to 5 ratings (alignment, visual fidelity) (Supp. C, p.24, Tab. 6).

## c. Resolutions

- 2K (2048 x 2048) and 4K (4096 x 4096), "Following the settings in [3, 34]" ([3] HiFlow, [34] FreeScale) (Sec. 4.1, p.13). 8K (8192 x 8192) extra (Sec. 4.3, Tab. 3, p.15). Square only. HxW vs WxH: not relevant (square).

## d. Sampler settings

Paper (Supp. B, p.20 to 21):
- "following prior work [3,7,20], we adopt a cascaded generation strategy for synthesizing images at 1K (1024 x 1024), 2K (2048 x 2048), 4K (4096 x 4096), and 8K (8192x8192) resolutions. For FLUX, the sampling steps at these resolutions are 30, 16, 10, and 8, respectively; for Stable Diffusion 3, the corresponding steps are 28, 8, 6, and 5."
- "Following prior work [3], we further apply a commonly used technique: balancing the entropy shift of self-attention [19]. For a fair comparison, all baseline methods in our experiments adopt the same cascaded generation strategy and incorporate this technique."
- Hyperparameters: Nscope = 50, rc = 0.1 (Sec. 4.1, p.14); SPA bundle N = 3 at 2K, N = 5 at 4K (Supp. B, p.20).
- Guidance scale: NOT STATED in paper. Scheduler / shift: NOT STATED. Precision: NOT STATED.
- "Direct" baseline is defined as "passing them higher-resolution initial noises" (Sec. 4.2, p.14), yet Supp. B says all baselines use the same cascade. Whether Direct also cascades is ambiguous as printed.

Released code (inference.py, configs/pipeline.json; may differ from the paper runs):
- 1024 base stage: `guidance_scale=3.5`, `num_inference_steps=30`, `max_sequence_length=512`.
- High-res stages: `target_heights=[2048, 4096]`, `num_inference_steps_highres=[17, 10]` (paper says 16 for 2K), `guidance_scale_highres=[4.5, 6]`, `ntk_factor=[4.0, 10.0]`, `proportional_attention=True`, `structure_guidance="fft"`, `alphas=[1.0, 0.25]`, `betas=[0.5]`, `filter_ratio=[0.2]`, `upsampling_choice="latent"`, `flow_choice="v_theta"` (HiFlow-derived).
- Each high-res stage decodes the previous latent, bicubic-upsamples the image, re-encodes it, re-noises to the first of the last `num_inference_steps_highres` timesteps of the 30-step base schedule, then denoises (hrdit/pipeline.py lines 914 to 933).
- `pipe.scheduler.use_dynamic_shifting = False` with sigmas `np.linspace(1.0, 1/num_inference_steps, num_inference_steps)`; the static shift then comes from the FLUX.1-dev scheduler config (code-derived, not executed).
- Precision: `--dtype` default `fp16` (README: "33 GiB is the FLUX.1-dev weights in fp16"; peak about 46 GiB at 4096x4096).

## e. Metrics and implementation

- Metrics: FID, KID, FIDp, KIDp, CLIP score, per-image latency (Sec. 4.1, p.14), "following [20]" (DiffuseHigh). User study (Tab. 6).
- FID/KID reference set: "we follow [20] and compute all four metrics with the help of 10,000 real images sampled from LAION-5B [36]" (Supp. B, p.20).
- Patch metrics: "During computing FIDp and KIDp, we further follow [27] and randomly crop ten local patches from each generated image" (Supp. B, p.20). [27] = AccDiffusion.
- Crop size: NOT STATED. Whether crops are resized: NOT STATED. How reference patches are made from the 10,000 LAION images: NOT STATED.
- FID implementation (clean-fid / torchmetrics / pytorch-fid / torch-fidelity): NOT STATED. Resize to 299 and interpolation: NOT STATED. How the 4K images are downsampled for global FID: NOT STATED.
- CLIP score backbone: NOT STATED.
- KID subset size: NOT STATED.
- No IS, ImageReward, HPS, PickScore, MUSIQ, MANIQA, CLIP-IQA, ARNIQA, Q-Align, aesthetic score, GLCM, or compression ratio (none appear in the text).
- No evaluation code released: the HRDiT repo has only inference code; HRDiT-HAP has MoA/, flux_utils/, scripts/flux_pipeline/ (profiling, plan search, generation). Metric code: UNVERIFIED (not public).

## f. Latency hardware

- "We conduct our experiments on NVIDIA H200 GPU" (Sec. 4.1, p.14). Preparation stage timed on an H200: about 45 min + 3 min = about 48 min for FLUX 4K (Supp. C, p.24, Tab. 7).

## g. Tables (as printed; columns FID down, FIDp down, KID down, KIDp down, CLIP Score up, Latency(s) down)

### Table 1 (p.13): "Results on 2K and 4K image generation. Notably, by their design, I-Max is applicable to FLUX but not to Stable Diffusion 3 (SD3), whereas FreCaS is applicable to SD3 but not to FLUX."

Checked against the rendered page (papers/2608.07003/table1.png). Bold in the PDF: every "+ Ours" row on FID, FIDp, KID, KIDp, CLIP and Latency. Bold below marks the FLUX 4K block only.

| Res | Method | FID | FIDp | KID | KIDp | CLIP | Latency(s) |
|---|---|---|---|---|---|---|---|
| 2k | SDXL + FreeScale (U-Net) | 73.23 | 65.04 | 0.0093 | 0.0132 | 30.57 | 26 |
| 2k | SD3 (Direct) | 81.34 | 70.88 | 0.0215 | 0.0194 | 30.23 | 17 |
| 2k | SD3 + DemoFusion | 72.21 | 59.24 | 0.0132 | 0.0125 | 31.40 | 35 |
| 2k | SD3 + DiffuseHigh | 70.57 | 57.39 | 0.0119 | 0.0112 | 31.59 | 20 |
| 2k | SD3 + FreCaS | 73.27 | 63.55 | 0.0139 | 0.0132 | 31.41 | 16 |
| 2k | SD3 + HiFlow | 70.49 | 57.35 | 0.0126 | 0.0110 | 31.62 | 18 |
| 2k | SD3 + Ours | 67.90 | 52.59 | 0.0081 | 0.0073 | 31.81 | 11 |
| 2k | FLUX (Direct) | 73.96 | 69.56 | 0.0173 | 0.0164 | 29.63 | 47 |
| 2k | FLUX + DemoFusion | 68.95 | 60.53 | 0.0127 | 0.0120 | 30.83 | 94 |
| 2k | FLUX + DiffuseHigh | 67.40 | 58.63 | 0.0115 | 0.0108 | 31.01 | 52 |
| 2k | FLUX + I-Max | 68.37 | 56.96 | 0.0121 | 0.0106 | 30.67 | 85 |
| 2k | FLUX + HiFlow | 67.45 | 56.79 | 0.0117 | 0.0105 | 30.73 | 49 |
| 2k | FLUX + Ours | 64.42 | 51.55 | 0.0074 | 0.0067 | 31.27 | 35 |
| 4k | SDXL + FreeScale (U-Net) | 74.85 | 68.89 | 0.0112 | 0.0157 | 30.49 | 248 |
| 4k | SD3 (Direct) | 84.29 | 74.45 | 0.0243 | 0.0231 | 30.18 | 108 |
| 4k | SD3 + DemoFusion | 73.39 | 60.75 | 0.0141 | 0.0133 | 31.16 | 229 |
| 4k | SD3 + DiffuseHigh | 71.88 | 58.76 | 0.0130 | 0.0126 | 31.35 | 115 |
| 4k | SD3 + FreCaS | 79.30 | 72.82 | 0.0226 | 0.0213 | 30.32 | 86 |
| 4k | SD3 + HiFlow | 71.39 | 58.40 | 0.0133 | 0.0118 | 31.59 | 112 |
| 4k | SD3 + Ours | 68.63 | 53.04 | 0.0089 | 0.0079 | 31.79 | 58 |
| **4k** | **FLUX (Direct)** | **75.68** | **72.73** | **0.0192** | **0.0188** | **29.57** | **203** |
| **4k** | **FLUX + DemoFusion** | **71.18** | **62.38** | **0.0150** | **0.0142** | **30.48** | **649** |
| **4k** | **FLUX + DiffuseHigh** | **69.23** | **59.13** | **0.0132** | **0.0125** | **30.97** | **215** |
| **4k** | **FLUX + I-Max** | **70.11** | **59.83** | **0.0142** | **0.0133** | **30.52** | **484** |
| **4k** | **FLUX + HiFlow** | **68.36** | **57.43** | **0.0124** | **0.0114** | **30.62** | **209** |
| **4k** | **FLUX + Ours** | **64.91** | **51.84** | **0.0078** | **0.0074** | **31.17** | **116** |

### Table 3 (p.15): "Additional results on 8K image generation."

| Res | Method | FID | FIDp | KID | KIDp | CLIP | Latency(s) |
|---|---|---|---|---|---|---|---|
| 8k | SDXL + FreeScale | 76.51 | 70.50 | 0.0123 | 0.0176 | 30.44 | 3958 |
| 8k | SD3 (Direct) | 86.75 | 77.68 | 0.0258 | 0.0251 | 30.01 | 822 |
| 8k | SD3 + DemoFusion | 76.07 | 65.22 | 0.0207 | 0.0206 | 30.96 | 2720 |
| 8k | SD3 + DiffuseHigh | 73.30 | 60.97 | 0.0166 | 0.0161 | 31.19 | 889 |
| 8k | SD3 + FreCaS | 81.19 | 76.06 | 0.0241 | 0.0232 | 30.16 | 635 |
| 8k | SD3 + HiFlow | 73.72 | 61.77 | 0.0179 | 0.0173 | 31.44 | 832 |
| 8k | SD3 + Ours | 69.51 | 54.47 | 0.0096 | 0.0083 | 31.71 | 454 |
| 8k | FLUX (Direct) | 78.47 | 74.59 | 0.0212 | 0.0201 | 29.34 | 1708 |
| 8k | FLUX + DemoFusion | 75.72 | 68.71 | 0.0201 | 0.0187 | 30.03 | 5749 |
| 8k | FLUX + DiffuseHigh | 71.02 | 63.87 | 0.0169 | 0.0158 | 30.69 | 1835 |
| 8k | FLUX + I-Max | 72.54 | 65.34 | 0.0196 | 0.0183 | 30.23 | 4488 |
| 8k | FLUX + HiFlow | 70.67 | 60.79 | 0.0142 | 0.0136 | 30.43 | 1721 |
| 8k | FLUX + Ours | 65.73 | 53.55 | 0.0089 | 0.0084 | 31.04 | 827 |

### Table 2 (p.15): "Evaluation on key components in HRDiT." (4K, FLUX)

| Method | FID | FIDp | KID | KIDp | CLIP | Latency(s) |
|---|---|---|---|---|---|---|
| FLUX (Direct) | 75.68 | 72.73 | 0.0192 | 0.0188 | 29.57 | 203 |
| Ours (w/o SPA) | 75.85 | 72.83 | 0.0195 | 0.0192 | 29.55 | 110 |
| Ours (w/o HAP) | 64.82 | 51.81 | 0.0076 | 0.0074 | 31.19 | 212 |
| Ours (full) | 64.91 | 51.84 | 0.0078 | 0.0074 | 31.17 | 116 |

### Supplementary ablations, all 4K FLUX (Supp. C, "on the 4K image generation task using FLUX", p.21)

Table 4 (p.22) SPA variants: w/o whole SPA 75.85 / 72.83 / 0.0195 / 0.0192 / 29.55 / 110; Down-scale 74.87 / 71.68 / 0.0186 / 0.0174 / 29.60 / 111; w/o slide operation 69.43 / 63.08 / 0.0130 / 0.0129 / 30.52 / 111; Ours 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116.

Table 5 (p.23) HAP variants: w/o whole HAP 64.82 / 51.81 / 0.0076 / 0.0074 / 31.19 / 212; Fixed local window 69.47 / 60.77 / 0.0116 / 0.0101 / 30.83 / 115; Random window 73.45 / 70.90 / 0.0168 / 0.0162 / 29.79 / 117; Intermediate-feature-losses-guided window 68.02 / 58.21 / 0.0104 / 0.0093 / 30.92 / 116; Ours 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116.

Table 6 (p.24) user study (Prompt-Image Alignment, Visual fidelity): FLUX (Direct) 2.2, 1.8; + DemoFusion 3.2, 2.4; + DiffuseHigh 3.6, 3.7; + I-Max 3.3, 2.6; + HiFlow 3.8, 3.7; + Ours 4.7, 4.8.

Table 7 (p.24) preparation time: step 1 about 45 min; step 2 about 3 min; whole about 48 min.

Table 8 (p.25) Nscope = 10: 66.03 / 55.72 / 0.0093 / 0.0084 / 31.01 / 115; 30: 65.17 / 53.29 / 0.0082 / 0.0079 / 31.13 / 115; 50: 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116; 70: 64.89 / 51.84 / 0.0078 / 0.0074 / 31.18 / 116; 90: 64.88 / 51.83 / 0.0078 / 0.0074 / 31.18 / 116.

Table 9 (p.25) N = 3: 65.87 / 52.88 / 0.0090 / 0.0082 / 31.07 / 114; N = 4: 65.09 / 52.26 / 0.0083 / 0.0076 / 31.11 / 115; N = 5: 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116; N = 6: 64.92 / 51.82 / 0.0081 / 0.0073 / 31.15 / 117; N = 7: 65.04 / 52.29 / 0.0082 / 0.0075 / 31.09 / 117.

Table 10 (p.26) rc = 0.01: 94.51 / 91.34 / 0.0383 / 0.0379 / 27.39 / 107; 0.05: 75.66 / 67.49 / 0.0197 / 0.0181 / 30.07 / 111; 0.1: 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116; 0.3: 64.90 / 51.84 / 0.0078 / 0.0074 / 31.18 / 134; 0.5: 64.88 / 51.83 / 0.0077 / 0.0074 / 31.18 / 155.

Table 11 (p.26, extra column Preparation Time(min)): 10 prompts 66.13 / 54.68 / 0.0092 / 0.0081 / 31.02 / 116 / 18; 20: 65.02 / 52.15 / 0.0081 / 0.0077 / 31.14 / 116 / 33; 30: 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116 / 48; 40: 64.91 / 51.80 / 0.0077 / 0.0072 / 31.17 / 116 / 63; 50: 64.90 / 51.79 / 0.0077 / 0.0071 / 31.17 / 116 / 78.

Table 12 (p.26): HiFlow 68.36 / 57.43 / 0.0124 / 0.0114 / 30.62 / 209; Preparation once for all testing resolutions 66.09 / 53.11 / 0.0087 / 0.0079 / 31.04 / 116; Preparation once per testing resolution 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116.

Table 13 (p.27): HiFlow 68.36 / 57.43 / 0.0124 / 0.0114 / 30.62 / 209; Single plan using all denoising steps 64.91 / 51.84 / 0.0077 / 0.0073 / 31.17 / 116; Separate plan per denoising step 64.90 / 51.84 / 0.0077 / 0.0073 / 31.18 / 117; Single plan using only the final denoising step 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116.

Table 14 (p.27): ChatGPT prompts (first strategy) 64.92 / 51.84 / 0.0079 / 0.0075 / 31.17 / 116; ChatGPT prompts (second strategy) 64.93 / 51.82 / 0.0080 / 0.0074 / 31.17 / 116; Prompts from LAION-5B 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116.

Table 15 (p.28) vs SUPIR: 2k SD3 + SUPIR 68.49 / 53.64 / 0.0088 / 0.0077 / 31.75 / 59; 2k SD3 + Ours 67.90 / 52.59 / 0.0081 / 0.0073 / 31.81 / 11; 2k FLUX + SUPIR 65.25 / 53.67 / 0.0083 / 0.0072 / 31.22 / 59; 2k FLUX + Ours 64.42 / 51.55 / 0.0074 / 0.0067 / 31.27 / 35; 4k SD3 + SUPIR 69.33 / 54.57 / 0.0097 / 0.0084 / 31.70 / 305; 4k SD3 + Ours 68.63 / 53.04 / 0.0089 / 0.0079 / 31.79 / 58; **4k FLUX + SUPIR 65.89 / 54.96 / 0.0092 / 0.0083 / 31.04 / 305**; **4k FLUX + Ours 64.91 / 51.84 / 0.0078 / 0.0074 / 31.17 / 116**.

## Notes for the SCA comparison

- HRDiT is a cascaded (1K to 2K to 4K re-noise) method; its 4K numbers are not single-pass 4096x4096 generation.
- Its FID is against 10,000 LAION-5B real images (not Aesthetic-4K), so its absolute FID/FIDp values cannot be compared with Aesthetic-4K numbers.
- Code discrepancy: 2K high-res stage uses 17 steps in configs/pipeline.json vs 16 in the paper.
