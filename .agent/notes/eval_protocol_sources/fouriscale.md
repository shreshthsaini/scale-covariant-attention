# FouriScale (Huang et al.), arXiv 2403.12963

Source read: arXiv 2403.12963v1 (stamp "19 Mar 2024", 26 PDF pages, ECCV-style; main paper p.1-15,
appendix p.16-22, references p.23-26). Page numbers are PDF pages.

## a. Base model(s)
- SD 1.5, SD 2.1, SDXL 1.0. "We default use FreeU [39] in all experimental settings." (Section 4
  "Experimental setup", p.12)

## b. Benchmark prompt set
- Copied from ScaleCrafter ([15] = He et al.): "Following [15], we assess performance using the Laion-5B
  dataset ... For tests conducted at an inference resolution of 1024x1024, we select a subset of 30,000
  images, each paired with randomly chosen text prompts from the dataset. Given the substantial
  computational demands, our sample size is reduced to 10,000 images for tests at inference resolutions
  exceeding 1024x1024." (Section 4 "Testing dataset and evaluation metrics", p.12)
- So all SDXL settings use 10,000 images with random LAION-5B prompts. No aesthetic filtering stated.
- Images per prompt: NOT STATED. Seeds: NOT STATED. Prompt list release: NOT STATED.

## c. Resolutions
- SD 1.5 / SD 2.1 (train 512^2): 1024^2, 1280^2, 2048x1024, 2048^2.
- SDXL (train ~1024^2): 2048^2, 2560^2, 4096x2048, 4096^2. (Section 4, p.12)
- Table labels 4x 1:1, 6.25x 1:1, 8x 1:2, 16x 1:1. Highest = 16x = 4096^2 for SDXL.
- Qualitative 2048x4096 and 4096x4096 with SD 2.1 and AnimeArtXL in the supplement (Fig. 4, p.22).

## d. Sampler
- 50 inference timesteps for all models (Supp. Table 1 "Experiment settings", p.20). Sampler type:
  NOT STATED. CFG scale: NOT STATED. [Sinit, Sstop] = [20, 35] for SDXL; for SD 1.5/2.1, [10, 30] at
  4x and 6.25x, [20, 35] at 8x and 16x. FouriScale blocks SDXL: [DB2, MB, UB0, UB1].
- Timing: at SDXL 16x, ScaleCrafter 577 s per image vs FouriScale 540 s per image on one A100.
  (Section 4.1, p.13)

## e. Metrics and implementation
- FIDr / KIDr "between generated images and real images" and FIDb / KIDb "between the generated images
  at the base training resolution and the inference resolution", both following [15] (ScaleCrafter).
  (Section 4, p.12)
- Which real images and how many: NOT STATED. Resize protocol: NOT STATED. FID library: NOT STATED
  (ScaleCrafter used clean-fid; FouriScale does not say it does).
- KID values "are scaled by 10^2" in Table 1 (caption, p.13).
- No crop/patch FID, no IS, no CLIP score, no user study.
- Baselines: Vanilla, Attn-Entro [25], ScaleCrafter [15]. Supplement adds SD 2.1 + SD Upscaler-4x on
  10,000 SD 2.1 images (Supp. C.1, Table 2, p.20-21) and ElasticDiffusion at SDXL 2048^2 (Supp. C.2,
  Table 3, p.21).

## f. Main table at the highest resolution
Table 1 (p.13): "Quantitative comparisons among training-free methods ... The values of KIDr and KIDb
are scaled by 10^2." Headers per model: FIDr down, KIDr down, FIDb down, KIDb down. 16x 1:1 rows
(SD 1.5 and SD 2.1 at 2048^2; SDXL 1.0 at 4096^2):

| Method | SD1.5 FIDr | KIDr | FIDb | KIDb | SD2.1 FIDr | KIDr | FIDb | KIDb | SDXL FIDr | KIDr | FIDb | KIDb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 67.90 | 2.37 | 66.49 | 2.18 | 84.01 | 3.28 | 82.25 | 3.05 | 116.40 | 5.45 | 109.19 | 4.84 |
| Attn-Entro | 67.45 | 2.35 | 66.16 | 2.17 | 83.68 | 3.30 | 81.98 | 3.04 | 113.25 | 5.44 | 106.34 | 4.81 |
| ScaleCrafter | 32.00 | 1.01 | 27.08 | 0.71 | 40.91 | 1.32 | 33.23 | 0.90 | 84.58 | 3.53 | 85.91 | 3.39 |
| Ours | 30.84 | 0.95 | 23.29 | 0.57 | 39.49 | 1.27 | 28.14 | 0.73 | 56.66 | 2.18 | 49.59 | 1.63 |

8x 1:2 SDXL (4096x2048): Vanilla 90.23/4.20/79.32/3.42; Attn-Entro 89.87/4.15/79.00/3.40; ScaleCrafter
106.57/5.15/108.67/5.23; Ours 71.77/2.79/70.70/2.65.

Note: FouriScale's re-run of ScaleCrafter at SDXL 16x (FIDr 84.58) differs from ScaleCrafter's own
Table 1 (131.03); the two papers' "Vanilla" SDXL 16x numbers also differ (116.40 vs ScaleCrafter's
Direct-Inf 153.33). Same nominal protocol, different numbers; cause NOT STATED.

## Protocol takeaways for SCA comparison
- Protocol inherited verbatim from ScaleCrafter: random LAION-5B prompts, 10k images above 1024^2,
  real-image FIDr/KIDr plus self-reference FIDb/KIDb. No crop FID, no preference metrics.
