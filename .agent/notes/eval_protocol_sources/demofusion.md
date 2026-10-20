# DemoFusion (Du et al.), arXiv 2311.16973, and the crop-FID lineage

Source read: arXiv 2311.16973v2 (stamp "15 Dec 2023", 18 PDF pages; CVPR 2024). Cross-checked against
v1 (stamp "24 Nov 2023"): the metrics paragraph and Table 1 are identical in v1 and v2.
Lineage root also read: Chai et al., "Any-resolution training for high-resolution image synthesis",
arXiv 2204.07156v2 (stamp "5 Aug 2022", ECCV 2022). Page numbers are PDF pages.

## a. Base model
- SDXL. "all the results in this paper are obtained based on SDXL with a DDIM scheduler of 50 steps."
  (App. B, p.11)

## b. Benchmark prompt set
- "We evaluate on the LAION-5B dataset [31] with 1K randomly sampled captions." (Section 4.1
  "Quantitative Results", p.7). Plain LAION-5B; no aesthetic filter stated.
- Seed for sampling captions: NOT STATED. Images per prompt: NOT STATED. Caption list: NOT RELEASED in
  the paper (App. F lists only the prompts of qualitative figures, "taken from the internet", p.16-17).

## c. Resolutions
- 2048x2048, 2048x4096, 4096x4096 (Table 1, p.7). Qualitative 3072^2 ablations and 4096^2 galleries.

## d. Sampler
- DDIM, 50 steps; guidance 7.5 for all denoising paths; MultiDiffusion crop h = w = 128 latent for
  SDXL, stride h/2, w/2, crop positions randomly perturbed. (App. B, p.11)
- Time measured on one RTX 3090. (Section 4.1, p.7)

## e. Metrics and implementation (the crop-FID protocol)
Exact text (Section 4.1, p.7): "we adopt 3 widely-used metrics: FID [10], IS [30], and CLIP Score [27].
Considering that FID and IS require resizing images to 299^2, which is not very suitable for
high-resolution image assessment, inspired by [4], we additionally crop local patches of 1x resolution
and then resize them to calculate these metrics, termed FIDcrop and IScrop. The CLIP Score assesses the
entire image's semantics; thus, we do not consider evaluating local patches here. ... Note that the
results of FID and IS are related to the number of samples; therefore, the scores of FIDcrop and IScrop
might be better than FID and IS due to more samples."
- Crop size: "local patches of 1x resolution" = 1024x1024 for SDXL. Crops are then RESIZED (to the
  Inception input; 299 implied by the preceding sentence).
- Crops per image: NOT STATED (only "more samples" than the full-image metric, so more than one).
- Random vs fixed crop positions: NOT STATED.
- Reference set for FID and FIDcrop (real LAION images or SDXL 1024 outputs; how reference crops are
  made): NOT STATED. The paper never says "real images".
- KID: NOT USED (no KID or KIDcrop in DemoFusion). KIDc first appears in this lineage in FreeScale.
- FID implementation library: NOT STATED. CLIP backbone: NOT STATED (cites Radford et al. 2021 only).
- User study: none.
- Baselines: SDXL direct inference, MultiDiffusion, SDXL+BSRGAN, ScaleCrafter. (Section 4.1, p.5)

## f. Main table at the highest resolution (4096x4096)
Table 1 (p.7) "Quantitative comparison results." 4096 x 4096 block, headers
FID down, IS up, FIDcrop down, IScrop up, CLIP up, Time:

| Method | FID | IS | FIDcrop | IScrop | CLIP | Time |
|---|---|---|---|---|---|---|
| SDXL Direct Inference | 105.65 | 14.01 | 98.59 | 19.47 | 25.64 | 8 min |
| MultiDiffusion | 97.98 | 13.84 | 79.45 | 19.73 | 28.62 | 15 min |
| SDXL + BSRGAN | 66.44 | 16.21 | 77.20 | 22.42 | 29.63 | 1 min |
| ScaleCrafter | 87.50 | 15.20 | 84.36 | 20.32 | 29.04 | 19 min |
| DemoFusion (Ours) | 74.11 | 16.11 | 70.34 | 24.28 | 29.57 | 25 min |

The ablation Table 2 (p.11) ends with the same full-method row (74.11 / 16.11 / 70.34 / 24.28 / 29.57),
so the ablation is at 4096^2 (resolution not printed in its caption).

## Lineage root: Chai et al. 2022 patch-FID (cited as [4] by DemoFusion and [8] by FreeScale)
- "Our patch-FID randomly resizes and crops patches from the HR dataset, and computes FID on real and
  generated patches, sampled at corresponding scales and locations. We use 50k patches, matching
  standard FID. By avoiding downsampling, our patch-FID is more sensitive to blurriness or artifacts."
  (2204.07156v2, Section 4.1 "Patch-FID metric", p.9)
- Appendix 7.2 (p.23): patches of size p sampled from REAL images at random scale s and location, with
  the generator rendering the matching patch; "50,000 is typically used"; when p > 299 a random
  299-pixel crop is taken from each patch, so the generated content is never downsampled.
- So the origin of crop FID is: real-image references, 50k random patches, no downsampling, matched
  scale and location. DemoFusion changed it to 1x-resolution (1024) crops that ARE resized, with
  unstated crop count and unstated references; FreeScale then renamed it FIDc/KIDc, described it as
  "cropped local patches ... without resizing" (crop size and count not stated), cited Chai et al.
  rather than DemoFusion, and used the base model's own 1024 outputs as the reference set.
