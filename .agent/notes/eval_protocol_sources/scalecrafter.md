# ScaleCrafter (He et al.), arXiv 2310.07702

Source read: arXiv 2310.07702v1 (stamp "11 Oct 2023", 18 PDF pages; ICLR 2024 format). Page numbers are
PDF pages.

## a. Base model(s)
- SD 1.5, SD 2.1, SD XL 1.0 (text-to-image); LVDM for text-to-video. (Section 4 "Experiment setup", p.7;
  Section 4.3, p.9)

## b. Benchmark prompt set
- "We evaluate performance on the dataset of Laion-5B ... When the inference resolution is 1024^2, we
  sample 30k images with randomly sampled text prompts from the dataset. Due to massive computation, we
  sample 10k images when the inference resolution is higher than 1024^2." (Section 4 "Testing dataset and
  evaluation", p.7)
- So for SDXL (all settings are above 1024^2: 2048^2, 2560^2, 4096x2048, 4096^2): 10k images with random
  LAION-5B prompts. Plain LAION-5B, no aesthetic filter stated.
- Images per prompt: NOT STATED. Seeds: NOT STATED. Prompt list release: NOT STATED.
- Video: 2048 sampled videos, evaluated on WebVid-10M. (Section 4.3, p.9)

## c. Resolutions
- SD 1.5 / SD 2.1 (train 512^2): 1024^2, 1280^2, 2048x1024, 2048^2.
- SD XL (train 1024^2): 2048^2, 2560^2, 4096x2048, 4096^2. (Section 4, p.7)
- Table 1 labels these 4x 1:1, 6.25x 1:1, 8x 1:2, 16x 1:1. Highest = 16x = 4096^2 for SDXL.

## d. Sampler
- 50 inference timesteps in every setting table (App. A, Tables 6-13, p.14-15). Sampler type (DDIM,
  Euler, ...): NOT STATED.
- "All samples are generated using the default classifier-free guidance scale of the corresponding
  pre-trained model (i.e. SD 1.5 and SD 2.1 use 7.5, SD XL 1.0 uses 5.0)." (App. A, p.14)
- 4096^2 SDXL (Table 13, p.15): re-dilated [DB2, UB1] scale [2,2]; dispersed [DB3, MB, UB0] scale [2,2],
  kernel 3x3 to 5x5; progressive on; noise-damped CFG on; 50 steps; tau 35.
- Tiled VAE decoding with synchronized GroupNorm statistics above 2048^2 (App. A, p.15).

## e. Metrics and implementation
- FIDr / KIDr: "between generated images and real images ... We adopt the implementation of clean-fid
  (Parmar et al., 2022) to avoid discrepancies in the image pre-processing steps." (Section 4, p.7)
  - Which real images and how many: NOT STATED (implied LAION-5B). Resize of real images: NOT STATED
    beyond clean-fid. The paper notes "the calculation of FID and KID requires image downsampling to
    229 x 229 [sic] which cannot measure the definition of detailed texture." (Section 4.1, p.9)
- FIDb / KIDb: "between the generated samples under the base training resolution and inference
  resolution" (self-reference to the base model's own native-resolution samples). (Section 4, p.7)
- No crop/patch FID. No IS. No CLIP score.
- User study: only vs SD 2.1 upscaler-4x on the 16x setting, "choose an image with a better texture
  definition", "The responses of 300 questions are collected"; reported as Texture Definition (TD):
  SD+SR 38%, Ours 62%. Number of users: NOT STATED. (Section 4.1, p.9, Table 2)
- Baselines: Direct-Inf and Attn-SF (Jin et al. 2023). MultiDiffusion and SyncDiffusion "are not
  compared". (Section 4.1, p.8)

## f. Main table at the highest resolution
Table 1 (p.7): "Quantitative comparisons among training-free methods." Headers per model:
FIDr, KIDr, FIDb, KIDb. 16x 1:1 rows (SD 1.5 and SD 2.1 at 2048^2; SD XL 1.0 at 4096^2):

| Method | SD1.5 FIDr | KIDr | FIDb | KIDb | SD2.1 FIDr | KIDr | FIDb | KIDb | SDXL FIDr | KIDr | FIDb | KIDb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Direct-Inf | 111.34 | 0.046 | 106.70 | 0.042 | 104.70 | 0.043 | 104.10 | 0.040 | 153.33 | 0.070 | 144.99 | 0.061 |
| Attn-SF | 110.10 | 0.046 | 105.42 | 0.042 | 104.34 | 0.043 | 103.61 | 0.041 | 153.68 | 0.070 | 144.84 | 0.061 |
| Ours | 78.22 | 0.027 | 65.86 | 0.023 | 59.40 | 0.021 | 57.26 | 0.018 | 131.03 | 0.063 | 124.01 | 0.055 |

8x 1:2 rows (SDXL 4096x2048): Direct-Inf SDXL 122.41/0.062/82.51/0.037; Attn-SF 122.21/0.062/82.35/0.037;
Ours 119.58/0.057/50.70/0.019.

Table 2 (p.9, SD 2.1 base): SD+SR FIDr-4x 12.59, KIDr-4x 0.005, TD-16x 38%; Ours 20.88, 0.008, 62%.

## Protocol takeaways for SCA comparison
- Random LAION-5B prompts; 10k images above 1024^2; real-image FID via clean-fid plus a self-reference
  FIDb. No crop FID, no CLIP, no aesthetic or preference metric beyond a 300-response TD study.
