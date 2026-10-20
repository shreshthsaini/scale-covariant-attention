# FreeScale (Qiu et al.), arXiv 2412.09626

Source read: arXiv 2412.09626v2 (stamp "11 Jul 2025", 16 PDF pages), text extracted from the PDF.
Cross-checked against v1 (stamp "12 Dec 2024", 13 pages). Page numbers below are PDF pages of v2.

## a. Base model(s)
- Image: SDXL. Video: VideoCrafter2. (v2, Section 4 "Experimental Settings", p.5)
- FLUX appears only qualitatively: FLUX is used to produce the intermediate 1x image of one 8k gallery
  example, and Fig. 13 shows a "structure gap" (DiT models blur). No FLUX quantitative result.
  (v2, Supp. C.3 and D, p.14 and p.16)

## b. Benchmark prompt set
- "We evaluate image generation on the LAION-5B dataset with 1024 randomly sampled captions.
  Specifically, to better align with human preference, we randomly selected prompts from the
  LAION-Aesthetics-V2-6.5plus dataset". (v2, Supp. A "Datasets", p.12; in v1 it is in the main
  Section 4 Experimental Settings, p.5)
- Count: 1024 captions. Subset of LAION-Aesthetics-V2-6.5plus, random. Seed for the subset: NOT STATED.
  The caption list is not released in the paper. Repo check (github.com/ali-vilab/FreeScale, 2026-10-20):
  the visible tree has no evaluation script and no caption list; prompts/ holds demo.txt and imgen.txt
  with one prompt each. The FIDc/KIDc crop protocol cannot be recovered from the repo either.
- Images per prompt: NOT STATED (implicitly one; 1024 images per method is the natural reading but the
  paper does not say it).
- Video: 512 random WebVid-10M captions (same paragraph).

## c. Resolutions
- Quantitative: 2048x2048 and 4096x4096 (Table 1, Table 3), plus 2048x4096 (Table 8, p.14).
  "Considering the computing resources that can be afforded, we evaluate the image generation at
  resolutions of 2048^2 and 4096^2". (Section 4, p.5)
- 8192x8192 is qualitative only (Fig. 1, Fig. 8, Fig. 12 gallery). No 8k metric table.

## d. Sampler
- DDIM, 50 steps, eta 0; SDXL base 1024x1024; CFG 7.5; K = 700; alpha = 2 (1 for 8k).
  (v2, Supp. A "Implementation Details", p.12)
- Single A800 GPU. (Section 4, p.5)

## e. Metrics and implementation
- Reference images are NOT real images: "we calculate all metrics between the originally generated
  low-resolution images/videos and the corresponding high-resolution outputs." (Section 4 "Evaluation
  Metrics", p.5). So FID/KID reference = SDXL's own 1024x1024 generations for the same prompts.
- FID, KID, IS. "FID and KID need to resize the images to 299 before the comparison and this operation
  may cause quality loss for high-resolution images." (p.5). IS added in v2 only (v1 Table 1 has no IS).
- Crop metrics: "Inspired by previous work [8], we also use cropped local patches to calculate these
  metrics without resizing, termed FIDc and KIDc." [8] = Chai et al., "Any-resolution training for
  high-resolution image synthesis", ECCV 2022. (p.5)
  - Crop size: NOT STATED. Crops per image: NOT STATED. Random vs fixed: NOT STATED.
  - How reference crops are made from the 1024 references: NOT STATED.
- FID/KID implementation library (clean-fid, torchmetrics, etc.): NOT STATED.
- CLIP score: NOT REPORTED.
- Inference time (min) reported per method.
- User study: 23 users; images of all methods shown in random order; pick the best for image-text
  alignment, image quality, visual structure. Number of prompts/questions: NOT STATED. Resolution of the
  user-study images: NOT STATED. (Supp. B.2, p.12-13, Table 5)
  Results (Table 5): SDXL-DI 0.87/0.00/0.00%; ScaleCrafter 7.83/5.22/7.83%; DemoFusion 17.39/14.35/18.26%;
  FouriScale 2.17/2.61/1.74%; Ours 71.74/77.83/72.17% (Text Alignment / Image Quality / Visual Structure).
- Baselines: SDXL direct inference (SDXL-DI), ScaleCrafter, DemoFusion, FouriScale; "FreeU is used if
  compatible." (Section 4.1, p.5)

## f. Main table at the highest resolution (4096x4096)
Table 1 (v2, p.5): "Image quantitative comparisons with other baselines." 4096^2 block, headers
FID down, KID down, FIDc down, KIDc down, IS up, Time (min) down:

| Method | FID | KID | FIDc | KIDc | IS | Time (min) |
|---|---|---|---|---|---|---|
| SDXL-DI | 134.075 | 0.044 | 42.383 | 0.009 | 7.036 | 5.456 |
| ScaleCrafter | 100.419 | 0.033 | 116.179 | 0.053 | 8.805 | 9.255 |
| DemoFusion | 72.378 | 0.020 | 94.975 | 0.045 | 12.450 | 11.382 |
| FouriScale | 93.079 | 0.029 | 128.862 | 0.068 | 8.248 | 8.446 |
| Ours | 49.796 | 0.004 | 71.369 | 0.029 | 12.572 | 6.240 |

2048^2 block of the same table: SDXL-DI 64.313/0.008/31.042/0.004/10.424/0.648; ScaleCrafter
67.545/0.013/60.151/0.020/11.399/0.653; DemoFusion 65.864/0.016/63.001/0.024/13.282/1.441; FouriScale
68.965/0.016/69.655/0.026/11.055/1.224; Ours 44.723/0.001/36.276/0.006/12.747/0.853.

v1 Table 1 has identical FID/KID/FIDc/KIDc/Time numbers, without the IS column.

Note in text (p.7): SDXL-DI gets the best FIDc/KIDc at 4096^2 "because it tends to generate multiple
duplicated objects and its crop may be closer to the reference images".

Other 4096^2 rows: Table 4 (p.12) SDXL+Real-ESRGAN 43.476/0.000/73.524/0.024/12.599 vs Ours
49.796/0.004/71.369/0.029/12.572. Table 8 (p.14) is 2048x4096.

## Protocol takeaways for SCA comparison
- Prompt set is a random 1024-caption LAION-Aesthetics-V2-6.5plus subset, not a fixed public split.
- FID/KID reference set is the base model's own 1024 generations (self-reference), not real images.
- Crop-FID details (crop size, count) are not given; attributed to Chai et al. 2022.
