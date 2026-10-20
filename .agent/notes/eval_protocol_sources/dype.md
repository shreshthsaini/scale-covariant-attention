# DyPE: evaluation protocol audit

Paper: "DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion", Issachar et al.
arXiv 2510.20766v2 (v1 23 Oct 2025, v2 29 Jan 2026; the fetched PDF carries the stamp
"arXiv:2510.20766v2 [cs.CV] 29 Jan 2026"). 22 pages. Code README title: "[ICML 2026] DyPE".

Sources read:
- PDF text: papers/2510.20766/paper.txt. v1 was not read; any v1 vs v2 differences are UNVERIFIED.
- Code: /scratch/09032/saini_2/constancy_rope/third_party/DyPE, commit 436fe61 (2026-05-18), equal to
  the remote HEAD of github.com/guyyariv/DyPE (git ls-remote, 2026-10-20). Files: run_dype.py,
  run_dype_qwen.py, flux/{pipeline_flux,transformer_flux}.py, qwen/transformer_qwenimage.py. No
  evaluation or metric scripts and no prompt lists.
- For the patch-FID lineage only: HiFlow, arXiv 2504.06232v2 (papers/2504.06232/paper.txt).

Citation format: [v2, Sec X, p. P] = arXiv 2510.20766v2, section X, PDF page P.

## a. Base model and checkpoint

- "We evaluate DYPE on top of the pre-trained FLUX (Labs, 2024), specifically the FLUX.1-Krea-dev
  version, whose effective generation resolution is 1024 x 1024." [v2, Sec 4.1.1, p. 5]
- Table 1 caption: "All methods are built on FLUX." [v2, Table 1, p. 6]
- Code: run_dype.py loads `black-forest-labs/FLUX.1-Krea-dev` for the transformer and pipeline with
  `torch_dtype=torch.bfloat16` (lines 53-65). Precision is not stated in the paper.
- Other backbones: FiTv2-XL/2 on ImageNet (Table 4, Sec 4.2), Qwen-Image (Table 8, App. E), Wan2.1
  1.3B video (Table 9, App. E). Not FLUX; not transcribed.

## b. Prompt sets

1. DrawBench: "a set of 200 text prompts" [v2, Sec 4.1.1 Benchmarks, p. 5]. Table 1 uses DrawBench
   (count per cell not restated; presumably all 200, NOT STATED explicitly). Ablations (Tables 6, 7)
   use "50 DrawBench prompts (sampled every 4th index)" [v2, Tables 6-7 captions, p. 16-17].
2. Aesthetic-4K: "We use its 4K subset (Aesthetic-Eval@4096), which comprises 195 curated image-prompt
   pairs, and downsample them to match the target test resolutions for fair comparison. Following the
   official protocol, we report (i) CLIPScore, (ii) ImageReward, (iii) Aesthetics score, and (iv) FID"
   [v2, Sec 4.1.1, p. 6-7]. So the full 195-pair Aesthetic-Eval@4096 split, no subset, for Table 1.
   Independent check: the Hugging Face Aesthetic-4K tree (dumped by a sibling agent to
   papers/aesthetic4k_hf/) has 195 images under eval/size_4096 and a 195-line
   eval4096/metadata.jsonl ({"file_name", "text"} records), matching the 195 count.
3. Human study: "a curated subset of 20 prompts from Aesthetic-4K, obtained by sampling every fourth
   entry to ensure uniform coverage" [v2, Sec 4.1.1 Perceptual Evaluation, p. 7]. Note: stride 4 over
   195 entries gives 49 prompts, so which 20 were kept is NOT STATED.
4. Resolution scaling (Fig. 5): "Using 20 Aesthetic-4K prompts sampled at intervals of 10" [v2, Sec
   4.1.1, p. 7].
5. Table 3 (non-PE baselines, patch metrics): the prompt set is NOT STATED in the caption or in Sec
   4.1.2. Inference only: the DY-YaRN FID at 4096 x 4096 (186.00) is identical to its Aesthetic-4K FID
   in Table 1, which is consistent with Aesthetic-4K.
- Images per prompt: NOT STATED. Seeds: NOT STATED (code default `--seed 0`, run_dype.py line 27).

## c. Resolutions

- Table 1: 2048 x 3072, 3072 x 2048, 3072 x 3072, 4096 x 4096 [v2, Table 1, p. 6]. DyPE does not
  state the HxW vs WxH convention (NOT STATED); the text says "FLUX performs relatively better on
  landscape resolutions than portrait" without mapping them to columns. Code arguments are
  `--height`, `--width`. (For comparison only, HiFlow's Table 1 header says "Resolution (height x
  width)".)
- Table 3: 2048 x 2048 and 4096 x 4096 [v2, Table 3, p. 8].
- Fig. 5: six square resolutions from 1024^2 to 6144^2 [v2, Sec 4.1.1, p. 7].
- Human study: 4096^2 [v2, p. 7]. Ablations: 3072^2 [v2, Tables 6-7].

## d. Sampler settings

- Paper: "Diffusion inference is performed with 28 sampling steps. For our method, we apply
  lambda_s = lambda_t = 2." and "We set alpha = 1, beta = 32, and use an effective resolution of
  L = 1024." [v2, App. C, p. 14]
- Guidance scale: NOT STATED in the paper. Scheduler / shift: NOT STATED in the paper.
- Code (run_dype.py): `--steps` default 28 (line 26); `guidance_scale=4.5` hard-coded (line 75);
  `--method` in {yarn, ntk, base}, default yarn; `--no_dype` flag (lines 28-39). The FLUX, NTK and
  YaRN baselines run through the same script and pipeline, so they share steps, guidance and shift.
  Whether the paper's baselines used exactly this script: UNVERIFIED.
- Time shift (flux/pipeline_flux.py lines 847-854): `calculate_shift(image_seq_len,
  base_image_seq_len=256, image_seq_len, base_shift=0.5, max_shift=1.15)` with the original
  `# self.scheduler.config.get("max_image_seq_len", 4096),` line commented out. Hence mu = 1.15 at
  every resolution (shift e^1.15, about 3.16), instead of the stock linear extrapolation of mu.
- Non-PE baselines (Table 3): settings NOT STATED; which FLUX checkpoint HiFlow and I-Max used is NOT
  STATED.

## e. Metrics and implementation

- DrawBench: CLIP-Score (Hessel et al., 2022) "based on CLIP (Radford et al., 2021)", ImageReward
  (Xu et al., 2023), "Aesthetic-Score-Predictor (Schuhmann et al., 2022)" [v2, Sec 4.1.1, p. 5-6].
  CLIP backbone NOT STATED; aesthetic predictor version NOT STATED.
- Aesthetic-4K: CLIPScore, ImageReward, Aesthetics score, FID (Heusel et al., 2017), "following the
  official protocol" (Diffusion-4K) [v2, p. 6-7].
- FID: implementation (clean-fid / torchmetrics / pytorch-fid / torch-fidelity) NOT STATED; Inception
  resize and interpolation NOT STATED. Reference images: the 195 Aesthetic-Eval@4096 images,
  "downsample[d] ... to match the target test resolutions" [v2, p. 6]; resampling filter NOT STATED.
- patch-FID, IS, patch-IS (Table 3 only): "Consistent with previous works (Bu et al., 2025), our
  evaluation focuses on patch-based fidelity and detail preservation. Specifically, we report: (i) FID
  ..., (ii) patch-FID, (iii) Inception Score (IS) ..., and (iv) patch-Inception Score (patch-IS)."
  [v2, Sec 4.1.2, p. 7]. Crop size, crops per image, crop positions, resizing, reference crops: ALL
  NOT STATED.
  - The cited protocol source, HiFlow (arXiv 2504.06232v2, Sec 4.1 Evaluation, p. 7), also gives no
    crop size: "the patch-version FIDpatch and ISpatch are calculated based on local image patches to
    quantify the quality of synthesized details." HiFlow computes FID against "10K real high-quality
    images (with at least 1024 x 1024 resolution) sourced from LAION-High-Resolution" over "1K
    high-quality captions"; DyPE does not say it uses those references.
- KID, HPS, PickScore, MUSIQ, MANIQA, CLIP-IQA, GLCM, compression ratio: not used.
- User study: 50 raters, 20 prompts, pairwise at 4096^2 on FLUX, three pairs per prompt (NTK vs
  DY-NTK, TASR vs DY-NTK, YaRN vs DY-YaRN), three questions (text alignment, geometry/structure,
  textures/details) [v2, Sec 4.1.1, p. 7].

## f. Hardware / latency

- "Unless otherwise stated, all experiments are conducted on a single L40S GPU." [v2, App. C, p. 14]
- No latency or runtime numbers (grep for latency/runtime/seconds/overhead).

## g. Table transcriptions (exactly as extracted; bold/underline not preserved by text extraction)

### Table 1 [v2, p. 6]
Caption: "High-resolution image generation on DrawBench and Aesthetic-4K on multiple resolutions. Each
row reports CLIPScore (CLIP), ImageReward (IR), Aesthetics (Aesth) for DrawBench, and CLIP, IR, Aesth,
and FID for Aesthetic-4K. All methods are built on FLUX."

Block 1: per resolution, DrawBench CLIP IR Aesth | Aesthetic-4K CLIP IR Aesth FID
| Method | 2048x3072 DB CLIP | DB IR | DB Aesth | A4K CLIP | A4K IR | A4K Aesth | A4K FID | 3072x2048 DB CLIP | DB IR | DB Aesth | A4K CLIP | A4K IR | A4K Aesth | A4K FID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FLUX | 26.64 | -0.28 | 5.14 | 28.64 | 0.32 | 6.11 | 186.31 | 26.56 | 0.16 | 5.33 | 28.74 | 0.97 | 6.17 | 148.29 |
| NTK | 27.68 | 0.21 | 5.31 | 29.13 | 0.99 | 6.49 | 180.87 | 27.28 | 0.51 | 5.39 | 28.97 | 1.17 | 6.25 | 146.74 |
| TASR | 27.86 | 0.30 | 5.15 | 29.13 | 0.97 | 6.12 | 201.40 | 27.40 | 0.55 | 5.22 | 29.05 | 1.15 | 5.95 | 186.21 |
| DY-NTK | 27.91 | 0.48 | 5.54 | 29.14 | 1.10 | 6.56 | 176.13 | 27.44 | 0.60 | 5.55 | 29.11 | 1.21 | 6.53 | 146.40 |
| YaRN | 28.27 | 0.52 | 5.63 | 29.28 | 1.01 | 6.59 | 179.54 | 27.79 | 0.62 | 5.48 | 29.12 | 1.24 | 6.49 | 147.12 |
| DY-YaRN | 28.43 | 0.71 | 5.69 | 29.44 | 1.17 | 6.61 | 179.51 | 28.17 | 0.81 | 5.68 | 29.20 | 1.28 | 6.51 | 146.84 |

Block 2:
| Method | 3072x3072 DB CLIP | DB IR | DB Aesth | A4K CLIP | A4K IR | A4K Aesth | A4K FID | 4096x4096 DB CLIP | DB IR | DB Aesth | A4K CLIP | A4K IR | A4K Aesth | A4K FID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FLUX | 25.11 | -0.53 | 5.01 | 28.62 | 0.46 | 6.16 | 187.96 | 16.43 | -1.97 | 3.29 | 25.50 | -0.73 | 5.42 | 195.68 |
| NTK | 26.07 | -0.14 | 5.05 | 28.68 | 0.96 | 6.45 | 182.38 | 17.49 | -1.88 | 3.57 | 24.88 | -0.54 | 5.50 | 203.85 |
| TASR | 26.87 | 0.18 | 5.01 | 28.79 | 1.00 | 6.01 | 194.23 | 21.21 | -1.69 | 3.56 | 25.09 | -0.09 | 5.96 | 221.39 |
| DY-NTK | 27.02 | 0.30 | 5.36 | 28.83 | 1.10 | 6.57 | 179.98 | 21.51 | -1.22 | 4.25 | 28.06 | 0.79 | 6.42 | 183.72 |
| YaRN | 27.92 | 0.41 | 5.37 | 29.26 | 1.14 | 6.67 | 184.16 | 25.71 | -0.34 | 4.85 | 28.57 | 0.85 | 6.47 | 192.19 |
| DY-YaRN | 28.12 | 0.66 | 5.55 | 29.75 | 1.24 | 6.70 | 179.82 | 26.94 | 0.15 | 5.17 | 29.28 | 1.09 | 6.67 | 186.00 |

Aesthetic-4K at 4096 x 4096 (FLUX.1-Krea-dev), extracted: FLUX 25.50 / -0.73 / 5.42 / 195.68;
NTK 24.88 / -0.54 / 5.50 / 203.85; TASR 25.09 / -0.09 / 5.96 / 221.39; DY-NTK 28.06 / 0.79 / 6.42 /
183.72; YaRN 28.57 / 0.85 / 6.47 / 192.19; DY-YaRN 29.28 / 1.09 / 6.67 / 186.00 (CLIP / IR / Aesth /
FID). Note the best FID at 4096^2 in Table 1 is DY-NTK (183.72), not DY-YaRN.

### Table 2 [v2, p. 7]: human evaluation on Aesthetic-4K
Caption: "Human evaluation on Aesthetic-4K. Each cell reports the percentage of pairwise comparisons in
which DYPE was preferred."
| Comp. | Txt | Str | Det |
|---|---|---|---|
| NTK vs. DY-NTK | 88.5 | 88.7 | 88.3 |
| TASR vs. DY-NTK | 70.5 | 80.6 | 94.2 |
| YaRN vs. DY-YaRN | 90.1 | 87.3 | 88.1 |

### Table 3 [v2, p. 8]
Caption: "Evaluation at 2048^2 and 4096^2 resolutions on non-PE approaches. Baselines include
U-Net-based methods (*), tuning-based approaches ([ss]), progressive refinement pipelines (double
dagger), and diffusion combined with super-resolution (dagger)." ([ss] stands for the section-sign
glyph used in the caption.)
| Method | 2048 FID | 2048 patch-FID | 2048 IS | 2048 patch-IS | 4096 FID | 4096 patch-FID | 4096 IS | 4096 patch-IS |
|---|---|---|---|---|---|---|---|---|
| DemoFusion* | 205.59 | 199.00 | 10.03 | 10.49 | 205.86 | 195.69 | 10.93 | 7.92 |
| FreCaS* | 201.07 | 195.73 | 11.45 | 10.52 | 200.95 | 202.41 | 11.14 | 8.02 |
| DiffuseHigh* | 178.08 | 117.43 | 14.47 | 10.68 | 186.25 | 96.99 | 12.62 | 7.56 |
| FreeScale* | 199.87 | 126.45 | 9.89 | 10.55 | 259.24 | 191.23 | 10.66 | 7.74 |
| I-Max (double dagger) | 174.35 | 107.81 | 13.91 | 9.77 | 187.29 | 87.71 | 13.44 | 5.78 |
| FLUX+BSRGAN (dagger) | 175.26 | 106.88 | 13.84 | 10.51 | 201.12 | 98.51 | 10.11 | 8.34 |
| UltraPixel (double dagger) | 181.06 | 114.69 | 14.21 | 11.08 | 186.75 | 88.99 | 13.83 | 8.54 |
| Diffusion-4K [ss] | 178.25 | 98.35 | 13.41 | 10.37 | 198.16 | 94.82 | 13.82 | 4.72 |
| HiFlow (double dagger) | 173.00 | 106.65 | 13.36 | 10.32 | 174.39 | 78.38 | 13.38 | 6.67 |
| DY-YaRN (Ours) + HiFlow (double dagger) | 166.71 | 103.06 | 13.60 | 10.74 | 169.46 | 79.64 | 14.18 | 7.06 |
| DY-YaRN (Ours) | 142.74 | 96.34 | 14.76 | 11.95 | 186.00 | 78.33 | 13.96 | 10.13 |

### Table 5 [v2, App. B, p. 15]
Caption: "Comparison of NTK-aware, Time-Aware Scaled RoPE, and DY-NTK-aware on the Aesthetic-4K
benchmark" (resolution not given; values equal Table 1's 4096 x 4096 Aesthetic-4K columns).
| Method | CLIPScore | ImageReward | Aesthetic-Score | FID |
|---|---|---|---|---|
| NTK-aware | 24.88 | -0.54 | 5.50 | 203.85 |
| Time-Aware Scaled RoPE | 25.09 | -0.09 | 5.96 | 221.39 |
| DY-NTK-aware | 28.06 | 0.79 | 6.42 | 183.72 |

### Table 6 [v2, App. D, p. 16]
Caption: "Comparison of scheduler designs for DY-NTK-aware on FLUX at 3072^2 resolution. Evaluated on 50
DrawBench prompts (sampled every 4th index). Baselines (FLUX, NTK-Aware) are included. Metrics:
CLIP-Score (CLIP), ImageReward (IR), and Aesthetics-Score (Aesth)."
| Variant | lambda_s | lambda_t | CLIP | IR | Aesth |
|---|---|---|---|---|---|
| FLUX | - | - | 25.33 | -0.52 | 5.12 |
| NTK-Aware | - | - | 25.83 | -0.13 | 4.99 |
| Multiplicative | 1.0 | 1.0 | 25.67 | 0.11 | 5.08 |
| Multiplicative | 1.5 | 1.0 | 25.75 | 0.10 | 5.18 |
| Multiplicative | 2.0 | 1.0 | 26.09 | 0.16 | 5.31 |
| Multiplicative | 2.5 | 1.0 | 26.38 | 0.21 | 5.34 |
| Multiplicative | 2.0 | 0.5 | 26.28 | 0.21 | 5.34 |
| Multiplicative | 2.0 | 2.0 | 26.12 | 0.17 | 5.40 |
| Exponential | 1.0 | 1.0 | 25.81 | -0.13 | 5.03 |
| Exponential | 1.5 | 1.0 | 26.02 | 0.10 | 5.26 |
| Exponential | 2.0 | 1.0 | 26.52 | 0.29 | 5.39 |
| Exponential | 2.5 | 1.0 | 26.21 | 0.10 | 5.35 |
| Exponential | 2.0 | 0.5 | 26.69 | 0.24 | 5.34 |
| Exponential | 2.0 | 2.0 | 26.51 | 0.30 | 5.41 |

### Table 7 [v2, App. D, p. 17]
Caption: "Comparison of scheduler application strategies for DY-YaRN on FLUX at 3072^2 resolution.
Evaluated on 50 DrawBench prompts (sampled every 4th index), with baselines (FLUX, YaRN) included. ...
All experiments use the best scheduler configuration from Sec. D (exponential with lambda_s = 2,
lambda_t = 2)."
| Variant | NTK term kappa(t) | By-parts kappa(t) | CLIP | IR | Aesth |
|---|---|---|---|---|---|
| FLUX | - | - | 25.33 | -0.52 | 5.12 |
| YaRN | - | - | 27.32 | 0.36 | 5.47 |
| kappa(t) on NTK only | yes | - | 27.35 | 0.37 | 5.50 |
| kappa(t) on by-parts only | - | yes | 27.78 | 0.58 | 5.56 |
| kappa(t) on NTK & kappa(t) on by-parts | yes | yes | 27.76 | 0.36 | 5.41 |

(Tables 4, 8, 9 are FiTv2 ImageNet, Qwen-Image and Wan2.1; not FLUX; not transcribed.)

## Protocol summary

FLUX.1-Krea-dev; 28 steps (paper); guidance 4.5 and mu = 1.15 shift cap only in code; L40S GPU.
DrawBench 200 prompts plus the full Aesthetic-Eval@4096 split (195 image-prompt pairs), with
references downsampled to the test resolution, metrics CLIPScore, ImageReward, LAION aesthetic score,
FID. A separate non-PE comparison (Table 3) adds patch-FID and patch-IS "consistent with" HiFlow,
with no crop protocol given in DyPE or HiFlow. No metric implementation is named.

## Open questions / NOT STATED

- Guidance scale and shift in the paper (code only: 4.5 and mu = 1.15).
- FID implementation, Inception resize, reference downsampling filter.
- patch-FID / patch-IS crop size, crops per image, positions, resizing, reference crops.
- Prompt set and reference set for Table 3.
- Images per prompt and seeds.
- CLIP backbone and aesthetic predictor version.
- Which 20 prompts the human study used (stride 4 over 195 gives 49).
- v1 vs v2 table differences (v1 not read): UNVERIFIED.
