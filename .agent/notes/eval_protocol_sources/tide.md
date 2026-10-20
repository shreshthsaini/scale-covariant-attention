# TIDE (arXiv 2603.08928v1): evaluation protocol audit

Paper: "TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers", Yihua Liu, Fanjiang Ye, Bowen Lin, Rongyu Fang, Chengming Zhang. arXiv 2603.08928v1, submitted 9 Mar 2026 by Chengming Zhang; v1 is the only version (arXiv abstract page, "Submission history"). PDF footer: "Preprint. March 11, 2026." (page 1).

Sources read:
- PDF text: `papers/2603.08928/paper.txt` (17 pages). Bold and underline were recovered from the PDF itself with PyMuPDF (font names plus vector underline rules), script `papers/tide_tools/table_spans.py` and `papers/tide_tools/hlines.py`.
- Code: the paper links NO code repository (the only github.com URL in the PDF is the FLUX citation "B. F. Flux. https://github.com/black-forest-labs/flux, 2024", page 10). A GitHub search found `github.com/Wendyl42/TIDE` (README: "Code release for "TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers""; commit 01f04a6, author "yihua", 2026-03-14). Shallow clone at `papers/tide_repo`, read only, never executed. It is not linked from the paper, so treat it as the authors' likely release, not as part of the paper's stated protocol. It contains generation code only: `run.py`, `run_qwen.py`, `flux/*.py`, `qwen/*.py`, `requirements.txt`, `README.md`. There are no evaluation scripts, metric code, FID or crop code, or prompt lists.

Citation format: (Section, Table/Figure, page) of arXiv 2603.08928v1. Printed page numbers match PDF page indices.

## a. Base model and checkpoint

- FLUX.1-dev: "We apply TIDE on FLUX.1-dev (Labs, 2024), an advanced open-source model based on DiT architecture with default generation resolution as 1024 × 1024." (Section 4.1, page 7). Table 1 caption: "All methods are built on FLUX." (Table 1, page 8).
- No other checkpoint is evaluated. FLUX.1-Krea-dev, FLUX.1-schnell and SD3 are not evaluated. Qwen-Image is mentioned only as an example DiT (Section 2.1, page 3); the paper reports no Qwen-Image results.
- Repo (not paper): `run.py` lines 70 and 78 load `"black-forest-labs/FLUX.1-dev"` for both transformer and pipeline. The repo also has `run_qwen.py` for `"Qwen/Qwen-Image"` (lines 52, 58; 40 steps default at line 26, `true_cfg_scale=4.0` at line 90), with no matching results in the paper.

## b. Benchmark prompt sets

Main quantitative evaluation (Table 1): "we use all 200 text prompts from DrawBench (Saharia et al., 2022) and all 195 image-prompt pairs from the 4K-resolution subset of Aesthetic-4K (Zhang et al., 2025) to generate images. The results are summarized in Table 1." (Section 4.3, page 7).

| Set | Source | Count | Selection | Used for |
|---|---|---|---|---|
| DrawBench | Saharia et al., 2022 | 200 prompts | full set ("all 200") | Table 1 CLIP, IR, Aesth |
| Aesthetic-4K, "4K-resolution subset" | Zhang et al., 2025 (Diffusion-4K, CVPR 2025) | 195 image-prompt pairs | full subset ("all 195") | Table 1 FID, KID, CLIP, IR, Aesth |
| Aesthetic-4K, random subset | Zhang et al., 2025 | 50 prompts | "randomly selected" | Ablations, Tables 3, 4, 5 (Section 4.5, page 8; Appendix B, page 11) |
| Aesthetic-4K, random subset | Zhang et al., 2025 | 20 prompts | "randomly selected" | User study, Table 2 (Section 4.4, page 8) |

- Mapping to Diffusion-4K's split name (cross-check from the Diffusion-4K paper, arXiv 2503.18352, Section 3.1, page 4, not from TIDE): "Within the evaluation set, 195 images have a short side greater than 4096, which we refer to as Aesthetic-Eval@4096." The parent evaluation set (Aesthetic-Eval) has 2,781 images selected from LAION-Aesthetics V2 6.5+ with short side greater than 2048. TIDE's "all 195 image-prompt pairs from the 4K-resolution subset" therefore matches Aesthetic-Eval@4096, all of it. TIDE never uses the name "Aesthetic-Eval@4096".
- The same 195-pair set appears to be used for both the 2048 × 2048 and the 4096 × 4096 rows of Table 1. The text names one prompt set and Table 1 has Aesthetic-4K columns at both resolutions (Section 4.3 and Table 1, pages 7 and 8). The paper does not say so per resolution.
- Ablation prompts: "We randomly selected 50 prompts from Aesthetic-4K (Zhang et al., 2025) to guide image generation at 4096 × 4096 resolution" (Section 4.5, page 8). Appendix B: "The experimental setting is same as Section 4.5 unless otherwise specified." (Appendix B, page 11). The paper does not say whether the 50 (and the 20 user-study prompts) come from the 195-image 4K subset or from the full evaluation set, and does not give a selection seed.
- Images per prompt: NOT STATED (one per prompt is implied by "all 195 image-prompt pairs ... to generate images", but this is not explicit).
- Seeds: NOT STATED in the paper. Repo: `run.py` line 35 `--seed` default 0, line 109 `torch.Generator(device=device).manual_seed(args.seed)`. The seed policy used for the paper's runs is UNVERIFIED.
- Figures: Figure 1 caption "Prompts are from DrawBench and Aesthetic-4K." (page 2). Figure 9 is DrawBench, Figure 10 is Aesthetic-4K (pages 14, 15).

## c. Resolutions

- Native: 1024 × 1024 (Section 4.1, page 7).
- Table 1: "2048 × 2048" and "4096 × 4096" (Table 1, page 8).
- Table 4: "2048 × 2048", "3072 × 3072", "4096 × 4096" (Table 4, page 12; caption writes 2048², 3072², 4096²).
- Tables 3 and 5 and the user study: 4096 × 4096 (Section 4.5 and Section 4.4, page 8).
- "We experiment with resolution up to 4096×4096." (Section 4.1, page 7).
- Qualitative only: Figure 1 labels 2048×4096, 4096×2304, 4096×1536, 2048×2048, 4096×2048, 4096×4096 (page 2). Panoramas "at 4096×1536 resolution" (Appendix C, page 13; Figures 11 and 12, pages 16 and 17). Figure 3 compares at 4096 × 4096 against 1024 × 1024 (page 4).
- H×W vs W×H convention: NOT STATED. Calling 4096×1536 "panoramic" implies width × height for that label. Repo filenames use `res_{width}x{height}` (`run.py`). All quantitative results are square, so the convention does not affect any number.

## d. Sampler settings (method and baselines)

- Steps and guidance: "The sampling steps are set to 28, and the guidance scale is set to 3.5." (Section 4.1, page 7). This is stated once under "Settings", with no separate baseline settings, so it presumably covers all methods. Repo: `run.py` line 34 steps default 28, line 108 `guidance_scale=3.5` hard-coded. In FLUX.1-dev this is the distilled guidance embedding. The repo pipeline has no true-CFG path for FLUX.
- Time shift, applied to all methods: Appendix A (page 11). Default FLUX.1 shift is linear in the image-token count LI with anchors LI = 256 (µ = 0.5) and LI = 4096 (µ = 1.15). At 4096 × 4096 (LI = 65536) "the linearly extrapolated µ can reach approximately 11.5", so "the effective t [would] remain near 1.0 for the first 27 steps". DyPE's code "clamp[s] µ at 1.15 for resolutions beyond 1024²". TIDE instead uses "a logarithmic time-shifting schedule for µ with respect to LI: µ = k · ln(LI) + b (23)" with the same anchors. "To ensure a fair comparison, all experiments in this paper (including baseline evaluations) have used this logarithmic time-shifting schedule." (Appendix A, page 11).
  - Derived by me from Eq. (23), not printed in the paper: k = 0.65/ln 16 = 0.2344, b = -0.800. That gives µ = 1.475 at 2048² (LI = 16384), 1.665 at 3072² (LI = 36864), and 1.800 at 4096² (LI = 65536). Linear µ at 4096² = 11.55, consistent with the paper's "approximately 11.5".
  - Consequence: the Dy-YaRN baseline was not run with DyPE's own clamped shift (µ = 1.15). It was re-run with TIDE's log shift.
- Scheduler: NOT STATED in the paper. Repo: `flux/pipeline_flux.py` imports `FlowMatchEulerDiscreteScheduler` (line 19), `sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)` (line 896), and `run.py` line 112 passes `shift_mode="log"` (`pipeline_flux.py` line 73 `'log': # suggested by us`, line 75 `'clamp': # same as DyPE`).
- Precision: NOT STATED in the paper. Repo: `torch.bfloat16` (`run.py` lines 72, 80). diffusers==0.35.2, torch==2.6.0 (`requirements.txt`).
- TIDE hyperparameters: αlow = 0.6, αhigh = 0.2 (Section 3.3 page 7; Section 4.1 page 7). τmax = 1.0 and τmin is the YaRN temperature of Eq. (10), sqrt(1/τ) = 0.1 ln(s) + 1 (Section 3.3, page 6; Section 2.3, page 4). Text-anchoring bias β = ln(λ) = 2 ln(s) (Eqs. 17 and 18, page 6). Repo: bias `math.log(width / 1024.0) + math.log(height / 1024.0)` added to the first `TEXT_TOKENS = 512` key positions (`run.py` lines 16 and 18), and only when width × height > 1024 × 1024 (line 95).
- Baselines: "(1) Direct Extrapolation; (2) YaRN (Peng et al., 2023); (3) Dy-YaRN (Issachar et al., 2025)." (Section 4.1, page 7). Direct Extrapolation is the row labelled "FLUX" in Table 1 (caption: "We compare our TIDE with FLUX, YaRN, and Dy-YaRN", page 8). Dy-YaRN is DyPE applied to YaRN (Issachar et al., 2025, "Dype: Dynamic position extrapolation for ultra high resolution diffusion", arXiv 2510.20766; reference list, page 9). NTK-Aware and NTK-by-Parts appear only qualitatively (Figures 3 and 4, pages 4 and 5). Baseline code versions: NOT STATED. Other high-resolution methods such as cascades, patch-based methods or SR are not compared quantitatively.
- What "TIDE (Ours)" is built on in Table 1: NOT STATED explicitly. In the ablations the base is Dy-YaRN, and "Full Method" is Dy-YaRN + TA + DTC (Section 4.5, page 8; Tables 3 to 5). Repo default is "YARN + DyPE + TIDE" (README; `run.py` line 40 `default='yarn'`, DyPE enabled unless `--disable_dype`).

## e. Metrics and implementation details

Metric list (Section 4.1, page 7): "Fréchet Inception Distance (Heusel et al., 2017) (FID) and Kernel Inception Distance (Bińkowski et al., 2018) (KID), CLIP Score (Radford et al., 2021), ImageReward (Xu et al., 2023) and Aesthetic-Score-Predictor (Schuhmann et al., 2022)." Also: "Both FID and KID measure the similarity between the ground images and the generated images. ... Note that since the count of images is small, KID, which is unbiased, has better reliability than FID." (Section 4.1, page 7). Table headers: CLIP↑, IR↑, Aesth↑, FID↓, KID↓ (Table 1, page 8).

| Metric | What the paper states | Implementation details |
|---|---|---|
| FID | Aesthetic-4K only; generated vs "ground images" (Section 4.1, page 7) | Library NOT STATED. Resize to 299 and interpolation NOT STATED. Reference count NOT STATED (implicitly the 195 Aesthetic-Eval@4096 images). How the non-square, short side over 4096 references are resized or cropped against square outputs NOT STATED. Repo `requirements.txt` lists `clean-fid==0.1.35` and `torchmetrics==1.8.2` but has no metric script, so the implementation is UNVERIFIED. |
| KID | Aesthetic-4K only; reported raw (e.g. 0.0081), not ×10³ (Table 1, page 8) | Same unknowns as FID. Subset size and number of subsets NOT STATED. |
| Patch / crop FID (FIDp, pFID, FID_patch, FID_c) | Not used. The words "patch" and "crop" never appear in the evaluation (the one "patches" mention, page 1, describes prior methods). | NOT REPORTED |
| IS | Not used | NOT REPORTED |
| CLIP Score | Radford et al., 2021; DrawBench and Aesthetic-4K | Backbone, input resize and scaling NOT STATED. Values about 13 to 29 suggest 100 × cosine, but the paper does not say so. |
| ImageReward ("IR") | Xu et al., 2023 | Version and input size NOT STATED. Repo `requirements.txt` lists `image-reward==1.5` (UNVERIFIED as the evaluation tool). |
| Aesthetic score ("Aesth") | "Aesthetic-Score-Predictor (Schuhmann et al., 2022)", where the citation is the LAION-5B paper (reference list, page 10) | Predictor version (v1 or v2, CLIP backbone) and input size NOT STATED |
| HPS (v2 or v2.1), PickScore, MUSIQ, MANIQA, CLIP-IQA, ARNIQA, Q-Align, GLCM, compression ratio, pyiqa | Not used | NOT REPORTED |
| User study | Section 4.4 and Table 2, page 8 | "We conduct a User Study with researchers familiar with generative AI models. We randomly selected 20 prompts from Aesthetic-4K to generate images with Dy-YaRN and our TIDE at 4096 × 4096 resolution. For each prompt, participants were required to rate images on a scale from 1 to 5 at three levels: (i) Alignment with the text. (ii) Structural integrity and proportions. (iii) Refined textures and details." Mean scores reported. Number of participants, blinding, display order and display resolution NOT STATED. |

## f. Latency and hardware

- "All experiments are conducted on NVIDIA RTX 6000 Ada GPUs." (Section 4.1, page 7). Number of GPUs, memory settings, and VAE tiling or offloading NOT STATED. The repo offers `--tiled_vae` and `--cpu_offloading`, both off by default (README).
- No latency, runtime, memory or throughput numbers are reported anywhere. Claims are qualitative: "without additional sampling overhead" (Abstract, page 1; Section 5, page 8) and "a training-free framework without additional sampling steps" (Section 1 contributions, page 2).

## g. Quantitative tables, transcribed exactly

Bold and underline come from the PDF fonts and vector rules (PyMuPDF), not from the plain-text extraction. **x** = bold, <u>x</u> = underlined. Every number was checked twice: once against `paper.txt` and once against the PDF span positions.

### Table 1 (page 8)

Caption: "Table 1. Quantitative comparison with baselines. We compare our TIDE with FLUX, YaRN, and Dy-YaRN on DrawBench and Aesthetic-4K. All methods are built on FLUX. Best results are bolded."

Column groups (confirmed from PDF x-positions): DrawBench = CLIP↑, IR↑, Aesth↑; Aesthetic-4K = FID↓, KID↓, CLIP↑, IR↑, Aesth↑.

| Resolution | Method | DB CLIP↑ | DB IR↑ | DB Aesth↑ | A4K FID↓ | A4K KID↓ | A4K CLIP↑ | A4K IR↑ | A4K Aesth↑ |
|---|---|---|---|---|---|---|---|---|---|
| 2048 × 2048 | FLUX | 26.63 | 0.46 | 5.53 | 162.59 | 0.0087 | 27.87 | 1.02 | 6.54 |
| 2048 × 2048 | YaRN | 26.76 | 0.66 | 5.76 | **141.93** | **0.0045** | 28.07 | 1.14 | 6.59 |
| 2048 × 2048 | Dy-YaRN | 26.52 | 0.72 | 5.74 | 161.32 | 0.0068 | 28.09 | 1.10 | 6.62 |
| 2048 × 2048 | **TIDE (Ours)** | **27.34** | **0.83** | **5.77** | 159.35 | 0.0066 | **28.60** | **1.21** | **6.68** |
| 4096 × 4096 | FLUX | 15.16 | -2.19 | 2.92 | 341.79 | 0.1952 | 13.46 | -2.25 | 2.58 |
| 4096 × 4096 | YaRN | 23.25 | -0.83 | 4.39 | 211.48 | 0.0494 | 23.86 | -0.24 | 5.75 |
| 4096 × 4096 | Dy-YaRN | 23.21 | -0.84 | 4.39 | 211.50 | 0.0496 | 23.90 | -0.24 | 5.75 |
| 4096 × 4096 | **TIDE (Ours)** | **26.13** | **0.18** | **5.34** | **158.39** | **0.0081** | **27.33** | **0.75** | **6.44** |

"FLUX" = Direct Extrapolation (Section 4.1, page 7). Aesthetic-4K = all 195 pairs of the 4K subset (Section 4.3, page 7). DrawBench = all 200 prompts.

FLUX.1-dev, Aesthetic-4K, 4096 × 4096 (Table 1), complete:
- FLUX (direct): FID 341.79, KID 0.1952, CLIP 13.46, IR -2.25, Aesth 2.58
- YaRN: FID 211.48, KID 0.0494, CLIP 23.86, IR -0.24, Aesth 5.75
- Dy-YaRN: FID 211.50, KID 0.0496, CLIP 23.90, IR -0.24, Aesth 5.75
- TIDE (Ours): FID 158.39, KID 0.0081, CLIP 27.33, IR 0.75, Aesth 6.44

### Table 2 (page 8), user study

Caption: "Table 2. User study results. We conduct a subjective evaluation comparing TIDE and Dy-YaRN on 20 randomly selected prompts. Participants rated images on a scale of 1 to 5 (higher is better). We report the mean score for each aspect."

| Method | TIDE (Ours) | Dy-YaRN |
|---|---|---|
| Text Alignment | **4.57** | 3.60 |
| Global Structure | **4.33** | 3.21 |
| Texture Quality | **4.46** | 2.48 |

### Table 3 (page 8), component ablation, 50 random Aesthetic-4K prompts, 4096 × 4096

Caption (typo "Asethetic" as printed): "Table 3. Component-wise ablation study. We investigate the impact of Text Anchoring and Dynamic Temperature Control on Asethetic-4K. The baseline model is Dy-YaRN. Best results are bolded."

| Configurations | CLIP↑ | IR↑ | Aesth↑ |
|---|---|---|---|
| Baseline | 23.60 | -0.44 | 5.64 |
| TA Only | 26.60 | 0.66 | **6.54** |
| DTC Only | 23.28 | -0.30 | 5.65 |
| Full Method | **26.76** | **0.86** | 6.46 |

No FID or KID in any ablation.

### Table 4 (page 12), text-anchoring bias ablation (setting "same as Section 4.5", i.e. 50 random Aesthetic-4K prompts)

Caption: "Table 4. Ablation study on Text Anchoring bias across different resolutions. We evaluate the impact of bias strategies on image generation quality at 2048², 3072², and 4096² resolutions. Dy-YaRN is adopted as the baseline. The best results for each resolution are bolded, and the second-best results are underlined."

| Method | Bias Scale | 2048² CLIP↑ | 2048² IR↑ | 2048² Aesth↑ | 3072² CLIP↑ | 3072² IR↑ | 3072² Aesth↑ | 4096² CLIP↑ | 4096² IR↑ | 4096² Aesth↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Dy-YaRN | - | 29.10 | 1.12 | 6.83 | 26.87 | 0.69 | 6.56 | 23.60 | -0.44 | 5.64 |
| TIDE (Ours) | 1.0 | **29.16** | 1.15 | **6.86** | 27.81 | 0.94 | 6.78 | 26.05 | 0.54 | 6.26 |
| TIDE (Ours) | 2.0 | 28.87 | **1.19** | 6.80 | 28.08 | 1.08 | 6.76 | <u>26.91</u> | <u>0.83</u> | **6.47** |
| TIDE (Ours) | 3.0 | 28.42 | 1.07 | 6.84 | 28.14 | 1.03 | 6.77 | **27.25** | 0.71 | 6.41 |
| TIDE (Ours) | **Adaptive** | <u>29.12</u> | <u>1.17</u> | <u>6.85</u> | **28.28** | **1.09** | **6.78** | 26.76 | **0.86** | <u>6.46</u> |

"TIDE (Ours)" is one multirow cell in the PDF. The 3072² columns have no underlines in the PDF, despite the caption.

### Table 5 (page 13), DTC exponent sensitivity (setting as Section 4.5, 4096 × 4096)

Caption: "Table 5. Hyperparameter sensitivity of Dynamic Temperature Control. We evaluate the impact of low-frequency (αlow) and high-frequency (αhigh) exponents. Dy-YaRN serves as the baseline. The overall best results are bolded."

| Method | αlow | αhigh | CLIP↑ | IR↑ | Aesth↑ |
|---|---|---|---|---|---|
| Dy-YaRN | - | - | 23.60 | -0.44 | 5.64 |
| TIDE | 0.0 | 0.0 | 26.60 | 0.66 | 6.54 |
| TIDE | 0.6 | 0.1 | **27.25** | 0.84 | 6.49 |
| TIDE | 0.6 | 0.2 | 26.76 | **0.86** | 6.46 |
| TIDE | 0.6 | 0.5 | 27.09 | 0.84 | **6.52** |
| TIDE | 0.6 | 1.0 | 26.03 | 0.53 | 6.19 |
| TIDE | 1.0 | 0.2 | 27.19 | 0.84 | 6.49 |
| TIDE | 1.0 | 1.0 | 27.03 | 0.76 | 6.46 |
| TIDE | 2.0 | 0.2 | 26.99 | 0.78 | 6.38 |
| TIDE | 2.0 | 1.0 | 26.97 | 0.73 | 6.37 |

αlow is printed once per group as a vertically centred multirow cell (0.6 spans the four αhigh ∈ {0.1, 0.2, 0.5, 1.0} rows; 1.0 and 2.0 each span two rows). I filled it per row from the PDF y-positions. Aesth 6.52 is bolded although the (0.0, 0.0) row shows 6.54, as printed.

Internal consistency (my check): Table 3 Baseline = Table 4 Dy-YaRN at 4096² = Table 5 Dy-YaRN (23.60, -0.44, 5.64). Table 3 TA Only = Table 5 (0.0, 0.0) (26.60, 0.66, 6.54). Table 3 Full Method = Table 4 Adaptive at 4096² = Table 5 (0.6, 0.2) (26.76, 0.86, 6.46). The 50-prompt numbers differ from the 195-pair Table 1 numbers (for example Dy-YaRN 4096² CLIP 23.60 vs 23.90), as expected for a different prompt subset.

## Protocol summary

FLUX.1-dev, 28 steps, guidance 3.5, a logarithmic time shift µ = k ln(LI) + b applied to all methods (µ about 1.80 at 4096², my derivation), on RTX 6000 Ada GPUs. Precision is not stated (the repo uses bf16). The main benchmark is all 200 DrawBench prompts (CLIP, ImageReward, LAION aesthetic) plus all 195 Aesthetic-4K 4K-subset image-prompt pairs (the Diffusion-4K Aesthetic-Eval@4096 split), scored with FID, KID, CLIP, ImageReward and aesthetic score at 2048² and 4096². Baselines are direct extrapolation, YaRN and Dy-YaRN (DyPE). No patch or crop FID, no IS, HPS, PickScore or no-reference IQA, and no latency numbers. FID and KID implementation, reference preprocessing, image counts per prompt, seeds and CLIP/aesthetic backbones are all unstated. The ablations (Tables 3 to 5) use 50 random Aesthetic-4K prompts and report only CLIP, IR and Aesth. The user study uses 20 random prompts at 4096², TIDE vs Dy-YaRN only.

## Open questions / NOT STATED

1. FID/KID library (clean-fid vs torchmetrics vs pytorch-fid vs torch-fidelity): NOT STATED. The repo requirements list clean-fid 0.1.35 and torchmetrics 1.8.2, but there is no metric code: UNVERIFIED.
2. FID/KID preprocessing: resize to 299 and interpolation, and how the non-square references (short side over 4096) and the square 2048² or 4096² outputs are resized or cropped: NOT STATED.
3. Reference set for FID/KID: "ground images" only. Count (presumably 195) NOT STATED. Whether the same 195 references are used for the 2048² rows: NOT STATED.
4. KID subset size and number of subsets: NOT STATED.
5. Patch or crop FID: not used at all.
6. CLIP Score backbone (ViT-B/32, ViT-L/14, ...) and scaling: NOT STATED. Aesthetic predictor version and backbone: NOT STATED (the citation is LAION-5B). ImageReward version: NOT STATED.
7. Images per prompt and seeds: NOT STATED (repo default seed 0).
8. Precision (bf16) and scheduler (FlowMatch Euler): NOT STATED in the paper; repo only.
9. Which positional method TIDE sits on in Table 1 (YaRN vs Dy-YaRN): NOT STATED. The ablations and the repo default use YaRN + DyPE + TIDE.
10. Baseline implementations and versions (YaRN, Dy-YaRN code): NOT STATED. Dy-YaRN was re-run with TIDE's log shift, not DyPE's clamped µ (Appendix A, page 11).
11. Source pool and seed for the 50 ablation prompts and the 20 user-study prompts (the 195-image 4K subset or the full 2,781-image evaluation set): NOT STATED.
12. User study: number of participants, blinding, ordering and viewing conditions: NOT STATED.
13. Runtime, latency and memory: not reported. GPU count: NOT STATED.
14. H×W vs W×H convention: NOT STATED (irrelevant for the square quantitative results).
15. The code repo `github.com/Wendyl42/TIDE` is not linked from the paper. It is attributed to the authors by its README title and commit author "yihua" only: UNVERIFIED as official.
