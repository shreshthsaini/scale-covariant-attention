# SEGA: evaluation protocol audit

Paper: "SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers",
Rajabi, Shaban, Roohi, Lindell, Taati. arXiv 2605.22668v1 (21 May 2026; v1 is the only version, checked
on the abs page 2026-10-20). 27 pages, NeurIPS 2026 template (README says "[NeurIPS 2026]").

Sources read:
- PDF text: papers/2605.22668/paper.txt (page markers "===== PAGE n =====").
- arXiv TeX source (e-print), extracted .tex only: papers/2605.22668/src/x/ (section/*.tex, tables/*.tex).
  Checked every comment line; none holds protocol details.
- Code: /scratch/09032/saini_2/constancy_rope/third_party/sega, commit ddaab08 (2026-09-30), which equals
  the remote HEAD of github.com/rajabi2001/sega (git ls-remote, 2026-10-20). Inference code only.
- Project page https://rajabi2001.github.io/sega/ (papers/sega_page/index.html). No protocol details.

Citation format: [vN, Sec X, p. P] = arXiv 2605.22668v1, section X, PDF page P.

## a. Base model and checkpoint

- "We evaluated our proposed method, SEGA on both Flux [1] and Qwen [2]." [v1, Sec 6 Experimental
  Settings, p. 7]
- "For the Flux model, we specifically utilized the dev.Krea checkpoint." [v1, App. C, p. 17]
  i.e. FLUX.1-Krea-dev, not FLUX.1-dev.
- "all model weights and latent activations were cast to bfloat16." [v1, App. C, p. 17]
- Code agrees: flux_sega/run_flux.py `--checkpoint` default `'Krea-dev'`, expanded to
  `black-forest-labs/FLUX.1-Krea-dev` (lines 48-70); transformer and pipeline loaded with
  `torch_dtype=torch.bfloat16` (lines 81-91).
- SEGA is composed with NTK by default: "we use NTK [35] as the default length extrapolation method
  for SEGA ... we set gamma to 1.5 and kappa to 0.08." [v1, Sec 6, p. 7]
- Other backbones: Qwen-Image (Table 2, Tables 5 to 7); Table 4 compares SEGA (Flux rows, identical
  numbers to Table 1) with SDXL-based U-Net methods and Diffusion-4K [v1, App. F.1, p. 18].

## b. Prompt sets

1. Aesthetic-4K (Zhang et al. 2025, Diffusion-4K, ref [20]).
   - "We used prompts and reference images from the Aesthetic-4K [20] dataset." [v1, Sec 6
     Evaluation, p. 8]. That is the entire description for the main tables (1 to 4).
   - Which split (Aesthetic-Eval@4096 vs @2048 vs train): NOT STATED.
   - Number of prompts / images for Tables 1 to 4: NOT STATED.
   - Subset rule for Tables 1 to 4: NOT STATED (no subset is mentioned).
   - Tables 6 and 7 (5120^2, 6144^2): "we evaluate on a randomly selected subset of 20 prompt-image
     pairs from Aesthetic-4K." [v1, App. F.3, p. 20]. Random seed of the selection NOT STATED.
   - Weak hint only (UNVERIFIED inference): the project page slider assets are named
     aesthetic/p00036, p00100, p00166, i.e. zero-padded prompt indices up to at least 166.
2. "Zero-Shot" benchmark: "we use an LLM to generate 200 curated, high-detail prompts ... with care
   taken to minimize overlap with the Aesthetic-4K dataset." [v1, App. F.2, p. 19]. Used only for
   Table 5 at 4096^2. LLM identity NOT STATED. Prompts not released in the repo.
- Images per prompt: NOT STATED. Seeds for the main evaluation: NOT STATED (App. B.2 analysis figure
  is "averaged across different seeds" [v1, App. B.2, p. 17]; code default `--seed 0`, run_flux.py
  line 47).

## c. Resolutions

- Main: "2048 x 4096, 4096 x 2048, 3072^2, and 4096^2" [v1, Sec 6 Evaluation, p. 8]. HxW vs WxH
  convention NOT STATED in the paper. Code arguments are `--height` / `--width` (run_flux.py lines
  44-45), and the README examples write `--height 4096 --width 4096`.
- Appendix: 5120^2 (Table 6), 6144^2 (Table 7) [v1, App. F.3, p. 19-20]; Table 9 at 4096^2, 5120^2,
  6144^2 [v1, App. p. 21]. Limitation section mentions SEGA "can scale up to 8192^2" (qualitative).
- Native resolution assumed: "the 1024^2 training resolution of Flux" [v1, App. F.3, p. 20].

## d. Sampler settings

- Paper: "We followed the standard inference settings provided by the official model implementations
  of Flux and Qwen, using their default samplers, number of denoising steps, and guidance scales."
  [v1, App. C, p. 17]. No numbers are given in the paper: steps NOT STATED, guidance NOT STATED,
  shift NOT STATED.
- "SEGA was applied at every denoising step throughout the entire trajectory, with no warmup" [v1,
  App. C, p. 17].
- Code (flux_sega/run_flux.py, commit ddaab08):
  - `--steps` default 28 (line 46); `guidance_scale=4.5` hard-coded in the pipe call (line 179);
    `--seed` default 0 (line 47); `max_sequence_length` pipeline default 512.
  - Time shift: flux_sega/pipeline_flux.py lines 871-877 call
    `calculate_shift(image_seq_len, base_image_seq_len=256, image_seq_len, base_shift=0.5,
    max_shift=1.15)`, i.e. the max_seq_len argument is replaced by image_seq_len, so
    mu = max_shift = 1.15 at every resolution (shift e^1.15, about 3.16). Stock diffusers would use
    max_image_seq_len 4096 and extrapolate mu linearly (mu about 11.5 at 4096^2). This code layout is
    copied from DyPE (README: "adapts the inference layout and scripts from DyPE"; DyPE's pipeline has
    the identical edit). Whether the paper's tables used exactly this code: UNVERIFIED.
  - Default sigmas `np.linspace(1.0, 1/num_inference_steps, num_inference_steps)` (line 867), Flux
    FlowMatchEuler scheduler from the checkpoint.
- Baselines: "the multi-stage guidance methods are exclusively evaluated on Flux to align with their
  official implementations." [v1, Sec 6 Baselines, p. 8]. Baseline steps / guidance / shift: NOT
  STATED. Whether HiFlow / I-Max / ScaleDiff ran on Krea-dev or FLUX.1-dev: NOT STATED.

## e. Metrics and implementation

Listed: "We evaluate image quality using FID [39] and the reference-free metrics MUSIQ (MSQ) [40], and
CLIP-IQA (CQA) [41]. Semantic alignment is measured by CLIP Score (CS) [42, 43], while ... ImageReward
(IR) [44], PickScore (PS) [45], and HPSv2 [46]." [v1, Sec 6 Evaluation, p. 8]

- FID: cites Heusel et al. [39]. Implementation (clean-fid / torchmetrics / pytorch-fid), resize,
  interpolation: NOT STATED. Reference set: "reference images from the Aesthetic-4K dataset"; number of
  references and how they are resized: NOT STATED.
- FIDp (column "FID_p" in Tables 1, 2, 4; TeX `FID$_p$`): never defined in the paper text, captions,
  appendix, or TeX comments. Crop size, crops per image, crop positions, resizing, reference crops:
  ALL NOT STATED. The released code contains no evaluation scripts (files: flux_sega/{run_flux,
  pipeline_flux, transformer_flux}.py, qwen_sega/..., ideogram4_sega/...). The project page has no
  metric details. So the crop protocol cannot be recovered from any SEGA source.
- MUSIQ, CLIP-IQA: implementation (pyiqa or other) and input resolution: NOT STATED.
- CLIP Score: cites CLIP [42] and CLIPScore [43]; backbone NOT STATED.
- HPS: "HPSv2 [46]" (ref [46] = Wu et al., arXiv 2306.09341). v2 vs v2.1 weights: NOT STATED. Reported
  on a 0 to 1 scale (e.g. 0.291 in Table 3, 0.29 in Tables 5 to 7).
- ImageReward, PickScore: implementations NOT STATED. PickScore is on the x100 logit scale (about 23).
- KID, IS, aesthetic predictor, GLCM, compression ratio, user study: not used.

## f. Hardware / latency

- "executed on NVIDIA H100 GPUs" [v1, App. C, p. 17]. No latency or runtime numbers anywhere
  (grep for latency/runtime/seconds/overhead). README: FLUX at 6144^2 or higher needs `--multi_gpu`.

## g. Table transcriptions (exactly as extracted; bold/underline not preserved by text extraction)

### Table 1 [v1, p. 8]
Caption: "Comparison of SEGA against state-of-the-art baselines on Flux across four high-resolution
settings on Aesthetic-4K [20]. Best and second-best results are shown in bold and underlined."

Block 1. Columns: Method | 2048 x 4096: IR PS CS MSQ FID FIDp | 4096 x 2048: IR PS CS MSQ FID FIDp
| Method | IR | PS | CS | MSQ | FID | FIDp | IR | PS | CS | MSQ | FID | FIDp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | 0.39 | 21.64 | 27.32 | 52.96 | 158.86 | 67.31 | -0.74 | 20.61 | 26.99 | 50.92 | 173.18 | 70.65 |
| HiFlow | 1.14 | 22.61 | 27.72 | 44.30 | 152.24 | 69.23 | 0.89 | 22.31 | 28.00 | 43.45 | 169.60 | 69.35 |
| I-Max | 1.11 | 22.71 | 27.60 | 32.58 | 157.82 | 68.21 | 0.87 | 22.54 | 28.30 | 38.71 | 162.30 | 66.92 |
| ScaleDiff | 1.17 | 22.84 | 29.03 | 55.84 | 157.62 | 72.21 | 0.97 | 22.51 | 28.58 | 56.85 | 162.38 | 77.65 |
| PI | -0.83 | 19.77 | 22.87 | 38.73 | 235.83 | 182.50 | -1.10 | 19.50 | 23.05 | 37.48 | 225.65 | 179.96 |
| NTK | 0.80 | 21.96 | 27.75 | 48.32 | 157.21 | 66.86 | 0.54 | 22.01 | 27.93 | 48.90 | 156.22 | 61.84 |
| YaRN | 0.97 | 22.63 | 28.48 | 52.30 | 156.97 | 83.35 | 0.30 | 21.82 | 27.99 | 52.10 | 154.44 | 76.42 |
| DyPE | 1.10 | 22.90 | 28.87 | 53.35 | 159.81 | 85.12 | 0.53 | 22.15 | 28.26 | 52.85 | 158.49 | 85.39 |
| UltraImage | 0.82 | 22.41 | 28.57 | 55.40 | 157.33 | 59.19 | 0.60 | 21.99 | 28.84 | 55.13 | 157.97 | 63.11 |
| SEGA | 1.21 | 22.91 | 29.18 | 53.65 | 151.93 | 64.54 | 0.86 | 22.58 | 28.99 | 53.30 | 153.10 | 55.85 |

Block 2. Columns: Method | 3072 x 3072: IR PS CS MSQ FID FIDp | 4096 x 4096: IR PS CS MSQ FID FIDp
| Method | IR | PS | CS | MSQ | FID | FIDp | IR | PS | CS | MSQ | FID | FIDp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | 0.49 | 23.00 | 28.18 | 49.93 | 162.25 | 68.63 | -0.72 | 20.31 | 25.34 | 26.52 | 183.33 | 121.93 |
| HiFlow | 1.26 | 23.22 | 28.48 | 43.45 | 154.32 | 64.72 | 1.26 | 23.17 | 28.40 | 33.37 | 155.56 | 60.70 |
| I-Max | 1.28 | 23.15 | 28.42 | 35.23 | 151.98 | 64.56 | 1.26 | 23.10 | 28.45 | 26.36 | 151.74 | 64.06 |
| ScaleDiff | 1.30 | 23.22 | 28.73 | 53.58 | 153.73 | 72.92 | 1.23 | 23.16 | 28.64 | 44.51 | 153.05 | 76.14 |
| PI | 0.19 | 21.03 | 25.64 | 48.61 | 200.04 | 155.60 | -0.28 | 20.54 | 24.16 | 29.09 | 208.48 | 202.01 |
| NTK | 0.90 | 22.47 | 28.27 | 35.99 | 156.10 | 63.25 | -0.29 | 20.82 | 25.52 | 23.85 | 182.29 | 138.61 |
| YaRN | 1.11 | 22.89 | 29.10 | 50.49 | 157.04 | 84.31 | 0.88 | 22.21 | 28.30 | 42.87 | 160.48 | 98.52 |
| DyPE | 1.21 | 23.15 | 29.17 | 51.71 | 156.91 | 82.45 | 1.01 | 22.56 | 28.79 | 43.23 | 156.21 | 97.88 |
| UltraImage | 1.17 | 22.65 | 29.30 | 53.56 | 155.92 | 61.61 | 0.61 | 21.74 | 28.16 | 43.61 | 167.04 | 63.91 |
| SEGA | 1.30 | 23.26 | 29.30 | 52.97 | 151.08 | 43.86 | 1.26 | 23.18 | 29.22 | 45.73 | 150.05 | 51.28 |

(Base = Flux, i.e. FLUX.1-Krea-dev per App. C, run directly at the target resolution.)

### Table 2 [v1, p. 9]: Qwen only ("Quantitative comparison on Qwen across all four resolutions on
Aesthetic-4K [20]"; columns Resolution, Method, IR, PS, HPS, CS, MSQ, CQA, FID, FIDp). Not FLUX; not
transcribed.

### Table 3 [v1, p. 9]
Caption: "Ablation study on Flux at 4096x4096 resolution on Aesthetic-4K. [20]"
| Method | IR | PS | HPS | CS | MSQ | CQA | FID |
|---|---|---|---|---|---|---|---|
| NTK + Fixed Scaling | 0.66 | 22.10 | 0.270 | 28.38 | 42.21 | 0.690 | 162.92 |
| SEGA W/ Axis-only | 1.15 | 23.05 | 0.290 | 28.86 | 44.08 | 0.701 | 152.98 |
| SEGA W/ Global-only | 1.13 | 22.89 | 0.286 | 28.81 | 43.55 | 0.671 | 153.58 |
| YaRN + SEGA | 0.94 | 22.21 | 0.280 | 28.23 | 44.36 | 0.707 | 159.80 |
| DyPE + SEGA | 1.18 | 22.78 | 0.287 | 28.84 | 45.18 | 0.720 | 154.51 |
| SEGA | 1.26 | 22.35 | 0.291 | 29.22 | 45.72 | 0.725 | 150.05 |

Internal inconsistency: SEGA at 4096^2 has PS 22.35 / MSQ 45.72 here but PS 23.18 / MSQ 45.73 in
Table 1 (and PS 23.18, MUSIQ 45.73 in Table 9; MSQ 45.72 in Table 4).

### Table 4 [v1, p. 18]
Caption: "Comparison of SEGA against state-of-the-art baselines on SDXL [47] and Diffusion-4K across
four high-resolution settings on Aesthetic-4K [20]. Best and second-best results are shown in bold and
underlined." (Baselines are SDXL-based U-Net methods plus fine-tuned Diffusion-4K, whose backbone is
NOT STATED here; the SEGA rows are the Flux rows of Table 1.)

Block 1 (2048 x 4096 | 4096 x 2048), columns IR PS CS MSQ FID FIDp each:
| Method | IR | PS | CS | MSQ | FID | FIDp | IR | PS | CS | MSQ | FID | FIDp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Diffusion-4K | 0.71 | 21.89 | 27.94 | 35.76 | 156.68 | 55.95 | 0.48 | 21.89 | 27.92 | 36.97 | 154.68 | 55.03 |
| DemoFusion | 0.41 | 22.03 | 28.73 | 48.39 | 165.57 | 77.62 | -0.30 | 21.56 | 27.27 | 53.46 | 169.48 | 81.01 |
| FreCas | 0.93 | 22.34 | 28.89 | 55.21 | 156.14 | 100.80 | 0.47 | 21.62 | 28.73 | 53.05 | 158.95 | 99.78 |
| FreeScale | 0.89 | 22.23 | 28.78 | 50.94 | 162.86 | 98.66 | 0.69 | 21.92 | 28.97 | 44.61 | 170.32 | 94.87 |
| DiffuseHigh | 0.73 | 22.44 | 28.29 | 45.44 | 158.73 | 82.73 | 0.49 | 22.18 | 28.98 | 48.03 | 160.31 | 77.27 |
| SEGA | 1.21 | 22.91 | 29.18 | 53.65 | 151.93 | 64.54 | 0.86 | 22.58 | 28.99 | 53.30 | 153.10 | 55.85 |

Block 2 (3072 x 3072 | 4096 x 4096):
| Method | IR | PS | CS | MSQ | FID | FIDp | IR | PS | CS | MSQ | FID | FIDp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Diffusion-4K | 0.87 | 22.29 | 28.28 | 34.49 | 154.83 | 55.79 | 0.52 | 21.74 | 27.48 | 24.13 | 161.62 | 70.05 |
| DemoFusion | 0.89 | 23.00 | 29.35 | 51.45 | 153.65 | 76.56 | 0.88 | 22.99 | 29.45 | 41.27 | 156.94 | 72.86 |
| FreCas | 1.13 | 23.12 | 29.20 | 51.70 | 155.98 | 91.26 | 1.06 | 22.90 | 29.82 | 39.32 | 156.57 | 82.27 |
| FreeScale | 1.05 | 22.78 | 29.74 | 43.63 | 170.63 | 86.80 | 1.06 | 22.82 | 29.75 | 33.94 | 167.83 | 74.72 |
| DiffuseHigh | 0.91 | 23.17 | 29.59 | 49.48 | 159.33 | 73.97 | 0.93 | 23.16 | 29.21 | 36.44 | 160.43 | 63.19 |
| SEGA | 1.30 | 23.26 | 29.29 | 51.89 | 151.08 | 43.86 | 1.26 | 23.18 | 29.22 | 45.72 | 150.05 | 51.28 |

Inconsistency: SEGA 3072^2 here CS 29.29 / MSQ 51.89 vs Table 1 CS 29.30 / MSQ 52.97.

### Table 5 [v1, p. 18], zero-shot benchmark (200 LLM prompts), 4096^2
Caption: "Quantitative comparison at 4096^2 resolution on the zero-shot benchmark. Methods are grouped
by backbone model; best and second-best results are bolded and underlined within each group. dagger
denotes a closed-source proprietary model." No reference set, so no FID.
| Backbone | Method | IR | PS | CS | HPS | MSQ |
|---|---|---|---|---|---|---|
| Flux | Base | -1.50 | 19.41 | 22.17 | 0.23 | 25.83 |
| Flux | NTK | -0.30 | 20.66 | 25.36 | 0.25 | 29.23 |
| Flux | YaRN | 0.70 | 21.83 | 28.95 | 0.27 | 41.18 |
| Flux | DyPE | 0.78 | 22.01 | 29.65 | 0.27 | 41.62 |
| Flux | UltraImage | 0.42 | 21.36 | 28.35 | 0.28 | 39.29 |
| Flux | SEGA | 1.05 | 22.50 | 29.26 | 0.28 | 42.36 |
| Qwen | Base | 0.13 | 21.33 | 28.89 | 0.27 | 30.90 |
| Qwen | DyPE | 1.13 | 22.58 | 29.55 | 0.29 | 39.10 |
| Qwen | UltraImage | 0.71 | 21.60 | 29.62 | 0.27 | 35.71 |
| Qwen | SEGA | 1.58 | 23.86 | 30.06 | 0.30 | 45.27 |
| Prop. (dagger) | Nano Banana 2 | 1.37 | 23.43 | 30.02 | 0.30 | 42.03 |

### Table 6 [v1, p. 19], 5120^2 on Aesthetic-4K, random 20 prompt-image pairs (App. F.3)
| Backbone | Method | IR | PS | CS | HPS | MSQ | CQA | FID |
|---|---|---|---|---|---|---|---|---|
| Flux | Base | -2.03 | 18.22 | 16.43 | 0.21 | 25.72 | 0.37 | 351.73 |
| Flux | NTK | -0.05 | 21.31 | 24.53 | 0.25 | 27.96 | 0.48 | 244.52 |
| Flux | YaRN | 0.40 | 21.53 | 26.74 | 0.26 | 40.11 | 0.61 | 235.33 |
| Flux | DyPE | 0.58 | 21.85 | 27.98 | 0.26 | 40.40 | 0.66 | 225.30 |
| Flux | UltraImage | 0.24 | 21.34 | 26.39 | 0.27 | 36.25 | 0.74 | 251.02 |
| Flux | SEGA | 1.13 | 23.22 | 28.92 | 0.29 | 40.64 | 0.72 | 221.99 |
| Qwen | Base | -0.54 | 20.08 | 25.52 | 0.24 | 21.90 | 0.48 | 270.94 |
| Qwen | DyPE | -0.37 | 20.34 | 24.72 | 0.24 | 20.58 | 0.41 | 250.45 |
| Qwen | UltraImage | 0.18 | 20.88 | 26.13 | 0.25 | 23.13 | 0.49 | 255.24 |
| Qwen | SEGA | 1.39 | 23.94 | 29.55 | 0.30 | 41.62 | 0.74 | 218.47 |

### Table 7 [v1, p. 19], 6144^2 on Aesthetic-4K, random 20 prompt-image pairs (App. F.3)
| Backbone | Method | IR | PS | CS | HPS | MSQ | CQA | FID |
|---|---|---|---|---|---|---|---|---|
| Flux | Base | -2.25 | 17.06 | 12.48 | 0.19 | 25.77 | 0.31 | 453.97 |
| Flux | NTK | -2.03 | 17.11 | 10.83 | 0.19 | 23.79 | 0.34 | 528.45 |
| Flux | YaRN | -0.33 | 20.18 | 24.16 | 0.24 | 35.28 | 0.56 | 288.66 |
| Flux | DyPE | -2.23 | 16.78 | 11.68 | 0.18 | 27.65 | 0.34 | 274.82 |
| Flux | UltraImage | -0.74 | 20.27 | 25.29 | 0.26 | 36.15 | 0.69 | 290.75 |
| Flux | SEGA | 0.75 | 22.47 | 27.92 | 0.27 | 43.43 | 0.65 | 232.18 |
| Qwen | Base | -0.95 | 19.78 | 23.96 | 0.23 | 22.50 | 0.44 | 279.78 |
| Qwen | DyPE | -0.88 | 19.64 | 23.75 | 0.23 | 36.15 | 0.35 | 279.77 |
| Qwen | UltraImage | -0.47 | 20.14 | 23.94 | 0.24 | 21.89 | 0.43 | 254.09 |
| Qwen | SEGA | 1.36 | 23.97 | 28.75 | 0.30 | 38.77 | 0.74 | 210.21 |

### Table 9 [v1, p. 21], m_ref form ablation on Flux (gamma = 1.5, kappa = 0.08)
| Resolution | m_ref form | CS | ImageReward | HPS | PickScore | FID | MUSIQ |
|---|---|---|---|---|---|---|---|
| 4096 x 4096 | logarithmic | 28.46 | 1.23 | 0.29 | 23.10 | 150.16 | 44.65 |
| 4096 x 4096 | power-law | 29.22 | 1.26 | 0.29 | 23.18 | 150.05 | 45.73 |
| 5120 x 5120 | logarithmic | 28.41 | 0.80 | 0.28 | 22.93 | 265.01 | 46.36 |
| 5120 x 5120 | power-law | 28.92 | 1.13 | 0.29 | 23.22 | 221.99 | 40.64 |
| 6144 x 6144 | logarithmic | 27.54 | 0.67 | 0.27 | 22.37 | 269.58 | 42.92 |
| 6144 x 6144 | power-law | 27.92 | 0.75 | 0.27 | 22.47 | 232.18 | 43.43 |

## Protocol summary

FLUX.1-Krea-dev, bf16, H100, "default" sampler/steps/guidance (code: 28 steps, guidance 4.5, mu fixed at
1.15, seed 0). Aesthetic-4K prompts plus reference images; split and count not stated for the main
tables; random 20 pairs at 5120^2 and 6144^2; separate 200-prompt LLM zero-shot set. Metrics IR, PS,
CS, MUSIQ, CLIP-IQA, HPSv2, FID, FIDp. No implementation stated for any metric and FIDp is undefined
anywhere in paper, TeX, code, or project page.

## Open questions / NOT STATED

- Aesthetic-4K split and number of prompts/images for Tables 1 to 4.
- Images per prompt; seeds for the main tables.
- FID implementation, Inception resize, number and resizing of reference images.
- FIDp crop size, crops per image, positions, resizing, reference crops (entirely undefined).
- MUSIQ / CLIP-IQA implementation and input resolution; CLIP backbone; HPS v2 vs v2.1.
- Baseline sampler settings and baseline checkpoint (Krea-dev vs dev) for HiFlow / I-Max / ScaleDiff.
- Whether the shift cap mu = 1.15 in the released code was used for the tables (UNVERIFIED).
