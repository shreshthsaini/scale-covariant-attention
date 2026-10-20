# SigMa protocol audit

Paper: "σ: Sigmoid Modulation for Ultra High Resolution Diffusion" (SigMa), Bingxuan Zhao, Qing Zhou, Yu Wang, Chuang Yang, Qi Wang. ICML 2026, PMLR 306:162993-163008.

## Sources and versions

- **PMLR PDF (used for every claim below).** Local copy `papers/sigma_pmlr/paper.pdf`, 16 pages, sha256 `d7ecce4b90e79d68abc07ad6ae84763b6a6f1be35949ea696d3baa97342cc9ae`, pdfTeX creation date 2026-05-27. Citation convention: "p.N" is printed page N of this PDF, which is proceedings page 162992+N.
- **arXiv version: none found.** The arXiv API title search `ti:"Sigmoid Modulation"` returned 0 results on 2026-10-20. The author search for "Bingxuan Zhao" lists only other papers, such as SHARP, 2603.21783v4, a remote-sensing paper. Two web searches also found no arXiv listing. A listing under a different title is UNVERIFIED. There are no arXiv numbers to compare against PMLR.
- **OpenReview forum 47JZSOkw5C: UNVERIFIED.** Both the API and the page are blocked by a browser challenge.
- **PMLR landing page:** links to the PDF, OpenReview and code. It lists no supplementary PDF.
- **Code: github.com/bxuanz/SigMa.** Shallow clone at HEAD `fbe3a863` ("add info", 2026-07-12), in `papers/sigma_repo`. The GitHub API shows 3 commits: `ae6b7411` "v1" (authored 2026-05-11, committed 2026-05-24), `94aa7ece` "Add SigMa Qwen support" (2026-06-30) and `fbe3a863`. The repo contains `README.md`, `run_sigma.py`, `run_sigma_qwen.py`, `flux/pipeline_flux.py`, `flux/transformer_flux.py`, `qwen/transformer_qwenimage.py`, `requirements.txt` and `docs/assets.png`. **There is no evaluation code, metric code, FID or crop code, or prompt list.** The code was read only, never executed.

## a. Base model(s) and checkpoint

- **Paper: FLUX.1-dev.**
  - Section 4 intro and Section 4.1 "Implementation Details" (p.6): "SigMa is deployed as a training-free adapter for FLUX.1-dev (Labs, 2024) with no per-resolution search."
  - The citation is "Labs, B. F. Flux. https://github.com/black-forest-labs/flux, 2024".
  - The Baseline row is "Direct inference with FLUX.1-dev" (Section 4.1, Baselines, p.7).
  - The Table 1 caption says "All methods are built on FLUX." (p.7).
  - Table 4 and Appendix A.4 probe FLUX.1-dev hidden states (p.13).
- **Second backbone: Qwen-Image** (Wu et al., 2025, arXiv:2508.02324), in Appendix C.2 and Table 6 (p.14-15).
- **Checkpoint revision or hash:** NOT STATED in the paper.
- **Released code uses FLUX.1-Krea-dev, not FLUX.1-dev, in every commit.**
  - In v1, `run_sigma.py` loads `'black-forest-labs/FLUX.1-Krea-dev'`.
  - From `94aa7ece` on, it loads `.../models--black-forest-labs--FLUX.1-Krea-dev/snapshots/8162a9c7b05a641be098422bf2fcf335615c2f28`.
  - The only other FLUX name in the repo is a FLUX.1-schnell docstring example in `pipeline_flux.py`. The string "FLUX.1-dev" appears nowhere.
  - The Qwen runner loads `.../models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6`.
- **UNVERIFIED:** which checkpoint produced the published numbers. The paper says FLUX.1-dev; the code says FLUX.1-Krea-dev.

## b. Benchmark prompt sets

Main benchmarks, from Section 4.1 "Benchmarks & Metrics" (p.7):

- **DrawBench** (Saharia et al., 2022), "(200 prompts)", "to assess compositional alignment". 200 is the whole DrawBench set, so the full set is used. Images per prompt: NOT STATED. Seeds: NOT STATED.
- **Aesthetic-4K**, quoted exactly: "the Aesthetic-Eval@4096 subset of Aesthetic-4K (Zhang et al., 2025) (195 pairs, downsampled references) for ultra-high-resolution fidelity".
  - 195 is the full official Aesthetic-Eval@4096 split. Diffusion-4K (arXiv 2503.18352v2, Section 3.1, p.4) says: "Within the evaluation set, 195 images have a short side greater than 4096, which we refer to as Aesthetic-Eval@4096."
  - **Subset rule:** the full official 4096 split. It is not a random subset or the first N.
  - Which captions serve as prompts: NOT STATED.
  - Images per prompt: NOT STATED. Seeds: NOT STATED.
  - The paper never says that the same 195 pairs are reused at 3072 × 2048, 2048 × 3072 and 3072². The tables only say "Aesthetic-4K".

Other prompt usage:

- **Table 3 (4096 × 1024):** prompt set NOT STATED (Section 4.4, p.9).
- **Table 4 (feature-STD probe):** "evaluate 100 Aesthetic-4K prompts" (Appendix A.4, p.13). How the 100 were chosen: NOT STATED.
- **Figure 7 gallery:** "all prompts were sampled from the Aesthetic-4K test set and DrawBench" (p.16).

Not stated in the paper, but present in the code:

- The FLUX runner default seed is 42 (`run_sigma.py`). It runs one prompt per call and has no batch prompt file.
- The Qwen runner (`run_sigma_qwen.py`) has a `--prompt_file` option (one prompt per line, `--start_idx/--end_idx`) and default seed 0. It re-seeds that same seed for every prompt. It appends ", Ultra HD, 4K, cinematic composition." to each prompt and uses negative prompt " ".

## c. Resolutions (exactly as written)

- **Table 1** (p.7): "Landscape Extrapolation (3072 × 2048)" and "Square Extrapolation (4× Scale, 4096²)".
- **Table 5** (p.14): "Portrait Extrapolation (2048 × 3072)" and "Square Extrapolation (3× Scale, 3072²)".
- **Table 3** (p.9): "4096 × 1024" (aspect ratio 4:1).
- **Table 6** (p.15): Qwen-Image "4096 × 4096", "Starting from the 1024 × 1024 base resolution".
- **Table 4** (p.13): 1024 × 1024, 2048 × 1024, 3072 × 1024, 4096 × 1024.
- **Abstract** (p.1): "up to 16 megapixels".
- **Section 4.1** (p.6): "from standard squares (3072², 4096²) to panoramic layouts (4 : 1)".
- **Convention is W × H.** 3072 × 2048 is called "Landscape". Section 3.2 (p.6) says "For a 4096 × 1024 image (sw = 4, sh = 1)". Table 4 says "The height is fixed at 1024 while the width scales from 1024 to 4096".
- **Base resolution:** the extrapolation factors (4× for 4096², 3× for 3072²) imply a 1024 base. The code sets `base_resolution = 1024` and `patch_size = 16`, so the base is 64 tokens per axis.
- **Qwen base differs between paper and code.** `qwen/transformer_qwenimage.py` sets `base_resolution = 1328`, which gives a factor of about 3.08 at 4096. The paper says 1024 × 1024.

## d. Sampler settings

**SigMa, per the paper** (Section 4.1, p.6):

- Quoted: "We utilize an Euler-flow solver (28 steps, guidance 3.5) across a spectrum of resolutions". Also: "with no per-resolution search".
- SigMa "adds only scalar schedule updates during sampling" (p.7).
- Schedule (Eq. 10 and Corollary 3.4, p.5-6): µd(t) = σ(√sd [ℓ(t) − ℓ(1/sd)]), with tc = 1/sd and γ = √sd.

**Not stated in the paper:**

- Scheduler time shift: NOT STATED.
- Precision: NOT STATED.
- Negative prompt and true CFG for FLUX: NOT STATED.
- Qwen-Image sampler (steps, CFG): NOT STATED.

**Baselines, per the paper** (Section 4.1, Baselines, p.7):

- Baseline: direct FLUX.1-dev inference.
- Static: NTK-aware (bloc97, 2023) and YaRN (Peng et al., 2024). Their parameters (β thresholds, attention temperature) are NOT STATED. Section 2.2 describes them only generically.
- DyPE (Issachar et al., 2025, arXiv:2510.20766), quoted: "µDyPE(t) = λs · t^λt. We set λs = 2, λt = 2 (quadratic decay)".
- Whether every baseline uses the same 28 steps and guidance 3.5: implied by the shared setup sentence, but not stated per baseline.
- The DyPE code version: NOT STATED.

**Released code** (behavior read from source, not stated in the paper):

- FLUX sampling uses `guidance_scale=4.5` (not 3.5), 28 steps by default, `torch.bfloat16` and `enable_model_cpu_offload()`.
- **Time shift.** `flux/pipeline_flux.py` lines 851-858 replace `max_image_seq_len` (4096) with `image_seq_len` in `calculate_shift`. For any token count above 256, mu therefore equals `max_shift`. The local FLUX.1-dev and FLUX.1-Krea-dev snapshots (Krea snapshot `8162a9c7...`, the same as the repo path) both have `base_shift 0.5`, `max_shift 1.15`, `use_dynamic_shifting true` and `FlowMatchEulerDiscreteScheduler`. So mu = 1.15 and exp(mu) ≈ 3.16 at every resolution. The same edit is already present in v1.
- **YaRN constants** in `flux/transformer_flux.py`: beta_0 = 1.25, beta_1 = 0.75, gamma_0 = 16, gamma_1 = 2, and attention mscale = 0.1 ln(s) + 1.
  - With SigMa on, the betas, the gammas and (mscale − 1) are multiplied by µ(t), which matches Eq. 21.
  - In v1 the betas were instead raised to the power log2(s)·µ(t), and the gammas to gamma_factor·µ(t). The multiplicative form arrived in commit `94aa7ece` (2026-06-30), after the PDF was built (2026-05-27).
- **NTK:** the factor is s^(D/(D−2)), raised to µ(t) when SigMa is on (Eq. 20).
- **Timestep:** t is the scheduler sigma in (0, 1], clamped to [1e-6, 1 − 1e-6].
- **Modes:** `--no_sigma` gives static YaRN or NTK; `--method base` gives native RoPE.
- **No DyPE implementation** is in the repo.
- **Qwen runner:** 40 steps, `true_cfg_scale=4.0`, bf16, VAE tiling. It offers only native or SigMa-YaRN, so there is no static YaRN or DyPE option for Qwen.
- **UNVERIFIED:** which code version produced the paper's numbers.

## e. Metrics and implementation details

Metric list (Section 4.1, p.7, and the Table 1 caption):

- DrawBench: CLIPScore, Aesthetic Score and HPSv2.
- Aesthetic-4K: the same three, "supplemented by FID".

| Metric | What the paper states (p.7 unless noted) | Not stated |
|---|---|---|
| CLIP | "CLIP-Score (Hessel et al., 2022)". Printed values run from 17.80 to 29.48, which looks like a ×100 scale. | Backbone, scale, input resize |
| Aesthetic | "Aesthetic Score (Schuhmann et al., 2022)", which cites the LAION-5B paper. Values run from 3.59 to 6.88. | Predictor version, CLIP backbone, input resize |
| HPS | Text: "prioritizing HPSv2.1 (Wu et al., 2023)". Tables label it "HPSv2" (Tables 1, 2, 3, 6) or "HPS" (Table 5). Values run from 0.118 to 0.297, a raw-score scale. | Exact version used in the tables (v2 vs v2.1), input size |
| FID | "FID (Heusel et al., 2017) is reported for completeness despite sample size limitations." The reference set is Aesthetic-Eval@4096, "195 pairs, downsampled references". | Implementation (clean-fid, torchmetrics, pytorch-fid or torch-fidelity), resize to 299 and its interpolation, the size the references are downsampled to and the method, whether generated images are resized, sample counts (presumably 195 vs 195, but not stated) |

Metrics that do not appear in the paper at all (case-sensitive grep):

- Patch, crop or FIDp FID: not reported, so crop size, crop count, positions and reference crops do not apply.
- KID, IS, ImageReward, PickScore, MUSIQ, MANIQA, CLIP-IQA, ARNIQA, Q-Align, GLCM, compression ratio, user study, human evaluation.

Context only, not SigMa's own statement: DyPE (arXiv 2510.20766) describes Aesthetic-Eval@4096 as "195 curated image–prompt pairs" and says it downsamples them "to match the target test resolutions". SigMa's phrase "downsampled references" looks like the same procedure, but SigMa does not say so.

## f. Latency, runtime, hardware

None reported. The GPU type, latency, memory and throughput are all NOT STATED. The only cost claim is that SigMa "adds only scalar schedule updates during sampling" (p.7). The README says "without retraining or extra sampling cost".

## g. Tables, transcribed exactly

**How the tables were checked:**

- Cells were read twice, independently: from `paper.txt` and from a PyMuPDF position-based extraction. The two agree on every number.
- **Bold**, shown here as `**x**`, was recovered from the PDF font. Bold cells use NimbusRomNo9L-Medi; regular cells use NimbusRomNo9L-Regu. Bold is not visible in the text extraction.
- No underline strokes were detected under any numeric cell.
- Only the Table 5 caption defines bold ("Bold indicates best in group"). Tables 1, 2, 3 and 6 use bold but do not define it.

### Table 1 (p.7)

Caption: "Quantitative Results on High-Resolution Synthesis. Performance comparison on DrawBench and Aesthetic-4K across two resolutions per row. We report CLIPScore (CLIP), HPSv2 (HPS), and Aesthetic Score (Aes) for DrawBench, supplemented by FID for Aesthetic-4K. All methods are built on FLUX."

The column groups are "Landscape Extrapolation (3072 × 2048)" and "Square Extrapolation (4× Scale, 4096²)". Each has a DrawBench sub-group (CLIP↑ Aes↑ HPSv2↑) and an Aesthetic-4K sub-group (CLIP↑ Aes↑ HPSv2↑ FID↓). The row groups are "Baseline", "NTK-based Methods" and "YaRN-based Methods". Column prefixes: L = 3072 × 2048, S = 4096², DB = DrawBench, A4K = Aesthetic-4K.

| Method | L DB CLIP↑ | L DB Aes↑ | L DB HPSv2↑ | L A4K CLIP↑ | L A4K Aes↑ | L A4K HPSv2↑ | L A4K FID↓ | S DB CLIP↑ | S DB Aes↑ | S DB HPSv2↑ | S A4K CLIP↑ | S A4K Aes↑ | S A4K HPSv2↑ | S A4K FID↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Baseline* | | | | | | | | | | | | | | |
| FLUX.1-dev | 26.43 | 5.32 | 0.241 | 28.61 | 6.54 | 0.267 | 152.32 | 19.99 | 3.81 | 0.139 | 25.70 | 5.42 | 0.200 | 197.21 |
| *NTK-based Methods* | | | | | | | | | | | | | | |
| NTK-aware | 27.45 | 5.52 | 0.253 | 28.86 | 6.62 | 0.275 | 150.11 | 17.80 | 3.59 | 0.118 | 25.35 | 5.62 | 0.197 | 204.56 |
| DyPE-NTK | 27.67 | 5.66 | 0.261 | 29.02 | 6.69 | **0.281** | 149.77 | 22.23 | 4.43 | 0.177 | 27.73 | 6.38 | 0.254 | **184.99** |
| **SigMa-NTK** | **27.69** | **5.71** | **0.264** | **29.10** | **6.72** | **0.281** | **148.76** | **22.29** | **4.46** | **0.179** | **27.77** | **6.41** | **0.257** | 185.35 |
| *YaRN-based Methods* | | | | | | | | | | | | | | |
| YaRN | 27.47 | 5.67 | 0.264 | 29.19 | 6.79 | 0.282 | 151.21 | 25.78 | 4.98 | 0.222 | 28.11 | 6.40 | 0.261 | 194.76 |
| DyPE-YaRN | 27.89 | 5.73 | 0.270 | 29.38 | 6.82 | 0.287 | 150.33 | 26.88 | 5.24 | 0.244 | 28.72 | 6.63 | 0.279 | 188.53 |
| **SigMa-YaRN** | **27.91** | **5.76** | **0.271** | **29.46** | **6.88** | **0.288** | **149.98** | **26.90** | **5.36** | **0.247** | **29.23** | **6.71** | **0.287** | **184.73** |

**FLUX at 4096 × 4096 on Aesthetic-4K, extracted from Table 1:**

| Method | CLIP | Aes | HPSv2 | FID |
|---|---|---|---|---|
| FLUX.1-dev | 25.70 | 5.42 | 0.200 | 197.21 |
| NTK-aware | 25.35 | 5.62 | 0.197 | 204.56 |
| DyPE-NTK | 27.73 | 6.38 | 0.254 | 184.99 |
| SigMa-NTK | 27.77 | 6.41 | 0.257 | 185.35 |
| YaRN | 28.11 | 6.40 | 0.261 | 194.76 |
| DyPE-YaRN | 28.72 | 6.63 | 0.279 | 188.53 |
| SigMa-YaRN | 29.23 | 6.71 | 0.287 | 184.73 |

### Table 2 (p.8)

Caption: "Ablation of Scaling Laws on Aesthetic-4K (4096²). We conduct a controlled ablation to isolate the impact of Geometric Center (tc) and Dynamic Rate (γ) within the YaRN framework." It is FLUX, Aesthetic-4K at 4096². The "Ours" columns equal the SigMa-YaRN 4096² Aesthetic-4K row of Table 1.

(a) Impact of Geometric Center (tc)

| Metric | Const (tc = 1) | Late (1/s²) | Early (1/√s) | Ours (1/s) |
|---|---|---|---|---|
| CLIP ↑ | 28.05 | 27.42 | 28.89 | **29.23** |
| Aes. ↑ | 6.15 | 6.01 | 6.52 | **6.71** |
| HPSv2 ↑ | 0.248 | 0.241 | 0.274 | **0.287** |
| FID ↓ | 199.50 | 208.30 | 189.10 | **184.73** |

(b) Impact of Dynamic Rate (γ)

| Metric | Const (γ = 1) | Inv (1/s) | Lin (s) | Ours (√s) |
|---|---|---|---|---|
| CLIP ↑ | 28.65 | 27.60 | 28.75 | **29.23** |
| Aes. ↑ | 6.38 | 6.08 | 6.45 | **6.71** |
| HPSv2 ↑ | 0.262 | 0.245 | 0.269 | **0.287** |
| FID ↓ | 194.20 | 205.10 | 191.40 | **184.73** |

The Figure 5 heatmap labels the same tc candidates "1/s² (Fast)" and "1/√s (Slow)", while Table 2 calls them "Late" and "Early".

### Table 3 (p.9)

Caption: "Evaluation on Extreme Aspect Ratios (4096 × 1024). This 4:1 regime stress-tests the "geometric repetition" failure mode of scalar-based methods. See Appendix C.3 for visual evidence." The benchmark is NOT STATED, and there is no FID.

| Method | CLIP Score↑ | Aesthetic↑ | HPSv2↑ |
|---|---|---|---|
| YaRN | 26.15 | 5.42 | 0.208 |
| DyPE-YaRN | 26.92 | 5.85 | 0.231 |
| **SigMa (Ours)** | **27.45** | **6.28** | **0.254** |

### Table 5 (p.14, Appendix C.1)

Caption: "Additional Quantitative Evaluation. Complementing the main text, we provide results for Portrait Extrapolation (2048×3072) and intermediate 3× Square Super-Resolution (3072²). Consistent with the main results, SigMa maintains strong overall performance across both additional geometries. Bold indicates best in group."

The columns are laid out as in Table 1, with the HPS header printed as "HPS↑". Column prefixes: P = Portrait Extrapolation (2048 × 3072), S3 = Square Extrapolation (3× Scale, 3072²).

| Method | P DB CLIP↑ | P DB Aes↑ | P DB HPS↑ | P A4K CLIP↑ | P A4K Aes↑ | P A4K HPS↑ | P A4K FID↓ | S3 DB CLIP↑ | S3 DB Aes↑ | S3 DB HPS↑ | S3 A4K CLIP↑ | S3 A4K Aes↑ | S3 A4K HPS↑ | S3 A4K FID↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Baseline* | | | | | | | | | | | | | | |
| FLUX.1-dev | 26.58 | 5.63 | 0.253 | 28.74 | 6.33 | 0.262 | 188.45 | 24.98 | 4.99 | 0.217 | 27.92 | 6.16 | 0.253 | 189.21 |
| *NTK-based Methods* | | | | | | | | | | | | | | |
| NTK-aware | 27.45 | 5.54 | 0.260 | 29.17 | 6.60 | 0.273 | 186.98 | 25.73 | 5.20 | 0.231 | 28.71 | 6.42 | 0.271 | 184.72 |
| DyPE-NTK | 27.42 | 5.60 | 0.265 | 28.94 | 6.66 | 0.275 | 181.23 | 26.38 | 5.42 | 0.248 | 28.43 | 6.51 | 0.278 | 180.18 |
| **SigMa-NTK** | **27.77** | **5.63** | **0.266** | **29.20** | **6.73** | **0.281** | **179.92** | **26.41** | **5.45** | **0.249** | **28.47** | **6.57** | **0.279** | **179.13** |
| *YaRN-based Methods* | | | | | | | | | | | | | | |
| YaRN | 27.47 | 5.72 | 0.270 | 28.95 | 6.72 | 0.280 | 181.12 | 27.35 | 5.49 | 0.255 | 29.02 | 6.62 | 0.283 | 183.93 |
| DyPE-YaRN | 27.92 | 5.81 | 0.277 | 29.26 | 6.76 | 0.282 | **180.11** | 27.77 | 5.60 | 0.267 | 29.32 | 6.69 | **0.297** | 182.71 |
| **SigMa-YaRN** | **27.96** | **5.85** | **0.279** | **29.31** | **6.84** | **0.285** | 181.27 | **28.05** | **5.62** | **0.268** | **29.48** | **6.78** | 0.294 | **181.88** |

**Bolding error:** in the 3072² Aesthetic-4K CLIP column, SigMa-NTK's 28.47 is bold, but NTK-aware scores 28.71 in the same group.

### Table 6 (p.15, Appendix C.2, Qwen-Image, not FLUX)

Caption: "Evaluation on Qwen-Image (4096 × 4096). Starting from the 1024 × 1024 base resolution, we extrapolate to 4096 × 4096 on Aesthetic-4K and compare the native Qwen-Image baseline, YaRN, DyPE-YaRN, and SigMa-YaRN."

| Method | CLIP↑ | Aes.↑ | HPSv2↑ | FID↓ |
|---|---|---|---|---|
| Qwen-Image baseline | 27.42 | 5.61 | 0.212 | 200.73 |
| YaRN | 28.73 | 6.24 | 0.239 | 198.67 |
| DyPE-YaRN | 28.81 | 6.33 | 0.277 | 197.01 |
| **SigMa-YaRN (Ours)** | **28.93** | **6.36** | **0.281** | **196.69** |

### Table 4 (p.13, Appendix A, FLUX.1-dev analysis, not a benchmark)

Caption (abridged): "We extract Step-4 hidden states from FLUX.1-dev, average them over all attention heads in the middle layers, and evaluate 100 Aesthetic-4K prompts."

| Scale Factor (sw) | Target Resolution | Theoretical Multiplier (√sw) | Normalized Empirical Feature STD |
|---|---|---|---|
| 1× | 1024 × 1024 | 1.000 = 1 × √1 | 1.000 |
| 2× | 2048 × 1024 | 1.414 = 1 × √2 | 1.398 |
| 3× | 3072 × 1024 | 1.732 = 1 × √3 | 1.715 |
| 4× | 4096 × 1024 | 2.000 = 1 × √4 | 1.972 |

### Consistency checks

- **Prose matches the tables.** The text figures "HPSv2 score of 0.139 on DrawBench" (p.7), "FID: 208.30 vs. 199.50", "CLIP: 29.23", "184.73", "205.10", "0.287" (p.8), and "0.208", "0.231", "5.85" (p.9) all match the tables.
- **Baselines were not copied from DyPE.** None of SigMa's Table 1 baseline values (152.32, 150.33, 188.53, 194.76, 197.21, 184.99, 204.56, 26.88, 28.72) appear in the text of the DyPE arXiv paper (2510.20766). DyPE also does not report HPSv2.

## Protocol summary

- **Model and sampler:** FLUX.1-dev per the paper, with a 28-step Euler flow solver and guidance 3.5. The time shift and precision are not stated.
- **Benchmarks:** DrawBench (all 200 prompts) for CLIP, Aes and HPS. The full official Aesthetic-Eval@4096 split (195 pairs, "downsampled references") for CLIP, Aes, HPS and FID.
- **Resolutions:** 3072 × 2048 and 4096² in Table 1; 2048 × 3072 and 3072² in Table 5; 4096 × 1024 in Table 3 (no FID, prompt set not stated).
- **FID details:** none given. There is no patch or crop FID, no KID, IS or IQA metrics, no user study, and no latency.
- **Released code differs from the paper.** It uses FLUX.1-Krea-dev with embedded guidance 4.5, bf16, and a dynamic shift pinned to mu = 1.15. It contains no evaluation code and no DyPE baseline.

## Open questions / NOT STATED / UNVERIFIED

1. Which checkpoint produced the numbers: FLUX.1-dev (paper) or FLUX.1-Krea-dev (code)? Guidance 3.5 (paper) or 4.5 (code)? UNVERIFIED.
2. Scheduler time shift: NOT STATED. The code pins mu to 1.15 at every resolution.
3. Precision: NOT STATED (bf16 in the code). Seeds and images per prompt: NOT STATED.
4. Which Aesthetic-Eval@4096 captions serve as prompts: NOT STATED. Whether the same 195 pairs are used at 3072 × 2048, 2048 × 3072 and 3072²: NOT STATED.
5. FID: the implementation, the Inception resize and interpolation, the size the references are downsampled to, the resize method, and the sample counts are all NOT STATED.
6. Backbones and versions for CLIP, the aesthetic predictor and HPS (v2 vs v2.1, given the text/table mismatch): NOT STATED. Metric input resolution: NOT STATED.
7. Prompt set for Table 3 (4096 × 1024): NOT STATED. Selection of the 100 prompts for Table 4: NOT STATED.
8. Baseline hyperparameters for NTK-aware and YaRN, and the DyPE code version: NOT STATED. Code constants are listed in Section d.
9. Qwen-Image sampler: NOT STATED. Base resolution: the paper says 1024, the code says 1328.
10. GPU, latency and memory: NOT STATED.
11. Which code commit produced the numbers: UNVERIFIED. The YaRN gating rule changed on 2026-06-30, after the PDF build date.
12. OpenReview version: UNVERIFIED (access blocked). arXiv version: none found, UNVERIFIED beyond the title and author searches.
