# ResDiT protocol audit

Paper: "ResDiT: Evoking the Intrinsic Resolution Scalability in Diffusion Transformers", Ma et al. arXiv 2512.01426v1 (only version; submitted 1 Dec 2025; arXiv comment "8 pages").
Local copy: papers/2512.01426/paper.pdf, paper.txt (13 pages: 8 main, references, appendix A/B on pages 11 to 13).
Code: none. The paper gives no code or project link (its only github.com URL is the FLUX reference [19]). A GitHub repository search for "ResDiT" (API, 2026-10-20) returned no official repository. UNVERIFIED.

## a. Base models and checkpoint

- "If not specified, we deploy ResDiT on FLUX.1-dev [19], an advanced open-source model based on DiT architecture" (Sec. 4.1, p.6). No other base model is evaluated quantitatively. FLUX.1-Krea-dev: not used.

## b. Prompt sets

- "We collect 500 high-quality captions and generate images corresponding to each caption" (Sec. 4.1 Evaluation, p.7).
- Source of the 500 captions: NOT STATED. Selection rule beyond "high-quality": NOT STATED. Prompt list: not released.
- Aesthetic-4K: NOT USED (no mention). HRGen-1K, DrawBench, GenEval, DPG, COCO: NOT USED. LAION appears only as the KID reference set.
- Images per prompt: "generate images corresponding to each caption" (count NOT STATED). Seeds: NOT STATED.
- User study: "Twenty participants independently rated images on a scale from 1 to 5 for their image visual quality based on 40 randomly selected prompts per method" (Sec. 4.1, p.7).

## c. Resolutions

- Table 1: 3072 x 3072 and 4096 x 4096 (p.6). Qualitative: 2048 x 4096 and 4096 x 2048 (Fig. 5 caption, p.7); HxW vs WxH convention for these NOT STATED.

## d. Sampler settings

- ResDiT: "The sampling steps are set to 35, and the guidance scale is set to 3.5. Following prior work [25, 26], we set the normalized frequency cutoff of the spectral mask to 0.2" (Sec. 4.1, p.6). [25] FreeLong++, [26] FreeLong.
- Stage schedule (single-pass, no base-resolution image): "we use a global branch for the first 10 timesteps, a local branch for the last 15 timesteps, and Patch-wise Spectral Fusion for the remaining steps" (Sec. 4.1, p.7). With 35 steps, PSF covers the middle 10.
- Baselines: DemoFusion [8], DiffuseHigh [18], I-Max [9], HiFlow [4]; "They all adopt a two-stage paradigm: first generate a base-resolution image and then perform high-resolution extrapolation. All methods are evaluated using their official implementations" (Sec. 4.1, p.7). Baseline steps / guidance: NOT STATED (official defaults implied).
- Scheduler / shift: NOT STATED. Precision: NOT STATED.

## e. Metrics and implementation

- Metrics: KID [2], IS [34], CLIP Score [30], patch KID (KIDp), patch IS (ISp), user study (Sec. 4.1, p.7; Table 1). FID is NOT reported at all.
- KID reference: "The KID is calculated between generated images and 2K real high-quality images sourced from LAION-Aesthetics-v2 6.5plus [35]" (Sec. 4.1, p.7). How the 2K references were picked: NOT STATED.
- Patch metrics: "we also adopted patch KID and patch IS as our evaluation metrics" (p.7). Crop size, crops per image, random or fixed positions, resizing, reference patches: all NOT STATED.
- KID / IS implementation, resize to 299, interpolation, KID subset size: NOT STATED.
- CLIP Score backbone: NOT STATED (0 to 100 scale, for example 32.71).
- No ImageReward, HPS, PickScore, MUSIQ, MANIQA, CLIP-IQA, ARNIQA, Q-Align, aesthetic score, GLCM, or compression ratio.

## f. Latency hardware

- "All experiments are conducted on RTX 4090 GPU" (Sec. 4.1, p.7). No latency numbers are reported.

## g. Tables (as printed)

### Table 1 (p.6): "Quantitative comparisons with baselines. The best results are highlighted in bold, and the second-best results are underlined. Compared to approaches that rely on base-resolution images for high-resolution generation, ResDiT achieves competitive, near state-of-the-art performance at 3072 x 3072 resolution. A slight performance drop is observed at 4096 x 4096, which is further analyzed in the experimental section."

All rows are FLUX.1-dev.

| Resolution | Method | KID down | KIDp down | IS up | ISp up | CLIP up | User Study up |
|---|---|---|---|---|---|---|---|
| 3072 x 3072 | Demofusion [8] | 0.0211 | 0.0342 | 12.20 | 10.21 | 31.92 | 3.1 |
| 3072 x 3072 | DiffuseHigh [18] | 0.0195 | 0.0213 | 12.61 | 10.13 | 32.74 | 4.2 |
| 3072 x 3072 | I-Max [9] | 0.0192 | 0.0207 | 12.96 | 10.48 | 32.73 | 4.2 |
| 3072 x 3072 | HiFlow [4] | 0.0190 | 0.0194 | 12.87 | 10.67 | 32.76 | 4.6 |
| 3072 x 3072 | ResDiT(Ours) | 0.0189 | 0.0199 | 12.91 | 10.87 | 32.85 | 4.8 |
| **4096 x 4096** | **Demofusion [8]** | **0.0236** | **0.0374** | **10.56** | **9.23** | **30.41** | **2.1** |
| **4096 x 4096** | **DiffuseHigh [18]** | **0.0215** | **0.0298** | **11.40** | **9.81** | **32.70** | **3.9** |
| **4096 x 4096** | **I-Max [9]** | **0.0208** | **0.0275** | **11.79** | **9.95** | **32.69** | **4.0** |
| **4096 x 4096** | **HiFlow [4]** | **0.0203** | **0.0245** | **11.65** | **10.12** | **32.74** | **4.3** |
| **4096 x 4096** | **ResDiT(Ours)** | **0.0217** | **0.0252** | **11.46** | **9.97** | **32.71** | **4.3** |

Bold/underline checked on the rendered page (papers/2512.01426/table1.png). At 4096 x 4096, the PDF bolds HiFlow on KID 0.0203, KIDp 0.0245, ISp 10.12, CLIP 32.74, I-Max on IS 11.79, and HiFlow and ResDiT tied on User Study 4.3. ResDiT is underlined (second) on KIDp 0.0252, ISp 9.97, CLIP 32.71. So at 4096 x 4096 ResDiT is not best on any automatic metric; the paper attributes this to single-stage generation drifting from the base model's distribution (Sec. 4.2, p.7).

### Table 2 (Appendix A, p.11): "Quantitative ablation study of PE Scaling (PES) / Patch-wise Independent PE (PIPE) / Patch-wise Spectral Fusion (PSF) components at different resolutions. The best results are marked in bold."

| Resolution | PES | PIPE | PSF | KID | KIDp | IS | ISp | CLIP |
|---|---|---|---|---|---|---|---|---|
| 3072x3072 | x | x | x | 0.0836 | 0.1958 | 7.80 | 4.06 | 25.08 |
| 3072x3072 | x | v | v | 0.0906 | 0.0842 | 11.05 | 6.97 | 30.72 |
| 3072x3072 | v | x | v | 0.2218 | 0.4426 | 8.54 | 3.03 | 24.17 |
| 3072x3072 | v | v | x | 0.0227 | 0.0336 | 12.04 | 9.84 | 32.51 |
| 3072x3072 | v | v | v | 0.0189 | 0.0199 | 12.91 | 10.87 | 32.85 |
| 4096x4096 | x | x | x | 0.2703 | 0.3164 | 6.9 | 3.53 | 21.28 |
| 4096x4096 | x | v | v | 0.1334 | 0.1239 | 10.16 | 6.56 | 26.63 |
| 4096x4096 | v | x | v | 0.2208 | 0.3976 | 8.56 | 3.77 | 24.69 |
| 4096x4096 | v | v | x | 0.0488 | 0.0760 | 10.35 | 8.10 | 31.03 |
| 4096x4096 | v | v | v | 0.0217 | 0.0252 | 11.46 | 9.97 | 32.71 |

(v = check mark, x = cross, as printed. The all-off row is direct FLUX.1-dev generation: "the performance without PIPE is even worse than that of direct generation", Appendix A.)

## Notes for the SCA comparison

- ResDiT reports no FID and no Aesthetic-4K results. Its 4096 x 4096 FLUX numbers are KID/IS against 2K LAION-Aesthetics-v2 6.5plus images on 500 captions of unstated source.
