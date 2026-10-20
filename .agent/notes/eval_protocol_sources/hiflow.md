# HiFlow: evaluation protocol audit

Sources read (fetched 2026-10-20, read only, nothing executed):

- A = "HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned Guidance", Bu et al., arXiv 2504.06232v2 (16 May 2025, latest arXiv version), 27 PDF pages. Local: papers/2504.06232/.
- N = NeurIPS 2025 camera-ready from papers.neurips.cc (hash d4ecde5d18d36150c529d1b9c5f0d727), 35 PDF pages including the checklist. Local: papers/hiflow_neurips/paper.pdf.
- G = github.com/Bujiazi/HiFlow at main, commit f397a1f (2025-09-18). Read README.md, inference.sh, run_hiflow.py. The repo has **no evaluation scripts and no prompt list**. Files: flux_pipeline_hiflow.py, transformer_flux.py, utils.py, run_hiflow.py, inference.sh, requirements.txt, assets.
- The OpenReview forum (id koG76YqOwo) was blocked by a bot challenge, so the reviews and rebuttal are UNVERIFIED.

## a. Base model / checkpoint

- "If not specified, the generated images are based on Flux.1.0-dev [26]" (A Section 4.1, p.7; same in N). Ref [26] = Black Forest Labs Flux GitHub. G loads `black-forest-labs/FLUX.1-dev` with `torch_dtype=torch.float16` (run_hiflow.py l.9-12). The paper does not state precision.
- Baselines (A Section 4.1): training-free DemoFusion, DiffuseHigh, I-Max; training-based UltraPixel, Diffusion-4k; super-resolution BSRGAN. "For a fair comparison, the training-free methods are tested on the same Flux model according to their official implementations." `*` = "methods adapted from U-Net architecture" (DemoFusion*, DiffuseHigh*). **N drops the training-based baselines** (UltraPixel, Diffusion-4k rows removed from Table 1 and from the Baselines sentence). All other numbers are unchanged.

## b. Prompt set

- "We collect 1K high-quality captions across various scenarios for diverse image generation." (A Section 4.1 "Evaluation", p.7; N identical.) Source, selection rule, release: NOT STATED, and the prompts are not in G. Not a named benchmark: no LAION/COCO/DrawBench prompt set for generation.
- Images per prompt: NOT STATED. Seeds for the quantitative eval: NOT STATED. The appendix shows qualitative seeds 0/1/2 (A App. B, Fig. F.9; N Fig. H.10). G uses seed 3407.
- FID reference set: "FID is calculated between generated images and 10K real high-quality images (with at least 1024 x 1024 resolution) sourced from LAION-High-Resolution [44]" (A Section 4.1; [44] = LAION-5B paper). How the 10K were sampled, and their resize: NOT STATED. Generated count is presumably 1K (one per caption), but this is not stated explicitly.

## c. Resolutions

2048 x 2048 and 4096 x 4096 (Table 1, Tables 2-3). Both are cascades from 1024. The paper cascade for 4K is 1K -> 2K -> 3K -> 4K (Section 4.1). Qualitative figures also show 4096x2048, 2048x4096 and 3072x4096 (Fig. 1).

## d. Sampler settings

- Paper (A Section 4.1, p.7; N identical): "The sampling steps are set as 30, the noise-adding ratio tau in initialization alignment is set as [0.6, 0.3, 0.3] for 1K -> 2K -> 3K -> 4K cascade generation, and the normalized cutoff frequency is set as D = 0.4. The guidance weights in direction/acceleration alignment are set as alpha_t = beta_t = t/tau." Guidance scale (CFG / distilled guidance): NOT STATED. Scheduler/shift: NOT STATED in the main text. App. A: "NTK-aware scaled RoPE, balancing the entropy shift of self-attention and balancing the image/text sequence length ratio for MMDiT", taken from I-Max [11].
- G defaults (run_hiflow.py l.24-48) differ from the paper. 1K stage: guidance_scale 3.5, 30 steps, max_sequence_length 512. High-res stages `target_heights=[2048, 4096]` (1K -> 2K -> 4K, not 1K -> 2K -> 3K -> 4K), `num_inference_steps_highres=[16, 10]`, `guidance_scale_highres=[4.5, 6]`, `filter_ratio=[0.2, 0.2]`, `alphas=[1.0, 0.25]`, `betas=[0.5, 0.5]`, `ntk_factor=[10.0, 10.0]`, `proportional_attention=True`, `text_duplication=False`, `swin_pachify=True`, `structure_guidance="fft"`, `upsampling_choice="latent"`, an optional LoRA loaded (aidmaFLUXPro1.1). Which settings produced Table 1 is UNVERIFIED.

## e. Metrics

- "The CLIP [39] score is used to assess the prompt-following capability, Frechet Inception Distance [16] (FID) and Inception Score [43] (IS) are reported to measure image quality ... Furthermore, the patch-version FIDpatch and ISpatch are calculated based on local image patches to quantify the quality of synthesized details." (A Section 4.1, p.7.) That sentence is the entire protocol.
- FID implementation (clean-fid / torchmetrics / pytorch-fid / torch-fidelity): NOT STATED. Resize to 299 and how the 10K reference images are resized: NOT STATED.
- FIDpatch / ISpatch: crop size, crops per image, random vs fixed positions, crop resize, and which reference patches: **all NOT STATED** in A or N. No eval code in G.
- CLIP score backbone: NOT STATED (cites Radford et al. CLIP only).
- KID, ImageReward, HPS, PickScore, MUSIQ, MANIQA, CLIP-IQA, aesthetic predictor, GLCM, compression ratio: not used.
- User study / GPT-4o preference: none. The N checklist item 14 says "This paper does not involve crowdsourcing nor research with human subjects". Error bars: none (N checklist item 7: "[No] ... variance analysis is not included").
- Aesthetic-4K: **not used** anywhere. Diffusion-4k appears only as a baseline model, evaluated on HiFlow's own 1K captions.

## f. Latency hardware

"All experiments are conducted on a single NVIDIA A100 GPU." (A Section 4.1.) Table 2 reports latency in seconds per image. Batch size and whether it is end-to-end including the 1K stage are NOT STATED.

## g. Table transcriptions (exact, as printed)

Table 1 (A, p.7): Quantitative comparison with other baselines. The best result is highlighted in bold, while the second-best result is underlined. * indicates methods adapted from U-Net architecture. The paper says the training-free methods run "on the same Flux model". The checkpoints used for UltraPixel and Diffusion-4k are NOT STATED. Diffusion-4k's public checkpoint is a fine-tune of FLUX.1-dev.

| Resolution (height x width) | Method | FID ↓ | FIDpatch ↓ | IS ↑ | ISpatch ↑ | CLIP Score ↑ |
|---|---|---|---|---|---|---|
| 2048 x 2048 (2K) | DemoFusion* [10] | 56.07 | 51.69 | 27.23 | 13.48 | 35.05 |
| | DiffuseHigh* [24] | 61.62 | 50.25 | 26.76 | 13.10 | 34.83 |
| | I-Max [11] | 57.57 | 54.56 | 28.84 | 12.07 | 34.96 |
| | Flux + BSRGAN [52] | 60.25 | 52.06 | 25.85 | 13.39 | 35.34 |
| | UltraPixel [40] | 59.67 | 49.02 | 28.49 | 13.74 | 35.16 |
| | Diffusion-4k [51] | 60.96 | 54.67 | 27.15 | 12.31 | 34.76 |
| | HiFlow (Ours) | 55.39 | 47.70 | 28.67 | 13.86 | 35.32 |
| 4096 x 4096 (4K) | DemoFusion* [10] | 56.72 | 49.48 | 21.17 | 8.49 | 35.27 |
| | DiffuseHigh* [24] | 62.01 | 50.98 | 20.60 | 8.09 | 34.98 |
| | I-Max [11] | 53.27 | 52.93 | 22.21 | 7.65 | 35.05 |
| | Flux + BSRGAN [52] | 59.53 | 54.12 | 19.32 | 8.87 | 35.37 |
| | UltraPixel [40] | 60.90 | 47.19 | 24.73 | 9.47 | 35.01 |
| | Diffusion-4k [51] | 77.95 | 70.07 | 15.79 | 6.42 | 32.67 |
| | HiFlow (Ours) | 52.55 | 45.01 | 24.62 | 9.73 | 35.40 |

N Table 1 (camera-ready, p.7) is the same table without the UltraPixel and Diffusion-4k rows (reference numbers shift: DemoFusion* [12], DiffuseHigh* [26], I-Max [13], BSRGAN [58]). All remaining values are identical. No plain "FLUX.1-dev direct at 4096" row exists in either version.

Table 2 (A p.9; N p.9): Comparison in latency.

| Resolution | Method | Latency (sec.) ↓ |
|---|---|---|
| 2048^2 | DemoFusion* [10] | 106 |
| | DiffuseHigh* [24] | 59 |
| | I-Max [11] | 94 |
| | HiFlow (Ours) | 56 |
| 4096^2 | DemoFusion* [10] | 972 |
| | DiffuseHigh* [24] | 533 |
| | I-Max [11] | 735 |
| | HiFlow (Ours) | 379 |

Table 3 (A p.9; N p.9): Quantitative results of ablation study (Ai initialization, Ad direction, Aa acceleration alignment).

| Resolution | Method | FID ↓ | FIDpatch ↓ | IS ↑ | ISpatch ↑ | CLIP ↑ |
|---|---|---|---|---|---|---|
| 2048^2 | −Aa, −Ad, −Ai | 67.87 | 71.79 | 23.47 | 9.16 | 33.81 |
| | −Aa, −Ad | 58.26 | 54.22 | 27.46 | 12.58 | 34.47 |
| | −Aa | 58.40 | 50.38 | 28.92 | 13.14 | 34.90 |
| | HiFlow (Ours) | 55.39 | 47.70 | 28.67 | 13.86 | 35.32 |
| 4096^2 | −Aa, −Ad, −Ai | 234.38 | 198.21 | 9.33 | 3.31 | 11.42 |
| | −Aa, −Ad | 56.20 | 48.30 | 19.85 | 6.51 | 33.64 |
| | −Aa | 55.36 | 51.79 | 22.78 | 8.77 | 35.44 |
| | HiFlow (Ours) | 52.55 | 45.01 | 24.62 | 9.73 | 35.40 |

## Notes for the SCA comparison

- HiFlow's numbers are on a private 1K-caption set with a 10K LAION-HR reference, so they are not comparable to Aesthetic-4K numbers.
- Because the FIDpatch protocol is unstated, any "FIDpatch as in HiFlow" claim must define its own crop protocol.
- Statistical note, not from the paper: with about 1K generated images, Inception-feature FID is strongly biased upward. Absolute values in the 50s are typical of that regime.
