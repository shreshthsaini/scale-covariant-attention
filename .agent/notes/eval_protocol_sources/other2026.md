# Other 2025-2026 papers: training-free FLUX at 4096^2 on Aesthetic-4K or HRGen-1K

Scope: papers other than AReA 2610.17227, SEGA 2605.22668, HRDiT 2608.07003, UltraImage 2512.04504,
DyPE 2510.20766, SigMa (ICML 2026), HiFlow, I-Max, ScaleDiff, ResDiT, TIDE 2603.08928.
Search done 2026-10-20 with WebSearch (9 queries) and arXiv listing searches ("Aesthetic-4K",
"training-free high-resolution image generation" newest 100 of 228, "resolution extrapolation
diffusion transformer", "training-free 4K FLUX"). Every paper below was opened as a PDF and grepped,
unless marked "abstract only".

## Bottom line
- No paper outside the excluded list reports training-free FLUX results at 4096^2 on Aesthetic-4K
  (Aesthetic-Eval@4096) or on HRGen-1K.
- HRGen-1K is introduced by AReA (2610.17227, appendix "HRGEN-1K BENCHMARK"), which says it follows
  "SEGA (Rajabi et al., 2026) and JoLT (Koroglu et al., 2026)". No other paper using HRGen-1K was found.
- Definition check: Aesthetic-Eval@4096 = the 195 images of Aesthetic-Eval (2,781 image-text pairs
  selected from LAION-Aesthetics V2 6.5+ with short side > 2048) whose short side is > 4096.
  (Diffusion-4K, arXiv 2503.18352v2, Section 3.1 "Aesthetic-4K Benchmark", p.4.) The same LAION-Aesthetics-V2-6.5+ pool is where
  FreeScale draws its 1024 random captions.

## 1. JoLT (Koroglu, Jeanneret, Caselles-Dupre, Cord, Dapogny), arXiv 2608.15395v1 (15 Aug 2026)
- Training-free: yes ("We propose JoLT, a training-free pipeline", Section 1, p.3).
- Model: four-step-distilled FLUX.2 Klein-9B, not FLUX.1-dev. All baselines (DemoFusion, SEGA, ResDiT)
  re-implemented on the same FLUX.2 Klein-9B backbone. (Section 4.1, p.7-8)
- Resolution: 4096^2 canvas from 1024^2 HR windows, 512 px stride, MultiDiffusion; LR stream 1024^2.
  (Section 4.1, p.7)
- Prompt set: 200 prompts generated with "GPT-5.6 Sol-high", 30-80 words (mean 36.3), at least four
  disparate objects each; released in the project repository. Not Aesthetic-4K, not HRGen-1K.
  (Section 4.1 "Evaluation Dataset", p.8)
- Metrics: complexity (IC9600, DCT energy, JPEG ratio); text alignment (LMM4LMM-Correspondence);
  aesthetics/quality (LMM4LMM-Perception, Qwen-Image-Bench Quality); latency on one H100 (Table 1,
  p.9). Global-local coherence: Qwen3.6-27B judges each full image plus four crops (Table 2, p.9;
  App. C.1). User study: AMT 2AFC, 2,700 judgments over 120 prompts, three questions (Section 4.4,
  p.10; Table 3, p.11). App. B: HPSv2.1, ImageReward, PickScore. App. C.6 Table 7 (p.25) "legacy"
  metrics: CLIP B/32, CLIP L/14, MUSIQ, CLIP-IQA, CLIP-IQA+, IS. No FID, no KID, no patch FID.
- All 195 Aesthetic-4K prompts: NO (Aesthetic-4K not used).

## 2. UltraFlux, arXiv 2511.18050v1 (22 Nov 2025)
- Training-free: NO (trained Flux-based 4K model). Listed because it reports training-free baselines
  on Aesthetic-Eval@4096.
- Prompt set: "All methods are evaluated on the Aesthetic-Eval@4096 benchmark [42]" (Section 4.1, p.7).
  Number of prompts actually evaluated: NOT STATED (195 never written).
- Resolution: 4096x4096 (Table 2 "under 4K res."; Fig. 6 "at 4096x4096").
- Metrics: FID, HPSv3, PickScore, ArtiMuse, CLIP Score, Q-Align, MUSIQ. FID reference set and resize:
  NOT STATED. Patch/crop FID: none.
- Training-free baselines: ScaleCrafter and FouriScale; their base model is NOT STATED (both are UNet
  methods). Table 2 (p.7), FID / HPSv3 / PickScore / ArtiMuse / CLIP / Q-Align / MUSIQ:
  ScaleCrafter 164.02 / 6.83 / 21.68 / 67.88 / 33.36 / 4.30 / 38.21;
  FouriScale 164.71 / 11.19 / 21.86 / 65.87 / 33.11 / 4.50 / 38.96;
  Sana 144.17 / 10.83 / 23.18 / 63.72 / 35.49 / 4.89 / 45.08;
  Diffusion-4K 152.43 / 8.92 / 21.88 / 63.76 / 33.00 / 4.69 / 27.51;
  UltraFlux 143.11 / 11.47 / 22.69 / 68.36 / 34.62 / 4.85 / 46.13.
- All 195: implied by "evaluated on the Aesthetic-Eval@4096 benchmark", count NOT STATED.

## 3. UltraImageGen, arXiv 2510.16325v4 (1 Jul 2026)
- Training-free: NO (LoRA-trained hierarchical local attention; trained on 256^2 to 1K data; App.
  Table 4 reports FID "after 10K training steps").
- Model: FLUX.1-dev. Training-free FLUX.1-dev baselines: DemoFusion, DiffuseHigh, I-MAX, HiFlow
  ("evaluated on the same FLUX.1-dev base model, following their official implementations").
  (Section 4.1-4.2, p.9-10)
- Prompt set: 1,000 prompts generated with GPT-4o. Not Aesthetic-4K (Aesthetic-4K real images appear
  only in an appendix power-spectrum comparison, App. 11.2).
- Resolution: 4096x4096 (Table 1 "at 4K x 4K"); qualitative 8192^2 and 4096x10240.
- Metrics: CLIP Score, CLIP-IQA, FID (vs 10,000 real LAION-High-Resolution images), IS, FIDpatch,
  ISpatch ("calculated on local image patches"; crop size and count NOT STATED). (Section 4.1, p.9-10)
- All 195 Aesthetic-4K prompts: NO.

## 4. Ultra-High-Resolution Image Synthesis: Data, Method and Evaluation, arXiv 2506.01331v1 (2 Jun 2025)
- Journal extension of Diffusion-4K. Training-free: NO (fine-tunes SD3-2B and Flux-12B). Restates
  Aesthetic-Eval = 2,781 images, 195 with short side > 4096 = Aesthetic-Eval@4096 (p.4). No
  training-free FLUX 4096 results found in its tables.

## Checked and out of scope
- CineScale, arXiv 2508.15774v3 (16 Sep 2026): FreeScale extension, now titled "Tuning-Free
  High-Resolution Video Generation"; no Aesthetic-4K mention in the text.
- PixelRush, arXiv 2602.12769v2 (27 Feb 2026): training-free but SDXL / SDXL-Turbo, 1000 prompts
  "randomly sampled from the LAION/LAION2B aesthetic dataset", 2048^2 and 4096^2. No FLUX, no Aesthetic-4K.
- SAFE-DiT, arXiv 2606.29360v1 (28 Jun 2026): acceleration for high-resolution DiTs, up to 3072^2; no
  Aesthetic-4K.
- Abstract only (UNVERIFIED beyond the arXiv abstract): MrFlow 2607.01642 (staged-resolution
  acceleration, FLUX.1-dev, OneIG), SelfLift 2609.02036 (FLUX.2-Klein acceleration), PhotoQuilt
  2606.30968 (photomosaics), Spectral Progressive Diffusion 2605.18736, UltraDiffEdit 2607.06136
  (editing), EditCrafter 2604.10268 (editing). None mentions Aesthetic-4K, HRGen-1K or 4096 in its abstract.
