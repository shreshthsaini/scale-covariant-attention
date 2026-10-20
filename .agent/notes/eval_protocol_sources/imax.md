# I-Max: evaluation protocol audit

Sources read (fetched 2026-10-20, read only, nothing executed):

- I = "I-Max: Maximize the Resolution Potential of Pre-trained Rectified Flow Transformers with Projected Flow", Du et al., arXiv 2410.07536v2 (15 Oct 2024, latest version). It is labeled "Technical Report" in the page header. 18 PDF pages. Local: papers/2410.07536/. A web search found no later conference version.
- G = github.com/PRIS-CV/I-Max at main, commit 3b38c11 (2024-10-16). Files: README.md, gradio_demo.py, pipeline_flux_imax.py, transformer_flux.py, attention_processor.py, figures. **No evaluation scripts and no prompt list.**

## a. Base models / checkpoints

- "we use a self-trained Lumina-Next-2K model and the open-source Flux.1-dev" (I Section 3.1, p.6). Lumina-Next-2K = Lumina-Next after "60,000 iterations of supervised fine-tuning on 800K self-collected high-quality data of the 2048^2 resolution" (not public as far as stated). Flux.1-schnell is used only for the 2-step and 4-step timing examples (Section 3.6, p.10).
- G QuickStart: `black-forest-labs/FLUX.1-dev`, `torch_dtype=torch.bfloat16`, `pipe.scheduler.config.use_dynamic_shifting = False`.

## b. Prompt sets and counts

- No named benchmark (no LAION subset, COCO or DrawBench for generation) and **no FID/IS/CLIP evaluation at all**. Stated reason (Section 3.1, p.6): "existing metrics are not well-suited for high-resolution image evaluation because they require downsampling the images. Additionally, considering the efficiency of generating high-resolution images, it is challenging to produce enough test images (typically in tens of thousands) for reliable metric calculation in each experiment. Therefore, human evaluation remains the most reliable method." They then use GPT-4o preference as "a low-cost alternative to human evaluation", following PixArt-Sigma.
- Single-ablation study (Fig. 5): "We randomly generate 300 images at the 4096 x 4096 resolution using Lumina-Next-2K and Flux.1-dev for each setting and conduct a comparison using GPT-4o preference voting between w/ and w/o a specific module." (Section 3.2, p.8.) The prompt source is NOT STATED, as are images per prompt and seeds. The number of images for Fig. 6 and Fig. 7 is NOT STATED. Presumably the same 300 protocol, but this is not stated.
- Section 3.5 (p.9) analysis only (velocity loss vs timestep, Fig. 9, no generation metric): "the 1K Lumina-Next model and 1,000 randomly sampled high-resolution text-image pairs". The source dataset is NOT STATED.

## c. Resolutions

- Flux.1-dev: 1K -> 4K extrapolation; sequential ablation at 2048^2, 3072^2 and 4096^2 (Fig. 4). Native resolution is taken as 1024^2 for the toolkit computations (Section 2.3). "limits us to generate up to 4K" for Flux (Section 4, p.10).
- Fig. 6 x-axis pairs (read from the rendered figure, Lumina-Next-2K / Flux.1-dev): 2560^2/1536^2, 4096^2/2048^2, 6144^2/3072^2, 8192^2/4096^2. Lumina-Next-2K goes 2K -> 6K in App. A.2 and up to 16x in Fig. 1.

## d. Sampler settings

- Paper: projected-flow guidance weight with cosine decay alpha_t = 1 + 0.5 cos(pi t) (Section 2.2, p.5). Projection = low-pass filter. Toolkit (Section 2.3, p.5-6): NTK-aware scaled RoPE with b' = 2.5 b s for Flux.1-dev; time shifting with s* = 1.5 s for Flux.1-dev; proportional attention; text duplication (repeat text tokens s^2 times with grid position ids, Flux only).
- Steps: Fig. 10 shows 10/20/30 steps (Flux.1-dev) and 2/4 steps (Flux.1-schnell) at 4096^2 with a 30-step 1024^2 guidance image. The default step count for the GPT-4o studies is NOT STATED. Guidance scale is NOT STATED in the paper. Precision is NOT STATED in the paper.
- G QuickStart defaults: `num_inference_steps1=30, num_inference_steps2=30, guidance_scale1=3.5, guidance_scale2=7, height=4096, width=4096, ntk_factor=10, time_shift_1=3, time_shift_2=6, proportional_attention=True, text_duplication=True, swin_pachify=True, guidance_schedule="cosine_decay"`, seed `torch.random.manual_seed(25)`. Stage 1 is the native 1024 image. Stage 2 is direct 4096 guided by stage 1, with no 2K/3K intermediate.

## e. Metrics

- Only GPT-4o pairwise preference win rate. The prompt is in App. A.1, Fig. 11, p.14: system "You are an evaluator specialized in image quality analysis for high-resolution text-to-image generation models."; user "Please evaluate the following two generated images in terms of clarity, richness of detail, and overall quality. The prompt is: <prompt>. Please do not be affected by the order of the images. Output [Image 1] if the first image is better, [Image 2] if the second image is better, and give me the reason."
- FID / patch FID / KID / IS / CLIP score / ImageReward / HPS / PickScore / MUSIQ / MANIQA / CLIP-IQA / aesthetic predictor / GLCM / compression ratio: **none used**.
- User study: none. The paper argues for human evaluation but runs GPT-4o instead.

## f. Latency hardware

"We report the inference time of the denoising process, which is evaluated on a single A100 GPU." (Section 3.6, p.10.) Fig. 10 labels: 1024 x 1024 "30 steps - 26s". At 4096 x 4096: "2 steps - 53s", "4 steps - 107s" (Flux.1-schnell), "10 steps - 269s", "20 steps - 539s", "30 steps - 809s" (Flux.1-dev).

## g. Quantitative tables

**I-Max has no numeric tables.** Its quantitative results are three bar charts (Fig. 5, 6, 7, p.7) of GPT-4o win rate with **no printed values**. Approximate readings from the rendered page (about +/-0.02, not exact, Flux.1-dev bars only):

- Fig. 5 (I-Max win rate vs. removing each module, 4096^2, 300 images per setting): w/o Projected Flow ~0.95; w/o NTK-aware Scaled RoPE ~0.86; w/o Time-shifting ~0.78; w/o Proportional Attention ~0.89.
- Fig. 6 (extrapolated vs. native-resolution win rate): 1536^2 ~0.65; 2048^2 ~0.60; 3072^2 ~0.54; 4096^2 ~0.48.
- Fig. 7 (Projected Flow win rate vs. other low-resolution guidance): SDEdit ~0.75; Skip-residual ~0.72; Time-aware Scaled RoPE ~0.93. (Values re-read from a 260-dpi crop of p.7. The x-axis labels are confirmed.)

The only published FID-type numbers for I-Max on FLUX.1-dev at 4096 x 4096 come from HiFlow's Table 1, where I-Max scores FID 53.27, FIDpatch 52.93, IS 22.21, ISpatch 7.65 and CLIP 35.05 on HiFlow's private 1K captions (see hiflow.md). They are not from I-Max itself.
