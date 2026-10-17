# Baseline and presentation audit, 2026-10-17

The current manuscript already contains six methods at 4K: full attention, AReA,
DyPE, official SEGA, guided UltraImage, and SCA. Their 191-prompt results are complete.
The scientific gap is direct evaluation of recent head-adaptive and positional
methods, especially HRDiT and SigMa. The presentation gap is that these comparisons
are harder to see than the two-method AReA/SCA headline. This audit prepares new
experiments; it adds no measured baseline results and changes no existing method.

The strongest next addition is HRDiT, since it already assigns different attention
scopes to different heads. SigMa is a useful recent direct-inference control and
appears in AReA's comparison. ResDiT is also close to SCA's local/global design, but
an official implementation was not found in the primary paper, CVF entry, or bounded
repository search. This is a code-discovery limitation, not proof that no code exists.
Do not present an invented implementation as an official ResDiT baseline.

| Method | Primary comparison and protocol | SCA coverage and action |
|---|---|---|
| AReA | Aesthetic-4K 195 prompts and HRGen-1K; FLUX.1/Qwen; four square/wide settings. Main tables include full, DyPE, SigMa, SEGA, and direct UltraImage; multistage comparisons are separate. Metrics cover alignment, preference, distribution, local quality, and latency. H100 timing. | Complete 191-prompt dev comparison, 200 HRGen prompts, two wide settings and exploratory 8K; Krea/6K completion still pending. Our guided UltraImage protocol differs from AReA's direct UltraImage. |
| DyPE | FLUX.1-Krea-dev; DrawBench and Aesthetic-4K, square and landscape. Static NTK/YaRN and dynamic variants; CLIP, ImageReward, aesthetics, FID. Resolution-scaling and human studies are separate. | Complete dev baseline; Krea comparison is needed for author-backbone fairness and is underway. |
| SEGA | Krea and Qwen, Aesthetic-4K plus a separate prompt set; 2K-by-4K, transposed, 3K, 4K. Compares static PE, DyPE, UltraImage and FLUX-only HiFlow/I-Max/ScaleDiff; preference and quality metrics, FID and patch FID. NTK is default, gamma1.5 and kappa0.08, bf16 on H100. | Complete official dev baseline; Krea inherited outputs now have a seven-image provenance repair overlay. Final Krea comparison remains incomplete. |
| UltraImage | Direct resolution, guided resolution, and guided view extrapolation are separate. Main table uses FLUX at3600² and Qwen at4096², reporting FID/KID/CLIP against positional and entropy controls and guided alternatives. | Complete guided 30+30-step, guidance3.5/7, 4096² comparison and budget controls. This does not reproduce its direct protocol or original3600² table. |
| HRDiT | Table1: SD3/FLUX at2K/4K; direct, DemoFusion, DiffuseHigh, HiFlow, I-Max where compatible, FreCaS onSD3, FreeScale onSDXL. 1000 LAION captions; FID/KID, patch versions, CLIP and H200 latency. | Missing quantitative result. Pinned official runner prepared with released FLUX scope plan and progressive1K→2K→4K,30+17+10 steps. Compare as a separate-budget method. |
| ResDiT | FLUX.1-dev,35steps,guidance3.5; first10 global, middle10 spectral fusion, final15 local. 500captions, LAION-Aesthetics references, KID/IS and patch versions, CLIP and human ratings; RTX4090. Baselines DemoFusion/DiffuseHigh/I-Max/HiFlow. Table1 acknowledges weaker4K performance. | Missing quantitative result, official code not verified. Preserve mechanism discussion and missing-baseline limitation. |
| SigMa | Final ICML Table1 compares full, NTK/YaRN, DyPE variants and SigMa variants on DrawBench/Aesthetic-4K, with CLIP/aesthetics/HPSv2/FID. Paper states dev,28Eulersteps,guidance3.5. Released CLI instead hardcodes Krea and4.5. | Missing quantitative result. Prepare shared dev/guidance4.5 pilot; separate paper-backbone/guidance and released-Krea controls are available. |

Primary sources, checked directly:
[AReA](https://arxiv.org/html/2610.17227v1),
[DyPE](https://arxiv.org/html/2510.20766v1),
[SEGA](https://arxiv.org/html/2605.22668v1),
[UltraImage](https://arxiv.org/html/2512.04504v1),
[HRDiT](https://arxiv.org/html/2608.07003v1),
[ResDiT](https://arxiv.org/html/2512.01426v1),
[ResDiT final CVF entry](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_ResDiT_Evoking_the_Intrinsic_Resolution_Scalability_in_Diffusion_Transformers_CVPR_2026_paper.html),
[SigMa final PMLR entry and PDF](https://proceedings.mlr.press/v306/zhao26cc.html).
Source-byte hashes are in `repro/baseline_expansion_20261017/primary_sources.json`.
Cross-paper numerical scores must not be placed in the same ranking as our reruns.
Reference sets, backbone, preprocessing, sample count, hardware and sampling differ.

**Implementation findings.**

- [HRDiT official code](https://github.com/zylwithxy/HRDiT) is pinned to
  `f3be935da797c7dc98ecff07cf4a5f01e1ddb51d`. Repository license MIT; inherited
  source notices are retained. It needs FlexAttention and publishes a FLUX scope
  plan. Default stages use fp16, guidance3.5/4.5/6, SPA3/0 steps, HAP throughout
  high-resolution stages, group80 and global-anchor32. We retain those settings.
- [SigMa official code](https://github.com/bxuanz/SigMa) is pinned to
  `fbe3a863c20b24fd870c9912e77ce0e39e85e1b4`, MIT with inherited Apache notices.
  Its pipeline explicitly caps the scheduler shift by passing the current sequence
  length as the maximum sequence length, so4K mu equals1.15. Reading only the
  scheduler JSON would incorrectly suggest extrapolated mu. The prepared shared
  profile retains this schedule. The paper/code backbone-guidance discrepancy is
  recorded rather than silently choosing one definition of author default.
- The local dev snapshot is `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`; Krea is
  `8162a9c7b05a641be098422bf2fcf335615c2f28`. Krea contains transformer/scheduler,
  so the adapter uses dev text encoders/VAE, consistent with the current campaign.
  Metadata hashes, blob identities and sizes are checked. Weight bytes were not
  rehashed; the receipt says so. No weight downloads are allowed by the runner.
- The existing environment is Torch2.11.0+cu128, Diffusers0.39.0,
  Transformers5.19.0, Accelerate1.15.0. Both official transformers pass tiny CPU
  forwards. HRDiT additionally needs PyWavelets1.9.0 and pytorch-wavelets1.3.0,
  installed in an isolated managed uv environment with102 file hashes. The existing
  generation environment is unchanged. This does not validate GPU FlexAttention,
  half-precision numerics, pretrained4K generation or output quality.
- Other verified repository licenses: AReA and UltraImage Apache-2.0; DyPE MIT.
  GitHub reports no license for SEGA. Existing SEGA results do not establish blanket
  redistribution rights; this addition does not vendor SEGA. Model licenses remain
  separate from implementation licenses.

**Bounded experiment sequence.**

1. Freeze this new package and enqueue one4K image each for `sigma_dev_matched`
   and `hrdit_dev_author` in the existing fleet. Candidate tasks require3600seconds
   admission, use GNU compilers, preserve source hashes and run offline. No new
   jobs or live-queue changes were made by this audit.
2. Check finite generation, exact4096² dimensions, sidecar/model/source/PNG hashes,
   actual sigmas and representative visual fidelity. The first image includes
   compilation and has no excluded warmup, so its time is a planning observation,
   not clean benchmark latency. Preserve failures and compiler/OOM logs.
3. `prepare_tasks.py --stage pilot12` refuses until both smoke outputs validate.
   It prepares six bounded task candidates for the remaining11 images per method.
   Admission uses twice the observed per-image duration plus setup, floored at45min.
   It neither queues tasks nor submits jobs. Use measured runtime to revise budget.
4. Evaluate paired scalar metrics on the same12 prompts and inspect all images.
   Promotion checks implementation fidelity, finite/non-corrupt output, measured
   runtime and affordable cost, never whether SCA wins. Preserve unfavorable results.
   Expand to46 after protocol checks; then consider matched191-prompt runs within
   the authorized SU budget. The initial46 is preflight scope, not project completion.
   No automatic191-prompt expansion is prepared.
   The46 prompts are the existing abl50 order minus dev4. First12 form a nested
   pilot chosen without looking at scores. All existing191-prompt methods can be
   subset to those IDs. Keep guided/progressive methods in their own budget group.
5. Reuse the corrected evaluator with reference-image clusters. Report small-sample
   distribution estimates as exploratory. Use2000 paired prompt draws for scalar
   metrics and1000 cached-feature draws for distribution metrics, keeping GPU
   extraction separate from CPU bootstraps. New outputs use unique method labels.

The adapter differs from the release CLIs in necessary, explicit ways: deterministic
campaign CPU seeds per prompt, local pinned component loading, resident GPU execution,
VAE tiling, resumable PNG/JSON records, and intermediate paths isolated per prompt.
These choices are recorded; they must not be described as exact author-result
reproductions. In particular HRDiT's57 steps span three resolutions, so57 does not
mean the same amount of work as57 full4K steps.

**Other recent work and presentation.**

[ScaleDiff](https://arxiv.org/html/2510.25818v1) uses neighborhood patch attention
and low-resolution structure guidance in a progressive1K→2K→4K pipeline. It is
relevant to local/global design but not a missing direct28-step positional control.
[JoLT](https://arxiv.org/abs/2608.15395) jointly evolves low/high-resolution streams
with detail conditioning; its [official project](https://obvious-research.github.io/jolt/)
demonstrates FLUX.2 Klein-9B. Adding that backbone would broaden the campaign beyond
the immediate same-model comparison. [TIDE](https://arxiv.org/html/2603.08928v1)
compares FLUX/YaRN/DyPE on DrawBench/Aesthetic-4K and is already cited. These are
secondary expansion candidates after HRDiT/SigMa and current incomplete experiments.

[SANA's final ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/34345e243156da67605d4b63d71c8d98-Paper-Conference.pdf)
provides a useful Figure1 image-and-latency teaser and Figure2 system/algorithm
latency decomposition with hardware and batch size stated. Transfer that visual
discipline, not its speedup claims. For SCA, show matched whole images and identical
crops beside clean GH200 timings, retain failure examples, and distinguish campaign
timing for other baselines. Do not use old arXiv SANA teaser versions with visible
author TODOs.

[Nemotron-Labs-Diffusion-Image](https://arxiv.org/html/2606.29814v1) uses an image-first
Figure1, a concrete token-neighborhood example in Figure2, and an overall framework
in Figure3. It is a trained discrete1024px generator, not a direct4K FLUX
extrapolation baseline. The useful pattern is a concrete example tied to the
mechanism, followed by readable evidence. For SCA, connect a visible structural
failure and detail crop to the coarse/local outputs and fitted head-noise weights.
Separate method groups in the main table, keep a short representative metric set
readable, and put complete metrics/intervals in the appendix. No claim should imply
that head roles, sparse local/global attention, or head adaptation begin with SCA.
