# Evaluation-protocol survey and gap analysis (2026-10-20)

Sources were read from the arXiv PDFs (text extraction checked against the tables), the released code, and dataset
cards. Per-paper notes with page-level citations and full table transcriptions:
`.agent/notes/eval_protocol_sources/*.md`. NS = not stated in the paper. UV = could not verify.

## 1. What each paper does

| Paper (arXiv) | Model, checkpoint | Prompts (count, selection) | Sizes | Sampler | Metrics | FID references; patch protocol |
|---|---|---|---|---|---|---|
| AReA (2610.17227v1) | "FLUX.1" (Sec. 4). Released code defaults to FLUX.1-Krea-dev, 28 steps, guidance 4.5, one seed for all prompts | Aesthetic-4K, "195 prompts paired with high-resolution reference images" (Sec. 4, Evaluation). HRGen-1K, all 1000 (Table 2, App. H) | 6144x2048, 2048x4096, 4096^2, 5120^2 (HxW or WxH NS) | NS in paper; 8 of 28 steps in Phase I (Sec. 4 Setup) | IR, PickScore, CLIP, Aesthetic, HPSv2 (version NS), MUSIQ, CLIP-IQA, ARNIQA, IS, FID, FIDp, latency (H100) | FID resizes to 299^2 (App. G.3). FIDp "computed on local crops" (Sec. 4), crop protocol NS. No eval code released |
| SEGA (2605.22668v1) | FLUX.1-Krea-dev ("dev.Krea checkpoint", App. C). Code: 28 steps, guidance 4.5 | Aesthetic-4K, count NS (Sec. 6). Random 20-prompt subsets only at 5120^2/6144^2 (App. F.3); 200 LLM prompts for a zero-shot table (App. F.2) | 2048x4096, 4096x2048, 3072^2, 4096^2 (Table 1) | Model defaults (App. C) | IR, PickScore, CLIP, MUSIQ, CLIP-IQA, HPSv2, FID, FIDp | References from Aesthetic-4K; FIDp never defined (Tables 1, 2, 4) |
| DyPE (2510.20766v2) | FLUX.1-Krea-dev (Sec. 4; code) | DrawBench 200 (Sec. 4.1.1); Aesthetic-Eval@4096, "195 curated image-prompt pairs" (Sec. 4.1.1); human study on 20 of them | up to 4096^2 and beyond | Code defaults | CLIP, ImageReward, Aesthetic, FID, patch metrics (Table 3, prompt set NS) | Aesthetic-4K references |
| Diffusion-4K / Aesthetic-4K (2503.18352v2) | SD3-2B and FLUX.1-dev with F16 VAE (trained) | Aesthetic-Eval@4096 = 195 images with short side > 4096 (Sec. 3.1, p. 4; supp. Table 9); GPT-4o captions | 2048 and 4096 | 50 Euler steps, guidance 7 (supp. Sec. 9, code) | At 4096: CLIPScore (ViT-B/16), Aesthetics, GLCM, compression ratio. FID only at 2048 (Table 3, torchmetrics, refs = split images) | No FID or patch FID at 4096 in either Diffusion-4K paper |
| HRDiT (2608.07003v1) | FLUX.1-dev (Supp. B), cascade 30/16/10 steps | 1000 random LAION-5B captions (Sec. 4.1) | 2048^2, 4096^2, 8192^2 | Cascade; guidance NS (code 3.5/4.5/6) | FID, KID, FIDp, KIDp, CLIP, latency (H200), user study | 10K LAION images; 10 random crops per image (after AccDiffusion), crop size NS |
| UltraImage (2512.04504v1) | FLUX.1-dev (Supp. 8.1) | 1K random LAION-5B captions (Sec. 6.1) | FLUX at 3600^2 only (4096^2 only for Qwen-Image) | NS (code: 30+30 steps, guidance 3.5 then 7) | FID, KID, CLIP | 10K LAION images; no patch FID |
| ResDiT (2512.01426v1) | FLUX.1-dev, 35 steps, guidance 3.5 | 500 captions, source NS (Sec. 4.1) | 3072^2, 4096^2 | single pass | KID, KIDp, IS, ISp, CLIP, user study; no FID | 2K LAION-Aesthetics images; patch protocol NS |
| ScaleDiff (2510.25818v1) | FLUX (code: FLUX.1-dev), 30 steps, guidance 3.5 | 1000 random LAION-5B pairs (Sec. 4.1) | 2048^2, 4096^2 | cascade | FID, KID, IS, FIDp, KIDp, ISp, CLIP | Real images resized to 299^2; patches "following DemoFusion", size NS |
| HiFlow (2504.06232v2) | FLUX.1-dev | 1K "high-quality captions", source NS, not released (Sec. 4.1) | up to 4096^2 | 30 steps, cascade | FID, FIDpatch, IS, ISpatch, CLIP | 10K LAION-High-Resolution images; crop protocol NS |
| I-Max (2410.07536v2) | FLUX.1-dev, Lumina-Next | 300 images per setting, prompt source NS | 4096^2 | NS (code 30+30) | GPT-4o win rates only (Sec. 3.1) | none |
| FreeScale (2412.09626v2) | SDXL | 1024 random LAION-Aesthetics-V2-6.5+ captions (Supp. A) | 2048^2, 4096^2, 2048x4096 | DDIM 50, CFG 7.5 | FID, KID, IS, FIDc, KIDc, user study | References = SDXL's own 1024^2 outputs (Sec. 4); crops "without resizing", size NS |
| ScaleCrafter (2310.07702v1) / FouriScale (2403.12963v1) | SD1.5, SD2.1, SDXL | random LAION-5B, 10k images above 1024^2 (Sec. 4) | up to 4096^2 | 50 steps | FIDr/KIDr (clean-fid), FIDb/KIDb; no crop FID | Real LAION images (set NS) |
| DemoFusion (2311.16973v2) | SDXL | 1K random LAION-5B captions | up to 4096^2 | DDIM 50, CFG 7.5 | FID, IS, FIDcrop, IScrop, CLIP | Crops at 1x resolution, then resized (Sec. 4.1); count NS |
| JoLT (2608.15395v1), UltraFlux (2511.18050v1) | FLUX.2 Klein / trained | JoLT: own 200 prompts; UltraFlux: Aesthetic-Eval@4096, count NS | 4096^2 | | JoLT: no FID; UltraFlux: FID, HPSv3, PickScore, CLIP, MUSIQ | |

TIDE (2603.08928, notes in tide.md): FLUX only (Krea-dev, schnell and SD3 not evaluated). Its Table 1 uses "all 200
text prompts from DrawBench ... and all 195 image-prompt pairs from the 4K-resolution subset of Aesthetic-4K" with
CLIP, ImageReward and aesthetic scores. No code is linked. SigMa (ICML 2026, PMLR v306, notes in sigma.md): the paper says FLUX.1-dev with 28 Euler steps and guidance 3.5
(Sec. 4.1). The released code loads FLUX.1-Krea-dev with guidance 4.5, so the checkpoint behind its tables is UV.
It evaluates all 200 DrawBench prompts and the full Aesthetic-Eval@4096 split (195 pairs) with CLIPScore, aesthetic
score, HPSv2.1 and FID; it reports no patch FID (Table 1, p. 7).

## 2. The field standard

1. Two lineages exist. Direct-inference FLUX attention and position methods (AReA, SEGA, DyPE) evaluate on the
   Aesthetic-4K evaluation split at 4096 and report the whole split: AReA, DyPE, TIDE and SigMa state all 195 prompts
   explicitly; SEGA does not state a count. AReA, SEGA and DyPE run FLUX.1-Krea-dev: SEGA and DyPE say so, and AReA's released code defaults to it. TIDE and SigMa
   state FLUX.1-dev, although SigMa's code loads Krea-dev.
   Cascade and SD-era methods (HRDiT, UltraImage, ScaleDiff, HiFlow, DemoFusion, FreeScale, ScaleCrafter, FouriScale,
   ResDiT) each draw their own random 1K LAION captions and use their own LAION reference sets, so their numbers cannot
   be compared across papers.
2. Nobody reports a headline table on a hand-picked or partial Aesthetic-4K subset. Subsets appear only in ablations,
   human studies, and the largest sizes: AReA Table 5 uses 20 prompts; SEGA App. F.3 uses a random 20 at 5120^2 and
   6144^2; DyPE's human study uses 20 prompts.
3. The shared metric set (AReA Table 1/6): ImageReward, PickScore, CLIP, LAION aesthetic, HPSv2, MUSIQ, CLIP-IQA, ARNIQA,
   IS, FID against the 195 references at 299^2, and patch FID. No paper defines patch FID reproducibly. The lineage runs
   Chai et al. 2204.07156 (50k random patches) -> DemoFusion (1x crops, resized) -> FreeScale (FIDc, size unstated). Every
   paper that reports patch FID must therefore define its own.
4. HRGen-1K exists only in AReA, evaluated on all 1000 prompts at 4096^2 with no-reference metrics (Table 2).
5. The Aesthetic-4K split has 195 rows but only 168 distinct images: 15 groups hold 27 redundant files, and each copy has
   a different caption (HF LFS hashes). Diffusion-4K's size statistics (supp. Table 9) include the duplicates.

Decision: the headline protocol is all 195 Aesthetic-4K prompts at 4096^2 (also 2048x4096 and 2048x6144), with the
AReA metric set, FID against all 195 reference rows, and our patch FID defined explicitly. HRGen-1K uses all 1000
prompts. The 4 development prompts are included for every method. The paper discloses that they were used during
development, a small bias in SCA's favour.

## 3. Evidence that AReA's published numbers are FLUX.1-Krea-dev

Our 4096^2 runs on Aesthetic-4K rows 4-103 compared with AReA's published Table 1 and Table 6 (195 prompts):

| | CLIP | PickScore | Aesthetic | MUSIQ | HPSv2 | IR |
|---|---|---|---|---|---|---|
| AReA, published | 29.71 | 23.31 | 6.37 | 45.22 | 0.297 | 1.325 |
| AReA, our run on Krea-dev | 29.69 | 23.23 | 6.36 | 45.25 | 0.294 | 1.240 |
| AReA, our run on FLUX.1-dev | 27.67 | 22.35 | 6.28 | 39.4 | 0.286 | 0.92 |

Patch FID does not reproduce (115.5 against 58.2), so AReA's unstated crop rule differs from ours.

## 4. Deviations in the current draft, and fixes

| # | Deviation | Fix (task, status) |
|---|---|---|
| D1 | Main, all-metric, non-square, and added-baseline tables use 191 prompts (dev 4 excluded) | z960 (main, 6 methods), z964/z965 (non-square) on 195: ready now. HRDiT and SigMa rows 0-3 generated 05:50 CT (a190_dev4_*, done) |
| D2 | Table 2 panel (SigMa, HRDiT budget rows, progressive, 42/56 steps, guidance 7) and seeds 1-2 on 46/50 prompts | z961 (panel, 195), gate already passes; seeds z962/z963 after a191_6-9 |
| D3 | HRGen-1K uses rows 0-199 of 1000 | hold_hrgen1k_krea (SCA-Krea, 1000; compare with AReA Table 2, 22.7 GPU-h) and optional hold_hrgen1k_dev (SCA + AReA rows 200-999, about 70 GPU-h) |
| D4 | Published numbers are Krea-dev; our protocol is FLUX.1-dev, so the draft calls them "not comparable" without the cause | hold_krea195 (SCA-Krea + AReA-Krea on the 95 missing rows, 8.6 GPU-h) then z966; results/published_compare |
| D5 | Patch FID differs from AReA's undefined FIDp | Keep ours, define it (4 fixed random 1024^2 crops per image and per reference row, Inception at 299^2), never compare FIDp across papers |
| D6 | Seeds: AReA code uses one seed for all prompts; we derive one per prompt | Disclose; same distribution |
| D7 | Reference duplicates (168 distinct of 195) | Keep the 195 official rows and weights, as now; state it |
| D8 | Latency on H100 (AReA) vs GH200 (ours) | Compare only our matched timings |
| D9 | 6144^2 on 100 prompts, 8192^2 on 12, FLUX.2-dev on 50, Krea on 100 | Label the first three as subsets, as SEGA does at 5120^2+. Extend Krea via hold_krea195 |
| D10 | Ablations on a fixed 46-prompt subset | Field practice (AReA Table 5: 20 prompts). Keep, and say it was fixed before the ablations ran |

## 5. Paper edits (for the paper agent; phrases quoted from main.tex as of 2026-10-20 05:40 CT)

1. Table `tab:main`: "191 Aesthetic-4K prompts" -> "all 195 Aesthetic-4K prompts". Refill every number from
   results/a4k195_main (base/extra summaries; FID_p from corrected_fidp) and the HRDiT row from results/a4k195_panel.
2. Data paragraph: replace "191 prompts after excluding four development prompts" with "all 195 prompts; four of them
   (boats, Moai, woman and bear, underwater surfer) were used during development, which favours SCA slightly".
   Change "(191 prompts each)" to "(195 prompts each)". Change "seeds 1 and 2 repeat the 46-prompt subset" to
   "seeds 1 and 2 repeat all 195 prompts". Change "HRGen-1K ... adds 200 prompts" to "all 1000 HRGen-1K prompts"
   (after D3; until then, say "the first 200 of 1000").
3. Baselines paragraph: "run on all 191 prompts, and SigMa on the 46-prompt subset" -> all baselines, SigMa and HRDiT
   included, "on all 195 prompts". "Both run on all 191 prompts" -> 195.
4. Table `tab:panel46` -> 195 prompts (results/a4k195_panel). Retitle it ("Multi-stage recipes and sampling budgets"),
   drop "46-image FID is exploratory", and refill the campaign times. The quality-against-time figure caption:
   "46 prompts" -> "195 prompts". The qualitative figure caption: "picked ... from the 191 evaluation prompts" -> 195.
5. Table `tab:generality`: seed rows n=195; HRGen n=1000 after D3.
6. Appendix `tab:protocol`: "Prompt counts exclude four development prompts" -> "include the four development
   prompts". Sizes row -> 195. Prompts row: "191 prompts" -> 195. Baselines row: drop "on the 46-prompt subset also
   SigMa and HRDiT".
7. "Prompts and references" paragraph: replace "are excluded from generation evaluation, leaving 191" with the
   disclosure above. Add: "The 195 reference rows hold 168 distinct images (27 duplicate rows, each with its own
   caption); we keep the official rows and weights, as Diffusion-4K's statistics do." HRGen sentence -> all 1000.
8. Distribution-metrics paragraph: "(191, or 46 on the subset)" -> "(195)"; "764 generated and 780 reference crops" ->
   "780 generated and 780 reference crops". State that patch FID is our own definition and not comparable to the
   FIDp of AReA or SEGA, which neither paper defines.
9. Intervals paragraph: replace the SCA-AReA FID/patch numbers (-7.63 [-12.79,-2.94], etc.) with
   results/a4k195_main/corrected_fidp_ci after hold_cpu_finalize/a4k195_main_bootstrap.sh.
10. "Reproducing AReA": replace "AReA's paper reports ImageReward 1.32 ... our shared protocol gives 0.92, so cross-paper
    numbers are not comparable" with "AReA's published numbers match FLUX.1-Krea-dev, its code default: rerunning its
    released code on Krea-dev reproduces its CLIP, PickScore, aesthetic and MUSIQ within 0.1 (Table
    `tab:published_compare`); on FLUX.1-dev the same code gives ImageReward 0.92." The Krea paragraph: 100 -> 195
    prompts after D4.
11. Other-baselines appendix: SigMa "runs on the 46-prompt subset" and HRDiT "on all 191 prompts" -> 195. The
    corrected-patch sentence about "The 191-prompt HRDiT evaluation" -> refill from a4k195_panel. FLUX.2-dev: "the first
    50 Aesthetic-4K prompts remaining after development-prompt exclusion" -> "rows 4-53, a 50-prompt subset".
12. Appendix tables whose captions say 191 (`tab:app_4k`, all-metrics, non-square, the DyPE/SEGA/UltraImage/HRDiT
    deltas, the added-baselines patch table) -> 195, refilled from a4k195_*. Tables that say "46 prompts" for the
    panel, budget, and progressive deltas -> 195 from a4k195_panel. The ablation and band-guard tables stay at 46 and
    are labelled as the fixed ablation subset.
13. New appendix table: `\input{figures/snippets/published_compare.tex}` (label `tab:published_compare`). Add one sentence to
    the Evaluation paragraph: "Table~\ref{tab:published_compare} places our numbers beside the published ones under
    AReA's own protocol." Regenerate the snippet with results/published_compare/build.py after z966.
