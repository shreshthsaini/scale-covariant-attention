# AReA's idea and ours, stated the same way (positioning agent, 2026-11-05)

Sources: paper/main.tex (build delivery_eeofzli6), AReA arXiv 2610.17227v1 (text and renders in
/scratch/09032/saini_2/tmp/area_digest/, project page), the three round-1 reviews and meta_review.md,
HANDOVER.md Sections 24 to 27 and the 2026-11-05 addendum, duplication_fix_20261027.md, duplication_fix_round6.md,
calib/*.json and *.npz, results/heads_view/summary.csv, results/dup_audit_20261027/human_rates_95.txt.
Every number below is in the paper or in results/ unless it is marked NEW (derived on CPU from the archived
calibration tables by evidence/variance_split.py; not yet in results/ or the paper).

## 1. AReA in one sentence, as its abstract sells it

> Attention's receptive field narrows as denoising proceeds, so AReA makes it follow that progression: coarse
> attention over pooled keys while the layout forms, then local windows that shrink to native size as detail
> emerges, with one schedule for every head.

The last clause is our reading. AReA never says "shared by heads", but nothing in its method varies by head.
The idea lands because it has five parts, and the name is the principle:

| Part | AReA | Where |
|---|---|---|
| Diagnosis | Uniform full attention causes both problems: quadratic cost and out-of-range token counts and distances | abstract, intro paragraph 3 |
| Observation | Distance to dominant keys (at least 1/4 of the maximum) shrinks 5.5x over denoising, 1.4x over all keys; vanilla FLUX.1 at 1024^2, averaged over heads, blocks and 24 prompts | Figure 3, page 3, inside the introduction |
| Principle | The receptive field should follow the coarse-to-fine progression of denoising | abstract, intro, method opening, conclusion |
| Mechanism | Phase I: pooled keys on a native-size grid, f ramped 4 to 2. Phase II: windows shrinking from about 70% of the grid side (180 tokens) to 64, DWT splits, halo, global-KV bridge | Figure 4 |
| Name | Adaptive Receptive-Field Attention | title |

Each component is explained by the same principle: the pooling ramp gets finer "as the layout forms", the
windows shrink "as fine details emerge", and the bridge is the patch for duplication. A reader can restate the
paper from its title.

What AReA does not have: its Figure 3 measurement never sets a parameter. sigma* is "chosen empirically",
w_start comes from a cost budget, and rho = 0.5 and delta = 8 are fixed by hand. There are no intervals, no seeds,
no sigma* ablation and no limitations section (area_vs_sca_presentation.md). Its observation is an average over heads.

## 2. Ours today, in the same form

> Each head's balance between a local window and a coarse view of the whole image can be measured at the
> training resolution and transfers to larger images, so SCA calibrates one weight per head, noise level and
> image-to-window ratio in closed form at 1024^2 and, after a global opening, mixes the two views with a window
> that shrinks to native size over four steps.

What a reader keeps after the abstract: "AReA plus a per-head weight fitted by least squares".

| Part | SCA today | Where |
|---|---|---|
| Diagnosis | Unseen token counts and offsets, quadratic cost: AReA's diagnosis, restated | abstract sentence 1 |
| Observation | Heads differ (SD 0.24 to 0.26 on a 0 to 1 scale), repeat on disjoint prompts (0.989), track measured reach (rho = 0.77) | Section 3 and Figure 5, page 6, after the method |
| Principle | "can be measured at the training resolution and transfers to larger images" | abstract sentence 2, conclusion |
| Mechanism | Opening for 8 of 28 steps "as in AReA", block-centre keys ("AReA's first phase at pooling factor s"), window plus 1,024 bridge tokens "with AReA's selection rule", 4-step shrink, per-head weight table | abstract, intro paragraph 3, Section 2 |
| Name | Scale-Covariant Attention: a property the introduction narrows to "a narrow sense" | title, intro paragraph 4 |

## 3. Where ours reads weaker, and why

1. **The principle is phrased as a procedure.** "Fitting one weight per head, noise level, and image-to-window
   ratio in closed form" names three indices and a solver. AReA's abstract never names a parameter. Ours tells the
   reader what was fitted before it says what was learned about the model, so the reader hears "fit some weights".
2. **Our observation has no picture before the method.** Page 2 holds the Moai pair, which argues for the global
   opening, AReA's Phase I. Our own observation (Figure 5) arrives on page 6, after two pages of operators. AReA's
   Figure 3 sits inside its introduction.
3. **Most of the visible machinery is AReA's, and the paper says so.** The opening, the block-centre keys, the
   bridge rule and the idea of shrinking windows are borrowed and attributed, which is correct. Once they are
   subtracted, what is left on the page is a weight table.
4. **The title does not survive the paper.** Reviewer 1 (item 6) asked us to define "scale covariance" or rename
   it, and the introduction now says "in a narrow sense". A reader cannot restate the idea from the title.
5. **The contribution bullets list a procedure, our own failure, and benchmark numbers.** None states a finding
   about the model, and AReA's abstract leads with one.
6. **Our weakness sits exactly where AReA's design works.** Human-validated duplication on the 95 person or
   animal prompts is 26.7% [24.6, 36.5] for SCA against 4.2% [1.6, 17.0] for AReA
   (results/dup_audit_20261027/human_rates_95.txt). The bridge that does most of our reduction is AReA's
   duplication fix. The current build's "about 7% corrected" for AReA is wrong (HANDOVER addendum 2026-11-05) and
   must read 4.2% in every framing below.
7. **"Measured" has visible exceptions.** The gate, four window sizes, sigma bins, bridge size and tile are set by
   hand (Table F.2). Calibration omits the bridge and sees 16 times fewer keys per branch than the target. Output
   error cannot see layout: the best hard switch it finds is local attention from step 1 (sigma_star = 1.0 in
   calib/k1_1024.json at both ratios), which v0 shows is catastrophic (ImageReward -1.33, 26 of 26 prompts
   repeat). So a human decides when the layout is fixed, as in AReA.
8. **The closed form is a scalar projection** (reviewer 1, item 11). Sold as a contribution, it reads as trivial
   math. The idea that matters is where the reference comes from, not how the quadratic is solved.

## 4. What a reviewer would call it today

- "A calibrated mixture of AReA's two phases."
- "AReA with per-head blend weights fitted by least squares, plus a shorter window shrink."
- "A faster, tuned AReA that repeats subjects six times as often."
- "A speed paper: on Krea, AReA's checkpoint, quality ties" (reviewer 2's most damaging question).
- "Head heterogeneity is known": UltraImage fixes head pattern types from one step at run time and noted stable
  local and global head roles; DuoAttention splits retrieval and streaming heads; Sparse VideoGen splits spatial and
  temporal heads.
- "Calibrated per-head attention is known": HRDiT fits per-head scopes with a MoA-style profiling and plan search
  on 30 calibration prompts, once per target size (about 48 min of preparation for FLUX at 4K on an H200, its
  Supp. C, Table 7). The related-work section must answer this directly.

## 5. What ours has that AReA lacks, and the current framing hides

- **The measurement is the parameter.** AReA measures attention at 1024^2 (its Figure 3) and then sets its schedule
  by hand. SCA's weights are fitted at 1024^2 against the model's own full attention.
- **Head-level evidence in two currencies.** At ratio 4 the per-head table has relative output error 0.200 against
  0.223 per layer, 0.230 shared and 0.261 for the best hard switch. Image metrics confirm the order: a shared weight
  costs 0.09 ImageReward and 10.4 MUSIQ_c, and local attention alone after the opening costs 0.36 ImageReward and
  repeats subjects in 25 of 26 prompts against 15 of 26 with the per-head mix.
- **Transfer.** The ratio-2 table explains ratio 4 with 2.4% more output error (0.205 against 0.200). The 1024^2
  table scored at 1280^2 gives 0.0589 against 0.0585.
- **Statistics and scope.** Paired intervals over 15 metrics, three seeds, 1000 HRGen-1K prompts, human-validated
  duplication, three checkpoints, and linear cost (fitted exponent 0.99 against 1.12; 309.7 against 952.0 s at
  8192^2).
- **NEW: a stronger observation than AReA's, unused so far.** In the model's own full attention at the training
  resolution, after the first 8 of 28 steps (sigma < 0.89), the head (its layer and its place in the layer)
  accounts for 97.5% of the variance in how much image attention a query keeps within 16 tokens, and the noise
  level for 2.5%. The head-averaged curve over noise level, which is what AReA's Figure 3 plots, accounts for 0.65%.
  The same holds on FLUX.1-Krea-dev (97.6% and 2.4%), on FLUX.2-dev (99.1% and 0.9%) and at 1280^2 (97.8% and 2.2%).
  Of the total variance, place within the layer carries 81 to 89% and the layer 10 to 17%.
  Source: evidence/variance_split_out.txt; inputs calib/{k1_1024,krea_1024,flux2_1024,k1_1280b}.npz, hashes in
  evidence/inputs.sha256.

Caveats for the NEW numbers. (a) AReA measures distance to dominant keys, which shrinks 5.5x. We measure the share
of image attention mass within r tokens, the quantity that sets the output by Lemma 1 (softmax split). The head
average narrows on our statistic too (share within 8 tokens 0.29 to 0.47 across all 13 noise bins), so the two
observations agree on direction. (b) The split covers sigma < 0.89 for every table. Including the
opening bins, the noise-level share rises to 7 to 10% on FLUX.1-dev and the head share is still 90 to 93%.
(c) 256 sampled queries per prompt, 32 prompts (16 for Krea, FLUX.2 and 1280^2).

The data do not support a binary taxonomy of heads. On the scale-4 table, 19% of heads keep pi above 0.9 in at
least three quarters of the 13 noise bins, 5% keep it below 0.3, and the rest spread in between (evidence/pi_stats_out.txt). Reach is
a continuous property of each head.
