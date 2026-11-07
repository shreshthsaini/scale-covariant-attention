# Root-cause audit of late duplication in SCA (2026-11-06, CPU only)

Scope: the operator's math and implementation (sca/attn.py, sca/calib.py, calib/k1_1024.npz) and the saved probes
(results/round6_mechanism, /scratch/09032/saini_2/sca/r6/probe/sca_v2/*/probe.npz). No code or paper edits, no GPU.
Numbers below are new unless cited. Probe conventions: masses are softmax fractions, "main" = mass on the main
subject's box, mean over 57 blocks x 24 heads at 6 labeled copies; "mix" = pi O_loc + (1 - pi) O_glob read as a mass.

## Ranking (expected effect on copies / risk to quality)

| rank | issue | effect | risk | cheap fix |
|---|---|---|---|---|
| 1 | per-head constant pi where full attention has a per-query mass (two softmaxes, one weight) | high | moderate | per-query weight from image-only, outside-window, count-corrected partition functions; head mean pinned to pi |
| 2 | calibration: output-MSE optimum over-weights the window by about 2x against the near-field mass; emulated 16-token windows are not native canvases | high | moderate | blend pi toward the measured near share at the hand-off; recalibrate at 2048^2 with native 64 windows |
| 3 | text is normalized twice: the global branch reads the prompt at native strength (0.21 to 0.29), so the mix's text share cannot fall below (1 - pi) x 0.23 | medium | moderate | fold the text into the single three-part softmax of fix 1 |
| 4 | text queries read only the coarse canvas; AReA's read the windows | medium-low | low | text queries over all image keys at coarse ids (9% of local cost) |
| 5 | bridge keys inside the window counted twice; bridge at true ids | low | none | mask in-window bridge keys (score_mod) |
| 6 | coarse query ids c(p) shared by 16 queries | low | low | none needed (frac ids tested, no effect) |
| 7 | fixed sliding windows vs jittered tiles | low | low | none (sliding windows have no seams; only jitter differs) |

## 1. Per-head weight versus per-query mass (the softmax split)

Issue. Lemma 1 gives O_full = m_i O_near + (1 - m_i) O_far with m_i the near set's softmax mass of that query. SCA
replaces m_i by pi_lh, a constant over all 65,536 queries of a head. The window softmax is normalized without the far
keys, so at a query whose window holds a strong local match (a prop plus the full prompt drive) O_loc is a full-strength
"draw the subject here" output no matter how much far mass the query would have, and the mix adds O_glob at a fixed
0.3. Nothing in the operator can say "this query's mass belongs elsewhere".

Evidence (calibration, calib/k1_1024.npz, scale 4, bins 0.83 to 0.97). The exact near-field image share of full
attention at the emulated window (Chebyshev radius 8 of the stored masses, divided by 1 - text) is 0.36 at sigma 0.896
against a calibrated mean pi of 0.69; including the text keys the local key set holds 0.47 of the mass. 75% of
(head, bin) cells keep less than half their image mass inside the window, but only 19% have pi < 0.5; 37% have far
share > 0.6 together with pi > 0.6. The gap pi - m has percentiles 10/50/90 of +0.06 / +0.32 / +0.70. Correlation
between pi and m across heads is only 0.56 to 0.61, so the calibrated weight ranks heads by something other than
how much of their mass is near.

Evidence (probe at copy sites, steps 8 to 12, sigma 0.888 to 0.808). Full attention's main-subject mass seen from the
copy site rises 0.067 -> 0.092 across the seeding steps; the global branch tracks it (0.052 -> 0.073); the window loses
it (0.059 -> 0.041); the mix stays flat (0.055 -> 0.052): 82% of full attention's subject mass at step 8, 56% at step
12, exactly while the copy forms. The per-head weight on O_loc that would restore full attention's subject mass at the
copy site is (median over heads) 0.40, 0.32, 0.28, 0.26, 0.17 at steps 8 to 12, against pi = 0.65 to 0.70. At the main
subject the window over-concentrates (0.57 to 0.71 vs full 0.39 to 0.45), at background it under-delivers (0.020 vs
0.036); the matching weights there are 0.12 to 0.30 and 0.02 to 0.16. In terms of where mass goes the whole image
wants 0.2 to 0.4 on the window's output; the operator gives 0.7 everywhere.

Why round 5's LSE mix (mix_lse, --lse-tau 1, pilot dup_lse1 6/9 vs base 6/9) could not work. Its d = lse_loc -
lse_glob used lse_loc over [text, bridge, window] and lse_glob over [text, all 4,096 block centres]. (a) Both partitions
contain the text keys and the block centres inside the window (256 of 4,096 at w = 64, 1,024 at w = 128): a strong
local match raises both LSEs, so d is blind to the one situation it should detect. (b) The global branch scores at
coarse ids, the local at true ids, so the same key gets different logits and d carries a RoPE artifact. (c) No s^2
count correction for keys that stand for 16 tokens. The form (head mean pinned at logit pi, per-query swing) is right;
d measured the wrong thing. The bridge did not rescue it (dup_bridge_lse1 4/9 vs bridge 3/9).

Fix (keeps calibrated per-head receptive fields, cost under 5% of the local branch). Per query and head compute
  Z_T   = text partition (textgate.text_part, 65,536 x 512),
  Z_in  = partition of the global branch's block centres inside the query's window (one small attention, 256 keys),
  Z_far = s^2 (exp(lse_glob) - Z_T - Z_in)          (lse_glob from flex return_lse, already wired),
  Z_loc = exp(lse_loc) - Z_T'                        (lse_loc from flex return_lse),
  d_i   = log Z_loc - log Z_far,  w_i = sigmoid(logit pi_h + tau (d_i - mean_i d_i)).
Mean-centring keeps every head's mean weight at its calibrated pi, so the published tables and the paper's
calibration story stand; only the distribution across queries changes, toward the global branch where the far field
holds the mass. tau = 1 is the exact Lemma 1 slope; start there. Quality risk: lse1 lost PickScore, but it was
mis-specified; this version leaves queries whose window holds the subject (the main site) at or above pi.

## 2. Calibration objective

Issue. pi* = clip(B/A) minimizes output MSE pooled over all queries of a head per noise bin.
(a) The MSE optimum is not the mass share: at bin 12 the per-head table has rel. error 0.205; pi = exact image near
share m costs 0.256, m with text 0.237 (closed form from the saved sums). The objective over-weights the window 2x
because the window's output carries the high-energy near-field values and the global branch at coarse ids is a poor
O_far (0.59 alone). MSE buys sharpness and gives up the allocation of mass, which decides whether a window redraws.
(b) The sum is dominated by queries where the branches disagree in energy, not by the rare tiles where they disagree
in meaning; copy-prone sites (a few tiles of 1,024) have no leverage. Split-half correlation 0.99: this is the
objective, not noise.
(c) Emulation mismatch: the scale-4 emulation at 1024^2 uses 16-token windows, texture patches to the model. At
4096^2 the native 64-token window is a whole training canvas with the full prompt, and the object prior acts inside it
(the paper's "native canvas" hypothesis). Ratio covariance holds for cost and offsets, not for a prior anchored to
absolute token counts. sca_area20 (AReA's windows and bridge inside SCA, pi at the matched ratio, still 12/21) points
at the weights rather than the windows.

Fix. Cheapest: at the hand-off steps read pi' = (1 - lambda) pi + lambda m_h, m_h the stored near share (mass_mean
is in the npz; no GPU), lambda 0.5 to 1 over sigma > 0.8. Better: recalibrate at 2048^2 with scale-2 emulation (window
64 = native canvas; full attention at 2048^2 is sane) and use that table in the native phase; s2 -> s4 tables transfer
at +0.005 rel. error (calib/k1_1024.json), so the risk is low. Both combine with fix 1.

## 3. Text normalized twice

Issue. Each branch gives the prompt its own share. At 4096^2 the probe measures text mass of full attention 0.035,
the window 0.06 to 0.10, the global branch 0.21 to 0.29 (its native share), the mix 0.12 to 0.15. Two thirds of the
mix's text drive comes from the global branch, which no variant touched: ctw lowered the window's part only (copies
4 -> 3 of 26, n.s.), gtx raised the window's part to the global's (4 -> 8), txtmid biased the window's part (destroyed
IR). The direction is consistent (more text share, more copies); the larger term was never pulled. Under one softmax
over (text, near image, far image) the prompt is counted once and competes with the far subject keys. Full
attention's 0.035 at 4096^2 is a broken reference (subject missing in 22 of 37 images); the aim is one normalization,
not that number.

Fix. Fold into fix 1: compute O = (Z_T O_T + Z_near O_near + Z_far O_far) / (Z_T + Z_near + Z_far) with O_T the text
part (textgate.merge_lse exists), O_near the window+bridge image part, O_far the global branch's image part, and the
per-head pi entering only as the calibrated offset on log Z_near. Risk: prompt adherence (ctw lost IR and PickScore),
so keep it to the seeding steps first (sigma > 0.8).

## 4. Text queries read the coarse canvas only

Issue. In every block the text tokens are updated from O_glob (text + 4,096 block centres at coarse ids) and become
the next block's text keys for every window; AReA's text queries average over the fine windows. The text stream thus
carries a subject state from a view where an 8-token copy is 2 coarse tokens. No saved measurement bears on it (the
probe reads image queries only); the morph rungs sca+C and area+text are the test. Cheap fix: text queries attend all
image keys at coarse ids c(p) plus text (512 x 66,048 pairs, 9% of the local branch), offsets in range. Risk low.

## 5. Bridge double counting and bridge RoPE

Issue. Keys are a multiset: bridge tokens inside the window enter twice (about 64 of 5,120 keys at w = 64, 256 of
17,408 at w = 128), and they are the outlier keys, the most distinctive tokens in the window. Doubling their logit mass
inside the very window that is forming a copy is the wrong direction, but the share is at most a few percent.
Bridge keys at true ids reach offsets up to 255, as in AReA (shared, not a differentiator). Fix: a score_mod or mask
that drops a bridge key when its position lies inside the query's window (free; bridge indices are per block, so use
score_mod with a captured buffer rather than a new BlockMask).

## 6. Coarse query ids

c(p) = floor(p / 4) gives the 16 queries of a block identical relative positions in the global branch; content still
separates them, and AReA's Phase I has the same geometry. --glob-pos frac (frac_ramp60 3/9 vs ramp60 4/9) moved
nothing. Low rank.

## 7. Fixed sliding windows versus jittered tiles

The seam premise does not apply: SCA's windows slide per 8 x 8 tile (each tile centred in its own clamped window), so
there are no partition seams, and coverage equals AReA's at the same side (49% of the image at w = 180). Only the
step-to-step variation of what a location sees (jitter, Halo) differs. The morph rungs sca+B and area+fixed measure
it; expected small next to items 1 to 3.

## Text RoPE offsets

The window's text logits use offsets up to 255, the global branch's up to 63; full attention and AReA share the
out-of-range case, so it is not a differentiator. Remark rem:text's refinement (text offsets through c(p) in the
window) would raise the window's text share, the wrong direction by the gtx evidence.

## Order of GPU tests

Fix 1 alone (all post-gate steps, and sigma > 0.8 only); fix 2's cheap form alone and with fix 1; fix 3 only if copies
remain. All reuse sca_dedup_forward's return_lse wiring and textgate's text_part / merge_lse; the new pieces are the
256-key in-window global LSE and the s^2 count correction.
