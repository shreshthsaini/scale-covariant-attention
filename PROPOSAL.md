# Scale-Covariant Attention: proposal and literature map

Status (2026-10-15): proposal only. Nothing implemented, nothing submitted.
Origin: deep read of AReA (arXiv 2610.17227, code github.com/KooroshRH/AReA) plus two
literature sweeps. Next step is the user's choice of direction, then the kill test.

## 1. What AReA does and where it is weak

Training-free resolution extrapolation for MM-DiTs (FLUX.1, Qwen-Image, FLUX.2).
- Phase I (sigma > sigma*): full-res queries, K/V subsampled to one token per f x f
  block, query and key coordinates divided by f so RoPE distances stay native; f ramps
  linearly from f0 to 2.
- Phase II: windows shrink geometrically to native size; DWT splits smooth windows;
  halo ring; random shifts; "global-KV bridge" of distinctive tokens against duplication.
- Gains: 4K FLUX ImageReward 1.09 vs SEGA 0.79 (2048x4096); 3.2x faster at 6144^2.

Weaknesses we can attack:
1. Every head gets the same receptive field at a given step. Their locality evidence
   (Fig. 3) is averaged over heads and blocks, which hides head heterogeneity.
2. Six hand-set schedule pieces (sigma*, f0, ramp, w_start, geometric shrink, DWT
   threshold) plus three patches (halo, shifts, bridge). The bridge exists because
   pure-local late steps lose the global context some heads still need.
3. Pooling is block-center subsampling (mean pooling hurt in their ablation), a biased
   estimate of the far-field softmax partition.

## 2. Literature map (condensed)

Training-free high-res, attention topology: HiDiffusion 2311.17528 (MSW-MSA),
DemoFusion 2311.16973 (dilated sampling), FAM 2411.18552 (native global maps steer
local), FreeScale 2412.09626 (global low-freq + local high-freq fusion), ScaleDiff
2510.25818, FreeSwim 2511.14712 (video, inward sliding windows), ResDiT 2512.01426
(global + patch branches, Gaussian/Fourier fusion), UltraImage 2512.04504 (per-head
entropy focus, static), HRDiT 2608.07003 (ECCV'26: bundled positions + per-head static
local scope chosen by Taylor-estimated quality loss and an integer program, 10% cost
budget; no timestep dependence, no zoom, no cross-resolution measurement).
RoPE/temperature: Jin'23 2306.08645 (log N), Lumina-Next, I-Max, DyPE 2510.20766,
SigMa, SEGA 2605.22668, TIDE 2603.08928 (text-token dilution fix, ln s^2 bias).
Efficient DiT attention: ToDo 2402.13573 (full Q, pooled KV = Phase I topology),
DiTFastAttn(V2), SVG/SVG2, STA, Radial, Sparse-vDiT, SPADE 2608.03335, RoPeSLR 2605.20659.
LLM analogues: Self-Extend 2401.01325 (= Phase I coordinates), Ms-PoE 2403.04797
(per-head position ratios, heuristic), MoA 2406.14909 (per-head span rules),
DuoAttention 2410.10819, RazorAttention, StreamingLLM.
Trained multi-scale ViTs: Focal 2107.00641, PVT, Twins, HiLo/LITv2 2205.13213 (heads
split local vs pooled-global), Shunted 2111.15193 (per-head KV reduction rates),
DilateFormer 2302.01791, MANO 2507.02748.

Ideas killed by the prior-art check:
- Text-mass dilution fix: published as TIDE (2603.08928).
- Focal/multipole single softmax with two coordinate systems: Focal + ToMe proportional
  attention, and it has a RoPE seam discontinuity (far keys look closer than near ones).

## 3. Lead proposal: Scale-Covariant Attention (SCA)

Principle. A head's attention, viewed across resolutions the model was trained on,
decomposes into a pixel-anchored part (extent fixed in tokens: texture, edges) and an
image-anchored part (extent fixed as a fraction of the image: layout, counting,
identity), plus a text part. The split is a property of the trained model, it is
identifiable from several IN-distribution resolutions, and it depends on noise level.
Resolution extrapolation then needs no schedule: each part is extended by its own law,
in its own native-range coordinate system, with the mass the fitted law predicts.

Calibration (once per model, about an hour on one GPU): encode real images at
512/768/1024/1280, renoise to each sigma (same content at every resolution, all in
distribution), hook every joint-attention layer, and record per (layer, head, sigma)
the attention mass on text and the image-key distance histogram. Fit
P_n(d) = pi * p(d) + (1 - pi) * g(d / n) / n per head (p fixed in tokens, g fixed in
image fraction). Output: a table pi_h(sigma), tau_h(sigma) (text share).

Inference at target grid n_T = s * n0, per head, every step:
- pixel branch: sliding neighborhood (native radius) at full resolution, original
  RoPE coordinates;
- image branch: K/V pooled by s on the native grid, coordinates / s (Phase-I style);
- text keys in both; merge the two normalized outputs with weights from pi_h(sigma)
  (or via log-sum-exp with a calibrated per-head bias, which keeps query adaptivity).
Cost per query: about 2 x native key count, linear in N. No sigma*, ramps, DWT,
halos, shifts or bridge: AReA's schedule becomes one hand-tuned row of a measured table.

What is new vs neighbors: per-head AND per-sigma; zoom (pooling + coordinate
renormalization) rather than span; and the split is read off the model's own
cross-resolution behavior rather than hand-set (AReA, ResDiT, FreeScale), cost-searched
(HRDiT, MoA), heuristic (Ms-PoE) or trained (HiLo, Shunted).

Kill test K1 (measurement only, no high-res generation, about 1-2 GPU hours):
- two-component fit explains the histograms (median R^2 > 0.9);
- test-retest stability of pi_h(sigma) across prompt halves (corr > 0.8);
- non-trivial spread: >= 20% of (head, sigma) cells with pi in [0.2, 0.8], OR clear
  sigma dependence per head (otherwise it collapses to a DuoAttention-style head split);
- held-out check: fit on 512/768, predict 1024 attention mass split; SCA output vs true
  full attention at 1024, per-head cosine, must beat AReA's uniform choice.
K2 (generation): FLUX.1-dev 2048^2 and 4096^2, 32 Aesthetic-4K prompts, vs full
attention and AReA official code; then the full table (DyPE, SEGA, HRDiT, AReA).

Main risks: heads mix a sharp local peak with a sparse long-range tail (handled by the
two-branch mixture, not a single zoom); aliasing from center-sampled pooled keys at low
sigma (component B below addresses it); measurement boundary bias at 512^2 (use
boundary-corrected histograms).

## 4. Backup / component B: cumulant-corrected pooled keys

Pooling replaces log sum_{j in B} exp(q.k_j) with q.k_B (block center). Second-order
cumulant expansion: log|B| + q.mean_B + 0.5 q^T diag(var_B) q. A query-dependent
far-field estimate for O(d) extra per pooled key, drop-in for any pooled-KV method
(AReA Phase I, ToDo, SCA's image branch). Closest: ToMe proportional attention
(first order), ERA 2606.31982 (first order + radius bound), FMA (learned). No
second-order correction found. Test: per-head output fidelity vs full attention at
1024 (f = 2) and 2048 (f = 2, 4), then generation.

## 5. Backup C: native sink anchoring

Detect FLUX's native high-norm / sink image tokens and include them in every local
window at high resolution (StreamingLLM for DiTs) instead of AReA's distinctive-token
bridge. Confidence it is open: about 60%. Needs a sink-existence check first.

## 6. Reusable assets

- FLUX.1-dev snapshot cached under /scratch/09032/saini_2/tmp/hf (see
  constancy_rope/crope/runtime.py loader).
- constancy_rope harness: Aesthetic-4K prompts (195) and references exported under
  /scratch/09032/saini_2/constancy_rope/prompts/, metrics (CLIP, ImageReward,
  PickScore, MUSIQ, CLIP-IQA, FID/FID_p), SEGA/DyPE baselines, fleet spool worker.
