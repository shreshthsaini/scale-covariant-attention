# methodfigs (v6): Figures 2, 3, 4, appendix method zoom and analysis plots

Status 2026-11-10: main figures 2, 3, 4 done (kept candidates installed under paper/figures/v6/); appendix in progress.
Builds and renders: /scratch/09032/saini_2/sca/paper_builds/v6_methodfigs_1/ (png/, obs/, emu/, tikz/out/, testdoc/).

## New files (all mine; announce)
- Scripts (paper/figures/src): v6_attn_extract.py (data), v6_method_assets.py (method assets), v6_method_common.tex,
  v6_method_a.tex, v6_method_b.tex (TikZ), v6_tikz_build.py (compiles TikZ in scratch, `--install` copies kept),
  v6_observation.py, v6_emulation.py, v6_figcheck.py (PyMuPDF check: size, smallest span, words, PNG).
- Data: paper/figures/src/data/v6_attn_block12.npz + .json (1.1 MB): full-attention rows of all 24 heads of block 12 at
  query (43, 11) of the 1024^2 capture at steps 8, 20, 26 (sigma 0.897, 0.612, 0.327), plus the r = 4 near and far
  views at step 20. Read from the CPU capture /scratch/09032/saini_2/sca/out/attn_focus/sheep_church_seed0 (no GPU).
- Outputs (paper/figures/v6): observation.pdf, method.pdf, emulation.pdf, observation_provenance.json,
  emulation_provenance.json, method_assets/ (images, data.tex, provenance.json).
- Rebuild: `PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python; cd paper/figures/src;
  $PY -I v6_observation.py; $PY -I v6_emulation.py; $PY -I v6_method_assets.py; $PY -I v6_tikz_build.py --install`.

## Figure 2: paper/figures/v6/observation.pdf (5.5 x 1.9 in, full width, label fig:observation unchanged)
Claim: how far a head looks is set by the head, not by the denoising step.
Panels: (a) two heads x three noise levels, one query, real maps; (b) all 1,368 heads as lines across the 11 post-gate
noise bins, heads 4 and 7 bold, head average black; (c) variance split drawn large with a zoom into the noise-level
share (0.7% head-average curve inside the 2.5%), plus FLUX.1-Krea-dev and FLUX.2-dev head shares. 30 words, text 7 to
8.5 pt.
Proposed caption:
  \textbf{Reach belongs to the head, not to the denoising step.} (a)~Full attention of heads 4 and 7 of block 12 at
  one query of a $1024^2$ image, at three noise levels: amber within 16 tokens of the query (dashed), blue beyond,
  with that near share. (b)~Near share of all 1,368 FLUX.1-dev heads after the first 8 steps, 32 prompts; black: head
  average. (c)~Variance of the near share; the head-averaged curve is part of the noise level
  (Table~\ref{tab:variance_split}).
Data: calib/k1_1024.npz (mass_mean), results/heads_view/variance_split.csv (checked against the recomputation and
headline.tex \hlVarHead/\hlVarStep/\hlVarAvg/\hlVarHeadKrea/\hlVarHeadFluxTwo), src/data/v6_attn_block12.npz,
paper/figures/attn_focus/image_1024.png.
Selection rule (deterministic): block 12, heads 4 and 7 = highest and lowest 32-prompt share within 16 tokens at the
bin centred at 0.62 (the rule of fig_attn_focus.py; asserted in the script). Query = the face of the leftmost sheep,
the query of v5.1 Figure 3/attn_focus (fixed earlier, not re-chosen).
Rubric (0/1/2): one claim at a glance A2 / B1; words <= 30 A1 (30) / B1 (29); text >= 6 pt A1 (7.0) / B1; vector
except images A1 / B1; lettered named panels A1 / B1; no title repeating caption A1 / B1; palette <= 4 hues A1 / B1;
example tied to population (same heads coloured in a and b) A2 / B1; layout density A1 / B0 (y label collides with
tiles); 97.5 / 2.5 / 0.7 drawn large A2 / B1 (B drops the 2.5/0.7 nesting). Mean A 1.3, B 0.9. Kept A.
A: lines + zoomed split. B: sorted matrix (v5.1 style) + three-checkpoint bars. The matrix reads as a vertical
gradient (by construction of the sort) and hides flatness in colour; lines show "flat and far apart" directly.

## Figure 3: paper/figures/v6/method.pdf (5.5 x 2.3 in, full width, label fig:method unchanged)
Claim: SCA mixes two native-size views per head with one number, fitted once at 1024^2 against full attention in
closed form, and one table serves every size and sampler stage.
Panels: (a) top row, the 4096^2 output of progressive SCA: near view (64-token window, amber) and far view (64 x 64
block-centre grid, blue), pi_h (+) (1 - pi_h); bracket and "÷ r" arrow down to (b): the same views on the 1024^2
stage of the same run (r = 4: 16-token window, 16 x 16 grid) with the real attention of head 10 of block 12, the
picture equation pi [near] + (1 - pi) [far] ≈ [full attention], the real parabola J(pi) (32 prompts, block 12,
sigma 0.62, r = 4) with pi* = 0.79, and pi* = clip(B/A, 0, 1); (c) pi* fills its cell (green outline) of the reach
table card (block 12: 24 heads x 13 noise bins, r = 4 in front of r = 2), which the progressive stages read (1024^2
full attention -> 2048^2 at r = 2 -> 4096^2 at r = 4). 16 words, 4 hues, text 6.2 to 9 pt (scripts >= 6.2 pt).
Proposed caption:
  \textbf{One number per head, measured once at $1024^2$.} (a)~A head mixes a native-size window (amber) and grid
  (blue) by its reach $\pi_h$; $4096^2$ output of progressive \sca{}. (b)~On the $1024^2$ stage of the same run both
  views shrink by $r$, and the mix is fitted to full attention in closed form (head 10 of block 12, $r=4$).
  (c)~$\pi^\star$ fills one cell of the reach table, which each stage reads at its $r$. Figure~\ref{fig:method_zoom}
  zooms in.
  (Measured in the ICLR test page: the longer first draft ran 5 lines; this one is 4 including the lead.)
Data: method_assets/ (v6_method_assets.py): img4096.jpg (attn_focus/target_4096_at1024.jpg), img1024.jpg
(attn_focus/image_1024.png), b_near/b_full/b_far.png from v6_attn_block12.npz, J curves and tables from
results/evidence/calib/k1_1024.npz. Head 10 rule (asserted): of block 12's heads, the one whose full attention at this
query puts the share closest to one half inside the r = 4 window (0.522).
Honesty note for the caption owner: the 1024^2 stage (white sheep) and the final 4096^2 output (brown sheep, autumn
trees) differ in colour because HRDiT's sampler re-noises between stages; layout and the window region match. The
caption says "1024^2 stage of the same run". The reach table itself is fitted on the 32 emulation prompts, not on this
image; (b) illustrates the operation on this image. Single-pass mode is not drawn (caption may add "single-pass mode
reads r(sigma)").
Rubric (0/1/2): one idea readable without text B2 / A1 (A: 3 overlapping J curves, (a)/(b) boundary weaker); words
15-30 B1 (16) / A0 (13); lettered panels B1 / A1; <= 4 hues B1 / A1; data flowing through (real image, real attention,
real J, real table) B2 / A2; arrows 0.6 pt orthogonal B1 / A1; blocks 2 pt radius, 0.8 pt border, tint fill B1 / A1;
labels 7-9 pt, >= 6 pt B1 / A1; layout density B1 / A1. Mean B 1.2, A 1.0. Kept B (two rows that rhyme: the 1024^2
views sit under the 4096^2 views at the same proportions).

## Figure 4: paper/figures/v6/emulation.pdf (5.5 x 1.6 in, full width; new label, suggest fig:emulation)
Claim: the fitted weight measures the head's reach, per-head weights fit full attention best at both ratios, and one
table transfers across prompts, ratios and resolutions.
Panels: (a) pi_h (ratio-2 table) vs near share, 1,368 heads at sigma 0.62, rho 0.77, heads 4 and 7 coloured; (b)
relative output error per head / per layer / shared / best hard switch at r = 2 and r = 4 (two small panels, values
printed, per head green and bold); (c) own table (dark green) vs a table used where it was not fitted (light green)
on one error axis: held-out prompt half (0.0574 vs 0.0576, label carries the split-half rho 0.989), ratio-2 table at
r = 4 (0.200 vs 0.205), 1024^2 table at 1280^2 (0.0585 vs 0.0589). 30 words, text 6.04 to 8.5 pt.
Proposed caption:
  \textbf{The fitted weight measures reach, fits full attention best, and transfers.} (a)~$\pi_h$ of the ratio-2
  table against the near share, 1,368 heads at $\sigma=0.62$ ($\rho=\hlEmuShareRho$); amber and blue: heads 4 and 7
  of Figure~\ref{fig:observation}. (b)~Relative output error of four mixing rules, 32 prompts. (c)~A table used where
  it was not fitted (light) against the table fitted there (dark); held-out: each prompt half's table on the other
  half. Appendix~\ref{app:emulation} expands each panel.
Data: results/heads_view/pi_vs_mass.csv + summary.csv (rho, recomputed), src/data/k1_1024_pi.npz + k1_1024.json
(errors via k1data, asserted), results/evidence/calib/k1_1024.npz + k1_1024_split_half_stats.npz (split-half rho,
asserted against the report), src/data/k1_1280_pi.npz + k1_1280.json (resolution transfer, asserted).
Macros available in headline.tex: \hlEmuShareRho 0.77, \hlEmuSplitHalfTwo 0.989, \hlEmuXferRatio 0.205,
\hlEmuErrHeadFour 0.200, \hlEmuXferRes 0.0589, \hlEmuOwnRes 0.0585, \hlEmuErr{Head,Layer,Shared,Switch}{Two,Four}.
The held-out pair (0.0574 in-sample vs 0.0576 cross-half, k1_1024.json cross_half_rel_err_per_head) has no macro; it
is printed only in the figure.
Rubric (0/1/2): one claim per panel B2 / A1 (A's grouped bars flatten the r = 2 differences); values on bars, best
bold B1 / A1; text >= 6 pt B1 / A1; spines/grid B1 / A1; palette B1 / A1; words <= 60 B1 (30) / A1 (25); layout
density B1 / A1. Mean B 1.1, A 1.0. Kept B. (A hybrid C, B's (b) with A's split-half histogram, crowded (b) and
clipped (c) at this width; the split-half histogram stays in appendix pi_heads.)

## Appendix method zoom: paper/figures/v6/method_zoom.pdf (5.5 x 3.42 in; label fig:method_zoom unchanged)
New path, same stem (announce: appendix.tex should include figures/v6/method_zoom.pdf). Source v6_method_zoom.tex on
Figure 3's preamble (same fonts, 9 pt titles, query marker, block and arrow styles, near/far vocabulary). Content as
v5.1 D.1 with fewer in-figure equations: (a) Far view, toy grid -> n0 x n0 keys with kappa(b), c(p); (b) Near view,
two query tiles, windows (w = n0 filled, 2 n0 dotted, clamp dashed), bridge dots, block mask (text / bridge / window);
(c) Gate and mix: head 24 of block 12 at r = 2 (dashed) and r = 4 (solid) with the weight applied at the 20 post-gate
steps (amber: the four wide-window steps; green: native), grey band sigma > sigma_g, and the mix O^far, O^near -> O_i;
(d) the r = 4 table of all 57 blocks (fig_method_zoom.py's strip, unchanged) with a raster colour bar (the old TikZ
shading rendered wrongly in MuPDF). Data: paper/figures/method_zoom/{data.tex, pi_strip.png, block_mask.png} from the
existing fig_method_zoom.py (read only) and method_assets/picmap.png. 32 words, text 6.2 to 9 pt.
Why head 24 in (c), not Figure 3's head 10: head 10's r = 2 and r = 4 weights are nearly equal and flat in sigma,
which hides what (c) shows (the table depends on r; window steps read between the tables). Head 24 is
fig_method_zoom.py's example (largest visible r gap, rule in method_zoom_provenance.json); the caption must name it.
Proposed caption:
  \textbf{The operators of Figure~\ref{fig:method} at component level.} Toy grid: $H=W=24$ tokens, $s=4$, $n_0=6$,
  $t=2$. (a)~The far view keys each block by its centre token (Equation~\ref{eq:glob}). (b)~A query tile's near view:
  text keys, bridge tokens (dots) and a border-clamped window, with its block mask (Equations~\ref{eq:loc}
  and~\ref{eq:bridge}). (c)~Head 24 of block 12 at $r=2$ (dashed) and $r=4$ (solid); dots: the weight applied at each
  post-gate step at $4096^2$ (amber: wide windows), and the mix (Equation~\ref{eq:mix}). (d)~The $r=4$ table,
  $57\times24$ heads by 14 noise bins.
Note: the zoom writes O^far, O^near and K^loc; if the text renames the operators, tell me and I will change the labels.

## Requests to other owners
- text: Figure 4 is new (suggest \label{fig:emulation}, wrapfigure is not advised at 5.5 x 1.6 in; full width beside
  3.2). Fig 2 caption no longer says "m_i"; panel (a) now uses the share within 16 tokens, the same quantity as (b).
  Figure 3 and 4 captions above use macros where they exist.
- tablesplots: none needed (macros exist). If (c) of Figure 4 is quoted in prose, add a macro for the held-out error
  pair (k1_1024.json scales/2: rel_err/per_head 0.0574 and cross_half_rel_err_per_head 0.0576).

## Not done yet
- Appendix: method zoom redraw, analysis plots restyle (heads_view, pi_heads, rel_err, pi_sigma, pi_heatmap,
  reach_tables).
