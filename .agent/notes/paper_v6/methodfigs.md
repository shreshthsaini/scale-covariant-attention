# methodfigs (v6): Figures 2, 3, 4, appendix method zoom and analysis plots

Status 2026-11-10: done (round 2 after the lead's review: Figure 3 rebuilt as two bands, Figure 2 (c) and
Figure 4 (c) fixed). Main figures 2, 3, 4, the method zoom and the six Appendix B plots are installed under
paper/figures/v6/ (same stems). All pass the PyMuPDF check (no span under 6 pt) and were read at print size in an ICLR
test page (/scratch/09032/saini_2/sca/paper_builds/v6_methodfigs_1/testdoc/test_v6figs.pdf and test_v6app.pdf).
LIVE NOW: appendix.tex's \apxgraphic already picks up v6/heads_view, pi_heads, pi_heatmap, rel_err, pi_sigma and
method_zoom, so their v5.1 captions are mismatched until the appendix owner applies the fixes listed below.
Builds and renders: /scratch/09032/saini_2/sca/paper_builds/v6_methodfigs_1/ (png/, obs/, emu/, tikz/out/, testdoc/).

## New files (all mine; announce)
- Scripts (paper/figures/src): v6_attn_extract.py (data), v6_method_assets.py (method assets), v6_method_common.tex,
  v6_method_a.tex, v6_method_b.tex, v6_method_c.tex (TikZ; C is kept), v6_tikz_build.py (compiles TikZ in
  scratch, `--install` copies the kept ones),
  v6_observation.py, v6_emulation.py, v6_figcheck.py (PyMuPDF check: size, smallest span, words, PNG).
- Data: paper/figures/src/data/v6_attn_block12.npz + .json (1.1 MB): full-attention rows of all 24 heads of block 12 at
  query (43, 11) of the 1024^2 capture at steps 8, 20, 26 (sigma 0.897, 0.612, 0.327), plus the r = 4 near and far
  views at step 20. Read from the CPU capture /scratch/09032/saini_2/sca/out/attn_focus/sheep_church_seed0 (no GPU).
- Also: v6_method_zoom.tex (zoom TikZ, built by v6_tikz_build.py), v6_app_{heads_view,pi_heads,rel_err,pi_sigma,
  pi_heatmap,reach_tables}.py (Appendix B plots; each writes v6/<stem>.pdf and v6/<stem>_provenance.json).
- Outputs (paper/figures/v6): observation.pdf, method.pdf, emulation.pdf, method_zoom.pdf, heads_view.pdf,
  pi_heads.pdf, rel_err.pdf, pi_sigma.pdf, pi_heatmap.pdf, reach_tables.pdf, their *_provenance.json, and
  method_assets/ (images, data.tex, picmap.png, provenance.json).
- Rebuild: `PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python; cd paper/figures/src;
  $PY -I v6_observation.py; $PY -I v6_emulation.py; $PY -I v6_method_assets.py; $PY -I v6_tikz_build.py --install`.

## Figure 2: paper/figures/v6/observation.pdf (5.5 x 1.9 in, full width, label fig:observation unchanged)
Claim: how far a head looks is set by the head, not by the denoising step.
Panels: (a) two heads x three noise levels, one query, real maps; (b) all 1,368 heads as lines across the 11 post-gate
noise bins, heads 4 and 7 bold, head average black; (c) variance split drawn large, every label outside the bars and
one per line (round 2): "head 97.5%" above the bar, a zoom wedge labelled "noise level 2.5%" into the magnified noise
share, and under it "shared schedule 0.7%" (black: the part of the 2.5% a schedule shared by all heads can follow, the
head-averaged curve of (b)); then FLUX.1-Krea-dev 97.6% and FLUX.2-dev 99.1%. 30 words, text 7 to 8.5 pt.
Final caption:
  \textbf{Reach belongs to the head, not to the denoising step.} (a)~Full attention of heads 4 and 7 of block 12 at
  one query of a $1024^2$ image, at three noise levels: amber within 16 tokens of the query (dashed), blue beyond, with
  that near share. (b)~Near share of all 1,368 FLUX.1-dev heads after the first 8 steps, 32 prompts. (c)~Variance of
  the near share; a schedule shared by all heads can follow only the head average (Table~\ref{tab:variance_split}).
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

## Figure 3: paper/figures/v6/method.pdf (5.5 x 2.46 in, full width, label fig:method unchanged)
ROUND 2 (lead review 2026-11-10): kept candidate C, source v6_method_c.tex (B stays as v6_method_b.tex for reference).
Claim: each head's reach is measured once at 1024^2, in closed form against the model's own full attention, stored in
one table, and read at any size and sampler stage.
Layout: two bands read left to right, green band titles at the left.
  Band 1 "Measure once at 1024^2": (a) Reach emulation: full attention of head 10 of block 12 at one query of the
  1024^2 stage (the reference) ~ pi [window / r] + (1 - pi) [grid / r], each view with its real attention; the real
  parabola J(pi) (32 emulation prompts, sigma 0.62, r = 4) and pi* = clip(B/A, 0, 1); then (b) Reach table: block
  12's 24 heads (rows) by 13 noise levels (columns, sigma falling left to right), r = 4 sheet in front of r = 2,
  pi* outlined in green, rows of heads 4 and 7 marked. The table is the hinge: it sits at the right end of band 1 and
  reaches down to the sampler strip of band 2 (short arrows r = 2, r = 4), so no long return arrow.
  Band 2 "Generate at any size": (c) the 4096^2 output; head 4 = 0.93 x window + 0.07 x grid, head 7 = 0.00 x window +
  1.00 x grid (table weights at r = 4, sigma 0.62, printed in chips coloured like their table cells; each view drawn
  at an opacity that follows its weight); the progressive stages 1024^2 (full) -> 2048^2 -> 4096^2.
  22 words, four hues, text 6.6 to 9 pt (lead asked >= 6.5 pt).
Image: one image throughout, the run whose 1024^2 stage was captured (prompt a4k_eval_4096_minus8343371271909676658,
seed 0): band 1 its 1024^2 stage, band 2 its 4096^2 progressive SCA output, both lightly desaturated (half colour,
lifted 16% towards white). CORRECTION of my first report: there is no brown/white sheep mismatch. In colour the
1024^2 stage and the 4096^2 output match (same beige sheep, church, orange trees; contact sheet
/scratch/09032/saini_2/sca/paper_builds/v6_methodfigs_1/cand4096.png); the "white sheep" came from my own light-grey
rendering of the 1024^2 tiles in round 1. No image was re-chosen.
Heads: head 10 (band 1) by the rule "share closest to one half inside the r = 4 window at this query"; heads 4 and 7
(band 2) are Figure 2's heads (highest and lowest 32-prompt near share in block 12), and their r = 4 weights do show
the contrast (0.928 and 0.000; asserted in v6_method_assets.py).
Two-sentence test (figure alone): "At 1024^2 each head's full attention is fitted, in closed form, by a weighted mix of
a window and a grid shrunk by r, and the weight goes into a table of heads by noise level. At 4096^2 each head mixes
its native window and grid by that weight (head 4 mostly window, head 7 only grid), and each sampler stage reads the
table at its ratio r." Passes.
Rubric C / B (0/1/2): one idea readable without text 2 / 1; words 15-30 1 (22) / 1 (17); panels lettered in reading
order 1 / 0 (B read a, c, b); <= 4 hues 1 / 1; real data flowing through (one recognisable image, real attention, J,
table, per-head weights) 2 / 1 (B's images near white); arrows short and orthogonal 1 / 0 (B's long return arrow);
blocks per recipe 1 / 1; text >= 6.5 pt 1 / 0 (B 6.18); width filled, no dead quarter 1 / 0; tied to Figure 2 2 / 0.
Mean C 1.3, B 0.5. Kept C.
Final caption:
  \textbf{Reach is measured once at $1024^2$ and read at any size.} (a)~Full attention of one head at $1024^2$ is
  fitted by a mix of its window and grid views shrunk by $r$; the fit has a closed form. (b)~The weight fills one cell
  of the reach table: heads by noise level, one sheet per $r$. (c)~At $4096^2$, head 4 of
  Figure~\ref{fig:observation} mixes mostly its window and head 7 only its grid; each sampler stage reads the table at
  its $r$. Figure~\ref{fig:method_zoom} zooms in.
  (4 lines in the ICLR test page.) Optional addition if space allows: "Head 10 of block 12; the table is fitted on 32
  emulation prompts." Single-pass sampling (reads r(sigma)) is shown in Figure D.1 (c), not here.
Data: method_assets/ (v6_method_assets.py): c_full/c_near/c_far.png (1024^2 tiles), c_img4096.jpg, table_r2/r4.png,
data.tex (geometry, J curve, table cell, \PiFour 0.93, \PiSeven 0.00, chip colours, marked rows).

## Figure 4: paper/figures/v6/emulation.pdf (5.5 x 1.6 in, full width; new label, suggest fig:emulation)
Claim: the fitted weight measures the head's reach, per-head weights fit full attention best at both ratios, and one
table transfers across prompts, ratios and resolutions.
Panels: (a) pi_h (ratio-2 table) vs near share, 1,368 heads at sigma 0.62, rho 0.77, heads 4 and 7 coloured; (b)
relative output error per head / per layer / shared / best hard switch at r = 2 and r = 4 (two small panels, values
printed, per head green and bold); (c) own table (dark green) vs a table used where it was not fitted (light green)
on one error axis: held-out prompt half (0.0574 vs 0.0576, label carries the split-half rho 0.989), ratio-2 table at
r = 4 (0.200 vs 0.205), 1024^2 table at 1280^2 (0.0585 vs 0.0589); round 2: a two-entry legend above the bars,
"fitted there" (dark) and "transferred" (light). 33 words, text 6.04 to 8.5 pt.
Final caption:
  \textbf{The fitted weight measures reach, fits full attention best, and transfers.} (a)~$\pi_h$ of the ratio-2
  table against the near share, 1,368 heads at $\sigma=0.62$ ($\rho=\hlEmuShareRho$); amber and blue: heads 4 and 7
  of Figure~\ref{fig:observation}. (b)~Relative output error of four mixing rules, 32 prompts. (c)~Each table where
  it was fitted and where it is transferred; held-out: each prompt half's table on the other half.
  Appendix~\ref{app:emulation} expands each panel.
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
(c) Gate and mix (sigma falls left to right, as everywhere in v6): head 24 of block 12 at r = 2 (dashed) and r = 4
(solid) with the weight applied at the 20 post-gate
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
  post-gate step at $4096^2$ (amber: wide windows; $\sigma$ falls left to right), and the mix
  (Equation~\ref{eq:mix}). (d)~The $r=4$ table, $57\times24$ heads by 14 noise bins.
Note: the zoom writes O^far, O^near and K^loc; if the text renames the operators, tell me and I will change the labels.

## Appendix B analysis plots (restyled; same stems under paper/figures/v6/; scripts v6_app_<stem>.py)
Restyle by a subagent from my brief, reviewed by me (all six read at 300 dpi and in the test page). Data code is the
old scripts' (imported or copied verbatim, every original assertion kept; no assertion failed); presentation only.
Common: "(a) Name" titles, near/far wording, "fitted weight pi_h", "near share", ratio r (not "scale"), noise-level
axes fall left to right with "-> denoising" (as Figure 2), block numbers one-based, values printed, one SCA green.
| stem | size (in) | words | min pt | change |
|---|---|---|---|---|
| heads_view | 5.5 x 1.62 | 25 | 6.47 | near/far chips on the photo; heads near/mid/far; (d) x label "near share" |
| pi_heads | 5.5 x 1.34 (was 1.22) | 14 | 6.04 | titles "Blocks 1-19 / 20-38 / 39-57" (stream type now only in caption) |
| rel_err | 5.5 x 1.62 | 21 | 6.5 | "Ratio 2/4"; "per head, other ratio", "near only", "far only"; coarse rules one grey |
| pi_sigma | 5.5 x 2.05 | 32 | 7.0 | direct labels "per-head mean", "shared", "AReA switch" (AReA purple) |
| pi_heatmap | 4.95 x 2.78 | 16 | 6.47 | built at 0.9 width (as included); panels now sigma 0.97, 0.54, 0.10 top to bottom |
| reach_tables | 5.5 x 3.50 (was 3.30) | 46 | 6.04 | (a-c) now sigma 0.83, 0.46, 0.10; (e) grouped bars "scored at ..." instead of rotated labels; "far only" band |
heads_view note: the old --no-results-csv check compares results/heads_view/SHA256SUMS byte for byte and fails today
because that file gained two lines on Nov 7; v6_app_heads_view.py recomputes the four CSVs into scratch, requires them
byte-identical to results/heads_view (they are) and their checksum lines to appear in the archived SHA256SUMS. Nothing
was written to results/ or to the v5.1 PDFs (mtimes checked).

## Requests to other owners
- appendix (live caption mismatches, v6 figures already included):
  * fig:heads: "Local window (amber)" -> "Near window (amber)"; the blue keys are the far view; "share within 16 tokens"
    -> "near share"; "global-only opening" -> "far-only opening"; (c) sigma now falls left to right.
  * fig:pi_heads: "(a to c) ... blocks 1 to 19 (double-stream), 20 to 38 and 39 to 57 (single-stream)"; "scale 2" ->
    "ratio 2"; rho can be \hlEmuSplitHalfTwo.
  * fig:relerr: "emulated scales 2 (a) and 4 (b)" -> "ratios 2 (a) and 4 (b)"; "other scale" -> "other ratio";
    "equals local only" -> "equals near only".
  * fig:pisigma: "scale 2" -> "ratio 2"; "(b) local beats global everywhere" -> "near beats far everywhere"; sigma now
    falls left to right (sigma = 1, pure noise, is at the left).
  * fig:heatmap: "at sigma = 0.10 (a), 0.54 (b), and 0.97 (c)" -> "at sigma = 0.97 (a), 0.54 (b), and 0.10 (c)";
    "scale-2" -> "ratio-2".
  * fig:method_zoom: replace the caption with the one proposed above (head 24 named, near/far, sigma direction).
  * fig:reach_tables: figures/snippets/reach_tables.tex (written by the old fig_reach_tables.py) still includes
    figures/reach_tables.pdf. Switch it to \apxgraphic{reach_tables} (or edit the path to figures/v6/) and change the
    caption to: "(a to c)~$\pih$ of the ratio-4 reach table at $\sigma=0.83$, $0.46$, $0.10$; (d)~mean and
    interquartile range across heads of three tables, shaded: far-only opening. (e)~Excess error over each setting's
    own table: at most 5.4\% transferred, 12 to 24\% per layer." (the 5.4 and 11.6 to 24.0 are printed in (e)). I did
    not edit the shared snippet; say if you want me to.
- text: Figure 4 is new (suggest \label{fig:emulation}; full width beside Section 3.2; a wrapfigure is not advised at
  5.5 x 1.6 in). Figure 2 panel (a) now shows the share within 16 tokens (same quantity as (b)); the old "m_i" in the
  caption is gone. Captions above use existing macros (\hlVar*, \hlEmu*). The figures use "near view"/"far view" and
  the zoom writes O^far, O^near and K^loc; if the text renames the operators, tell me and I relabel.
- tablesplots: none required. If prose quotes Figure 4(c)'s held-out pair, add macros for k1_1024.json scales/2
  rel_err/per_head (0.0574) and cross_half_rel_err_per_head (0.0576).

## Not done or not verified
- Not compiled inside the real main.tex/appendix.tex (owned by others); verified only in my ICLR test pages.
- Single-pass sampling is not drawn in Figure 3 (the zoom's (c) shows it).
- Figure 3 (a) illustrates emulation on the 1024^2 stage of an evaluation prompt; the table itself comes from the 32
  emulation prompts (the caption must not imply the table was fitted on this image).
