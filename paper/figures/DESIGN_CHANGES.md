# Figure design pass, 2026-10-18: changes the lead must know about

All figures now share one visual language, defined in `src/scafig.py` (matplotlib) and
`src/sca_colors.tex` (TikZ): Helvetica text with Computer Modern math, labels at 8.5 pt and
ticks at 7.5 pt, bold lowercase panel letters, global branch blue `#2F6DB3`, local branch
amber `#E08A2B`, the local weight pi on a blue to neutral to amber map, SCA as the single
green accent `#13865F`, baselines in neutral greys, and crop windows plus their zoom tiles in
one yellow `#FFC21A`. Captions in snippets are 2 to 4 lines with the takeaway first.
Nothing in `main.tex` was edited.

## Main-text figures

| Figure | Path | Old | New | Action for the lead |
|---|---|---|---|---|
| `fig:teaser` | `figures/snippets/teaser_six.tex` (unchanged path and label) | in-flow `center`/`minipage` with 12 JPEGs, images 5.5 x 2.35 in, 5-line caption | `figure*[t]` including `figures/teaser.pdf` (5.5 x 3.19 in, about 3.75 in with its 4-line caption, about 34% of a page) | Intended placement: keep `\input` right after the abstract; as a top float it lands at the top of page 2. Nothing to change unless you want it elsewhere. |
| `fig:method` | `figures/method.pdf` (unchanged path and label) | 5.5 x 5.0 in matplotlib drawing, panels a to c, 12-line caption | 5.5 x 2.84 in TikZ (paper fonts), panels a and b; the old panel c schedule is now a strip inside b | Saves about 2.2 in of main text. Caption already rewritten in `snippets/method.tex`. Line 192's "Figure~\ref{fig:method}a" still points at the real fitted-weight inset. |
| `fig:qual_main` | `figures/snippets/qual_main.tex` (unchanged path and label) | `figure*` with 6 prompts, about 5.4 in tall | `figure[t]` including `figures/qual_main.pdf`, 5.5 x 4.24 in, 4 of the same 6 prompts (dropped 8142852677968042257 sunset lake and 1052899493017962912 roast chickens); full attention as a thumbnail strip; AReA and SCA crops widened to 736x512 by the same edge-energy rule, at least 256 px from every border | Saves about 1.2 in. `src/fig_qual_main.py` writes the caption. |

Proposed captions (already in the snippets):

- `fig:teaser`: "SCA keeps layout and fine detail up to 6144x2048 without training and runs 2.85x faster than AReA at 4096^2. Top: curated SCA outputs with native-pixel zooms. Bottom: one curated prompt and the same 768x768 window for six methods, seed 0 (*UltraImage uses its two-stage defaults). Badges: batch-1 seconds per image on one GH200, where measured." The sentence about the earlier hero selection is gone.
- `fig:method`: "SCA fits one local/global weight per head and noise level at native resolution, then reuses the table at the target size. (a) At 1024^2, sums against the frozen model's own full attention give pi_h(sigma) in closed form; the inset shows real fitted values for block 29. (b) At the target, block-centre global and native-window local attention are mixed per head, and the first 8 of 28 steps use the global branch only. Grids are schematic."
- `fig:qual_main`: "At 4096^2 full attention collapses, AReA blurs or tiles fine structure, and SCA keeps both layout and detail. Seed 0; four curated prompts, hand-picked for variety from 191 evaluation prompts. Beside each AReA and SCA image is the same native 736x512 crop, auto-placed at maximal edge energy averaged over the three methods. Uncurated samples: Figure~\ref{fig:gallery_4096}." (written by `src/fig_qual_main.py`)

Timing rule: the teaser shows both kinds of seconds per image and labels every value. Clean
batch-1 medians after warm-up (`results/latency_curve_summary.csv`) cover full attention
(186.5 s), AReA (202.4 s) and SCA (71.1 s); DyPE (196.1 s), SEGA (190.5 s) and UltraImage
(308.5 s) show their 191-prompt campaign means on shared nodes. An unlabelled time is never shown.

## New and superseded scripts

- New: `src/fig_hero.py` (teaser, writes `src/teaser.tex`, `figures/teaser.pdf`, `figures/teaser/`),
  `src/fig_qual_main.py` (writes `src/qual_main.tex`, `figures/qual_main.pdf`,
  `figures/qual_main/cmp_*`), `src/scafig.py`, `src/sca_colors.tex`.
- Rewritten: `src/fig_method.py` now generates `src/method_diagram.tex` and compiles it with
  tectonic; run it with `uv run --with numpy --with matplotlib --with pymupdf` so `method.svg`
  is refreshed too.
- Superseded, do not run: `src/fig_teaser_six.py`, and the `main`/`qual_main` mode of
  `src/fig_qual.py` (it would overwrite `snippets/qual_main.tex`). Never run `fig_qual.py all`.
- Now unused by the paper, safe to drop from the source ZIP: `figures/teaser_six/*.jpg`
  (3.4 MB) and the non-`cmp_` files in `figures/qual_main/` (2.8 MB).

## Plots (same paths and labels; captions in snippets already shortened)

| Figure | Old size | New size | Notes |
|---|---|---|---|
| `fig:calib` (`calib.pdf`) | 5.5 x 3.55 in | 5.5 x 3.30 in | pi map, one colourbar (global 0, local 1); panel e is now grouped bars of extra error with the per-layer fit as a grey reference. |
| `fig:baseline_quality` (`baseline_deltas.pdf`) | 5.5 x 4.15 in | 5.5 x 1.88 in | One row of four panels, baselines named once; filled marker when the 95% interval excludes zero; statistics CSV byte-identical. |
| `fig:latency` (`latency.pdf`) | 5.5 x 2.34 in | 5.5 x 2.05 in | Clean batch-1 curves, plus hollow 46-prompt campaign-mean markers at 4096^2 (HRDiT 106.5 s, UltraImage 322.0 s, SCA 71.6 s) and an in-plot note (full attention: 6144^2 pending, 8192^2 not run by design); caption written by `src/fig_latency.py`; fill between AReA and SCA, fitted exponents (SCA N^1.00, AReA N^1.12; full attention N^1.51 in the caption), top row of speedups over AReA. |
| `fig:pisigma`, `fig:relerr`, `fig:heatmap` | | rel_err now built at 5.5 in | Same visual language; rule colours keep their meaning. |

Edits needed in `main.tex` (appendix figures whose captions live there):

1. `fig:relerr`: change `width=0.9\linewidth` to `width=\linewidth`.
2. `fig:pisigma` caption (its "(grey)" is now wrong):
   `\caption{\textbf{In output energy, no noise level prefers the global branch.} (a)~Fraction of the 1368 heads per $\pi$ bin and noise bin (shading), with the mean of $\pih$, the best shared $\pi(\sigma)$, and AReA's schedule read as a weight; emulated scale 2, 32 prompts. (b)~Relative output error per noise bin: local alone beats global alone everywhere, and per-head mixing beats both.}`
3. `fig:relerr` caption:
   `\caption{\textbf{Per-head weights explain full attention best, and transfer across scales.} Relative output error of each mixing rule at emulated scales 2 (a) and 4 (b), 32 prompts. ``Other scale'' applies the scale-4 table at scale 2 and vice versa; the best hard switch has $\sigma^\star=1$ and so equals local only. In-sample and held-out errors agree within 0.0003.}`
4. `fig:heatmap` caption (numbers printed by `fig_pi_heatmap.py`):
   `\caption{\textbf{Heads in the same block differ widely at every noise level.} $\pih$ of the scale-2 table for all 57 blocks and 24 heads of FLUX.1-dev, 32 prompts; heads are sorted within each block by their weight at $\sigma=0.54$, in the same order in every panel. The median within-block range is 0.66 to 0.84. Figure~\ref{fig:calib} shows the scale-4 table used at inference.}`
5. The unused fallback caption inside `\snippet{calib}{...}` in `main.tex` says every block spans almost purely global to almost purely local; only 17 to 24 of 57 blocks span from below 0.1 to above 0.9. Remove or fix that fallback.

## Appendix image grids (same paths, labels, prompts and crop windows)

- All ten grid snippets (`gallery_4096`, `gallery_4096_p2`, `gallery_4096x2048`,
  `gallery_6144x2048`, `ablation`, `failures`, `gallery_baselines`, `gallery_6144`,
  `gallery_8192_complete`, `gallery_flux2`) now `\input` a shared, guarded macro file
  `figures/snippets/_scagrid.tex`: vector crop windows in the zoom yellow over a dark casing,
  matching frames on crop tiles, ruled sans headers with SCA (ours) in the accent, one tile
  height per figure, 1 pt pair / 4 pt group / 2 pt row gutters, rotated "Full image" and
  "1:1 crop" row labels in the ablation and baseline grids. Captions are at most 4 lines.
- JPEGs re-encoded at quality 85, full images at 500 dpi of printed size, crops native:
  gallery assets fell from 33.0 MB to 22.3 MB. Fresh build with every current edit:
  28.9 MB (was 39.7 MB).
- `figures/failure/*.jpg` (tiles of the fallback `fig:failure` in `main.tex`) were re-made at
  406 px from the same PNGs by the new `fig_qual.py failure_fallback` mode; names unchanged.
- Each `fig_qual` grid now has `<figure>/provenance.json`; phase-2 manifests gained an
  `"outputs"` key only. `tests/test_gallery_readiness.py` passes (8 tests).
- Suggestions: align the ablation column "No gate" with Table 2's row "no opening phase (v0)";
  the `6144_complete` and `krea_complete` captions (not yet rendered) are still long.

## Appendix showcase `fig:hero` (same path `snippets/hero.tex`, same label)

- Rebuilt by the new `src/fig_showcase.py` as `figures/showcase.pdf` (5.5 x 4.63 in, 2.7 MB):
  seven curated SCA outputs (one 6144x2048, four 4096^2, two 4096x2048), none shared with the
  teaser, qual_main, galleries or the development set; badges and native-pixel insets in the
  teaser style; 3-line caption. Float changed from `[t]` to `[!ht]` so it stays below the
  "Qualitative galleries" heading.
- `fig_qual.py hero` (and `all`) would overwrite this snippet: retire that mode. `figures/hero/`
  (2.0 MB) is now unused, like `figures/teaser_six/` and the non-`cmp_` files in
  `figures/qual_main/`.
