# Figure 3, D.1 and calibration figures redesign (2026-11-02, Opus figure subagent)

## State
- Figure 3 (fig:method): rebuilt from scratch as candidate B, iterated twice. Source paper/figures/src/method.tex
  (self-contained TikZ), data and images from paper/figures/src/fig_method_flow.py into paper/figures/method_flow/
  (calib.jpg, noise.png, output.jpg, crop.jpg, data.tex), provenance paper/figures/method_provenance.json.
  PDF paper/figures/method.pdf (5.5 in x 128 pt, 0.32 of text width). Render: paper/figures/candidates/fig3_final_print300.png.
- Figure D.1 (fig:method_zoom): redrawn in the same language. Source paper/figures/src/method_zoom.tex, data from
  fig_method_zoom.py (example head now the Figure 3 head, block 12 head 24 one-based; read step 20). Render:
  paper/figures/candidates/figD1_print200.png.
- Candidates: paper/figures/candidates/fig3_candidate_{A,B}.pdf and *_print300.png (sources in scratch).
- NOT DONE: main.tex captions of fig:method and fig:method_zoom still describe the old figures (see next steps).
  heads_view (Figure 5) redraw was delegated to a background subagent on i617-044; the node ends 04:12, so its
  edits to fig_heads_view.py / heads_view.pdf may be partial and are NOT committed. Check git diff before using.

## Self-critique of the two Figure 3 candidates (gold-standard diagram recipe)
- A (L-shape: calibration column left, sampling row top right, mixing row bottom right). Meets the recipe on palette
  (green, amber, blue, grey), 0.8 pt rounded pale blocks, 0.6 pt orthogonal arrows, lettered panels, about 17 words,
  real data in every panel. Weaker: the calibration column is cramped (fit box and table compete with the key maps),
  the flow turns twice (down, then right), the table-to-bars arrow runs along the bottom, the right third of (c) is
  empty, and there is no zoom panel, so (c) reads as a separate row rather than the inside of one sampling step.
- B (two left-to-right strips plus a pale green side panel, LILIE Figure 4 layout). One reading direction for (a)
  and (b) on a shared left edge; (c) reads as a zoom into one attention call and is fed directly by the table, so
  calibrate, weight, mix is one path. Weaker in v1: spare space at the bottom of (c), the sigma_g label crowded the
  (b) title, 6.5 pt numbers, a green-filled fit box competing with the green panel, the step tabs over the crop.
  Chosen: B. Iteration 1: all text >= 7 pt, fit box white with green border, coloured operator edges (blue
  1 - pi_h, amber pi_h), sigma_g inside the plot, "step" and "heads" labels. Iteration 2: noise aligned with the
  plot, "13+" tab moved off the spire, thinner frame halos, "head" label under the dot column, r = 4 label.

## Facts encoded (all checked by the scripts)
- Key maps and table excerpt at emulated ratio r = 4 (to scale on 64 x 64; window 16, lattice 16 x 16). Table rows:
  heads of Figure 5 (blocks/heads 7/10, 25/21, 12/24 one-based), sigma = 0.105, 0.536, 0.896 shown as 0.1, 0.5, 0.9.
- Fit parabola: J(pi)/E for block 12 head 24, r = 4, sigma bin 7; pi* = B/A = 0.4046 = outlined table cell 0.40.
- pi_h curve: block 12 head 24 over the 28 steps at 4096^2; the per-step mean over all 1368 heads reproduces the
  stored pi_trace of the output image at all 20 post-gate steps (|d| < 1e-9).
- Bars: the 24 heads of block 12 at step 20 (sigma 0.5994, r = 4), sorted; outlined head value 0.4275 = curve dot.
- Calibration thumbnail is a stand-in (calibration prompt calib00, seed 0, SCA 2048^2 smoke output resized to 1024):
  the calibration pass saved statistics, not images. Replace with a true full-attention 1024^2 sample if a GPU run
  happens (same prompt and seed).
- Window schedule is hand-set (128, 112, 96, 80, then 64); only the weights are calibrated. The figure follows
  the paper, not the brief's "calibrated schedule" wording.

## Next steps
1. Replace the fig:method caption in main.tex with:
   \textbf{Calibrate once, then mix per head.} (a)~At $1024^2$, $\pi_h$ is fit to full attention per head,
   $\sigma$, and $r$; (b)~at $4096^2$, sampling opens globally, shrinks the window over steps 9 to 12, and
   (c)~mixes both operators by $\pi_h$ at $r=256/w$. Figure~\ref{fig:method_zoom} draws the operators, the bridge
   tokens, and the mix at component level.
2. Replace the fig:method_zoom caption (panels: (a) global operator, (b) local operator with block mask,
   (c) gate and mix with pi of the Figure 3 head vs sigma rising 0 to 1, (d) r = 4 table, sigma rising upward).
3. main.tex text (owner only): "Figures~\ref{fig:method}b" in Appendix B's opening now means sampling; calibration
   is panel (a).
4. Finish and review heads_view (Figure 5), rebuild with paper/build_delivery.py, check pages 3 and the D.1 page.
