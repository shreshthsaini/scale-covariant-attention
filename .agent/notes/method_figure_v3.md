# Method figure v3 (fig:method), 2026-11-07 night, paper owner

User directive: redraw the main method figure as a mathematical picture in the spirit of AReA's pipeline figure
(arXiv 2610.17227 Figure 4: dashed phase panels, Q x pooled K^T = cost bar, sigma timeline with real images; project
page: one formula per component card), but our own: the exact split O_full = m O_near + (1 - m) O_far as a picture
equation, the emulation (both views shrunk by r at 1024^2) as a visual mapping, the closed-form fit
pi* = clip(B/A, 0, 1) with its quadratic J(pi), and the progressive stages using the per-head mix.

Sources: paper/figures/src/method_v3_common.tex (macros), method_v3a.tex, method_v3b.tex (candidates),
method_v3.tex (chosen = A), fig_method_v3.py (assets in paper/figures/method_v3/, provenance.json there).
Real data in the figure: the 1024^2 canvas (emulation prompt 1, stand-in render, as in every earlier method figure),
the 4096^2 progressive SCA output (sheep and church, seed 0), the reach-table cards of block 12, and three real
quadratics J(pi) of block 12 at sigma = 0.618, r = 4 (heads 7, 14, 9: B/A = -0.185, 0.749, 1.017, so
pi* = 0, 0.749, 1), each rescaled to [0, 1].

## Critique against the gold-standard diagram recipe (0 below, 1 meets, 2 above)

| Item | A: aligned two-line equation, (b) and (c) as columns, (d) lane | B: one-line equation, (b) beside (c), (d) lane |
|---|---|---|
| Vector, photos embedded | 1 | 1 |
| Palette (4 hues: green SCA, amber window, blue grid, grey) | 1 | 1 |
| Words (<= 60 for a formulation figure) | 2 (15 words) | 1 (20 words) |
| Lettered panels, 9 pt titles, labels >= 6 pt | 1 | 1 |
| Data flowing through it | 2 (canvas, target, real quadratics, real table) | 2 |
| Reads as mathematics | 2 (near -> loc and far -> glob stacked: the approximation is visible column by column) | 1 (correspondence only via a legend) |
| Layout | 1 (dense, all space used, 196 pt tall) | 0 (row 2 half empty, clip formula collides with the table cards, 206 pt) |

Chosen: A. Iteration on A: operators re-spaced so "+" never touches "(1 - m_i)"; per-query and per-head carried by the
indices m_i and pi_h instead of text; (c) title shortened to fit; window fades lightened to 0.68 so the photo reads.
