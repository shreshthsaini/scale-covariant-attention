# Latest verified paper delivery, 2026-10-28 (6 pt figure-text pass)

- [Paper PDF](../../paper/main.pdf), 51 pages: main text pages 1 to 10 (Section 6 ends at y = 673 pt on page 10),
  reproducibility statement and references from page 11, appendix after.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 292 source files.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_mzzf_m71/` (`primary/`, `extracted/`, `standalone/`).

The source ZIP builds on its own from a fresh unzip (51 pages). Both builds have zero overfull boxes and zero
undefined references; the PDF text has no em dash, no section symbol and no "??". PDF size 27,736,505 bytes.

What changed in this pass (details: HANDOVER.md Section 19):
- Figure text floor. PyMuPDF finds no span under 6.0 pt in any of the 17 figure PDFs the paper places (all at
  scale 1.0). New `scafig.script_floor` raises matplotlib's mathtext script size, otherwise a fixed 0.7 of the
  parent: Figure 4 (heads_view, h of pi_h 5.6 to 6.05 pt), Figure 6 (paired_area, subscripts of MUSIQ_c, FID_p,
  KID_p 4.9 to 6.04 pt), Figure B.2 (pi_heads, sigma_g 4.9 to 6.04 pt, pi_h 5.6 to 6.9 pt; y-label pad 1.5 pt
  keeps the label inside the left edge) and Figure B.5 (pi_heatmap, 5.6 to 6.05 pt; the earlier review missed it).
  Figure 1's size badges and Figure 5's MUSIQ_c subscript went from 6 TeX pt (5.98 pt) to 6.1 TeX pt (6.08 pt);
  the badges use a text times sign so it scales with them. Data, figure sizes and layout are unchanged, and the
  teaser and zoom4k JPEG assets are byte-identical after regeneration.
- `fig_paired_area.py` expands the headline.tex macros before checking the main-text sentence and caption (the
  check broke when the prose moved to macros); its provenance now hashes headline.tex too.
- `teaser/provenance.json` regenerated: its stale note on prompt minus3268147817845827695 now matches fig_hero.py.
- Sections 4.1 and 4.2 open every finding paragraph with a bold claim sentence instead of a label. The AReA
  paragraph adds Table 1's before and after values (ImageReward 0.92 to 1.23, FID 145.8 to 138.1) to the paired
  intervals, Coverage quotes the 6144^2 ImageReward interval of Table K.1, and the progressive paragraph quotes the
  campaign times (49.5 and 110.4 s, headline macros). No reported number changed.
- Gold-standard rubric: 22 items, sum 27, mean 1.23, no 0 items (HANDOVER.md Section 19).
- Known and left: 5 pt second-level math scripts outside figures (body equations, algorithm comments such as
  1024^2, appendix table marks); the public code URL under the anonymous author block; Table J.1 has no HRDiT or
  SigMa rows; Table H.1's ranking note (results/published_compare/build.py).

PDF SHA256: `d6ffee017a9cf94f1910951e2b6bd928a2a5402f3a2f5ce002c61a9fbceb703f`.
ZIP SHA256: `8cd8c689d8cca09d99d2a4373693d372aeb1df8c6677627b3a4587f615d11d34`.
