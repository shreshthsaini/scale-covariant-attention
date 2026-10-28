# Latest verified paper delivery, 2026-10-28 (AReA presentation pass)

- [Paper PDF](../../paper/main.pdf), 51 pages: main text pages 1 to 10 (Section 6 ends at y = 662 pt on page 10),
  reproducibility statement and references from page 11, appendix after.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 292 source files.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_3ip7uv4h/` (`primary/`, `extracted/`, `standalone/`).

The source ZIP builds on its own from a fresh unzip (51 pages). Both builds have zero overfull boxes and zero
undefined references; the PDF text has no em dash, no section symbol and no "??". PDF size 27,734,255 bytes.

What changed in this pass (details: HANDOVER.md Section 19, .agent/notes/state_area_pass_20261027.md):
- New Figure 4 (paper/figures/heads_view.pdf, results/heads_view): key sets on a 4096^2 output, measured reach of
  three heads, their pi(sigma), pi against near-field share over 1368 heads (r = 0.77, scale 2, 16 tokens,
  sigma = 0.62). The old pi_heads figure is now Figure B.2.
- Tables 1 to 4 are \input from paper/figures/snippets/main_tables (gen_main_tables.py); headline numbers in the
  prose are macros from headline.tex. PDF text was identical before and after the swap. Table 4 now marks
  significantly worse cells (sigma_g = 0.85 on HPS and MUSIQ_c); Tables 1 and 2 carry direction arrows.
- Related-work table (Table 5) fits the text width; related-work prose shorter, all 27 citations kept.
- Page fit: prose tightened in Sections 1 to 6; the published-numbers table moved from Section 4 to Appendix H
  (Table H.1); Figure 6 wraps beside the Metrics paragraph; the qualitative grid is now Figure 5.
- Two render reviews (fresh Opus reviewers) fixed stale appendix openings (B, D, K, L), a wrong figure pointer,
  the compute appendix's campaign times (now 72, 264 and 196 s from results/a4k195_main), the Table 2 caption,
  and the Table 5 offsets note (image keys only, Remark 1).
- Known and left: figure subscripts under 6 pt in Figures 4, 6 and B.2; the public code URL under the anonymous
  author block; Table J.1 has no HRDiT or SigMa rows; Table H.1's ranking note (results/published_compare/build.py).

PDF SHA256: `a7077ccac9d219d3ee11396326d86955204b9d29b527ac1ca7e6d123b7201074`.
ZIP SHA256: `3c930b53c3eb6e8212b2f1057601ac0452da2efb84e670e2743f659b4beed634`.
