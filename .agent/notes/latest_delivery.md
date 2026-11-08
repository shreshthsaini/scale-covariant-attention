# Latest verified paper delivery, 2026-11-08 03:20 CST (speed headline, attention-focus figure, newer-model switches)

- [Paper PDF](../../paper/main.pdf), 55 pages: main text pages 1 to 10 (Section 6 ends on page 10 with about four
  spare lines), reproducibility statement and references from page 11, appendix after. SHA256
  2ac3325c98f9b38ebc5ee030a2ca69dfa481803347de812b4779e372856d5d8c.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 425 source files (main.tex, appendix.tex, and only the
  snippets and figures the build reads), SHA256 6b48050d4cf3377db141a200a39107564e76f047cee0ee72933c08df907f1326.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_gvyduue4/`.

Both builds: zero overfull boxes, zero undefined references, and the fresh unzip builds on its own to the same 55 pages.
PDF text has no calib*, curat*, hand-pick*, "picked by eye", em dash, section symbol or "??"; no file name in the ZIP
contains calib or curat.

What changed (HANDOVER Section 37): Figure 2 (speed: ImageReward against seconds per image at 4096^2 with arrows to
progressive SCA and its speedups, clean over clean or campaign over campaign, plus clean latency against image side)
sits under the hero on page 1 and is cited in the abstract and introduction. Figure 3 (attention focus) shows full
attention of two heads of block 12 on a real 1024^2 FLUX.1-dev image (CPU attention capture, scripts/attn_focus.py),
split into near and far parts, with SCA's window and block-centre grid, reach emulation on the same image, and the
97.5% / 2.5% variance split as one bar. The 1,368-head curve bundle and scatter (heads_view) moved to Appendix B.
Sections 1-2 run near view, far view, reach, reach emulation. Newer-model switches (\resfluxtwo, \resqwen,
\resideogram) are wired and off; tested on fake snippets with all three on: Figure 2 stays on page 1 and the main
text still ends on page 10.

Render review: two cycles. Cycle 1: main pages by the paper owner (3 MUST: the 2.5 heading stranded at the bottom of page 5, fixed with needspace; Figure 5's caption overlapping text after the wrap shrank, fixed with a 21-line wrap; nested parentheses in related work); appendix by two reviewer subagents (MUST: Figure D.1 cited Equations 4 to 3 after the operator reorder, now 2 and 3 with 4 for panel a; 'randomly drawn' in Appendix L, gallery and FLUX.2 body text, reworded, captions keep the neutral 'drawn at random (selection seed 0)'; prompts recurring between Figures S.2/S.3 and the seeded galleries, now stated: the sixteen prompts were set later, so a prompt can appear in more than one figure). Also fixed: Figure B.1 caption (gate line and band explained), Figure D.1 placed after its section opening, Figure N.1 caption sign, FLUX.2 portrait phrasing, Figure 2 caption ties 195 prompts to ImageReward. Cycle 2 (paper owner: main pages and appendix pages 15, 19, 32, 40): no MUST item. Open SHOULD items, all pre-existing: blank space under Algorithms 4-5 (p22) and on some table and gallery pages, Table L.1 at 6 pt and 'MUSIQ, crops', Table D.1's first column, Table K.5 caption clause, Figure S.2 says sixteen prompts while showing eight (eight more in S.3), a results/ path in Appendix N prose, mathtext glyphs that copy as fractions in Figures B.1-B.3.
Previous deliveries: `delivery_yio2nt1w` (commit 29a6870, before the cycle-1 appendix fixes) and `/scratch/09032/saini_2/sca/paper_builds/delivery_mqhyl3id` (commit b131b66).
