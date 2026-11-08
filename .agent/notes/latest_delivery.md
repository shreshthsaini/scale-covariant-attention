# Latest verified paper delivery, 2026-11-07 22:15 CST (no "calibration", no curation language, method figure v3)

- [Paper PDF](../../paper/main.pdf), 54 pages: main text pages 1 to 10 (Section 6 ends on page 10 with
  two spare lines), reproducibility statement and references from page 11, appendix after. SHA256
  70bd495900b91e1fc0185db03503a28ebbd0180c4139d4bfb90cdb80d3b895b0.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 421 source files (main.tex, appendix.tex, and only the
  snippets and figures the build reads), SHA256 403a408b2cdaca3e8fd984f571e3643ffe7b783cad69c17208bcb68c4b82b5c9.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_mqhyl3id/`.

Both builds: zero overfull boxes, zero undefined references, and the fresh unzip builds on its own to the same
54 pages. PDF text has no calib*, curat*, hand-pick*, "picked by eye", em dash, section symbol or "??";
no file name or .tex file in the ZIP contains calib or curat. Terms: the procedure is reach emulation, its output the
reach table (both defined once in the introduction). Figure 2 is the new mathematical method figure
(figures/method_v3.pdf). The new-model showcase figure is wired but off (\showcasefalse in main.tex): at delivery
web/build/showcase_manifest.json held 23 FLUX.1-dev images and no FLUX.2-dev or Qwen-Image image. When those land, run
paper/figures/src/fig_showcase_models.py and set \showcasetrue (tested on fake manifests of both schemas: the main
text still ends on page 10, 0 overfull). Render review: two cycles (main pages by the paper owner, appendix pages by
two reviewer subagents per cycle). Cycle 1 found 3 MUST items (pages 11 to 32) and 16 SHOULD items; cycle 2 found no
MUST item. Open by choice: the size of Figure T.1, free space on gallery pages 43, 47, 48 and 51 to 53, and the code
release's own calib file names (sca/calib.py, scripts/calibrate.py, scripts/calib_prompts.jsonl, calib/), which the
paper no longer prints. Previous delivery: `/scratch/09032/saini_2/sca/paper_builds/delivery_rjy3ai7t` (commit
6ad15ae). HANDOVER Section 33 has the details.
