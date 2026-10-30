# Latest verified paper delivery, 2026-10-30 09:20 CDT (rewrite around the scheduled operator, sca_v2)

- [Paper PDF](../../paper/main.pdf), 53 pages: main text pages 1 to 10 (Section 6 ends on page 10), reproducibility
  statement and references from page 11, appendix after. SHA256 0569a016201c1774a00158486be47d3561274a59bd425e4c6397600031026fc4,
  27771842 bytes.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 308 source files, SHA256 99ae690c15898bcd3bacebea62bc2d95bdc0830c7f1cc266e7d00d99f31a0794.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_32xjiemn/` (`primary/`, `extracted/`, `standalone/`).

Both builds have zero overfull boxes and zero undefined references; the fresh unzip builds on its own (53 pages);
the PDF text has no em dash, no section symbol and no "??".

Numbers folded in (final operator, label sca_v2): a4k195_main, 2048x4096, 2048x6144, seeds 1 and 2, progressive,
Krea-195, latency_curve_v2, and the 46-prompt variants dup_i_160 / bridge32 / ext75. PENDING (grey cells with caption
notes): hrgen1k_dev_sca_v2 and hrgen1k_krea_sca_v2 (generation 85/100 and 0/100 tasks at 09:13). To fold them in:
`bash paper/figures/src/swap_all.sh`, then the hand edits listed in .agent/notes/paper_v2_rewrite_20261029.md
(Swap procedure), build_delivery.py, commit, push.

Commits of this pass: eccbd70 (structure), 597d7cc (main and non-square swap, CPU reference bootstraps, images),
5cf219f (seed and progressive CSVs), 548520f (curation), 64a0a9d (render review 2, rubric), 58eb7e1 (Krea and
latency), ab5a41b (galleries), then this delivery commit. Rubric: .agent/notes/rubric_20261030.md (mean 1.23, no 0).
