# Latest verified paper delivery, 2026-10-17 10:32 PDT

- [Paper PDF](../../paper/main.pdf), 38 pages.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 473 source files.
- [Build receipt](../../paper/delivery_receipt.json).
- [Independent appearance checks](../../results/evidence/presentation_review_20261017.json).
- [Revamp and numeric evidence](../../results/evidence/paper_revamp_20261017.json).

The source ZIP builds independently. All 38 pages render identically at 72 dpi.
Changed pages were inspected at 144 dpi. No overfull boxes or undefined references.
The revised method diagram has PDF, editable SVG, source and data provenance.
The main paper now shows six methods, paired quality intervals, a six-method
portrait with identical crop coordinates, and the full verified latency curve.
Main results and ablations use readable compact tables; full metrics remain in
the appendix. The ablation data font increased from 6.96 pt to 8.97 pt.

Verified AReA/SCA medians are 482.3/161.6 seconds at 6K and 952.0/287.6 seconds
at 8K: speedups 2.99x and 3.31x. Timing logs identify NVIDIA GH200 120GB.
All 30 compact main-table values and headline deltas passed independent audit.
All 42 retained ablation values exactly match the previously audited table.
The figures retain UltraImage's preference-score advantages and unresolved
comparisons. The portrait is explicitly curated, not a random sample.

PDF SHA256: `abf64459d0aae84a207b592ab53f0be80cbd503c04f68a3f6d81fcf671b4ff3e`.
ZIP SHA256: `a9d7fc5c03ca6552a3a46dd36a2d83c75ebe73362bc1c06dbf68668ae4c10588`.

This is a verified revised draft, not project completion. E2 6K, E9 Krea,
full-attention 6K timing, repaired GPU smoke, final evaluation and complete
galleries remain pending. HRDiT and SigMa are being added as validated baseline
experiments; two frozen smoke tasks are queued after18CPU checks. CPU preflight
does not count as a GPU result.

At 17:47:50 UTC, SCA10 1077447 and SCA11 1077473 remain queued. CPU continuation
1077042 completed baseline promotion/evaluation hardening with 28 combined tests.
Its recorded successor 1077572 is pending on dependency; the controller owns it.
Evidence: `results/evidence/continuation/unit_validation_1077042.json`.
The PDF and source ZIP above remain unchanged. Read HANDOVER.md for ownership.
Do not duplicate the watcher. Render history is under
`/scratch/09032/saini_2/sca/paper_review_20261017/redesign_final_bibliography`.

## 2026-10-19 00:15 delivery (Claude operator)
- paper/main.pdf: 42 pages, main text 9 pages, 24.9 MB, SHA256 785f1947d36dfb73a32aaa71b53aeaa3d765a02b1effb2058afd5a29bd3aa123
- paper/sca_overleaf.zip: SHA256 1b7e79c96faa6d8085763c5a15bb48b7fa9c1ec19c71554d3fdbd30141b28c63
- Fresh-unzip build: /scratch/09032/saini_2/sca/zipcheck/fable_fix_20261019_001102/delivery_hmyrl93d (42 = 42 pages, 0 overfull, 0 undefined)
- Content: supervisor (Fable) fixes, 8-method Table 2, corrected HRDiT timing, calibration panel restored in Fig 2.
- Pending rows: HRDiT 191, HRDiT clean timing, SCA 42 steps, progressive SCA, E2 6144/8192, E9 Krea.

## 2026-10-19 03:50 delivery (final text polish, Claude subagent)
- paper/main.pdf: 47 pages, main text 9 pages (references begin on page 9), 26.1 MB (26,102,171 bytes), SHA256 58be9fdab82fe4dc2669caa2fbe434dd28ab9bed6e3770b394f31fb0bcf42685
- paper/sca_overleaf.zip: 273 source files, SHA256 259fc17fd0d0e81db24918f70f2acd0f638fa6e5fdde1967d1ce14a14b5fc78a
- Fresh-unzip build: /scratch/09032/saini_2/sca/zipcheck/final2_20261019_034855/delivery_3fjwxmt_ (47 = 47 pages, 0 overfull, 0 undefined). No red markers in any page.
- Text: abstract 169 words with three numbers; HRDiT overclaim replaced by "matches HRDiT's preference scores (no per-image metric significantly worse; HPSv2 and ARNIQA significantly better)" in abstract, contribution 4, Section 4.1, conclusion and the Table 23 caption; Section 4.1 at 54% and 4.3 at 58% of their former word counts; related work in three paragraphs; limitations 12 lines.
- New numbers: HRDiT corrected patch FID 103.9 (cached crop features via repro/evaluate_corrected.py, output /scratch/09032/saini_2/tmp/tmp/claude-883311/hrdit_fidp/hrdit191_corrected; not yet archived under results/). Full attention 762.7 s at 6144^2 (median of the three timed images after the excluded warm-up, results/latency_curve_batch1.csv). Table 2 FID marks for 42 steps and progressive from results/progressive46/base_paired_deltas_distribution.csv. Budget-run patch FIDs 175.1 and 167.2 from results/evidence/distributions/checks/budget_base_rows_*patch_fid*.csv.
- Figures: only caption text and the latency data point changed (generators updated to match); every includegraphics path and label is unchanged for the restyle. The quality-versus-time scatter stays in Appendix J.
