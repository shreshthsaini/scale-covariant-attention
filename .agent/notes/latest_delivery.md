# Latest verified paper delivery, 2026-10-17 10:19 PDT

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

PDF SHA256: `13832c6ee2774271999010e6ad063177f6fd9908f6f91c225baf63d34a36fb82`.
ZIP SHA256: `cdb8c611a44864f3ba5dcba4f4cbc38498f5e2452856e0b618f87bb53847dccd`.

This is a verified revised draft, not project completion. E2 6K, E9 Krea,
full-attention 6K timing, repaired GPU smoke, final evaluation and complete
galleries remain pending. HRDiT and SigMa are being added as validated baseline
experiments; CPU preflight does not count as a GPU result.

SCA10 1077447 and SCA11 1077473 are queued. Durable continuation 1077042 is
queued with its dependency satisfied. Read HANDOVER.md for current ownership.
Do not duplicate the watcher. Render history is under
`/scratch/09032/saini_2/sca/paper_review_20261017/redesign_final`.
