# Latest verified paper delivery, 2026-10-19 (gold-standard fix pass)

- [Paper PDF](../../paper/main.pdf), 52 pages: main text pages 1 to 10, references from page 10, appendix from page 14.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 279 source files.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_e20al59t/` (`primary/`, `extracted/`, `standalone/`).

The source ZIP builds on its own from a fresh unzip (52 pages). Both builds have zero overfull
boxes and zero undefined references. PDF size 26,975,860 bytes (under the 35 MB cap).

What changed in this pass (scorecard zeros of HANDOVER Section 13):
- Page 1: hero under the abstract with a panel-walking caption; code URL in the abstract and
  conclusion through the `\codeurl` macro (GitHub link; swap for an anonymous mirror at submission).
- Introduction: three "We" bullets, a motivating wrapfigure (Figure 2, Moai duplication), and no
  sentence over 35 words in the main text.
- Method: lettered method diagram with a brighter cathedral, appendix zoom (Figure D.1, all 57
  blocks), and Algorithms 1 and 2 side by side at the top of page 5.
- Experiments: setup run-ins (Data, Models, Baselines, Evaluation, Hyperparameters, Compute),
  "Key result:" in every main table caption, Tables 3 and 4 side by side, value labels and light
  gridlines on plots, wrapfigures for quality-time (Figure 4) and latency (Figure 6).
- Reproducibility statement at the end of the main text.
- Appendix: contents A to S, calibration prompts (C), method zoom (D), core code listings (E),
  "expands Table N" openings, letter-numbered floats.

Compute claim: the 1.5 GPU-hour calibration figure stays out (unsupported per
`results/compute/claims_audit/`); the paper cites the 123.3 GH200-hour worker-log lower bound.

PDF SHA256: `d2150b1128be760a6d7e06e7595a3e2b3be66b106dc1e0e89097965542168bc3`.
ZIP SHA256: `ea8419c92c51102e08aef40f8ba02f710fa1dd05fcd8c81f75aee6e9d417f96c`.
Nothing is committed; HANDOVER.md Section 14 holds the self-score.
