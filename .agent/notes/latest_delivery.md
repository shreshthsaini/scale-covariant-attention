# Latest verified paper delivery, 2026-10-19 (four-defect fix pass)

- [Paper PDF](../../paper/main.pdf), 50 pages: main text pages 1 to 10, references from page 10, appendix from page 14.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 279 source files.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_i3592753/` (`primary/`, `extracted/`, `standalone/`).

The source ZIP builds on its own from a fresh unzip (50 pages, page-for-page identical text layout).
Both builds have zero overfull boxes and zero undefined references. PDF size 26,974,785 bytes
(under the 35 MB cap). No reported number changed.

What changed in this pass (the lead's four defects, `paper/main.tex` only):
- Algorithm 1: the B-line comment is now `# over rows`, so no listing line wraps; both algorithms
  have 17 lines. Algorithm 2 had no wraps (its two-line `flex_attention` call is a deliberate
  continuation). The one-line-shorter float would jump to the top of page 4, so its source now sits
  after Proposition 3; it stays at the top of page 5.
- `\codeurl` is `\href` with roman display text `github.com/shreshthsaini/scale-covariant-attention`
  in an `\mbox` (one line in the abstract and conclusion; swap the macro for an anonymous mirror).
  The reproducibility statement and Appendix E opening say "the code repository (link in the abstract)".
- Figure D.1 is `[t]` (was `[!htbp]`, which made a float-only page 19); Appendix D text now follows it
  on page 19, Tables D.1 and D.2 and Figure D.2 share page 20 (75% full), and Section E starts on
  page 21 under the end of the latency paragraph (old page 21 was 37% full).
- `\clearpage` replaces `\FloatBarrier` before Appendix S, so Figure S.1 sits at the top of page 50
  with the section below it (85% full) instead of a half-empty text page plus a figure-only last page.
- Pages 2 to 4, 6 to 9, 11 to 13 and 15 to 18 are line-for-line identical to the previous build, so the
  hand-tuned wrapfigures (pages 2, 7, 9) are unchanged; old pages 29 to 50 are new pages 28 to 49.
- Only page below two-thirds full in the appendix: page 33, a table-only float page (Tables K.4 and
  K.5, 57% full) between other K float pages; no running text exists that could fill it.

Previous pass (gold-standard fixes, build `delivery_e20al59t`): page-1 hero, three "We" bullets,
Moai wrapfigure, lettered method diagram, setup run-ins, "Key result:" captions, reproducibility
statement, appendix contents, method zoom, core-code listings; see HANDOVER Section 13 and 14.

Compute claim: the 1.5 GPU-hour calibration figure stays out (unsupported per
`results/compute/claims_audit/`); the paper cites the 123.3 GH200-hour worker-log lower bound.

PDF SHA256: `2c80e8f43127a0b5a55429f7f56e7c041b1b960ccdd9cd31773ef863ec68cab7`.
ZIP SHA256: `d341629b13126e1bda6d07634e48a9bbce11c1345c6ef41505f14958b6fd703c`.
Nothing is committed; HANDOVER.md Section 14 holds the self-score.
