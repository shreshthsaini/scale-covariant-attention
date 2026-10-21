# Latest verified paper delivery, 2026-10-20 23:00 CT (final pass: sample curation, Krea-195, HRGen-1K)

- [Paper PDF](../../paper/main.pdf), 49 pages: main text pages 1 to 10 (Section 6 ends at y = 694 pt on page 10),
  reproducibility statement and references from page 11, appendix after.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 286 source files.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_9un4nd1w/` (`primary/`, `extracted/`, `standalone/`).

The source ZIP builds on its own from a fresh unzip (49 pages). Both builds have zero overfull boxes and zero
undefined references; the PDF text has no em dash, no section symbol and no "??". PDF size 26,497,556 bytes.

What changed in this pass (details: HANDOVER.md Section 17, .agent/notes/figure_curation_20261020.md):
- Qualitative figures: every SCA image audited at full resolution. Curated swaps in Figure 1 (portrait, six-method
  strip), Figure 6 (all three rows), R.1 (panorama, still life) and R.7 (owl row); random galleries keep their draws
  and their captions now describe what is shown, including SCA's extra subjects (results/visual_audit_20261020).
- Numbers: Krea-195 and HRGen-1K (1000) folded into Table 2 (HRGen block; AReA's multi-stage rows dropped), Table 5,
  Tables K.1 and K.3, new Table H.1 (published HRGen-1K rows), abstract, Sections 4.4 and 6, Appendix H. Table 3 no
  longer repeats Table 1's rows. J.1/J.2 band-guard entries stay on the 191-prompt run (captions say so).
- Render review fixes: split citation that broke page-10 links, unreferenced Figure D.2 and Table J.7, Figure S.1
  above its heading, "Fig-ure" hyphenation, Table 2 caption wording.
- Known and left: page 27 holds Table J.2 alone (61 percent full); the code link is the public GitHub URL while the
  author block is anonymous (swap \codeurl at submission).

PDF SHA256: `3e121764d2288efa78c94a842fb9f8c9fd8b293756ae1e017ed53a5f87c6981b`.
ZIP SHA256: `a26310cf5f972e4be7d575245ca6b3c5193f5f345fff734b259ec4e797962388`.
