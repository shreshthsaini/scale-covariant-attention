# Latest verified paper delivery, 2026-11-10 00:42 CST (v5: measurement-first rewrite, ICLR 2027 style)

- [Paper PDF](../../paper/main.pdf), 54 pages: main text pages 1 to 9 (Section 6 ends on page 9, about 40% of the
  page used), reproducibility statement and references from page 10, appendix after. SHA256
  e17bc42e8d26b2daeebfc0d42a486cb9d6065e08b94eaa93173ea20b5d4ea690.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 427 source files (main.tex, appendix.tex, iclr2027 style files,
  and only the snippets and figures the build reads), SHA256
  ef4f0f5305fc334c0b2ec157c92afe7926c0e45745b1bee13b769c24bf7901d3.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_gy4l47kq/`.

Both builds: zero overfull boxes, zero undefined references, and the fresh unzip builds on its own to the same 54 pages.
PDF text has no calib*, curat*, "drawn at random", "random draw", "random quer*", "fixed random", em dash, section
symbol or "??"; no file name in the ZIP contains calib or curat. Abstract: 150 words (counted from the PDF).

What changed (HANDOVER Section 41; details in .agent/notes/paper_v5_20261110.md): the story now runs observation ->
two native-size views with one number per head -> reach emulation -> use -> evidence. New Figure 3 (observation:
two heads' attention beside the 1,368-head by noise-level matrix of near share, with the head-average strip and the
97.5% / 2.5% split), new three-panel Figure 4 (method, pi* = clip(B/A, 0, 1) at its centre), cleaner Figure 2 (speed:
arrows only from AReA and HRDiT). Lean Table 1 (time, IR, HPS, Pick, M_c, FID_p; bold best per group) and ablation
(reach weights, then views). Switches \resprogabl and \resra wired to results/prog195_ablation/ and results/ra195/,
off; tested on with fake snippets (0 overfull, main text still on 9 pages).

Previous deliveries: `delivery_gvyduue4` (v4, HANDOVER Section 37, 55 pp, main text 1 to 10) and `delivery_yio2nt1w`.
