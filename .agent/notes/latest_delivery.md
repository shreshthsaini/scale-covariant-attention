# Latest verified paper delivery, 2026-10-20 (195-prompt switch and layout pass)

- [Paper PDF](../../paper/main.pdf), 48 pages: main text pages 1 to 10 (Section 6 ends at y = 694 pt on page 10),
  reproducibility statement and references from page 11, appendix from page 14.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 285 source files.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_m1gkdr8f/` (`primary/`, `extracted/`, `standalone/`).

The source ZIP builds on its own from a fresh unzip (48 pages). Both builds have zero overfull boxes and zero
undefined references. PDF size 26,441,395 bytes.

What changed in this pass:
- Evaluation protocol: all 195 Aesthetic-4K prompts (`\nmain`, `\npanel`, `\nseed` = 195; ablations stay at the
  fixed 46-prompt subset, HRGen-1K at 200, FLUX.1-Krea-dev at 100). Tables 1, 2, 3, 5, J.1 to J.3, K.2, K.3, K.5,
  K.6, L.1, L.2 and Figures 5, 7, K.1 regenerated from `results/a4k195_{main,panel,2048x4096,2048x6144,seed1,seed2}`
  through `paper/figures/src/data195.py`; table script `/scratch/09032/saini_2/sca/finish_agent/switch/apply_tables.py`
  (seeds: `apply_seeds.py`). The band-guard row of J.1 and columns of J.2 stay on the 191-prompt run (no 195 run), as
  their captions say.
- Headline numbers: still 11 of 15 metrics significantly better than AReA; ImageReward +0.31 [0.23, 0.38] (was
  [0.24, 0.38]); FID -7.7 [-13.1, -2.6] (was -7.6 [-12.8, -2.9]). Progressive SCA vs HRDiT: ImageReward -0.02
  [-0.09, 0.03] at 49.5 s against 110.4 s; its FID gap to single-pass SCA is no longer significant (patch FID still is).
  At 2048x6144 SCA wins nine metrics (was ten) and now loses patch FID. Both extra seeds improve all ten per-image
  metrics.
- Layout: Design choices and Inference are proper paragraphs; Section 4 order is Main results, Other sizes and
  latency, Ablations, Generality, so Tables 4 and 5 (page 10) follow their first citation (page 9); Figure 7 sits
  beside the HRDiT paragraph at the top of page 8 and Figure 8 beside Coverage; `\newpage` before the
  reproducibility statement keeps page 10 at natural spacing. Appendix J order is app_4k, app_wide, app_ps2,
  app_abl, band, band2 (also in gen_tables.py); J float pages are 71 to 97 percent full.

PDF SHA256: `8c8e1bb1c60146139f397f3c533dc4f9abe894e12413c396ad9f31351cb2ecbf`.
ZIP SHA256: `251c3d8dfdd8a417dd0e2ce7ede1f0e1782c54ea3fcabf3711bcdd28598ac7e1`.
Nothing is committed.
