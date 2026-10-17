# Closest-work primary-source audit

Checked 2026-10-17. Scope: AReA, HRDiT, ResDiT, UltraImage. No performance results were independently reproduced in this audit. Existing author order, full titles, years and arXiv identifiers for all four entries in `paper/references.bib` match the primary abstract pages. Added canonical arXiv URLs; retained preprint metadata without inventing venue acceptance.

## Mechanisms and claim corrections

- [AReA, Sections 3.2–3.3](https://arxiv.org/html/2610.17227v1): block-centre coarse K/V attention followed by shrinking windows; also content-driven splitting, halo and bridge mechanisms. Its shared schedule is a valid specific contrast. Do not generalize that contrast to every prior method. SCA explicitly reuses its coarse global geometry and global-opening motivation.
- [HRDiT, Section 3.2, Equations 7–8](https://arxiv.org/html/2608.07003v1): per-head scope allocation is prior art. It estimates candidate-scope quality effects using final-image-loss gradients and a softmax-aware Taylor approximation, then solves a cost-constrained integer program. The resulting assignment is prepared before inference. It also discusses stable head roles. The current compact description of a precomputed per-head scope is supported, but an assertion that all previous heads share one receptive field is false.
- [ResDiT, Sections 3.2–3.4 and 4.1](https://arxiv.org/html/2512.01426v1): replace the incorrect scalar fixed-weight fusion description. It applies patch-wise FFT masking to combine global low frequencies with local high frequencies, plus Gaussian spatial overlap weights. Its 35-step recipe uses global attention first, spectral fusion in the middle, and local attention last. This precedes both the global-opening idea and the general global/local spectral combination. The optional SCA band guard needs this citation as well as FreeScale.
- [UltraImage, Sections 5.1–5.2 and 6.1](https://arxiv.org/html/2512.04504v1): head-specific concentration and stable local/global roles are explicit prior art. Distinguish its entropy-derived logit sharpening from SCA's output mixture. It also corrects dominant positional frequencies. Its paper evaluates direct and guided extrapolation; our two-stage FLUX run covers a guided configuration, not all UltraImage variants.

## Manuscript directions sent to root

Root delegated `main.tex` back for these corrections after its initial build. The requested introduction, related-work, optional-band, and untested-baseline limitations were applied; the next delivery build must include them.

1. State near the introduction that head-specific adaptation already exists. Ask specifically whether a local/global output mixture fitted at native resolution transfers across scales.
2. Correct ResDiT's fusion description and separate UltraImage's head-wise concentration from purely positional methods.
3. Retain novelty boundaries around emulated-scale output matching, a closed-form per-head/per-noise mixture, empirical transfer, and the efficient combined operator. The least-squares minimizer, linear pair-count calculation, head specialization, global opening, and two-branch design alone do not establish new general principles.
4. Missing direct HRDiT and ResDiT comparisons remain an evaluation limitation. Literature differences alone cannot prove comparative efficacy.
5. The complete bibliography was not audited. In particular, the pre-existing SigMa entry still has a comment saying its OpenReview title/venue need verification. This audit does not certify that entry.

Primary metadata: [AReA](https://arxiv.org/abs/2610.17227), [HRDiT](https://arxiv.org/abs/2608.07003), [ResDiT](https://arxiv.org/abs/2512.01426), [UltraImage](https://arxiv.org/abs/2512.04504).

Upstream repositories cross-checked for provenance: [HRDiT](https://github.com/zylwithxy/HRDiT), [UltraImage](https://github.com/thu-ml/DiT-Extrapolation). No upstream files were modified.
