# Method figure redesign, 2026-10-17

The method figure now separates native calibration, target-resolution geometry,
and the opening schedule. Source is `paper/figures/src/fig_method.py`; deliverables
are `paper/figures/method.pdf`, `method.svg`, and `method_provenance.json`.
`paper/figures/snippets/method.tex` includes the new PDF at full text width and
contains the proposed caption. The previous TikZ sources and `method_sca.pdf`
remain available in the repository. No manuscript source was edited by this agent.

## Primary-source visual references

- [AReA Figure 4](https://arxiv.org/html/2610.17227v1#S2.F4): spatial key sampling
  is shown before the attention shape. This informed the explicit retained-key
  grids, rather than another abstract branch label.
- [ResDiT Figure 3](https://arxiv.org/html/2512.01426v1): distinguishes global and
  local geometry before fusion. SCA's figure uses its own native output fit and
  scalar head mixture, without importing ResDiT's spectral or Gaussian modules.
- [UltraImage mechanism figures](https://arxiv.org/html/2512.04504v1): frequency
  and attention-concentration explanations connect operations to measured
  evidence. SCA's only heatmap therefore uses actual archived fitted coefficients.
- [NVIDIA/MIT Sana Figure 5](https://arxiv.org/html/2410.10629v1#S2.F5) and
  [NVIDIA research page](https://research.nvidia.com/labs/eai/publication/sana/):
  separates overview from operator detail and labels tensor dimensions. SCA uses
  this level of specificity while depicting its own operators and protocol.

These are presentation references, not reproduced artwork or evidence that SCA
shares their training regimes. The four reference images were inspected directly.

## Scientific content and scope

- Native calibration uses the same full-attention trajectory. At emulated scale 4,
  a 64x64 native image-token grid supplies 16x16 global keys and 16x16 local windows.
  The diagram states the clipped B/A fit and its A=0 convention.
- The displayed 24x14 coefficient slice is layer index 28, the middle of 57 layers,
  from `paper/figures/src/data/k1_1024_pi.npz`, key `s4_pi_head`. Heads and bins stay
  in their recorded order. There is no synthetic heatmap, sorting or smoothing.
  Every table entry is independently rederived from archived sufficient statistics
  in the generator, including clipping and empty-bin filling.
- Deployment shows the 4096-square example, 256x256 image tokens, 64x64 global keys,
  and 64x64 local windows around 8x8 query tiles. Relative window/query extents are
  exact; the drawn grid cells aggregate tokens and are explicitly schematic.
- Global attention has (N+L) queries and 64^2+L keys; each local query tile has 8^2
  queries and 64^2+L keys. The matrices show support/shapes, not observed attention
  weights or a claim that the implementation materializes those matrices.
- Text keys occur in both branches. The output mixture applies to image queries;
  text queries use the global branch. At sigma>0.89 the local branch is skipped.
  The 28 segments show steps, not a uniform sigma axis.
- The coordinate labels describe the algorithm. The caption explicitly avoids an
  exact-bf16-equivalence claim. No performance metric or new experimental claim
  is added by the diagram.

## Caption and integration

Suggested lead: **Calibrate the mixture at native resolution, then transfer it
to a larger attention geometry.** The full caption in the owned snippet explains
panels a/b/c, source of the real coefficient inset, text treatment, and the
coordinate caveat. The snippet previously selected `method_sca.pdf`, overriding
the fallback in main.tex; it now selects the redesigned `method.pdf`.

Native export is 5.5x5in. Include at `width=\linewidth`, with no 0.86 shrink.
The main operator owns final page placement, manuscript fallback and delivery.
The figure and caption together are intended to occupy most of a page.

## Validation and files

Render command (existing managed environment, compute node):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/csd-figs/bin/python python paper/figures/src/fig_method.py
```

The PDF has no raster images and embeds TrueType fonts. SVG keeps text elements
and editable paths. All coefficient fits pass 1e-12 absolute/relative tolerance.
PDF text bounds are checked for clipping; a first-render clipped local caption
was shortened before delivery. Printed-width previews at 72dpi and 144dpi are in
`/scratch/09032/saini_2/sca/method_redesign_20261017/`; they are QA intermediates,
not paper assets. The source/data/output hashes are in `method_provenance.json`.
