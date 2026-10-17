# Figure snippets

Each `.tex` file holds one complete `figure`/`figure*` environment. Use it with `\input{figures/snippets/<name>}` from `main.tex`. Image paths are relative to `paper/`. The snippets need `graphicx` and `xcolor` (both already loaded in `main.tex`), plus the `\sca`, `\Oloc`, `\Oglob` and `\pih` macros.

| Snippet | Label | What it shows | Where it goes |
|---|---|---|---|
| `hero.tex` | `fig:hero` | Six SCA results: 6144x2048, 4096x2048 and four at 4096^2. Three native 1:1 zoom crops. A strip comparing full attention, AReA and SCA on one prompt, as native crops. | Page 1, `figure*[t]` right after `\maketitle` |
| `method.tex` | `fig:method` | TikZ pipeline with three panels: (a) gate phase, then the per-head mix over the 28 steps; (b) the two coordinate systems; (c) the calibration loop at 1024^2. | Method section; replaces the old `figures/method.pdf` |
| `calib.tex` | `fig:calib` | (a-c) pi_h(sigma) heatmaps of the scale-4 table at three sigma values. (d) Mean pi against sigma with the IQR for 1024/s2, 1024/s4 and 1280/s2. (e) Excess error of transferred tables over each target's own table. | "What the heads want" section; can replace `pi_heatmap.pdf` |
| `qual_main.tex` | `fig:qual_main` | 6 prompts x {full attention, AReA, SCA} at 4096^2, each with a native 512 px crop. | Experiments; replaces the old `fig:qual` |
| `latency.tex` | `fig:latency` | Log-log resolution curve: SCA from 2048^2 through 8192^2, AReA and full attention through 4096^2. Median of three timed prompts; whiskers are observed ranges. | Cost appendix |
| `ablation.tex` | `fig:ablation_grid` | 4 prompts x {no gate, pi=1, uniform pi, pi=0, mean pooling, SCA}, full images plus crops. | Appendix, next to the ablation table |
| `failures.tex` | `fig:failures` | Top row: duplication without the gate (window 64 and window 16, dev prompts). Then four errors SCA still makes: a staircase sword, a blocky patch, a foot in place of an arm, speckle at a border. | Appendix, limitations |
| `gallery_4096.tex`, `gallery_4096_p2.tex` | `fig:gallery_4096`, `fig:gallery_4096_p2` | 32 randomly sampled, uncurated prompts, AReA vs SCA at 4096^2, with crops. | Appendix gallery |
| `gallery_4096x2048.tex` | `fig:gallery_4096x2048` | 6 random prompts, AReA vs SCA at 4096x2048 (W x H). | Appendix gallery |
| `gallery_6144x2048.tex` | `fig:gallery_6144x2048` | 6 random prompts, AReA vs SCA at 6144x2048 (W x H). | Appendix gallery |
| `gallery_baselines.tex` | `fig:gallery_baselines` | 3 random prompts, AReA, DyPE, SEGA, UltraImage and SCA at 4096^2; full-image row followed by matching crops. UltraImage uses its own two-stage defaults. | Appendix gallery |
| `gallery_flux2.tex` | `fig:gallery_flux2` | 4 random paired FLUX.2-dev prompts, full attention vs SCA at 4096^2. | Appendix gallery |
| `gallery_6144.tex`, `gallery_8192.tex` | `fig:gallery_6144`, `fig:gallery_8192` | 4 prompts each from the completed AReA/SCA pairs at the frozen snapshot, explicitly labeled partial qualitative evidence. | Appendix gallery |
| `gallery_8192_complete.tex` | `fig:gallery_8192_complete` | 4 random prompts from all 12 completed 8K pairs, preserving visible grid/stripe failures. | Appendix gallery |
| `gallery_6144_complete.tex` (prepared mode; not rendered) | `fig:gallery_6144_complete` | 4 random prompts, only after all 100 archived E2 IDs have complete AReA/SCA pairs. Preserves the older 14-prompt pool separately. | Future appendix gallery |
| `gallery_krea_complete.tex` (prepared mode; not rendered) | `fig:gallery_krea_complete` | 2 random prompts across all 4 methods, only after all 100 archived E9 IDs have complete outputs. Two prompts leave room for readable full images, crops and protocol caveats on one page. | Future appendix gallery |

`main.tex` already uses the label `fig:failure` for the body figure. `fig:failures` is a different figure.

## Regenerate (compute node, sca env)

```
P=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
cd paper/figures/src
uv run --no-project --python "$P" python fig_qual.py all
uv run --no-project --python "$P" python fig_phase2.py all
uv run --no-project --python "$P" python fig_calib.py
uv run --no-project --python "$P" python fig_latency.py  # uses results/latency_curve.csv
tectonic -o .. method_sca.tex           # method diagram
uv run --no-project --python "$P" python make_test_doc.py
cd ../snippets
tectonic -Z search-path=../.. _test.tex  # layout check
```

- **New baselines (DyPE, SEGA, UltraImage):** add a `ROOTS` entry in `imgsel.py`. Then run, for example, `fig_qual.py gallery --methods area dypes sca` (one prompt per row once there are 3 or more methods) or `fig_qual.py qual_main --methods full area dypes sca`.
- **New resolutions:** run `fig_qual.py gallery --res 6144x6144` or `--res 8192x8192` (the crop scales with resolution). For non-square sizes, use `fig_qual.py gallery_wide --wide HxW ...`.
- **HRGen or other prompt sets:** add the output directory to `ROOTS`. The short ids come from the file names.
- **Selection:** hero, main and failure prompts and their crop windows are hand-picked in the specs at the top of `fig_qual.py`. Gallery prompts are random and disjoint across galleries. Automatic crops take the window with the most edge energy, averaged over the methods shown.
- `contact_sheet.py` builds thumbnail sheets for choosing prompts.
- `_test.tex` and `_test.pdf` are scratch files. Do not commit `_test.pdf` (25 MB).

## Phase-2 provenance

`src/fig_phase2.py` builds the phase-2 galleries without changing the existing
selection. On its first run it samples sorted eligible prompt IDs using
`random.Random(17)`, excluding the four development prompts. An image is eligible
only when all compared methods have a complete seed-zero sidecar and a PNG with
the expected dimensions. The original partial 6K/8K galleries use frozen incomplete
pools that may reflect completion order; their captions retain that limitation.

Each gallery directory contains `provenance.json` with the frozen eligible pool,
selected prompts, source SHA256 hashes, full generation configurations, snapshot
time, and shared crop coordinates. Later runs reuse the selection and check source
hashes. Crops maximize edge energy averaged over all methods shown. No score or
visual judgment enters prompt selection. JPEG paper assets are archived here;
the source PNG paths point to scratch and remain purge-eligible.

The first phase-2 render was checked as four separate pages with the paper's
5.5-inch text width. The frozen 6K and 8K eligible pools contain 14 and 11 paired
prompts respectively. The displayed FLUX.2 portrait distortion and residual 8K
grid/stripe artifacts are retained and identified in their captions.

## Prepared complete-campaign galleries

The new `6144_complete` and `krea_complete` modes are explicit commands, excluded
from `all`. They require every ID in the archived
`results/evidence/prompts/eval100.jsonl`, not merely 100 common filenames. Extra
prompts cannot replace a missing target. Each method must have a complete seed-zero
PNG/sidecar pair, matching image and metadata resolution, exact prompt text, 28
steps and guidance 4.5. Duplicate completed outputs for a target are rejected.
Readiness is checked before creating a gallery directory, and on later rebuilds.

From the repository root, these checks write no figures or selection manifests:

```bash
uv run --no-project --python "$P" python paper/figures/src/fig_phase2.py 6144_complete --check-ready
uv run --no-project --python "$P" python paper/figures/src/fig_phase2.py krea_complete --check-ready
```

Only after a check succeeds, rerun its command without `--check-ready` to render.
Neither prepared gallery has been rendered or added to the manuscript. The first
successful render samples the sorted complete pool with seed 17 and freezes its
IDs, prompt-manifest hash, PNG and sidecar hashes, configurations and shared crop
coordinates. Later renders reject changed sources or evaluation manifests.

Krea uses AReA and recalibrated SCA from `sca/out/krea/`, plus the inherited DyPE
and SEGA outputs from `constancy_rope/out_krea/main/`. It requires the archived
`results/evidence/krea_baseline_provenance/receipt.json`, which connects the older
baseline runs to the Krea transformer snapshot through launch scripts and loader
code. Their sidecars label the FLUX.1-dev base pipeline and omit that override.
The gallery records the receipt hash and this metadata limitation, retains each
selected runtime configuration, and reports any unresolved historical source
snapshots counted by the receipt. It does not present the varying historical
runtime revisions as one frozen implementation. The caption also distinguishes
the shared sampling protocol from separately tuned defaults and equal runtime.

`tests/test_gallery_readiness.py` exercises missing-target substitution, protocol
and metadata mismatches, duplicate outputs, deterministic selection, four-method
coverage and the required checkpoint receipt. It uses tiny synthetic images and
does not render an incomplete campaign.
