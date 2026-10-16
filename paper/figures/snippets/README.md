# Figure snippets

Each `.tex` file holds one complete `figure`/`figure*` environment. Use it with `\input{figures/snippets/<name>}` from `main.tex`. Image paths are relative to `paper/`. The snippets need `graphicx` and `xcolor` (both already loaded in `main.tex`), plus the `\sca`, `\Oloc`, `\Oglob` and `\pih` macros.

| Snippet | Label | What it shows | Where it goes |
|---|---|---|---|
| `hero.tex` | `fig:hero` | Six SCA results: 6144x2048, 4096x2048 and four at 4096^2. Three native 1:1 zoom crops. A strip comparing full attention, AReA and SCA on one prompt, as native crops. | Page 1, `figure*[t]` right after `\maketitle` |
| `method.tex` | `fig:method` | TikZ pipeline with three panels: (a) gate phase, then the per-head mix over the 28 steps; (b) the two coordinate systems; (c) the calibration loop at 1024^2. | Method section; replaces the old `figures/method.pdf` |
| `calib.tex` | `fig:calib` | (a-c) pi_h(sigma) heatmaps of the scale-4 table at three sigma values. (d) Mean pi against sigma with the IQR for 1024/s2, 1024/s4 and 1280/s2. (e) Excess error of transferred tables over each target's own table. | "What the heads want" section; can replace `pi_heatmap.pdf` |
| `qual_main.tex` | `fig:qual_main` | 6 prompts x {full attention, AReA, SCA} at 4096^2, each with a native 512 px crop. | Experiments; replaces the old `fig:qual` |
| `latency.tex` | `fig:latency` | Seconds per image. For now this is the 4096^2 bar chart; it becomes a resolution curve once the latency CSV exists. | Experiments or cost section |
| `ablation.tex` | `fig:ablation_grid` | 4 prompts x {no gate, pi=1, uniform pi, pi=0, mean pooling, SCA}, full images plus crops. | Appendix, next to the ablation table |
| `failures.tex` | `fig:failures` | Top row: duplication without the gate (window 64 and window 16, dev prompts). Then four errors SCA still makes: a staircase sword, a blocky patch, a foot in place of an arm, speckle at a border. | Appendix, limitations |
| `gallery_4096.tex`, `gallery_4096_p2.tex` | `fig:gallery_4096`, `fig:gallery_4096_p2` | 32 randomly sampled, uncurated prompts, AReA vs SCA at 4096^2, with crops. | Appendix gallery |
| `gallery_4096x2048.tex` | `fig:gallery_4096x2048` | 6 random prompts, AReA vs SCA at 4096x2048 (W x H). | Appendix gallery |
| `gallery_6144x2048.tex` | `fig:gallery_6144x2048` | 6 random prompts, AReA vs SCA at 6144x2048 (W x H). | Appendix gallery |

`main.tex` already uses the label `fig:failure` for the body figure. `fig:failures` is a different figure.

## Regenerate (compute node, sca env)

```
P=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
cd paper/figures/src
$P fig_qual.py all                      # hero, qual_main, gallery, gallery_wide, ablation, failures
$P fig_calib.py; $P fig_latency.py      # latency: add --csv /scratch/09032/saini_2/sca/results_latency_curve.csv
tectonic -o .. method_sca.tex           # method diagram
$P make_test_doc.py && cd ../snippets && tectonic -Z search-path=../.. _test.tex   # layout check
```

- **New baselines (DyPE, SEGA, UltraImage):** add a `ROOTS` entry in `imgsel.py`. Then run, for example, `fig_qual.py gallery --methods area dypes sca` (one prompt per row once there are 3 or more methods) or `fig_qual.py qual_main --methods full area dypes sca`.
- **New resolutions:** run `fig_qual.py gallery --res 6144x6144` or `--res 8192x8192` (the crop scales with resolution). For non-square sizes, use `fig_qual.py gallery_wide --wide HxW ...`.
- **HRGen or other prompt sets:** add the output directory to `ROOTS`. The short ids come from the file names.
- **Selection:** hero, main and failure prompts and their crop windows are hand-picked in the specs at the top of `fig_qual.py`. Gallery prompts are random and disjoint across galleries. Automatic crops take the window with the most edge energy, averaged over the methods shown.
- `contact_sheet.py` builds thumbnail sheets for choosing prompts.
- `_test.tex` and `_test.pdf` are scratch files. Do not commit `_test.pdf` (25 MB).
