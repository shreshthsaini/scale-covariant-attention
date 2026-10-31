# Ideogram4 RoPE Length Extrapolation

RoPE-based length-extrapolation (NTK, YaRN, SEGA) for the Ideogram4
transformer, to generate at resolutions beyond the native range.

## Training range

Although the official Ideogram4 training resolution is not published, we
observe that the model generates very well at **4096×4096** without any RoPE
scaling, quality is sharp, coherent, and stable at that size. We therefore
treat 4096 px as the effective training extent and set `--train-grid` to
**256** latent patches by default (256 × 16 = 4096 px). RoPE extrapolation
methods only engage above this grid; at 4096² and below the RoPE path is
bit-identical to the unmodified model.

## Why Ideogram4's RoPE is different

Flux and Qwen-Image use **per-axis block RoPE**: each axis (t, h, w) owns a
contiguous slice of the head dim (Flux: 16/56/56 of 128) carrying its own
*complete* geometric frequency ladder `theta^(-2j/d_axis)`. NTK/YaRN/SEGA were
formulated for exactly this: rescale each spatial axis's private ladder.

Ideogram4 instead uses **Qwen3-VL-style interleaved M-RoPE**:

- One shared ladder of `head_dim/2 = 128` frequencies, `base = 5e6`.
- With `mrope_section = (24, 20, 20)`, the h axis owns global frequency indices
  `{1, 4, ..., 58}` (20 freqs), w owns `{2, 5, ..., 59}` (20 freqs), and t owns
  the remaining 88. So h/w each get a **subsampled comb** of the shared ladder
  (wavelengths ~7 to ~6800 patches), not a full ladder.
- Text tokens carry positions `(i, i, i)` — plain 1D RoPE over all 128 freqs.
- Image tokens sit at `(65536, 65536+h, 65536+w)`: a large constant offset
  (`IMAGE_POSITION_OFFSET`) separates text from image positions, so the trained
  text<->image relative phase on every frequency contains a baked-in `65536*f`
  term.

Consequences for extrapolation:

1. Rescaling the shared ladder touches **all axes and text tokens at once**.
2. Any change to a frequency `f -> f'` rotates the trained text<->image offset
   phase by `65536*(f - f')` — essentially a random large rotation per
   frequency — unless the offset term is explicitly anchored.
3. YaRN's index-inversion correction formula assumes a full ladder per axis;
   on a 20-entry comb the split between "interpolate" and "extrapolate"
   frequencies must be computed from per-frequency rotation counts instead.

## Implementation:

Each method (NTK, YaRN, SEGA) is implemented in `Ideogram4MRoPE`
(`transformer_ideogram4.py`), selected with `--rope-scaling`:

1. **Per-axis comb scaling**: only the h/w frequency combs are modified, each
   with its own factor `s_h = grid_h / train_grid`, `s_w = grid_w / train_grid`.
   The t axis (which also carries the text<->image separation) and all 88
   t-frequencies stay untouched.
2. **NTK-by-parts on the comb**: the comb is treated as its own ladder of
   effective dim `2K` (K = 20): entry k is scaled by `s^(-k/(K-1))` — highest
   comb frequency (local texture) preserved, lowest fully interpolated.
3. **Rotation-count YaRN ramps**: `rot_j = train_grid * f_j / 2pi`; `rot > 16`
   keeps the base frequency, `rot < 0.75` gets full PI (`f/s`), blended
   PI<->NTK<->base in between (same beta/gamma thresholds as the references).
   (The full-ladder index-inversion formula does not apply to a 20-entry comb.)
4. **Offset anchoring**: image-token phase is computed as
   `65536 * f_original + (pos - 65536) * f_scaled`. Text tokens (pos < 65536)
   have zero delta, so text RoPE and trained text<->image offset phases are
   preserved exactly; image<->image relative phases use the scaled combs.

Scaling only engages when the grid exceeds `--train-grid` (default 256 patches
= 4096 px); below that the RoPE path is bit-identical to the original model.

## The horizontal-band bug (fixed)

Every 4096x4096 run (baseline included) showed a garbled horizontal band
starting at pixel y=3840 and ~128 px tall. Root cause: the conditional pass
packs 2048 text + 65536 image = 67584 tokens, and the dense boolean segment
mask then holds 67584^2 = 4.57e9 > 2^32 elements. The attention backend
indexes the mask with 32-bit arithmetic, so query rows >= 2^32/67584 = 63550
(image row 240 -> y=3840) read wrapped mask rows: the first ~2048 corrupted
queries land on text/pad mask rows (attend only to padding -> the ~8-latent-row
garbled band), later ones land on almost-correct image mask rows (coherent but
misaligned content below the band).

Fix (both parts exact, no behavior change in-distribution):
1. The pipeline trims the shared left-padding prefix of the packed layout
   (padding is segment-isolated, so this is lossless).
2. The transformer passes `attention_mask=None` whenever all segment ids are
   uniform (always true for single-prompt after trimming), avoiding the giant
   mask entirely and enabling the unmasked fast attention path.

## Files

- `transformer_ideogram4.py` — model + `Ideogram4MRoPE`; configure via
  `model.set_rope_scaling(method=..., train_grid=...)`.
- `pipeline_ideogram4.py` — pipeline with the padding-trim / maskless fix.
- `run_ideogram4.py` — CLI entry point; adds `--rope-scaling {none,ntk,yarn,sega}`
  and `--train-grid N`. Applies the setting to both the conditional and
  unconditional transformers.

## Usage

Requires the Ideogram4 diffusers checkpoint in the local Hugging Face cache
(`ideogram-ai/ideogram-4-nf4-diffusers`).

```bash
# baseline at native resolution (no scaling needed)
python run_ideogram4.py --height 4096 --width 4096

# SEGA extrapolation at 8192^2 (default train_grid=256)
python run_ideogram4.py --height 8192 --width 8192 --rope-scaling sega --multi-gpu

# YaRN at 5120^2 with a different assumed training extent
python run_ideogram4.py --height 5120 --width 5120 --rope-scaling yarn --train-grid 320
```

Outputs are saved under `outputs/ideogram4/` with filenames tagged
`..._{method}_tg{train_grid}.png` (`_base` for the unscaled run). Two runs at
the same size/method but different `--train-grid` therefore never collide on
the same filename.
