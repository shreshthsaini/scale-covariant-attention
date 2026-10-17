# Frozen baseline generation code

These 14 files are unchanged copies from the account-authored `constancy_rope`
commit `c05ebdcccddafd6b8fe4ec2e14247d72775df9a1`. `SOURCE.json` records every Git
blob and SHA256 hash. Its composite generation fingerprint matches all archived
DyPE and official-SEGA sidecars exactly. The calibration and probe scripts are
included because the historical runtime includes their bytes in that fingerprint.

`scripts/sample.py` with `--method dype` or `--method sega_official` is the runner
used for the manuscript's shared FLUX.1-dev comparison. Runtime dependencies are
contained in this directory; model weights and Python packages are separate.
The original source has no top-level license file. This snapshot does not assign
it a new license. `crope/THIRD_PARTY.md` preserves source attribution and the
distinction between the numerical ports and upstream paper protocols.
Existing upstream license metadata is in `../baseline_metadata/`.

The preserved tests cover RoPE formulas, direct upstream parity, seeds, resume
integrity, and metadata. Upstream parity tests require separately restored source
checkouts via `CROPE_THIRD_PARTY`; they skip explicitly if those sources are absent.
These CPU checks do not regenerate the benchmark images.

Use a fresh output directory. Relocation changes repository-level provenance even
when the account-authored runtime fingerprint remains identical. Scratch defaults
in the original files remain unchanged; set the cache and tracking environment
variables described in `../README.md` before running on another system.
