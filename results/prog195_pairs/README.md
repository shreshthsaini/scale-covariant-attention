# Progressive SCA (sca_v2_prog) against AReA and sca_v2, 195 Aesthetic-4K prompts, 4096^2, seed 0

These pairs are built on CPU from cached evaluations, and no image was rescored. The per-image metrics come from
eval/a4k195_main_sca_v2 (AReA, sca_v2) and eval/prog195_sca_v2 (sca_v2_prog, hrdit_dev_author). Both ran on
2026-10-30 with the same evaluator and caches, and AReA's per-image values are identical across the two AReA
evaluations. scripts/paired_bootstrap.py draws 2000 prompt resamples (seed 0) for each `<a>_vs_<b>.csv`.
dup_rates_people.csv gives the detector people/animal rates over the 95 eligible prompts, with the reference set to
AReA (exact McNemar). cfid_vs_*/ (added when ready) holds the corrected FID/KID/patch FID/patch KID with 1000-draw
intervals (repro/evaluate_corrected.py, reference clusters = images). Recipe: scratch
eval/prog195_pairs/cpu_setlevel.sh.
