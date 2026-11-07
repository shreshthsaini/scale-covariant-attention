# Progressive SCA (sca_v2_prog) against every 4096^2 baseline, 195 Aesthetic-4K prompts, seed 0

Item 1 of the submission list (lead relay, 2026-11-07). These are built on CPU from cached evaluations, all made
by the same evaluator, and no image was rescored. The per-image metrics come from eval/a4k195_main (AReA, DyPE, SEGA,
UltraImage, full attention), eval/a4k195_main_sca_v2 (sca_v2) and eval/prog195_sca_v2 (sca_v2_prog,
hrdit_dev_author). AReA's values are identical in the two evaluations that both scored it.
- sca_v2_prog_vs_<baseline>.csv: per-image paired bootstrap, 2000 prompt resamples, seed 0
  (scripts/paired_bootstrap.py), for IR, aesthetic, HPSv2, HPSv2.1, CLIP-IQA, ARNIQA, CLIP, PickScore, MUSIQ and
  MUSIQ crop. delta = sca_v2_prog minus the baseline.
- dup_rates_people.{csv,txt}: detector person/animal rates over the 95 eligible prompts with exact McNemar against
  sca_v2_prog. The blind human census of sca_v2_prog, AReA and HRDiT is in
  results/dup_audit_20261027/human_rates_prog95.txt.
- cfid/ (added when the CPU job finishes): corrected FID, KID, patch FID and patch KID with 1000-draw intervals
  (168 reference identities, repro/evaluate_corrected.py, reference sca_v2_prog). Recipe: cpu_setlevel.sh.
- IS is reported per method with its interval in each source evaluation's extra_summary.csv.
