Set-level intervals of the progressive-schedule ablation (2026-11-10).
Each variant against sca_v2_prog: repro/evaluate_corrected.py --reference-method sca_v2_prog --reference-clusters images
--res 4096x4096 --metrics fid,kid,patch_fid,patch_kid --device cpu --bootstrap 1000 --threads 8, one process per variant
on a root holding that variant and sca_v2_prog, with the Inception features of ../corrected_fidp reused.
Point estimates equal ../corrected_fidp/paired_deltas.csv. The seconds rows are campaign times, not the clean batch-1
times of ../latency.csv. Run folders: /scratch/09032/saini_2/sca/eval/prog195_ablation/cfid_ci/<variant>/out.
