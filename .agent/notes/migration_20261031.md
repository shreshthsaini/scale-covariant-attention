# Migration off TACC Vista, 2026-10-31 (Opus migration agent, subagent of the SCA lead)

Goal: everything needed to continue SCA from another machine is on GitHub (code, notes, paper, web, results,
handover) or on private Hugging Face repos (heavy artifacts). Host: compute node i617-044 (job cntrl-cmd 1111046,
gg, ends 2026-11-02 04:12). No job submission. Uploads with nice -n 19, at most 4 workers.

## Log
- 2026-10-31 start. Repo clean except `.agent/notes/web_v2_20261030.md` and `results/hrgen1k_dev_sca_v2/` (SHA256SUMS
  verify OK). Git LFS 3.6.1 available. HF account shreshthsaini is a FREE account (isPro False): private storage
  cap is 100 GB for the whole account (huggingface.co/docs/hub/storage-limits, checked 2026-10-31). Current private
  usage about 0 GB (only two empty private datasets). `~/.local/bin/hf` is broken here (python3.9 site-packages on
  scratch lost `requests`); use `/work/09032/saini_2/vista/library/uv-envs/sca/bin/hf` (huggingface_hub 2.2.0).
- Scratch inventory (MiB): out 289108, eval 2048 (88,755 npz features, 1148 csv/json/jsonl = 305 MB), calib 269,
  code 395, third_party 5 (AReA only), web_assets 1068, taskq 9, logs 30, wandb 5, prompts 1.
- Spool: 1 pending task (a000_z988_eval_hrgen1k_krea_sca_v2.sh), 0 running, 2463 done. No SCA job in squeue.
- Concurrent agent: another agent is swapping the HRGen-1K sca_v2 numbers into paper/ and web/ right now (modified
  paper/*.tex snippets, main.pdf, zip, web/*; check_page.py running). Commit only explicit migration paths
  (`git add <paths> && git commit -- <paths>`), never `-a`/`-A`.
- Commit 9b56c20 (local): .gitattributes (calib npz in LFS, 91 objects), calib/ (269 MB), third_party/ (AReA,
  UltraImage, DyPE, SEGA copies, no local patches; README with all six pins), repro/taskq_archive/ (2463 done scripts
  57 KB xz, spool state 142 KB xz), results/eval_sidecars/ (1507 files, 308 MB, SHA256SUMS), git lfs hooks installed.
- `git push origin main` was DENIED by the Claude Code auto-mode permission classifier ("Out-of-Place Publication").
  Not retried. Local commits wait for a push by the caller or the user.
- HF: shreshthsaini/sca-calibration (model, private) uploaded and verified (112 files, 281.7 MB, 0 mismatches,
  commit f41cf5f). `hf upload` (hub 2.2.0) is the resumable large-folder uploader; no upload-large-folder command.
- Not in the caller's inventory: sca/cache/ultraimage_blockavg (270 GB, UltraImage attention cache, regenerable),
  metric_restore_20261017 (11.5 GB), repro_env_cache_20261017 (7.1 GB), paper_builds (4.8 GB), flux2_embeds (1 GB):
  all regenerable, not migrated. Krea DyPE/SEGA baseline images live in constancy_rope scratch out_krea/main.
- HF private budget plan (free cap 100 GB): phase 1 = headline 4096^2 main-table sets (34.2 GB) + sca_v2 reruns
  without HRGen (21.3 GB) + latency (2.8 GB) + small repos (about 3.6 GB), then dupfix_hrgen_krea (23.8 GB, input of
  the one pending eval) if account usage stays under about 90 GB. Everything else is phase 2 (needs HF PRO).
- HF uploads verified: sca-eval-outputs (122 files, 1.88 GB, commit via hf upload, 0 mismatches), sca-frozen-code
  (29 files, 1.51 GB, commit db88753, 0 mismatches). Image archives 1a-1c built (52 tars, 58.2 GB) and uploading;
  1d (dupfix_hrgen_krea, 3 parts) building.
- Written: MIGRATION.md, README (read-first, layout, heavy data, restart recipe), HANDOVER Section 21,
  repro/migration/{inventory_out.py, plan_archives.py, build_archives.sh, upload_repo.sh, verify_hf.py,
  write_manifest.py, relocate_paths.sh}. Staging on scratch: /scratch/09032/saini_2/sca/hf_staging (plan.tsv, lists/,
  sums/, logs/, one folder per HF repo).
- HF final (all private, every file re-verified after the manifest upload, 0 mismatches): sca-calibration 112 files
  0.28 GB; sca-eval-outputs 122 files 1.88 GB; sca-frozen-code 29 files 1.51 GB; sca-generated-images 57 files
  82.02 GB (55 tars: 1a headline 14, 1b rerun 6, 1c latency 32, 1d dupfix_hrgen_krea 3 parts). Account private
  total about 85.7 GB of the 100 GB cap. results/migration_manifest.csv: 420 rows, 297 uploaded_verified, 123
  planned_not_uploaded (phase 2, 221.15 GB of source).
- The paper owner's commit 61dbfd3 (HRGen-1K dev row) was pushed by that agent and carried 9b56c20 to origin.
  The pending eval a000_z988 also needs out/hrgen_krea/sca_krea (phase 2); MIGRATION.md Section 4 gives both routes.
- GitHub check: fresh clone of origin (61dbfd3) plus `git lfs pull` fetched all 91 LFS objects; k1_1024.npz SHA256
  7c54b727... matches; eval_sidecars, third_party and taskq_archive present. Local commits after 61dbfd3 (MIGRATION.md,
  manifest, README/HANDOVER pointers, scripts, this log) are NOT on GitHub: the migration agent's push is blocked by the
  permission classifier. Push them with `git push origin main` from this directory.
