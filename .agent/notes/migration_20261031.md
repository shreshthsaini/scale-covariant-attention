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
