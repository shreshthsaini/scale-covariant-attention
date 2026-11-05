# Round 6 (method owner, duplication fix): live state, rewritten at each checkpoint

## Goal
User directive 2026-11-05: remove small repeated subject copies at 4096^2 and keep SCA SOTA and novel.
Step 1 human-validated bar; Step 2 mechanism; Step 3 principled fix (>= 4 candidates, 46 then 95 then 195);
Step 4 operator definition + paragraph in .agent/notes/duplication_fix_round6.md, rerun task set prepared (gated, ask lead).
Report each step to the lead (SendMessage main, < 250 words); final report via SubagentHandback.

## Status (2026-11-05 01:45 CST)
- Step 1 DONE, reported (commit 9f464ba): human-corrected rates on 95 prompts: AReA 4.2% [1.6,17.0], HRDiT 6.3%,
  full 4.2% (subject missing in 22/37 labeled), sca_v2 26.7% [24.6,36.5]. Bar = <= AReA 4.2% on the same prompts.
  Files results/dup_audit_20261027/human_labels_95*.{csv,jsonl,json,md,txt}, human_rates_95.*; scripts/human_rates.py.
- Code: sca/textgate.py (--text-mode ctw|gtg|gtx|bw, commit 09492d5, freeze code/09492d5); slurm/round6_tasks.py;
  scripts/r6_attn_probe.py (freeze code/ff59add).
- Queue (taskq/pending, a6_ sorts first): a6_a000_smoke (identity PSNR >= 24 dB releases taskq/hold_r6: 24 gen tasks
  r6_ctw, r6_gtg, r6_gtx, r6_bw x 46 prompts), a6_p00_probe_sca_v2 (6 copies), a6_t0{0,1}_trace_{sca_v2,area}
  (10 copies, selection /scratch/09032/saini_2/sca/r6/trace_selection.json). Evals a6_z97_eval_* in hold_dup_eval,
  self-released -> eval/r6_46_<label>.
- Fleets SCA21 1123398 (campaign) + SCA22 1123443 (mine) both PENDING at 01:45.
- Findings: copies spread over the image (not RoPE text offsets); sca_v2 copies median 135 px, 1/3 copy-like.

## Next
1. When GPU runs: check smoke (r6/smoke/VERDICT, timing), probe (r6/probe/*/probe.npz), traces (r6/trace).
2. Write decision script (human check of flagged images of each candidate, paired metrics vs sca_v2 and AReA, time).
3. Step 2 report (mechanism) after probe + traces (+ review item 2 when r2_ lands); Step 3 round 2 on the best.

## Decisions
- Human-corrected rate = (TP among flagged + N_unflagged * k/30) / 95; Wilson on k/30 carried through.
- Candidate references on 46 prompts: out/dupfix/sca_v2 and out/main/area (both cover the 46).
