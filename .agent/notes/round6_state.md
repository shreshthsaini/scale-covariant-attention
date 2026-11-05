# Round 6 (method owner, duplication fix): live state, rewritten at each checkpoint

## Goal
User directive 2026-11-05: remove small repeated subject copies at 4096^2 and keep SCA SOTA and novel.
Step 1 human-validated bar; Step 2 mechanism; Step 3 principled fix (>= 4 candidates, 46 then 95 then 195);
Step 4 operator definition + paragraph in .agent/notes/duplication_fix_round6.md, rerun task set prepared (gated, ask lead).
Report each step to the lead (SendMessage main, < 250 words); final report via SubagentHandback.

## Status (2026-11-05 14:00 CST)
- Step 1 DONE, reported (commit 9f464ba): human rates on 95 prompts AReA 4.2% [1.6,17.0], HRDiT 6.3%, full 4.2%
  (subject missing in 22/37), sca_v2 26.7% [24.6,36.5]. Bar = <= AReA's rate on the same prompts.
- Step 2 interim, reported: copies seeded at the first post-gate step (sigma .888) on ambiguous blobs, formed by .85
  (web traces of published SCA; results/round6_mechanism/*.jpg); windows over-read the prompt 2.0-2.3x median, 4.5x in
  heads with pi < .5 (results/round6_mechanism/text_overread.csv). Lead: prove candidate (c) ctw first.
- GPU: SCA21, SCA22 (4x10h) + lead's backfill SCA23 (gh-dev 4x2h), SCA24 x2 (gh 1x3h) all PENDING at 13:40.
- Queue order (code freeze a4f72f2 except smoke/gtg/gtx/bw on 09492d5): a6_a000_smoke -> a6_b200..b205_r6_ctw (46 p)
  -> a6_b3_r6_ctw_extra (10 traced prompts; releases taskq/hold_r6_ctw: (c) traces a6_c20x, probe a6_c6) ->
  a6_b99_eval_r6_ctw (self-released) -> a6_c00x/c01x traces sca_v2/area, a6_c5 probe sca_v2 -> gtg/gtx/bw a6_d2xx
  (hold_r6, released by smoke) + evals a6_z97 (hold_dup_eval). Superseded tasks in taskq/hold_r6/superseded_20261105.
- Lead wants on (c)'s 46-prompt result: human-checked dup count, paired deltas, over-read after the fix (probe c6).

## Next
1. Smoke verdict (r6/smoke/VERDICT, PSNR, timing). 2. When eval/r6_46_r6_ctw/DONE: prep human labels
   (scratchpad step3/prep_labels.py SP r6_ctw), labelers, adjudicate -> results/dup_audit_20261027/human_labels_r6.csv;
   python scripts/r6_decide.py r6_ctw; probe summaries (scripts/r6_mechanism.py probe r6/probe/<label>); report lead.
3. Then gtg/gtx/bw results; round 2 (native opening --native-open K, combos) as needed; promotion to 95/195.

## Decisions
- Human-corrected rate = (TP among flagged + N_unflagged * k/30) / 95; Wilson on k/30 carried through.
- Candidate references on 46 prompts: out/dupfix/sca_v2 and out/main/area (both cover the 46).
