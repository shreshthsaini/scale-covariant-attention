# Round 6 (method owner, duplication fix): live state, rewritten at each checkpoint

## Goal
User directive 2026-11-05: remove small repeated subject copies at 4096^2 and keep SCA SOTA and novel.
Step 1 human-validated bar; Step 2 mechanism; Step 3 principled fix (>= 4 candidates, 46 then 95 then 195);
Step 4 operator definition + paragraph in .agent/notes/duplication_fix_round6.md, rerun task set prepared (gated, ask lead).
Report each step to the lead (SendMessage main, < 250 words); final report via SubagentHandback.

## Status (2026-11-05 00:50 CST)
- Controller node i615-133 (job 1123393 ends 11-06 23:31). Fleets: SCA21 1123398 (campaign, 4x10h) pending;
  SCA22 1123443 (mine, 4x10h) pending. Lead told about SCA22.
- Step 1: sample = flagged + 30 random unflagged (seed 20261105) of area/hrdit/full/sca_v2 on 95 people/animal prompts
  (166 images, key SP/step1/sample_key.json, tiles SP/step1/tiles, protocol SP/step1/PROTOCOL.md). 42 batches of 4;
  labelers batch 01-14 launched 00:45; 15-42 still to launch. Labels -> SP/step1/labels/batch_XX.jsonl.
  SP = /scratch/09032/saini_2/tmp/tmp/claude-883311/-work-09032-saini-2-vista-Projects-Exploration/4c03b520-8e32-49ed-8c50-e539347fc179/scratchpad
- Finding: extra boxes spread uniformly (published SCA top-left 1024 px cell 5.5% vs 6.25%), so out-of-range
  text RoPE in windows is NOT the cause. Copies small: median side 4.7% (SCA), 3.1% (sca_v2) of the image.

## Next
1. Launch labelers 15-42; then adjudicate detector/labeler disagreements; write results/dup_audit_20261027/human_labels_95.csv,
   rates with Wilson intervals; commit; report Step 1.
2. GPU (a6_ prefix): traces for 10 flagged sca_v2 prompts (custom selection json, box on a copy), candidates on 46 prompts.

## Decisions
- Human-corrected rate = (TP among flagged + N_unflagged * k/30) / 95; interval = Wilson on k/30 propagated.
