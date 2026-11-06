# Round 6 (method owner, duplication fix): live state, rewritten at each checkpoint

## Goal
User directive 2026-11-05: remove small repeated subject copies at 4096^2 and keep SCA SOTA and novel.
Step 1 human-validated bar; Step 2 mechanism; Step 3 principled fix (>= 4 candidates, 46 then 95 then 195);
Step 4 operator definition + paragraph in .agent/notes/duplication_fix_round6.md, rerun task set prepared (gated, ask lead).
Report each step to the lead (SendMessage main, < 250 words); final report via SubagentHandback.

## Status (2026-11-05 19:00 CST)
- Step 1 DONE (9f464ba). Step 2: copies seed at sigma .888-.85 on ambiguous blobs (web traces); probe at 6 copies
  (results/round6_mechanism/probe_*.csv): window text over-read ~2x vs full attention, AND window sees the main subject
  with 0.4-0.5x full attention's mass; global branch's subject-word mass at copy sites as high as at the main subject.
- Candidate (c) r6_ctw FAILED (reported 18:45): dup 4/26 = sca_v2 4/26; human 14% vs 17% vs AReA 2.2%; 8 significant
  losses (IR -0.057, PickScore -0.098, ...); right-edge garbage band in ~1/3 of images (text reduction at mid noise);
  over-read after fix ~1.0 but copies stay -> context, not prompt strength, drives copies.
- GPU: gh-dev SCA23 nodes c642-{052,061,081,082} running; SCA21/22/24 pending. Smoke2 PASS (split identity 34/32 dB).
- Queue: round-1 a6_d2xx gtg/gtx/bw (running) -> a6_d3xx r6_ffc (far-field calibrated bridge, freeze 4ef8ab9) ->
  a6_e12xx/e8xx native opening K=12/8 (freeze 049d062); evals z97 (gtg/gtx/bw), z99 (ffc), z98 (nop) self-release.
  (c) traces deferred in taskq/hold_r6_ctw. Debug outputs r6/debug/{split0,ctw_mid}.

## Next
1. As evals land: prep labels (scratchpad step3/prep_labels.py SP <label>), 2 labelers per candidate, adjudicate,
   append results/dup_audit_20261027/human_labels_r6.csv; scripts/r6_decide.py <label>; check right-edge band
   (scratchpad step3/strip_scan.py <dir>).
2. Round 2 on the best (ffc variants: all steps vs mid; bridge size; combine with nop); promote winner to 95 then 195.
3. Step 2 report (mechanism) to lead with traces + probe once sca_v2/AReA traces land (r6/trace).

## Decisions
- Human-corrected rate = (TP among flagged + N_unflagged * k/30) / 95; Wilson on k/30 carried through.
- Candidate references on 46 prompts: out/dupfix/sca_v2 and out/main/area (both cover the 46).
