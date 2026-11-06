# Round 6 (method owner, duplication fix): live state, rewritten at each checkpoint

## Goal
User directive 2026-11-05: remove small repeated subject copies at 4096^2 and keep SCA SOTA and novel.
Step 1 human-validated bar; Step 2 mechanism; Step 3 principled fix; Step 4 only if a variant passes.

## Status (2026-11-06 06:20 CST): round closed, reported to the lead
- Step 1 DONE (9f464ba): human-corrected rates on 95 prompts AReA 4.2%, HRDiT 6.3%, full 4.2%, sca_v2 26.7%.
- Step 2 DONE: copies seeded at sigma .888-.85 on open props of the coarse layout; context deficit (window sees the main
  subject at 0.4-0.5x full attention), not prompt strength. Reported.
- Step 3 DONE, no pass: 13 variants (results/dup_audit_20261027/round6_frontier_46.csv). Best neutral: hand-off bridge
  0.25 (3/26). Lowest duplication intact: lattice far field sigma > .5 (1/26, 9 losses). Step 4 not triggered.
- No round-6 GPU tasks pending or running; deferred: taskq/hold_r6_ctw ((c) traces). HANDOVER Section 27 written.

## Next (for whoever continues)
Head-selective or sigma-decaying lattice far field (keep 1/26 at lower cost), or a finer opening layout (AReA-style f=2
in the last opening steps) to remove the props that seed copies. Code: sca/textgate.py (ffl), slurm/round6_tasks.py.

## Decisions
- Human-corrected rate = (TP among flagged + N_unflagged * k/30) / 95; Wilson on k/30 carried through.
- Candidate references on 46 prompts: out/dupfix/sca_v2 and out/main/area.
