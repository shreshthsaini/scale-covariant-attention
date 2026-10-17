# SCA live state, 2026-10-17 01:50 CDT

## Goal
Finish Phase 2 experiments, verified evaluation, figures, paper and reproducibility package.
User explicitly transferred full project ownership to Codex; continue autonomously.

## Status and next action
- Main operator: root. Parallel agents own main.tex (paper), figures (figures), results and audit scripts (evidence).
- Last inherited commit cda71e3, October 16 21:59 CDT. No remote configured.
- Complete: Phase 1; Phase 2 E1, E5, E6, E7, E8. E4 HRGen and budget controls now finished on scratch.
- In flight: E2 6K/8K on six MF9 guest workers; E9 Krea and E3 latency queued.
- 01:44 spool snapshot: 90 pending, 6 running, 855 done, 0 failed; no STOP; no held tasks.
- MF9 1073907 nodes c609-092,c609-151,c610-011,c610-052,c638-141,c638-142, ends about 05:29 CDT.
- SCA6 1075813 (4 nodes/24h), SCA7 1075814 (4 nodes/12h) pending.
- Fresh telemetry and task logs confirm all six workers advancing; do not restart or duplicate them.
- Paper PDF: 27 pages, October 16 11:29, completed-result placeholders remain; ZIP is older.
- NEXT: archive E4/budget and validate paired comparisons; integrate completed results and galleries;
  keep campaign moving, recover only verified orphan tasks, build and inspect paper and standalone ZIP.
- No jobs submitted or cancelled during initial takeover. Current session on compute i618-051.
- srun directly from compute rejected by TACC wrapper; use single guarded login ssh for Slurm steps.

## Map
| Note | Read it when |
| --- | --- |
| HANDOVER.md | Method, historical results, paths and Phase 2 plan |
| .agent/notes/state_20261016_checkpoint.md | Exact inherited chronology and prior results |
| .agent/notes/phase2_experiments.md | E1-E7/E9 task design and protocol |
| .agent/notes/flux2_port.md | FLUX.2 implementation, scheduler and memory decisions |
| .agent/notes/area_code.md | AReA release implementation details |
| paper/figures/snippets/README.md | Figure generation and snippets |

## Decisions
- Preserve frozen campaign code and shared FLUX.1-dev protocol for comparability.
- UltraImage own two-stage defaults and FLUX.2 own scheduler/guidance must be explicit.
- UltraImage wins some preference metrics; SCA wins detail metrics and speed. No universal superiority claim.
- Four-prompt calibration shows no detected loss on 46 prompts; absence of significance is not equivalence.
- Preserve negative results and protocol limitations; paired CIs do not establish novelty or universal efficacy.
- No source checkout changes in another project. No scratch deletion. Archive compact paper crux on work.
- Existing guest authorization applies; inspect resource ownership and foreign processes before adding workers.

## Open questions
- Need final E2/E3/E9 outputs, coverage checks and paper integration before claiming completion.
- Need final clean build, print-size review, source ZIP rebuild and independent ZIP build.
- No Git remote; keep local commits and defer repository creation/publication to user instruction.
