# SCA live state, 2026-10-29 21:30 CDT (paper rewrite around sca_v2; Opus paper owner, subagent of the SCA lead)

## Goal
Rewrite paper/ around sca_v2 (opening + shrinking windows 128/112/96/80 + matched pi + 1,024-token bridge in the LOCAL
softmax, code sca/attn.py sca_dedup_forward); swap every table to sca_v2 numbers as groups land; deliver build + notes.

## Status and next action
- 22:30 main text rewritten in main.tex (abstract to reproducibility; old text in scratchpad main_before_rewrite.tex):
  Sections 2.3 schedule (eq:sched), 2.4 root cause (sec:rootcause), 4.2 late duplication (sec:dup), new appendix
  app:dupfix, Argument moved to App. A, hparams rows, Prop 1/2 proofs, failure paragraph, band-guard link, bib entry.
  Status macros (\pend, \pendImg*, \hlDup*, \hlWorked*) are \providecommand'ed after headline.tex in main.tex.
- Quick build (scratchpad/quick_build.sh): compiles, 0 overfull, main text ends p. 11 (needs ~1 page cut after
  agents deliver), undefined tab:dupfix until agent B writes snippets/dupfix_variants.tex.
- Agents running (opus): A1 main tables+headline macros (gen_main_tables.py), A2 appendix tables + Fig 6/7 + swap_all.sh,
  B dup figure (figures/dup.pdf) + dupfix_variants + dup_examples_rule snippets, C method.pdf, method_zoom.pdf,
  alg drafts (figures/src/alg_drafts), cost_v2.txt. Waiter: scratchpad/wait_groups.sh (seen_groups.txt), bg task.
- NEXT: integrate agent outputs (algorithms from C, captions from B/C, A2's \input replacements), page-fit to 10,
  render review 1, commit+push "structure"; then per-group swaps (swap_all.sh) + prose claims check.
## Map
| note | read it when |
|---|---|
| .agent/notes/duplication_fix_20261027.md (end) | exact sca_v2 definition, 46-prompt evidence |
| .agent/notes/state_20261020_final_pass.md | previous STATE (Oct 20 final pass) |
| .agent/notes/latest_delivery.md | last delivery hashes |
| results/dup_audit_20261027/rates_*_people.txt | duplication rates (46 and 195 prompts) |

## Decisions
- Bridge lives in the local softmax (code), not the global branch as the lead's brief says; paper follows the code.
- Window schedule is fixed by hand (not calibrated); only the weight is read from the table. Paper says so.
- Old numbers stay visible behind \pend{} until each group lands (generator emits it); no \todo in the build.
- main.tex has one writer (me); subagents edit only paper/figures/src, snippets and figure PDFs.

## Open questions
- None yet.
