# SCA live state, 2026-11-07 22:20 CST (paper owner hand-back; HANDOVER.md Sections 31 to 33 are the log)

## Goal
Finish SCA for submission with progressive SCA (HRDiT's three stages, SCA attention at 2048^2 and 4096^2) as the
headline and single-pass SCA as the fast mode. User wording rules now in force: no "calibration" (the procedure is
"reach emulation", its output the "reach table", weights are "fitted"), no "curated"/"uncurated" or selection phrases.

## Status and next action
- Paper delivered: /scratch/09032/saini_2/sca/paper_builds/delivery_mqhyl3id, commit b131b66, 54 pp, main text
  pages 1-10 with two spare lines, 0 overfull, 0 undefined, fresh unzip passes (.agent/notes/latest_delivery.md).
- Next for the paper: when web/build/showcase_manifest.json lists FLUX.2-dev and Qwen-Image SCA images, run
  paper/figures/src/fig_showcase_models.py, set \showcasetrue in paper/main.tex, rebuild with
  paper/build_delivery.py, check the main text still ends on page 10, update latest_delivery.md, commit and push.
- Open by choice (render review SHOULD items): Figure T.1 size; free space on gallery pages 43, 47, 48, 51 to 53;
  the code release still names files calib (sca/calib.py, scripts/calibrate.py, scripts/calib_prompts.jsonl,
  calib/), which the paper no longer prints.

## In flight
- Showcase campaign (HANDOVER Section 32): fleets SCS1 and SCS2 on gh, outputs in /scratch/09032/saini_2/sca/out/
  showcase/; the manifest held 23 FLUX.1-dev images at 22:15 and no FLUX.2-dev or Qwen-Image image.
- Web owner: page v3 already uses method_v3 and the new terms (commit fc200a7).
- Nothing of the paper owner's is running.

## Map
| file | what |
|---|---|
| paper/main.tex, paper/appendix.tex | paper source; \showcasefalse switch near the top of main.tex |
| paper/figures/src/method_v3*.tex, fig_method_v3.py | Figure 2 (TikZ) and its data assets (figures/method_v3/) |
| paper/figures/src/fig_showcase_models.py | manifest -> figures/showcase_models/ + snippets/showcase_models.tex |
| paper/build_delivery.py | build, ZIP of the files the build read, fresh-unzip check |
| .agent/notes/method_figure_v3.md | candidates A and B and the critique against the gold-standard recipe |
| .agent/notes/latest_delivery.md | hashes and checks of the latest delivery |

## Decisions
- Terms are defined once in the introduction (\emph{reach emulation}, \emph{reach table}) and used everywhere,
  including algorithm names, section titles, labels and shipped file names.
- Figure captions give prompt set and seed only; random galleries say how prompts were drawn; no figure claims
  randomness it does not have.
- The showcase figure stays off until the FLUX.2-dev and Qwen-Image images exist; it opens Appendix S, with one
  sentence in Section 4.4, so the main text keeps its 10 pages.
