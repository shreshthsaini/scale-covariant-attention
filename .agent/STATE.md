# SCA live state, 2026-11-08 03:10 CST (paper owner; HANDOVER.md Sections 31 to 37 are the log)

## Goal
Finish SCA for submission with progressive SCA (HRDiT's three stages, SCA attention at 2048^2 and 4096^2) as the
headline and single-pass SCA as the fast mode. Wording rules in force: no "calibration" (the procedure is "reach
emulation", its output the "reach table", weights are "fitted"), no "curated"/"uncurated" or selection phrases, no em
dashes or section symbols. Main text at most 10 pages.

## Status and next action
- Paper v4 delivered: /scratch/09032/saini_2/sca/paper_builds/delivery_gvyduue4 (55 pp, main text 1-10, 0 overfull,
  fresh unzip passes; HANDOVER Section 37): speed figure (Figure 2) under the hero on page 1; attention-focus figure (Figure 3,
  real CPU attention capture of FLUX.1-dev at 1024^2); Sections 1-2 ordered near view, far view, reach, reach
  emulation; heads_view moved to Appendix B; newer-model switches wired and off.
- Next for the paper: when FLUX.2-dev / Qwen-Image showcase images or Ideogram 4 results land, run their snippet
  scripts (contract in the main.tex preamble comment), set the switch (\resfluxtwotrue, \resqwentrue,
  \resideogramtrue), rebuild, and check the main text still ends on page 10 (about 0.3 line spare with all three on
  and one-line result sentences); then paper/build_delivery.py, latest_delivery.md, commit, push.

## In flight
- Showcase campaign (FLUX.2-dev, Qwen-Image) and Ideogram 4 at 8192^2 (HANDOVER 32, 36): all wait on SCS1/SCS2/SCSd.
- Nothing of the paper owner's is running. The attention capture (scripts/attn_focus.py) finished; its 9.7 GB of q,k
  sit on scratch (purge-eligible; the figure needs only results/attn_focus/ and paper/figures/attn_focus/).

## Map
| file | what |
|---|---|
| paper/main.tex, paper/appendix.tex | paper source; switches \ifresfluxtwo, \ifresqwen, \ifresideogram in the preamble |
| paper/figures/src/fig_speed.py | Figure 2 + snippets/speed_numbers.tex + web speed exports |
| paper/figures/src/fig_attn_focus.py, attn_focus_data.py | Figure 3 (--extract reads the scratch capture) |
| scripts/attn_focus.py | CPU/GPU attention capture (q,k of all blocks at chosen steps) |
| paper/figures/src/fig_showcase_models.py | --model FLUX.2-dev / Qwen-Image writes the per-model slots |
| paper/build_delivery.py | build, ZIP of the files the build read, fresh-unzip check |
| .agent/notes/paper_v4_20261108.md | data decisions, layout facts, page budget, scratch test tools |

## Decisions
- Speed ratios divide two times of the same kind: clean over clean (prog_latency sweep 3, latency curves) or campaign
  over campaign (UltraImage, DyPE, SEGA); hollow markers mark campaign-only methods.
- The focus figure's heads follow a rule (block 12, highest and lowest share within 16 tokens in the reach archive);
  the 1024^2 image is a CPU re-run of progressive SCA's first stage, said so in the caption.
- Newer models never add an abstract line: the single-pass sentence swaps "keeps more fine detail" for the model list.
