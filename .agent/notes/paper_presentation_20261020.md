# Paper presentation overhaul, 2026-10-20 (lead: Opus subagent of the SCA coordinator)

Brief: Figure 3 revamp; >=2 new main-text visuals; Appendix E code -> one-page pseudo-algorithms;
more method + protocol detail; spacing pass; main text <= 10 pages; two render-review cycles;
delivery build + fresh-unzip check; update latest_delivery.md. No table numbers / prompt counts
changed (protocol agent owns those). No commits.

Helpers (opus, background, disjoint files), resumed 04:21 after the usage-limit stop:
- method figure -> figures/method_flow.pdf (src figures/src/method_flow.tex)
- zoom 4096^2 (MFlow Fig 22 style) -> figures/zoom4k.pdf (replaces Figure 5 qual_main)
- plots -> figures/paired_area.pdf (forest plot, wrap in 4.1), figures/pi_heads.pdf (Section 3)
- text -> paper/main.tex only (Jobs A+B); Section 6 must end y<=600pt on p10 with old figures

Lead integration plan: swap includes, new captions, spacing via \setlength (textfloatsep,
floatsep, intextsep, wrapfig), widow/club penalties, renders at 105 dpi (helpers in scratchpad
render.py / sheet.py / pos.py), two review cycles, then build_delivery.py.
Baseline geometry (old build): Fig 3 image y 92-270 on p3; Section 6 ends y~550 p10;
Reproducibility y=559, References y=680 on p10.

## Update 06:00 (lead additions)
- Lead: method figure must match AReA/BudgetPix/LoT diagrams; final source figures/src/method.tex -> figures/method.pdf (forwarded to method helper).
- Lead: explicit scale-coverage statement in 4.4 (2048..8192, 2048x4096, 2048x6144, FLUX.2-dev, Krea), optional compact table (forwarded to text helper).
- Lead: after zero-overfull fresh-unzip build, COMMIT paper/ + .agent/ (plain message, no trailers), git push origin main, update HANDOVER Section 15 with build path; report <150 words.
- Coordinator protocol decision: move 191 and 46-panel tables to 195 (ablations stay 46 = fixed ablation subset), disclose 4 dev prompts, patch FID ours only, add tab:published_compare (figures/snippets/published_compare.tex) in Section 4, HRGen-1K at 1000. Edits list: .agent/notes/eval_protocol_survey.md Section 5. Numbers when /scratch/09032/saini_2/sca/eval/a4k195_{main,panel}/BOOTSTRAP_DONE (+ a4k195_2048x4096, a4k195_2048x6144) appear: regenerate tables via gen_tables.py pattern, copy CSVs to results/ with SHA256SUMS. Seeds/krea195/hrgen1k_krea later: keep old numbers and counts unless they land.
- Background waiter (bash until-loop, 10 min, 8 h cap) watches the two BOOTSTRAP_DONE markers.
- Done: plots helper (paired_area.pdf 182.8x173.4pt wrap; pi_heads.pdf 397.5x88.2 strip), zoom helper (zoom4k.pdf 397.5x220.7, MUSIQ_c per image verified).
- Pending: method helper, text helper; then protocol helper (fresh opus) for the structural edits.

## Update (after helpers 1-4)
- Done: method.pdf rebuilt (src figures/src/method.tex, 397.5x122.4pt, 24 words, block 8 pi table, assets figures/method_flow/);
  text helper: "Algorithms in full" appendix (Alg 3-5, p22), method/protocol detail, 4.4 coverage statement; Section 6 ended y=557 p10 (old figs).
- Launched integrator helper (opus): integrate method/zoom4k/pi_heads/paired_area with lead captions, protocol structural edits
  (survey Section 5, tab:published_compare, count macros \nmain/\npanel/\nabl), spacing pass, fit 10 pages.
- Numbers phase (when BOOTSTRAP_DONE markers land): regenerate tables (gen_tables.py), flip macros to 195, regenerate
  dependent figures (paired_area from final4k_nb_* -> 195 CSVs, quality_time, baseline_deltas), copy CSVs to results/ + SHA256SUMS.
- Then: lead review cycles (2), build_delivery.py, commit paper/ + .agent/ (no trailers), push origin main, HANDOVER Section 15, latest_delivery.md.

## Update (lead review cycle 1 done on build lead_r1, 48 pp, 0 overfull, 0 undefined)
- Integrator finished: figures in (Fig 3 method, Fig 4 pi_heads p5, Fig 5 paired wrap p7, Fig 6 zoom p8, Fig 7 quality_time wrap p8),
  Table 2 published_compare p7, macros \nmain \npanel \nseed \nabl \nhrgen \nkrea, spacing (\wraptop/\wrapbot), Section 6 ends y=722 p10.
- Cycle-1 findings: "Design choices." run-in mid-paragraph p3; no slack on p10; Tables 4/5 on p10 (cited p8);
  appendix p27 holds Table J.3 alone (tab:app_wide, generated region); p21 bottom gap (acceptable).
- Launched finisher helper (opus): Step 1 layout fixes, Step 2 195 switch when markers land (6 h cap, own waiter),
  Step 3 two review passes + build_delivery.py + latest_delivery.md. Lead waiter stopped (PIDs 2851782, 2990978).
- After finisher: lead final visual check, then commit paper/ + .agent/ (no trailers), push origin main, HANDOVER Section 15.
