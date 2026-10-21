# SCA live state, 2026-10-18 11:40 CDT (operator: Claude session, resumed)

## Operator change (read first)
- The Codex continuation chain ended: SCA10/SCA11 were cancelled 2026-10-17 13:27, successor 1077572 never
  ran, no SCAC job exists. Do NOT restart the controller unless the user asks; Claude orchestrates now.
- User 2026-10-18: request many GPUs in ONE job and hold them until all work is done; finish experiments
  and polish the paper to the level of top industry papers (NVIDIA SOL / Nemotron style).
- Submitted SCA12 = job 1078982 (gh, 8 nodes, 16 h, IDLE_MAX=14400) via login2 after balance check
  (7679 SU available; keep >= 1600). Guest worker on c608-101 (gpu_workspace) runs a_000 repro smoke.
- Codex probe failed (treat unavailable); use Claude subagents (opus).
- Remaining GPU work in taskq/pending: 4 a_ (repro smoke, HRDiT/SigMa smokes, full 6144 timing),
  20 i (6144/8192), 36 j (Krea). Then baseline promotion per repro/baseline_promotion_1077042/README.md.


## Goal
Finish experiments, verified evaluation, the paper and reproducibility delivery.
Codex owns the whole project. Preserve unfavorable results and novelty boundaries.

## Status and next action
- Unit1077042 completed baseline promotion/evaluation hardening. All28 combined CPU
  tests passed:15 promotion and13 evaluation. The separate helper prepares reviewed
  12/46 candidates, sizing missing prompts from twice the maximum observed time
  plus explicit startup headroom. It never enqueues or submits jobs; it retains the
  frozen runner, manifest, output root and WandB. Partial/corrupt outputs refuse.
- Evaluator now preserves all4 GPU reports, including paired deltas; binds reference
  image bytes across195 rows/168 images; checks PNG CRC; explicitly uses4 CLI threads.
  Verified39 frozen files,46 identities for each historical control,6 sampled PNGs.
  New baseline GPU generation/scoring and CPU intervals remain unrun. Smokes absent.
- NEXT: existing GPU event. At17:47:50 UTC, SCA10/1077447 and SCA11/1077473 remain
  PENDING. No SCA GPU allocation active;60 pending,0 running,0 failed,894 done.
  All60 admission headers fit fresh2h windows with120s cleanup. This is not runtime
  proof. No submission, task mutation or controller change. Do not duplicate retries.
- CPU1077042 runs on i618-043; deadline16:41:12 America/Chicago(21:41:12 UTC).
  Successor1077572 is PENDING on dependency. Frozen controller hashes unchanged.
  Currently actionable CPU work is done; return waiting so the controller parks on
  existing GPU events. Do not change successor/dependency or launch another watcher.
- Generation:6K AReA56/100,SCA64/100; Krea AReA0/100,SCA20/100. Strict readiness
  remains closed. Neither E2/E9 CPU READY nor CPU COMPLETE marker exists. When READY
  appears, run its hold_cpu_finalize script once on CPU,1000 reference-image-cluster
  draws with bounded threads. Do not rerun a completed bootstrap.
- After the GCC fresh GPU smoke, review its distinct receipt and promote to canonical
  environment/fresh_gpu_smoke.json, preserving the original failure. Full6K gridfix
  timing remains pending,4800s+cleanup.13 clean groups audited; only full6K pending.
  AReA6K482.2844s/8K951.9874s; SCA161.5585s/287.6398s;2.9852x/3.3097x.
  All7 timing logs identify GH200120GB.
- Baseline next: inspect smoke images/profile fidelity, write a hash-bound review
  with the new helper, prepare12 candidates sized to the fleet, inspect live task
  ownership, then publish. After12 review, prepare46 and evaluate exact8-method
  pairs. Consider affordable191 only after measured cost/protocol review and a
  separate frozen manifest. Promotion must never depend on SCA winning.
- Krea overlay:100DyPE+100SEGA,193 originals+7 replacements. All7 repaired PNGs
  are byte-identical to originals;200 original sidecars unchanged;207 PNGs audited.
- Paper revamp4011238:38 pages,473 source files, independent ZIP build,38 identical
  72dpi renders, changed pages inspected at144dpi, no overfull boxes. Method diagram
  PDF/SVG/source; authentic6-method portrait; paired intervals; readable tables.
  Main30 cells independently verified;42 ablation cells preserved. Paper unchanged
  by this CPU unit. PDF/ZIP hashes remain those in latest_delivery.md.
- After GPU evidence: archive validated E2/E9 results; exact tables; complete6K/Krea
  galleries; reconcile abstract/main interpretation/appendix/limitations. Rebuild
  PDF+ZIP; independently build extracted sources; review appearance and numbers.
- Complete:E1/E4/E5/E6/E7/E8,budget controls, exploratory12-prompt8K metrics/gallery.
  Three fresh metric envs pass bounded CPU parity exactly:39 assets,12.08GB,
  0 downloads,42 focused/71 combined tests. GPU/native4K MUSIQ/full reruns untested.

## Map
- HANDOVER.md first85 lines: current authoritative summary before historical material.
- results/evidence/continuation/unit_validation_1077042.json: latest live checks,
  pending task hashes, frozen hashes and28 tests; read before task actions.
- .agent/notes/unit_1077042.md: completed unit and exact next commands.
- repro/baseline_promotion_1077042/README.md: after smokes, exact12/46 commands.
- .agent/notes/baseline_eval_audit_1077042.md: evaluator fixes and planner review.
- repro/baseline_evaluation_20261017/audit_validation_1077042.json: control audit.
- repro/baseline_evaluation_20261017/README.md: exact8-method GPU/CPU handoff.
- .agent/notes/baseline_audit_20261017.md: primary sources/protocol/novelty review.
- repro/baseline_expansion_20261017/README.md: original frozen generation profiles.
- repro/runtime_repairs_20261017/README.md: repairs already released; no duplicates.
- results/evidence/metric_environment/validation_summary_1076947.json: CPU parity.
- .agent/notes/latest_delivery.md: current PDF/ZIP hashes and appearance proof.
- .agent/notes/paper_numeric_audit_20261017.md: result insertion map; old line counts.
- slurm/CONTINUATION.md: frozen controller; scratch continuation/status.json is live.

## Decisions
- Keep originals, negative evidence, frozen packages and all scratch artifacts.
- No remote; commit explicit paths locally. No posting or external messages.
- Before any submission use one sequential login2 SSH; check balance+reservations,
  retain1600SU and respect96% burn cap. Historical balance7908/40000SU is stale.
- No submissions in this unit. Preserve foreign SSH/jobs; guard before fan-out.
- Promotion uses fidelity, finite valid output and measured cost, never quality wins.
- HRDiT57 progressive steps are a separate budget. SigMa paper dev/g3.5 differs from
  released Krea/g4.5; released mu=max_shift. Identical aliases are not extra controls.
- Reference195 rows/168 images: main patchFID/KID gain unresolved.8K has6 resolved
  gains/4 unresolved; MUSIQ means favor AReA; stripe/grid failures retained.
  8K CPU float32 HPS differs from main GPU precision. Calibration runtime unknown.
- Native emulation+fitted/transferred head/noise mixing is bounded novelty; ingredients
  established. Exact coordinates do not repair bf16 drift;0.015625 is observed error.

## Open questions
- Waiting for existing SCA fleets; new baseline GPU fidelity/cost unknown until smokes.
- E2/E9,full6K timing,fresh GPU,final evaluation,broader baselines,paper remain unfinished.
- No official ResDiT code verified. Do not claim an invented reproduction.
## 2026-10-18 12:00 env repair: HRDiT smoke failed (torch._thread_safe_fork missing; scratch purge deleted
zero-byte files). Restored 329 empty files from package RECORDs in uv-envs/sca (scripts/restore_empty_files.py;
rerun after purges). HRDiT smoke requeued. User asked for separate max-effort Opus design agent(s) for all
figures/plots/hero/diagrams and a final max-effort polish of the whole paper; paper editor told to leave figures.
## 12:20 smokes done: HRDiT valid coherent; SigMa valid but lattice-degraded (kept). Review /scratch/09032/saini_2/sca/baseline_review12_1077042.json; published pilot12 tasks (a_b_pilot12_*). After both complete: repeat prepare.py --stage pilot46 with a new review (see repro/baseline_promotion_1077042/README.md).
## 12:35 Figure redesign committed 9ee46b6 (house style src/scafig.py + sca_colors.tex; hero on p2; TikZ method; PDF ~29.6 MB). Final max-effort editor-in-chief agent running (owns paper/; applies DESIGN_CHANGES.md, fixes method-diagram overlaps, latency label, gallery floats, header; 2 render cycles; builds PDF+zip). SCA12 still pending.
## 18:05 pilot12 done (HRDiT coherent and strong; SigMa consistently degraded); review46 filled; published
pilot46 tasks (a_b_pilot46_*: HRDiT 012-046, SigMa 012-045 + 045-046). Next: when done, run the exact 8-method
46-prompt eval per repro/baseline_evaluation_20261017/README.md. Final editor agent resumed after rate limit.
## 19:25 Paper polish complete (commit above): 38 pp, main text 9 pp, teaser p2, PDF 29.6 MB, ZIP verified at
/scratch/09032/saini_2/sca/zipcheck/final_20261018_191838. Header = anonymous ICLR 2026 submission (\iclrfinalcopy
restores draft header). WARNING: some table blocks and gallery snippets are hand-edited; rerunning gen_tables.py,
gen_final_tables.py, fig_qual.py gallery_4096 or fig_phase2.py flux2 undoes them (see figures/snippets/README.md).
Remaining: E2 (6144/8192) and E9 (Krea) placeholders (13 \todo, 8 \tbd), HRDiT/SigMa 46-prompt 8-method table.
## 21:10 pilot46 done (HRDiT, SigMa). Eval workspace /scratch/09032/saini_2/sca/eval/baseline_expansion46_v1 (368 samples, 8 methods x 46). Queued a_eval_baseline46_gpu.sh; after it: run intervals_cpu.sh on a CPU node, archive to results/, fill paper baseline table. SCA12 start estimate 2026-10-19 11:39.
## 22:10 8-METHOD BASELINE EVAL (46 prompts, 4096^2; results/baseline46/; eval /scratch/09032/saini_2/sca/eval/baseline_expansion46_v1):
IR/HPS2.1/Pick/CLIP/MUSIQc/FID: HRDiT 1.373/.316/23.43/29.38/54.1/216.9; UltraImage 1.260/.315/23.31/28.77/42.9/213.8;
SCA 1.124/.300/22.96/29.41/62.5/206.4; AReA .833/.278/22.31/27.85/52.0/212.7; SEGA .469; DyPE -.234; SigMa -.771; full -2.214.
Budget (results via eval/budget_*): SCA 56 steps (140 s) IR 1.221, HPS2.1 .310, Pick 23.13, CLIP-IQA .727, MUSIQc 64.7.
=> SCA is best single-pass, best FID/patch FID/detail, fastest; HRDiT (progressive multi-stage, ~180 s) leads preference metrics.
Asked user: (1) frame honestly now, or (2) implement SCA inside a progressive (HRDiT-like) pipeline to beat HRDiT.
## 22:30 User: use ANY idle node/GPU to finish; keep checking. Added slurm/guest_worker_shared.sh (claims tasks when
GPU util <= 10% and >= 56 GB free; for reserved-idle nodes). Submitted SCAdev 1080047 (gh-dev, 2 nodes, 2 h).
Started progressive-SCA implementation agent (tasks into taskq/hold_prog; freeze code to code/PROG before release).
User: use Fable as a background supervisor/reviewer; Fable review of paper + framing + progressive plan running.
## 23:00 FABLE REVIEW (5/10 now; 6.5-7 after fixes). CORRECTION: HRDiT campaign mean 106.5 s (CI 103-111), SCA 71.5 s
-> SCA 1.5x faster than HRDiT (NOT 180 s / 2.5x; 180 s was one smoke image). Framing: "strongest single-pass
training-free operator at 4K; best FID/patch FID/detail of 8 methods; fastest; multi-stage recipes lead preference
metrics; 56-step SCA closes most of the gap". Queued: c_hrdit191_* (HRDiT on 145 remaining eval prompts ->
out/baseline_expansion_20261017_rest145), a_latency_hrdit_batch1.sh (clean timing, lat3 -> out/latency_hrdit),
b*_sca_s42 (42-step matched wall clock, abl50 -> out/budget/sca_s42). Progressive SCA = operator swap in HRDiT
schedule (agent told). Paper-fix agent (max effort) implementing Fable items A-E. Human 2AFC study not feasible.
## 23:20 Progressive SCA committed 57feaef, frozen code/PROG. Smoke released (pending/a_p_prog_a_smoke_000_002.sh). After it passes (4096x4096, timing): mv taskq/hold_prog/* to pending (10 gen chunks + gated eval vs hrdit, sca_s42, sca_nb, ultraimage).
## 23:41 SCA12 RUNNING on 8 nodes (c610-[001-002],c636-[042,051-052],c637-[081-082,091]); SCAdev cancelled (not needed).
## 23:45 prog smoke OK (4096^2; steady 56 s/img: stages 5.5/13.8/31.5 s; clean layout, slightly softer texture with faint sharpening halo). Released hold_prog (gen chunks + gated eval).
## 23:58 Paper fixes committed (Table 2 = 8-method panel; 11 of 15 vs AReA after corrected patch KID CI; 56-step 'narrows' gap). Editor asked to restore a compact calibration panel in Fig 2. TODO lead: update .agent/notes/latest_delivery.md after final build; rerun Fable review after progressive results.
## 01:17 c_hrdit191 done (145+46 HRDiT). Queued q_eval_hrdit191.sh -> eval/hrdit191_{extra,base} (HRDiT vs sca_nb/area/ultraimage, 191 prompts). E2 6144 eval running; Krea last chunks; progressive p_* next.
## 02:40 PROGRESSIVE SCA RESULT (46 prompts, 4096^2; results/progressive46/): HRDiT schedule with SCA attention (sca_prog)
vs HRDiT: IR 1.296 vs 1.373 (-0.077 n.s.), HPS2.1 .319 vs .316 (n.s.), Pick 23.41 vs 23.43 (n.s.), CLIP-IQA, aes, CLIP n.s.;
HPSv2 +0.005*, ARNIQA +0.030*; 49.0 s vs 106.5 s (2.2x faster; fastest of all methods). => ties HRDiT on preference at
less than half the time; HRDiT's lead is a schedule effect, not its attention. Single-pass SCA keeps best FID/detail.
sca_s42 (110 s): IR 1.185. Claim (Fable rule "tie"): SCA as drop-in operator matches the best multi-stage method at 2.2x speed
and leads single-pass. CPU bootstraps for E2 6144 / E9 Krea queued as a_cpu_* on SCA12 (local 1-core run stopped).
## 02:45 ALL GPU WORK DONE; SCA12 released (scancel 1078982), taskq/STOP set. Archived:
- results/hrdit191: HRDiT vs SCA single pass, 191 prompts: IR +0.109*, HPS2.1 +0.013*, Pick +0.36*, CLIP-IQA +0.028*,
  aes +0.091*; SCA better MUSIQ -5.0*, crop -7.7*; FID 138.4 (SCA) vs 139.1, patch FID 106.5 vs 110.7; CLIP tie.
- results/e2_6144 (100 prompts): SCA IR 0.841 vs AReA -0.421; FID 169.9 vs 216.7; patch FID 122.4 vs 188.6; 165 vs 508 s.
- results/e9_krea (100 prompts, Krea checkpoint): SCA ~ AReA (IR 1.198 vs 1.240; HPS2.1 .297 vs .290; FID 165.8 vs 166.6;
  patch FID 123.9 vs 115.5); 73 vs 217 s. SEGA strong on Krea (FID 164.2).
Next: paper editor (running) integrates these; then Fable final review; rebuild; send PDF+zip.
## 03:35 (Fable operator) User: figures/diagrams/plots must follow the style of the reference papers (AReA/Toronto,
"left loop diffusion transformer", "budget picks", a ByteDance paper, a DeepMind paper) and carry much LESS text.
Running: final text/placeholder editor (max effort) on main.tex; reference-style study -> paper/figures/REFERENCE_STYLE.md.
Then: figure restyle pass (remove text from diagrams, follow REFERENCE_STYLE house rules), lead re-review, deliver.
All compute done; all results archived (incl. latency_curve_batch1.csv, progressive46 distribution intervals).
## 2026-10-19 12:15 (Fable operator) Opus WEEKLY limit hit (resets Oct 22 12:00 CT): Opus subagents unavailable.
Teaser restyle finished before cutoff and committed (e683c34: 53 -> 9 labels). Remaining restyle running on
Fable subagents with disjoint ownership: method diagram (method.tex/pdf + Fig 2 caption), data plots (latency,
scatter, deltas, calibration plots + captions), galleries/qualitative (fig_qual.py, fig_phase2.py, galleries + captions).
Then: lead rebuilds PDF+zip, re-reviews pages, delivers. Text is final at c123913.
