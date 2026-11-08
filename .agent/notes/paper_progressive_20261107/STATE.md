# Paper owner state: progressive-SCA rewrite (2026-11-07, paper/ and web/ owner)

## Goal
Rewrite paper/ and web/ with progressive SCA as the headline and single-pass SCA as the fast mode; no duplication
content except the one dictated limitations sentence. Lead's binding priorities (later message): (1) one-place
protocol, (2) margins with paired deltas (seeds 0 and 1), (3) curated main-text comparison grid prog/AReA/HRDiT/
UltraImage + large curated appendix gallery + one random gallery, (4) method diagram rebuilt from two candidates,
(5) interim delivery (DONE). Then two render-review cycles, final delivery, latest_delivery.md, HANDOVER 31, web.

## Status and next action
- DONE, pushed: main text rewrite; appendix rewrite (agent); hero, zoom figure, uncurated progressive gallery
  (agent); clean latency (49.2/107.0/202.4 s, half-up from raw 49.153); set-level vs all baselines; seed 1; one-place
  protocol; interim delivery /scratch/09032/saini_2/sca/paper_builds/delivery_ru0txknm (commit 80078a9, lead told);
  Figure 5 wrap fix; render-review cycle 1 done (reviewers A p12-33: 13 MUST, B p34-55: 16 MUST), forwarded to the
  appendix agent in two batches.
- RUNNING: appendix agent (cycle-1 fixes + gallery placeholders); figure agent (fig:qual_prog grid
  prog/AReA/UltraImage/HRDiT, gallery_prog_curated, gallery_prog enlargement); diagram agent (method_v2a/b -> method_v2);
  web agent; watcher (seed 2, HRGen-1K).
- NEXT: integrate qual_prog into main text (replace or complement Figure 6; keep page budget); integrate method_v2
  (replaces concept wrapfigure + Figure 3; repoint fig:concept/fig:method refs; caption); cycle 2 render review (main
  by me, appendix by two reviewers); final delivery; notes; HANDOVER 31; web commit.

## Results status
- LANDED: prog195_vs_baselines (+cfid), prog195_pairs, prog_latency, prog195_s1.
- PENDING: prog195_s2, HRGen-1K progressive. Krea unsupported (single-pass only).

## Decisions
- Table 1 colours every row against progressive SCA; HRDiT/progressive clean times from sweep 3.
- Commit only my files while agents work; no autostash pulls.
- Report to lead: VAE tiled-decode row seams (0.5-2 grey levels) seen in progressive images by the figure agent.
