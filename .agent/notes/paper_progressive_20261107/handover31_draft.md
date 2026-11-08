## 31. 2026-11-07 evening: paper and web rewritten with progressive SCA as the headline (paper owner)
User decisions (binding, via lead): progressive SCA (HRDiT's three-stage schedule with sca_v2 attention at the
2048^2 and 4096^2 stages) is the headline, single-pass SCA the fast mode; framing per
.agent/notes/positioning_20261105/recommendation.md (reach is a property of the head; variance split as the hook);
no duplication content anywhere in paper or web except one dictated limitations sentence.
- Source layout: paper/main.tex (preamble + main text) and paper/appendix.tex (\input after the bibliography).
- Main text: new title "Scale-Covariant Attention: Per-Head Receptive Fields Calibrated at the Training Resolution";
  abstract, introduction (concept figure fig:concept beside paragraph 2), bullets, method with Section 2.5 progressive
  SCA and Algorithm 3 (alg:prog_main), Figure 3 lane (d), Section 3 "Reach is set by the head" led by the variance
  split (97.5% head, 2.5% noise level; Krea 97.6%, FLUX.2 99.1%; Table B.1), experiments around Table 1 with
  progressive SCA as the headline row and every other row coloured against it, Figure 5 (both modes minus AReA),
  Figure 6 (four crops; progressive softer than single-pass in every crop), Table 2 ablations and Table 3
  generality without Dup. columns, related work, limitations with the fidelity trade-off and the dictated sentence,
  conclusion. Main text ends on page 10.
- Appendix: duplication appendix, band guard, frequency argument, repeated-subject failure row and every duplication
  sentence removed; Appendix L progressive SCA (protocol, Table L.1 prog_all against every baseline, Table L.2
  prog_deltas, Figure progressive); SigMa/budget table (tab:panel46) in Appendix N; six-method figure (fig:teaser)
  and uncurated progressive gallery (fig:gallery_prog) in Appendix S; configuration named on every table and figure.
- Generators: gen_main_tables.py (Table 1 headline, colours vs progressive, generality with progressive rows,
  prog_all, variance_split, headline macros incl. sweep-3 latency), fig_concept.py, fig_paired_prog.py,
  fig_hero_prog.py, fig_gallery_prog.py, fig_zoom4k.py; scripts/variance_split.py ->
  results/heads_view/variance_split.csv and concept_curves.csv.
- Results folded in: results/prog195_vs_baselines (per-image pairs vs every baseline), prog195_pairs/cfid_vs_{area,
  sca_v2}, prog_latency (sweep 3: progressive 49.1 s, HRDiT 107.0 s, AReA 202.4 s; 4.1x and 46%).
- PENDING at delivery: (fill in)
- Delivery: (fill in) ; rubric: (fill in)
