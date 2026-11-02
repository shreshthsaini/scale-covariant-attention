# Reviewer 2: benchmarking and evaluation (round 1, 2026-11-02)

Sources: main.pdf (53 pp.), main.tex, results/, eval_protocol_survey.md. SCA = final operator with the schedule
(sca_v2); nb = the operator without it. No measured number is questioned.

## 1. Coverage: setting x baseline

P = present with final SCA. S = stale, only against nb (not Table 1's method). M = missing. CSV = fillable from
results/ by regenerating snippets; run = needs GPU time.

| Setting (page) | Full | AReA | DyPE | SEGA | SigMa | UltraImage | HRDiT | Published |
|---|---|---|---|---|---|---|---|---|
| 4096^2, 195 (Tab. 1 p6, Tab. 2 p8, K.2 p29, L.2 p34) | P | P | P | P | P | P | P | P (H.1 p26) |
| 4096^2 FID/KID deltas vs SCA (L.5 p34) | M | P | S | S | M | S | M | n/a |
| 2048x4096, 195 (K.3 p30) | M | P | M | M | M | M | M | M (AReA Tab. 1 has all baselines here) |
| 2048x6144, 195 (K.3 p30) | M | P | M | M | M | M | M | M (AReA uses 6144x2048) |
| 6144^2, 100 (L.1 p33) | M | S | M | M | M | M | M | M (SEGA: 20 prompts) |
| 8192^2, 12 (N.1 p38) | latency only | S | M | M | M | M | M (HRDiT paper has 8K) | M |
| Seeds 1, 2, 195 (Tab. 4 p9, L.3) | M | P | M | M | M | M | M | n/a |
| HRGen-1K, 1000 (Tab. 4, H.2 p26) | M | P dev, M Krea | M | M | M | M | M | P; SCA-Krea row grey "pending" though results/hrgen1k_krea_sca_v2 holds sca_v2_krea: CSV |
| Krea-dev, 195 (L.1, H.1) | M | P | 100, dagger | 100, dagger | M | M | M | P; schedule row in results/krea195_sca_v2: CSV |
| FLUX.2-dev, 50 (Tab. 4, L.3) | S | n/a | M | M | M | M | M | n/a |
| Progressive, 195 (Tab. 2, M.2 p37) | n/a | M | n/a | n/a | n/a | P | P | n/a |
| Budget rows, 195 (Tab. 2, M.1) | M | M | M | M | M | P | P | n/a |
| Ablations, 46 (Tab. 3, K.5) | SCA variants only; field practice (AReA Tab. 5 uses 20) |
| Duplication 4096^2 (Tab. 1, Fig. 4b) | P | P | P | P | P | P | P | n/a |
| Duplication non-square, seeds, Krea | M | P | M | M | M | M | M | n/a |
| Clean batch-1 latency (D.2 p21, Fig. 8 p9) | P (not 8K) | P | n/m | n/m | n/m | n/m | 4096^2 only | H100 numbers in text only |

Gaps: (a) non-square sizes carry AReA alone, while AReA's Table 1 shows every baseline at four sizes; (b) 6144^2,
8192^2, FLUX.2 and L.5 use the retired operator; (c) clean latency exists for three of eight methods. Fills: regenerate the two pending Krea snippets (CSV); rerun SCA at 6144^2 (100 prompts, about 5 GH200-h), 8192^2
(12, 1 h), FLUX.2 (50); one back-to-back latency sweep of all eight methods at 4096^2 (about 30 images, under 2 h);
DyPE/SEGA on the 95 missing Krea rows (about 10 h).

## 2. Protocol issues

| # | Issue | Severity | Fix |
|---|---|---|---|
| P1 | Operator mismatch: 6144^2 "14 of 15", 8192^2, FLUX.2, sensitivity, L.5 and M.1 describe nb, not Table 1's method; M.1 says every SCA point is without the schedule, yet Table 2's Prog. SCA 50.7 s is sca_v2_prog. | High | Rerun the few missing sca_v2 cells; until then put "SCA w/o schedule" in the column head, not only the caption. |
| P2 | Duplication detector validated on 20 SCA-nb images only (validation_labels.csv has no baseline rows). AReA's 4% may be under-detection on blurrier outputs. | High | Label 20 images per method (160 images), report per-method agreement; add Wilson intervals to Table 1's Dup. column. |
| P3 | Timing: AReA and full attention from sweep 1, SCA from sweep 2; AReA's campaign/clean ratio is 1.31, SCA's 1.02, so campaign means (the only numbers for DyPE, SEGA, SigMa, UltraImage, Prog. SCA) are not comparable within the Time column. "2.63x" is sweep 1 over sweep 2. | Medium-high | One sweep, all methods, same GPU, same three prompts; campaign means in a separate unranked column. |
| P4 | Checkpoint confound is handled for AReA (H.1), but DyPE/SEGA on Krea are 100 prompts and SigMa, HRDiT, UltraImage have no Krea run. | Medium | Finish DyPE/SEGA Krea at 195; note that SigMa's code also defaults to Krea. |
| P5 | FID with 195 images is dominated by finite-sample bias (values 138 to 312); the headline -5.6 is a 4% shift on it. KID is unbiased and already computed. | Medium | Keep FID (convention) but lead the distribution claim with KID and state the bias once in Section 4. |
| P6 | Patch FID: 780 crops, own definition, not comparable to AReA's 58.2 (our rule gives AReA 115). Table 1 still bolds FID_p beside AReA. | Medium | Dagger in Table 1 ("own crop rule, Appendix G"); keep it out of H.1, as now. |
| P7 | Interval methods differ across tables: 1000 vs 2000 draws, reference-row vs identity clusters, "historical" patch intervals omitted; Table 1 and Figure 7 do not say which. | Medium | One recipe (2000 draws, identity clusters) re-derived from the per-image CSVs; CPU only. |
| P8 | "12 of 15 metrics" counts HPSv2/HPSv2.1, MUSIQ/MUSIQ_c, FID/KID/FID_p/KID_p as independent; unadjusted (p10). | Medium | Report wins by family (6 preference, 4 no-reference, 5 distribution) and a Holm-adjusted count in K.2. |
| P9 | Dev prompts (4) and the schedule-selection subset (46) sit inside the 195; disclosed; the 15% vs 26% gap shows the optimism was real. | Low-medium | Also report the 149 prompts outside the subset, one line. |
| P10 | One seed per prompt vs AReA's single seed; 8K HPS in fp32; MUSIQ chunking at 8K. All disclosed. | Low | Move the fp32 remark into N.1's caption. |

## 3. Presentation against AReA, HRDiT, LILIE, MFlow

- Table 1 (p6): bold and underline run across both groups, so HRDiT holds bold on five columns and "best single-pass"
  lives only in the caption. LILIE/MFlow bold per group and set context rows in gray. Fix: bold within group, gray
  text for the multi-stage rows.
- Time column mixes clean and campaign (bold 77.1 against n/m). Rank the clean column only.
- Dup. header says "share of the 95" but prints counts. Print "26/95" or a percentage with its interval.
- Table 2 (p8) carries +/o/* significance markers; Table 1, "columns and bold as in Table 1", carries none. One
  convention for both.
- Table 4 (p9): three metrics, no intervals, no Dup.; "+12.56" MUSIQ_c needs its scale stated. Add Dup. (rates exist
  for seeds and Krea, 29.5%).
- Non-square results are AReA-only on p30; AReA puts four sizes in its Table 1. Add a two-row-per-size summary in
  Section 4.3 with Dup. (24%, 33% vs 18%, 6%).
- H.1 ranks published and our-run rows together (bold published AReA 1.32, underline our SCA-Krea 1.30). Rank within
  block; seeds differ across blocks.
- Captions ending in "Key result:", claim-first appendix captions, Figure 7's forest plot and "Table K.1 adds all
  metrics" meet the bar. Figure 8 shows three methods; a per-size table of campaign means would make "faster than
  every baseline we timed" checkable.

## 4. The most damaging question

"On the checkpoint AReA was built for (Krea-dev), SCA ties AReA on ImageReward (+0.02 [-0.02, 0.06]), CLIP and FID,
loses patch FID, and repeats subjects in 29.5% of prompts against about 2%; the 12-of-15 win appears only on
FLUX.1-dev, where AReA's code runs off its defaults. Isn't the contribution speed, not quality?" (L.1 p33, H.1 p26,
krea195_sca_v2/dup_rates_people.csv.)

Pre-empt in the main text: (i) a three-checkpoint block in Section 4.5 (FLUX.1-dev, Krea, FLUX.2) with wins, losses,
Dup. and time, stating that on Krea the result is parity in a third of the time with more duplication; (ii) complete
the Krea block (DyPE/SEGA at 195, SigMa) so SCA >= AReA > others holds on both checkpoints; (iii) per-method detector
validation (P2); (iv) back-to-back latency of all eight methods (P3) in Table 1, since speed is the claim that
survives on every checkpoint.
