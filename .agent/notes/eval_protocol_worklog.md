# Eval-protocol owner worklog (2026-10-20, Opus subagent; scratch facts, not the deliverable)
- Prompt rows: aesthetic4k_eval.jsonl 195 unique ids; dev4 = rows 0-3 (= /scratch/09032/saini_2/sca/dev4.jsonl);
  abl50 = rows 0-49; pilot46 = rows 4-49; hrdit_rest145 = rows 50-194.
- Outputs with all 195 at 4096^2: out/main/{area,sca_nb,sca,dype,full,sega_official,ultraimage}; prog/sca_prog;
  budget/{sca_g7,sca_s42,sca_s56}; seed1/sca_nb. area/sca_nb also 195 at 2048x4096, 2048x6144.
- Missing rows 0-3: hrdit_dev_author (v1 46 + rest145 145), sigma_dev_matched (v1 46 + a4k191 145).
- Seeds in flight: seed1/area 70, seed2/{sca_nb,area} 50 (a191_7/8/9 pending, 79 chunks at 04:20 CT).
- Spool: workers take `ls pending | sort | head -1`; rc 0 -> done else failed. Templates: done/c_hrdit191_000_010.sh,
  done/a191_1_sigma_050_055.sh (+ hold_191/body), hold_eval/z910_eval_sigma191.sh (sleeps 1200 s: do NOT copy that).
- AReA 2610.17227 Table 1 FLUX.1 4096^2 (Lat, IR, Aes, CLIP, Pick, FID, FIDp): Base 314 -0.7177 5.80 25.72 20.34 179.8 112.0;
  UltraImage 256 -0.1379 6.11 26.77 20.85 169.4 82.0; DyPE 314 1.1070 6.32 28.98 22.79 155.2 95.4;
  SigMa 322 1.0149 6.16 28.36 22.47 156.8 83.3; SEGA 319 1.2702 6.31 29.16 23.16 150.6 71.8; AReA 151 1.3250 6.37 29.71 23.31 148.5 58.2.
- AReA Table 6 (4096^2 A4K; Lat IR Aes MUSIQ CLIP-IQA ARNIQA CLIP HPS FID FIDp): HiFlow 182 1.1710 6.3147 38.674 0.6561 0.4811 29.3 0.2951 153.0 59.0;
  I-Max 440 1.1716 6.3043 41.086 0.6838 0.5013 29.2 0.2956 154.8 60.1; ScaleDiff 135 1.2339 6.2785 44.496 0.7066 0.5663 29.2 0.2935 152.5 73.7;
  AReA-Bootstrap 126 1.2658 6.3402 43.224 0.7027 0.5479 29.4 0.2955 152.0 61.2; AReA 151 1.3250 6.3717 45.217 0.7132 0.5942 29.7 0.2969 148.5 58.2.
- AReA Table 2 HRGen-1K FLUX.1 4096^2 (IR Aes CLIP Pick HPS IS): Base -1.4649 5.12 23.36 19.04 0.232 7.20; UltraImage -0.0277 5.77 28.26 20.46 0.269 10.17;
  DyPE 0.3280 5.83 29.35 20.94 0.266 10.93; SigMa 0.0627 5.63 27.91 20.54 0.258 10.70; SEGA 0.4422 5.82 29.63 21.18 0.270 11.99; AReA 0.6001 5.87 29.89 21.44 0.275 13.90.
- AReA Table 3 full row: Lat 151 IR 1.32 Aes 6.37 MUSIQ 45.22 Pick 23.31 HPS 0.297 CLIP 29.71 FID 148.50.
- AReA text: Sec 4 "Aesthetic-4K ... 195 prompts paired with high-resolution reference images"; HRGen-1K 1000 prompts (App. H);
  sizes 6144x2048, 2048x4096, 4096x4096, 5120x5120; FIDp "computed on local crops" (no crop protocol); App. G.3 "FID also resizes every image to 299^2".
  No steps/guidance/checkpoint/seed/metric-implementation stated; "FLUX.1"; H100; all baselines official implementations. Repo: no eval code; default Krea-dev, 28 steps, g4.5, one seed for all prompts.
- Research agents (4) resumed 04:21 CT; outputs to scratchpad research/*.md.
- 04:40 CT: KEY FINDING. Our FLUX.1-Krea-dev runs at 4096^2 (results/e9_krea, rows 4-103) reproduce AReA's published Table 1
  numbers: AReA CLIP 29.69 vs 29.71, Pick 23.23 vs 23.31, Aes 6.36 vs 6.37, MUSIQ(full) 45.25 vs 45.217, HPSv2 0.2941 vs 0.2969,
  IR 1.24 vs 1.325; on FLUX.1-dev our AReA gives IR 0.92, CLIP 27.67, Pick 22.35. => AReA Table 1 = Krea-dev (repo default, 28 steps, g4.5).
  FIDp does not match (ours 115.5 vs 58.2): crop protocol differs and is unstated.
- Published dev4 generation to pending: a190_dev4_1_hrdit_000_004.sh, a190_dev4_2_sigma_000_004.sh (bodies+wrappers in
  taskq/hold_195/; out root out/baseline_expansion_20261017_dev4; ~0.4 GPU-h).
- link_common.py defaults --exclude dev4.jsonl: never use it for 195 evals. Eval pipeline = z910 clone (eval_extra + CR eval.py
  + evaluate_corrected bootstrap 0 + paired_bootstrap); CPU set-level CIs via evaluate_corrected --device cpu --bootstrap 1000.
- Templates: Krea gen done/j005_{sca,area}_krea_4096x4096.sh (code/PHASE2, eval100=rows 4-103); HRGen gen done/h000_{sca_nb,area}_4096x4096.sh
  (code/2d11740, hrgen200.jsonl = AReA HRGen-1K rows 0-199 + dataset/index keys); HRGen eval done/h*_eval_e4_hrgen.sh.
- Timing: eval ~3-4 s/img (q_eval_hrdit191 764 imgs 49 min); SigMa 195 s/img, HRDiT 111 s/img, SCA 71.5, AReA 202-262, SCA-Krea 72.7, AReA-Krea 217.5.
- 05:00 CT agent B (research/{hrdit,ultraimage,resdit,scalediff}.md): none uses Aesthetic-4K or HRGen-1K; all FLUX.1-dev; prompts =
  1K random LAION-5B captions (HRDiT Sec 4.1, UltraImage Sec 6.1, ScaleDiff Sec 4.1) or 500 captions (ResDiT Sec 4.1);
  FID refs = 10K LAION (HRDiT, UltraImage), 2K LAION-Aes (ResDiT KID only), ~1K paired LAION (ScaleDiff); patch: 10 random crops
  (HRDiT, after AccDiffusion), DemoFusion-style (ScaleDiff); crop size NOT STATED anywhere. UltraImage FLUX only at 3600^2.
  HRDiT Tab1 4K FLUX: Ours FID 64.91 FIDp 51.84 KID .0078 KIDp .0074 CLIP 31.17 116 s (H200).
- Generated (not released): hold_eval z960-z968 (+ CPU CI scripts in hold_cpu_finalize/{a4k195_*,krea195}_bootstrap.sh);
  hold_krea195 38 tasks; hold_hrgen1k_krea 100; hold_hrgen1k_dev 240. Generators: /scratch/09032/saini_2/sca/code/eval195_20261020/.
- 05:15 CT agent D (research/{freescale,scalecrafter,fouriscale,demofusion,other2026}.md): classic SD papers use LAION subsets
  (FreeScale 1024 LAION-Aes-6.5+ captions, refs = own 1024^2 outputs; ScaleCrafter/FouriScale 10k LAION-5B, clean-fid FIDr/KIDr;
  DemoFusion 1K LAION-5B, FIDcrop = 1x-res crops resized, count unstated). Crop FID never reproducibly defined (root: Chai et al.
  2204.07156, 50k random patches). HRGen-1K only in AReA. Aesthetic-Eval@4096 = 195 imgs from LAION-Aes V2 6.5+ (Diffusion-4K v2 Sec 3.1).
  UltraFlux 2511.18050 (trained) uses Aesthetic-Eval@4096 (count not stated). JoLT 2608.15395: FLUX.2 Klein, own 200 prompts, no FID.
- AReA Tables 1/2/6 transcriptions verified programmatically vs PDF text (24 + 5 + 6 rows, 0 mismatches) -> results/published_compare/published.csv.
- hold_eval: z910-z950 moved to hold_eval/superseded_191/; README.md written (release order, GPU-h, CPU finalize, build.py).
- TODO: survey note (.agent/notes/eval_protocol_survey.md), build.py + compare.csv + README + paper/figures/snippets/published_compare.tex
  (no \scarow macro: use \rowcolor[HTML]{EAF4F1}), final report. Waiting on agents A (SEGA/SigMa/DyPE/TIDE) and C (Diffusion-4K/HiFlow/I-Max).
- 05:55 CT DONE: survey .agent/notes/eval_protocol_survey.md (+ sources/); results/published_compare/{published,ours,compare}.csv,
  build.py, README.md; paper/figures/snippets/published_compare.tex (compiles with library/texlive pdflatex, rendered OK).
  dev4 HRDiT+SigMa generated (gate 195/195). Ready to release now: z960, z961, z964, z965. TIDE: all 195 + DrawBench 200.
