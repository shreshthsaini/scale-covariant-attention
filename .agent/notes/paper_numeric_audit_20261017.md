# Paper numerical audit, October 17, 2026

Completed on the CPU continuation allocation, without GPU work, SSH, task changes,
controller changes, new results, or manuscript result-table edits. Current handover
corrections were treated as authoritative. The structured receipt is
`results/evidence/paper_numeric_audit_20261017.json`.

The existing numerical evidence agrees with the manuscript. The project remains
incomplete because E2 6K, E9 Krea, three latency groups, fresh GPU validation,
and final delivery reconciliation remain outstanding. Metric-environment bounded
CPU parity subsequently passed; GPU metric parity remains untested.
Root rebuilt the PDF and source ZIP after both prose corrections. All37 primary and
independent-source pages render identically; only page27 changes from the prior delivery.
See results/evidence/delivery_validation_1076947.json.

## Checks completed

- Rehashed 195 archived files against the current evidence manifest. No mismatch.
- Independently rederived 762 scalar summary means from archived per-image rows.
  No mismatch; all method groups have matching prompt IDs and no duplicate rows.
- Refit all 13 scale-specific per-head calibration tables from seven archived
  sufficient-statistic bundles. Fitted weights and reported errors match within
  1e-12. These checks do not rerun the model or collect new calibration observations.
- Regenerated all six generated appendix tables from existing CSVs and corrected
  distribution checks. Their complete source is identical to the embedded block
  in `paper/main.tex`; 82 table rows were checked.
- Independently checked 143 populated main-table cells at lines 258 to 272 against
  source means, corrected patch distributions, and the appropriate clean or
  campaign timing source. Every cell agrees after its declared display rounding.
- Checked 450 numeric values in the additional-baseline, generality, sensitivity,
  and sampling-budget tables at lines 780 to 891 against the separately saved
  2,000-draw paired CSVs. All deltas and endpoints agree after display rounding.
  Those existing bootstrap runs were not rerun.
- Regenerated the complete 12-prompt 8K table using `gen_final_tables.py`, including
  its 2,000-draw paired prompt bootstrap. Its bytes match the existing snippet.
- Rechecked all 11 populated clean-latency groups directly against archived logs
  and sidecars. Each has an excluded generated warm-up followed by three ordered
  timed images, with matching settings, log hashes and medians. The logs contain no
  skipped/resumed image within those groups. Saved logs alone do not independently
  establish that the rest of each node was idle.
- Root subsequently rebuilt `paper/main.pdf` and `paper/sca_overleaf.zip` after
  both prose edits. Their456 source hashes and37 rendered pages match the
  independent extracted build. Page27 was inspected at144dpi. Current proof is
  `results/evidence/delivery_validation_1076947.json`.

## Applied fixes

1. `paper/main.tex:982` claimed lower energy per image than full attention, although
   the archived measurements establish generation time and occupied worker time.
   Replaced this with a $4096^2$ generation-time reduction under the measured
   protocol and the sentence `Energy use remains unmeasured.` No result changed.
2. `results/evidence/README.md:22` still reported 740 scalar means. After rerunning
   `scripts/audit_evidence.py`, refreshed this to 762 and stated the current 195
   verified archive hashes. The authoritative refreshed report is
   `results/evidence/audit.json`. The extra 22 scalar means are the completed
   12-prompt 8K evaluation, including its two campaign-time summaries.

## Interpretation boundaries confirmed

- Reference correction: `paper/main.tex:598` and 600 retain 195 official row
  weights, 168 reference-image identities, corrected per-row patch features,
  and image-cluster uncertainty for the corrected main comparison. Main SCA
  patch FID is 99.8156146855 versus AReA 103.4110620759; SCA-minus-AReA is
  -3.5954473904 with interval [-7.3886775756, 1.1524725330]. Patch KID also
  remains unresolved, with delta -0.002270261 and interval spanning zero.
  The whole-image FID gain remains resolved. Historical intervals and corrected
  point estimates are not combined silently in the generated appendix.
- Native range: `paper/main.tex:108`, 165, and 526 distinguish exact mathematical
  coordinates from the frozen FLUX.1 bf16 implementation. The 6K and 8K rounded
  endpoints can give coarse coordinate 64, and a local-window offset of 64 rather
  than the exact bound 63. No new fp32-coordinate result is claimed.
- GPU numerical check: `paper/main.tex:524` reports the observed 0.015625 maximum
  absolute difference and explicitly declines to claim a per-element ULP bound.
- Novelty: `paper/main.tex:388` attributes head roles, two branches, global opening
  phases, spectral fusion, and head-specific scopes to prior work. The claimed
  contribution is native emulation and fitted/transferred per-head/noise mixing.
  No new primary-source literature verification was performed in this bounded
  offline audit; this checks consistency with the established project boundary.
- Calibration cost: `paper/main.tex:584` and 975 say the complete runtime was not
  retained. The unsupported historical 1.5-GPU-hour claim is absent from the paper.
- Exploratory 8K: `paper/main.tex:897` and 899 preserve 12 paired prompts, no
  distributional metrics, remaining stripe/grid artifacts, CPU evaluation,
  float32 HPS versus main mixed precision, and native MUSIQ query chunks of 256
  retaining all keys/patches. Six intervals favor SCA; aesthetic, MUSIQ, crop MUSIQ,
  and CLIP-IQA remain unresolved. The latter three means favor AReA.
- Clean timing: `paper/main.tex:303` and 553 distinguish current medians of three
  after one same-process warm-up from historical repeats and campaign means.
  Full/SCA at 4K is 2.6249; AReA/SCA is 2.8486. The reported 2.6 and 2.85 agree.
  SCA 8K/2K runtime is 15.8031 versus 16 times the pixel count.
- Compute total: `paper/main.tex:975` remains a historical lower bound, explicitly
  cut off at 06:59 UTC, not a complete campaign total or GPU kernel activity.

## Required next paper edits when real results arrive

1. E2 6K must fill the existing main-table rows at 274 to 276, protocol at 233 to
   235, interpretation at 285, limitations at 395, appendix marker at 892, and
   gallery insertion at 961. Use exact complete coverage and 1,000 reference-image
   cluster draws before archiving or rendering final tables.
2. E3 must fill 6K AReA/full and 8K AReA at 314 to 315; update interpretation at
   346 and 553. Do not promote interrupted or resumed timing fragments.
3. E9 must update the main generality table at 341, interpretation at 383,
   limitations at 395, common-sampler provenance paragraph at 770, appendix at
   892, and AReA protocol at 948. Add real Krea means, paired differences,
   corrected distributions and complete gallery with the repaired SEGA overlay.
4. Root qualified `paper/main.tex:977` to distinguish task-declared frozen SCA code
   from incomplete inherited Krea baseline revision evidence. When E9 is added, describe the
   recovered revisions, common relevant functions, seven regenerated SEGA images,
   and immutable repair overlay. The archived provenance README explicitly says
   that the inherited outputs are not evidence of one frozen baseline revision.
5. Retain the reference correction, bf16 caveat, untested HRDiT/ResDiT boundary,
   UltraImage protocol/budget differences, 8K negative examples, two-seed subset
   limits, and unadjusted multiplicity limits in the final paper.
6. Rebuild delivery, independently build extracted sources, inspect warnings and
   appearance, rehash artifacts, and refresh delivery/report links after all edits.

## Reproduction and owned paths

Scratch scripts and generated comparison files are preserved under
`/scratch/09032/saini_2/sca/paper_numeric_audit_20261017/`. Their hashes are recorded
in the structured report. All scripts used the existing managed `sca` interpreter
with OMP, OpenBLAS and MKL threads bounded to four. No environment was modified.
The standalone audit script plus the main-table and phase2 scripts are sequential:
`audit.py`, `main_table.py`, then `phase2_tables.py`. The last two augment the
standalone JSON. Root archived these scripts in `repro/paper_audit_20261017/`, reran
all three after the final prose edits, and recorded their hashes in the report.
Use the managed sca interpreter with OMP/OpenBLAS/MKL threads4. Refresh current
source hashes after any later manuscript edit.

Changed paths owned by this unit:

- `paper/main.tex`
- `results/evidence/README.md`
- `results/evidence/audit.json`
- `results/evidence/paper_numeric_audit_20261017.json`
- `.agent/notes/paper_numeric_audit_20261017.md`

No commit was made by the audit worker. Root owns the final checkpoint and commit.
