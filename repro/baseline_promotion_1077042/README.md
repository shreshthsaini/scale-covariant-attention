# Bounded baseline promotion

This helper fills the missing12-to46 generation preparation path. It also sizes
12-prompt tasks to the available fleet window. It creates candidates only; it
does not submit jobs, enqueue tasks, poll Slurm, or change frozen generation code.
The existing two smoke tasks remain the prerequisites. No new GPU result is
established by these CPU tests.

Use the managed campaign Python on a compute allocation:

```bash
SCA_PROMOTION_PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
uv run --no-project --python "$SCA_PROMOTION_PY" python \
  repro/baseline_promotion_1077042/prepare.py --stage pilot12 \
  --review-template /scratch/09032/saini_2/sca/baseline_review12_1077042.json
```

This refuses until both smoke outputs are complete. The helper verifies every
frozen deployment file, exact runtime/source/model/prompt resume identity,
PNG checksum,4096² dimensions, PNG CRCs, full decoding, recorded finite checks,
and consistent positive pipeline timing. A partial output is preserved and
refuses promotion. It neither repairs nor replaces output files.

Inspect the actual smoke images and source/profile settings. Fill the template
with the reviewing operator's name, per-image fidelity/validity observations and
notes. The template starts with false review flags. It binds each reviewed image
and sidecar checksum. Review requires valid implementation and outputs; no
quality score or advantage over SCA is required. Retain unfavorable valid images.
The main authorized operator can perform this review without additional user
approval. A synthetic test image never supplies campaign review evidence.

Then prepare candidates into a new scratch directory:

```bash
uv run --no-project --python "$SCA_PROMOTION_PY" python \
  repro/baseline_promotion_1077042/prepare.py --stage pilot12 \
  --review /scratch/09032/saini_2/sca/baseline_review12_1077042.json \
  --tasks-dir /scratch/09032/saini_2/sca/baseline_candidates12_1077042 \
  --max-task-seconds 6000 --startup-seconds 600
```

The admission budget excludes the frozen dispatcher's120-second cleanup margin.
6000+120seconds fits a fresh2-hour fleet; it may not fit a nearly expired fleet.
For a suitable4-hour fleet, an explicitly reviewed13200-second admission budget
leaves time for cleanup and launch delay. The helper refuses if even one image
cannot fit. Use an appropriately sized fleet rather than ignoring the task.

Each task reserves twice the maximum observed per-image pipeline duration plus
the explicit startup allowance. This allowance is a planning assumption because
the generation receipt excludes model loading. Contiguous missing prompts are
grouped within that budget. Existing complete images are validated and excluded;
partial images cause refusal. No completed image is regenerated. Tasks use the
original frozen runner,46-prompt manifest, output namespace and WandB tracking.

After both methods have valid outputs for every first12 prompt, repeat the same
two commands with `--stage pilot46`, new review/candidate paths and an appropriate
admission budget. All24 prerequisite outputs need hash-bound fidelity/validity
review before46 preparation. This produces only missing prompt indices up to46.
The existing exact12/46 evaluation helper remains compatible with these outputs.

`receipt.json` records each candidate checksum, review checksum, measured sample
count, central incremental pipeline cost, planning range and task reservation
hours. The range is1times mean to2times maximum observed pipeline time; it is
not a confidence interval. Add evaluation, actual loading, rerun reserve, charge
rates and existing reservations when reviewing SU headroom. Before any new
Slurm submission check the live balance on login2, keeping at least1600SUs.
This helper never performs a submission.

Immediately before publishing candidates, inspect live tasks and ownership in
pending/running/done/failed. The helper does not reserve work. Do not duplicate
running tasks or replace dispatcher timeout retries. Keep candidates outside the
spool until this review. A191-prompt campaign still requires a separate frozen
manifest and provenance audit; no191 generation or scale-up is claimed here.

Tests use temporary synthetic outputs and the unchanged frozen source package:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
uv run --no-project --python "$SCA_PROMOTION_PY" python -m pytest -q \
  repro/baseline_promotion_1077042/test_prepare.py
```
