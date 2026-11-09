#!/bin/bash
# After the 000q evaluation (HANDOVER Section 38), run by 000q_z_eval.sh once results/qwen_5312_vs_area exists:
# refresh the showcase manifest and the page data (Qwen-Image block: SCA at 50 and 40 steps, AReA, full attention),
# write the Qwen-Image paper snippets (figure and \qwenResult; the \resqwen switch stays off) and run the headless
# page check, all in the live repository on CPU under nice. Nothing is committed or published: review the diff and the
# screenshots, commit web/ and paper/figures, and the lead publishes. Log: /scratch/09032/saini_2/sca/logs/area_qwen_finish.log
set -uo pipefail
R=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention
PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
PW=/work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python
LOG=/scratch/09032/saini_2/sca/logs/area_qwen_finish.log
CHECK=/scratch/09032/saini_2/sca/web_assets/check_area_qwen
cd $R || exit 1
{
  echo "$(date +%FT%T) finish start on $(hostname -s)"
  step(){ echo "+ $*"; nice -n 19 "$@" || { echo "$(date +%FT%T) step failed: $*"; return 1; }; }
  step $PY web/build/build_showcase.py && step $PY web/build/set_showcase_backbones.py \
    && step $PY web/build/build_showcase_page.py --procs 8 \
    && step $PY paper/figures/src/fig_showcase_models.py --model Qwen-Image \
    && step $PY paper/figures/src/gen_qwen_area_text.py \
    && PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright step $PW web/build/check_page.py --http --sections --out $CHECK
  echo "$(date +%FT%T) finish rc=$? (screenshots in $CHECK; nothing committed)"
} >> $LOG 2>&1
exit 0
