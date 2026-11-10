#!/bin/bash
# Qwen-Image tuning review page (HANDOVER Section 39): build web/review_qwen and run its headless check once the TEST
# evaluation is archived (marker of 000r_g_eval_test5312.sh) and every full-attention task of the TEST grids
# (000r_h_*) has finished (done marker, or moved to failed). Called at the end of 000r_g and of every 000r_h task, so
# whichever finishes last builds the page; a lock directory keeps it to one build. The chosen configuration is read
# from out/qwen_tuned/chosen.env (TUNED, TUNED_NAME, optional ALSO=label=name; written by qwen_tune_tasks.py test
# --name). CPU only, nice, live
# repository. Nothing is committed or published. Log: /scratch/09032/saini_2/sca/logs/review_qwen_finish.log
# Rebuild by hand: rmdir /scratch/09032/saini_2/sca/out/qwen_tuned/.review_built and run this script.
set -uo pipefail
R=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention
S=/scratch/09032/saini_2/sca
PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
PW=/work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python
LOG=$S/logs/review_qwen_finish.log
CHECK=$S/web_assets/check_review_qwen
CHOSEN=$S/out/qwen_tuned/chosen.env
{
  echo "$(date +%FT%T) $(hostname -s) finish hook (caller ${1:-manual})"
  [ -f $CHOSEN ] || { echo "no $CHOSEN yet"; exit 0; }
  [ -f $S/out/showcase/_done/000r_g_eval_test5312.sh ] || { echo "TEST evaluation not archived yet"; exit 0; }
  left=""
  for t in $(ls $S/taskq/pending $S/taskq/running $S/taskq/done 2>/dev/null | grep '^000r_h_' | sed -E 's/(\.sh).*/\1/' | sort -u); do
    [ -f $S/out/showcase/_done/$t ] || [ -f $S/taskq/failed/$t ] || left="$left $t"
  done
  [ -z "$left" ] || { echo "full attention still running or pending:$left"; exit 0; }
  mkdir $S/out/qwen_tuned/.review_built 2>/dev/null || { echo "already built (lock present)"; exit 0; }
  # shellcheck disable=SC1090
  ALSO=""
  . $CHOSEN
  cd $R || exit 1
  EXTRA=(); [ -n "$ALSO" ] && EXTRA=(--also "$ALSO")
  echo "+ build_review_qwen.py --tuned $TUNED --tuned-name '$TUNED_NAME' ${EXTRA[*]}"
  nice -n 19 $PY web/build/build_review_qwen.py --tuned "$TUNED" --tuned-name "$TUNED_NAME" "${EXTRA[@]}" --processes 8 \
    || { echo "$(date +%FT%T) build failed"; rmdir $S/out/qwen_tuned/.review_built; exit 0; }
  echo "+ check_review_qwen.py -> $CHECK/report.json"
  mkdir -p $CHECK
  PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright nice -n 19 timeout 900 $PW -I \
    web/build/check_review_qwen.py web/review_qwen $CHECK > $CHECK/report.json 2>&1
  echo "$(date +%FT%T) check rc=$? (report $CHECK/report.json; nothing committed)"
} >> $LOG 2>&1
exit 0
