#!/bin/bash
# Release held evaluation tasks when their generation group has drained, then run the CPU bootstrap
# finalizers when an eval writes eval/<group>/DONE. Loops every 5 min until taskq/STOP exists.
#   nohup bash slurm/release_evals.sh > /scratch/09032/saini_2/sca/logs/release_evals.log 2>&1 &
S=/scratch/09032/saini_2/sca; Q=$S/taskq; H=$Q/hold_eval; F=$Q/hold_cpu_finalize
drained(){ [ -z "$(ls $Q/pending $Q/running 2>/dev/null | grep -E "$1")" ]; }
rel(){ [ -f $H/$1 ] && drained "$2" && mv $H/$1 $Q/pending/a000_$1 && echo "$(date +%FT%T) released $1"; }
while [ ! -f $Q/STOP ]; do
  rel z962_eval_a4k195_seed1.sh '^a19[01]_[67]_'
  rel z963_eval_a4k195_seed2.sh '^a19[01]_[89]_'
  rel z966_eval_krea195.sh '^a192_'
  rel z967_eval_hrgen1k_krea.sh '^a193_'
  rel z968_eval_hrgen1k_dev.sh '^a194_'
  for d in $S/eval/*/DONE; do g=$(basename $(dirname $d)); b=$F/${g}_bootstrap.sh
    [ -f "$b" ] && [ ! -f $S/eval/$g/BOOTSTRAP_DONE ] && { echo "$(date +%FT%T) bootstrap $g"; bash $b > $S/logs/bootstrap_$g.log 2>&1 && touch $S/eval/$g/BOOTSTRAP_DONE; }
  done
  sleep 300
done
