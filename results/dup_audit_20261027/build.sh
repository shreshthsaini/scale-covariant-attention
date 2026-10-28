#!/bin/bash
# Rebuild every duplication-rate CSV in this directory from the detector output (CPU, about a minute).
# Detector run: scripts/dup_detect.py, raw boxes in /scratch/09032/saini_2/sca/eval/dup_detect_20261027.
set -euo pipefail
cd "$(dirname "$0")"
P=../..; PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python; S=/scratch/09032/saini_2/sca
DET=$S/eval/dup_detect_20261027; SUB=subjects_a4k195.json
M195=sca_nb,sca_nb_seed1,sca_nb_seed2,sca_band,sca_prog,sca_g7,sca_s42,sca_s56,area,area_seed1,area_seed2,full,dype,sega,ultraimage,hrdit,sigma
MABL=sca_nb,sca_band,sca_g85,sca_g93,abl_v0,abl_nogate,abl_localonly_nb,abl_globalonly,abl_uniform_nb,abl_layer_nb,abl_mean_nb,abl_noband,abl_localonly,abl_uniform,abl_layer,abl_mean,abl_route30,abl_route50
for kind in any people; do
  K=""; [ $kind = people ] && K="--kinds person,animal"
  $PY $P/scripts/dup_rate.py --dets $DET --subjects $SUB --thr 0.35 $K --methods $M195 --ref sca_nb \
    --per-image per_image_a4k195_$kind.csv --out rates_a4k195_$kind.csv > rates_a4k195_$kind.txt
  $PY $P/scripts/dup_rate.py --dets $DET --subjects $SUB --thr 0.35 $K --methods $MABL --ref sca_nb \
    --ids $S/abl50.jsonl --exclude-ids $S/dev4.jsonl --out rates_abl46_$kind.csv > rates_abl46_$kind.txt
  for col in dup dup_reviewed; do
    $PY $P/scripts/dup_rate.py --dets $DET --subjects $SUB --thr 0.35 $K --methods sca_nb --labels validation_labels.csv \
      --label-col $col | grep -E "^validation|disagree" > validation_${kind}_${col}.txt
  done
done
sha256sum $(ls *.csv *.txt *.json | sort) > SHA256SUMS
# other sizes (detector run 2026-10-28 into eval/dup_detect_20261027_sizes)
D2=$S/eval/dup_detect_20261027_sizes
for kind in any people; do
  K=""; [ $kind = people ] && K="--kinds person,animal"
  for sz in 2048x4096 2048x6144; do
    $PY $P/scripts/dup_rate.py --dets $D2 --subjects $SUB --thr 0.35 $K --methods sca_nb_$sz,area_$sz --ref sca_nb_$sz \
      --out rates_${sz}_$kind.csv > rates_${sz}_$kind.txt
  done
done
sha256sum $(ls *.csv *.txt *.json | sort) > SHA256SUMS
