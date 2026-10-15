#!/bin/bash
# enqueue.sh METHOD RES [CHUNK] [EXTRA...]: one task per prompt chunk into the sca spool.
# METHOD: area | full | sca | sca_* (sample.py variants). RES: HxW. EXTRA is passed to the sampler.
set -e
M=$1; R=$2; C=${3:-5}; shift 3 || shift $#; EXTRA="$*"
S=/scratch/09032/saini_2/sca; P=$S/code/${CODE:-041bbf1}; PSRC=/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention
PR=${PRF:-/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl}; N=$(wc -l < $PR)
LBL=${LABEL:-$M}; OUT=${OUTROOT:-$S/out/main}/$LBL/$R
if [ "$M" = area ]; then CMD="scripts/run_area.py"; else CMD="scripts/sample.py --method $M"; fi
for ((a=0; a<N; a+=C)); do b=$((a+C)); [ $b -gt $N ] && b=$N
  T=$(printf "p%03d_%s_%s.sh" $a $LBL $R)
  cat > $S/taskq/pending/$T <<EOT
#!/bin/bash
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
cd $P
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python $CMD --res $R --prompts $PR --start $a --end $b --seed 0 --guidance 4.5 --out $OUT $EXTRA
EOT
done
echo "enqueued $LBL $R -> $OUT"
