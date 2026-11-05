#!/bin/bash
# add_guard.sh <script> <canonical name> <need seconds>: insert the remaining-walltime guard after the first
# "set -uo pipefail" line and a "# GUARD_NEED=<s>" header line after the SCA_MIN_SECONDS line. Idempotent.
f=$1; B=$2; NEED=$3
grep -q '^# GUARD_NEED=' "$f" && { echo "already guarded: $f"; exit 0; }
g=$(mktemp)
cat > $g <<GUARD
# Walltime guard (review-round campaign owner, 2026-11-05): the bash workers ignore SCA_MIN_SECONDS and short fleets
# (SCA23 2 h, SCA24 3 h) share this queue, so a task that cannot finish parks itself in the hold directory (exit 3);
# the release loop puts it back once a running fleet has enough time left.
if [ -n "\${SLURM_JOB_ID:-}" ]; then
  GL=\$(squeue -h -j \$SLURM_JOB_ID -o %L 2>/dev/null | head -1)
  if [[ \$GL =~ ^[0-9:-]+\$ ]]; then
    GLEFT=\$(echo "\$GL" | awk -F'[-:]' '{n=NF; s=\$n+60*\$(n-1); if(n>=3) s+=3600*\$(n-2); if(n>=4) s+=86400*\$(n-3); print s}')
    if [ "\$GLEFT" -lt $NEED ]; then
      GH=/scratch/09032/saini_2/sca/taskq/hold_review_round
      [ -e \$GH/$B ] || { cp "\$0" \$GH/.$B.tmp.\$\$ && mv -f \$GH/.$B.tmp.\$\$ \$GH/$B; }
      echo "GUARD: job \$SLURM_JOB_ID has \${GLEFT} s left (< $NEED s); parked $B in \$GH"; exit 3
    fi
  fi
fi
GUARD
awk -v gf="$g" -v need="$NEED" '
  /^# SCA_MIN_SECONDS=/ && !h {print; print "# GUARD_NEED=" need; h=1; next}
  /^set -uo pipefail$/ && !s {print; while ((getline l < gf) > 0) print l; s=1; next}
  {print}' "$f" > "$f.guarded" && chmod --reference="$f" "$f.guarded" && mv -f "$f.guarded" "$f"
rm -f $g; bash -n "$f" && echo "guarded $f (need $NEED s)"
