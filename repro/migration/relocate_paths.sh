#!/usr/bin/env bash
# Rewrite the TACC Vista absolute paths baked into scripts, task files and prompt lists for a new machine.
# Usage: SCA_SCRATCH=/data/sca_root SCA_ENVS=/opt/uv-envs SCA_REPO=$PWD [CROPE_REPO=...] \
#        bash repro/migration/relocate_paths.sh [--apply] DIR [DIR...]
# DIR defaults to the git checkout (only tracked text files are touched there); any other DIR (for example an
# extracted taskq/ or prompts/) is scanned for text files. Without --apply it only lists the files it would change.
# Prefix map, longest first:
#   /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention -> $SCA_REPO
#   /work/09032/saini_2/vista/Projects/Exploration/constancy_rope            -> $CROPE_REPO (default $SCA_SCRATCH/constancy_rope_repo)
#   /work/09032/saini_2/vista/library/uv-envs                                -> $SCA_ENVS
#   /work/09032/saini_2/vista/library/uv-cache                               -> $SCA_SCRATCH/uv-cache
#   /scratch/09032/saini_2                                                   -> $SCA_SCRATCH
set -euo pipefail
APPLY=0; [[ ${1:-} == --apply ]] && { APPLY=1; shift; }
: "${SCA_SCRATCH:?set SCA_SCRATCH}" "${SCA_ENVS:?set SCA_ENVS}" "${SCA_REPO:?set SCA_REPO}"
CROPE_REPO=${CROPE_REPO:-$SCA_SCRATCH/constancy_rope_repo}
SED=(-e "s#/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention#${SCA_REPO}#g"
     -e "s#/work/09032/saini_2/vista/Projects/Exploration/constancy_rope#${CROPE_REPO}#g"
     -e "s#/work/09032/saini_2/vista/library/uv-envs#${SCA_ENVS}#g"
     -e "s#/work/09032/saini_2/vista/library/uv-cache#${SCA_SCRATCH}/uv-cache#g"
     -e "s#/scratch/09032/saini_2#${SCA_SCRATCH}#g")
PAT='/scratch/09032/saini_2|/work/09032/saini_2/vista'
[[ $# -eq 0 ]] && set -- "$SCA_REPO"
for dir in "$@"; do
  if git -C "$dir" rev-parse --show-toplevel >/dev/null 2>&1 && [[ $(git -C "$dir" rev-parse --show-toplevel) == $(cd "$dir" && pwd) ]]; then
    mapfile -t files < <(git -C "$dir" grep -I -l -E "$PAT" -- . ':!*.pdf' ':!*.zip' | sed "s#^#$dir/#")
  else
    mapfile -t files < <(grep -r -I -l -E "$PAT" "$dir" || true)
  fi
  echo "$dir: ${#files[@]} files reference Vista paths"
  if (( APPLY )); then
    for f in "${files[@]}"; do [[ -L $f ]] || sed -i "${SED[@]}" "$f"; done
  else
    printf '  %s\n' "${files[@]:0:20}"
  fi
done
