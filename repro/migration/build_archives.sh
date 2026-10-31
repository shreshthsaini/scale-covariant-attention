#!/usr/bin/env bash
# Build the planned HF archives for the given phase regex (default: every phase-1 row), at most 4 at a time.
# Usage: bash build_archives.sh <staging> [phase_regex] [repo_regex]
# Each archive goes to <staging>/<repo>/<path_in_repo>; its SHA256 (computed while writing) and byte size go to
# <staging>/sums/<archive_id>.tsv. Finished archives are skipped, so the script is resumable.
set -euo pipefail
STAGING=$1; PHASE=${2:-^1}; REPO=${3:-.}
mkdir -p "$STAGING/sums" "$STAGING/logs"
build_one() {
  local aid=$1 repo=$2 path=$3 root=$4 comp=$5 list=$6 staging=$7
  local out="$staging/$repo/$path" sums="$staging/sums/$aid.tsv"
  [[ -s $sums && -s $out ]] && return 0
  mkdir -p "$(dirname "$out")"
  local sha
  if [[ $comp == zstd ]]; then
    sha=$(tar --sort=name --format=gnu --no-recursion -cf - -C "$root" -T "$list" | zstd -T2 -3 -q | tee "$out.tmp" | sha256sum | cut -d' ' -f1)
  else
    sha=$(tar --sort=name --format=gnu --no-recursion -cf - -C "$root" -T "$list" | tee "$out.tmp" | sha256sum | cut -d' ' -f1)
  fi
  mv "$out.tmp" "$out"
  printf '%s\t%s\t%s\t%s\n' "$aid" "$path" "$sha" "$(stat -c %s "$out")" > "$sums"
}
export -f build_one
awk -F'\t' -v ph="$PHASE" -v rp="$REPO" 'NR>1 && $6 ~ ph && $2 ~ rp {print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$11}' "$STAGING/plan.tsv" |
  while IFS=$'\t' read -r aid repo path root comp list; do printf '%s\0%s\0%s\0%s\0%s\0%s\0%s\0' "$aid" "$repo" "$path" "$root" "$comp" "$list" "$STAGING"; done |
  nice -n 19 ionice -c3 xargs -0 -n 7 -P 4 bash -c 'build_one "$0" "$@" || echo "FAILED $0" >&2'
echo "built $(ls "$STAGING/sums" | wc -l) archives"
