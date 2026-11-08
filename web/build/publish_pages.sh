#!/bin/bash
# Publish web/ from origin/main to the gh-pages branch (GitHub Pages, public site:
# https://shreshthsaini.github.io/scale-covariant-attention/). Run after every web/ change is pushed to main.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd); T=$(mktemp -d /scratch/09032/saini_2/tmp/ghpages.XXXX)
cd "$R" && git fetch -q origin && git archive origin/main web | tar -x -C "$T"
cd "$T/web" && cp "$R/paper/main.pdf" paper.pdf && sed -i 's#\.\./paper/main\.pdf#paper.pdf#g' index.html && touch .nojekyll
git init -q -b gh-pages && git add -A && git -c user.name=shreshthsaini -c user.email=shreshthfeel@gmail.com commit -q -m "Project page $(date +%F)"
git remote add origin git@github.com:shreshthsaini/scale-covariant-attention.git && git push -q -f origin gh-pages
echo "published $(date +%FT%T)"
