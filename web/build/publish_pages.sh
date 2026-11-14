#!/bin/bash
# Publish web/ from origin/main to the gh-pages branch (GitHub Pages, public site:
# https://shreshthsaini.github.io/scale-covariant-attention/). Run after every web/ change is pushed to main.
# web/build (scripts, manifests with scratch paths), web/README.md (working notes) and the paper owner's PNG and SVG
# figure exports (the page shows their WebP) and assets/img/showcase (build_showcase.py's exports; the page renders its
# own zoom files from the PNGs) stay out of the public site,
# and the publish stops if any published text file names an internal path or uses a word the page must not use.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd); T=$(mktemp -d /scratch/09032/saini_2/tmp/ghpages.XXXX)
cd "$R" && git fetch -q origin && git archive origin/main web ':(exclude)web/build' ':(exclude)web/README.md' ':(exclude)web/assets/img/figs/*.png' ':(exclude)web/assets/img/figs/*.svg' ':(exclude)web/assets/img/showcase' ':(exclude)web/assets/img/ideogram4' ':(exclude)web/assets/js/ideogram4.js' ':(exclude)web/dev' | tar -x -C "$T"
cd "$T/web" && cp "$R/paper/main.pdf" paper.pdf && sed -i 's#\.\./paper/main\.pdf#paper.pdf#g' ./*.html && sed -i 's#\.\./\.\./paper/main\.pdf#../paper.pdf#g' review/index.html && touch .nojekyll
if grep -rIl --include='*.html' --include='*.js' --include='*.json' --include='*.css' --include='*.vtt' --include='*.srt' \
     -e '/scratch/' -e '/work/09032' -e '/home1/' . ; then
  echo "publish_pages.sh: internal paths in the files above; not published" >&2; exit 1
fi
if grep -rIil --include='*.html' --include='*.vtt' --include='*.srt' -e 'calibrat' -e 'uncurated' -e '\bcurated\b' . ; then
  echo "publish_pages.sh: forbidden wording in the files above; not published" >&2; exit 1
fi
git init -q -b gh-pages && git add -A && git -c user.name=shreshthsaini -c user.email=shreshthfeel@gmail.com commit -q -m "Project page $(date +%F)"
git remote add origin git@github.com:shreshthsaini/scale-covariant-attention.git && git push -q -f origin gh-pages
echo "published $(date +%FT%T)"

# Deploy through the workflow (60 min timeout) rather than the legacy branch build.
gh workflow run pages.yml --repo shreshthsaini/scale-covariant-attention --ref main && echo "deploy workflow started"
