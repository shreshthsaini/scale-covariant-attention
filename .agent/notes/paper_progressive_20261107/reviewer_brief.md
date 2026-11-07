# Render-review brief (appendix pages), progressive-SCA rewrite, 2026-11-07

You review rendered pages of the SCA paper as a strict ICLR reviewer and copy editor. Repo root:
/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention. Load the skills gold-standard-paper and
no-ai-slop first. Do not edit any file; report findings only.

## What the paper now is (binding decisions)
- Headline configuration: progressive SCA = HRDiT's three-stage schedule (1024^2 stock attention 30 steps, then
  2048^2 for 17 and 4096^2 for 10 steps) with SCA attention at the 2048^2 and 4096^2 stages. Single-pass SCA
  (final operator sca_v2, 28 steps, 77.1 s) is the fast mode. "SCA without the schedule" (sca_nb) is the first
  operator, used in some appendix results; every table, figure and sentence must say which configuration it uses.
- The paper must NOT mention or discuss repeated subjects / duplication / copies of people or animals / the
  detector / Dup. columns / McNemar anywhere, except ONE sentence in the main-text limitations ("In single-pass mode,
  small additional figures occasionally appear ..."). Any other occurrence on your pages is a MUST-fix.
- Numbers come only from results/*.csv through generators; flag any number that looks inconsistent between a
  caption, a table and the text, or between appendix and main text (Table 1 is on page 7).

## Check each page for
1. MUST: duplication content (see above); wrong or missing configuration labels; numbers that disagree; undefined
   references (??); text or figure labels below 6 pt; overfull lines; captions without a bold lead; em dashes or
   LaTeX triple dashes; the section symbol; the words license/licenses/warrant/plainly; sentences over 35 words;
   floats more than one page from their first reference; half-empty pages caused by float placement.
2. SHOULD: AI-slop patterns (binary contrasts, colon reveals, puffery, vague adjectives instead of numbers),
   unclear sentences, appendix section openings that do not say what they contain and which main-text item they
   expand, tables whose caption does not end with a key result when it is a results table.
3. NICE: layout polish.

## How
Pages are PNG renders at 110 dpi in the directory given to you (pg_pNN.png = page NN). View each assigned page.
Zoom when needed by rendering a region at higher dpi with pymupdf from the PDF given to you
(~/.local/bin/uv run --no-project --with 'pymupdf<1.26' python -I your_script.py). Cross-check numbers against the
CSVs under results/ when a number looks off (grep is fine). Never cat files over 200 lines.

## Report
A list per page: "pNN | MUST/SHOULD/NICE | location (section, table/figure, line) | problem | concrete fix".
End with a count of MUST items and the three most important fixes. Keep it under 900 words.
