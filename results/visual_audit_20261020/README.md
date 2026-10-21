# Visual audit of the SCA images in the random galleries (2026-10-20)

`random_galleries_sca.csv`: one row per distinct SCA image shown in the uncurated galleries of Appendix R
(Figures R.2 to R.6 and R.8 to R.11; the R.8 dancer and hiker are the R.2 and R.3 images and are listed once).
Each image was viewed at full resolution (overview plus 1:1 crops) by two independent auditors; one row, the R.3
astronaut, was re-checked by the lead. Columns `extra_people_or_animals` and `repeated_or_extra_objects` record
content the prompt does not ask for. Samples were drawn at random (seeds and pools in each figure's
`paper/figures/<figure>/provenance.json`), so these rows are an unbiased but small sample.
The paper's failure-case appendix and limitations quote the FLUX.1-dev 4096x4096 rows (R.2, R.3, R.8).
