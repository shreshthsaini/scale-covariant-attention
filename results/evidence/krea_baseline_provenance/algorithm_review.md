# Baseline source comparison and seven-image repair

The recovered revisions preserve the algorithm paths used by DyPE and SEGA.
Across the four archived historical revisions and frozen `crope_c05ebdc`,
`rope.py`, `sega_official.py`, and `utils.py` are byte-identical. Fourteen relevant
functions also have identical parsed syntax trees, including model loading,
sampling, transformer step hooks, the attention processor, and SDPA dispatch.
`algorithm_review.json` records these hashes and the comparison scope.

The changed code adds constancy-only methods and their calibration machinery,
registers the existing scheduler setting before recording metadata, normalizes
scheduler bookkeeping fields during resume checks, and adds an optional
`CROPE_ROPE` override. The selected baseline sidecars all record the intended
DyPE or SEGA RoPE and one identical sigma schedule. Removing the inactive
constancy branch from the gain-update function leaves identical syntax trees.
The added gate defaults to `None`; time-matched constancy is inactive for both
baselines. Frozen `crope_c05ebdc` has exactly the ten-file implementation hash
of historical revision `59205da`.

This is a static code and metadata audit. It does not establish full-model
numerical equivalence through a new GPU run, and it cannot resolve code that
was never recovered. The known frozen SCA source directories and archived
generation snapshot did not contain the seven-image unmatched hash. No broad
scratch scan was performed.

The seven affected SEGA prompt records are preserved exactly in
`unmatched_sega_prompts.jsonl`. Their recorded pipeline times sum to 1,501.34
seconds, or 25.02 minutes. A rerun using the same Krea transformer, seed 0,
28 steps, guidance 4.5, and frozen `crope_c05ebdc` is estimated at 30 GPU-minutes,
with a 28 to 40 minute range including model loading and image I/O. The estimate
assumes a comparable GH200 node and batch size one. It requires no new allocation.

The approved repair writes seven new images to an SCA-owned output directory.
An SCA-owned baseline view selects those seven outputs and the 193 unchanged
inherited outputs. It preserves all original files and their provenance. A
prepared task is subject to operator review before queueing; preparing the view
does not count as completing the rerun.
