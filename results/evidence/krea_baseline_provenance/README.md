# Reused Krea baseline provenance

This archive records the evidence connecting the inherited DyPE and SEGA outputs
to the Krea transformer. It covers the 100 selected E9 prompts for each method.
The original outputs and sidecars have not been changed.

All 56 completed generation scripts export `CROPE_TRANSFORMER` to
`black-forest-labs/FLUX.1-Krea-dev`, snapshot
`8162a9c7b05a641be098422bf2fcf335615c2f28`. Their prompt ranges cover all 100
selected IDs, and each has a matching successful completion event. The scripts
use seed 0, 28 steps, guidance 4.5, and resolution 4096 by 4096. The archived loader
at every declared code revision reads that environment variable and supplies the
selected transformer to the FLUX.1-dev pipeline. These loader functions have
identical source hashes. The 200 selected sidecars also record one shared sigma
schedule and the requested DyPE or SEGA RoPE method.

The sidecars themselves label the base pipeline as FLUX.1-dev and do not record
the transformer override. The Krea attribution therefore relies on launch scripts
and loader code, not that model field alone.

The inherited campaign used a changing worktree. Its selected outputs span four
declared commits and five implementation hashes. The ten-file implementation hash
is reproduced exactly for 193 of 200 selected sidecars. Seven SEGA sidecars have
an unmatched hash, `4b7e9f42ee011ab831ea9171209b6f6d1754f17d8481406e99df9ea8ea7be24a`.
Some sidecars declaring an earlier commit match the exact bytes of a later
archived commit. This is not evidence of one frozen baseline implementation.
Algorithm-level differences and the unresolved source subset require separate
assessment before making a stronger reproducibility claim.

`receipt.json` lists source paths, hashes, selected coverage, revision strata,
and the unmatched count. `tasks/` contains exact completed generation scripts.
`runtime/` contains the ten fingerprinted files at commit `7405462`; the other
declared revisions are under `runtime_variants/`. `matched_log_events.jsonl`
preserves relevant worker events and saved-image lines. The receipt includes
hashes of their complete original log files. `selected_sidecars.jsonl.gz`
preserves the 200 complete sidecar records and each original file hash.
`source_generation_prompts.jsonl` preserves the manifest indexed by the tasks;
the selected E9 manifest is already archived at `../prompts/eval100.jsonl`.

No model weights were loaded and no PNG bytes were rehashed in this audit.
Sidecar PNG hashes are retained as original declarations. The evaluation-promotion
helper separately verifies PNG hashes before allowing the final evaluation.
