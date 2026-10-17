# Completed Krea baseline overlay audit

Verified at 12:29 CDT on 17 October 2026, on CPU workspace i618-051.
`scripts/audit_krea_overlay.py` completed successfully without changing any
scratch outputs, rescoring images, importing model weights, or submitting work.

The overlay at `/scratch/09032/saini_2/sca/out/krea_baseline_overlay_20261017`
contains exactly 100 DyPE and 100 SEGA prompt pairs. The selected manifest is
byte-identical to the archived `results/evidence/prompts/eval100.jsonl`, with
100 unique IDs and no development prompts. Every prompt text, method, resolution,
seed label, derived seed, step count, guidance, dtype, batch size, RoPE method,
filename, config fingerprint, symlink target, and image/sidecar pairing passed.
All 200 outputs share one recorded sigma schedule.

All 207 distinct PNG files were SHA256 hashed, checked as 4096-square PNGs, and
verified for PNG structure and chunk CRCs. This includes the 200 selected outputs
and the seven preserved original SEGA images. All original 200 sidecar hashes
match the pre-repair archive. The selected view consists of 193 unchanged outputs
and exactly the seven intended replacements; no original was overwritten.

The seven replacement sidecars record frozen `crope_c05ebdc` implementation hash
`fe1a23e318cf1e350d64a1236d2f7481cef5d87fd744e96b29b9be93a99a5392`.
The current ten frozen source files reproduce that hash. The runtime launch
receipt, done-task script, seven-ID manifest, and Krea transformer configuration
hash agree. All seven regenerated PNGs are byte-identical to their preserved
originals. Thus this repair changes provenance without changing selected image
bytes for these seven prompts; it does not establish equivalence on other inputs.
The completed task hash is
`f16673bda44660a47fc0906fb5fbb934d607a2c74558b0e3e76bcd6d9db442df`.

Compact durable evidence is under `results/evidence/krea_overlay_complete/`:

- `receipt.json`: exact coverage, verification scope, source counts, and hashes.
- `verified_pairs.jsonl`: one record per selected pair, including resolved paths,
  current and original hashes, and replacement identity.
- `repair_sidecars.jsonl.gz`: all seven complete new sidecars.
- `repair_launch_receipt.json` and `a_krea_sega_source_repair_7.sh`: exact launch proof.

Krea attribution still relies on the archived task and loader evidence because
the inherited sidecars label the base FLUX.1-dev pipeline components rather than
the transformer override. The 193 inherited outputs span recovered implementations;
this audit does not turn them into one frozen historical revision. All 200 selected
implementation fingerprints now belong to recovered source variants, with counts
83, 63, 21, and 33 across the four known hashes. The earlier seven unmatched
fingerprints survive only in the preserved originals, not the selected overlay.

This closes the image-byte and repair-completion provenance gap for the two
inherited baselines. It does not certify completion of the separate SCA/AReA Krea
generation campaign, final E9 metrics, GPU equivalence, or model-weight byte hashes.
The historical preparation receipt remains unchanged; this new completion receipt
supersedes its prepared-only status for the seven-image repair.
