# Calibration timing and generated-image count

The manuscript's approximate 1.5 GPU-hour calibration cost could not be re-derived.
All 48 calibration part files lack elapsed-time fields. The surviving successful
1024 run logs 32 completed forward loops whose displayed times sum to 834 seconds.
The successful 1280 run logs 16 loops totaling 322 displayed seconds, and its worker
start/end timestamps span 354 seconds including setup and reporting. These are
partial measurements, not a complete calibration cost: full 1024 task boundaries,
setup, and complete failed-attempt accounting are unavailable in the checked logs.
The initial 1280 attempt in `k1.log` fails before completing a prompt. Remove the
1.5-hour claim rather than replacing it with the sum of successful forward loops.

The image-count claim is supported after deduplication by recorded PNG SHA256:

| Archived subset | Complete sidecars | Distinct generated PNG hashes |
|---|---:|---:|
| Main comparisons, including the two wide sizes | 2,101 | 2,101 |
| Main comparisons plus ablations | 2,791 | 2,745 |
| All archived experiments | 3,797 | 3,751 |

The 46 duplicates are shared artifacts between `abl_noband` and `sca_nb`. Repeated
evaluation groups are not counted again. Counts use the already archived complete
sidecars and their recorded PNG hashes; this narrow audit does not reread image bytes.
Thus “more than 2,000 generated images” holds even for the main subset alone.

`source_manifest.json` hashes six compressed raw log/config/task sources.
`part_metadata_inventory.json` records the fields and hashes of all 48 calibration
parts without copying their arrays. `audit.json` contains the numerical derivation.
Reproduce the audit from WORK alone with the existing uv environment:

```sh
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python \
  results/compute/claims_audit/derive.py
```
