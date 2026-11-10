# Latest verified paper deliveries, 2026-11-10 14:30 (v6.2: full edition and short review edition)

Git tag `paper-v6.2`. Always build the full edition first, then the short one:
`python paper/build_delivery.py` then `python paper/build_delivery.py --main main_short.tex --stem short`.

## Full edition (67 pages)
- [PDF](../../paper/main.pdf), 23.8 MiB, SHA256 061a2cc26ec0ff21504c6f3c29d2930c8996e652d032b78384331cdee943fb38.
- [Overleaf ZIP](../../paper/sca_overleaf.zip), 23.3 MiB, SHA256
  bc01a7f646019abb4afeb87bdb383d9db756d0fb41c3fd1357bb54ecdf1d92fb.
- Receipt paper/delivery_receipt.json; build `/scratch/09032/saini_2/sca/paper_builds/delivery_9qo2y1mn/`.
- Main text pages 1 to 9 (page 9 about 95% full), reproducibility statement page 10, appendix after the references.

## Short review edition (31 pages)
- [PDF](../../paper/main_short.pdf), 16.6 MiB, SHA256 052de05e14cfbec0ec2dd0e6f5466527ba235be12db7fddc1f58186b1c0f5fb5.
- [Source ZIP](../../paper/sca_overleaf_short.zip), 16.4 MiB, SHA256
  dafa5c34b62b2f1528deba5dff847c07e583186c765efa87930c1d3463c29719.
- Receipt paper/delivery_receipt_short.json; build `/scratch/09032/saini_2/sca/paper_builds/delivery_ccu88_hn/`.
- Same source (\ifshortversion); same nine main pages except pointers; no table prints an interval.

Both: zero overfull boxes, zero undefined references, standalone ZIP builds to the same page count, no calibrat*,
curat*, "random" as a selection word, em dash, section symbol or "??". All four files are under the 30 MiB chat
upload limit.

Open before submission: AI use statement (authors), anonymous code URL.
Previous: tags `paper-v6.1` (62 / 26 pp), `paper-v6` (62 pp), `paper-v5.1` (55 pp).
