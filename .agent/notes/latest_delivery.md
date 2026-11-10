# Latest verified paper delivery, 2026-11-10 04:00 CST (v5.1: v5 plus the progressive-schedule ablation)

- [Paper PDF](../../paper/main.pdf), 55 pages: main text pages 1 to 9, reproducibility statement and references from
  page 10, appendix after. SHA256 8af9545fcace987727fc2c0a78aa07d6f7dc1396c83bca1d9b7c907a14c35a90.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 429 source files, SHA256
  260d4287aba5968ffd3c1ecef806476b818b250bbb6f03aa1bc338ca6111ac9e.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_5revguaf/`.

Both builds: zero overfull boxes, zero undefined references, and the fresh unzip builds on its own to the same 55 pages.
PDF text has no calib*, curat*, "drawn at random", "random draw", "random quer*", "fixed random", em dash, section
symbol or "??"; no file name in the ZIP contains calib or curat.

What changed against v5 (HANDOVER Section 42): switch \resprogabl on. Table 2 has the progressive group (shared,
per-layer, window only against per-head weights, 195 prompts), Section 4.2 two generated sentences, appendix Table L.3
all 14 metrics with intervals (set-level ones from results/prog195_ablation/cfid_ci). Switch \resra stays off.

Previous deliveries: `delivery_gy4l47kq` (v5, HANDOVER Section 41, 54 pp), `delivery_gvyduue4` (v4, Section 37).
