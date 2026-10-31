# Latest verified paper delivery, 2026-10-31 14:50 CDT (HRGen-1K dev row folded in; operator sca_v2)

- [Paper PDF](../../paper/main.pdf), 53 pages: main text pages 1 to 10 (Section 6 ends on the last line of page 10),
  reproducibility statement and references from page 11, appendix after. SHA256
  b4d22b9fecb42e98970e9124ee1ae66b1109d6adc48cb18e05df896944e43f53, 27771529 bytes.
- [Overleaf source ZIP](../../paper/sca_overleaf.zip), 308 source files, SHA256
  68b264132d9b70829b409f684e34b41ef4a145aaa53d69322cfb621157a045b4.
- [Build receipt](../../paper/delivery_receipt.json); build and fresh-unzip check in
  `/scratch/09032/saini_2/sca/paper_builds/delivery_tj1oh6am/` (`primary/`, `extracted/`, `standalone/`).

Both builds have zero overfull boxes and zero undefined references; the fresh unzip builds on its own (53 pages); the
PDF text has no em dash, no section symbol and no "??"; no word box crosses a margin on any page. Every page of the
previous build was reviewed as rendered (subagent, 2026-10-31); its five findings are fixed (HANDOVER Section 22).

Numbers folded in this pass: hrgen1k_dev_sca_v2 (Table 4 HRGen-1K row, Table L.3 HRGen-1K column, Table H.2 FLUX.1-dev
SCA row, Limitations sentence on the schedule's cost). SCA minus AReA on all 1000 HRGen-1K prompts: 10 of 10 per-image
metrics significantly better, ImageReward +0.764 [0.714, 0.816]. CSVs and SHA256SUMS in results/hrgen1k_dev_sca_v2.

PENDING (grey cells with caption notes): hrgen1k_krea_sca_v2, the "SCA (ours), FLUX.1-Krea-dev" rows of Tables H.1
and H.2 (p. 26). Its eval task `taskq/pending/a000_z988_eval_hrgen1k_krea_sca_v2.sh` needs a GPU worker. To fold it
in: `bash paper/figures/src/swap_all.sh`, then `paper/build_delivery.py`, check p. 26, commit, push.
