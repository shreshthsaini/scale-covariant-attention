#!/usr/bin/env python3
"""Fill the "backbones" list of web/build/showcase_page.json (the page's Other backbones section) from the showcase
manifest: for FLUX.2-dev and Qwen-Image, every prompt whose SCA image and full-attention image at the same size both
exist (complete pairs only), the first three per group in the order of showcase_prompts.jsonl, largest sizes first
(qwen_5312, flux2_5120, then qwen_4096, flux2_4096). Only that line of the file changes. Then run
build_showcase_page.py. Run with the sca env python from the repository root."""
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent
GROUPS = (("qwen_5312", "qwen_sca", "qwen_full"), ("flux2_5120", "flux2_sca_v2", "flux2_full"),
          ("qwen_4096", "qwen_sca", "qwen_full"), ("flux2_4096", "flux2_sca_v2", "flux2_full"))


def main(per_group=3):
    man = json.loads((R / "showcase_manifest.json").read_text())
    order = [json.loads(l)["id"] for l in (R / "showcase_prompts.jsonl").read_text().splitlines() if l.strip()]
    have = {}
    for e in man["images"]:
        have.setdefault((e["group"], e["prompt_id"]), set()).add(e["method"])
    pairs = []
    for g, sca, full in GROUPS:
        ids = [p for p in order if {sca, full} <= have.get((g, p), set())]
        pairs += [{"group": g, "prompt_id": p} for p in ids[:per_group]]
    path = R / "showcase_page.json"
    text = path.read_text()
    block = ' "backbones": [' + (",".join("\n  " + json.dumps(x) for x in pairs) + "\n ]" if pairs else "]")
    new = re.sub(r' "backbones": \[[^\]]*\]', block, text)
    json.loads(new)
    path.write_text(new)
    print(f"{len(pairs)} backbone pairs:", ", ".join(f"{x['group']}/{x['prompt_id']}" for x in pairs))


if __name__ == "__main__":
    main()
