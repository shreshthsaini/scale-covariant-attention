#!/usr/bin/env python3
"""Fill the "backbones" and "qwen_area" lists of web/build/showcase_page.json from the showcase manifest.

backbones (the page's Other backbones cards): FLUX.2-dev, every prompt whose SCA image and full-attention image at the
same size both exist (complete pairs only), the first three per group in the order of showcase_prompts.jsonl, largest
size first (flux2_5120, then flux2_4096).
qwen_area (the Qwen-Image block under it, HANDOVER Section 38): Qwen-Image at 5312^2, every prompt of the 15-prompt
set (the first three prompts of each of the five prompt groups, the rule of slurm/area_qwen_tasks.py) whose SCA image
and AReA image both exist, in that set's order; full attention joins a grid wherever it exists.
Only those two entries of the file change. Then run build_showcase_page.py. Run with the sca env python from the
repository root."""
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent
GROUPS = (("flux2_5120", "flux2_sca_v2", "flux2_full"), ("flux2_4096", "flux2_sca_v2", "flux2_full"))
QA_GROUP, QA_PREFIXES = "qwen_5312", ("sc_p", "sc_l", "sc_h", "sc_a", "sc_d")


def qwen_area_ids(order):
    return [p for g in QA_PREFIXES for p in [x for x in order if x.startswith(g)][:3]]


def put(text, key, items):
    block = f' "{key}": [' + (",".join("\n  " + json.dumps(x) for x in items) + "\n ]" if items else "]")
    if re.search(rf' "{key}": \[[^\]]*\]', text):
        return re.sub(rf' "{key}": \[[^\]]*\]', block, text)
    return re.sub(r"\n}\s*$", ",\n" + block + "\n}\n", text)


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
    qa = [{"group": QA_GROUP, "prompt_id": p} for p in qwen_area_ids(order)
          if {"qwen_sca", "qwen_area"} <= have.get((QA_GROUP, p), set())]
    path = R / "showcase_page.json"
    new = put(put(path.read_text(), "backbones", pairs), "qwen_area", qa)
    json.loads(new)
    path.write_text(new)
    print(f"{len(pairs)} backbone pairs:", ", ".join(f"{x['group']}/{x['prompt_id']}" for x in pairs))
    print(f"{len(qa)} qwen_area prompts:", ", ".join(x["prompt_id"] for x in qa))


if __name__ == "__main__":
    main()
