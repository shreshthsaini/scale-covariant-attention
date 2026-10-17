"""Reproducible phase-2 galleries with frozen selection and source provenance.

Run with the existing sca uv environment on a compute node:
  python paper/figures/src/fig_phase2.py all

The first run freezes selection from complete paired outputs. Later runs use that
manifest and verify source hashes. No metric or visual inspection enters selection.
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import imgsel
from fig_qual import COLORS, auto_box, header, make_tiles, row, snippet

SPECS = {
    "baselines": {
        "res": "4096x4096", "n": 3,
        "methods": ["area", "dype", "sega_official", "ultraimage", "sca"],
        "caption": (
            r"\textbf{Additional baselines at $4096^2$.} "
            r"Three randomly sampled paired prompts from the completed evaluation set, "
            r"with no selection by visual quality or metric score. Each prompt occupies "
            r"two rows: full images, then native-pixel crops of the outlined windows. "
            r"FLUX.1-dev, seed 0. AReA, DyPE, SEGA and SCA use 28 steps and guidance 4.5; "
            r"UltraImage uses its two-stage defaults (30+30 steps, guidance 3.5 then 7). "
            r"The gallery illustrates outputs under these protocols, not equal-cost comparisons."
        ),
    },
    "flux2": {
        "res": "4096x4096", "n": 4,
        "methods": ["flux2_full", "flux2_sca"],
        "caption": (
            r"\textbf{FLUX.2-dev at $4096^2$.} Four randomly sampled paired prompts "
            r"from the completed 50-prompt evaluation, without quality-based selection. "
            r"For each method, a full image is followed by its native-pixel crop. "
            r"Seed 0, 28 steps, guidance 4.0, with the FLUX.2 calibration table for SCA. "
            r"SCA can still distort face geometry and fine texture, as the last row shows."
        ),
    },
    "6144": {"res": "6144x6144", "n": 4, "methods": ["area", "sca"]},
    "8192": {"res": "8192x8192", "n": 4, "methods": ["area", "sca"]},
    "8192_complete": {
        "res": "8192x8192", "n": 4, "methods": ["area", "sca"], "expected": 12,
        "caption": (
            r"\textbf{Exploratory $8192^2$ generation.} Four prompts sampled uniformly "
            r"from all 12 paired evaluation prompts, without selection by visual quality "
            r"or metric score. Each full image is followed by its native-pixel crop. "
            r"FLUX.1-dev, seed 0, 28 steps, guidance 4.5. Grid and stripe artifacts "
            r"remain at this scale; this small evaluation does not establish general 8K quality."
        ),
    },
}

NAMES = {"area": "AReA", "dype": "DyPE", "sega_official": "SEGA",
         "ultraimage": "UltraImage", "sca": r"\textbf{SCA (ours)}",
         "flux2_full": "Full attention", "flux2_sca": r"\textbf{SCA (ours)}"}
for method in ("dype", "sega_official", "ultraimage"):
    imgsel.ROOTS[method] = imgsel.OUT + "/main/" + method + "/{res}"
for method in ("flux2_full", "flux2_sca"):
    imgsel.ROOTS[method] = imgsel.OUT + "/flux2/" + method + "/{res}"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete(method, res):
    """Only complete, matching-resolution, seed-zero images with sidecars."""
    valid = {}
    for pid, png in imgsel.index(method, res).items():
        sidecar = Path(png).with_suffix(".json")
        if not sidecar.exists():
            continue
        meta = json.loads(sidecar.read_text())
        config = meta.get("config", meta)
        if not meta.get("complete", False) or config.get("seed") != 0:
            continue
        with Image.open(png) as image:
            if image.size != tuple(reversed([int(v) for v in res.split("x")])):
                continue
        valid[pid] = png
    return valid


def build(tag):
    spec = SPECS[tag]
    name = "gallery_" + tag
    out = Path(imgsel.FIGDIR) / name
    out.mkdir(exist_ok=True)
    manifest_path = out / "provenance.json"
    methods, res = spec["methods"], spec["res"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["prompts"]:
            for source in entry["sources"].values():
                if sha256(source["png"]) != source["sha256"]:
                    raise ValueError(f"Changed source image: {source['png']}")
    else:
        indexes = {m: complete(m, res) for m in methods}
        common = set.intersection(*(set(v) for v in indexes.values()))
        dev = imgsel.dev4_ids()
        eligible = sorted(pid for pid in common if imgsel.short_id(pid) not in dev)
        if "expected" in spec and len(eligible) != spec["expected"]:
            raise ValueError(f"{tag}: expected {spec['expected']} complete pairs, found {len(eligible)}")
        selected = sorted(random.Random(17).sample(eligible, min(spec["n"], len(eligible))))
        if not selected:
            raise ValueError(f"No completed pairs for {tag}")
        manifest = {
            "schema": 1, "figure": name, "resolution": res,
            "snapshot_utc": datetime.now(timezone.utc).isoformat(),
            "selection": "random.Random(17).sample(sorted(eligible_ids), n); dev4 excluded",
            "eligible_ids": eligible, "eligible_count": len(eligible),
            "methods": methods, "generation_seed": 0,
            "source_counts": {m: len(v) for m, v in indexes.items()},
            "crop_rule": "Maximum average edge energy across every method in this figure",
            "prompts": [],
        }
        for pid in selected:
            images = [imgsel.load(indexes[m][pid]) for m in methods]
            crop = auto_box(images, 512 * int(res.split("x")[0]) // 4096)
            for image in images:
                image.close()
            entry = {"id": pid, "short_id": imgsel.short_id(pid), "crop_xy_size": crop, "sources": {}}
            prompt_values = set()
            for method in methods:
                png = indexes[method][pid]
                meta = json.loads(Path(png).with_suffix(".json").read_text())
                config = meta.get("config", meta)
                prompt_values.add(config["prompt"])
                entry["sources"][method] = {"png": png, "sha256": sha256(png), "config": config}
            if len(prompt_values) != 1:
                raise ValueError(f"Prompt mismatch: {pid}")
            entry["prompt"] = prompt_values.pop()
            manifest["prompts"].append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    lines = []
    for i, entry in enumerate(manifest["prompts"]):
        sid, box = entry["short_id"], tuple(entry["crop_xy_size"])
        tiles = {}
        for method in methods:
            _, tiles[method] = make_tiles(dict(out=str(out), name=f"{sid}_{method}",
                label=method, res=res, sid=sid, side=800, boxes=[(box, COLORS[0])],
                crops=[(box, COLORS[0], "crop")]))
        if len(methods) > 2:
            full, height = row([(1, tiles[m]["full"]) for m in methods])
            crops, _ = row([(1, tiles[m]["crop"]) for m in methods])
            if i == 0:
                lines.append(header([(1, NAMES[m]) for m in methods], height))
            lines.extend([full, crops])
        else:
            items = [(1, tiles[m][kind]) for m in methods for kind in ("full", "crop")]
            line, height = row(items)
            if i == 0:
                lines.append(header([(2 + 0.004 / height, NAMES[m]) for m in methods], height))
            lines.append(line)
    caption = spec.get("caption")
    if caption is None:
        caption = (
            rf"\textbf{{Partial qualitative results at ${res.split('x')[0]}^2$.}} "
            rf"{len(manifest['prompts'])} prompts sampled uniformly from the "
            rf"{manifest['eligible_count']} non-development prompts with completed AReA and SCA outputs "
            r"at the figure snapshot; this is not the final evaluation set. "
            r"Each full image is followed by its native-pixel crop. FLUX.1-dev, "
            r"seed 0, 28 steps, guidance 4.5. The subset was not selected by image quality; "
            r"completion order may differ from the full benchmark."
        )
        if tag == "8192":
            caption += r" Visible grid and stripe artifacts remain in both methods at this scale."
    caption += (r" Crop locations maximize edge energy averaged across the methods shown. "
                r"Prompt selection uses seed 17; source hashes and crop coordinates are archived with the figure.")
    comments = [f"Generated by fig_phase2.py {tag}; frozen selection in {name}/provenance.json."]
    comments += [f"{e['short_id']}: {e['prompt'].replace(chr(0x2014), ' - ')}" for e in manifest["prompts"]]
    snippet(name, lines, caption, "fig:" + name, comments=comments)
    snippet_path = Path(imgsel.FIGDIR) / "snippets" / f"{name}.tex"
    generated = snippet_path.read_text().splitlines()
    generated[0] = f"% Generated by paper/figures/src/fig_phase2.py {tag}; do not edit by hand."
    snippet_path.write_text("\n".join(generated) + "\n")
    print(name, manifest["eligible_count"], "eligible; selected", len(manifest["prompts"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=[*SPECS, "all"])
    args = parser.parse_args()
    for name in SPECS if args.figure == "all" else [args.figure]:
        build(name)
