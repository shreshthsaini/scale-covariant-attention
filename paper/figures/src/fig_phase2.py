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

REPO = Path(__file__).resolve().parents[3]
EVAL100 = REPO / "results/evidence/prompts/eval100.jsonl"
KREA_OVERLAY = Path("/scratch/09032/saini_2/sca/out/krea_baseline_overlay_20261017")

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
    "6144_complete": {
        "res": "6144x6144", "n": 4, "methods": ["area", "sca"],
        "expected": 100, "prompt_manifest": EVAL100,
        "caption": (
            r"\textbf{Paired $6144^2$ generation.} Four prompts sampled uniformly "
            r"from all 100 paired evaluation prompts, without selection by visual quality "
            r"or metric score. Each full image is followed by its native-pixel crop. "
            r"FLUX.1-dev, seed 0, 28 steps, guidance 4.5. This gallery illustrates "
            r"the shared protocol; four examples do not establish general generation quality."
        ),
    },
    "krea_complete": {
        "res": "4096x4096", "n": 2,
        "methods": ["krea_area", "krea_dype", "krea_sega", "krea_sca"],
        "expected": 100, "prompt_manifest": EVAL100,
        "provenance_receipts": [REPO / "results/evidence/krea_baseline_provenance" / name
                                for name in ("receipt.json", "repair_preparation.json", "algorithm_review.json")],
        "caption": (
            r"\textbf{FLUX.1-Krea-dev transformer at $4096^2$.} Two prompts sampled "
            r"uniformly from all 100 four-method evaluation prompts, without selection "
            r"by visual quality or metric score. Each prompt occupies two rows: "
            r"full images, then matching native-pixel crops. Seed 0, 28 steps, "
            r"guidance 4.5, with FLUX.1-dev text encoders and VAE. SCA is recalibrated "
            r"for Krea; DyPE and SEGA use the earlier Krea campaign with the recorded source repairs. This compares "
            r"the recorded implementations under the shared sampling settings, "
            r"rather than each method's separately tuned defaults or equal runtime. "
            r"The inherited baselines span multiple recorded runtime revisions."
        ),
    },
}

NAMES = {"area": "AReA", "dype": "DyPE", "sega_official": "SEGA",
         "ultraimage": "UltraImage", "sca": r"\textbf{SCA (ours)}",
         "flux2_full": "Full attention", "flux2_sca": r"\textbf{SCA (ours)}",
         "krea_area": "AReA", "krea_dype": "DyPE", "krea_sega": "SEGA",
         "krea_sca": r"\textbf{SCA (ours)}"}
for method in ("dype", "sega_official", "ultraimage"):
    imgsel.ROOTS[method] = imgsel.OUT + "/main/" + method + "/{res}"
for method in ("flux2_full", "flux2_sca"):
    imgsel.ROOTS[method] = imgsel.OUT + "/flux2/" + method + "/{res}"
imgsel.ROOTS.update({
    "krea_area": imgsel.OUT + "/krea/area/{res}",
    "krea_sca": imgsel.OUT + "/krea/sca_krea/{res}",
    "krea_dype": str(KREA_OVERLAY / "dype/{res}"),
    "krea_sega": str(KREA_OVERLAY / "sega_official/{res}"),
})


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete(method, res, targets=None):
    """Only complete, matching-resolution, seed-zero images with sidecars."""
    valid = {}
    paths = []
    for png in sorted(Path(imgsel.root(method, res)).glob("*.png")):
        match = imgsel.NAME.match(png.name)
        if match and int(match["seed"]) == 0:
            paths.append((match["pid"], str(png)))
    for pid, png in paths:
        if targets is not None and pid not in targets:
            continue
        sidecar = Path(png).with_suffix(".json")
        if not sidecar.exists():
            continue
        meta = json.loads(sidecar.read_text())
        config = meta.get("config", meta)
        if not meta.get("complete", False) or config.get("seed") != 0:
            continue
        if targets is not None:
            if pid in valid:
                raise ValueError(f"Duplicate completed seed-zero image: {method}/{pid}")
            if config.get("prompt_id") != pid or config.get("res") != res:
                raise ValueError(f"Sidecar identity/resolution mismatch: {png}")
            if config.get("prompt") != targets[pid]:
                raise ValueError(f"Prompt differs from evaluation manifest: {png}")
            if config.get("steps") != 28 or config.get("guidance") != 4.5:
                raise ValueError(f"Unexpected sampling protocol: {png}")
            expected_model = ("black-forest-labs/FLUX.1-Krea-dev" if method in ("krea_area", "krea_sca")
                              else "black-forest-labs/FLUX.1-dev")
            if config.get("model") != expected_model:
                raise ValueError(f"Unexpected base/model identity: {png}")
        with Image.open(png) as image:
            if image.size != tuple(reversed([int(v) for v in res.split("x")])):
                continue
        valid[pid] = png
    return valid


def ready_pool(tag):
    """Read headers/sidecars only; strict modes require exact archived prompt IDs."""
    spec = SPECS[tag]
    dev = imgsel.dev4_ids()
    targets, prompt_source = None, None
    if "prompt_manifest" in spec:
        source = Path(spec["prompt_manifest"])
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        targets = {r["id"]: r["prompt"] for r in rows}
        if len(targets) != len(rows) or len(targets) != spec["expected"]:
            raise ValueError(f"{tag}: evaluation manifest must contain {spec['expected']} unique prompt IDs")
        if any(imgsel.short_id(pid) in dev for pid in targets):
            raise ValueError(f"{tag}: evaluation manifest includes development prompts")
        prompt_source = {"path": str(source), "sha256": sha256(source), "count": len(targets)}
    indexes = {m: complete(m, spec["res"], targets) for m in spec["methods"]}
    common = set.intersection(*(set(v) for v in indexes.values()))
    eligible = sorted(pid for pid in common if imgsel.short_id(pid) not in dev)
    if targets is not None and set(eligible) != set(targets):
        counts = ", ".join(f"{m}={len(v)}/{len(targets)}" for m, v in indexes.items())
        raise ValueError(f"{tag}: incomplete evaluation pool; {counts}; paired={len(eligible)}/{len(targets)}")
    if "expected" in spec and len(eligible) != spec["expected"]:
        raise ValueError(f"{tag}: expected {spec['expected']} complete pairs, found {len(eligible)}")
    if not eligible:
        raise ValueError(f"No completed pairs for {tag}")
    for receipt in spec.get("provenance_receipts", []):
        if not Path(receipt).is_file():
            raise ValueError(f"{tag}: required provenance receipt not yet archived: {receipt}")
    if tag == "krea_complete":
        evidence = json.loads(Path(spec["provenance_receipts"][0]).read_text())
        if (evidence.get("transformer_snapshot") != "8162a9c7b05a641be098422bf2fcf335615c2f28"
                or any(evidence.get("coverage", {}).get(m, {}).get("selected_prompts") != spec["expected"]
                       for m in ("dype", "sega_official"))):
            raise ValueError("Krea provenance receipt does not cover the expected checkpoint and prompt count")
        validate_krea_overlay(json.loads(Path(spec["provenance_receipts"][1]).read_text()), indexes, targets)
    return indexes, eligible, prompt_source


def validate_krea_overlay(repair, indexes, targets):
    """Refuse fallback to the seven historical files with unresolved sources."""
    links = repair["links"]
    expected = {(m, pid) for m in ("dype", "sega_official") for pid in targets}
    actual = {(r["method"], r["prompt_id"]) for r in links}
    if (actual != expected or len(links) != len(expected)
            or repair["selected_pairs"] != len(expected)
            or repair["replacement_pairs"] != 7 or sum(r["replacement"] for r in links) != 7):
        raise ValueError("Krea repair receipt does not cover the exact selected baseline pool")
    for entry in links:
        method = {"dype": "krea_dype", "sega_official": "krea_sega"}[entry["method"]]
        sidecar = Path(indexes[method][entry["prompt_id"]]).with_suffix(".json")
        target = Path(entry["target_json"])
        if (sidecar != Path(entry["overlay_json"])
                or not sidecar.is_relative_to(Path(repair["overlay"]))
                or sidecar.resolve() != target.resolve()
                or sidecar.with_suffix(".png").resolve() != target.with_suffix(".png").resolve()):
            raise ValueError(f"Krea selected source differs from the prepared overlay: {sidecar}")
        if entry["replacement"]:
            if (entry["method"] != "sega_official" or not target.is_relative_to(Path(repair["output"]))
                    or target.resolve() == Path(entry["original_json"]).resolve()):
                raise ValueError(f"Krea repair still selects an original unresolved source: {sidecar}")
        elif sha256(sidecar) != entry["original_sidecar_sha256"]:
            raise ValueError(f"Krea inherited sidecar changed: {sidecar}")


def select_prompts(eligible, count):
    eligible = sorted(eligible)
    return sorted(random.Random(17).sample(eligible, min(count, len(eligible))))


def build(tag):
    spec = SPECS[tag]
    name = "gallery_" + tag
    out = Path(imgsel.FIGDIR) / name
    manifest_path = out / "provenance.json"
    methods, res = spec["methods"], spec["res"]
    pool = ready_pool(tag) if "prompt_manifest" in spec or not manifest_path.exists() else None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if pool is not None:
            _, eligible, prompt_source = pool
            if manifest.get("prompt_manifest") != prompt_source or manifest["eligible_ids"] != eligible:
                raise ValueError(f"Changed evaluation pool for frozen gallery: {tag}")
        for entry in manifest["prompts"]:
            for source in entry["sources"].values():
                if sha256(source["png"]) != source["sha256"]:
                    raise ValueError(f"Changed source image: {source['png']}")
                if "sidecar_sha256" in source and sha256(source["sidecar"]) != source["sidecar_sha256"]:
                    raise ValueError(f"Changed source metadata: {source['sidecar']}")
        for receipt in manifest.get("provenance_receipts", []):
            if sha256(receipt["path"]) != receipt["sha256"]:
                raise ValueError(f"Changed provenance receipt: {receipt['path']}")
    else:
        indexes, eligible, prompt_source = pool
        selected = select_prompts(eligible, spec["n"])
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
        if prompt_source is not None:
            manifest["prompt_manifest"] = prompt_source
            manifest["completion_policy"] = "all archived evaluation prompt IDs complete for every displayed method"
        if spec.get("provenance_receipts"):
            manifest["provenance_receipts"] = [{"path": str(p), "sha256": sha256(p)}
                                               for p in spec["provenance_receipts"]]
            manifest["historical_baseline_metadata_note"] = (
                "DyPE and SEGA sidecars identify the FLUX.1-dev base pipeline and omit the Krea "
                "transformer override. The archived launch/runtime receipt establishes the override; "
                "historical runtime revisions may differ and each selected sidecar is preserved. "
                "The repair receipt selects seven regenerated SEGA outputs instead of the unresolved originals.")
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
                sidecar = Path(png).with_suffix(".json")
                entry["sources"][method] = {"png": png, "sha256": sha256(png), "config": config,
                    "sidecar": str(sidecar), "sidecar_sha256": sha256(sidecar)}
            if len(prompt_values) != 1:
                raise ValueError(f"Prompt mismatch: {pid}")
            entry["prompt"] = prompt_values.pop()
            manifest["prompts"].append(entry)
        out.mkdir(exist_ok=True)
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
    if tag == "krea_complete":
        repair = json.loads(Path(spec["provenance_receipts"][1]).read_text())
        caption += (rf" {repair['replacement_pairs']} SEGA outputs were regenerated from an archived runtime "
                    r"to replace historical outputs whose exact source snapshots could not be recovered.")
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
    parser.add_argument("--check-ready", action="store_true",
                        help="Validate source headers, sidecars and required prompt coverage without rendering or writing")
    args = parser.parse_args()
    # New completion-gated modes are explicit, so the established `all` command
    # never promotes a partial campaign or changes the historical gallery pool.
    names = [k for k, v in SPECS.items() if "prompt_manifest" not in v] if args.figure == "all" else [args.figure]
    for name in names:
        if args.check_ready:
            try:
                _, ids, _ = ready_pool(name)
            except ValueError as exc:
                parser.exit(2, str(exc) + "\n")
            print(f"{name}: ready, {len(ids)} complete paired prompts; no files written")
        else:
            build(name)
