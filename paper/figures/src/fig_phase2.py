"""Reproducible phase-2 galleries with frozen selection and source provenance.

Run with the existing sca uv environment on a compute node:
  python paper/figures/src/fig_phase2.py all

The first run freezes selection from complete paired outputs. Later runs use that
manifest and verify source hashes. No metric or visual inspection enters selection.

Final operator (2026-10-30): for specs with "final_sca" the SCA column shows the final
operator (sca_v2, out/dupfix/sca_v2; on FLUX.1-Krea-dev sca_v2_krea, out/dupfix_krea/sca_v2_krea)
instead of the published operator (sca_nb; sca_krea). final_sca() keeps the frozen draw,
points each prompt's SCA source at the final operator's image of the same prompt and seed
(imgsel.final_twin or imgsel.final_twin_krea), and recomputes the crop window by the frozen
rule over every method (after checking that the rule reproduces the frozen window from the
published images); the published source and window stay under "published_sca". Specs without
"final_sca" show the published operator and head its column "SCA w/o schedule".
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import imgsel
from fig_qual import (G_GROUP, G_PAIR, G_ROW, G_SECTION, OURS_PUB, OURS_PUB_SHORT, Geo, auto_box, count_word, g_row,
                      grid_snippet, grid_tiles, method_head, rel, t_crop, t_full)

REPO = Path(__file__).resolve().parents[3]
EVAL100 = REPO / "results/evidence/prompts/eval100.jsonl"
KREA_OVERLAY = Path("/scratch/09032/saini_2/sca/out/krea_baseline_overlay_20261017")

# Captions: a bold claim, then at most 35 words in all. The shared protocol (FLUX.1-dev, seed 0, 28 steps,
# guidance 4.5, automatic crop placement, archived selection seeds and hashes, the Krea and FLUX.2 settings
# and the SEGA regeneration) is stated once in the appendix text (main.tex, app:galleries).
SPECS = {
    "baselines": {
        "res": "4096x4096", "n": 3,
        "methods": ["area", "dype", "sega_official", "ultraimage", "sca"],
        "final_sca": True,  # SCA column: the final operator (sca_v2), see final_sca()
        # The frozen draw (2026-10-17) was over the main set without the four development prompts. The paper now
        # evaluates all 195 prompts, so the caption names that pool; both counts come from the frozen manifest.
        "caption": lambda manifest: (
            r"\textbf{At $4096^2$ DyPE and SEGA leave speckle, UltraImage softens the planet, and SCA with the "
            r"schedule is sharp but adds people in two prompts.} "
            rf"{count_word(len(manifest['prompts']))} of {manifest['eligible_count']} prompts drawn at random; "
            r"1:1 crops below."
        ),
    },
    "flux2": {
        "res": "4096x4096", "n": 4,
        "methods": ["flux2_full", "flux2_sca"],
        "caption": (  # laid out by fig_gallery_large.py (display record); kept equal to its caption
            r"\textbf{FLUX.2-dev, $4096^2$: full attention blurs and repeats; SCA without the schedule is sharp "
            r"but duplicates subjects in three rows and distorts the portrait.} Four of 50 prompts drawn at "
            r"random; 1:1 crops follow each image."
        ),
    },
    "6144": {"res": "6144x6144", "n": 4, "methods": ["area", "sca"]},
    "8192": {"res": "8192x8192", "n": 4, "methods": ["area", "sca"]},
    "8192_complete": {
        "res": "8192x8192", "n": 4, "methods": ["area", "sca"], "expected": 12,
        "caption": (
            r"\textbf{At $8192^2$ both fail: AReA blurs and speckles, and SCA without the schedule breaks into "
            r"blocks and stripes.} Exploratory: four of 12 paired prompts drawn at random; 1:1 crops follow each "
            r"image."
        ),
    },
    "6144_complete": {
        "res": "6144x6144", "n": 4, "methods": ["area", "sca"],
        "expected": 100, "prompt_manifest": EVAL100,
        "caption": (
            r"\textbf{At $6144^2$ AReA's crops dissolve into speckle or streaks; SCA without the schedule keeps "
            r"layouts but shows stripes, block speckle, or extra figures.} Four of 100 prompts drawn at random; "
            r"1:1 crops follow each image."
        ),
    },
    "krea_complete": {
        "res": "4096x4096", "n": 2,
        "methods": ["krea_area", "krea_dype", "krea_sega", "krea_sca"],
        "expected": 100, "prompt_manifest": EVAL100,
        "provenance_receipts": [REPO / "results/evidence/krea_baseline_provenance" / name
                                for name in ("receipt.json", "repair_preparation.json", "algorithm_review.json")],
        "final_sca": "krea_sca",  # SCA column: the final operator on Krea (sca_v2_krea), see final_sca()
        "caption": (
            # Audited 2026-10-30 at 1:1 (sca_v2_krea): the portrait is clean; the lone figure's legs melt into
            # hanging strands and a second person stands on the left rocks. DyPE ghosts the face in both versions.
            r"\textbf{On FLUX.1-Krea-dev at $4096^2$ DyPE ghosts the face; SCA melts the lone figure's legs and "
            r"adds a second person.} Two of 100 prompts drawn at random; SCA with the schedule; 1:1 crops below."
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


# SCA column -> (final-operator twin of a published image, method label, image description for the record)
FINAL_TWINS = {
    "sca": (imgsel.final_twin, "sca_v2", "out/dupfix/sca_v2 (imgsel.final_twin of each published image: same "
                                         "prompt, prompt id, derived seed and file name)"),
    "krea_sca": (imgsel.final_twin_krea, "sca_v2_krea",
                 "out/dupfix_krea/sca_v2_krea (imgsel.final_twin_krea of each published Krea image: same prompt, "
                 "prompt id, derived seed, file name and Krea calibration table)"),
}


def final_sca(tag, manifest):
    """Switch a frozen record's SCA column to the final operator (sca_v2); the draw is unchanged."""
    col = "sca" if SPECS[tag]["final_sca"] is True else SPECS[tag]["final_sca"]
    twin, method, images = FINAL_TWINS[col]
    methods, size = manifest["methods"], 512 * int(manifest["resolution"].split("x")[0]) // 4096
    for entry in manifest["prompts"]:
        src = entry["sources"][col]
        pub = entry.setdefault("published_sca", dict(
            {k: src[k] for k in ("png", "sha256", "sidecar", "sidecar_sha256") if k in src},
            crop_xy_size=list(entry["crop_xy_size"])))
        if sha256(pub["png"]) != pub["sha256"] or ("sidecar" in pub and sha256(pub["sidecar"]) != pub["sidecar_sha256"]):
            raise ValueError(f"Changed source image: {pub['png']}")
        png, digest = twin(pub["png"])
        sidecar = Path(png).with_suffix(".json")
        entry["sources"][col] = {"png": png, "sha256": digest, "config": json.loads(sidecar.read_text())["config"],
                                 "sidecar": str(sidecar), "sidecar_sha256": sha256(sidecar)}
        others = {m: imgsel.load(entry["sources"][m]["png"]) for m in methods if m != col}
        published = [others[m] if m != col else imgsel.load(pub["png"]) for m in methods]
        if list(auto_box(published, size)) != pub["crop_xy_size"]:
            raise ValueError(f"{tag} {entry['short_id']}: the crop rule does not reproduce the frozen window")
        final = [others[m] if m != col else imgsel.load(png) for m in methods]
        entry["crop_xy_size"] = list(auto_box(final, size))
    manifest["final_operator"] = dict(
        method=method, images=images,
        switched="2026-10-30, by fig_phase2.final_sca: the frozen draw is unchanged; every SCA source is the final "
                 "operator's image; every crop window is recomputed by crop_rule over every method; the published "
                 "source and window are kept per prompt under published_sca")


def select_prompts(eligible, count):
    eligible = sorted(eligible)
    return sorted(random.Random(17).sample(eligible, min(count, len(eligible))))


def build(tag):
    spec = SPECS[tag]
    name = "gallery_" + tag
    out = Path(imgsel.FIGDIR) / name
    manifest_path = out / "provenance.json"
    if manifest_path.exists() and "display" in json.loads(manifest_path.read_text()):
        print(f"{name}: skipped; fig_gallery_large.py lays out this gallery (display record in provenance.json)")
        return
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
        if spec.get("final_sca"):
            final_sca(tag, manifest)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
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

    # Layout: the shared vector grid of fig_qual (snippets/_scagrid.tex). Two methods: one prompt per
    # row, each full image followed by its crop. More methods: one column per method, each prompt a
    # block of two rows, full images above their crops (no row labels; the caption says so).
    vertical = len(methods) > 2
    if vertical:
        geo = Geo(len(methods), (len(methods) - 1) * G_GROUP)
    else:
        geo = Geo(2 * len(methods), len(methods) * G_PAIR + (len(methods) - 1) * G_GROUP)
    ours = "SCA (ours)" if spec.get("final_sca") else (OURS_PUB_SHORT if vertical else OURS_PUB)
    head = []
    for k, method in enumerate(methods):
        head += ([G_GROUP] if k else []) + [method_head(method, 1, 0, NAMES, ours=ours) if vertical
                                            else method_head(method, 2, G_PAIR, NAMES, ours=ours)]
    rows = [(g_row(head), G_ROW)]
    written = {}
    for entry in manifest["prompts"]:
        sid, box = entry["short_id"], tuple(entry["crop_xy_size"])
        tiles = {}
        for method in methods:
            _, tiles[method] = grid_tiles(dict(out=str(out), name=f"{sid}_{method}",
                png=entry["sources"][method]["png"], px=geo.px(), crops=[(box, "crop")]))
            written[method, sid] = tiles[method]
        if vertical:
            full, crops = [], []
            for k, method in enumerate(methods):
                full += ([G_GROUP] if k else []) + [t_full(tiles[method]["full"], box, res)]
                crops += ([G_GROUP] if k else []) + [t_crop(tiles[method]["crop"])]
            rows += [(g_row(full), G_PAIR), (g_row(crops), G_SECTION)]
        else:
            cells = []
            for k, method in enumerate(methods):
                cells += ([G_GROUP] if k else []) + [t_full(tiles[method]["full"], box, res), G_PAIR,
                                                     t_crop(tiles[method]["crop"])]
            rows.append((g_row(cells), G_ROW))
    caption = spec.get("caption")
    if callable(caption):
        if manifest["eligible_count"] != len(manifest["eligible_ids"]):
            raise ValueError(f"{tag}: eligible_count disagrees with eligible_ids in the frozen manifest")
        caption = caption(manifest)
    if tag == "krea_complete":  # the appendix text states the regeneration; the count must still match
        repair = json.loads(Path(spec["provenance_receipts"][1]).read_text())
        if repair["replacement_pairs"] != 7:
            raise ValueError("app:galleries in main.tex states seven regenerated SEGA outputs; update it")
    if caption is None:
        side = res.split("x")[0]
        claim = (rf"\textbf{{Partial results at ${side}^2$: grid and stripe artifacts remain.}}" if tag == "8192"
                 else rf"\textbf{{Partial results at ${side}^2$ from an incomplete pool.}}")
        caption = (
            rf"{claim} {count_word(len(manifest['prompts']))} of the {manifest['eligible_count']} prompts paired "
            r"at the snapshot, drawn at random; the pool may reflect completion order. After each image, a 1:1 crop."
        )
    comments = [f"Generated by fig_phase2.py {tag}; frozen selection in {name}/provenance.json."]
    comments += [f"{e['short_id']}: {e['prompt'].replace(chr(0x2014), ' - ')}" for e in manifest["prompts"]]
    grid_snippet(name, geo, rows, caption, "fig:" + name, comments=comments, source=f"fig_phase2.py {tag}")
    # Output hashes are the only part of the frozen manifest a re-render may change.
    paths = sorted(p for t in written.values() for p in t.values())
    manifest["outputs"] = {rel(p): sha256(p) for p in paths}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
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
