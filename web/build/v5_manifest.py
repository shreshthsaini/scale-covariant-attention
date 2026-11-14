"""Audit and finish web/assets/video/v5/manifest.json (run after any v5_*.py build).

For every item: the MP4 and poster exist; ffprobe says H.264 High, yuv420p, even size, no audio stream, moov first;
size within its limit (loops 2.5 MB, formation videos 4 MB, the explainer 10 MB); the recorded width, height, seconds and bytes match the
file; poster size equals the video size; the wording rules hold in alt and caption. It adds "source" (one line of
where the pixels or numbers come from, the field design.md asks for), fixes the item order, and writes totals.
Exit status 1 on any failure.

  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_manifest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_media as M  # noqa: E402

ORDER = ["hero_16x9", "hero_9x16", "hero_16x9_tagged", "hero_9x16_tagged", "speedrace_16x9", "speedrace_9x16",
         "method_window", "method_emulation", "method_stages", "compare_p25", "compare_p50", "compare_p75",
         "formation_silver_braids", "formation_jumping_spider", "formation_fjord_village", "formation_opera_house",
         "formation_antique_shop", "formation_silver_braids_9x16", "formation_jumping_spider_9x16",
         "formation_fjord_village_9x16", "formation_opera_house_9x16", "formation_antique_shop_9x16",
         "process_silver_braids", "process_jumping_spider", "process_fjord_village", "process_opera_house",
         "process_antique_shop", "explainer"]


def source_line(e: dict) -> str:
    i, s = e["id"], e.get("sources")
    if i.startswith("hero"):
        ids = ", ".join(x["id"] for x in s)
        return f"progressive SCA 4096^2 showcase PNGs, seed 0 ({ids}); paper hero draw"
    if i.startswith("speedrace"):
        return ("numbers.json main.*.time_clean (= paper headline macros); images: paper decile prompts 50, 60, 40, "
                "70 from the 195-prompt evaluation pool")
    if i == "method_window":
        return "reach table results/evidence/calib/k1_1024.npz (block 12, heads 4 and 7, r = 4); sheep-church 4096^2"
    if i == "method_emulation":
        return "attention capture paper/figures/src/data/v6_attn_block12.npz; J sums method_assets/provenance.json"
    if i == "method_stages":
        return f"recorded progressive SCA run {s['trace']}"
    if i.startswith("compare"):
        return "paper Figure 5 rows (qual_main_provenance.json): AReA, HRDiT, progressive SCA 4096^2 PNGs"
    if i.startswith("formation"):
        return f"recorded progressive SCA run {s['trace']} (previews, per-step weights) and its final PNG"
    if i.startswith("process"):
        return f"traces {s['progressive_trace']} and {s['area_trace']}"
    if i == "explainer":
        return "paper headline macros; v5 loop frames; v6_observation data; native and full-attention images"
    return ""


def main() -> int:
    data = json.loads(M.MANIFEST.read_text())
    errs = []
    total = 0
    for e in data["items"]:
        mp4, poster = M.OUT / e["file"], M.OUT / e["poster"]
        limit = 10_000_000 if e["id"] == "explainer" else (4_000_000 if e["id"].startswith("formation") else 2_500_000)
        try:
            p = M.check_video(mp4, max_bytes=limit)
        except Exception as ex:  # noqa: BLE001
            errs.append(str(ex))
            continue
        for k, v in (("width", p["width"]), ("height", p["height"]), ("bytes", p["bytes"])):
            if e[k] != v:
                errs.append(f"{e['id']}: manifest {k} {e[k]} != file {v}")
        if abs(e["seconds"] - p["seconds"]) > 0.05:
            errs.append(f"{e['id']}: seconds {e['seconds']} != {p['seconds']}")
        if not poster.exists():
            errs.append(f"{e['id']}: poster missing")
        else:
            pw, ph = Image.open(poster).size
            if (pw, ph) != (p["width"], p["height"]):
                errs.append(f"{e['id']}: poster {pw}x{ph} != video {p['width']}x{p['height']}")
            e["poster_bytes"] = poster.stat().st_size
        if e.get("vtt") and not (M.OUT / e["vtt"]).exists():
            errs.append(f"{e['id']}: vtt missing")
        try:
            M.check_words(e["alt"], e["caption"], where=e["id"])
        except ValueError as ex:
            errs.append(str(ex))
        e["source"] = source_line(e)
        total += p["bytes"] + e.get("poster_bytes", 0)
    rank = {k: n for n, k in enumerate(ORDER)}
    data["items"].sort(key=lambda x: (rank.get(x["id"], 999), x["id"]))
    data["order"] = [x["id"] for x in data["items"]]
    data["total_bytes"] = total
    data["total_mb"] = round(total / 1e6, 2)
    M.MANIFEST.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    for e in data["items"]:
        print(f"{e['id']:24s} {e['role']:18s} {e['width']}x{e['height']:<5d} {e['seconds']:6.2f} s "
              f"{e['bytes'] / 1e6:5.2f} MB loop={e['loop']}")
    print(f"total {total / 1e6:.2f} MB ({len(data['items'])} items incl. posters)")
    for x in errs:
        print("ERROR", x)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
