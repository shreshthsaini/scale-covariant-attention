"""The second page's prompt draw: N prompts drawn once with a fixed seed (random.Random(2026)) from the 195 Aesthetic-4K
evaluation prompts that have an image for every method shown there (progressive SCA, single-pass SCA, AReA, HRDiT and
UltraImage at 4096^2, seed 0). Writes web/review/items.json ({i, id, prompt}); the page's zoom grids, overviews and
native-pixel tiles are built from it by `build_zoom.py review` (web/review/items.js, web/review/img/z/).

Run (repository root): python web/build/review_gallery.py web/review 96
"""
import glob, json, os, random, sys
O = "/scratch/09032/saini_2/sca/out"
OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 48
M = {"prog": f"{O}/dupfix_prog/sca_v2_prog/4096x4096", "sca": f"{O}/dupfix/sca_v2/4096x4096",
     "area": f"{O}/main/area/4096x4096", "hrdit": f"{O}/hrdit191eval/hrdit_dev_author/4096x4096",
     "ultra": f"{O}/main/ultraimage/4096x4096"}
def stems(d):
    return {os.path.basename(p)[:-4] for p in glob.glob(f"{d}/*.png")}
common = sorted(set.intersection(*(stems(d) for d in M.values())))
pick = random.Random(2026).sample(common, min(N, len(common)))
items = []
for i, s in enumerate(pick):
    meta = json.load(open(f"{M['prog']}/{s}.json"))
    items.append({"i": i, "id": s, "prompt": meta.get("config", meta).get("prompt", "")})
json.dump(items, open(f"{OUT}/items.json", "w"), indent=1)
print(f"{OUT}/items.json: {len(items)} prompts")
