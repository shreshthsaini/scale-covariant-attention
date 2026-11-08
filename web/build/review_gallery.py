"""Build a private review gallery: N prompts drawn at random (fixed seed) from the 195 Aesthetic-4K
evaluation prompts, each shown for progressive SCA, single-pass SCA, AReA, HRDiT and UltraImage at 4096^2.
Writes 1600 px WebP views plus one 768 px native-pixel crop per image (same window for all methods)."""
import glob, json, os, random, sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
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
os.makedirs(f"{OUT}/img", exist_ok=True)
items = []
for i, s in enumerate(pick):
    meta = json.load(open(f"{M['prog']}/{s}.json"))
    prompt = meta.get("config", meta).get("prompt", "")
    rng = random.Random(i)
    cx, cy = rng.randint(384, 4096 - 384), rng.randint(384, 4096 - 384)
    for k, d in M.items():
        im = Image.open(f"{d}/{s}.png").convert("RGB")
        im.resize((1600, 1600), Image.LANCZOS).save(f"{OUT}/img/{i:02d}_{k}.webp", quality=80, method=6)
        im.crop((cx - 384, cy - 384, cx + 384, cy + 384)).save(f"{OUT}/img/{i:02d}_{k}_crop.webp", quality=85, method=6)
    items.append({"i": i, "id": s, "prompt": prompt, "crop": [cx, cy]})
    print(i, s, flush=True)
json.dump(items, open(f"{OUT}/items.json", "w"), indent=1)
