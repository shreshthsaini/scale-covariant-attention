import json, sys
from PIL import Image, ImageDraw
T = "/scratch/09032/saini_2/sca/web_assets/trace"; OUT = sys.argv[1]
STEPS = [5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 21, 27]
jobs = {"climber_copy": ("997293625843621481", (2948, 927), 384), "climber_edge": ("997293625843621481", (2720, 90), 384),
        "ball_a": ("3943607367684487957", (2522, 1044), 512), "ball_b": ("3943607367684487957", (769, 781), 512)}
for name, (sid, (cx, cy), side) in jobs.items():
    sig = json.load(open(f"{T}/sca/{sid}/trace.json"))
    sigmas = {s["step"]: s["sigma"] for s in sig["steps"]} if "steps" in sig else {}
    rows = []
    for m in ("sca", "sca_v2", "area"):
        row = []
        for k in STEPS:
            im = Image.open(f"{T}/{m}/{sid}/steps/x0_{k:02d}.webp").convert("RGB")
            s = im.size[0] / 4096
            x0, y0 = max(0, (cx - side / 2) * s), max(0, (cy - side / 2) * s)
            c = im.crop((int(x0), int(y0), int(x0 + side * s), int(y0 + side * s))).resize((128, 128), Image.LANCZOS)
            d = ImageDraw.Draw(c); d.rectangle((0, 0, 127, 11), fill=(0, 0, 0))
            d.text((2, 0), f"{m[:4]} {k} {sigmas.get(k, 0):.2f}" if sigmas else f"{m[:4]} {k}", fill=(255, 255, 0))
            row.append(c)
        rows.append(row)
    sheet = Image.new("RGB", (128 * len(STEPS), 128 * 3))
    for r, row in enumerate(rows):
        for j, c in enumerate(row): sheet.paste(c, (j * 128, r * 128))
    sheet.save(f"{OUT}/{name}.jpg", quality=90)
print("ok", list(json.load(open(f"{T}/sca/997293625843621481/trace.json")).keys())[:8])
