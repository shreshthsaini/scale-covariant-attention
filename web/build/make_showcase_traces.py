#!/usr/bin/env python3
"""Trace selection, frozen code and spool tasks for the showcase "how it works" videos (HANDOVER Section 32).

For the chosen showcase prompt ids (12 by default):
  web/build/showcase_trace_selection.json   sample_trace.py selection: per prompt the progressive SCA and AReA 4096^2
                                            campaign PNGs (out/showcase) and one 512 px box, the window with the most
                                            gradient energy in the progressive image (web/build/build_showcase.py rule)
  prompts/showcase12_trace.jsonl            the selected rows (AReA runs these first)
--freeze  code/web_trace_sc_<commit>: `git archive 7bd64c0` (the campaign tree of the progressive and AReA runs) plus
          the committed overlay scripts/sample_trace.py, scripts/sample_progressive.py (per-step hook) and the
          selection file; never modified once written.
--enqueue taskq/pending: 00s4_f_area4096tr_* (AReA 4096^2 on the selected prompts, campaign flags, code/7bd64c0),
          00s5_a_trace_prog_* (progressive SCA traces) and 00s5_b_trace_area_* (AReA traces, each waits for its AReA
          chunk), outputs under /scratch/09032/saini_2/sca/web_assets/trace_showcase/<method>/<prompt id>/.

Run with the sca env python: python web/build/make_showcase_traces.py --ids a,b,... [--freeze] [--enqueue]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "slurm"))
sys.path.insert(0, str(HERE))
import showcase_tasks as stq  # noqa: E402
from build_showcase import crop_boxes, record, stem  # noqa: E402

S = Path("/scratch/09032/saini_2/sca")
SEL = HERE / "showcase_trace_selection.json"
TRACE_PROMPTS = S / "prompts/showcase12_trace.jsonl"
OUT_ROOT = S / "web_assets/trace_showcase"
CAMPAIGN = "7bd64c0"
FROZEN = ["sca", "scripts", "slurm", "tests", "README.md", "HANDOVER.md", ".gitignore"]
OVERLAY = ["scripts/sample_trace.py", "scripts/sample_progressive.py", "web/build/showcase_trace_selection.json"]
BOX = 512


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(ids):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    rows = {r["id"]: r for r in (json.loads(l) for l in open(stq.PROMPTS) if l.strip())}
    items = []
    for pid in ids:
        row = rows[pid]
        prog = S / "out/showcase/flux1_sca_v2_prog/4096x4096" / f"{stem(pid)}.png"
        if record(prog) is None:
            raise SystemExit(f"{pid}: no complete progressive SCA image {prog}")
        with Image.open(prog) as im:
            (x, y, s), = crop_boxes(im.convert("RGB"), n=1, side=BOX)
        items.append({"sid": pid, "prompt": row["prompt"], "label": row["title"], "role": row["category"],
                      "paths": {"sca_v2_prog": str(prog),
                                "area_sc": str(S / "out/showcase/flux1_area/4096x4096" / f"{stem(pid)}.png")},
                      "box_xys": [x, y, s]})
        print(pid, "box", [x, y, s], flush=True)
    SEL.write_text(json.dumps({"description": "showcase trace selection (web/build/make_showcase_traces.py): progressive "
                                              "SCA and AReA at 4096^2, seed 0; box = the 512 px window with the most "
                                              "gradient energy in the progressive image", "items": items}, indent=1) + "\n")
    TRACE_PROMPTS.write_text("".join(json.dumps(rows[p]) + "\n" for p in ids))
    print("wrote", SEL, TRACE_PROMPTS)


def freeze():
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--"] + OVERLAY, cwd=REPO, text=True).strip()
    if dirty:
        raise SystemExit(f"commit the overlay first: {dirty}")
    dest = S / "code" / f"web_trace_sc_{commit}"
    if dest.exists():
        print("exists, kept:", dest)
        return dest
    dest.mkdir(parents=True)
    tar = subprocess.Popen(["git", "archive", CAMPAIGN] + FROZEN, cwd=REPO, stdout=subprocess.PIPE)
    subprocess.check_call(["tar", "-x", "-C", str(dest)], stdin=tar.stdout)
    tar.wait()
    for rel in OVERLAY:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest / rel)
    same = sha(dest / "sca/attn.py") == sha(S / "code" / CAMPAIGN / "sca/attn.py")
    (dest / "COMMIT").write_text(commit + "\n")
    (dest / "OVERLAY.md").write_text(
        f"# web_trace_sc_{commit}\n\nFrozen copy for the showcase trace tasks. Never edit it.\n\n"
        f"- `git archive {CAMPAIGN}` of: {', '.join(FROZEN)}.\n"
        f"- `sca/attn.py` identical to the campaign copy {S / 'code' / CAMPAIGN}: {'yes' if same else 'NO'}.\n"
        f"- Overlay from commit {commit}:\n\n| File | SHA-256 |\n|---|---|\n"
        + "".join(f"| `{r}` | `{sha(dest / r)}` |\n" for r in OVERLAY))
    print("froze", dest, "attn identical:", same)
    return dest


def enqueue(dest, ids, dry=False):
    rows = list(range(len(ids)))
    chunks = [ids[a:a + 4] for a in range(0, len(ids), 4)]
    run = f"{stq.PY} scripts/run_area.py --seed 0 --guidance 4.5"
    for k, a in enumerate(range(0, len(rows), 4)):
        b = min(a + 4, len(rows))
        stq.task(f"00s4_f_area4096tr_{k}.sh", 20, "AReA on FLUX.1-dev at 4096^2, the traced prompts first",
                 str(S / "code" / CAMPAIGN), f"{run} --res 4096x4096 --prompts {TRACE_PROMPTS} "
                 f"--out {stq.OUT}/flux1_area/4096x4096 --start {a} --end {b}", dry, need=50)
    sel = dest / "web/build/showcase_trace_selection.json"
    for k, sids in enumerate(chunks):
        base = (f"{stq.PY} scripts/sample_trace.py --prompts {stq.PROMPTS} --selection {sel} --out-root {OUT_ROOT} "
                f"--sid {' '.join(sids)}")
        stq.task(f"00s5_a_trace_prog_{k}.sh", 45, "progressive SCA traces (57 steps) for the showcase videos", str(dest),
                 f"{base} --method sca_v2_prog", dry, need=50)
        stq.task(f"00s5_b_trace_area_{k}.sh", 60, "AReA traces (28 steps) for the showcase videos", str(dest),
                 f"{base} --method area_sc", dry, need=50, wait=f"{stq.OUT}/_done/00s4_f_area4096tr_{k}.sh")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", required=True)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--enqueue", action="store_true")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    ids = [i for i in a.ids.split(",") if i]
    if not a.freeze and not a.enqueue:
        select(ids)
        return
    dest = freeze() if a.freeze else None
    if a.enqueue:
        if dest is None:
            raise SystemExit("--enqueue needs --freeze (the tree the tasks run in)")
        enqueue(dest, ids, a.dry)


if __name__ == "__main__":
    main()
