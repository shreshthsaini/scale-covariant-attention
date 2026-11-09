#!/usr/bin/env python3
"""Generate one resumable AReA (official release) chunk with Qwen-Image (HANDOVER Section 38).

AReA's upstream release (github.com/KooroshRH/AReA, commit 3efb7e5 "Add Qwen-Image support"), vendored unmodified in
third_party/AReA_3efb7e5 (env AREA_QWEN_DIR overrides), is attached to the same Qwen-Image snapshot and diffusers
pipeline as scripts/sample_qwen.py. The defaults are AReA's own for --model qwen-image, read from the vendored
generate.py at run time (so they cannot drift from it): 40 steps, true CFG 4.0 with the empty negative prompt, the
quality suffix QWEN_SUFFIX appended to every prompt, VAE tiling on; the AReA settings are the AReA() defaults:
switch sigma 0.87 (the Qwen backbone's), subsample pooling, Halo 8, bridge ratio 0.5, AReA seed = run seed. The
backbone also replaces the timestep shift (mu 2.0 up to 4096^2, plus 0.5 ln(tokens / 256^2) above), so this script
never touches the pipeline's calculate_shift.

Differences from generate.py, all in the harness and none in the method:
- Seeding as in every project run: a CPU generator seeded with the per-prompt derived seed (sca.runtime.image_seed),
  so AReA starts from the same initial noise as SCA and full attention at the same size. generate.py seeds a CUDA
  generator with the run seed.
- Prompt embeddings come from QwenImagePipeline.encode_prompt through sca/qwen.py's cache, and the text encoder is not
  resident while sampling, exactly as for SCA; the timed scope is the same as sample_qwen.py's (denoising, two
  transformer calls per step, and VAE decode).
- The sidecar's "prompt" is the benchmark prompt, which the metrics score against; the text the model was conditioned
  on is config.conditioning_prompt (prompt + suffix).

Timing hygiene: before each image the GPU's compute processes are listed with nvidia-smi and the run waits up to
--exclusive-wait seconds for other processes to leave; while the image is generated a thread lists them every
--poll seconds. The sidecar records every other process seen (gpu_cotenants), gpu_shared and first_call_in_process.

--dry-run prints the config and AReA's per-step schedule without loading a model (CPU).
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AREA_DIR = Path(os.environ.get("AREA_QWEN_DIR", str(ROOT / "third_party" / "AReA_3efb7e5")))
AREA_COMMIT = "3efb7e57effa2f0c0ac0bc78d80d53455bd34cc7"
AREA_REPO = "https://github.com/KooroshRH/AReA"

from sca.runtime import (atomic_json, file_sha256, fingerprint, image_seed, image_stem, read_prompts,  # noqa: E402
                         resolution, runtime_environment)


# ----------------------------------------------------------------------------- upstream defaults (vendored generate.py)

def upstream_defaults(area_dir=AREA_DIR):
    """AReA generate.py's Qwen-Image settings, parsed (not imported: generate.py imports Ideogram 4 classes at the top):
    {"suffix", "steps", "guidance", "guidance_arg", "negative_prompt", "vae_tiling", "checkpoint"}."""
    tree = ast.parse((Path(area_dir) / "generate.py").read_text())
    suffix, models = None, None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "QWEN_SUFFIX":
                suffix = ast.literal_eval(node.value)
            elif node.targets[0].id == "MODELS":
                models = node.value
    if suffix is None or not isinstance(models, ast.Dict):
        raise RuntimeError(f"QWEN_SUFFIX or MODELS not found in {area_dir}/generate.py")
    for k, v in zip(models.keys, models.values):
        if isinstance(k, ast.Constant) and k.value == "qwen-image" and isinstance(v, ast.Call):
            kw = {}
            for a in v.keywords:
                if a.arg == "prompt_suffix":
                    if not (isinstance(a.value, ast.Name) and a.value.id == "QWEN_SUFFIX"):
                        raise RuntimeError("qwen-image prompt_suffix is no longer QWEN_SUFFIX")
                    continue
                if a.arg == "pipeline":
                    continue
                kw[a.arg] = ast.literal_eval(a.value)
            return {"suffix": suffix, "steps": int(kw["steps"]), "guidance": float(kw["guidance"]),
                    "guidance_arg": kw.get("guidance_arg", "guidance_scale"), "negative_prompt": kw["negative_prompt"],
                    "vae_tiling": kw["vae_tiling"], "checkpoint": kw["checkpoint"]}
    raise RuntimeError(f"no qwen-image entry in {area_dir}/generate.py MODELS")


def area_hash(area_dir=AREA_DIR):
    """SHA-256 over this script, the vendored package and generate.py, and the sca helpers used (sca/qwen.py,
    sca/runtime.py). Unrelated repository edits leave it unchanged, so resume keys stay stable."""
    d, area_dir = hashlib.sha256(), Path(area_dir)
    items = [(f"repo/{p.relative_to(ROOT)}", p) for p in (Path(__file__).resolve(), ROOT / "sca" / "qwen.py",
                                                           ROOT / "sca" / "runtime.py")]
    items += [(f"area/{p.relative_to(area_dir)}", p)
              for p in [area_dir / "generate.py"] + sorted((area_dir / "area").rglob("*.py"))]
    for name, p in items:
        d.update(name.encode())
        d.update(p.read_bytes())
    return d.hexdigest()


# ----------------------------------------------------------------------------- GPU co-tenants

def _ancestors(pid):
    out = set()
    p = int(pid)
    for _ in range(64):
        try:
            with open(f"/proc/{p}/stat") as fh:
                p = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if p <= 1:
            break
        out.add(p)
    return out


class GpuWatch:
    """Other processes on this job's GPU (nvidia-smi --query-compute-apps). Ours (this process and its descendants)
    are excluded. available is False when nvidia-smi is missing (CPU runs)."""

    def __init__(self, interval=10.0):
        dev = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip() or "0"
        self.index, self.interval, self.me = dev, float(interval), os.getpid()
        self.available = self._query() is not None
        self._seen, self._stop, self._thread = {}, None, None

    def _query(self):
        try:
            out = subprocess.run(["nvidia-smi", "-i", self.index, "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        apps = []
        for line in out.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                pid = int(parts[0])
                if pid != self.me and self.me not in _ancestors(pid):
                    apps.append((pid, int(parts[1]) if parts[1].isdigit() else None))
        return apps

    def others(self):
        return self._query() or []

    def wait_exclusive(self, limit, log=print):
        """Wait up to limit seconds (15 s polls) until no other process holds the GPU; returns (waited s, others)."""
        t0, others = time.time(), self.others()
        if others:
            log(f"{time.strftime('%FT%T')} GPU {self.index} shared with {others}; waiting up to {int(limit)} s")
        while others and time.time() - t0 < limit:
            time.sleep(15)
            others = self.others()
        return round(time.time() - t0, 1), others

    def start(self):
        self._seen = {}
        if not self.available:
            return
        self._stop = threading.Event()

        def loop():
            while True:
                stamp = time.time()
                for pid, mib in self.others():
                    rec = self._seen.setdefault(pid, {"pid": pid, "first_seen": stamp, "max_mib": mib})
                    rec["last_seen"] = stamp
                    if mib is not None and (rec["max_mib"] is None or mib > rec["max_mib"]):
                        rec["max_mib"] = mib
                if self._stop.wait(self.interval):
                    return
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=60)
            self._thread = None
        return sorted(self._seen.values(), key=lambda r: r["pid"])


# ----------------------------------------------------------------------------- AReA schedule

def area_mu(h, w):
    """mu AReA's Qwen backbone uses for an h x w patch grid (area/backbones/qwen.py _shift)."""
    from area.backbones import qwen as Q
    return Q.SHIFT + max(0.0, 0.5 * math.log(h * w / Q.SHIFT_TOKENS))


def area_schedule(res, steps, switch_sigma, txt_len):
    """AReA's per-step plan for this canvas: Phase I pooling factors, then Phase II window sides (patches)."""
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from area.backbones import qwen as Q
    from area.schedule import build_schedule
    from sca.qwen import scheduler_config
    h, w = res[0] // 16, res[1] // 16
    if max(h, w) <= Q.NATIVE_PATCHES:
        return {"note": "native or below: plain attention", "phase1_steps": 0, "factors": [], "windows": []}
    s = FlowMatchEulerDiscreteScheduler(**scheduler_config())
    s.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=area_mu(h, w))
    sigmas = s.sigmas[:steps].float().numpy()
    sig = Q.SWITCH_SIGMA if switch_sigma is None else switch_sigma
    p1 = int((sigmas > sig).sum())
    sch = build_schedule(h, w, steps, p1, Q.NATIVE_PATCHES, Q.WINDOW_FLOOR, txt_len=txt_len)
    return {"switch_sigma": sig, "phase1_steps": p1, "native_patches": Q.NATIVE_PATCHES, "window_floor": Q.WINDOW_FLOOR,
            "txt_len": int(txt_len), "mu": round(area_mu(h, w), 6), "factors": list(sch.factors),
            "windows": list(sch.windows)}


# ----------------------------------------------------------------------------- CLI

def parser(defaults):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=resolution, required=True, metavar="HxW")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default="qwen_area", help="Method name written to the sidecar")
    p.add_argument("--steps", type=int, default=defaults["steps"], help="AReA generate.py default for qwen-image")
    p.add_argument("--true-cfg", type=float, default=defaults["guidance"], help="AReA generate.py default (true CFG)")
    p.add_argument("--negative", default=defaults["negative_prompt"], help="AReA generate.py default: empty string")
    p.add_argument("--prompt-suffix", action=argparse.BooleanOptionalAction, default=True,
                   help="Append AReA's QWEN_SUFFIX (generate.py default on)")
    p.add_argument("--switch-sigma", type=float, default=None, help="sigma*; default the Qwen backbone's 0.87")
    p.add_argument("--pooling", default="subsample", choices=("subsample", "mean", "medoid", "random"))
    p.add_argument("--halo", type=int, default=8)
    p.add_argument("--bridge-ratio", type=float, default=0.5)
    p.add_argument("--area-seed", type=int, default=None, help="AReA jitter/Halo seed; default --seed (generate.py)")
    p.add_argument("--exclusive-wait", type=float, default=120.0,
                   help="Seconds to wait before each image for other GPU processes to leave (then run, flagged)")
    p.add_argument("--poll", type=float, default=10.0, help="Seconds between co-tenant polls during an image")
    p.add_argument("--embedding-cache", type=Path, help="Prompt-embedding cache directory (default sca/qwen.py)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="No model: print config and schedule, then exit")
    return p


def main(argv=None):
    defaults = upstream_defaults()
    args = parser(defaults).parse_args(argv)
    runtime_environment()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no network at run time
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(AREA_DIR))
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    if not 0 <= args.start < end <= len(prompts) or args.steps <= 0:
        raise ValueError("Require 0 <= start < end <= number of prompts and positive steps")
    rows = prompts[args.start:end]
    suffix = defaults["suffix"] if args.prompt_suffix else ""
    cfg_on = args.true_cfg > 1
    h, w = args.res
    grid = (h // 16, w // 16)
    import diffusers
    import torch
    from area import AReAConfig
    acfg = AReAConfig(switch_sigma=args.switch_sigma, pooling=args.pooling, halo=args.halo,
                      bridge_ratio=args.bridge_ratio, seed=args.seed if args.area_seed is None else args.area_seed)
    from area.backbones.qwen import SWITCH_SIGMA
    impl = area_hash()

    def config(row):
        return {"schema_version": 1, "method": args.label, "res": f"{h}x{w}", "prompt_id": str(row["id"]),
                "prompt": row["prompt"], "conditioning_prompt": row["prompt"] + suffix, "prompt_suffix": suffix,
                "seed": int(args.seed), "derived_seed": image_seed(row["id"], args.seed), "steps": args.steps,
                "true_cfg_scale": args.true_cfg, "negative_prompt": args.negative if cfg_on else None,
                "model": "Qwen/Qwen-Image", "dtype": "bfloat16", "batch_size": 1,
                "attention": "area_official (AReA qwen backbone; diffusers dispatch_attention_fn, default backend)",
                "diffusers": diffusers.__version__, "torch": torch.__version__, "implementation_sha256": impl,
                "grid": list(grid), "generator": "cpu, derived_seed", "vae_tiling": True,
                "mu_rule": "AReA Qwen backbone: 2.0 + max(0, 0.5 ln(tokens / 256^2))",
                "text_encoder": "Qwen2.5-VL via QwenImagePipeline.encode_prompt, precomputed (sca/qwen.py cache)",
                "area": {"repo": AREA_REPO, "commit": AREA_COMMIT, "pooling": acfg.pooling,
                         "switch_sigma": SWITCH_SIGMA if acfg.switch_sigma is None else acfg.switch_sigma,
                         "halo": acfg.halo, "bridge_ratio": acfg.bridge_ratio, "seed": acfg.seed,
                         "generate_py_defaults": {k: v for k, v in defaults.items() if k != "suffix"}}}

    def paths(row):
        stem = image_stem(row["id"], args.seed)
        return args.out / f"{stem}.png", args.out / f"{stem}.json"

    def done(row):
        png, meta = paths(row)
        return png.exists() and meta.exists() and json.loads(meta.read_text()).get("complete")

    if args.dry_run:
        c = config(rows[0])
        c["schedule_preview"] = area_schedule(args.res, args.steps, acfg.switch_sigma, txt_len=0)
        print(json.dumps(c, indent=2))
        for row in rows:
            print(f"{'done' if done(row) else 'todo'} {row['id']} derived_seed={image_seed(row['id'], args.seed)}")
        print(f"dry-run ok: {len(rows)} prompts (schedule preview with txt_len 0; the run uses the prompt's length)")
        return

    from area import AReA
    from sca.qwen import encode_prompts, free_cuda, load_pipeline, snapshot
    import sca.qwen as SQ
    cache = {"cache": args.embedding_cache} if args.embedding_cache is not None else {}
    todo = [r for r in rows if not done(r)]
    t0 = time.perf_counter()
    texts = [r["prompt"] + suffix for r in todo] + ([args.negative] if cfg_on else [])
    enc = encode_prompts(texts, args.device, **cache) if todo else []
    neg = enc.pop() if (cfg_on and todo) else None
    embeds = dict(zip([r["id"] for r in todo], enc))
    encode_seconds = time.perf_counter() - t0
    if not todo:
        print(f"nothing to do: {len(rows)} images complete", flush=True)
        return
    free_cuda()
    pipe = load_pipeline(args.device)
    if defaults["vae_tiling"] == "always":
        pipe.vae.enable_tiling()
    stock_shift = SQ.mu_for(grid[0] * grid[1], "default", dict(pipe.scheduler.config))   # the pipeline's own mu
    area = AReA(pipe, config=acfg)
    watch = GpuWatch(args.poll)
    host = socket.gethostname()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() and str(args.device).startswith("cuda") else None
    calls = 0
    try:
        for row in rows:
            png, meta = paths(row)
            c = config(row)
            c["snapshot"] = snapshot().name
            c["scheduler"] = dict(area.backbone._scheduler.config)       # the stock config AReA wraps
            digest = fingerprint(c)
            if png.exists() and meta.exists():
                saved = json.loads(meta.read_text())
                if saved.get("complete"):
                    if saved.get("config_fingerprint") != digest:
                        raise ValueError(f"Existing result has different settings: {meta}. Choose a new --out.")
                    if file_sha256(png) != saved["png_sha256"]:
                        raise ValueError(f"Existing PNG checksum differs: {png}")
                    print(f"skip {row['id']} {args.label} {c['res']}", flush=True)
                    continue
            waited, before = watch.wait_exclusive(args.exclusive_wait) if watch.available else (0.0, [])
            generator = torch.Generator(device="cpu").manual_seed(image_seed(row["id"], args.seed))
            cuda = str(args.device).startswith("cuda")
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            kw = {"negative_prompt_embeds": neg.to(args.device)} if cfg_on else {}
            watch.start()
            t_start = time.time()
            start = time.perf_counter()
            with torch.inference_mode():
                image = pipe(prompt=None, prompt_embeds=embeds[row["id"]].to(args.device), height=h, width=w,
                             num_inference_steps=args.steps, true_cfg_scale=args.true_cfg, generator=generator,
                             **kw).images[0]
            if cuda:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            t_end = time.time()
            cotenants = watch.stop()
            calls += 1
            sched = area.backbone._schedule
            plan = None if sched is None else {"factors": list(sched.factors), "windows": list(sched.windows),
                                               "phase1_steps": len(sched.factors)}
            sigmas = [float(s) for s in pipe.scheduler.sigmas[:-1]]
            args.out.mkdir(parents=True, exist_ok=True)
            tmp = png.with_name(png.name + f".tmp.{os.getpid()}")
            image.save(tmp, format="PNG")
            os.replace(tmp, png)
            shared = bool(before) or bool(cotenants)
            atomic_json(meta, {"complete": True, "config": c, "config_fingerprint": digest,
                               "code_revision": os.environ.get("SCA_CODE_REVISION", "frozen tree, see task"),
                               "image": str(png.resolve()), "png_sha256": file_sha256(png), "seconds": seconds,
                               "seconds_scope": "denoising (two transformer calls per step with true CFG) and VAE "
                               "decode; prompt embeddings precomputed (text encoder not resident); excludes model "
                               "loading and PNG I/O",
                               "encode_seconds_chunk": encode_seconds, "first_call_in_process": calls == 1,
                               "peak_memory_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                               "sigmas": sigmas, "area_schedule": plan,
                               "text_tokens": int(embeds[row["id"]].shape[1]),
                               "negative_text_tokens": int(neg.shape[1]) if neg is not None else None,
                               "mu_pipeline": stock_shift, "mu_area": area_mu(*grid),
                               "t_start": t_start, "t_end": t_end, "host": host, "gpu": gpu,
                               "gpu_watch": {"available": watch.available, "poll_seconds": watch.interval,
                                             "waited_for_exclusive_seconds": waited,
                                             "others_at_start": [{"pid": p, "mib": m} for p, m in before],
                                             "others_during": cotenants},
                               "gpu_shared": shared if watch.available else None})
            print(f"saved {row['id']} {args.label} {c['res']} seconds={seconds:.3f} "
                  f"peak_gb={(torch.cuda.max_memory_allocated() / 1e9) if cuda else 0:.1f} "
                  f"shared={shared if watch.available else 'n/a'}", flush=True)
    finally:
        watch.stop()
        area.remove()


if __name__ == "__main__":
    main()
