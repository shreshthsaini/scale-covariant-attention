#!/usr/bin/env python3
"""Generate one resumable AReA (official code) campaign chunk with FLUX.1-dev.

Thin wrapper around the official AReA checkout (github.com/KooroshRH/AReA) that keeps the
CLI, seeding, file naming and PNG + JSON sidecar format of
constancy_rope/scripts/sample.py, so constancy_rope/scripts/eval.py reads the outputs as
method "area". AReA defaults follow its generate.py for model family flux1: 28 steps,
guidance 4.5, Phase I 8 steps, subsample pooling, Halo 8, bridge ratio 0.5, AReA seed
equal to the run seed, VAE tiling always on, scheduler base_shift = max_shift (mu = 1.15).
The checkpoint is the local FLUX.1-dev snapshot instead of AReA's FLUX.1-Krea-dev default.

--dry-run parses arguments, loads prompts, builds the AReA config and the per-step
schedule and prints one sidecar config without importing a model; it runs on CPU.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CROPE = Path("/work/09032/saini_2/vista/Projects/Exploration/constancy_rope")
AREA_DIR = Path(os.environ.get("AREA_DIR", "/scratch/09032/saini_2/sca/third_party/AReA"))
HF_HOME = "/scratch/09032/saini_2/tmp/hf"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
TXT_LEN = 512                     # FLUX T5 max_sequence_length, the text token count AReA prices
OFFLOAD_BELOW_GB = 60             # AReA generate.py offloads below this much GPU memory


def environment():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(min(8, max(1, int(os.environ.get(name, "8")))))
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("TORCH_HOME", "/scratch/09032/saini_2/tmp/torch")
    os.environ.setdefault("XDG_CACHE_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("MPLCONFIGDIR", "/scratch/09032/saini_2/sca/cache/matplotlib")


sys.path.insert(0, str(CROPE))      # identical seeding, naming and prompt parsing to sample.py
sys.path.insert(0, str(AREA_DIR))   # official AReA package
from crope.utils import atomic_json, file_sha256, image_seed, image_stem, read_prompts, resolution  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=resolution, required=True, metavar="HxW")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=28, help="AReA flux1 default 28")
    p.add_argument("--guidance", type=float, default=4.5,
                   help="AReA flux1 default 4.5 (constancy_rope sample.py uses 3.5)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pooling", default="subsample", choices=("subsample", "mean", "medoid", "random"))
    p.add_argument("--phase1-steps", type=int, default=None, help="Phase I steps; default AReA flux1 value 8")
    p.add_argument("--halo", type=int, default=8, help="delta_max in patches")
    p.add_argument("--bridge-ratio", type=float, default=0.5, help="rho")
    p.add_argument("--area-seed", type=int, default=None,
                   help="AReA jitter/Halo seed; default --seed, as in AReA generate.py")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-offload", action="store_true",
                   help=f"Model CPU offload; also enabled automatically below {OFFLOAD_BELOW_GB} GB like AReA")
    p.add_argument("--allow-download", action="store_true", help="Allow missing FLUX cache files to download")
    p.add_argument("--dry-run", action="store_true", help="No model: print config and schedule, then exit")
    return p


def snapshot_dir():
    hub = Path(os.environ.get("HF_HOME", HF_HOME)) / "hub/models--black-forest-labs--FLUX.1-dev"
    revision = (hub / "refs/main").read_text().strip()
    return hub / "snapshots" / revision, revision


def git_revision(path):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL)
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path,
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return out.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def implementation_hash():
    digest = hashlib.sha256()
    paths = [Path(__file__).resolve(), CROPE / "crope/utils.py"]
    paths += sorted((AREA_DIR / "area").rglob("*.py"))
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprint(config):
    # The SCA repo revision changes with unrelated commits; resume keys on everything else.
    stable = {k: v for k, v in config.items() if k != "code_revision"}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def area_config(args):
    from area import AReAConfig
    return AReAConfig(phase1_steps=args.phase1_steps, pooling=args.pooling, halo=args.halo,
                      bridge_ratio=args.bridge_ratio, seed=args.seed if args.area_seed is None else args.area_seed)


def area_schedule(res, steps, cfg):
    from area.backbones.flux import NATIVE_PATCHES, PHASE1_STEPS
    from area.schedule import build_schedule
    h, w = res[0] // 16, res[1] // 16
    if max(h, w) <= NATIVE_PATCHES:
        return {"phase1_steps": 0, "factors": [], "windows": [], "note": "native or below: plain attention"}
    phase1 = PHASE1_STEPS if cfg.phase1_steps is None else cfg.phase1_steps
    s = build_schedule(h, w, steps, phase1, NATIVE_PATCHES, txt_len=TXT_LEN)
    return {"phase1_steps": phase1, "native_patches": NATIVE_PATCHES, "txt_len": TXT_LEN,
            "factors": list(s.factors), "windows": list(s.windows)}


class AReARunner:
    def __init__(self, args, dry_run=False):
        import diffusers
        import torch
        self.args, self.res = args, tuple(args.res)
        self.cfg = area_config(args)
        self.schedule = area_schedule(self.res, args.steps, self.cfg)
        self.pipe, self.area, self.offload, self.calls = None, None, False, 0
        try:
            source, self.model_revision = snapshot_dir()
        except OSError:
            source, self.model_revision = MODEL_ID, None
        self.versions = {"diffusers": diffusers.__version__, "torch": torch.__version__}
        self.base = {"code_revision": git_revision(ROOT), "area_revision": git_revision(AREA_DIR),
                     "implementation_sha256": implementation_hash()}
        if dry_run:
            self.scheduler_config = None
            return
        self.pipe = self.load(source)
        from area import AReA
        self.area = AReA(self.pipe, config=self.cfg)
        self.scheduler_config = {k: v for k, v in dict(self.pipe.scheduler.config).items()}

    def load(self, source):
        import torch
        from diffusers import FluxPipeline
        args = self.args
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for FLUX inference; use --dry-run on CPU nodes")
        if not args.device.startswith("cuda") and os.environ.get("CROPE_ALLOW_CPU") != "1":
            raise RuntimeError("Full FLUX inference on CPU is disabled (set CROPE_ALLOW_CPU=1 for a plumbing check)")
        torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))
        if args.allow_download:
            source = MODEL_ID
        pipe = FluxPipeline.from_pretrained(str(source), torch_dtype=torch.bfloat16,
                                            local_files_only=not args.allow_download)
        pipe.vae.enable_tiling()                        # AReA generate.py: always tiled VAE decode
        device = torch.device(args.device)
        small = device.type == "cuda" and \
            torch.cuda.get_device_properties(device).total_memory / 1024 ** 3 < OFFLOAD_BELOW_GB
        self.offload = bool(args.cpu_offload or small)
        if self.offload:
            pipe.enable_model_cpu_offload(device=args.device)
        else:
            pipe.to(args.device)
        return pipe

    def config(self, row, seed):
        a = self.args
        return {"schema_version": 1, "method": "area", "res": f"{self.res[0]}x{self.res[1]}",
                "prompt_id": str(row["id"]), "prompt": row["prompt"], "seed": int(seed),
                "derived_seed": image_seed(row["id"], seed), "steps": a.steps, "guidance": a.guidance,
                "model": MODEL_ID, "model_revision": self.model_revision, "dtype": "bfloat16", "batch_size": 1,
                "attention": "area_official (diffusers dispatch_attention_fn, default backend)",
                **self.versions, **self.base, "grid": [n // 16 for n in self.res],
                "area": {"pooling": self.cfg.pooling, "phase1_steps": self.schedule.get("phase1_steps"),
                         "halo": self.cfg.halo, "bridge_ratio": self.cfg.bridge_ratio, "seed": self.cfg.seed,
                         "schedule": self.schedule, "repo": "https://github.com/KooroshRH/AReA"},
                "vae_tiling": True, "max_sequence_length": TXT_LEN, "generator": "cpu, derived_seed",
                "cpu_offload": self.offload, "scheduler": self.scheduler_config}

    def generate(self, row, seed):
        import torch
        h, w = self.res
        generator = torch.Generator(device="cpu").manual_seed(image_seed(row["id"], seed))
        cuda = torch.cuda.is_available() and str(self.pipe._execution_device).startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.pipe(prompt=row["prompt"], height=h, width=w, num_inference_steps=self.args.steps,
                               guidance_scale=self.args.guidance, generator=generator,
                               max_sequence_length=TXT_LEN, output_type="pil")
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        sigmas = [float(s) for s in self.pipe.scheduler.sigmas]
        return result.images[0], elapsed, torch.cuda.max_memory_allocated() if cuda else None, sigmas

    def sample(self, row, out, seed):
        from PIL import Image
        out = Path(out)
        stem = image_stem(row["id"], seed)
        png, metadata = out / f"{stem}.png", out / f"{stem}.json"
        config = self.config(row, seed)
        digest = fingerprint(config)
        if png.exists() and metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved.get("complete"):
                if saved.get("config_fingerprint") != digest:
                    raise ValueError(f"Existing result has different settings: {metadata}. Choose a new --out.")
                with Image.open(png) as im:
                    if im.size != self.res[::-1]:
                        raise ValueError(f"Existing image has incorrect dimensions: {png}")
                    im.verify()
                if file_sha256(png) != saved["png_sha256"]:
                    raise ValueError(f"Existing PNG checksum differs: {png}")
                print(f"skip {row['id']} area {config['res']}", flush=True)
                return saved
        first = self.calls == 0
        image, seconds, peak_memory, sigmas = self.generate(row, seed)
        self.calls += 1
        out.mkdir(parents=True, exist_ok=True)
        temporary = png.with_name(png.name + f".tmp.{os.getpid()}")
        image.save(temporary, format="PNG")
        os.replace(temporary, png)
        record = {"complete": True, "config": config, "config_fingerprint": digest,
                  "image": str(png.resolve()), "png_sha256": file_sha256(png), "seconds": seconds,
                  "seconds_scope": "prompt encoding, denoising, and VAE decode; excludes model loading and PNG I/O",
                  "first_call_in_process": first, "peak_memory_bytes": peak_memory, "sigmas": sigmas}
        atomic_json(metadata, record)
        print(f"saved {row['id']} area {config['res']} seconds={seconds:.3f}", flush=True)
        return record

    def close(self):
        if self.area is not None:
            self.area.remove()


def main(argv=None):
    args = parser().parse_args(argv)
    environment()
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    if not 0 <= args.start < end <= len(prompts) or args.steps <= 0:
        raise ValueError("Require 0 <= start < end <= number of prompts and positive steps")
    rows = prompts[args.start:end]
    if args.dry_run:
        runner = AReARunner(args, dry_run=True)
        config = runner.config(rows[0], args.seed)
        print(json.dumps(config, indent=2))
        for row in rows:
            stem = image_stem(row["id"], args.seed)
            done = (args.out / f"{stem}.png").exists() and (args.out / f"{stem}.json").exists()
            print(f"{'done' if done else 'todo'} {row['id']} derived_seed={image_seed(row['id'], args.seed)} "
                  f"{args.out / (stem + '.png')}")
        print(f"dry-run ok: {len(rows)} prompts, fingerprint {fingerprint(config)[:12]}")
        return
    runner = AReARunner(args)
    try:
        for row in rows:
            runner.sample(row, args.out, args.seed)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
