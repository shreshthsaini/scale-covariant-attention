#!/usr/bin/env python3
"""Generate one resumable UltraImage (official code, FLUX branch) campaign chunk with FLUX.1-dev.

Thin wrapper around the official UltraImage checkout (thu-ml/DiT-Extrapolation, directory flux/)
that keeps the CLI, seeding, file naming and PNG + JSON sidecar format of
constancy_rope/scripts/sample.py and scripts/run_area.py, so constancy_rope/scripts/eval.py reads
the outputs as method "ultraimage". Settings follow flux/infer_unified.py with flux/base.yaml:
stage 1 at the 1024 training resolution (30 steps, guidance 3.5, time_shift_1 3), stage 2 at the
target resolution (30 steps, guidance 7, time_shift_2 from the yaml), guidance_schedule super_imax,
swin patchify, proportional attention, RIFLEx RoPE on both spatial axes with linear_factor = res/2048
and train_length = (1024/16)^2 + 512, and the attention_decay operator. Before each prompt a
2 + 2 step "get_block_avg" pass writes 57 block-average files (one per FLUX block) that
attention_decay then reads; they live under ULTRA_BLOCKAVG_ROOT/<HxW>/<stem>/ on scratch.

Deviations from infer_unified.py, all recorded in the sidecar under "ultraimage":
  * seeding: infer_unified seeds torch once with 42 and passes torch.Generator("cuda").manual_seed(0)
    to every call. Here every call gets torch.Generator("cpu").manual_seed(derived_seed) and the
    global torch RNG (used by the VAE latent_dist.sample() of super_imax) is reseeded with
    derived_seed before each call, so each prompt is reproducible independent of chunking.
  * block averages are keyed by the image stem (prompt id + seed), not by the first 20 prompt
    characters, and a "_blockavg.json" marker must exist before the cache is reused.
  * only images[0] (the final image) is saved; the low-resolution guidance image is discarded.
Compatibility shim: the pipeline imports PyWavelets (pywt) for its level-1 Haar low-pass in
super_imax. If pywt is not installed, an exact numpy Haar level-1 replacement for pywt.wavedec2 /
pywt.waverec2 is registered in sys.modules; it matches PyWavelets 1.8.0 bitwise on this path. The
backend goes into the sidecar record (outside the resume fingerprint) as compat.dwt_backend. No
UltraImage file is edited.

Known upstream behaviour kept as is: infer_unified assigns scheduler.config.shift = time_shift_*,
which with diffusers >= 0.3x does not reach FlowMatchEulerDiscreteScheduler._shift, so both stages
run with the FLUX scheduler config shift (3.0); the emulated sigma schedules are recorded.

--dry-run parses arguments, loads prompts, imports the UltraImage modules on CPU, builds the full
settings and both sigma schedules and prints one sidecar config and the todo list without loading
model weights.
"""

import argparse
import copy
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
CROPE = Path("/work/09032/saini_2/vista/Projects/Exploration/constancy_rope")
ULTRA_DIR = Path(os.environ.get("ULTRA_DIR", "/scratch/09032/saini_2/constancy_rope/third_party/UltraImage/flux"))
BLOCKAVG_ROOT = Path(os.environ.get("ULTRA_BLOCKAVG_ROOT", "/scratch/09032/saini_2/sca/cache/ultraimage_blockavg"))
HF_HOME = "/scratch/09032/saini_2/tmp/hf"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
UPSTREAM = "https://github.com/thu-ml/DiT-Extrapolation (flux/)"
N_BLOCK_FILES = 57                # 19 double + 38 single FLUX blocks, one block-average file each
TXT_LEN = 512                     # pipeline default max_sequence_length
MARKER = "_blockavg.json"


def environment(allow_download=False):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(min(8, max(1, int(os.environ.get(name, "8")))))
    os.environ.setdefault("HF_HOME", HF_HOME)
    if not allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TORCH_HOME", "/scratch/09032/saini_2/tmp/torch")
    os.environ.setdefault("XDG_CACHE_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("TRITON_CACHE_DIR", "/scratch/09032/saini_2/tmp/triton")
    os.environ.setdefault("MPLCONFIGDIR", "/scratch/09032/saini_2/sca/cache/matplotlib")


# Path order: UltraImage first so its top-level modules (utils, rope, transformer_flux,
# attention_processor, pipeline_flux_unified, jintao) win; constancy_rope is only reached through
# the package name crope, which UltraImage does not define. check_shadowing() verifies this.
sys.path.insert(0, str(CROPE))
sys.path.insert(0, str(ULTRA_DIR))
from crope.utils import atomic_json, file_sha256, image_seed, image_stem, read_prompts, resolution  # noqa: E402

ULTRA_MODULES = ("utils", "rope.rope", "jintao", "attention_processor", "transformer_flux", "pipeline_flux_unified")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=resolution, required=True, metavar="HxW")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default="ultraimage", help="method name written to the sidecar")
    p.add_argument("--model-path", type=Path, default=None,
                   help="FLUX.1-dev diffusers directory; default the local HF snapshot")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-download", action="store_true", help="Allow missing FLUX cache files to download")
    p.add_argument("--dry-run", action="store_true", help="No model: print config and todo list, then exit")
    return p


def snapshot_dir():
    hub = Path(os.environ.get("HF_HOME", HF_HOME)) / "hub/models--black-forest-labs--FLUX.1-dev"
    revision = (hub / "refs/main").read_text().strip()
    return hub / "snapshots" / revision, revision


def git_revision(path):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL)
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "."], cwd=path,
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return out.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def implementation_hash():
    digest = hashlib.sha256()
    paths = [Path(__file__).resolve(), CROPE / "crope/utils.py", ULTRA_DIR / "base.yaml"]
    paths += sorted(ULTRA_DIR.rglob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(ULTRA_DIR) if ULTRA_DIR in path.parents else path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _stable(x):
    """Sort set-derived lists (diffusers' _use_default_values) so the digest is process independent."""
    if isinstance(x, dict):
        return {k: (sorted(v) if k == "_use_default_values" and isinstance(v, list) else _stable(v)) for k, v in x.items()}
    return [_stable(v) for v in x] if isinstance(x, list) else x


def fingerprint(config):
    # The SCA repo revision changes with unrelated commits; resume keys on everything else.
    stable = _stable({k: v for k, v in config.items() if k != "code_revision"})
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# Compatibility shims ---------------------------------------------------------------------------

def _envfix():
    """Serve zero-byte torch/triton modules lost to the scratch purge (repo file sca/envfix.py).

    Loaded from its file path under a private name so the sca package itself is not imported."""
    path = ROOT / "sca/envfix.py"
    if not path.is_file():
        return 0
    spec = importlib.util.spec_from_file_location("_sca_envfix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return len(module.install())


def _haar_shim():
    """Exact stand-in for pywt.wavedec2 / pywt.waverec2 with wavelet 'haar', level 1, even sizes.

    Mirrors PyWavelets' float arithmetic: the separable analysis is s*a + s*b along axis -2 then
    axis -1, synthesis of the approximation band alone is s*c along axis -1 then axis -2, with
    s = 1/sqrt(2) cast to the input dtype. These are the only calls pipeline_flux_unified makes
    (split_frequency_components_dwt, which zeroes every detail band)."""
    import numpy as np

    def _check(wavelet, level):
        if wavelet != "haar" or level != 1:
            raise NotImplementedError("pywt shim supports only wavelet='haar', level=1; install PyWavelets")

    def wavedec2(data, wavelet="haar", mode="symmetric", level=None, axes=(-2, -1)):
        _check(wavelet, 1 if level is None else level)
        x = np.asarray(data)
        if x.dtype not in (np.float32, np.float64):
            x = x.astype(np.float64)
        if x.shape[-2] % 2 or x.shape[-1] % 2 or tuple(axes) != (-2, -1):
            raise NotImplementedError("pywt shim needs even sizes on axes (-2, -1); install PyWavelets")
        s = x.dtype.type(0.7071067811865476)
        bands = {}
        a = x[..., 0::2, :] * s + x[..., 1::2, :] * s          # axis -2 low
        d = x[..., 0::2, :] * s - x[..., 1::2, :] * s          # axis -2 high
        bands["aa"] = a[..., 0::2] * s + a[..., 1::2] * s
        bands["da"] = d[..., 0::2] * s + d[..., 1::2] * s
        bands["ad"] = a[..., 0::2] * s - a[..., 1::2] * s
        bands["dd"] = d[..., 0::2] * s - d[..., 1::2] * s
        return [bands["aa"], (bands["da"], bands["ad"], bands["dd"])]

    def waverec2(coeffs, wavelet="haar", mode="symmetric", axes=(-2, -1)):
        _check(wavelet, len(coeffs) - 1)
        c = np.asarray(coeffs[0])
        if any(np.any(np.asarray(band)) for band in coeffs[1]):
            raise NotImplementedError("pywt shim reconstructs only from the approximation band")
        s = c.dtype.type(0.7071067811865476)
        r = np.repeat(c * s, 2, axis=-1)                       # axis -1 first, as pywt idwtn
        return np.repeat(r * s, 2, axis=-2)

    module = types.ModuleType("pywt")
    module.wavedec2, module.waverec2 = wavedec2, waverec2
    module.__version__ = "shim-haar-level1"
    module.__file__ = __file__
    return module


def install_pywt():
    try:
        import pywt
        return f"pywt {pywt.__version__}"
    except ImportError:
        sys.modules["pywt"] = _haar_shim()
        return "numpy haar level-1 shim (PyWavelets not installed)"


def check_shadowing(mods):
    for name, module in mods.items():
        path = Path(getattr(module, "__file__", "") or "")
        if ULTRA_DIR.resolve() not in path.resolve().parents:
            raise ImportError(f"UltraImage module {name} resolved to {path}, not under {ULTRA_DIR}")


def import_ultraimage():
    """Import the official modules exactly as infer_unified.py does (CPU safe, no weights)."""
    zero_byte = _envfix()
    dwt_backend = install_pywt()
    import importlib
    mods = {name: importlib.import_module(name) for name in ULTRA_MODULES}
    check_shadowing(mods)
    return mods, {"dwt_backend": dwt_backend, "envfix_missing_modules": zero_byte}


# UltraImage settings ---------------------------------------------------------------------------

def load_yaml():
    import yaml
    return yaml.safe_load((ULTRA_DIR / "base.yaml").read_text())


def ultra_settings(res):
    """base.yaml method section plus the infer_unified.py call arguments for resolution res."""
    cfg = load_yaml()
    base, method = cfg["base"], copy.deepcopy(cfg["method"])
    h, w = res
    th, tw = base["train_resolution"]
    method["hyper_parameters_height"]["linear_factor"] = h / 2048
    method["hyper_parameters_width"]["linear_factor"] = w / 2048
    method["hyper_parameters_height"]["train_length"] = (th // 16) * (tw // 16) + 512
    method["hyper_parameters_width"]["train_length"] = (th // 16) * (tw // 16) + 512
    method["save_dir"] = str(BLOCKAVG_ROOT / f"{h}x{w}_v2")  # v1 caches were built at the 1024 default
    call = {"height": h, "width": w,
            "num_inference_steps1": 30, "num_inference_steps2": 30,
            "guidance_scale1": 3.5, "guidance_scale2": 7,
            "ntk_factor": method.get("ntk_factor", 10),
            "time_shift_1": 3, "time_shift_2": method.get("time_shift_2", 6),
            "proportional_attention": method.get("proportional_attention", True),
            "text_duplication": method.get("text_duplication", True),
            "swin_pachify": True,
            "guidance_schedule": method.get("guidance_schedule", "cosine_decay"),
            "sdedit_threshold": method.get("sdedit_threshold", 20),
            "native_resolution": th,
            "start_croords": method.get("start_croords", [0.5, 0.5]),
            "max_sequence_length": TXT_LEN}
    block_call = dict(call, num_inference_steps1=2, num_inference_steps2=2)
    return {"method": method, "call": call, "block_avg_call": block_call, "train_resolution": [th, tw],
            "yaml_seed_unused": base.get("seed")}


def check_geometry(res, native):
    """super_imax unpacks the native-resolution latents at (H, W) // round(sqrt(H W) / native);
    anything other than (native, native) fails only after the 30 low-resolution steps."""
    import math
    h, w = res
    k = int(math.sqrt(h * w / native ** 2) + 0.5)
    if k < 1 or (h // k, w // k) != (native, native):
        raise ValueError(f"UltraImage super_imax needs H/k = W/k = {native} with k = round(sqrt(HW)/{native}); "
                         f"{h}x{w} gives {h // max(k, 1)}x{w // max(k, 1)}")


def emulate_schedule(mods, scheduler_dir, res, call):
    """Sigma schedules the pipeline's own scheduler calls produce for both stages (CPU)."""
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    pf = mods["pipeline_flux_unified"]
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_dir), subfolder="scheduler")
    sched.config.use_dynamic_shifting = False          # infer_unified.py
    out = {}
    stages = ((1, call["native_resolution"] ** 2 // 256, call["time_shift_1"]),
              (2, (res[0] // 16) * (res[1] // 16), call["time_shift_2"]))
    for stage, seq_len, shift in stages:
        n = call[f"num_inference_steps{stage}"]
        sigmas = np.linspace(1.0, 1 / n, n)
        sched.config.shift = shift                     # pipeline assignment; see module docstring
        mu = pf.calculate_shift(seq_len, sched.config.base_image_seq_len, sched.config.max_image_seq_len,
                                sched.config.base_shift, sched.config.max_shift)
        pf.retrieve_timesteps(sched, n, "cpu", None, sigmas, mu=mu)
        out[f"stage{stage}"] = {"image_seq_len": seq_len, "requested_shift": shift,
                                "effective_shift": float(sched.shift),
                                "sigmas": [round(float(s), 7) for s in sched.sigmas]}
    return out


class UltraRunner:
    def __init__(self, args, dry_run=False):
        self.mods, self.compat = import_ultraimage()
        import diffusers
        import torch
        import transformers
        self.args, self.res = args, tuple(args.res)
        self.settings = ultra_settings(self.res)
        check_geometry(self.res, self.settings["call"]["native_resolution"])
        self.pipe, self.calls = None, 0
        if args.model_path is not None:
            source, self.model_revision = args.model_path, None
            if source.parent.name == "snapshots":
                self.model_revision = source.name
        else:
            try:
                source, self.model_revision = snapshot_dir()
            except OSError:
                source, self.model_revision = MODEL_ID, None
        self.source = source
        self.schedule = emulate_schedule(self.mods, source, self.res, self.settings["call"])
        self.versions = {"diffusers": diffusers.__version__, "torch": torch.__version__,
                         "transformers": transformers.__version__}
        self.base = {"code_revision": git_revision(ROOT), "ultraimage_revision": git_revision(ULTRA_DIR),
                     "implementation_sha256": implementation_hash()}
        if dry_run:
            return
        self.pipe = self.load(source)

    def load(self, source):
        import torch
        args = self.args
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for UltraImage (jintao Triton kernels); use --dry-run on CPU nodes")
        if not args.device.startswith("cuda"):
            raise RuntimeError("UltraImage attention_decay asserts CUDA tensors; CPU inference is not supported")
        torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))
        if args.allow_download:
            source = MODEL_ID
        local = not args.allow_download
        FluxPipeline = self.mods["pipeline_flux_unified"].FluxPipeline
        FluxTransformer2DModel = self.mods["transformer_flux"].FluxTransformer2DModel
        forward_rope = self.mods["rope.rope"].forward_rope
        # Same construction as infer_unified.py.
        transformer = FluxTransformer2DModel.from_pretrained(str(source), subfolder="transformer",
                                                             torch_dtype=torch.bfloat16, local_files_only=local)
        pipe = FluxPipeline.from_pretrained(str(source), transformer=None, torch_dtype=torch.bfloat16,
                                            local_files_only=local)
        pipe.transformer = transformer
        self.rope_method = copy.deepcopy(self.settings["method"])
        pipe.transformer.pos_embed.forward = partial(forward_rope, pipe.transformer.pos_embed, method=self.rope_method)
        pipe.scheduler.config.use_dynamic_shifting = False
        pipe.to(args.device)
        transformer.to(args.device)                    # in case the late assignment was not registered
        return pipe

    def config(self, row, seed):
        s = self.settings
        method = copy.deepcopy(s["method"])
        return {"schema_version": 1, "method": self.args.label, "res": f"{self.res[0]}x{self.res[1]}",
                "prompt_id": str(row["id"]), "prompt": row["prompt"], "seed": int(seed),
                "derived_seed": image_seed(row["id"], seed),
                "steps": s["call"]["num_inference_steps1"] + s["call"]["num_inference_steps2"],
                "guidance": s["call"]["guidance_scale2"],
                "model": MODEL_ID, "model_revision": self.model_revision, "dtype": "bfloat16", "batch_size": 1,
                "attention": "ultraimage_official (attention_decay: jintao int8 Triton sage_attn with entropy "
                             "decay mask; block averages from a get_block_avg pass)",
                **self.versions, **self.base, "grid": [n // 16 for n in self.res],
                "ultraimage": {"repo": UPSTREAM, "entry": "flux/infer_unified.py + flux/base.yaml",
                               "method": method, "call": s["call"], "block_avg_call": s["block_avg_call"],
                               "block_avg_files": N_BLOCK_FILES, "prompt_low": "prompt", "prompt_high": "prompt",
                               "train_resolution": s["train_resolution"],
                               "use_dynamic_shifting": False, "schedule": self.schedule,
                               "generator": "torch.Generator('cpu').manual_seed(derived_seed) per call; "
                                            "torch.manual_seed(derived_seed) before each call",
                               "upstream_generator": "torch.Generator('cuda').manual_seed(0); torch.manual_seed(42)",
                               "saved_outputs": "images[0] only (guidance image discarded)"},
                "vae_tiling": "pipeline tiled_encode/tiled_decode", "max_sequence_length": TXT_LEN,
                "generator": "cpu, derived_seed", "cpu_offload": False}

    def block_avg_ready(self, path, derived):
        marker = path / MARKER
        if not marker.is_file():
            return False
        saved = json.loads(marker.read_text())
        files = len(list(path.glob("*.pt")))
        return saved.get("complete") is True and saved.get("derived_seed") == derived and files >= N_BLOCK_FILES

    def generate(self, row, seed):
        import torch
        s = self.settings
        derived = image_seed(row["id"], seed)
        save_path = Path(s["method"]["save_dir"]) / image_stem(row["id"], seed)
        method = copy.deepcopy(s["method"])
        method["save_path"] = str(save_path)
        common = {k: v for k, v in s["call"].items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        block_seconds, cached = None, self.block_avg_ready(save_path, derived)
        if not cached:
            save_path.mkdir(parents=True, exist_ok=True)
            for stale in save_path.glob("*.pt"):
                stale.unlink()
            (save_path / MARKER).unlink(missing_ok=True)
            method["attention_operator"]["attn_func"] = "get_block_avg"
            method["num_steps"] = 2                    # infer_unified sets this (unused downstream)
            torch.manual_seed(derived)
            with torch.inference_mode():
                out = self.pipe(prompt=row["prompt"], prompt_high=row["prompt"],
                                generator=torch.Generator(device="cpu").manual_seed(derived),
                                return_dict=False, hyper_params=method, method=method, **s["block_avg_call"])
                del out
            method["attention_operator"]["attn_func"] = "attention_decay"
            torch.cuda.synchronize()
            block_seconds = time.perf_counter() - start
            files = len(list(save_path.glob("*.pt")))
            if files < N_BLOCK_FILES:
                raise RuntimeError(f"get_block_avg wrote {files} < {N_BLOCK_FILES} files in {save_path}")
            atomic_json(save_path / MARKER, {"complete": True, "derived_seed": derived, "files": files,
                                             "res": f"{self.res[0]}x{self.res[1]}", "prompt_id": str(row["id"]),
                                             "seconds": block_seconds})
        main_start = time.perf_counter()
        torch.manual_seed(derived)
        with torch.inference_mode():
            images = self.pipe(prompt=row["prompt"], prompt_high=row["prompt"],
                               generator=torch.Generator(device="cpu").manual_seed(derived),
                               return_dict=False, hyper_params=method, method=method, **common)
        torch.cuda.synchronize()
        end = time.perf_counter()
        timing = {"seconds": end - start, "seconds_main": end - main_start, "seconds_block_avg": block_seconds,
                  "block_avg_cached": cached, "block_avg_dir": str(save_path)}
        sigmas = [float(x) for x in self.pipe.scheduler.sigmas]
        return images[0], timing, torch.cuda.max_memory_allocated(), sigmas

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
                print(f"skip {row['id']} {config['method']} {config['res']}", flush=True)
                return saved
        first = self.calls == 0
        image, timing, peak_memory, sigmas = self.generate(row, seed)
        self.calls += 1
        if image.size != self.res[::-1]:
            raise RuntimeError(f"UltraImage returned {image.size}, expected {self.res[::-1]}")
        out.mkdir(parents=True, exist_ok=True)
        temporary = png.with_name(png.name + f".tmp.{os.getpid()}")
        image.save(temporary, format="PNG")
        os.replace(temporary, png)
        record = {"complete": True, "config": config, "config_fingerprint": digest,
                  "image": str(png.resolve()), "png_sha256": file_sha256(png), **timing, "compat": self.compat,
                  "seconds_scope": "get_block_avg pass (when not cached) plus the two-stage UltraImage call: "
                                   "prompt encoding, both denoising stages, VAE decode/encode/decode; "
                                   "excludes model loading and PNG I/O",
                  "first_call_in_process": first, "peak_memory_bytes": peak_memory, "sigmas": sigmas}
        atomic_json(metadata, record)
        print(f"saved {row['id']} {config['method']} {config['res']} seconds={timing['seconds']:.3f} "
              f"block_avg={'cached' if timing['block_avg_cached'] else round(timing['seconds_block_avg'], 3)}",
              flush=True)
        return record


def main(argv=None):
    args = parser().parse_args(argv)
    environment(args.allow_download)
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    if not 0 <= args.start < end <= len(prompts):
        raise ValueError("Require 0 <= start < end <= number of prompts")
    rows = prompts[args.start:end]
    if args.dry_run:
        runner = UltraRunner(args, dry_run=True)
        config = runner.config(rows[0], args.seed)
        print(json.dumps(config, indent=2))
        for row in rows:
            stem = image_stem(row["id"], args.seed)
            done = (args.out / f"{stem}.png").exists() and (args.out / f"{stem}.json").exists()
            print(f"{'done' if done else 'todo'} {row['id']} derived_seed={image_seed(row['id'], args.seed)} "
                  f"{args.out / (stem + '.png')} block_avg={Path(runner.settings['method']['save_dir']) / stem}")
        print(f"dry-run ok: {len(rows)} prompts, fingerprint {fingerprint(config)[:12]}, "
              f"dwt {runner.compat['dwt_backend']}")
        return
    runner = UltraRunner(args)
    for row in rows:
        runner.sample(row, args.out, args.seed)


if __name__ == "__main__":
    main()
