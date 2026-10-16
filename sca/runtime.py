"""FLUX loading, seeding and record helpers. Seeding, stems and sidecars follow
constancy_rope (crope/utils.py, crope/runtime.py) so its eval.py reads our outputs unchanged."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import torch

HF_HOME = "/scratch/09032/saini_2/tmp/hf"
SCRATCH = Path("/scratch/09032/saini_2/sca")


def runtime_environment():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(min(8, max(1, int(os.environ.get(name, "8")))))
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("TORCH_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("WANDB_DIR", str(SCRATCH / "wandb"))


def resolution(value):
    """HxW string (constancy_rope convention), multiples of 16."""
    m = re.fullmatch(r"(\d+)[xX](\d+)", value)
    if m is None:
        raise ValueError("Resolution must be HxW, for example 2048x4096")
    h, w = map(int, m.groups())
    if min(h, w) < 64 or h % 16 or w % 16:
        raise ValueError("Both dimensions must be multiples of 16 and at least 64")
    return h, w


def read_prompts(path):
    rows, seen = [], set()
    with open(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key = str(row["id"])
                if key in seen or not str(row.get("prompt", "")).strip():
                    raise ValueError(f"Duplicate ID or invalid prompt: {key}")
                seen.add(key)
                rows.append({**row, "id": key})
    return rows


def image_seed(prompt_id, seed):
    payload = json.dumps([str(prompt_id), int(seed)], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def image_stem(prompt_id, seed):
    key = str(prompt_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80].strip(".") or "prompt"
    return f"{safe}_{hashlib.sha256(key.encode()).hexdigest()[:10]}_seed{int(seed)}"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def file_sha256(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable(x):
    """Sort set-derived lists (diffusers' _use_default_values) so the digest is process independent."""
    if isinstance(x, dict):
        return {k: (sorted(v) if k == "_use_default_values" and isinstance(v, list) else _stable(v)) for k, v in x.items()}
    return [_stable(v) for v in x] if isinstance(x, list) else x


def fingerprint(config):
    return hashlib.sha256(json.dumps(_stable(config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(list((root / "sca").glob("*.py")) + list((root / "scripts").glob("*.py"))):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_pipeline(device="cuda"):
    from diffusers import FluxPipeline
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLUX inference; use pytest for CPU checks")
    hub = Path(os.environ.get("HF_HOME", HF_HOME)) / "hub/models--black-forest-labs--FLUX.1-dev"
    source = str(hub / "snapshots" / (hub / "refs/main").read_text().strip())
    pipe = FluxPipeline.from_pretrained(source, torch_dtype=torch.bfloat16, local_files_only=True)
    return pipe.to(device)


def set_mu(pipe, h, w, mu_mode):
    """max_shift: mu = max_shift at any resolution (upstream DyPE/SEGA, constancy_rope default).
    default: diffusers linear extrapolation of mu in sequence length (degenerate at 4K)."""
    seq = (h // 16) * (w // 16)
    if mu_mode == "max_shift":
        pipe.scheduler.register_to_config(max_image_seq_len=max(4096, seq))
    elif mu_mode == "default":
        pipe.scheduler.register_to_config(max_image_seq_len=4096)
    else:
        raise ValueError(mu_mode)
