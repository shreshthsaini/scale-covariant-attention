#!/usr/bin/env python3
"""AReA-style extra metrics for constancy_rope sample trees.

Metrics: ImageReward (ir), LAION improved aesthetic predictor (aesthetic),
HPSv2.0 (hpsv2), HPSv2.1 (hpsv21), CLIP-IQA (clipiqa), ARNIQA (arniqa),
Inception Score (is), and generation latency read from the sidecars (latency).
eval.py in constancy_rope covers clip, pickscore, musiq, fid, kid and patch
FID/KID; this script fills the rest of the AReA table.

Run it with the sca-metrics uv env, which pins transformers 4.x for
ImageReward and ships HPSv2, pyiqa and torch-fidelity:

  /work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python eval_extra.py \
      --root SAMPLES --prompts PROMPTS.jsonl --out OUT --device cuda

Sample discovery and the prompt-cluster bootstrap are imported from
constancy_rope (crope.metrics.io and crope.metrics.statistics), so image sets,
pairing rules and intervals match eval.py exactly.

Preprocessing follows each model's own pipeline:
  ir         ImageReward: bicubic resize 224, center crop, BLIP normalization.
  aesthetic  OpenAI CLIP ViT-L/14 transform (bicubic 224, center crop), L2
             normalized image embedding into sac+logos+ava1-l14-linearMSE MLP.
  hpsv2(1)   official HPSv2 val transform (longest side 224, pad), cosine of
             normalized image and text features, autocast on CUDA as upstream.
  clipiqa,   pyiqa defaults on the image after an antialiased bicubic
  arniqa     downscale so the long side is at most --iqa-long-side (1024 by
             default, the KonIQ scale both models were trained on; 0 = native).
  is         full image resized to 299x299 (PIL bicubic, antialiased) into the
             torch-fidelity inception-v3-compat logits; IS uses
             torch-fidelity's isc_features_to_metric (10 splits, shuffle 2020).

Per-image results are appended to OUT/extra_cache.jsonl, keyed by resolved
path, size, mtime_ns, prompt and metric settings, so reruns only compute what
is missing.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import csv
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

DEFAULT_CROPE = "/work/09032/saini_2/vista/Projects/Exploration/constancy_rope"
ALL_METRICS = ("ir", "aesthetic", "hpsv2", "hpsv21", "clipiqa", "arniqa", "is", "latency")
DEFAULT_METRICS = "ir,aesthetic,hpsv2,clipiqa,arniqa,is,latency"
AESTHETIC_WEIGHTS = "/scratch/09032/saini_2/tmp/torch/hub/checkpoints/laion_aesthetic/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Root containing sample PNG files and JSON sidecars")
    p.add_argument("--prompts", required=True, help="Prompt JSONL with id and prompt")
    p.add_argument("--out", required=True, help="Output directory; also holds the resumable cache")
    p.add_argument("--metrics", default=DEFAULT_METRICS, help=f"Comma list from {','.join(ALL_METRICS)} or 'all'")
    p.add_argument("--reference-method", default="sega_official")
    p.add_argument("--res", help="Optional resolution filter, for example 4096x4096")
    p.add_argument("--methods", help="Optional comma list of methods to keep")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=8, help="Decode and preprocessing threads (1 to 16)")
    p.add_argument("--iqa-long-side", type=int, default=1024, help="Downscale cap for CLIP-IQA and ARNIQA; 0 keeps native size")
    p.add_argument("--is-splits", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--crope-root", default=DEFAULT_CROPE, help="constancy_rope checkout providing crope.metrics")
    p.add_argument("--hf-home", default="/scratch/09032/saini_2/tmp/hf")
    p.add_argument("--torch-home", default="/scratch/09032/saini_2/tmp/torch")
    p.add_argument("--allow-download", action="store_true", help="Permit network downloads (default is offline)")
    p.add_argument("--allow-missing", action="store_true", help="Exit 0 even if a metric failed")
    return p


def setup_environment(args):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(args.threads))
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TORCH_HOME"] = args.torch_home
    os.environ.setdefault("XDG_CACHE_HOME", "/scratch/09032/saini_2/tmp/cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(Path(args.crope_root).resolve()))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def image_key(record) -> str:
    stat = Path(record["path"]).stat()
    return f"{record['path']}|{stat.st_size}|{stat.st_mtime_ns}|{digest(record['prompt'])}"


def to_tensor(image, uint8=False):
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
    return tensor if uint8 else tensor.float().div_(255.0)


def stack_by_shape(items):
    """Yield (indices, stacked tensor) for each distinct input shape."""
    import torch

    groups = defaultdict(list)
    for index, item in enumerate(items):
        groups[tuple(item.shape)].append(index)
    for indices in groups.values():
        yield indices, torch.stack([items[i] for i in indices])


# Scorers. prepare() runs in worker threads on a decoded RGB PIL image;
# run() runs on the main thread with a list of prepared inputs and records.

class ImageRewardScorer:
    name = "ir"

    def __init__(self, device):
        import ImageReward as reward
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download("THUDM/ImageReward", "ImageReward.pt")
        config = hf_hub_download("THUDM/ImageReward", "med_config.json")
        self.device = device
        self.model = reward.load(checkpoint, device=device, med_config=config).eval()
        self.settings = {"model": "THUDM/ImageReward v1.0", "checkpoint_sha": Path(checkpoint).parent.name,
                         "definition": "official normalized reward, batched copy of ImageReward.score"}

    def prepare(self, image, record):
        return self.model.preprocess(image)

    def run(self, inputs, records):
        import torch

        blip = self.model.blip
        out = [None] * len(inputs)
        with torch.inference_mode():
            for indices, images in stack_by_shape(inputs):
                text = blip.tokenizer([records[i]["prompt"] for i in indices], padding="max_length",
                                      truncation=True, max_length=35, return_tensors="pt").to(self.device)
                embeds = blip.visual_encoder(images.to(self.device))
                atts = torch.ones(embeds.size()[:-1], dtype=torch.long, device=self.device)
                hidden = blip.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                           encoder_hidden_states=embeds, encoder_attention_mask=atts,
                                           return_dict=True).last_hidden_state[:, 0, :].float()
                rewards = (self.model.mlp(hidden) - self.model.mean) / self.model.std
                for i, value in zip(indices, rewards.flatten().tolist()):
                    out[i] = {"ir": float(value)}
        return out


class AestheticScorer:
    name = "aesthetic"

    def __init__(self, device):
        import clip
        import torch
        from torch import nn
        from transformers import CLIPVisionModelWithProjection

        if sha256_file(AESTHETIC_WEIGHTS) != AESTHETIC_SHA256:
            raise RuntimeError(f"Unexpected checksum for {AESTHETIC_WEIGHTS}")
        self.device = device
        self.preprocess = clip.clip._transform(224)
        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
        self.mlp = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))
        state = torch.load(AESTHETIC_WEIGHTS, map_location="cpu", weights_only=True)
        self.mlp.load_state_dict({k.removeprefix("layers."): v for k, v in state.items()})
        self.mlp.eval().to(device)
        self.settings = {"model": "LAION improved-aesthetic-predictor sac+logos+ava1-l14-linearMSE",
                         "backbone": "openai/clip-vit-large-patch14", "sha256": AESTHETIC_SHA256,
                         "revision": getattr(self.clip.config, "_commit_hash", None)}

    def prepare(self, image, record):
        return self.preprocess(image)

    def run(self, inputs, records):
        import torch

        out = [None] * len(inputs)
        with torch.inference_mode():
            for indices, images in stack_by_shape(inputs):
                embeds = self.clip(pixel_values=images.to(self.device)).image_embeds.float()
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                for i, value in zip(indices, self.mlp(embeds).flatten().tolist()):
                    out[i] = {"aesthetic": float(value)}
        return out


class HPSv2Scorer:
    def __init__(self, device, version):
        import torch
        from huggingface_hub import hf_hub_download
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        filename = {"hpsv2": "HPS_v2_compressed.pt", "hpsv21": "HPS_v2.1_compressed.pt"}[version]
        self.name, self.device = version, device
        checkpoint = hf_hub_download("xswu/HPSv2", filename)
        # pretrained=None skips the redundant LAION ViT-H download; the HPS
        # checkpoint holds every weight and is loaded strictly below.
        model, _, self.preprocess = create_model_and_transforms(
            "ViT-H-14", None, precision="amp", device=device, jit=False, force_quick_gelu=False,
            force_custom_text=False, force_patch_dropout=False, force_image_size=None, pretrained_image=False,
            image_mean=None, image_std=None, light_augmentation=True, aug_cfg={}, output_dict=True,
            with_score_predictor=False, with_region_predictor=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        self.model = model.to(device).eval()
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.settings = {"model": f"xswu/HPSv2 {filename}", "checkpoint_sha": Path(checkpoint).parent.name,
                         "definition": "diag(image_features @ text_features.T), official val transform"}

    def prepare(self, image, record):
        return self.preprocess(image)

    def run(self, inputs, records):
        import torch

        out = [None] * len(inputs)
        cuda = str(self.device).startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", enabled=cuda):
            for indices, images in stack_by_shape(inputs):
                text = self.tokenizer([records[i]["prompt"] for i in indices]).to(self.device)
                result = self.model(images.to(self.device), text)
                scores = (result["image_features"] * result["text_features"]).sum(-1).float()
                for i, value in zip(indices, scores.tolist()):
                    out[i] = {self.name: float(value)}
        return out


class PyiqaScorer:
    def __init__(self, device, name, long_side):
        import pyiqa

        self.name, self.device, self.long_side = name, device, long_side
        self.model = pyiqa.create_metric(name, device=device).eval()
        self.settings = {"model": f"pyiqa {pyiqa.__version__} {name}", "long_side": long_side,
                         "resize": "PIL bicubic antialiased, aspect kept" if long_side else "native"}

    def prepare(self, image, record):
        from PIL import Image

        if self.long_side and max(image.size) > self.long_side:
            scale = self.long_side / max(image.size)
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.BICUBIC)
        return to_tensor(image)

    def run(self, inputs, records):
        import torch

        out = [None] * len(inputs)
        with torch.inference_mode():
            for indices, images in stack_by_shape(inputs):
                scores = self.model(images.to(self.device)).reshape(len(indices), -1).mean(1).float()
                for i, value in zip(indices, scores.tolist()):
                    out[i] = {self.name: float(value)}
        return out


class InceptionLogits:
    name = "is"

    def __init__(self, device):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        self.device = device
        self.model = FeatureExtractorInceptionV3("inception-v3-compat", ["logits_unbiased"]).eval().to(device)
        self.settings = {"model": "torch-fidelity inception-v3-compat logits_unbiased",
                         "resize": "PIL bicubic antialiased to 299x299"}

    def prepare(self, image, record):
        from PIL import Image

        return to_tensor(image.resize((299, 299), Image.Resampling.BICUBIC), uint8=True)

    def run(self, inputs, records):
        import numpy as np
        import torch

        out = [None] * len(inputs)
        with torch.inference_mode():
            for indices, images in stack_by_shape(inputs):
                logits = self.model(images.to(self.device))[0].float().cpu().numpy().astype(np.float32)
                for i, row in zip(indices, logits):
                    out[i] = {"is_logits_b64": base64.b64encode(row.tobytes()).decode()}
        return out


def build_scorer(metric, args):
    if metric == "ir":
        return ImageRewardScorer(args.device)
    if metric == "aesthetic":
        return AestheticScorer(args.device)
    if metric in ("hpsv2", "hpsv21"):
        return HPSv2Scorer(args.device, metric)
    if metric in ("clipiqa", "arniqa"):
        return PyiqaScorer(args.device, metric, args.iqa_long_side)
    if metric == "is":
        return InceptionLogits(args.device)
    raise ValueError(metric)


class Cache:
    """Append-only JSONL; the last valid line for a key wins, torn lines are skipped."""

    def __init__(self, path: Path):
        self.path, self.data = path, {}
        if path.exists():
            with path.open() as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        self.data[(row["key"], row["metric"], row["settings"])] = row
                    except (json.JSONDecodeError, KeyError):
                        continue
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a")
        if path.stat().st_size:  # terminate a torn final line so new rows start clean
            with path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    self.handle.write("\n")
                    self.handle.flush()

    def get(self, key, metric, settings):
        return self.data.get((key, metric, settings))

    def put(self, rows):
        for row in rows:
            self.data[(row["key"], row["metric"], row["settings"])] = row
            self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())


def score_images(records, metrics, args, cache, errors, provenance):
    """One decode per image feeds every metric still missing for it."""
    from PIL import Image

    keys = {r["path"]: image_key(r) for r in records}
    scorers = {}
    for metric in metrics:
        try:
            started = time.perf_counter()
            scorers[metric] = build_scorer(metric, args)
            provenance[metric] = {**scorers[metric].settings, "load_seconds": time.perf_counter() - started}
            print(f"loaded {metric} in {time.perf_counter() - started:.1f}s", flush=True)
        except Exception as exc:  # report and continue with the remaining metrics
            errors.append({"metric": metric, "stage": "load", "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED to load {metric}: {exc}", flush=True)
    settings = {m: digest(s.settings) for m, s in scorers.items()}
    todo = [(r, [m for m in scorers if cache.get(keys[r["path"]], m, settings[m]) is None]) for r in records]
    todo = sorted([t for t in todo if t[1]], key=lambda t: (t[0]["res"], t[0]["method"], t[0]["path"]))
    counts = {m: sum(m in needed for _, needed in todo) for m in scorers}
    print(f"{len(todo)} of {len(records)} images need scoring; missing per metric {counts}", flush=True)
    if not todo:
        return settings

    def prepare(item):
        record, needed = item
        started = time.perf_counter()
        with Image.open(record["path"]) as source:
            image = source.convert("RGB")
        prepared = {m: scorers[m].prepare(image, record) for m in needed}
        return prepared, time.perf_counter() - started

    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    done, started_all = 0, time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        pending = [pool.submit(prepare, item) for item in batches[0]]
        for b, batch in enumerate(batches):
            prepared = [f.result() for f in pending]
            pending = [pool.submit(prepare, item) for item in batches[b + 1]] if b + 1 < len(batches) else []
            rows = []
            for metric, scorer in scorers.items():
                members = [i for i, (_, needed) in enumerate(batch) if metric in needed]
                if not members:
                    continue
                started = time.perf_counter()
                try:
                    scores = scorer.run([prepared[i][0][metric] for i in members], [batch[i][0] for i in members])
                    if str(args.device).startswith("cuda"):
                        import torch
                        torch.cuda.synchronize()
                except Exception as exc:
                    errors.append({"metric": metric, "stage": "score", "batch": b,
                                   "error": f"{type(exc).__name__}: {exc}"})
                    print(f"FAILED {metric} batch {b}: {exc}", flush=True)
                    continue
                per_image = (time.perf_counter() - started) / len(members)
                for i, score in zip(members, scores):
                    rows.append({"key": keys[batch[i][0]["path"]], "metric": metric, "settings": settings[metric],
                                 "scores": score, "seconds": per_image, "decode_seconds": prepared[i][1]})
            cache.put(rows)
            done += len(batch)
            rate = done / (time.perf_counter() - started_all)
            print(f"batch {b + 1}/{len(batches)}: {done} images, {rate:.2f} img/s", flush=True)
    return settings


def inception_score_bootstrap(generated, baseline, *, splits, draws, seed):
    """IS over prompt clusters; a paired call resamples both sets with the same prompts."""
    import numpy as np
    import torch
    from torch_fidelity.metric_isc import isc_features_to_metric
    from crope.metrics.statistics import confidence

    keys = sorted(generated if baseline is None else generated.keys() & baseline.keys())
    if len(keys) < 2:
        raise ValueError("IS intervals need at least two prompt clusters")

    def score(bank, chosen):
        x = torch.from_numpy(np.concatenate([bank[k] for k in chosen], axis=0))
        return isc_features_to_metric(x, splits=min(splits, len(x)), shuffle=True, rng_seed=2020)

    full = score(generated, keys)
    estimate = full["inception_score_mean"]
    if baseline is not None:
        estimate -= score(baseline, keys)["inception_score_mean"]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        chosen = [keys[i] for i in rng.integers(len(keys), size=len(keys))]
        value = score(generated, chosen)["inception_score_mean"]
        if baseline is not None:
            value -= score(baseline, chosen)["inception_score_mean"]
        samples.append(value)
    result = {**confidence(samples, estimate, len(keys)), "n_images": sum(len(generated[k]) for k in keys)}
    if baseline is None:
        result["split_std"] = full["inception_score_std"]
    return result


def write_csv(path: Path, rows: list[dict]):
    columns = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["empty"])
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    args = parser().parse_args()
    if args.batch_size < 1 or not 1 <= args.threads <= 16 or args.bootstrap < 0 or args.is_splits < 1:
        raise SystemExit("--batch-size >= 1, --threads in 1..16, --bootstrap >= 0, --is-splits >= 1")
    metrics = list(ALL_METRICS) if args.metrics == "all" else [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = sorted(set(metrics) - set(ALL_METRICS))
    if unknown:
        raise SystemExit(f"Unknown metrics {unknown}; choose from {ALL_METRICS}")
    setup_environment(args)

    import numpy as np
    import torch
    import transformers
    from crope.metrics.io import discover_samples, read_jsonl, write_json
    from crope.metrics.statistics import paired_scalar_summaries, scalar_summaries

    if int(transformers.__version__.split(".")[0]) >= 5:
        raise SystemExit("ImageReward needs transformers 4.x; run with the sca-metrics uv env")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = discover_samples(args.root, read_jsonl(args.prompts), args.res)
    if args.methods:
        keep = {m.strip() for m in args.methods.split(",")}
        records = [r for r in records if r["method"] in keep]
    if not records:
        raise SystemExit("No samples left after filters")
    print(f"{len(records)} images, methods {sorted({r['method'] for r in records})}", flush=True)

    cache = Cache(out / "extra_cache.jsonl")
    errors, provenance = [], {}
    model_metrics = [m for m in metrics if m != "latency"]
    settings = score_images(records, model_metrics, args, cache, errors, provenance) if model_metrics else {}

    rows, is_banks = [], defaultdict(lambda: defaultdict(list))
    for record in records:
        key = image_key(record)
        row = {k: record[k] for k in ("method", "res", "prompt_id", "seed", "path")}
        row["metrics"] = {}
        if "latency" in metrics and record.get("seconds") is not None:
            row["metrics"]["latency"] = float(record["seconds"])
        for metric, setting in settings.items():
            saved = cache.get(key, metric, setting)
            if saved is None:
                continue
            if metric == "is":
                logits = np.frombuffer(base64.b64decode(saved["scores"]["is_logits_b64"]), dtype=np.float32)
                is_banks[(record["method"], record["res"])][(record["prompt_id"], str(record["seed"]))] = logits
            else:
                row["metrics"].update(saved["scores"])
        rows.append(row)

    summaries = scalar_summaries(rows, draws=args.bootstrap, seed=args.bootstrap_seed)
    paired = paired_scalar_summaries(rows, args.reference_method, draws=args.bootstrap, seed=args.bootstrap_seed)

    # IS is a set statistic: cluster logits by prompt, keyed by seed for pairing.
    def clusters(bank, seeds=None):
        grouped = defaultdict(list)
        for (prompt, seed), logits in sorted(bank.items()):
            if seeds is None or (prompt, seed) in seeds:
                grouped[prompt].append(logits)
        return {k: np.stack(v) for k, v in grouped.items()}

    for (method, res), bank in sorted(is_banks.items()):
        try:
            summaries.append({"method": method, "res": res, "metric": "is",
                              **inception_score_bootstrap(clusters(bank), None, splits=args.is_splits,
                                                          draws=args.bootstrap, seed=args.bootstrap_seed)})
            reference = is_banks.get((args.reference_method, res))
            if method != args.reference_method and reference:
                shared = bank.keys() & reference.keys()
                paired.append({"method": method, "res": res, "reference_method": args.reference_method,
                               "metric": "is", "delta_definition": "method_minus_reference",
                               **inception_score_bootstrap(clusters(bank, shared), clusters(reference, shared),
                                                           splits=args.is_splits, draws=args.bootstrap,
                                                           seed=args.bootstrap_seed)})
        except Exception as exc:
            errors.append({"metric": "is", "stage": "summary", "group": f"{method}/{res}",
                           "error": f"{type(exc).__name__}: {exc}"})

    import importlib.metadata as meta
    versions = {}
    for package in ("torch", "transformers", "image-reward", "hpsv2", "pyiqa", "torch-fidelity", "open-clip-torch"):
        try:
            versions[package] = meta.version(package)
        except meta.PackageNotFoundError:
            versions[package] = None
    report = {"config": vars(args), "metrics": metrics, "versions": versions, "provenance": provenance,
              "errors": errors, "summary": summaries, "paired_deltas": paired, "per_image": rows,
              "uncertainty": "95% percentile bootstrap over prompt clusters; seeds stay inside prompts",
              "paired_note": "Only matching resolution/prompt/seed pairs enter deltas",
              "is_note": "IS per method/res from torch-fidelity isc_features_to_metric; CI resamples prompts",
              "seconds": time.perf_counter() - started, "complete": not errors}
    flat = [{**{k: r[k] for k in ("method", "res", "prompt_id", "seed", "path")}, **r["metrics"]} for r in rows]
    write_csv(out / "extra_per_image.csv", flat)
    write_csv(out / "extra_summary.csv", summaries)
    write_csv(out / "extra_paired_deltas.csv", paired)
    write_json(out / "extra_metrics.json", report)
    print(f"extra eval: {len(rows)} images, {len(summaries)} summaries, {len(paired)} paired deltas, "
          f"{len(errors)} errors; reports in {out}", flush=True)
    for row in summaries:
        print(f"  {row['method']:>24} {row['res']:>10} {row['metric']:>10} {row['value']} "
              f"[{row['ci_low']}, {row['ci_high']}]", flush=True)
    return 2 if errors and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
