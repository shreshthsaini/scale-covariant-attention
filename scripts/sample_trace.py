#!/usr/bin/env python3
"""Per-step denoising traces of main-campaign images for the project page ("how it works" videos).

Re-runs Aesthetic-4K prompt ids at 4096x4096, seed label 0, with the campaign's derived seed, settings
and code path. Method sca builds the sca_nb sampler exactly as scripts/sample.py does (per-head table
k1_1024.npz, scale 4 at 4096^2, gate sigma 0.89); method area builds AReA through scripts/run_area.py
(official code on FLUX.1-dev). At each of the 28 steps it saves, under <out-root>/<method>/<short id>/,

  steps/x0_NN.webp   decoded x0 prediction (the pipeline's tiled VAE decode), downscaled to 1024 px, WebP q90
  steps/crop_NN.png  native-pixel 512 x 512 crop of the same decoded image at the prompt's box
                     (web/build/compare_selection.json, box_xys = [x, y, size])

then final.png and trace.json: per step index, sigma, timestep and phase (SCA: "global" while sigma > gate,
else "mixed"; AReA: its own Phase I "coarse" or Phase II "fine" state), for SCA the 57 x 24 pi values the
attention processors used, and whether final.png matches the campaign PNG (SHA-256, pixels, PSNR).

x0 rule: x0_hat = x_t - sigma_t * v, with x_t and v taken from forward hooks on the transformer and
computed as one Euler step from sigma_t to 0 in the scheduler's own arithmetic, so the last x0_hat equals
the final latents. Differencing consecutive latents in callback_on_step_end would amplify their bf16
rounding by 1 / (sigma_t - sigma_t+1), about 80x, so the hooks are used instead.

--dry-run resolves prompts, boxes, seeds, sigmas, phases, the pi table and output paths and checks them
against the campaign sidecars without loading a model; it runs on CPU and writes nothing (unless
--plan-json is given).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))   # sample.py and run_area.py, imported as modules

SCRATCH = Path("/scratch/09032/saini_2/sca")
PROMPTS = Path("/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl")
SELECTION = ROOT / "web/build/compare_selection.json"
OUT_ROOT = SCRATCH / "web_assets/trace"
RES = "4096x4096"
SEED = 0
GUIDANCE = 4.5
PI_TABLE = SCRATCH / "calib/k1_1024.npz"
GATE_SIGMA = 0.89
# The campaign command lines (taskq/done/b155_sca_nb_4096x4096.sh and the out/main/area/4096x4096 tasks)
# without --prompts, --out, --start and --end.
CAMPAIGN = {
    "sca": {"dir": SCRATCH / "out/main/sca_nb" / RES, "label": "sca_nb",
            "argv": ["--method", "sca", "--res", RES, "--seed", str(SEED), "--guidance", str(GUIDANCE),
                     "--pi-table", str(PI_TABLE), "--gate-sigma", str(GATE_SIGMA), "--label", "sca_nb"]},
    "area": {"dir": SCRATCH / "out/main/area" / RES, "label": "area",
             "argv": ["--res", RES, "--seed", str(SEED), "--guidance", str(GUIDANCE)]},
}
IGNORED = ("implementation_sha256", "code_revision")   # differ by construction: frozen tree with this file
VERSIONS = ("diffusers", "torch")                       # reported, not fatal
STEP_FILES = ("steps/x0_{:02d}.webp", "steps/crop_{:02d}.png")
PI_SOURCE_RUN = ("processor: SCAContext._pi_now, the [57, 24] float32 tensor every SCAProcessor of the step "
                 "multiplied into the mix, read by a transformer forward hook after each call")
PI_SOURCE_DRY = "table: SCAContext.pi (PiTable.__call__, the interpolation SCAContext.pi_now uses), dry run"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=tuple(CAMPAIGN))
    p.add_argument("--sid", required=True, nargs="+", help="short ids of web/build/compare_selection.json")
    p.add_argument("--prompts", type=Path, default=PROMPTS)
    p.add_argument("--selection", type=Path, default=SELECTION)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--preview", type=int, default=1024, help="preview width in px")
    p.add_argument("--crop", type=int, default=512, help="crop side in native px; must equal the box size")
    p.add_argument("--webp-quality", type=int, default=90)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="redo a trace that is already complete")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="run even if the rebuilt config differs from the campaign sidecar in a setting")
    p.add_argument("--dry-run", action="store_true", help="no model: resolve and check everything, then exit")
    p.add_argument("--plan-json", type=Path, help="dry run: also write the resolved plan (per-step sigma, "
                   "phase, pi) to this file")
    return p


# ----------------------------------------------------------------------------- resolution (no model)

def read_selection(path):
    data = json.loads(Path(path).read_text())
    return {item["sid"]: item for item in data["items"]}


def resolve(method, sid, args, selection, prompts):
    """Prompt, seed, box, campaign record and output paths of one (method, sid); checks every link."""
    from sca.runtime import image_seed, image_stem, resolution
    item = selection.get(sid)
    if item is None:
        raise SystemExit(f"{sid}: not in {args.selection}")
    camp = CAMPAIGN[method]
    sel_png = Path(item["paths"][method])
    sidecar = sel_png.with_suffix(".json")
    if not sidecar.exists() or not sel_png.exists():
        raise SystemExit(f"{sid}: campaign image or sidecar missing: {sel_png}")
    record = json.loads(sidecar.read_text())
    config = record.get("config", {})
    if not record.get("complete") or config.get("method") != camp["label"] or config.get("res") != RES:
        raise SystemExit(f"{sid}: {sidecar} is not a complete {camp['label']} {RES} record")
    pid = config["prompt_id"]
    row = prompts.get(pid)
    if row is None:
        raise SystemExit(f"{sid}: prompt id {pid} not in {args.prompts}")
    if not row["prompt"] == config["prompt"] == item["prompt"]:
        raise SystemExit(f"{sid}: prompt text differs between {args.prompts}, the sidecar and the selection")
    derived = image_seed(pid, SEED)
    if config.get("seed") != SEED or config.get("derived_seed") != derived:
        raise SystemExit(f"{sid}: derived seed {derived} differs from the sidecar ({config.get('derived_seed')})")
    png = camp["dir"] / f"{image_stem(pid, SEED)}.png"
    if png.resolve() != sel_png.resolve():
        raise SystemExit(f"{sid}: selection path {sel_png} is not the campaign file {png}")
    h, w = resolution(RES)
    x, y, s = (int(v) for v in item["box_xys"])
    if s != args.crop or x < 0 or y < 0 or x + s > w or y + s > h:
        raise SystemExit(f"{sid}: box {item['box_xys']} is not a {args.crop} px window inside {RES}")
    return {"method": method, "sid": sid, "label": camp["label"], "prompt_id": pid, "prompt": row["prompt"],
            "selection_label": item.get("label"), "role": item.get("role"), "seed": SEED, "derived_seed": derived,
            "res": RES, "box_xys": [x, y, s], "campaign_png": str(png), "campaign_json": str(sidecar),
            "campaign_record": record, "out": args.out_root / method / sid}


def sampler_args(method, args, run):
    """The campaign CLI parsed by the campaign script's own parser."""
    argv = CAMPAIGN[method]["argv"] + ["--prompts", str(args.prompts), "--out", str(run["out"]),
                                       "--device", args.device]
    if method == "sca":
        import sample
        return sample.parser().parse_args(argv)
    import run_area
    return run_area.parser().parse_args(argv)


def campaign_scheduler(method, h, w):
    """The scheduler a campaign run steps with, built on CPU from the FLUX.1-dev snapshot."""
    from types import SimpleNamespace
    from diffusers import FlowMatchEulerDiscreteScheduler
    from sca.runtime import set_mu, snapshot
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(str(snapshot("black-forest-labs/FLUX.1-dev")),
                                                            subfolder="scheduler", local_files_only=True)
    if method == "sca":
        set_mu(SimpleNamespace(scheduler=sched), h, w, "max_shift")     # sample.py
        return sched
    # AReA FluxBackbone.attach replaces the scheduler with this copy (area/backbones/flux.py)
    return type(sched).from_config(sched.config, base_shift=sched.config.max_shift)


def pipeline_timesteps(sched, h, w, steps):
    """FluxPipeline.__call__ step 5 (Prepare timesteps), on CPU."""
    import numpy as np
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
    sigmas = np.linspace(1.0, 1 / steps, steps)
    if hasattr(sched.config, "use_flow_sigmas") and sched.config.use_flow_sigmas:
        sigmas = None
    mu = calculate_shift((h // 16) * (w // 16), sched.config.get("base_image_seq_len", 256),
                         sched.config.get("max_image_seq_len", 4096), sched.config.get("base_shift", 0.5),
                         sched.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(sched, steps, "cpu", sigmas=sigmas, mu=mu)
    return [float(s) for s in sched.sigmas], [float(t) for t in timesteps]


def sca_config(a, row, info, grid, scheduler_config, impl):
    """The sidecar config of scripts/sample.py main(), field for field."""
    import diffusers
    import torch
    from sca.runtime import MODELS, image_seed
    h, w = a.res
    return {"schema_version": 1, "method": a.label or a.method, "res": f"{h}x{w}", "prompt_id": str(row["id"]),
            "prompt": row["prompt"], "seed": int(a.seed), "derived_seed": image_seed(row["id"], a.seed),
            "steps": a.steps, "guidance": a.guidance, "model": MODELS[a.model],
            "dtype": "bfloat16", "batch_size": 1, "attention": f"sca:{a.method}",
            "diffusers": diffusers.__version__, "torch": torch.__version__,
            "implementation_sha256": impl, "grid": list(grid), "mu_mode": a.mu_mode,
            "sca": info, "scheduler": dict(scheduler_config)}


def config_check(ours, campaign):
    from sca.runtime import _stable
    a, b = _stable(json.loads(json.dumps(ours))), _stable(campaign)
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    settings = [k for k in keys if k not in IGNORED + VERSIONS]
    return {"match": not settings, "differing_settings": settings,
            "differing_versions": [k for k in keys if k in VERSIONS],
            "ignored_differences": [k for k in keys if k in IGNORED],
            "detail": {k: {"trace": a.get(k), "campaign": b.get(k)} for k in settings if k != "scheduler"}}


def plan_sca(a, run, h, w):
    """Per-step sigma, phase and pi as the processors would see them, with checks against the sidecar."""
    import numpy as np
    import sample
    from sca.runtime import implementation_hash
    grid = (h // 16, w // 16)
    ctx, info = sample.build_context(a, grid)          # loads the pi table exactly as the campaign run did
    sched = campaign_scheduler("sca", h, w)
    sigmas, timesteps = pipeline_timesteps(sched, h, w, a.steps)
    steps = []
    for i in range(a.steps):
        sigma = sigmas[i]                                   # install() pre-hook: float(scheduler.sigmas[step])
        gated = ctx.gate_sigma > 0 and sigma > ctx.gate_sigma   # SCAProcessor.sca
        pi = None if gated else np.atleast_2d(np.asarray(ctx.pi(sigma), dtype=np.float64))   # SCAContext.pi_now
        steps.append({"index": i, "sigma": sigma, "timestep": timesteps[i], "phase": "global" if gated else "mixed",
                      "pi": pi})
    record = run["campaign_record"]
    trace = {e["step"]: e for e in record.get("pi_trace", [])}
    mixed = [s for s in steps if s["pi"] is not None]
    pi_diff = max((abs(float(s["pi"].mean()) - trace[s["index"]]["pi_mean"]) if s["index"] in trace else float("inf"))
                  for s in mixed) if mixed else 0.0
    checks = {
        "sigmas_max_abs_diff": max(abs(x - y) for x, y in zip(sigmas[:a.steps], record["sigmas"])),
        "pi_mean_vs_campaign_pi_trace_max_abs_diff": pi_diff,
        "campaign_pi_trace_steps": sorted(trace) == [s["index"] for s in mixed],
        "pi_shape": list(mixed[0]["pi"].shape) if mixed else None,
        "pi_scale": info.get("pi_scale"),
    }
    config = sca_config(a, {"id": run["prompt_id"], "prompt": run["prompt"]}, info, grid, sched.config,
                        implementation_hash())
    return steps, checks, config_check(config, record["config"])


def plan_area(a, run, h, w):
    import run_area
    from area.backbones.flux import NATIVE_PATCHES, PHASE1_STEPS
    from area.schedule import build_schedule
    runner = run_area.AReARunner(a, dry_run=True)
    phase1 = PHASE1_STEPS if runner.cfg.phase1_steps is None else runner.cfg.phase1_steps
    schedule = build_schedule(h // 16, w // 16, a.steps, phase1, NATIVE_PATCHES, txt_len=run_area.TXT_LEN)
    sched = campaign_scheduler("area", h, w)
    sigmas, timesteps = pipeline_timesteps(sched, h, w, a.steps)
    steps = []
    for i in range(a.steps):
        phase, size = schedule.at(i)                         # FluxBackbone._begin_step
        steps.append({"index": i, "sigma": sigmas[i], "timestep": timesteps[i], "phase": phase,
                      "area_size": int(size)})
    record = run["campaign_record"]
    checks = {"sigmas_max_abs_diff": max(abs(x - y) for x, y in zip(sigmas, record["sigmas"])),
              "phase1_steps": phase1, "schedule_matches_sidecar":
              runner.schedule == record["config"]["area"]["schedule"]}
    config = runner.config({"id": run["prompt_id"], "prompt": run["prompt"]}, run["seed"])
    config["scheduler"] = dict(sched.config)
    return steps, checks, config_check(config, record["config"])


def trace_complete(out, steps=28):
    meta = out / "trace.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except ValueError:
        return False
    files = [out / "final.png"] + [out / f.format(i) for i in range(steps) for f in STEP_FILES]
    return bool(data.get("complete")) and all(p.exists() for p in files)


def dry_run(args, runs):
    import numpy as np
    from sca.runtime import resolution
    h, w = resolution(RES)
    plans, failed = [], False
    for run in runs:
        a = sampler_args(run["method"], args, run)
        steps, checks, cfg = (plan_sca if run["method"] == "sca" else plan_area)(a, run, h, w)
        ok = cfg["match"] and checks["sigmas_max_abs_diff"] == 0.0 and (
            checks.get("pi_mean_vs_campaign_pi_trace_max_abs_diff", 0.0) == 0.0
            and checks.get("campaign_pi_trace_steps", True) and checks.get("schedule_matches_sidecar", True))
        failed |= not ok
        phases = {}
        for s in steps:
            phases[s["phase"]] = phases.get(s["phase"], 0) + 1
        print(f"{'ok  ' if ok else 'FAIL'} {run['method']} {run['sid']} prompt_id={run['prompt_id']} "
              f"derived_seed={run['derived_seed']} box={run['box_xys']}", flush=True)
        print(f"     out={run['out']} ({'complete, would skip' if trace_complete(run['out']) else 'todo'})")
        print(f"     phases={phases} first_after_switch={next((s['index'] for s in steps if s['phase'] != steps[0]['phase']), None)} "
              f"checks={json.dumps(checks)}")
        print(f"     config_vs_campaign={json.dumps({k: v for k, v in cfg.items() if k != 'detail'})}")
        if cfg["detail"]:
            print(f"     differing settings: {json.dumps(cfg['detail'])[:2000]}")
        plans.append({k: (str(v) if isinstance(v, Path) else v) for k, v in run.items() if k != "campaign_record"}
                     | {"checks": checks, "config_check": cfg, "a": vars(a) | {"res": list(a.res)},
                        "steps": [s | ({"pi": np.round(s["pi"], 6).tolist()} if s.get("pi") is not None else {})
                                  for s in steps]})
    if args.plan_json:
        args.plan_json.parent.mkdir(parents=True, exist_ok=True)
        args.plan_json.write_text(json.dumps({"dry_run": True, "plans": plans}, default=str, indent=1) + "\n")
    print(f"dry-run {'FAILED' if failed else 'ok'}: {len(runs)} run(s), method {args.method}, no model loaded",
          flush=True)
    return 1 if failed else 0


# ----------------------------------------------------------------------------- tracing (GPU)

def decode(pipe, packed, h, w):
    """FluxPipeline.__call__'s own decode for output_type="pil", applied to packed latents."""
    latents = pipe._unpack_latents(packed, h, w, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def atomic_save(image, path, **kwargs):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    image.save(tmp, **kwargs)
    os.replace(tmp, path)


def save_step(image, out, i, box, preview, quality):
    from PIL import Image
    x, y, s = box
    size = (preview, round(preview * image.height / image.width))
    atomic_save(image.resize(size, Image.Resampling.LANCZOS), out / STEP_FILES[0].format(i), format="WEBP",
                quality=quality, method=6)
    atomic_save(image.crop((x, y, x + s, y + s)), out / STEP_FILES[1].format(i), format="PNG")


class Tracer:
    """Forward hooks capture x_t, v and the attention state of each step; the step-end callback decodes
    x0_hat through the pipeline's VAE path and hands the image writes to a background thread."""

    def __init__(self, pipe, run, h, w, steps, args, ctx=None, area_state=None):
        self.pipe, self.run, self.h, self.w, self.n = pipe, run, h, w, steps
        self.args, self.ctx, self.area_state = args, ctx, area_state
        self.records, self.futures, self.handles = [], [], []
        self.calls, self.x_t, self.x0, self.info = 0, None, None, None
        self.decode_seconds, self.last_image, self.last_x0_equals_final_latents = 0.0, None, None
        self.pool = ThreadPoolExecutor(max_workers=2)

    def __enter__(self):
        tr = self.pipe.transformer
        self.handles = [tr.register_forward_pre_hook(self._pre, with_kwargs=True),
                        tr.register_forward_hook(self._post, with_kwargs=True)]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _pre(self, module, args, kwargs):
        self.x_t = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

    def _post(self, module, args, kwargs, output):
        import torch
        v = output[0] if isinstance(output, (tuple, list)) else output.sample
        sched = self.pipe.scheduler
        j = sched.step_index
        i = 0 if j is None else int(j)
        sigma, end = sched.sigmas[i], sched.sigmas[-1]
        if float(end) != 0.0:
            raise RuntimeError("scheduler sigmas do not end at 0")
        # FlowMatchEulerDiscreteScheduler.step with sigma_next = 0: x0_hat = x_t - sigma_t * v
        self.x0 = (self.x_t.to(torch.float32) + (end - sigma) * v).to(v.dtype)
        info = {"index": i, "sigma": float(sigma)}
        if self.ctx is not None:
            p = self.ctx._pi_now           # None when the gate kept every processor on the global branch
            info.update(sca_step=int(self.ctx.step), sca_sigma=float(self.ctx.sigma),
                        pi=None if p is None else p.detach().to(torch.float32).cpu().numpy())
        if self.area_state is not None:
            st = self.area_state
            info.update(area_phase=st.phase, area_size=None if st.phase is None else int(st.size),
                        area_step=None if st.phase is None else int(st.step))
        self.info = info
        self.calls += 1

    def callback(self, pipe, i, t, kwargs):
        import numpy as np
        import torch
        info = self.info
        if info is None or info["index"] != i or self.calls != i + 1:
            raise RuntimeError(f"trace: expected one transformer call for step {i}, got {self.calls} calls")
        start = time.perf_counter()
        image = decode(pipe, self.x0, self.h, self.w)
        seconds = time.perf_counter() - start            # postprocess copies to the CPU, so this is synced
        self.decode_seconds += seconds
        rec = {"index": i, "step": i + 1, "sigma": info["sigma"], "timestep": float(t),
               "preview": STEP_FILES[0].format(i), "crop": STEP_FILES[1].format(i), "decode_seconds": round(seconds, 3)}
        if self.ctx is not None:
            pi = info["pi"]
            rule = "global" if self.ctx.gate_sigma > 0 and info["sca_sigma"] > self.ctx.gate_sigma else "mixed"
            rec["phase"] = "global" if pi is None else "mixed"
            rec["phase_rule_agrees"] = rec["phase"] == rule
            rec["sca_sigma"] = info["sca_sigma"]
            if pi is not None:
                table = np.atleast_2d(np.asarray(self.ctx.pi(info["sca_sigma"]), dtype=np.float64))
                rec["pi"] = np.round(pi.astype(np.float64), 6).tolist()
                rec["pi_mean"] = float(pi.astype(np.float64).mean())
                rec["pi_vs_table_max_abs_diff"] = float(np.abs(pi - table).max())
            else:
                rec["pi"] = None
        if self.area_state is not None:
            rec.update(phase=info["area_phase"], area_size=info["area_size"], area_step=info["area_step"])
        if i == self.n - 1:
            self.last_x0_equals_final_latents = bool(torch.equal(self.x0, kwargs["latents"]))
            self.last_image = image
        self.futures.append(self.pool.submit(save_step, image, self.run["out"], i, self.run["box_xys"],
                                             self.args.preview, self.args.webp_quality))
        self.records.append(rec)
        self.info = None
        print(f"step {i + 1:02d}/{self.n} sigma={info['sigma']:.4f} phase={rec.get('phase')} "
              f"decode={seconds:.1f}s", flush=True)
        return {}

    def wait(self):
        for future in self.futures:
            future.result()
        self.pool.shutdown()


def compare_final(image, run, last_image):
    import numpy as np
    from PIL import Image
    from sca.runtime import file_sha256
    ours = np.asarray(image.convert("RGB"))
    with Image.open(run["campaign_png"]) as im:
        theirs = np.asarray(im.convert("RGB"))
    out = {"campaign_png": run["campaign_png"], "campaign_sha256": file_sha256(run["campaign_png"]),
           "campaign_sidecar_sha256": run["campaign_record"]["png_sha256"]}
    if ours.shape != theirs.shape:
        return out | {"pixels_identical": False, "note": f"shape {ours.shape} vs {theirs.shape}"}
    diff = np.abs(ours.astype(np.int16) - theirs.astype(np.int16))
    mse = float((diff.astype(np.float64) ** 2).mean())
    out.update(pixels_identical=bool(mse == 0.0), max_abs_diff=int(diff.max()),
               differing_pixel_fraction=float((diff.max(axis=-1) > 0).mean()),
               psnr_db=None if mse == 0.0 else float(10 * np.log10(255.0 ** 2 / mse)),
               psnr_note="null means identical pixels (infinite PSNR)")
    if last_image is not None:
        out["last_x0_decode_identical_to_final"] = bool(np.array_equal(np.asarray(last_image.convert("RGB")), ours))
    return out


def finish(run, tracer, image, config, check, args, seconds, extra):
    import torch
    from sca.runtime import atomic_json, file_sha256, implementation_hash
    tracer.wait()
    out = run["out"]
    final = out / "final.png"
    atomic_save(image, final, format="PNG")
    comparison = compare_final(image, run, tracer.last_image)
    sha = file_sha256(final)
    commit = (ROOT / "COMMIT").read_text().strip() if (ROOT / "COMMIT").exists() else None
    record = {
        "schema_version": 1, "complete": True, "synthetic": False,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": run["method"], "label": run["label"], "sid": run["sid"], "prompt_id": run["prompt_id"],
        "prompt": run["prompt"], "selection_label": run["selection_label"], "seed": run["seed"],
        "derived_seed": run["derived_seed"], "res": run["res"], "steps_total": len(tracer.records),
        "guidance": GUIDANCE, "box_xys": run["box_xys"], "preview_px": args.preview, "crop_px": args.crop,
        "webp_quality": args.webp_quality,
        "x0_rule": "x0_hat = x_t - sigma_t * v: forward hooks on the transformer give x_t and v; computed as one "
                   "Euler step from sigma_t to 0 with FlowMatchEulerDiscreteScheduler.step arithmetic, cast to the "
                   "latent dtype, then the pipeline's decode (unpack, scale and shift, tiled VAE, postprocess)",
        "code": {"root": str(ROOT), "commit": commit, "implementation_sha256": implementation_hash(),
                 "sample_trace_sha256": file_sha256(Path(__file__).resolve())},
        "campaign": {"png": run["campaign_png"], "json": run["campaign_json"],
                     "config_fingerprint": run["campaign_record"].get("config_fingerprint")},
        "config_check": check,
        "final": {"png": "final.png", "sha256": sha, "sha256_matches_campaign": sha == comparison["campaign_sha256"],
                  "last_x0_equals_final_latents": tracer.last_x0_equals_final_latents} | comparison,
        "timing": {"pipeline_seconds_with_trace": round(seconds, 2), "decode_seconds_total": round(tracer.decode_seconds, 2),
                   "campaign_seconds_without_trace": run["campaign_record"].get("seconds"),
                   "peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None},
        "steps": tracer.records,
    } | extra
    atomic_json(out / "trace.json", record)
    f = record["final"]
    print(f"saved trace {run['method']} {run['sid']}: final sha256 match={f['sha256_matches_campaign']} "
          f"pixels identical={f.get('pixels_identical')} psnr={f.get('psnr_db')} seconds={seconds:.1f}", flush=True)


def generate(pipe, run, h, w, a, tracer, max_sequence_length, **kwargs):
    import torch
    from sca.runtime import image_seed
    generator = torch.Generator(device="cpu").manual_seed(image_seed(run["prompt_id"], a.seed))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with tracer, torch.inference_mode():
        image = pipe(prompt=run["prompt"], height=h, width=w, num_inference_steps=a.steps, guidance_scale=a.guidance,
                     generator=generator, max_sequence_length=max_sequence_length,
                     callback_on_step_end=tracer.callback, callback_on_step_end_tensor_inputs=["latents"],
                     **kwargs).images[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return image, time.perf_counter() - start


def guard(check, args, run):
    if check["match"]:
        return
    msg = f"{run['method']} {run['sid']}: rebuilt config differs from the campaign in {check['differing_settings']}"
    if not args.allow_mismatch:
        raise SystemExit(msg + " (use --allow-mismatch to trace anyway)")
    print("WARNING " + msg, flush=True)


def trace_sca(args, runs):
    import sample
    from sca.attn import install
    from sca.runtime import implementation_hash, load_pipeline, runtime_environment, set_mu
    a = sampler_args("sca", args, runs[0])
    runtime_environment()
    h, w = a.res
    grid = (h // 16, w // 16)
    ctx, info = sample.build_context(a, grid)
    t0 = time.perf_counter()
    pipe = load_pipeline(a.device, a.model)
    uninstall = install(pipe.transformer, ctx)
    ctx.scheduler = pipe.scheduler
    if h * w >= 2048 * 4096:
        pipe.vae.enable_tiling()
    set_mu(pipe, h, w, a.mu_mode)
    print(f"model ready in {time.perf_counter() - t0:.1f}s", flush=True)
    impl = implementation_hash()
    try:
        for run in runs:
            config = sca_config(a, {"id": run["prompt_id"], "prompt": run["prompt"]}, info, grid,
                                pipe.scheduler.config, impl)
            check = config_check(config, run["campaign_record"]["config"])
            guard(check, args, run)
            (run["out"] / "steps").mkdir(parents=True, exist_ok=True)
            ctx.pi_trace.clear()
            tracer = Tracer(pipe, run, h, w, a.steps, args, ctx=ctx)
            image, seconds = generate(pipe, run, h, w, a, tracer, 512)
            campaign_trace = run["campaign_record"].get("pi_trace", [])
            extra = {"gate_sigma": ctx.gate_sigma,
                     "phase_rule": "global while sigma > gate_sigma (SCAProcessor.sca: global branch only, pi unused), "
                                   "else mixed: pi * O_loc + (1 - pi) * O_glob per head",
                     "pi_info": {"source": PI_SOURCE_RUN, "shape": [ctx.n_layers, 24], "table": str(a.pi_table),
                                 "table_sha256": info["pi_table_sha256"], "scale": info["pi_scale"],
                                 "variant": info["pi_variant"]},
                     "pi_trace": list(ctx.pi_trace), "pi_trace_matches_campaign": list(ctx.pi_trace) == campaign_trace,
                     "sca_log": sorted(set(ctx.log))}
            finish(run, tracer, image, config, check, args, seconds, extra)
    finally:
        uninstall()


def trace_area(args, runs):
    import run_area
    a = sampler_args("area", args, runs[0])
    run_area.environment()
    t0 = time.perf_counter()
    runner = run_area.AReARunner(a)
    h, w = runner.res
    print(f"model ready in {time.perf_counter() - t0:.1f}s (cpu_offload={runner.offload})", flush=True)
    try:
        for run in runs:
            config = runner.config({"id": run["prompt_id"], "prompt": run["prompt"]}, a.seed)
            check = config_check(config, run["campaign_record"]["config"])
            guard(check, args, run)
            (run["out"] / "steps").mkdir(parents=True, exist_ok=True)
            tracer = Tracer(runner.pipe, run, h, w, a.steps, args, area_state=runner.area.backbone.state)
            image, seconds = generate(runner.pipe, run, h, w, a, tracer, run_area.TXT_LEN, output_type="pil")
            extra = {"phase_rule": "AReA's own per-step state (FluxBackbone.state after _begin_step): coarse = "
                                   "Phase I, attention on a pooled grid (area_size = pooling factor); fine = Phase II, "
                                   "adaptive windows with Halo and bridge tokens (area_size = window side, tokens)",
                     "area_schedule": runner.schedule}
            finish(run, tracer, image, config, check, args, seconds, extra)
    finally:
        runner.close()


def main(argv=None):
    args = parser().parse_args(argv)
    from sca.runtime import read_prompts
    selection = read_selection(args.selection)
    prompts = {row["id"]: row for row in read_prompts(args.prompts)}
    runs = [resolve(args.method, sid, args, selection, prompts) for sid in dict.fromkeys(args.sid)]
    if args.dry_run:
        return dry_run(args, runs)
    todo = [r for r in runs if args.force or not trace_complete(r["out"])]
    for r in runs:
        if r not in todo:
            print(f"skip {r['method']} {r['sid']}: complete trace in {r['out']}", flush=True)
    if todo:
        (trace_sca if args.method == "sca" else trace_area)(args, todo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
