#!/usr/bin/env python3
"""Round 7 timing probe: decode the clean-image estimate x0_hat = x_t - sigma_t v_t at chosen steps of one run.

Same CLI as scripts/sample.py (operator flags, prompts, --start/--end, --out) plus --probe-steps. For every prompt it
writes <out>/<stem>_x0_sNN.png for each probed step NN. v_t comes from consecutive latents of the pipeline's
step-end callback: v_t = (x_{t+1} - x_t) / (sigma_{t+1} - sigma_t), exact for FlowMatchEuler. Question it answers:
is a copy already in the opening's estimate (step 7), or does it first appear on the window steps?
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample as S  # noqa: E402


def decode(pipe, x0, h, w):
    import torch
    lat = pipe._unpack_latents(x0.to(pipe.vae.dtype), h, w, pipe.vae_scale_factor)
    lat = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.inference_mode():
        img = pipe.vae.decode(lat, return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


def main():
    p = S.parser()
    p.add_argument("--probe-steps", default="7,9,11,13,15")
    args = p.parse_args()
    S.runtime_environment()
    import torch
    from sca.attn import install
    from sca.runtime import image_seed, image_stem, load_pipeline, read_prompts, set_mu
    h, w = args.res
    ctx, _ = S.build_context(args, (h // 16, w // 16))
    pipe = load_pipeline(args.device, args.model)
    uninstall = install(pipe.transformer, ctx)
    ctx.scheduler = pipe.scheduler
    pipe.vae.enable_tiling()
    set_mu(pipe, h, w, args.mu_mode)
    steps = [int(s) for s in args.probe_steps.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    try:
        for row in prompts[args.start:end]:
            lat = {}

            def cb(pipe_, i, t, kw):
                lat[i] = kw["latents"].detach().float().clone()
                return kw
            gen = torch.Generator(device="cpu").manual_seed(image_seed(row["id"], args.seed))
            with torch.inference_mode():
                pipe(prompt=row["prompt"], height=h, width=w, num_inference_steps=args.steps,
                     guidance_scale=args.guidance, generator=gen, max_sequence_length=512, callback_on_step_end=cb,
                     callback_on_step_end_tensor_inputs=["latents"], output_type="latent")
            sig = [float(s) for s in pipe.scheduler.sigmas]
            for st in steps:
                if st - 1 not in lat or st not in lat:
                    continue
                v = (lat[st] - lat[st - 1]) / (sig[st + 1] - sig[st])
                x0 = lat[st - 1] - sig[st] * v
                decode(pipe, x0, h, w).save(args.out / f"{image_stem(row['id'], args.seed)}_x0_s{st:02d}.png")
            print(f"probed {row['id']} steps {steps}", flush=True)
    finally:
        uninstall()


if __name__ == "__main__":
    main()
