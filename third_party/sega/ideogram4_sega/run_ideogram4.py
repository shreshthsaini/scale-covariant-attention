
import argparse
import os
import types


def _configure_compiler_caches() -> None:
    """Point Triton/TorchInductor caches at $SCRATCH (required on many HPC nodes)."""
    scratch = os.environ.get("SCRATCH", "/scratch/nova2001")
    for env_var, subdir in (
        ("TRITON_CACHE_DIR", ".triton/cache"),
        ("TORCHINDUCTOR_CACHE_DIR", ".cache/torchinductor"),
    ):
        if env_var not in os.environ:
            cache_dir = os.path.join(scratch, subdir)
            os.makedirs(cache_dir, exist_ok=True)
            os.environ[env_var] = cache_dir


_configure_compiler_caches()

import torch
from diffusers.models.autoencoders import AutoencoderKLFlux2
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3VLModel
from pipeline_ideogram4 import Ideogram4Pipeline, Ideogram4PromptEnhancerHead
from transformer_ideogram4 import Ideogram4Transformer2DModel

GPU_MEMORY_THRESHOLD_GB = 60

CHECKPOINTS = {
    "ideogram4": "ideogram-ai/ideogram-4-nf4-diffusers",
}
PROMPT_ENHANCER_HEAD = "diffusers/qwen3-vl-8b-instruct-lm-head"

DEFAULT_STEPS = 48
DEFAULT_GUIDANCE_SCHEDULE = (7.0,) * 45 + (3.0,) * 3


def get_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(x, device) for x in value)
    if isinstance(value, list):
        return [_move_to_device(x, device) for x in value]
    if isinstance(value, dict):
        moved = {k: _move_to_device(v, device) for k, v in value.items()}
        if type(value) is not dict:
            try:
                return type(value)(**moved)
            except Exception:
                pass
        return moved
    return value


def _distribute_transformer_blocks(transformer, num_gpus: int) -> None:
    """Spread one transformer's blocks (``.layers``) across GPUs; keep everything else on cuda:0."""
    for name, child in transformer.named_children():
        if name != "layers":
            child.to("cuda:0")

    blocks = transformer.layers
    blocks_per_gpu = max(1, len(blocks) // num_gpus)
    for i, block in enumerate(blocks):
        device = f"cuda:{min(i // blocks_per_gpu, num_gpus - 1)}"
        block.to(device)
        block._original_forward = block.forward.__func__
        block._target_device = device

        def forward_wrapper(self, *args, **kwargs):
            target_dev = self._target_device
            args = tuple(_move_to_device(a, target_dev) for a in args)
            kwargs = {k: _move_to_device(v, target_dev) for k, v in kwargs.items()}
            return _move_to_device(self._original_forward(self, *args, **kwargs), "cuda:0")

        block.forward = types.MethodType(forward_wrapper, block)

    transformer._original_transformer_forward = transformer.forward.__func__

    def transformer_forward_wrapper(self, *args, **kwargs):
        args = tuple(_move_to_device(a, "cuda:0") for a in args)
        kwargs = {k: _move_to_device(v, "cuda:0") for k, v in kwargs.items()}
        result = self._original_transformer_forward(self, *args, **kwargs)
        if hasattr(result, "sample"):
            result.sample = result.sample.to("cuda:0")
        elif isinstance(result, tuple):
            result = tuple(_move_to_device(r, "cuda:0") for r in result)
        return result

    transformer.forward = types.MethodType(transformer_forward_wrapper, transformer)


def _setup_multi_gpu(pipe, num_gpus: int) -> None:
    """Distribute both Ideogram4 transformers' blocks across all visible GPUs.

    The pipeline must already be placed on cuda:0 before calling this (so the bitsandbytes-quantized
    text encoder / VAE are correctly onloaded). Only the transformer blocks are scattered onto the
    remaining GPUs. Unlike the Qwen setup, the nf4-quantized text encoder cannot compute on CPU, so it
    stays on cuda:0.
    """
    _distribute_transformer_blocks(pipe.transformer, num_gpus)
    if pipe.unconditional_transformer is not None:
        _distribute_transformer_blocks(pipe.unconditional_transformer, num_gpus)

    orig_cls = type(pipe)
    pipe.__class__ = type(
        orig_cls.__name__,
        (orig_cls,),
        {"_execution_device": property(lambda self: torch.device("cuda:0"))},
    )
    print(f"[info] Multi-GPU: both transformers across {num_gpus} GPUs, text encoder + VAE on cuda:0")


def main():
    parser = argparse.ArgumentParser(description="Generate images with Ideogram 4")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a ginger cat wearing a tiny wizard hat reading a spellbook",
        help="Plain-text idea or pre-built JSON caption",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=sorted(CHECKPOINTS.keys()),
        default="ideogram4",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory; images are saved under outputs/<checkpoint>/",
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Force VAE tiling on (otherwise only auto-enabled for multi-GPU or GPUs under 60GB). "
        "Use for high-res single-GPU runs to avoid an OOM at the final decode.",
    )
    parser.add_argument(
        "--generator-device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Distribute both transformers' blocks across all visible GPUs (for high resolutions)",
    )
    parser.add_argument(
        "--no-prompt-upsampling",
        action="store_true",
        help="Skip prompt upsampling (use for pre-built JSON captions)",
    )
    parser.add_argument(
        "--guidance-mode",
        type=str,
        choices=["dual", "single"],
        default="dual",
        help=(
            "How to build the CFG negative signal. 'dual' (default) uses Ideogram4's two checkpoints "
            "(conditional + dedicated unconditional_transformer). 'single' uses only the conditional "
            "checkpoint, run with and without the prompt, and does not load the second checkpoint."
        ),
    )
    parser.add_argument(
        "--rope-scaling",
        type=str,
        choices=["none", "ntk", "yarn", "sega"],
        default="none",
        help=(
            "RoPE length-extrapolation method applied to both transformers (default: none). Each method is adapted "
            "to Ideogram4's interleaved MRoPE: it rescales only the per-axis h/w frequency combs, keeps text and "
            "the t axis untouched, and anchors the 65536 image-position offset at the original frequencies."
        ),
    )
    parser.add_argument(
        "--train-grid",
        type=int,
        default=256,
        help="Trained grid extent in latent patches; 256 = 4096 px (scaling engages only above this).",
    )

    args = parser.parse_args()
    prompt_upsampling = not args.no_prompt_upsampling and not args.prompt.lstrip().startswith("{")

    num_gpus = torch.cuda.device_count()
    enable_multi_gpu = args.multi_gpu and num_gpus >= 2
    if args.multi_gpu and num_gpus < 2:
        print("[warn] --multi-gpu requested but fewer than 2 GPUs visible; using single-GPU path")

    base_output_dir = args.output_dir
    if not base_output_dir.rstrip("/").endswith("outputs"):
        base_output_dir = os.path.join(base_output_dir, "outputs")
    output_dir = os.path.join(base_output_dir, args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    if args.generator_device == "cuda":
        gen_device = "cuda:0" if enable_multi_gpu else "cuda"
        generator = torch.Generator(gen_device).manual_seed(args.seed)
    else:
        generator = torch.manual_seed(args.seed)

    repo_id = CHECKPOINTS[args.checkpoint]
    model_path = snapshot_download(repo_id, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_path, "tokenizer"),
        local_files_only=True,
    )
    transformer = Ideogram4Transformer2DModel.from_pretrained(
        os.path.join(model_path, "transformer"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    # In "single" guidance mode the conditional transformer is reused (with/without prompt) for both CFG passes,
    # so the dedicated unconditional checkpoint is redundant and is not loaded at all.
    unconditional_transformer = None
    if args.guidance_mode == "dual":
        unconditional_transformer = Ideogram4Transformer2DModel.from_pretrained(
            os.path.join(model_path, "unconditional_transformer"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

    # Both transformers see the full extrapolated grid, so both need the scaled RoPE.
    if args.rope_scaling != "none":
        for tf in (transformer, unconditional_transformer):
            if tf is not None:
                tf.set_rope_scaling(method=args.rope_scaling, train_grid=args.train_grid)
        print(
            f"[info] RoPE scaling: {args.rope_scaling}, "
            f"train_grid={args.train_grid} ({args.train_grid * 16}px)"
        )
    prompt_enhancer_head = None
    if prompt_upsampling:
        prompt_enhancer_head = Ideogram4PromptEnhancerHead.from_pretrained(
            PROMPT_ENHANCER_HEAD,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

    pipe = Ideogram4Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(model_path, "scheduler"),
            local_files_only=True,
        ),
        vae=AutoencoderKLFlux2.from_pretrained(
            os.path.join(model_path, "vae"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ),
        text_encoder=Qwen3VLModel.from_pretrained(
            os.path.join(model_path, "text_encoder"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ),
        tokenizer=tokenizer,
        transformer=transformer,
        unconditional_transformer=unconditional_transformer,
        prompt_enhancer_head=prompt_enhancer_head,
    )

    if enable_multi_gpu:
        # Place everything on cuda:0 first (diffusers handles the bnb-quantized modules correctly),
        # then scatter the transformer blocks across the remaining GPUs.
        pipe = pipe.to("cuda")
        _setup_multi_gpu(pipe, num_gpus)
    elif args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    if args.vae_tiling or enable_multi_gpu or get_gpu_memory_gb() < GPU_MEMORY_THRESHOLD_GB:
        pipe.vae.enable_tiling()
        print(f"VAE tiling enabled (GPU memory: {get_gpu_memory_gb():.1f}GB)")

    pipe_kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "generator": generator,
        "num_inference_steps": args.steps,
        "prompt_upsampling": prompt_upsampling,
        "guidance_mode": args.guidance_mode,
    }
    if args.guidance_scale is not None:
        pipe_kwargs["guidance_scale"] = args.guidance_scale
        pipe_kwargs["guidance_schedule"] = None
    elif args.steps == DEFAULT_STEPS:
        pipe_kwargs["guidance_schedule"] = DEFAULT_GUIDANCE_SCHEDULE
    else:
        pipe_kwargs["guidance_scale"] = 7.0
        pipe_kwargs["guidance_schedule"] = None

    print(
        f"Generating {args.height}x{args.width} with {args.steps} steps "
        f"(checkpoint={args.checkpoint}, guidance_mode={args.guidance_mode}, "
        f"prompt_upsampling={prompt_upsampling})..."
    )
    image = pipe(**pipe_kwargs).images[0]

    prompt_part = args.prompt.split()[0:3]
    prompt_str = "_".join(prompt_part)
    rope_tag = "base" if args.rope_scaling == "none" else f"{args.rope_scaling}_tg{args.train_grid}"
    base_filename = f"{prompt_str}_{args.height}x{args.width}_s{args.seed}_{args.guidance_mode}_{rope_tag}.png"

    filename = os.path.join(output_dir, base_filename)
    image.save(filename)
    print(f"Image saved to: {filename}")


if __name__ == "__main__":
    main()
