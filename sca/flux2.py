"""Scale-covariant attention for FLUX.2-dev (diffusers Flux2Transformer2DModel).

Same method as sca/attn.py (global branch on the coarse grid, exact native-extent local window,
sigma gate, per-head pi mix, calib statistics); this module only adapts the parts that differ:

- Flux2 has two attention module types: Flux2Attention (8 double-stream blocks, separate text
  projections and outputs) and Flux2ParallelSelfAttention (48 single-stream blocks, QKV fused with
  the MLP input projection and the attention output fused with the MLP output projection).
- Position ids are 4-D (t, h, w, l): image tokens are (0, row, col, 0), text tokens (0, 0, 0, j).
  The coarse RoPE of the global branch replaces only the row / col axes by block indices.
- The 24B Mistral text encoder and the 32B transformer do not fit on one 96 GB GPU together:
  prompts are encoded first (cached on scratch), the text encoder is freed, then the
  transformer is loaded.
- The pipeline picks the timestep shift mu from compute_empirical_mu(seq_len, steps), which
  extrapolates linearly in the sequence length (mu = 11.5 at 4096^2, every sigma ~ 1). mu_mode
  "native" caps seq_len at the native 64 x 64 grid, the FLUX.2 analogue of FLUX.1's
  mu = max_shift protocol.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from sca.attn import SCAContext, SCAProcessor, attention_flops, rope, sdpa

REPO = "black-forest-labs/FLUX.2-dev"
HF_HOME = "/scratch/09032/saini_2/tmp/hf"
EMBED_CACHE = Path("/scratch/09032/saini_2/sca/flux2_embeds")
NATIVE_SEQ = 4096                     # 1024^2 image, 16 px per token: n0 = 64
TEXT_LAYERS = (10, 20, 30)            # Mistral hidden states stacked into the 15360-d text features
MAX_SEQ = 512


# ----------------------------------------------------------------------------- context + processors

class Flux2SCAContext(SCAContext):
    """SCAContext for 4-D (t, h, w, l) position ids."""

    def coarse_rope(self, f, device):
        """(rope_q, rope_k): image queries at floor(row / fh), floor(col / fw); pooled keys at
        block indices. Axes t and l keep their values (0 for image tokens)."""
        key = (f, str(device))
        if key not in self._rope_cache:
            ids = self._img_ids.to(device).float()
            q_ids = ids.clone()
            for ax, fa in ((1, f[0]), (2, f[1])):
                q_ids[:, ax] = torch.floor((ids[:, ax] - ids[:, ax].min()) / fa)
            H, W = self.grid
            k_ids = q_ids.view(H, W, -1)[f[0] // 2::f[0], f[1] // 2::f[1]].reshape(-1, q_ids.shape[1])
            self._rope_cache[key] = (self.pos_embed(q_ids), self.pos_embed(k_ids))
        return self._rope_cache[key]


class Flux2SCAProcessor(SCAProcessor):
    """Drop-in for Flux2AttnProcessor (double-stream blocks); same outputs in mode full."""

    def attend(self, q, k, v, rot, attention_mask=None):
        """q, k, v: normed, pre-RoPE [B, S, h, D], text first. Returns [B, S, h, D]."""
        ctx = self.ctx
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported by SCA")
        if ctx.mode == "full" or ctx.grid is None:
            if ctx.mode != "full":
                raise RuntimeError("SCA context has no image grid; install the transformer pre-hook")
            return sdpa(rope(q, rot), rope(k, rot), v)
        return self.sca(q, k, v, rot) if ctx.mode == "sca" else self.calibrate(q, k, v, rot)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None):
        from diffusers.models.transformers.transformer_flux2 import _get_qkv_projections
        q, k, v, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q, k, v = (t.unflatten(-1, (attn.heads, -1)) for t in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            eq, ek, ev = (t.unflatten(-1, (attn.heads, -1)) for t in (eq, ek, ev))
            q = torch.cat([attn.norm_added_q(eq), q], 1)
            k = torch.cat([attn.norm_added_k(ek), k], 1)
            v = torch.cat([ev, v], 1)
        hs = self.attend(q, k, v, image_rotary_emb, attention_mask).flatten(2, 3).to(q.dtype)
        txt = None
        if encoder_hidden_states is not None:
            L = encoder_hidden_states.shape[1]
            txt, hs = hs.split_with_sizes([L, hs.shape[1] - L], 1)
            txt = attn.to_add_out(txt)
        hs = attn.to_out[1](attn.to_out[0](hs))
        return hs if txt is None else (hs, txt)


class Flux2SCASingleProcessor(Flux2SCAProcessor):
    """Drop-in for Flux2ParallelSelfAttnProcessor (single-stream blocks, attention + parallel MLP)."""

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        h = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp = torch.split(h, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)
        q, k, v = (t.unflatten(-1, (attn.heads, -1)) for t in qkv.chunk(3, dim=-1))
        q, k = attn.norm_q(q), attn.norm_k(k)
        hs = self.attend(q, k, v, image_rotary_emb, attention_mask).flatten(2, 3).to(q.dtype)
        return attn.to_out(torch.cat([hs, attn.mlp_act_fn(mlp)], dim=-1))


def install(transformer, ctx: Flux2SCAContext):
    """Install SCA processors on all 8 + 48 Flux2 attention modules (layer index = double blocks
    first, then single blocks, as in FLUX.1) and a pre-hook that feeds grid and sigma."""
    if not isinstance(ctx, Flux2SCAContext):
        raise TypeError("FLUX.2 needs a Flux2SCAContext (4-D position ids)")
    ctx.pos_embed = transformer.pos_embed
    double, single = list(transformer.transformer_blocks), list(transformer.single_transformer_blocks)
    ctx.n_layers = len(double) + len(single)
    original = dict(transformer.attn_processors)
    for i, block in enumerate(double):
        block.attn.set_processor(Flux2SCAProcessor(ctx, i))
    for i, block in enumerate(single):
        block.attn.set_processor(Flux2SCASingleProcessor(ctx, len(double) + i))

    def pre_hook(module, args, kwargs):
        if kwargs.get("kv_cache_mode") is not None:
            raise NotImplementedError("SCA does not support FLUX.2 reference-image KV caching")
        img_ids = kwargs.get("img_ids")
        sigma = None
        if ctx.scheduler is not None and getattr(ctx.scheduler, "sigmas", None) is not None:
            j = ctx.scheduler.step_index
            ctx.step = 0 if j is None else int(j)
            sigma = float(ctx.scheduler.sigmas[ctx.step])
        elif kwargs.get("timestep") is not None:
            sigma = float(kwargs["timestep"].flatten()[0])
        ctx.set_inputs(img_ids, kwargs.get("txt_ids"), sigma)
        if ctx.calib is not None:
            ctx.calib.add_sigma(ctx.sigma)

    handle = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def uninstall():
        handle.remove()
        transformer.set_attn_processor(original)
    return uninstall


# ----------------------------------------------------------------------------- schedule

def empirical_mu(seq_len, steps, mu_mode="native"):
    """mu used by Flux2Pipeline. native: sequence length capped at the 64 x 64 grid."""
    from diffusers.pipelines.flux2 import pipeline_flux2 as P
    orig = getattr(P, "_sca_orig_mu", P.compute_empirical_mu)
    if mu_mode == "native":
        return float(orig(min(int(seq_len), NATIVE_SEQ), steps))
    if mu_mode == "default":
        return float(orig(int(seq_len), steps))
    raise ValueError(mu_mode)


def set_mu(mu_mode):
    """Patch the pipeline module's compute_empirical_mu (looked up at call time in __call__)."""
    from diffusers.pipelines.flux2 import pipeline_flux2 as P
    if not hasattr(P, "_sca_orig_mu"):
        P._sca_orig_mu = P.compute_empirical_mu
    P.compute_empirical_mu = lambda image_seq_len, num_steps: empirical_mu(image_seq_len, num_steps, mu_mode)


def sigma_schedule(steps, mu, scheduler_config=None):
    """The sigmas Flux2Pipeline uses (linspace(1, 1/steps) then the scheduler's exponential shift)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    cfg = scheduler_config or {"base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 4096,
                               "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 3.0,
                               "time_shift_type": "exponential", "use_dynamic_shifting": True}
    s = FlowMatchEulerDiscreteScheduler(**cfg)
    s.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    return s.sigmas[:-1].double().numpy()


def gate_from_steps(sigmas, global_steps):
    """Threshold that makes exactly the first `global_steps` steps global-only (midpoint rule)."""
    return round(float((sigmas[global_steps - 1] + sigmas[global_steps]) / 2), 3)


# ----------------------------------------------------------------------------- loading

def snapshot():
    hub = Path(os.environ.get("HF_HOME", HF_HOME)) / "hub/models--black-forest-labs--FLUX.2-dev"
    return hub / "snapshots" / (hub / "refs/main").read_text().strip()


def _embed_key(prompt):
    payload = json.dumps([REPO, snapshot().name, prompt, MAX_SEQ, list(TEXT_LAYERS)], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_prompts(prompts, device="cuda", cache=EMBED_CACHE, log=print):
    """Mistral-3 prompt embeddings [1, 512, 15360] (bf16, CPU), cached on scratch per prompt.

    The text encoder (24B, ~45 GB) is loaded only if some prompt is missing, used, then freed
    before the caller loads the transformer, so the two never share the GPU.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    paths = [cache / f"{_embed_key(p)}.pt" for p in prompts]
    missing = [(p, path) for p, path in zip(prompts, paths) if not path.exists()]
    if missing:
        from diffusers.pipelines.flux2 import pipeline_flux2 as P
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration
        src = snapshot()
        tok = AutoProcessor.from_pretrained(str(src / "tokenizer"), local_files_only=True)
        te = Mistral3ForConditionalGeneration.from_pretrained(str(src / "text_encoder"), dtype=torch.bfloat16,
                                                              device_map=device, local_files_only=True).eval()
        try:
            for p, path in missing:
                with torch.inference_mode():
                    e = P.Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                        te, tok, p, dtype=torch.bfloat16, device=te.device, max_sequence_length=MAX_SEQ,
                        system_message=P.SYSTEM_MESSAGE, hidden_states_layers=TEXT_LAYERS)
                tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
                torch.save(e.cpu().clone(), tmp)
                os.replace(tmp, path)
            log(f"encoded {len(missing)} prompts with the text encoder")
        finally:
            del te
            free_cuda()
    return [torch.load(path, map_location="cpu") for path in paths]


def load_pipeline(device="cuda"):
    """Flux2Pipeline without text encoder / tokenizer (pass prompt_embeds); transformer loaded
    straight onto the GPU (~64 GB bf16)."""
    from diffusers import Flux2Pipeline, Flux2Transformer2DModel
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLUX.2 inference; use pytest for CPU checks")
    src = str(snapshot())
    transformer = Flux2Transformer2DModel.from_pretrained(src, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                          device_map=device, local_files_only=True)
    pipe = Flux2Pipeline.from_pretrained(src, transformer=transformer, text_encoder=None, tokenizer=None,
                                         torch_dtype=torch.bfloat16, local_files_only=True)
    return pipe.to(device)


def flops(height, width, **kw):
    """sca.attn.attention_flops with FLUX.2-dev sizes (48 heads x 128, 56 blocks, 32B params)."""
    return attention_flops(height, width, heads=48, head_dim=128, layers=56, params=32.2e9, **kw)
