"""Scale-covariant attention for Qwen-Image (diffusers QwenImageTransformer2DModel).

Same method as sca/attn.py (global branch on the coarse grid, exact native-extent local window, sigma gate, per-head
pi mix, bridge and window schedule of sca_v2, calib statistics); this module adapts only what differs from FLUX:

- Qwen-Image is a 60-block MMDiT with one attention type (QwenDoubleStreamAttnProcessor2_0): separate image and text
  projections, RMS q/k norms per stream, joint attention over [text, image], separate output projections.
- RoPE is MSRoPE (QwenEmbedRope, scale_rope=True): complex frequencies over (frame, height, width) axes of 16, 56 and
  56 dims, applied to interleaved channel pairs. Image token (r, c) of an H x W grid sits at (0, r - ceil(H/2),
  c - ceil(W/2)); text token j sits at M + j on all three axes, M = max(H // 2, W // 2). The complex product equals
  diffusers' real interleaved RoPE with cos / sin repeated per pair, so the processor converts the model's own
  frequencies once per forward and every SCA kernel of sca/attn.py runs unchanged.
- Text positions depend on the grid (M grows with the canvas). The global branch must see the geometry of a native
  image of the coarse grid's size, Hc x Wc, where text sits at Mc + j with Mc = max(Hc // 2, Wc // 2). RoPE logits
  depend only on position differences, so the coarse image ids are shifted by M - Mc on every axis instead: text keeps
  the full-resolution positions both branches share, and every text-image and image-image offset of the global branch
  is exactly that of a native Hc x Wc image. The local branch keeps the model's own positions (as for FLUX).
- Native resolution is 1328^2 (83 x 83 tokens at 16 px per token). 83 is prime, so ratio-2 and ratio-4 pooling cannot
  tile the native grid; calibration runs at 1344^2 (84 x 84 tokens, 1.2% larger), the nearest size where both ratios
  tile it, and inference uses n0 = 83, the native extent.
- The pipeline is not guidance distilled: true classifier-free guidance (scale 4, negative prompt " ", 50 steps by
  default) runs the transformer twice per step with different text lengths. SCA needs no text mask: one prompt per
  call never pads.
- The pipeline extrapolates the timestep shift mu linearly in the sequence length (mu 3.8 at 4096^2). mu_mode
  "native" caps the sequence length at the native 83 x 83 grid, the analogue of FLUX.1's max_shift protocol and FLUX.2's
  native mode.
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

REPO = "Qwen/Qwen-Image"
HF_HOME = "/scratch/09032/saini_2/tmp/hf"
EMBED_CACHE = Path("/scratch/09032/saini_2/sca/qwen_embeds")
NATIVE = 1328                          # native side in pixels
N0 = 83                                # native grid side in tokens (1328 / 16)
NATIVE_SEQ = N0 * N0
CALIB_RES = 1344                       # 84 x 84 tokens: ratios 2 and 4 tile the grid
NEGATIVE = " "                         # the model card's negative prompt for true CFG
TRUE_CFG = 4.0
STEPS = 50


# ----------------------------------------------------------------------------- MSRoPE positions

def image_ids(H, W):
    """[H * W, 3] long (frame, height, width) positions of QwenEmbedRope with scale_rope=True."""
    ids = torch.zeros(H, W, 3, dtype=torch.long)
    ids[..., 1] = (torch.arange(H) - (H - H // 2))[:, None]
    ids[..., 2] = (torch.arange(W) - (W - W // 2))[None, :]
    return ids.reshape(-1, 3)


def text_ids(H, W, L):
    """[L, 3] long: text token j at max(H // 2, W // 2) + j on every axis."""
    j = torch.arange(L, dtype=torch.long) + max(H // 2, W // 2)
    return j[:, None].expand(L, 3).contiguous()


def raw_ids(H, W):
    """[H * W, 3] long (0, row, col): grid bookkeeping for SCAContext.set_inputs (not RoPE positions)."""
    ids = torch.zeros(H, W, 3, dtype=torch.long)
    ids[..., 1] = torch.arange(H)[:, None]
    ids[..., 2] = torch.arange(W)[None, :]
    return ids.reshape(-1, 3)


def qwen_freqs(pos_embed, ids, device=None, table=None):
    """Complex frequencies [S, sum(axes_dim) / 2] of integer positions ids [S, 3], read from the module's own tables
    (pos_freqs for p >= 0, neg_freqs for p < 0), so they equal what QwenEmbedRope computes for the same positions."""
    if table is None:
        table = torch.cat([pos_embed.neg_freqs, pos_embed.pos_freqs], 0)
    table = table.to(device) if device is not None else table
    n = pos_embed.pos_freqs.shape[0]
    ids = ids.to(table.device).long()
    if int(ids.min()) < -n or int(ids.max()) >= n:
        raise ValueError(f"positions outside the RoPE tables (+-{n})")
    parts, s0 = [], 0
    for a, d in enumerate(pos_embed.axes_dim):
        parts.append(table[ids[:, a] + n, s0:s0 + d // 2])
        s0 += d // 2
    return torch.cat(parts, 1)


def rot_from_freqs(freqs):
    """Complex [S, D / 2] -> (cos, sin) float32 [S, D], each value repeated per channel pair (diffusers' real RoPE
    with use_real_unbind_dim=-1 is then the same rotation as apply_rotary_emb_qwen(use_real=False))."""
    f = torch.view_as_real(freqs)
    return f[..., 0].float().repeat_interleave(2, -1).contiguous(), f[..., 1].float().repeat_interleave(2, -1).contiguous()


# ----------------------------------------------------------------------------- context + processor

class QwenSCAContext(SCAContext):
    """SCAContext for MSRoPE: coarse ids centred like a native Hc x Wc image and shifted so text offsets match."""

    def __post_init__(self):
        super().__post_init__()
        self.forward_id = 0
        self._rot = None
        self._tables = {}
        self._raw = {}
        if self.glob_pos != "floor":
            raise ValueError("Qwen-Image port: glob_pos floor only")

    def table(self, device):
        key = str(device)
        if key not in self._tables:
            self._tables[key] = torch.cat([self.pos_embed.neg_freqs, self.pos_embed.pos_freqs], 0).to(device)
        return self._tables[key]

    def grid_ids(self, H, W):
        if (H, W) not in self._raw:
            self._raw = {(H, W): raw_ids(H, W)}
        return self._raw[(H, W)]

    def rot(self, image_rotary_emb, device):
        """(cos, sin) [L + N, D] for [text, image] from the model's own (vid_freqs, txt_freqs); converted once per
        transformer call (the tensors are shared by every block of a call)."""
        vid, txt = image_rotary_emb
        c = self._rot
        if c is None or c[0] != self.forward_id or c[1] is not vid or c[2] is not txt:
            self._rot = c = (self.forward_id, vid, txt, rot_from_freqs(torch.cat([txt, vid], 0).to(device)))
        return c[3]

    def coarse_shift(self, f):
        H, W = self.grid
        Hc, Wc = H // f[0], W // f[1]
        return max(H // 2, W // 2) - max(Hc // 2, Wc // 2)

    def coarse_rope(self, f, device):
        """(rope_q, rope_k): image queries at their block (floor(row / fh), floor(col / fw)), pooled keys at block
        centres, both at the centred ids of a native Hc x Wc image plus the text shift M - Mc on every axis."""
        key = (f, self.glob_pos, str(device))
        if key not in self._rope_cache:
            H, W = self.grid
            fh, fw = f
            Hc, Wc = H // fh, W // fw
            d = self.coarse_shift(f)
            q_ids = torch.empty(H, W, 3, dtype=torch.long)
            q_ids[..., 0] = d
            q_ids[..., 1] = (torch.arange(H) // fh - (Hc - Hc // 2) + d)[:, None]
            q_ids[..., 2] = (torch.arange(W) // fw - (Wc - Wc // 2) + d)[None, :]
            k_ids = q_ids[fh // 2::fh, fw // 2::fw].reshape(-1, 3)
            tab = self.table(device)
            self._rope_cache[key] = (rot_from_freqs(qwen_freqs(self.pos_embed, q_ids.reshape(-1, 3), table=tab)),
                                     rot_from_freqs(qwen_freqs(self.pos_embed, k_ids, table=tab)))
        return self._rope_cache[key]


class QwenSCAProcessor(SCAProcessor):
    """Drop-in for QwenDoubleStreamAttnProcessor2_0; same outputs in mode full."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                 attention_mask=None, image_rotary_emb=None):
        ctx = self.ctx
        if encoder_hidden_states is None:
            raise ValueError("Qwen-Image attention needs encoder_hidden_states (text stream)")
        if attention_mask is not None:
            raise ValueError("pass encoder_hidden_states_mask, as for the stock processor")
        mask = None
        if encoder_hidden_states_mask is not None and not bool(encoder_hidden_states_mask.all()):
            if ctx.mode != "full":
                raise NotImplementedError("padded text needs a joint mask; SCA runs one prompt per transformer call")
            img = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
            mask = torch.cat([encoder_hidden_states_mask.bool(), img], 1)[:, None, None, :]
        L = encoder_hidden_states.shape[1]
        q = torch.cat([attn.add_q_proj(encoder_hidden_states), attn.to_q(hidden_states)], 1).unflatten(-1, (attn.heads, -1))
        k = torch.cat([attn.add_k_proj(encoder_hidden_states), attn.to_k(hidden_states)], 1).unflatten(-1, (attn.heads, -1))
        v = torch.cat([attn.add_v_proj(encoder_hidden_states), attn.to_v(hidden_states)], 1).unflatten(-1, (attn.heads, -1))
        # q/k norms per stream (RMSNorm over the head dim acts per token, so norming the split parts is exact)
        nq = [n or (lambda x: x) for n in (attn.norm_added_q, attn.norm_q, attn.norm_added_k, attn.norm_k)]
        q = torch.cat([nq[0](q[:, :L]), nq[1](q[:, L:])], 1)
        k = torch.cat([nq[2](k[:, :L]), nq[3](k[:, L:])], 1)
        rot = ctx.rot(image_rotary_emb, q.device)
        if ctx.mode == "full" or ctx.grid is None:
            if ctx.mode != "full":
                raise RuntimeError("SCA context has no image grid; install the transformer pre-hook")
            out = sdpa(rope(q, rot), rope(k, rot), v, attn_mask=mask)
        else:
            out = self.sca(q, k, v, rot) if ctx.mode == "sca" else self.calibrate(q, k, v, rot)
        hs = out.flatten(2, 3).to(q.dtype)
        txt, img = hs.split_with_sizes([L, hs.shape[1] - L], 1)
        img = attn.to_out[0](img.contiguous())
        if len(attn.to_out) > 1:
            img = attn.to_out[1](img)
        return img, attn.to_add_out(txt.contiguous())


def install(transformer, ctx: QwenSCAContext):
    """Install SCA processors on all 60 Qwen-Image blocks and a pre-hook that feeds grid and sigma."""
    if not isinstance(ctx, QwenSCAContext):
        raise TypeError("Qwen-Image needs a QwenSCAContext (MSRoPE positions)")
    ctx.pos_embed = transformer.pos_embed
    if type(ctx.pos_embed).__name__ != "QwenEmbedRope" or not ctx.pos_embed.scale_rope:
        raise NotImplementedError("Qwen-Image port: QwenEmbedRope with scale_rope=True only")
    blocks = list(transformer.transformer_blocks)
    ctx.n_layers = len(blocks)
    original = dict(transformer.attn_processors)
    for i, block in enumerate(blocks):
        block.attn.set_processor(QwenSCAProcessor(ctx, i))

    def pre_hook(module, args, kwargs):
        shapes = kwargs.get("img_shapes")
        if shapes is None:
            raise ValueError("SCA needs img_shapes (the grid) in the transformer call")
        first = shapes[0] if isinstance(shapes[0], list) else shapes
        if len(first) != 1 or any(s != first[0] for s in (x[0] if isinstance(x, list) else x for x in shapes)):
            raise NotImplementedError("SCA for Qwen-Image: one image shape per call (no edit / layered inputs)")
        frames, H, W = first[0]
        if frames != 1:
            raise NotImplementedError("SCA for Qwen-Image: single frame")
        ctx.known_grid = (int(H), int(W))
        sigma = None
        if ctx.scheduler is not None and getattr(ctx.scheduler, "sigmas", None) is not None:
            j = ctx.scheduler.step_index
            ctx.step = 0 if j is None else int(j)
            sigma = float(ctx.scheduler.sigmas[ctx.step])
        elif kwargs.get("timestep") is not None:
            sigma = float(kwargs["timestep"].flatten()[0])
        ctx.set_inputs(ctx.grid_ids(int(H), int(W)), None, sigma)
        ctx.forward_id += 1
        if ctx.calib is not None:
            ctx.calib.add_sigma(ctx.sigma)

    handle = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def uninstall():
        handle.remove()
        transformer.set_attn_processor(original)
    return uninstall


# ----------------------------------------------------------------------------- schedule

def scheduler_config():
    return {"base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": False, "max_image_seq_len": 8192,
            "max_shift": 0.9, "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": 0.02,
            "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
            "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False}


def mu_for(seq_len, mu_mode="native", cfg=None):
    """mu QwenImagePipeline uses: calculate_shift of the sequence length (native: capped at the 83 x 83 grid)."""
    from diffusers.pipelines.qwenimage import pipeline_qwenimage as P
    orig = getattr(P, "_sca_orig_shift", P.calculate_shift)
    c = cfg or scheduler_config()
    if mu_mode not in ("native", "default"):
        raise ValueError(mu_mode)
    n = min(int(seq_len), NATIVE_SEQ) if mu_mode == "native" else int(seq_len)
    return float(orig(n, c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 4096), c.get("base_shift", 0.5),
                      c.get("max_shift", 1.15)))


def set_mu(mu_mode):
    """Patch the pipeline module's calculate_shift (looked up at call time in __call__)."""
    from diffusers.pipelines.qwenimage import pipeline_qwenimage as P
    if not hasattr(P, "_sca_orig_shift"):
        P._sca_orig_shift = P.calculate_shift
    if mu_mode == "default":
        P.calculate_shift = P._sca_orig_shift
        return
    if mu_mode != "native":
        raise ValueError(mu_mode)
    P.calculate_shift = lambda image_seq_len, *a, **kw: P._sca_orig_shift(min(int(image_seq_len), NATIVE_SEQ), *a, **kw)


def sigma_schedule(steps, mu, cfg=None):
    """The sigmas QwenImagePipeline uses (linspace(1, 1/steps), exponential shift by mu, stretch to shift_terminal)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    s = FlowMatchEulerDiscreteScheduler(**(cfg or scheduler_config()))
    s.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    return s.sigmas[:-1].double().numpy()


def gate_from_steps(sigmas, global_steps):
    """Threshold that makes exactly the first `global_steps` steps global-only (midpoint rule, as sca/flux2.py)."""
    return round(float((sigmas[global_steps - 1] + sigmas[global_steps]) / 2), 5)


def win_sched_from_steps(sigmas, first, sizes, per_step=1):
    """--win-sched pairs (sigma threshold, window) giving window sizes[i] on steps first + i * per_step ... (per_step
    steps each); thresholds are midpoints below each size's last step."""
    out = []
    for i, w in enumerate(sizes):
        last = first + (i + 1) * per_step - 1
        out.append((round(float((sigmas[last] + sigmas[last + 1]) / 2), 5), int(w)))
    return out


# ----------------------------------------------------------------------------- loading

def snapshot():
    hub = Path(os.environ.get("HF_HOME", HF_HOME)) / "hub/models--Qwen--Qwen-Image"
    ref = hub / "refs/main"
    if ref.exists():
        return hub / "snapshots" / ref.read_text().strip()
    snaps = sorted(p for p in (hub / "snapshots").iterdir() if p.is_dir())
    if len(snaps) != 1:
        raise FileNotFoundError(f"expected one Qwen-Image snapshot under {hub}, found {len(snaps)}")
    return snaps[0]


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _embed_key(prompt):
    import diffusers
    payload = json.dumps([REPO, snapshot().name, prompt, "QwenImagePipeline.encode_prompt", diffusers.__version__],
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def encode_prompts(prompts, device="cuda", cache=EMBED_CACHE, log=print, pipe=None):
    """Qwen2.5-VL prompt embeddings [1, L, 3584] (bf16, CPU), cached on scratch per prompt. The text encoder (16 GB)
    is loaded only if some prompt is missing (or taken from `pipe`), used, then freed."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    paths = [cache / f"{_embed_key(p)}.pt" for p in prompts]
    missing = [(p, path) for p, path in zip(prompts, paths) if not path.exists()]
    if missing:
        own = pipe is None or pipe.text_encoder is None
        if own:
            from diffusers import QwenImagePipeline
            from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
            src = snapshot()
            te = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(src / "text_encoder"), dtype=torch.bfloat16,
                                                                    device_map=device, local_files_only=True).eval()
            tok = AutoTokenizer.from_pretrained(str(src / "tokenizer"), local_files_only=True)
            enc = QwenImagePipeline(scheduler=None, vae=None, text_encoder=te, tokenizer=tok, transformer=None)
        else:
            enc = pipe
        try:
            for p, path in missing:
                with torch.inference_mode():
                    e, m = enc.encode_prompt(prompt=p, device=torch.device(device) if isinstance(device, str) else device)
                if m is not None and not bool(m.all()):
                    raise RuntimeError("single prompt produced a padded mask")
                tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
                torch.save(e.to(torch.bfloat16).cpu().clone(), tmp)
                os.replace(tmp, path)
            log(f"encoded {len(missing)} prompts with the text encoder")
        finally:
            if own:
                del enc, te
                free_cuda()
    return [torch.load(path, map_location="cpu") for path in paths]


def load_pipeline(device="cuda"):
    """QwenImagePipeline without text encoder / tokenizer (pass prompt_embeds); transformer loaded straight onto the
    GPU (~41 GB bf16)."""
    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen-Image inference; use pytest for CPU checks")
    src = str(snapshot())
    transformer = QwenImageTransformer2DModel.from_pretrained(src, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                              device_map=device, local_files_only=True)
    pipe = QwenImagePipeline.from_pretrained(src, transformer=transformer, text_encoder=None, tokenizer=None,
                                             torch_dtype=torch.bfloat16, local_files_only=True)
    return pipe.to(device)


def flops(height, width, **kw):
    """sca.attn.attention_flops with Qwen-Image sizes (24 heads x 128, 60 blocks, image-stream weights ~6.8B per token,
    n0 = 83, text ~ 100 tokens, two transformer calls per step at 50 steps)."""
    kw.setdefault("n0", N0)
    kw.setdefault("text", 100)
    kw.setdefault("steps", 2 * STEPS)
    return attention_flops(height, width, heads=24, head_dim=128, layers=60, params=6.8e9, **kw)
