"""Scale-covariant attention for Ideogram 4 (ideogram-ai/ideogram-4-nf4-diffusers).

Same method as sca/attn.py (global branch on the coarse grid, exact native-extent local window, sigma gate, per-head
pi mix, bridge and window schedule of sca_v2, reach-emulation statistics); this module adapts only what differs:

- One code path for every method. Full attention, NTK, YaRN, SEGA and SCA all run the pinned SEGA copies of the
  Ideogram 4 pipeline and transformer (third_party/sega/ideogram4_sega). Those copies carry the padding trim and the
  maskless attention path (a dense mask over more than 65,536 tokens overflows 32-bit indexing and corrupts the bottom
  rows) that the diffusers 0.39 build lacks. NTK, YaRN and SEGA are its set_rope_scaling methods; SCA runs with RoPE
  scaling off.
- Ideogram 4 is a 34-block single-stream transformer (18 heads x 256) over one packed sequence [text][image]. Its CFG
  is asymmetric: a conditional transformer sees [text][image], a separately trained unconditional transformer sees
  [image] only (no text tokens, L = 0). Both see the full canvas, so both run SCA, each with its own reach table.
- RoPE is interleaved M-RoPE (Qwen3-VL style): one ladder of 128 frequencies (base 5e6); h owns indices 1, 4, ..., 58,
  w owns 2, 5, ..., 59, t the other 88. Text token j sits at (j, j, j); image token (r, c) at (65536, 65536 + r,
  65536 + c). The model applies it in the rotate-half layout (channel j pairs with j + 128). The processor permutes q
  and k channels once per call so that pair j becomes channels (2j, 2j + 1), and repeats each frequency per pair;
  that is diffusers' real interleaved RoPE, so every SCA kernel of sca/attn.py runs unchanged (logits are invariant
  to a common channel permutation of q and k; values are not permuted).
- Coarse positions of the global branch change only the h and w combs: image queries sit at (65536, 65536 + r // fh,
  65536 + c // fw), pooled keys at their block's coarse id, and the t component (the 65536 text-image offset) and the
  text tokens keep their positions. Every text-image and image-image phase of the global branch is then exactly that
  of a native Hc x Wc image, as the model saw in training. The local branch keeps the model's own positions.
- The native extent is 4096^2 (256 x 256 tokens at 16 px per token; the SEGA README's training-range assumption), so
  n0 = 256 and 8192^2 is ratio 2.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

from sca.attn import SCAContext, SCAProcessor, attention_flops

REPO = "ideogram-ai/ideogram-4-nf4-diffusers"
HEAD_REPO = "diffusers/qwen3-vl-8b-instruct-lm-head"     # prompt-enhancer LM head (caption upsampling only)
HF_HOME = "/scratch/09032/saini_2/tmp/hf"
SEGA_DIR = Path(__file__).resolve().parents[1] / "third_party" / "sega" / "ideogram4_sega"
CAPTIONS = Path("/scratch/09032/saini_2/sca/ideogram4/captions")
IMAGE_POSITION_OFFSET = 65536
TEXT, IMAGE = 3, 2                       # LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR of the transformer
NATIVE = 4096                            # native side in pixels (SEGA README: train_grid 256)
N0 = 256                                 # native grid side in tokens
STEPS = 48
GUIDANCE_SCHEDULE = (7.0,) * 45 + (3.0,) * 3   # pipeline default for 48 steps
MAX_TEXT = 2048                          # pipeline max_sequence_length


def sega_modules():
    """(transformer module, pipeline module) of the pinned SEGA copy; imported once by file name."""
    if str(SEGA_DIR) not in sys.path:
        sys.path.insert(0, str(SEGA_DIR))
    import pipeline_ideogram4 as P  # noqa: E402
    import transformer_ideogram4 as T  # noqa: E402
    return T, P


# ----------------------------------------------------------------------------- M-RoPE layout

def pair_perm(head_dim, device=None):
    """Channel order mapping the rotate-half pairs (j, j + D/2) onto adjacent pairs (2j, 2j + 1)."""
    h = head_dim // 2
    return torch.stack([torch.arange(h), torch.arange(h) + h], 1).reshape(-1).to(device)


def to_interleaved(cos, sin):
    """Model (cos, sin) [..., S, D], each cat(c, c), -> diffusers' interleaved layout [..., S, D] (c_j at 2j, 2j + 1)."""
    h = cos.shape[-1] // 2
    return cos[..., :h].repeat_interleave(2, -1), sin[..., :h].repeat_interleave(2, -1)


def rope_rotate_half(x, cos, sin):
    """The model's own application (Ideogram4AttnProcessor): x [B, S, h, D], cos / sin [B, S, D]."""
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


def image_positions(H, W):
    """[H * W, 3] long M-RoPE positions of the pipeline's image tokens (t, h, w) = 65536 + (0, r, c), row-major."""
    r = torch.arange(H).view(-1, 1).expand(H, W).reshape(-1)
    c = torch.arange(W).view(1, -1).expand(H, W).reshape(-1)
    return torch.stack([torch.zeros_like(r), r, c], 1) + IMAGE_POSITION_OFFSET


def text_positions(L):
    """[L, 3] long: text token j at (j, j, j) (plain 1-D RoPE over all 128 frequencies)."""
    j = torch.arange(L)
    return torch.stack([j, j, j], 1)


def coarse_positions(H, W, f):
    """(q_ids [H * W, 3], k_ids [Hc * Wc, 3]): image queries at 65536 + (0, r // fh, c // fw), pooled keys (block
    centres) at their block's id. Only the h and w components change; t keeps the 65536 offset."""
    fh, fw = f
    q = image_positions(H, W).view(H, W, 3).clone()
    q[..., 1] = IMAGE_POSITION_OFFSET + (torch.arange(H) // fh)[:, None]
    q[..., 2] = IMAGE_POSITION_OFFSET + (torch.arange(W) // fw)[None, :]
    k = q[fh // 2::fh, fw // 2::fw].reshape(-1, 3)
    return q.reshape(-1, 3), k


def raw_ids(H, W):
    """[H * W, 3] long (0, row, col): grid bookkeeping for SCAContext.set_inputs (not RoPE positions)."""
    ids = torch.zeros(H, W, 3, dtype=torch.long)
    ids[..., 1] = torch.arange(H)[:, None]
    ids[..., 2] = torch.arange(W)[None, :]
    return ids.reshape(-1, 3)


def packed_layout(position_ids, indicator, segment_ids):
    """(L, H, W) of one packed call [text (L)][image (H x W)], checked against the pipeline's layout. Raises for
    padding, mixed segments (SCA has no segment mask), batch rows with different layouts or non-grid image ids."""
    if position_ids.ndim != 3 or position_ids.shape[-1] != 3:
        raise ValueError(f"position_ids must be [B, S, 3], got {tuple(position_ids.shape)}")
    if not bool(torch.all(segment_ids == segment_ids[:, :1])):
        raise NotImplementedError("SCA needs uniform segment ids (one prompt per call, padding trimmed)")
    if position_ids.shape[0] > 1 and not (torch.equal(position_ids, position_ids[:1].expand_as(position_ids))
                                          and torch.equal(indicator, indicator[:1].expand_as(indicator))):
        raise NotImplementedError("SCA needs the same packed layout for every batch row")
    ind, pos = indicator[0], position_ids[0]
    img = ind == IMAGE
    n_img = int(img.sum())
    L = int(ind.numel()) - n_img
    if n_img == 0 or not bool(img[L:].all()) or (L and not bool((ind[:L] == TEXT).all())):
        raise ValueError("expected the packed layout [text][image] with no padding tokens")
    p = pos[L:].long() - IMAGE_POSITION_OFFSET
    if bool((p[:, 0] != 0).any()):
        raise ValueError("image tokens must have t = 65536")
    H, W = int(p[:, 1].max()) + 1, int(p[:, 2].max()) + 1
    if H * W != n_img or not torch.equal(p.cpu(), image_positions(H, W) - IMAGE_POSITION_OFFSET):
        raise ValueError("image position ids do not form the pipeline's row-major grid")
    if L:
        t = pos[:L].long()
        if not (torch.equal(t[:, 0], t[:, 1]) and torch.equal(t[:, 0], t[:, 2])) or int(t.max()) >= IMAGE_POSITION_OFFSET:
            raise ValueError("text tokens must sit at (j, j, j) below the image offset")
    return L, H, W


# ----------------------------------------------------------------------------- context + processor

class Ideogram4SCAContext(SCAContext):
    """SCAContext for interleaved M-RoPE with the 65536 image offset."""

    def __post_init__(self):
        super().__post_init__()
        if self.glob_pos != "floor":
            raise ValueError("Ideogram 4 port: glob_pos floor only")
        self.forward_id = 0
        self.text_len = 0
        self.rope_dtype = torch.bfloat16
        # flex kernel options of the local branch. sca/attn.py forces BLOCK_M 128 / BLOCK_N 64 with TMA, tuned for
        # head_dim 128 (FLUX); at head_dim 256 those tiles can exceed shared memory, so {} keeps torch's own
        # head-dim-256 configurations (their tiles divide the 128 x 64 sparse blocks). Same attention either way.
        self.flex_kernel = {}
        self._rot = None
        self._raw = {}
        self._perm = {}

    def flex_layout(self, L, tile, win, device):
        """sca/attn.py flex_layout with this context's kernel options, cached per (prefix, grid, tile, window)."""
        from sca.attn import flex_layout
        key = (L, self.grid, tuple(tile), tuple(win), str(device))
        if key not in self._flex_cache:
            self._flex_cache[key] = flex_layout(L, self.grid, tile, win, device, kernel_options=self.flex_kernel)
        return self._flex_cache[key]

    def grid_ids(self, H, W):
        if (H, W) not in self._raw:
            self._raw = {(H, W): raw_ids(H, W)}
        return self._raw[(H, W)]

    def perm(self, head_dim, device):
        key = (head_dim, str(device))
        if key not in self._perm:
            self._perm[key] = pair_perm(head_dim, device)
        return self._perm[key]

    def rot(self, image_rotary_emb, device):
        """(cos, sin) [S, D] in the interleaved layout from the model's (cos, sin) [B, S, D] (batch rows are equal,
        checked by the pre-hook); converted once per transformer call, shared by every block of the call."""
        cos, sin = image_rotary_emb
        c = self._rot
        if c is None or c[0] != self.forward_id or c[1] is not cos or c[2] is not sin:
            self.rope_dtype = cos.dtype
            self._rot = c = (self.forward_id, cos, sin, to_interleaved(cos[0].to(device), sin[0].to(device)))
        return c[3]

    def mrope(self, ids, device):
        """Model-exact (cos, sin) of integer M-RoPE positions ids [S, 3]: the transformer's own Ideogram4MRoPE in
        float32, cast to the dtype the transformer passes to its blocks, in the interleaved layout."""
        cos, sin = self.pos_embed(ids[None].to(device))
        return to_interleaved(cos[0].to(self.rope_dtype), sin[0].to(self.rope_dtype))

    def coarse_rope(self, f, device):
        """(rope_q, rope_k) of the global branch: coarse_positions on the h / w combs, t and text unchanged."""
        key = (f, self.glob_pos, str(device), self.rope_dtype)
        if key not in self._rope_cache:
            q_ids, k_ids = coarse_positions(*self.grid, f)
            self._rope_cache[key] = (self.mrope(q_ids, device), self.mrope(k_ids, device))
        return self._rope_cache[key]


class Ideogram4SCAProcessor(SCAProcessor):
    """Drop-in for Ideogram4AttnProcessor. mode full calls the stock processor itself (bit-identical); sca and calib
    run sca/attn.py's kernels on channel-permuted q / k with the interleaved RoPE."""

    def __init__(self, ctx, layer, stock):
        super().__init__(ctx, layer)
        self.stock = stock

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        ctx = self.ctx
        if ctx.mode == "full":
            return self.stock(attn, hidden_states, attention_mask, image_rotary_emb)
        if ctx.grid is None:
            raise RuntimeError("SCA context has no image grid; install the transformer pre-hook")
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported by SCA")
        shape = (attn.num_heads, attn.head_dim)
        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, shape))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, shape))
        v = attn.to_v(hidden_states).unflatten(-1, shape)
        p = ctx.perm(attn.head_dim, q.device)
        q, k = q.index_select(-1, p), k.index_select(-1, p)
        rot = ctx.rot(image_rotary_emb, q.device)
        out = self.sca(q, k, v, rot) if ctx.mode == "sca" else self.calibrate(q, k, v, rot)
        return attn.to_out[0](out.flatten(2, 3).to(q.dtype))


def install(transformer, ctx: Ideogram4SCAContext, scheduler=None):
    """Install SCA processors on all blocks of one Ideogram 4 transformer and a pre-hook that feeds grid, text length
    and sigma (from `scheduler` if given, else 1 - t of the call's timestep). Returns uninstall()."""
    if not isinstance(ctx, Ideogram4SCAContext):
        raise TypeError("Ideogram 4 needs an Ideogram4SCAContext (M-RoPE positions)")
    if getattr(transformer.rotary_emb, "rope_scaling", None) is not None:
        raise ValueError("SCA runs on the unmodified RoPE: set_rope_scaling(None) first")
    ctx.pos_embed = transformer.rotary_emb
    if scheduler is not None:
        ctx.scheduler = scheduler
    blocks = list(transformer.layers)
    ctx.n_layers = len(blocks)
    original = dict(transformer.attn_processors)
    for i, block in enumerate(blocks):
        block.attention.set_processor(Ideogram4SCAProcessor(ctx, i, block.attention.processor))

    def pre_hook(module, args, kwargs):
        if args:
            raise ValueError("call the Ideogram 4 transformer with keyword arguments (as the pipeline does)")
        L, H, W = packed_layout(kwargs["position_ids"], kwargs["indicator"], kwargs["segment_ids"])
        ctx.known_grid, ctx.text_len = (H, W), L
        sigma = None
        if ctx.scheduler is not None and getattr(ctx.scheduler, "sigmas", None) is not None:
            j = ctx.scheduler.step_index
            ctx.step = 0 if j is None else int(j)
            sigma = float(ctx.scheduler.sigmas[ctx.step])
        elif kwargs.get("timestep") is not None:
            sigma = 1.0 - float(kwargs["timestep"].flatten()[0])   # model time: 0 = noise, 1 = data
        ctx.set_inputs(ctx.grid_ids(H, W), None, sigma)
        ctx.forward_id += 1
        if ctx.calib is not None:
            ctx.calib.add_sigma(ctx.sigma)

    handle = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def uninstall():
        handle.remove()
        transformer.set_attn_processor(original)
    return uninstall


# ----------------------------------------------------------------------------- schedule

def sigma_schedule(steps=STEPS, height=NATIVE, width=NATIVE, mu=0.0, std=1.5):
    """The sigmas the pipeline uses: logit-normal schedule with mu shifted by 0.5 log(pixels / 512^2)."""
    _, P = sega_modules()
    m = P._resolution_aware_mu(height=height, width=width, base_mu=mu)
    return P._logit_normal_sigmas(steps, m, std=std).double().numpy()


def gate_from_steps(sigmas, global_steps):
    """Threshold that makes exactly the first `global_steps` steps global-only (midpoint rule, as sca/qwen.py)."""
    return round(float((sigmas[global_steps - 1] + sigmas[global_steps]) / 2), 5)


def win_sched_from_steps(sigmas, first, sizes, per_step=1):
    """--win-sched pairs (sigma threshold, window): window sizes[i] on steps first + i * per_step ... (per_step steps
    each); thresholds are midpoints below each size's last step (as sca/qwen.py)."""
    out = []
    for i, w in enumerate(sizes):
        last = first + (i + 1) * per_step - 1
        out.append((round(float((sigmas[last] + sigmas[last + 1]) / 2), 5), int(w)))
    return out


# ----------------------------------------------------------------------------- loading

def snapshot(repo=REPO):
    hub = Path(os.environ.get("HF_HOME", HF_HOME)) / "hub" / ("models--" + repo.replace("/", "--"))
    ref = hub / "refs" / "main"
    if ref.exists():
        return hub / "snapshots" / ref.read_text().strip()
    snaps = sorted(p for p in (hub / "snapshots").iterdir() if p.is_dir())
    if len(snaps) != 1:
        raise FileNotFoundError(f"expected one snapshot under {hub}, found {len(snaps)}")
    return snaps[0]


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_text_encoder(device="cuda"):
    """(tokenizer, nf4 Qwen3-VL text encoder) as run_ideogram4.py loads them, moved to `device`."""
    from transformers import AutoTokenizer, Qwen3VLModel
    src = snapshot()
    tok = AutoTokenizer.from_pretrained(str(src / "tokenizer"), local_files_only=True)
    te = Qwen3VLModel.from_pretrained(str(src / "text_encoder"), torch_dtype=torch.bfloat16, local_files_only=True)
    return tok, te.to(device).eval()


def load_pipeline(device="cuda", guidance_mode="dual"):
    """SEGA-copy Ideogram4Pipeline with both nf4 transformers, nf4 text encoder, VAE and scheduler on `device`
    (run_ideogram4.py's loading, single GPU). No prompt-enhancer head: captions are precomputed."""
    from diffusers.models.autoencoders import AutoencoderKLFlux2
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    T, P = sega_modules()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Ideogram 4 inference; use pytest for CPU checks")
    src = snapshot()
    tok, te = load_text_encoder("cpu")
    load = lambda sub: T.Ideogram4Transformer2DModel.from_pretrained(str(src / sub), torch_dtype=torch.bfloat16,
                                                                     local_files_only=True)
    pipe = P.Ideogram4Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(src / "scheduler"), local_files_only=True),
        vae=AutoencoderKLFlux2.from_pretrained(str(src / "vae"), torch_dtype=torch.bfloat16, local_files_only=True),
        text_encoder=te, tokenizer=tok, transformer=load("transformer"),
        unconditional_transformer=load("unconditional_transformer") if guidance_mode == "dual" else None,
        prompt_enhancer_head=None)
    return pipe.to(device)


@torch.no_grad()
def packed_inputs(pipe, prompt, height, width, latents, t_model):
    """(cond kwargs, uncond kwargs) of one transformer step exactly as Ideogram4Pipeline.__call__ builds them for a
    single prompt (encode, trim the shared padding, image-only slices for the unconditional pass); latents [1, N, 128]
    float32, t_model the model time in [0, 1] (0 = noise)."""
    device = pipe._execution_device
    gh, gw = height // 16, width // 16
    feats, pos, seg, ind = pipe.encode_prompt(prompt=prompt, grid_h=gh, grid_w=gw, max_sequence_length=MAX_TEXT,
                                              device=device)
    first = int((seg != -1).any(dim=0).to(torch.long).argmax())
    feats, pos, seg, ind = feats[:, first:], pos[:, first:], seg[:, first:], ind[:, first:]
    n = gh * gw
    L = pos.shape[1] - n
    dt = pipe.transformer.dtype
    t = torch.full((1,), float(t_model), device=device).to(dt)
    z = torch.cat([torch.zeros(1, L, latents.shape[-1], device=device), latents.to(device)], 1).to(dt)
    cond = dict(hidden_states=z, timestep=t, encoder_hidden_states=feats.to(dt), position_ids=pos, segment_ids=seg,
                indicator=ind, return_dict=False)
    uncond = dict(hidden_states=latents.to(device).to(dt), timestep=t, encoder_hidden_states=None,
                  position_ids=pos[:, -n:], segment_ids=seg[:, -n:], indicator=ind[:, -n:], return_dict=False)
    return cond, uncond


def transformers_of(pipe):
    """[("cond", transformer), ("uncond", unconditional_transformer)] (the second only in dual mode)."""
    out = [("cond", pipe.transformer)]
    if pipe.unconditional_transformer is not None:
        out.append(("uncond", pipe.unconditional_transformer))
    return out


def set_rope_scaling(pipe, method, train_grid=N0):
    """NTK / YaRN / SEGA (run_ideogram4.py --rope-scaling) on both transformers; None / "none" restores the RoPE."""
    for _, tf in transformers_of(pipe):
        tf.set_rope_scaling(method=method, **({} if method in (None, "none") else {"train_grid": int(train_grid)}))


# ----------------------------------------------------------------------------- captions

def caption_key(prompt, aspect, seed):
    """Cache key of one upsampled caption (the upsampler sees only the aspect ratio, not the size)."""
    payload = json.dumps([REPO, snapshot().name, HEAD_REPO, prompt, aspect, int(seed)], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def aspect_of(height, width):
    g = math.gcd(width, height) or 1
    return f"{width // g}:{height // g}"


def valid_caption(text):
    """True when text is the structured JSON caption the upsampler's schema asks for."""
    try:
        c = json.loads(text)
        comp = c["compositional_deconstruction"]
        els = comp["elements"]
        return (isinstance(c["high_level_description"], str) and isinstance(comp["background"], str)
                and isinstance(els, list) and len(els) > 0
                and all(isinstance(e, dict) and e.get("type") in ("obj", "text") and isinstance(e.get("desc"), str)
                        for e in els))
    except (ValueError, KeyError, TypeError):
        return False


# ----------------------------------------------------------------------------- cost model

def flops(height, width, **kw):
    """sca.attn.attention_flops with Ideogram 4 sizes: 18 heads x 256, 34 blocks, ~9.3B transformer parameters, n0
    = 256, ~600 text tokens, two transformer calls per step (conditional + unconditional) at 48 steps."""
    kw.setdefault("n0", N0)
    kw.setdefault("text", 600)
    kw.setdefault("steps", 2 * STEPS)
    return attention_flops(height, width, heads=18, head_dim=256, layers=34, params=9.3e9, **kw)
