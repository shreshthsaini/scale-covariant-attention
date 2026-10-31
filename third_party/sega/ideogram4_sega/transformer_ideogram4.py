# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---------------------------------------------------------------------------

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import logging

try:
    from diffusers.utils import apply_lora_scale
except ImportError:
    def apply_lora_scale(_name):
        def decorator(fn):
            return fn
        return decorator

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Per-token role indicators used to label entries of the packed text+image sequence.
SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3

# Image grid coordinates start at this offset so they never collide with text token indices.
IMAGE_POSITION_OFFSET = 65536


# Defaults for the rope-scaling config consumed by `Ideogram4MRoPE`. YaRN ramp
# thresholds (beta/gamma, in rotations at the training extent) and the SEGA
# spectral-allocation hyperparameters follow the grounded Flux/Qwen reference
# implementations in project/high-resolution/length_extrapolation.
ROPE_SCALING_DEFAULTS = {
    "method": "ntk",        # "ntk" | "yarn" | "sega"
    "train_grid": 256,      # trained grid extent in latent patches (256 = 4096 px)
    # YaRN two-pass ramp thresholds, expressed in rotations at train_grid.
    "beta_0": 1.25,
    "beta_1": 0.75,
    "gamma_0": 16.0,
    "gamma_1": 2.0,
    # SEGA spectral mscale allocation.
    "mscale_alpha": 0.25,
    "mscale_beta": 1.5,
    "mscale_min": 1.0,
    "base_mscale_coefficient": 0.1,
    "spread_min": 0.0,
    "spread_max": 1.0,
    "spread_alpha": 1.5,
}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@torch.no_grad()
def _sega_allocation(
    energy_profile: torch.Tensor,
    freqs: torch.Tensor,
    base_mscale: float,
    spread: float,
    alpha: float = 0.25,
    beta: float = 1.5,
    min_mscale: float = 1.0,
) -> torch.Tensor:
    """Per-frequency mscale allocation from a spectral energy profile (SEGA).

    Each RoPE frequency is mapped (by its period, log-scale) onto the bins of
    `energy_profile`; frequencies in high-energy bands get their mscale reduced
    below `base_mscale`, low-energy bands raised above it, modulated by
    `spread` and squashed through tanh(beta * z).
    """
    d_half = freqs.shape[0]
    eps = 1e-8

    if spread <= 0.0 or alpha <= 0.0:
        return torch.full((d_half,), float(base_mscale), device=freqs.device, dtype=torch.float32)

    periods = 2.0 * math.pi / freqs.clamp(min=eps)
    log_periods = torch.log(periods)
    min_lp, max_lp = log_periods.min(), log_periods.max()
    if (max_lp - min_lp).item() > 1e-6:
        lp_norm = (log_periods - min_lp) / (max_lp - min_lp)
    else:
        lp_norm = torch.zeros_like(log_periods)

    n_bins = energy_profile.shape[0]
    bin_pos = (1.0 - lp_norm) * (n_bins - 1)
    j_low = bin_pos.floor().long().clamp(0, n_bins - 1)
    j_high = (j_low + 1).clamp(0, n_bins - 1)
    frac = (bin_pos - j_low.to(bin_pos.dtype)).clamp(0.0, 1.0)

    energy = energy_profile.to(freqs.device).clamp(min=eps)
    log_energy = torch.log(energy)
    raw = log_energy[j_low] * (1.0 - frac) + log_energy[j_high] * frac

    z = raw - raw.mean()
    z = z / z.std().clamp(min=eps)
    s = torch.tanh(float(beta) * z)
    s = s - s.mean()

    m = float(base_mscale) * (1.0 - float(alpha) * float(spread) * s)
    return m.clamp(min=float(min_mscale)).to(torch.float32)


@torch.no_grad()
def compute_spectral_energy_profile(image_states: torch.Tensor, height: int, width: int, n_bins: int) -> torch.Tensor:
    """Radially-binned 2D power spectrum of the image-token hidden states (B, H*W, C)."""
    batch_size = image_states.shape[0]
    spatial = image_states[:, : height * width].reshape(batch_size, height, width, -1)
    spatial_map = spatial.float().mean(dim=(0, -1))
    spatial_map = spatial_map - spatial_map.mean()

    power = torch.fft.fftshift(torch.fft.fft2(spatial_map)).abs().pow(2)

    cy, cx = height / 2.0, width / 2.0
    y = torch.arange(height, device=power.device, dtype=torch.float32) - cy
    x = torch.arange(width, device=power.device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius_norm = torch.sqrt(yy**2 + xx**2)
    radius_norm = radius_norm / (radius_norm.max() + 1e-8)

    bin_idx = (radius_norm * n_bins).long().clamp(0, n_bins - 1).flatten()
    flat_pw = power.flatten()

    energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_sum.scatter_add_(0, bin_idx, flat_pw)
    energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(flat_pw))
    return energy_sum / (energy_cnt + 1e-8)


@torch.no_grad()
def compute_axis_spectral_profiles(
    image_states: torch.Tensor, height: int, width: int, n_bins_h: int, n_bins_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis 1D power-spectrum profiles of the image-token hidden states (B, H*W, C)."""
    batch_size = image_states.shape[0]
    spatial = image_states[:, : height * width].reshape(batch_size, height, width, -1)
    sm = spatial.float().mean(dim=(0, -1))
    sm = sm - sm.mean()

    def _axis_profile(sm, axis, n_bins, length):
        fft = torch.fft.fft(sm, dim=axis)
        power = fft.abs().pow(2).mean(dim=1 - axis)
        half = length // 2 + 1
        power = power[:half]
        freq_norm = torch.linspace(0.0, 1.0, half, device=power.device)
        bin_idx = (freq_norm * n_bins).long().clamp(0, n_bins - 1)
        energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_sum.scatter_add_(0, bin_idx, power.float())
        energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(power, dtype=torch.float32))
        return energy_sum / (energy_cnt + 1e-8)

    return (
        _axis_profile(sm, axis=0, n_bins=n_bins_h, length=height),
        _axis_profile(sm, axis=1, n_bins=n_bins_w, length=width),
    )


@torch.no_grad()
def compute_dynamic_spread(energy_profile: torch.Tensor, spread_min=0.0, spread_max=1.0, alpha=1.5) -> float:
    """Spectral-flatness-driven spread in [spread_min, spread_max] (SEGA)."""
    eps = 1e-8
    energy = energy_profile.clamp(min=eps)
    geo_mean = torch.exp(torch.log(energy).mean())
    arith_mean = energy.mean()
    flatness = (geo_mean / (arith_mean + eps)).clamp(0.0, 1.0)
    concentration = 1.0 - flatness.item()
    return spread_min + (spread_max - spread_min) * (1.0 - (1.0 - concentration) ** alpha)


class Ideogram4MRoPE(nn.Module):
    """Multi-axis (t, h, w) interleaved rotary position embedding.

    Optionally applies RoPE length-extrapolation (NTK / YaRN / SEGA) when
    `rope_scaling` is set (see `Ideogram4Transformer2DModel.set_rope_scaling`)
    and the current image grid exceeds the trained grid extent.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple[int, ...],
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim
        self.base = float(base)
        self.rope_scaling: dict | None = None

        # Which axis owns each entry of the shared frequency ladder under the
        # interleaved layout (0 = t, 1 = h, 2 = w). Mirrors the scatter in forward.
        axis_of_freq = torch.zeros(head_dim // 2, dtype=torch.long)
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3)
            axis_of_freq[idx] = axis
        self.register_buffer("axis_of_freq", axis_of_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        grid_hw: tuple[int, int] | None = None,
        energy_profiles: tuple[torch.Tensor, torch.Tensor] | None = None,
        mscale_spread: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, L, 3) of int (axes are t, h, w).
        if position_ids.ndim != 3 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must have shape (B, L, 3), got {tuple(position_ids.shape)}.")

        scaling = self.rope_scaling
        active = (
            scaling is not None
            and grid_hw is not None
            and max(grid_hw) > scaling["train_grid"]
        )
        if not active:
            return self._forward_base(position_ids)
        return self._forward_scaled(position_ids, grid_hw, energy_profiles, mscale_spread)

    def _forward_base(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = position_ids.shape

        # Ideogram4's image position ids start at IMAGE_POSITION_OFFSET (65536). If an ambient autocast downcasts the
        # matmul to bfloat16, the image positions will collapse to only a few distinct values because bfloat16 cannot
        # represent consecutive integers at this value (after pos 65536 each 512-integer block will collapse to the
        # same value), which causes the image to become essentially flat. Therefore, we need to disable autocast here.
        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)  # (3, B, L, inv_freq_size)

        # Interleaved mrope: pull H freqs into idx 1 mod 3, W freqs into idx 2 mod 3.
        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos().float(), emb.sin().float()

    def _forward_scaled(
        self,
        position_ids: torch.Tensor,
        grid_hw: tuple[int, int],
        energy_profiles,
        mscale_spread,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        inv_axis, mscale = self._build_scaled_ladders(grid_hw, energy_profiles, mscale_spread, device)

        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)  # (3, B, L)
        inv_axis = inv_axis.to(dtype=torch.float32)

        # Same bfloat16-collapse hazard as in `_forward_base`: keep everything float32.
        with torch.autocast(device_type=device.type, enabled=False):
            # Anchor the constant image-position offset (65536) at the ORIGINAL
            # frequencies and apply the scaled per-axis frequencies only to the
            # within-grid displacement. Text tokens (pos < offset) keep delta = 0,
            # so text RoPE and the trained text<->image offset phases are preserved;
            # image<->image relative phases use the extrapolation-scaled combs.
            anchor = pos.clamp(max=float(IMAGE_POSITION_OFFSET))
            delta = pos - anchor
            inv_base = self.inv_freq.to(device=device, dtype=torch.float32)
            freqs = anchor.unsqueeze(-1) * inv_base.view(1, 1, 1, -1) + delta.unsqueeze(-1) * inv_axis.view(
                3, 1, 1, -1
            )

        # Interleaved mrope scatter, identical to the base path.
        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        ms = torch.cat((mscale, mscale)).view(1, 1, -1)
        return emb.cos().float() * ms, emb.sin().float() * ms

    def _build_scaled_ladders(
        self,
        grid_hw: tuple[int, int],
        energy_profiles,
        mscale_spread,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-axis frequency ladders (3, head_dim//2) and per-frequency mscale (head_dim//2,)."""
        cfg = self.rope_scaling
        method = cfg["method"]
        train_grid = float(cfg["train_grid"])
        grid_h, grid_w = grid_hw
        s_h = max(1.0, grid_h / train_grid)
        s_w = max(1.0, grid_w / train_grid)

        num_freqs = self.head_dim // 2
        inv = self.inv_freq.to(device=device, dtype=torch.float32)
        mscale = torch.ones(num_freqs, dtype=torch.float32, device=device)

        profile_h = profile_w = None
        if energy_profiles is not None:
            profile_h, profile_w = energy_profiles

        # Rescale only the h/w frequency combs, per axis. The t axis (which also carries
        # the whole text<->image separation) is left untouched.
        inv_axis = inv.unsqueeze(0).repeat(3, 1)
        for axis, s_a, profile_a in ((1, s_h, profile_h), (2, s_w, profile_w)):
            if s_a <= 1.0:
                continue
            comb = torch.nonzero(self.axis_of_freq.to(device) == axis, as_tuple=False).flatten()
            f = inv[comb]  # descending frequencies along the comb
            num_comb = comb.numel()
            k = torch.arange(num_comb, dtype=torch.float32, device=device)
            denom = max(num_comb - 1, 1)

            # NTK-by-parts on the comb: geometric progression from x1 (highest comb
            # frequency, preserves local texture) to 1/s (lowest, full interpolation).
            # Equivalent to base rescaling of a ladder whose entries are just the comb.
            f_ntk = f * s_a ** (-(k / denom))

            if method == "ntk":
                f_new = f_ntk
            elif method == "yarn":
                # Rotation counts within the trained relative extent. The comb is a
                # subsampled ladder, so ramps are built directly from rotations instead
                # of the full-ladder index-inversion formula.
                rot = train_grid * f / (2.0 * math.pi)
                f_pi = f / s_a
                m1 = ((rot - cfg["beta_1"]) / (cfg["beta_0"] - cfg["beta_1"])).clamp(0.0, 1.0)
                f_new = f_pi * (1.0 - m1) + f_ntk * m1
                m2 = ((rot - cfg["gamma_1"]) / (cfg["gamma_0"] - cfg["gamma_1"])).clamp(0.0, 1.0)
                f_new = f_new * (1.0 - m2) + f * m2
                mscale[comb] = 0.1 * math.log(s_a) + 1.0
            elif method == "sega":
                # SEGA's aggressive NTK factor (2x exponent, log-damped), expressed on
                # the comb treated as its own ladder of effective dim 2*num_comb.
                factor = s_a ** (2.0 * num_comb / denom) / (1.0 + 0.1 * math.log(s_a))
                f_new = f * factor ** (-(k / num_comb))
                base_ms = s_a ** cfg["base_mscale_coefficient"]
                if profile_a is not None and mscale_spread and base_ms > 1.0 + 1e-8:
                    mscale[comb] = _sega_allocation(
                        profile_a,
                        f_new,
                        base_ms,
                        float(mscale_spread),
                        cfg["mscale_alpha"],
                        cfg["mscale_beta"],
                        cfg["mscale_min"],
                    )
                else:
                    mscale[comb] = base_ms
            else:
                raise ValueError(f"Unknown rope scaling method {method!r}.")

            inv_axis[axis, comb] = f_new

        return inv_axis, mscale


class Ideogram4AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "Ideogram4Attention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # MRoPE applied in (B, L, num_heads, head_dim) layout; cos/sin broadcast over the head axis.
        cos, sin = image_rotary_emb
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


class Ideogram4Attention(nn.Module, AttentionModuleMixin):
    """Self-attention with split Q/K/V, q/k RMSNorm, MRoPE and a block-diagonal segment mask."""

    _default_processor_cls = Ideogram4AttnProcessor
    _available_processors = [Ideogram4AttnProcessor]

    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_bias = False

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.norm_k = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0)])

        self.set_processor(self._default_processor_cls())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k in kwargs if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Ideogram4MLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@maybe_allow_in_graph
class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adaln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)
        self.attention_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)

        self.adaln_modulation = nn.Linear(adaln_dim, 4 * hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        adaln_input: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(hidden_states) * scale_msa,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + gate_msa * self.attention_norm2(attn_out)
        hidden_states = hidden_states + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(hidden_states) * scale_mlp)
        )
        return hidden_states


def _sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    """Sinusoidal scalar embedding followed by a small MLP."""

    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        if self.range_max <= self.range_min:
            raise ValueError("input_range[1] must be greater than input_range[0]")
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(in_dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adaln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adaln_dim, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(conditioning))
        return self.linear(self.norm_final(hidden_states) * scale)


class Ideogram4Transformer2DModel(ModelMixin, ConfigMixin, AttentionMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    The flow-matching transformer backbone used by the Ideogram 4 pipeline.

    The transformer operates on a single packed sequence containing both text-conditioning tokens (produced by a
    multimodal text encoder) and the patchified image latents. Per-token indicators distinguish the two roles, and a
    block-diagonal attention mask derived from `segment_ids` restricts each sample to attend only to itself within a
    packed batch.

    Args:
        in_channels (`int`, defaults to 128):
            Latent channel count after patchification (`ae_channels * patch_size ** 2`).
        num_layers (`int`, defaults to 34):
            Number of transformer blocks.
        attention_head_dim (`int`, defaults to 256):
            Dimension of each attention head; the total hidden size is `attention_head_dim * num_attention_heads`.
        num_attention_heads (`int`, defaults to 18):
            Number of attention heads.
        intermediate_size (`int`, defaults to 12288):
            Feed-forward hidden size used by the SwiGLU MLP inside each block.
        adaln_dim (`int`, defaults to 512):
            Dimensionality of the conditioning vector consumed by the AdaLN modulations.
        llm_features_dim (`int`, defaults to 53248):
            Dimensionality of the per-token text features fed into the model (typically a concatenation of hidden
            states from several layers of the text encoder).
        rope_theta (`int`, defaults to 5_000_000):
            Base used by the multi-axis rotary position embedding.
        mrope_section (`tuple[int, int, int]`, defaults to `(24, 20, 20)`):
            Number of frequencies allocated to each of the (t, h, w) axes of MRoPE.
        norm_eps (`float`, defaults to 1e-5):
            Epsilon used by the RMSNorm modules inside the transformer blocks.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Ideogram4TransformerBlock"]
    _repeated_blocks = ["Ideogram4TransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedding", "adaln_proj", "embed_image_indicator"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        num_layers: int = 34,
        attention_head_dim: int = 256,
        num_attention_heads: int = 18,
        intermediate_size: int = 12288,
        adaln_dim: int = 512,
        llm_features_dim: int = 53248,
        rope_theta: int = 5_000_000,
        mrope_section: tuple[int, int, int] = (24, 20, 20),
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        hidden_size = attention_head_dim * num_attention_heads
        head_dim = attention_head_dim

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.gradient_checkpointing = False

        self.input_proj = nn.Linear(in_channels, hidden_size, bias=True)
        self.llm_cond_norm = RMSNorm(llm_features_dim, eps=1e-6, elementwise_affine=True)
        self.llm_cond_proj = nn.Linear(llm_features_dim, hidden_size, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(hidden_size, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(hidden_size, adaln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, hidden_size)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=rope_theta,
            mrope_section=mrope_section,
        )

        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    norm_eps=norm_eps,
                    adaln_dim=adaln_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(
            hidden_size=hidden_size,
            out_channels=in_channels,
            adaln_dim=adaln_dim,
        )

    def set_rope_scaling(self, method: str | None = None, **kwargs) -> None:
        """Configure RoPE length-extrapolation on this model's rotary embedding.

        Args:
            method: `"ntk"`, `"yarn"`, `"sega"`, or `None`/`"none"` to disable.
            **kwargs: Overrides for `ROPE_SCALING_DEFAULTS` (e.g. `train_grid=256`,
                YaRN betas/gammas, SEGA mscale/spread params).

        Scaling only engages when the image grid of a forward pass exceeds
        `train_grid` patches on either axis; below that the RoPE is bit-identical
        to the unmodified model.
        """
        if method is None or method == "none":
            self.rotary_emb.rope_scaling = None
            return
        if method not in ("ntk", "yarn", "sega"):
            raise ValueError(f"Unknown rope scaling method {method!r}; expected 'ntk', 'yarn' or 'sega'.")
        unknown = set(kwargs) - set(ROPE_SCALING_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown rope scaling options: {sorted(unknown)}")
        cfg = dict(ROPE_SCALING_DEFAULTS)
        cfg.update(kwargs)
        cfg["method"] = method
        self.rotary_emb.rope_scaling = cfg

    def _rope_scaling_kwargs(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, indicator: torch.Tensor
    ) -> dict:
        """Derive the grid extent (and, for SEGA, spectral profiles) for the scaled RoPE path."""
        cfg = self.rotary_emb.rope_scaling
        if cfg is None:
            return {}

        image_mask = indicator[0] == OUTPUT_IMAGE_INDICATOR
        if not torch.any(image_mask):
            return {}
        image_pos = position_ids[0][image_mask]
        grid_h = int(image_pos[:, 1].max() - image_pos[:, 1].min()) + 1
        grid_w = int(image_pos[:, 2].max() - image_pos[:, 2].min()) + 1
        rope_kwargs: dict = {"grid_hw": (grid_h, grid_w)}

        if cfg["method"] == "sega" and max(grid_h, grid_w) > cfg["train_grid"]:
            image_states = hidden_states[:, image_mask]
            n_bins_h = max(grid_h // 2, 8)
            n_bins_w = max(grid_w // 2, 8)
            profile_h, profile_w = compute_axis_spectral_profiles(image_states, grid_h, grid_w, n_bins_h, n_bins_w)
            rope_kwargs["energy_profiles"] = (profile_h, profile_w)
            # Isotropic radial profile drives only the global spread scalar.
            profile_iso = compute_spectral_energy_profile(
                image_states, grid_h, grid_w, n_bins=max(max(grid_h, grid_w) // 2, 8)
            )
            rope_kwargs["mscale_spread"] = compute_dynamic_spread(
                profile_iso,
                spread_min=cfg["spread_min"],
                spread_max=cfg["spread_max"],
                alpha=cfg["spread_alpha"],
            )
        return rope_kwargs

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        attention_kwargs: dict | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        r"""
        Predict the flow-matching velocity for the image-token positions of the packed sequence.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, in_channels)`):
                Packed sequence of patchified noisy image tokens. Non-image positions are masked out internally.
            timestep (`torch.Tensor` of shape `(batch_size,)` or `(batch_size, sequence_length)`):
                Flow-matching time in `[0, 1]` (0 is pure noise, 1 is clean data).
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, llm_features_dim)`):
                Per-token text conditioning features. Non-text positions are masked out internally.
            position_ids (`torch.Tensor` of shape `(batch_size, sequence_length, 3)`):
                `(t, h, w)` coordinates consumed by the multi-axis RoPE.
            segment_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Per-token sample id within a packed batch. Positions sharing a `segment_id` attend to each other.
            indicator (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Per-token role: `LLM_TOKEN_INDICATOR` (text) or `OUTPUT_IMAGE_INDICATOR` (image).
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary passed along to the attention processor. A `"scale"` entry scales the LoRA weights
                (when the PEFT backend is active).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or a `tuple` whose first element is a tensor of shape
            `(batch_size, sequence_length, in_channels)` in the model's compute dtype. Only positions tagged with
            `OUTPUT_IMAGE_INDICATOR` carry meaningful velocity predictions.
        """
        batch_size, seq_len, in_channels = hidden_states.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected last dim {self.in_channels}, got {in_channels}.")

        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(hidden_states.dtype).unsqueeze(-1)

        hidden_states = hidden_states * output_image_mask
        hidden_states = self.input_proj(hidden_states) * output_image_mask

        # Keep shape (B, 1, ...) when t is per-sample so downstream adaln projections do not pay for L identical copies.
        t_cond = self.t_embedding(timestep)
        if timestep.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        # Text conditioning is nonzero only on LLM-token positions, which occupy a short prefix of the packed sequence;
        # the trailing image tokens carry zero text features. Gather just those rows so the very wide
        # (`llm_features_dim`) tensor is never normalized/projected at full sequence length -- at high resolution the
        # full-length RMSNorm alone would allocate tens of GBs. `encoder_hidden_states` aligns with the leading text
        # region (it may be shorter than the packed sequence) and is `None` for the unconditional pass, which carries
        # no text at all. This is mathematically identical to masking the full-length sequence.
        if encoder_hidden_states is not None:
            enc_len = encoder_hidden_states.shape[1]
            llm_token_mask = indicator[:, :enc_len] == LLM_TOKEN_INDICATOR
            if llm_token_mask.any():
                llm_rows = encoder_hidden_states[llm_token_mask]
                llm_rows = self.llm_cond_proj(self.llm_cond_norm(llm_rows))
                text_cond = torch.zeros(
                    batch_size, enc_len, self.hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
                )
                text_cond[llm_token_mask] = llm_rows.to(hidden_states.dtype)
                hidden_states[:, :enc_len] = hidden_states[:, :enc_len] + text_cond

        image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
        hidden_states = hidden_states + image_indicator_embedding

        rope_kwargs = self._rope_scaling_kwargs(hidden_states, position_ids, indicator)
        cos, sin = self.rotary_emb(position_ids, **rope_kwargs)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        image_rotary_emb = (cos, sin)

        # Block-diagonal mask from segment ids: tokens only attend within their segment. Shared by every block.
        # When every token shares one segment (single-prompt, padding trimmed by the pipeline) the mask is all-True,
        # so skip it: this enables the unmasked fast attention path and, critically, avoids materializing a dense
        # (L, L) mask whose element count exceeds 2^32 for L > 65536 (4096x4096 and above) — attention backends
        # index such masks with 32-bit arithmetic and silently corrupt the trailing query rows (visible as a
        # garbled horizontal band near the bottom of the image).
        if torch.all(segment_ids == segment_ids[:, :1]):
            attention_mask = None
        else:
            if seq_len * seq_len > 2**31:
                logger.warning(
                    f"Building a dense {seq_len}x{seq_len} segment attention mask (> 2^31 elements); attention "
                    "backends may overflow 32-bit indexing on masks this large. Prefer uniform segments "
                    "(trim padding) at high resolution."
                )
            attention_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        for block in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, attention_mask, image_rotary_emb, adaln_input
                )
            else:
                hidden_states = block(hidden_states, attention_mask, image_rotary_emb, adaln_input)

        output = self.final_layer(hidden_states, conditioning=adaln_input)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
