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

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import PIL.Image
import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedModel
from transformers.masking_utils import create_causal_mask

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_utils import ModelMixin

try:
    from diffusers.loaders import Ideogram4LoraLoaderMixin
except ImportError:
    class Ideogram4LoraLoaderMixin:  # noqa: N801
        """Stub for older diffusers builds without Ideogram4 LoRA support."""

from diffusers.models.autoencoders import AutoencoderKLFlux2
from transformer_ideogram4 import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Transformer2DModel,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring

try:
    from diffusers.utils import is_outlines_available
except ImportError:
    def is_outlines_available() -> bool:
        try:
            import outlines  # noqa: F401
            return True
        except ImportError:
            return False

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

PROMPT_UPSAMPLE_TEMPERATURE = 1.0


@dataclass
class Ideogram4PipelineOutput(BaseOutput):
    """Output class for the Ideogram 4 pipeline."""

    images: list[PIL.Image.Image] | np.ndarray


class Ideogram4PromptEnhancerHead(ModelMixin, ConfigMixin):
    """LM head that makes the head-less Qwen3-VL text encoder generative for prompt upsampling."""

    @register_to_config
    def __init__(self, hidden_size: int = 4096, vocab_size: int = 151936):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


def _build_prompt_enhancer(text_encoder, prompt_enhancer_head, device):
    from accelerate import init_empty_weights
    from transformers import Qwen3VLForConditionalGeneration

    with init_empty_weights():
        enhancer = Qwen3VLForConditionalGeneration(text_encoder.config)
    enhancer.model = text_encoder
    # Give the enhancer its OWN copy of the lm_head weights, pinned to `device`, rather than sharing
    # `prompt_enhancer_head.lm_head` directly. Under `enable_model_cpu_offload()`, `prompt_enhancer_head` sits
    # first in `model_cpu_offload_seq`; the hook attached to the NEXT component (`text_encoder`) holds a direct
    # reference to `prompt_enhancer_head` and force-offloads it back to CPU as a side effect the moment
    # `text_encoder`'s forward runs (accelerate's `CpuOffload.pre_forward`) -- this happens mid-`generate()`, well
    # before `lm_head` is reached, so sharing the module (even after detaching its own hook) still loses the race.
    # A decoupled copy sidesteps the hook chain entirely; the head is small (~1.2GB in bf16) so this is cheap.
    src = prompt_enhancer_head.lm_head
    lm_head = nn.Linear(src.in_features, src.out_features, bias=src.bias is not None)
    lm_head = lm_head.to(device=device, dtype=src.weight.dtype)
    lm_head.weight.data.copy_(src.weight.data)
    if src.bias is not None:
        lm_head.bias.data.copy_(src.bias.data)
    enhancer.lm_head = lm_head
    return enhancer.eval()


# docstyle-ignore
_CAPTION_SYSTEM_MESSAGE = """You convert a short user idea into a structured JSON caption for an image renderer. Output ONE minified single-line JSON object and NOTHING else (no markdown, no commentary).

SCHEMA — keys in this exact order:
{"high_level_description":"...","compositional_deconstruction":{"background":"...","elements":[ ... ]}}
- object element: {"type":"obj","desc":"..."}
- text element:   {"type":"text","text":"VERBATIM CHARS","desc":"..."}

STEP 1 — PICK THE MEDIUM. It decides what `background` and `elements` mean. Honor any medium or style the user implies; default to photograph only when nothing else fits. Render ANY subject faithfully — real, fantastical, sci-fi, surreal, abstract — in the chosen medium.

A) DESIGNED ARTIFACT — poster, logo, album/book cover, flyer, banner, sticker, packaging, app icon, infographic, menu, card, wordmark. THE FRAME IS THE ARTIFACT — never a photo of it hanging in a room.
   - high_level_description: name it as graphic design (e.g. "a minimalist jazz poster, flat graphic design...").
   - background: the design's OWN backdrop only — a flat color, gradient, or simple texture filling the frame. No room, wall, floor, easel, depth, or camera/photo language.
   - elements: the design's parts as a flat 2D layout — a `text` element for every headline/label (verbatim), `obj` elements for the central graphic/illustration/shapes/badges. Place by region (top / center / bottom).

B) SCENE — a photograph, illustration, painting, 3D render, anime frame, etc. of a real or imagined place or subject.
   - high_level_description: one sentence naming the subject and the medium/style.
   - background: the scene SHELL — surroundings, ground/sky/walls, atmosphere, ambient light; concrete and specific. The ground/floor/water surface lives here, never as an element.
   - elements: the main subject FIRST as an `obj`, then supporting `obj` elements (props, secondary subjects) that plausibly belong. Add `text` elements only where the scene would really carry text (signs, labels, brands).

C) ABSTRACT / CONCEPTUAL — "nostalgia", "chaos and order", "sound waves", pure pattern. Concretize the idea into a deliberate visual composition.
   - background: the dominant color field, gradient, or texture of the composition.
   - elements: the shapes, forms, motifs, or symbolic objects that carry the concept, as `obj` elements. Add `text` only if the idea calls for words.

UNIVERSAL RULES (every medium):
1. The user's core subject/concept MUST appear among the elements (as an `obj`, normally first). Naming it only in high_level_description or background is NOT enough.
2. Commit to ONE concrete value each (one color, one style, one count). No hedging: ban "various", "such as", "e.g.", "or similar", "maybe", "X or Y" for one property.
3. NEVER use a transparent, empty, or plain white background UNLESS the user explicitly says "transparent", "isolated", "sticker", or "cutout".
4. A coherent subject (one animal, person, vehicle, object) is exactly ONE element; its parts go inside its `desc`. Use separate elements for genuinely separate subjects.
5. Each `desc` is 25-55 words, identity-first, standalone. Do not mention shadows, depth of field, bokeh, lens, focus, or grain.
6. high_level_description: one sentence, at most 40 words, starts with the subject, names the medium. Preserve non-ASCII characters as-is.
7. Output STRICTLY VALID JSON: double quotes around every key and string, NO trailing commas, each element object closes with "}" right after its last value.
8. Catch the "warm" impulse. Only when you are about to describe light as "warm", "golden", "amber", or "honey", stop and check: is there a specific physical source in the scene casting that colour (candle, sunset, lamp, neon, fire)? If YES, name the source and the colour it casts instead of the mood word. If NO, you are just reaching for warmth as ambience — drop it and leave the light neutral ("soft" or "even"). Don't recolour or relight anything else; this only intercepts the warm reach, every other scene and mood the user wants is untouched.
9. Describe physical reality, not impressions. Avoid mood-words — "luminous", "radiant", "vibrant", "lush", "dynamic", "gorgeous", "stunning", "breathtaking", "mesmerizing", and metaphorical "glowing" — they produce a generic AI look (the same trap as "warm"). Use observable properties: "the cheekbone catches a small highlight", not "luminous complexion".
10. Every named thing must appear as its own element. Each subject, object, sign, and quoted phrase the user names gets its own element — quoted text (single or double quotes) becomes its own verbatim `text` element. Count the named units in the prompt; the element list must hold at least that many. Don't drop or merge them.
11. Don't add what wasn't asked for. No glitch art, wireframe overlay, body fragmentation, double-exposure, "dissolving", or extra stylization unless the prompt requests it. Asked for a cinematic photo of a journalist → render that, not a glitch-art composite.
12. Name attributes concretely, anchored to landmarks. People: skin tone, hair (colour + style), each visible garment with colour, expression, pose, one distinguishing feature. Objects: shape, material, colour, a distinctive part. Place things against named references — "resting on the lower-right corner of the table", not "on the surface".
13. Name real references by name. If the user names a brand, product, character, place, or person (Nike Dunk Low, Spider-Man, the Eiffel Tower), keep that exact name in the `desc`; don't swap it for a generic look-alike unless they ask for an anonymous one.
14. "Professional photo/headshot" of a person means professional CONTEXT — neutral attire, soft even daylight, neutral backdrop, friendly expression — not dramatic studio gear; no heavy rim-light or creamy bokeh unless asked.

EXAMPLES

User idea: a cup of coffee on a table
Output: {"high_level_description":"A white ceramic cup of black coffee on a worn wooden cafe table, a casual overcast-daylight phone photograph with an off-center composition.","compositional_deconstruction":{"background":"Scratched oak cafe table filling the lower frame, a pale grey mortar-lined brick wall a few feet behind slightly out of focus, a tall window on the left spilling soft overcast daylight across the table, neutral white balance, muted brown and green tones.","elements":[{"type":"obj","desc":"White ceramic cup of black coffee with a thin curved handle turned to the right and a faint crema ring at the rim, resting on a matching round saucer near the center of the table, a thin wisp of steam at the surface."},{"type":"obj","desc":"Brushed-steel teaspoon lying on the saucer to the right of the cup, handle angled toward the lower-right corner, a single small water droplet on the bowl of the spoon."}]}}

User idea: a minimalist poster for a jazz festival
Output: {"high_level_description":"A minimalist jazz festival poster, flat graphic design with bold typography and a single abstract saxophone motif on a deep teal background.","compositional_deconstruction":{"background":"Solid deep teal background filling the entire frame with a subtle fine paper-grain texture and a thin mustard-yellow keyline border just inside the edges, no scene and no depth.","elements":[{"type":"obj","desc":"A large flat geometric saxophone in mustard yellow and cream, centered in the upper two-thirds, built from simple bold shapes with no shading, angled diagonally from lower-left to upper-right."},{"type":"text","text":"JAZZ\\nFESTIVAL","desc":"Large bold condensed sans-serif headline in cream, stacked on two lines across the center of the poster, slightly overlapping the saxophone motif."},{"type":"text","text":"NOV 15 · CITY HALL","desc":"Small uppercase mustard-yellow caption centered near the bottom edge with wide letter spacing."}]}}"""

# docstyle-ignore
_CAPTION_USER_TEMPLATE = """TARGET IMAGE ASPECT RATIO: {aspect_ratio} (width:height).
User idea: {original_prompt}"""


def _build_caption_logits_processor(model, tokenizer):
    from typing import List, Literal, Union

    import outlines
    from pydantic import BaseModel, Field

    class ObjElement(BaseModel):
        type: Literal["obj"]
        desc: str

    class TextElement(BaseModel):
        type: Literal["text"]
        text: str
        desc: str

    class Composition(BaseModel):
        background: str
        elements: List[Union[ObjElement, TextElement]] = Field(min_length=1)

    class Caption(BaseModel):
        high_level_description: str
        compositional_deconstruction: Composition

    outlines_model = outlines.from_transformers(model, tokenizer)
    return outlines.Generator(outlines_model, Caption).logits_processor


def _generate_captions(
    prompt_enhancer,
    tokenizer,
    logits_processor,
    prompt: str | list[str],
    height: int,
    width: int,
    temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
    max_new_tokens: int = 1024,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | None = None,
) -> list[str]:
    device = device or prompt_enhancer.device
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    divisor = math.gcd(width, height) or 1
    aspect_ratio = f"{width // divisor}:{height // divisor}"

    sampling_seed = None
    if generator is not None:
        gen = generator[0] if isinstance(generator, list) else generator
        sampling_seed = int(torch.randint(0, 2**63 - 1, (1,), generator=gen, device=gen.device).item())
    fork_devices = [device] if getattr(device, "type", None) == "cuda" else []

    captions = []
    for i, text_prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": _CAPTION_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": _CAPTION_USER_TEMPLATE.format(aspect_ratio=aspect_ratio, original_prompt=text_prompt),
            },
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "use_cache": True,
        }
        if logits_processor is not None:
            logits_processor.reset()
            generate_kwargs["logits_processor"] = [logits_processor]
        with torch.random.fork_rng(devices=fork_devices, enabled=sampling_seed is not None):
            if sampling_seed is not None:
                torch.manual_seed(sampling_seed + i)
            generated = prompt_enhancer.generate(**inputs, **generate_kwargs)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        captions.append(tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip())
    return captions


# Hidden states of these Qwen3-VL decoder layers are concatenated to form the per-token
# text conditioning consumed by the Ideogram4 transformer.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import Ideogram4Pipeline

        >>> pipe = Ideogram4Pipeline.from_pretrained("ideogram-ai/ideogram-v4", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A photo of a cat holding a sign that says hello world"
        >>> # The defaults are the recommended settings for best quality.
        >>> image = pipe(prompt, height=2048, width=2048, generator=torch.Generator("cuda").manual_seed(0)).images[0]
        >>> image.save("ideogram4.png")
        ```
"""


def _logit_normal_sigmas(
    num_inference_steps: int,
    mu: float,
    std: float = 1.0,
    logsnr_min: float = -15.0,
    logsnr_max: float = 18.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    r"""
    Build a length-`num_inference_steps` sigma schedule using the Ideogram4 logit-normal flow-matching schedule.

    Sigmas are returned in `[0, 1]` in decreasing order (sigma close to 1 corresponds to pure noise, sigma close to 0
    to clean data), matching diffusers conventions.

    The Ideogram4 schedule applies `sigma(s) = 1 - logit_normal_cdf_inverse(1 - s)` to `s = linspace(0, 1, N + 1)` and
    keeps the first `N` entries; a terminal zero is appended downstream by the scheduler.
    """
    intervals = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=torch.float64)
    # Apply the inverse CDF of a normal then push through the logistic to obtain a logit-normal CDF inverse.
    z = torch.special.ndtri(intervals)
    y = mu + std * z
    t = 1.0 - torch.special.expit(y)
    t_min = 1.0 / (1.0 + math.exp(0.5 * logsnr_max))
    t_max = 1.0 / (1.0 + math.exp(0.5 * logsnr_min))
    t = t.clamp(t_min, t_max)
    # Convert from model time (0 = noise, 1 = data) to diffusers sigma (1 = noise, 0 = data) and reverse.
    sigmas = (1.0 - t).flip(0)
    # Drop the trailing 0; FlowMatchEulerDiscreteScheduler.set_timesteps appends one back internally.
    sigmas = sigmas[:-1].to(dtype=torch.float32, device=device)
    return sigmas


def _resolution_aware_mu(
    height: int,
    width: int,
    base_mu: float,
    base_resolution: tuple[int, int] = (512, 512),
) -> float:
    """Shift the schedule mean as a function of image resolution."""
    num_pixels = height * width
    base_pixels = base_resolution[0] * base_resolution[1]
    return base_mu + 0.5 * math.log(num_pixels / base_pixels)


def _expand_tensor_to_effective_batch(
    tensor: torch.Tensor,
    batch_size: int,
    num_per_prompt: int,
    tensor_name: str | None = None,
) -> torch.Tensor:
    """Replicate `tensor` along dim 0 from `batch_size` (or 1) to `batch_size * num_per_prompt`."""
    target_batch_size = batch_size * num_per_prompt

    if tensor.shape[0] == target_batch_size:
        return tensor

    if tensor.shape[0] == 1:
        repeat_by = target_batch_size
    elif tensor.shape[0] == batch_size:
        repeat_by = num_per_prompt
    else:
        tensor_name = f"`{tensor_name}`" if tensor_name is not None else "Tensor"
        raise ValueError(
            f"{tensor_name} batch size must be 1, `batch_size` ({batch_size}), or "
            f"`batch_size * num_*_per_prompt` ({target_batch_size}), but got {tensor.shape[0]}."
        )

    return torch.repeat_interleave(tensor, repeats=repeat_by, dim=0, output_size=tensor.shape[0] * repeat_by)


class Ideogram4Pipeline(DiffusionPipeline, Ideogram4LoraLoaderMixin):
    r"""
    Text-to-image pipeline for Ideogram4.

    Ideogram4 is a flow-matching model trained with asymmetric classifier-free guidance: a `transformer` consumes
    text-conditioned features alongside the image latents, while a separate `unconditional_transformer` denoises with
    zeroed text features. The two velocity predictions are linearly blended each step.

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler. The pipeline overrides the default sigma schedule with a resolution-aware
            logit-normal schedule.
        vae ([`AutoencoderKLFlux2`]):
            Variational auto-encoder used to decode latents back into images.
        text_encoder ([`PreTrainedModel`]):
            Multimodal text encoder. The pipeline consumes hidden states from a fixed set of intermediate decoder
            layers (see `QWEN3_VL_ACTIVATION_LAYERS`).
        tokenizer ([`AutoTokenizer`]):
            Tokenizer paired with `text_encoder`.
        transformer ([`Ideogram4Transformer2DModel`]):
            Conditional flow-matching transformer.
        unconditional_transformer ([`Ideogram4Transformer2DModel`]):
            Unconditional (asymmetric-CFG) flow-matching transformer.
    """

    model_cpu_offload_seq = "prompt_enhancer_head->text_encoder->transformer->unconditional_transformer->vae"
    _optional_components = ["prompt_enhancer_head", "unconditional_transformer"]
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: Ideogram4Transformer2DModel,
        unconditional_transformer: Ideogram4Transformer2DModel | None = None,
        prompt_enhancer_head: Ideogram4PromptEnhancerHead | None = None,
    ) -> None:
        super().__init__()

        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            unconditional_transformer=unconditional_transformer,
            prompt_enhancer_head=prompt_enhancer_head,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) is not None else 8
        )
        # Ideogram4 patchifies the VAE output by a factor of 2 before feeding into the transformer.
        self.patch_size = 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * self.patch_size)

        # Built lazily on first upsample: the head-less encoder body + `prompt_enhancer_head`, combined.
        self._prompt_enhancer = None
        # Outlines logits processor for schema-constrained captions; built lazily on first upsample.
        self._caption_logits_processor = None

    def upsample_prompt(
        self,
        prompt: str | list[str],
        height: int = 2048,
        width: int = 2048,
        temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
        max_new_tokens: int = 1024,
        generator: torch.Generator | list[torch.Generator] | None = None,
        device: torch.device | None = None,
    ) -> list[str]:
        """Rewrite each prompt into Ideogram4's native structured JSON caption.

        Requires the optional `prompt_enhancer_head` component, which is grafted onto the shared `text_encoder` body to
        make it generative. Generation is schema-constrained when `outlines` is installed, otherwise it runs
        unconstrained. Pass `generator` (the same one accepted by `__call__`) to make sampling reproducible.
        """
        if self.prompt_enhancer_head is None:
            raise ValueError(
                "Prompt upsampling requires the `prompt_enhancer_head` component, which is not loaded. Load it and "
                "pass it in, e.g.:\n"
                "    from diffusers import Ideogram4PromptEnhancerHead\n"
                "    head = Ideogram4PromptEnhancerHead.from_pretrained('diffusers/qwen3-vl-8b-instruct-lm-head')\n"
                "    pipe = Ideogram4Pipeline.from_pretrained(model_id, prompt_enhancer_head=head)"
            )
        if self._prompt_enhancer is None:
            self._prompt_enhancer = _build_prompt_enhancer(
                self.text_encoder, self.prompt_enhancer_head, device or self._execution_device
            )
        if self._caption_logits_processor is None and is_outlines_available():
            self._caption_logits_processor = _build_caption_logits_processor(self._prompt_enhancer, self.tokenizer)
        if self._caption_logits_processor is None:
            logger.warning_once(
                "`outlines` is not installed; prompt upsampling runs unconstrained and may not return schema-valid "
                "JSON. Install with `pip install outlines` for structured captions."
            )

        return _generate_captions(
            self._prompt_enhancer,
            self.tokenizer,
            self._caption_logits_processor,
            prompt,
            height,
            width,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            generator=generator,
            device=device,
        )

    @staticmethod
    def _prepare_ids(
        text_lengths: list[int],
        grid_h: int,
        grid_w: int,
        max_text_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the packed `[left-pad][text][image]` layout from the per-prompt text lengths and the image grid.

        Returns `position_ids` (3-axis MRoPE), `segment_ids` (block-diagonal attention) and `indicator` (per-token
        text/image/pad role).
        """
        batch_size = len(text_lengths)
        num_image_tokens = grid_h * grid_w
        total_seq_len = max_text_tokens + num_image_tokens

        # Image position ids (t=0, h, w); offset keeps them disjoint from text positions.
        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for b, num_text in enumerate(text_lengths):
            offset = max_text_tokens - num_text

            text_pos = torch.arange(num_text)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset + num_text :] = image_pos

            indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

            segment_ids[b, offset : offset + num_text + num_image_tokens] = 1

        return position_ids.to(device), segment_ids.to(device), indicator.to(device)

    @staticmethod
    def _get_text_encoder_hidden_states(
        text_encoder,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_2d: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the text encoder's decoder layers, returning the hidden states tapped at each activation layer."""

        language_model = text_encoder.language_model

        inputs_embeds = language_model.embed_tokens(token_ids)

        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured: dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states

        return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

    def encode_prompt(
        self,
        prompt: str | list[str],
        grid_h: int,
        grid_w: int,
        max_sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the conditioning for the packed text+image sequence (one entry per prompt).

        Returns a flat tuple `(prompt_embeds, position_ids, segment_ids, indicator)`. The unconditional branch carries
        no text, so the pipeline builds its (zeroed) inputs directly rather than encoding a negative prompt.
        """
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        batch_size = len(prompts)

        # Tokenize each chat-formatted prompt and left-pad to `max_sequence_length`. Only the text region is fed to
        # the encoder: the packed image tokens come after the text and the encoder is causal, so they never affect it.
        token_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        text_lengths = []
        for b, text_prompt in enumerate(prompts):
            messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            toks = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            n = int(toks.shape[0])
            if n > max_sequence_length:
                raise ValueError(f"prompt has {n} tokens, exceeds max_sequence_length={max_sequence_length}")
            text_lengths.append(n)
            offset = max_sequence_length - n
            token_ids[b, offset:] = toks
            attention_mask[b, offset:] = 1
            text_position_ids[b, offset:] = torch.arange(n)

        # To support enable_model_cpu_offload, we need to move the text_encoder inputs to the text encoder's actual
        # device te_device. This is necessary because the `CpuOffload` model offload hook attaches to a component's
        # `forward` method, but we call text_encoder's submodules directly below, so the hook never fires to onload the
        # model to the execution device. Other offloading techniques (group, sequential) would work without te_device
        # because they hook submodules, not just the top-level component module. Note that in the
        # enable_model_cpu_offload case te_device will actually be the offload device (e.g. CPU).
        te_device = self.text_encoder.device
        token_ids = token_ids.to(te_device)
        attention_mask = attention_mask.to(te_device)
        text_position_ids = text_position_ids.to(te_device)

        # Concatenate the tapped activation-layer hidden states into per-token text features, zeroing padding.
        selected = self._get_text_encoder_hidden_states(
            self.text_encoder, token_ids, attention_mask, text_position_ids
        )
        text_features = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(batch_size, max_sequence_length, -1)
        text_features = (text_features * attention_mask.to(text_features.dtype).unsqueeze(-1)).to(torch.float32)
        text_features = text_features.to(device)

        position_ids, segment_ids, indicator = self._prepare_ids(
            text_lengths, grid_h, grid_w, max_sequence_length, device
        )

        # Text features cover only the leading text region of the packed sequence. The transformer scatters them onto
        # the LLM-token positions itself, so there is no need to materialize a full-length image-padded copy here
        # (which at high resolution would be tens of GBs of mostly zeros).
        return text_features, position_ids, segment_ids, indicator

    def prepare_latents(
        self,
        batch_size: int,
        num_image_tokens: int,
        latent_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (batch_size, num_image_tokens, latent_dim)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self) -> float | None:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def attention_kwargs(self) -> dict[str, Any] | None:
        return self._attention_kwargs

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    def check_inputs(
        self,
        prompt,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        guidance_schedule,
        guidance_mode="dual",
        callback_on_step_end_tensor_inputs=None,
    ):
        if prompt is None:
            raise ValueError("`prompt` must be provided.")
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if guidance_mode not in ("dual", "single"):
            raise ValueError(
                f"`guidance_mode` must be one of 'dual' or 'single', got {guidance_mode!r}."
            )

        if (
            height % (self.vae_scale_factor * self.patch_size) != 0
            or width % (self.vae_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` ({height}) and `width` ({width}) must both be divisible by {self.vae_scale_factor * self.patch_size} "
                f"(vae_scale_factor * patch_size)."
            )

        # Guidance is controlled by either a constant `guidance_scale` or a per-step `guidance_schedule`; exactly
        # one must be set (the `guidance_schedule` default makes the no-arg call use the recommended schedule).
        if guidance_scale is not None and guidance_schedule is not None:
            raise ValueError("Only one of `guidance_scale` and `guidance_schedule` may be set.")
        if guidance_scale is None and guidance_schedule is None:
            raise ValueError("One of `guidance_scale` and `guidance_schedule` must be set.")
        if guidance_schedule is not None and len(guidance_schedule) != num_inference_steps:
            raise ValueError(
                f"`guidance_schedule` must have length `num_inference_steps` ({num_inference_steps}), "
                f"got {len(guidance_schedule)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        height: int = 2048,
        width: int = 2048,
        num_inference_steps: int = 48,
        guidance_scale: float | None = None,
        guidance_schedule: list[float] | torch.Tensor | None = (7.0,) * 45 + (3.0,) * 3,
        guidance_mode: str = "dual",
        mu: float = 0.0,
        std: float = 1.5,
        prompt_upsampling: bool = False,
        prompt_upsampling_temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
        max_sequence_length: int = 2048,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[["Ideogram4Pipeline", int, int, dict[str, Any]], dict[str, Any]] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> Ideogram4PipelineOutput | tuple[Any]:
        r"""
        Run text-to-image generation.

        Args:
            prompt (`str` or `list[str]`):
                Prompt(s) to guide image generation.
            height (`int`, *optional*, defaults to 2048):
                Output image height in pixels; must be a multiple of `vae_scale_factor * patch_size`.
            width (`int`, *optional*, defaults to 2048):
                Output image width in pixels; must be a multiple of `vae_scale_factor * patch_size`.
            num_inference_steps (`int`, *optional*, defaults to 48):
                Number of flow-matching steps. The default is the recommended setting for best quality.
            guidance_scale (`float`, *optional*):
                Constant classifier-free guidance scale applied at every step. The conditional and unconditional
                velocity predictions are blended as `v = guidance_scale * v_pos + (1 - guidance_scale) * v_neg`.
                Mutually exclusive with `guidance_schedule` (setting both raises). Defaults to `None`.
            guidance_schedule (`list[float]` or `torch.Tensor`, *optional*):
                Per-step guidance scale schedule; must have length `num_inference_steps`. The first entry corresponds
                to the first step (largest noise level). Mutually exclusive with `guidance_scale`; exactly one must be
                set. Defaults to the recommended schedule (7.0 for the main steps, dropping to 3.0 for the final 3
                "polish" steps). To use a constant scale instead, pass `guidance_scale` and `guidance_schedule=None`.
            guidance_mode (`str`, *optional*, defaults to `"dual"`):
                How the unconditional (negative) velocity is produced for classifier-free guidance.
                - `"dual"` (Ideogram4's native asymmetric CFG): the unconditional velocity comes from the separate
                  `unconditional_transformer` checkpoint, which was trained specifically to denoise without text.
                - `"single"` (standard CFG from a single checkpoint): the conditional `transformer` is reused for the
                  unconditional pass by feeding it no prompt (`encoder_hidden_states=None`). This avoids loading /
                  running a second checkpoint at the cost of using the conditional model out of its trained regime.
            mu (`float`, *optional*, defaults to 0.0):
                Base mean of the logit-normal flow-matching schedule. The schedule mean is shifted by half the log of
                the resolution ratio relative to 512x512.
            std (`float`, *optional*, defaults to 1.5):
                Standard deviation of the logit-normal flow-matching schedule.
            prompt_upsampling (`bool`, *optional*, defaults to `False`):
                If `True`, rewrite `prompt` into Ideogram4's native structured JSON caption via
                [`~Ideogram4Pipeline.upsample_prompt`] before encoding. Requires the optional `prompt_enhancer_head`
                component; install `outlines` for schema-constrained captions. `generator` is reused to make the
                upsampling reproducible.
            prompt_upsampling_temperature (`float`, *optional*, defaults to 1.0):
                Sampling temperature for prompt upsampling when `prompt_upsampling=True`.
            max_sequence_length (`int`, *optional*, defaults to 2048):
                Maximum number of text tokens per prompt.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator(s) used to make sampling deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noise of shape `(batch_size, num_image_tokens, latent_dim)`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                One of `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`~pipelines.ideogram4.Ideogram4PipelineOutput`].
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary passed along to the attention processor of each transformer. A `"scale"` entry
                scales the loaded LoRA weights (e.g. `{"scale": 0.7}`) when the PEFT backend is active.
            callback_on_step_end (`Callable`, *optional*):
                Callback invoked at the end of every denoising step.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Names of tensors to expose to the callback via `callback_kwargs`.

        Examples:

        Returns:
            [`~pipelines.ideogram4.Ideogram4PipelineOutput`] or `tuple`.
        """
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_schedule=guidance_schedule,
            guidance_mode=guidance_mode,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        if guidance_mode == "dual" and self.unconditional_transformer is None:
            raise ValueError(
                "`guidance_mode='dual'` requires the `unconditional_transformer` component, which is not loaded. "
                "Either load it, or pass `guidance_mode='single'` to reuse the conditional `transformer` for the "
                "unconditional pass."
            )

        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 0. Optionally rewrite the prompt(s) into Ideogram4's native structured JSON caption.
        if prompt_upsampling:
            prompt = self.upsample_prompt(
                prompt,
                height=height,
                width=width,
                temperature=prompt_upsampling_temperature,
                max_new_tokens=max_sequence_length,
                generator=generator,
                device=device,
            )

        # 1. Image grid (drives both the packed layout and the latent shape).
        grid_h, grid_w = (
            height // (self.vae_scale_factor * self.patch_size),
            width // (self.vae_scale_factor * self.patch_size),
        )
        num_image_tokens = grid_h * grid_w

        # 2. Encode prompts into the packed conditioning (one entry per prompt).
        llm_features, position_ids, segment_ids, indicator = self.encode_prompt(
            prompt=prompt,
            grid_h=grid_h,
            grid_w=grid_w,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # 3. Replicate the conditioning for num_images_per_prompt.
        llm_features = _expand_tensor_to_effective_batch(llm_features, batch_size, num_images_per_prompt)
        position_ids = _expand_tensor_to_effective_batch(position_ids, batch_size, num_images_per_prompt)
        segment_ids = _expand_tensor_to_effective_batch(segment_ids, batch_size, num_images_per_prompt)
        indicator = _expand_tensor_to_effective_batch(indicator, batch_size, num_images_per_prompt)

        # 3b. Trim the shared left-padding prefix of the packed layout. Positions, indicators and text features are
        # per-token, so dropping fully-padded columns is exact; padded tokens sit in their own segment and never
        # interact with real tokens. With a single prompt (the common case) this removes ALL padding, making the
        # segment ids uniform, which lets the transformer skip the dense (L, L) attention mask entirely. That mask
        # is not just wasteful at high resolution: for packed sequences longer than sqrt(2^32) = 65536 tokens
        # (e.g. 4096x4096 -> 67584) the attention backends' 32-bit indexing silently overflows on the mask and
        # corrupts the trailing query rows, which shows up as a garbled horizontal band near the bottom of the image.
        first_real = int((segment_ids != SEQUENCE_PADDING_INDICATOR).any(dim=0).to(torch.long).argmax())
        if first_real > 0:
            llm_features = llm_features[:, first_real:]
            position_ids = position_ids[:, first_real:]
            segment_ids = segment_ids[:, first_real:]
            indicator = indicator[:, first_real:]

        # 4. Unconditional (image-only) branch: just the image-region slices of the packed layout. It carries no text
        # tokens, so the transformer receives `encoder_hidden_states=None` and skips text conditioning entirely
        # (previously this allocated a full image-length tensor of zeros).
        neg_position_ids = position_ids[:, -num_image_tokens:]
        neg_segment_ids = segment_ids[:, -num_image_tokens:]
        neg_indicator = indicator[:, -num_image_tokens:]

        # 4. Set up the resolution-aware logit-normal schedule on the scheduler.
        schedule_mu = _resolution_aware_mu(height=height, width=width, base_mu=mu)
        sigmas = _logit_normal_sigmas(num_inference_steps, schedule_mu, std=std, device=device)
        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 5. Resolve the per-step guidance schedule (a constant `guidance_scale` broadcasts to every step, otherwise
        # use the provided `guidance_schedule`, validated by `check_inputs`) and the tensor of per-step weights `gw`.
        if guidance_scale is not None:
            guidance_schedule = [float(guidance_scale)] * num_inference_steps
        gw = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)

        # 6. Prepare latents in the packed (B, num_image_tokens, latent_dim) layout.
        latent_dim = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_image_tokens=num_image_tokens,
            latent_dim=latent_dim,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 7. Padding for the text region of the conditional packed sequence (image latents are appended after it).
        # After the 3b trim the text region is `position_ids.shape[1] - num_image_tokens` wide, not max_sequence_length.
        max_text_tokens = position_ids.shape[1] - num_image_tokens
        text_z_padding = torch.zeros(
            batch_size * num_images_per_prompt,
            max_text_tokens,
            latent_dim,
            dtype=torch.float32,
            device=device,
        )

        # Choose the model that produces the unconditional (negative) velocity. In "dual" mode this is Ideogram4's
        # dedicated `unconditional_transformer`; in "single" mode the conditional `transformer` is reused with no
        # prompt, so a single checkpoint provides both the with- and without-prompt predictions (standard CFG).
        neg_transformer = self.transformer if guidance_mode == "single" else self.unconditional_transformer

        # The transformers run in their loaded compute dtype; cast the (otherwise float32) text features to match.
        # `latents` stay float32 for scheduler precision and are cast per-step at the transformer call below.
        llm_features = llm_features.to(self.transformer.dtype)

        # 8. Denoising loop. The scheduler stores `num_train_timesteps`-scaled timesteps; convert back to model time.
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Map sigma-domain timestep to model time `t` in [0, 1] (0 = noise, 1 = clean data).
                t_model = 1.0 - (t.float() / num_train_timesteps)
                t_model = t_model.expand(batch_size * num_images_per_prompt).to(self.transformer.dtype)

                # Conditional pass operates on the full packed sequence.
                pos_z = torch.cat([text_z_padding, latents], dim=1).to(self.transformer.dtype)
                pos_out = self.transformer(
                    hidden_states=pos_z,
                    timestep=t_model,
                    encoder_hidden_states=llm_features,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                # Velocity (and guidance) is computed in float32 for scheduler precision; the transformers
                # return their compute dtype, so cast the predicted velocities up here.
                pos_v = pos_out[:, max_text_tokens:].to(torch.float32)

                # Unconditional pass uses image-only positions with no text features. `neg_transformer` is either the
                # dedicated unconditional checkpoint ("dual") or the conditional one run without a prompt ("single").
                neg_v = neg_transformer(
                    hidden_states=latents.to(neg_transformer.dtype),
                    timestep=t_model,
                    encoder_hidden_states=None,
                    position_ids=neg_position_ids,
                    segment_ids=neg_segment_ids,
                    indicator=neg_indicator,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0].to(torch.float32)

                # Expose the current step's guidance weight via `self.guidance_scale` so callbacks can read it.
                self._guidance_scale = guidance_schedule[i]
                gw_i = gw[i]
                v = gw_i * pos_v + (1.0 - gw_i) * neg_v

                latents = self.scheduler.step(-v, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        # 9. Decode: unpatch the latents, denormalize with the VAE batch-norm stats, and decode through the VAE.
        if output_type == "latent":
            image = latents
        else:
            z = latents
            # VAE bn stores per-channel statistics on the packed-channel latent space (ae_channels * patch ** 2).
            bn_mean = self.vae.bn.running_mean.view(1, 1, -1).to(device=z.device, dtype=z.dtype)
            bn_std = torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps).view(1, 1, -1)
            bn_std = bn_std.to(device=z.device, dtype=z.dtype)
            z = z * bn_std + bn_mean

            patch = self.patch_size
            ae_channels = z.shape[-1] // (patch * patch)
            z = z.view(batch_size * num_images_per_prompt, grid_h, grid_w, patch, patch, ae_channels)
            z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
            z = z.view(batch_size * num_images_per_prompt, ae_channels, grid_h * patch, grid_w * patch)

            decoded = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(decoded.float(), output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return Ideogram4PipelineOutput(images=image)
