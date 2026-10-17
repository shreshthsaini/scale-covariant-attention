"""Inference-only MUSIQ attention with bounded query memory and unchanged keys.

This adapter changes execution order, not patch selection or attention semantics.
In particular, it preserves PyIQA's finite -1000 mask even for masked queries.
The original installed implementation and archived evaluator remain untouched.
"""

import hashlib
import inspect
from pathlib import Path
from types import MethodType

POLICY = "musiq_all_keys_query_chunks_v1"


def chunked_attention(self, x, mask=None):
    if self.training or __import__("torch").is_grad_enabled():
        raise RuntimeError("MUSIQ chunked adapter requires eval mode and inference/no_grad")
    batch, length, channels = x.shape
    heads = self.num_heads
    q = self.query(x).reshape(batch, length, heads, channels // heads).permute(0, 2, 1, 3)
    k = self.key(x).reshape(batch, length, heads, channels // heads).permute(0, 2, 1, 3)
    v = self.value(x).reshape(batch, length, heads, channels // heads).permute(0, 2, 1, 3)
    # Allocate final output once. Only the query axis is split; every chunk sees
    # every key, with the same scaling, mask constant and row softmax.
    output = x.new_empty(batch, length, channels)
    for start in range(0, length, self._sca_query_chunk):
        end = min(start + self._sca_query_chunk, length)
        attn = (q[:, :, start:end] @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            pair_mask = mask[:, start:end].reshape(batch, 1, end - start, 1) * mask.reshape(batch, 1, 1, length)
            attn = attn.masked_fill(pair_mask == 0, -1e3)
        attn = self.attn_drop(attn.softmax(dim=-1))
        output[:, start:end] = (attn @ v).transpose(1, 2).reshape(batch, end - start, channels)
        del attn
    return self.out_drop(self.out(output))


def install(model, query_chunk=256):
    from pyiqa.archs.musiq_arch import MultiHeadAttention

    if query_chunk < 1 or model.training:
        raise ValueError("Positive query chunk and eval-mode model required")
    count = 0
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            module._sca_query_chunk = int(query_chunk)
            module.forward = MethodType(chunked_attention, module)
            count += 1
    if not count:
        raise ValueError("No MUSIQ attention modules found")
    source = Path(inspect.getfile(MultiHeadAttention))
    return {"execution_adapter": POLICY, "query_chunk": query_chunk,
            "attention_modules": count, "native_patches": "all, unchanged",
            "mask": "original finite -1000; all keys retained",
            "pyiqa_musiq_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
