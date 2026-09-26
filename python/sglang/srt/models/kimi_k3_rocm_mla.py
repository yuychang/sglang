# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Kimi-K3 ROCm MLA decode Q/cache fusion.

K3 MLA has no RoPE, so one AITER kernel with identity RoPE can build the
fused Q and write the latent KV cache. ``kimi_k3.py`` calls these behind
``_is_hip``.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs


def k3_init_mla_q_cache(attn: nn.Module) -> None:
    """Register the identity-RoPE buffers the fused kernel reads."""
    enabled = envs.SGLANG_ROCM_K3_AITER_MLA_Q_CACHE_FUSION.get()
    attn._k3_mla_q_cache_fusion = enabled
    buffers = {
        "_k3_identity_rope_cos": torch.ones((1, 32), dtype=torch.bfloat16),
        "_k3_identity_rope_sin": torch.zeros((1, 32), dtype=torch.bfloat16),
        "_k3_mla_q_cache_scale": torch.ones((1,), dtype=torch.float32),
    }
    for name, value in buffers.items():
        attn.register_buffer(name, value if enabled else None, persistent=False)


def _cached_buffer(
    attn: nn.Module,
    name: str,
    shape: Tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    zero_init: bool = False,
) -> torch.Tensor:
    """Persistent workspace, grown on demand and sliced to the token count."""
    buf = getattr(attn, name, None)
    if (
        not isinstance(buf, torch.Tensor)
        or buf.dtype != dtype
        or buf.device != device
        or buf.shape[1:] != shape[1:]
        or buf.shape[0] < shape[0]
    ):
        buf = torch.empty(shape, dtype=dtype, device=device)
        if zero_init:
            buf.view(torch.uint8).zero_()
        setattr(attn, name, buf)
    return buf[: shape[0]]


def k3_try_fused_mla_q_cache(
    attn: nn.Module,
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (q, k placeholder) with the KV cache already written, or None."""
    if not attn._k3_mla_q_cache_fusion:
        return None

    from sglang.kernels.ops.kimi_k3 import mla_q_cache_aiter_hip
    from sglang.srt.layers.attention.aiter_mla_gluon import prefer_mla_gluon_decode
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    kv_pool = get_token_to_kv_pool()
    if isinstance(kv_pool, HiSparseDSATokenToKVPool):
        return None
    kv_cache = kv_pool.get_key_buffer(attn.attn_mqa.layer_id)
    q_scale = attn._k3_mla_q_cache_scale
    k_scale = attn.attn_mqa.k_scale
    if k_scale is None:
        k_scale = q_scale
    if not isinstance(k_scale, torch.Tensor):
        return None

    tokens, heads = q_nope_out.shape[0], q_nope_out.shape[1]
    if (
        q_nope_out.shape != (tokens, heads, attn.kv_lora_rank)
        or q_pe.shape != (tokens, heads, attn.qk_rope_head_dim)
        or k_nope.shape != (tokens, 1, attn.kv_lora_rank)
        or k_pe.shape != (tokens, 1, attn.qk_rope_head_dim)
        or out_cache_loc.shape != (tokens,)
        or positions.shape != (tokens,)
    ):
        return None

    # Triton and Gluon decode take BF16 Q. Gluon keeps the native head count.
    # The asm decode takes FP8 Q padded to 16 heads: write the real heads into
    # a persistent zeroed buffer so decode can skip the per-layer Fill.
    triton_decode = attn.current_attention_backend in ("triton", "triton_mla")
    gluon_decode = not triton_decode and prefer_mla_gluon_decode(
        head_pad_mode="zero",
        num_head=heads,
        kv_cache_dtype=kv_cache.dtype,
        q_dtype=torch.bfloat16,
    )
    bf16_q = triton_decode or gluon_decode
    q_out_dtype = q_nope_out.dtype if bf16_q else kv_cache.dtype
    pad_heads = (
        heads if gluon_decode else (16 if (heads < 16 and 16 % heads != 0) else heads)
    )
    head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
    q_out = _cached_buffer(
        attn,
        "_k3_mla_q_out",
        (tokens, pad_heads, head_dim),
        q_out_dtype,
        q_nope_out.device,
        zero_init=pad_heads > heads,
    )
    cos = attn._k3_identity_rope_cos
    sin = attn._k3_identity_rope_sin
    out_cache_loc = out_cache_loc.contiguous()
    if mla_q_cache_aiter_hip.covered(
        q_nope_out,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        out_cache_loc,
        positions,
        k_scale,
        cos,
        sin,
        q_out,
        q_scale=q_scale,
    ):
        q = mla_q_cache_aiter_hip.run(
            q_nope=q_nope_out,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            kv_cache=kv_cache,
            slot_mapping=out_cache_loc,
            positions=positions,
            k_scale=k_scale,
            cos_cache=cos,
            sin_cache=sin,
            out=q_out,
            q_scale=q_scale,
        )
    else:
        q, _, _, _ = fused_qk_rope_cat_and_cache_mla(
            q_nope_out,
            q_pe,
            k_nope,
            k_pe,
            kv_cache,
            out_cache_loc,
            positions,
            cos,
            sin,
            k_scale,
            True,
            q_scale=q_scale,
            q_out_dtype=q_out_dtype,
            compute_all_q_rope=False,
            identity_rope=True,
        )
    # The cache is already written, so the backend never reads K.
    k = _cached_buffer(
        attn,
        "_k3_mla_k_placeholder",
        (tokens, 1, head_dim),
        k_nope.dtype,
        k_nope.device,
    )
    return q, k
