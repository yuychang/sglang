"""Per-head sigmoid-gated RMSNorm fused with per-token FP8 quantization.

Kimi-K3's KDA output path is ``o_proj(rmsnorm(core_attn_out) * w * sigmoid(g))``.
When ``o_proj`` runs a per-token per-channel FP8 GEMM, the unfused form costs
three passes over the ``[tokens, heads * head_dim]`` activation: the gated norm
reads it and writes bf16, the quant kernel reads that and writes fp8 plus a
scale, and only then does the GEMM read it. This kernel does all of it in one
pass, handing the GEMM its ``(fp8, scale)`` pair directly.

The gate is read through explicit (outer, head) strides so a column slice of the
fused in-projection output can be consumed in place, with no contiguous copy.

Ported from ATOM's ``atom/model_ops/kimi_k3/activations.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_gated_fp8_per_token_kernel(
    x_ptr,
    w_ptr,
    g_ptr,
    y_ptr,
    s_ptr,
    H,
    eps,
    fp8_max,
    stride_xm,
    stride_xh,
    stride_g_outer,
    stride_g_head,
    stride_ym,
    HEADS: tl.constexpr,
    HEADS_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tok = tl.program_id(0)
    head_ids = tl.arange(0, HEADS_POW2)
    cols = tl.arange(0, BLOCK)
    mask = (head_ids[:, None] < HEADS) & (cols[None, :] < H)  # [HEADS_POW2, BLOCK]
    # Padding heads (head_ids >= HEADS) are masked out on every load and store,
    # but their raw offset (head_ids * stride) can still address past the end of
    # the buffer -- forming an out-of-bounds pointer is UB in triton and faults
    # on ROCm when the allocation abuts an unmapped page. Clamp the head index
    # used for addressing to a valid row; the mask (other=0.0) still discards
    # the value, so numerics are unchanged.
    h_safe = tl.where(head_ids < HEADS, head_ids, 0)
    x_off = tok * stride_xm + h_safe[:, None] * stride_xh + cols[None, :]
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    # RMSNorm is per (token, head) over head_dim -- not over the flattened row.
    var = tl.sum(x * x, axis=1) / H  # [HEADS_POW2]
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=cols < H, other=0.0).to(tl.float32)  # [BLOCK]
    g_off = tok * stride_g_outer + h_safe[:, None] * stride_g_head + cols[None, :]
    gate = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
    normed = (x * rstd[:, None] * w[None, :]) * tl.sigmoid(gate)
    # The quant scale is per token, so the amax spans every head of the row.
    # Masked lanes hold 0.0 and cannot raise it.
    amax = tl.max(tl.abs(normed))
    scale = amax / fp8_max
    inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    q = normed * inv
    q = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
    y_off = tok * stride_ym + h_safe[:, None] * H + cols[None, :]
    tl.store(y_ptr + y_off, q.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + tok, scale)


def rmsnorm_gated_fp8_per_token(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    quant_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``rmsnorm(x) * weight * sigmoid(gate)``, quantized per token to FP8.

    Args:
        x: ``[tokens, heads, head_dim]`` attention output. Normed per head.
        weight: ``[head_dim]`` RMSNorm gain, shared across heads.
        gate: ``[tokens, heads, head_dim]``, may be strided.
        eps: RMSNorm epsilon.
        quant_dtype: the FP8 dtype the consuming GEMM expects.

    Returns:
        ``(out [tokens, heads * head_dim] quant_dtype, scale [tokens, 1] fp32)``
        -- the ``(q_input, x_scale)`` pair ``apply_fp8_ptpc_linear`` takes.
    """
    assert x.ndim == 3, f"expected [tokens, heads, head_dim], got {tuple(x.shape)}"
    assert gate.shape == x.shape, f"gate {tuple(gate.shape)} != x {tuple(x.shape)}"
    tokens, heads, head_dim = x.shape
    # Rows and heads are addressed by stride, but head_dim is walked with a
    # plain arange, so only the innermost dim has to be packed.
    if x.stride(-1) != 1:
        x = x.contiguous()
    if gate.stride(-1) != 1:
        gate = gate.contiguous()
    fp8_max = float(torch.finfo(quant_dtype).max)
    out = torch.empty((tokens, heads * head_dim), dtype=quant_dtype, device=x.device)
    scale = torch.empty((tokens, 1), dtype=torch.float32, device=x.device)
    if tokens == 0:
        return out, scale
    _rmsnorm_gated_fp8_per_token_kernel[(tokens,)](
        x,
        weight,
        gate,
        out,
        scale,
        head_dim,
        float(eps),
        fp8_max,
        x.stride(0),
        x.stride(1),
        gate.stride(0),
        gate.stride(1),
        out.stride(0),
        HEADS=heads,
        HEADS_POW2=triton.next_power_of_2(heads),
        BLOCK=triton.next_power_of_2(head_dim),
    )
    return out, scale
