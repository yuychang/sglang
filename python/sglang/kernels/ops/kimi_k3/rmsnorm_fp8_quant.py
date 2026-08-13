"""Ungated RMSNorm fused with per-token FP8 quantization for K3 ptpc_fp8."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fp8_per_token_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    s_ptr,
    N,
    eps,
    fp8_max,
    stride_xm,
    stride_ym,
    BLOCK: tl.constexpr,
):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x_off = tok * stride_xm + cols
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = x * rstd * w
    amax = tl.max(tl.abs(normed))
    scale = amax / fp8_max
    inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    q = tl.minimum(tl.maximum(normed * inv, -fp8_max), fp8_max)
    y_off = tok * stride_ym + cols
    tl.store(y_ptr + y_off, q.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + tok, scale)


@triton.jit
def _rmsnorm_fp8_residual_kernel(
    x_ptr,
    r_ptr,
    ro_ptr,
    w_ptr,
    y_ptr,
    s_ptr,
    N,
    eps,
    fp8_max,
    stride_xm,
    stride_rm,
    stride_ym,
    stride_rom,
    BLOCK: tl.constexpr,
):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + tok * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + tok * stride_rm + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + r
    tl.store(ro_ptr + tok * stride_rom + cols, x, mask=mask)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = x * rstd * w
    amax = tl.max(tl.abs(normed))
    scale = amax / fp8_max
    inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    q = tl.minimum(tl.maximum(normed * inv, -fp8_max), fp8_max)
    tl.store(y_ptr + tok * stride_ym + cols, q.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + tok, scale)


def rmsnorm_fp8_per_token(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
    quant_dtype: Optional[torch.dtype] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Fused RMSNorm (+ optional residual) and per-token FP8 quant."""
    if quant_dtype is None:
        from sglang.srt.layers.quantization.fp8_utils import is_fp8_fnuz

        quant_dtype = (
            torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
        )
    assert x.ndim == 2
    tokens, hidden = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    fp8_max = float(torch.finfo(quant_dtype).max)
    out = torch.empty((tokens, hidden), dtype=quant_dtype, device=x.device)
    scale = torch.empty((tokens, 1), dtype=torch.float32, device=x.device)
    if tokens == 0:
        if residual is None:
            return out, scale
        return (out, scale), residual
    block = triton.next_power_of_2(hidden)
    if residual is None:
        _rmsnorm_fp8_per_token_kernel[(tokens,)](
            x,
            weight,
            out,
            scale,
            hidden,
            float(eps),
            fp8_max,
            x.stride(0),
            out.stride(0),
            BLOCK=block,
        )
        return out, scale
    if residual.stride(-1) != 1:
        residual = residual.contiguous()
    residual_out = torch.empty_like(x)
    _rmsnorm_fp8_residual_kernel[(tokens,)](
        x,
        residual,
        residual_out,
        weight,
        out,
        scale,
        hidden,
        float(eps),
        fp8_max,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        residual_out.stride(0),
        BLOCK=block,
    )
    return (out, scale), residual_out
