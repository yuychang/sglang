"""CUDA JIT K3 MLA output gate: out = x * sigmoid(gate) in one kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_THREADS: int = 256


@triton.jit
def _mla_output_gate_fp8_per_token_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    scale_ptr,
    n_cols: tl.constexpr,
    stride_xm,
    stride_gm,
    stride_om,
    fp8_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(
        gate_ptr + row * stride_gm + cols, mask=mask, other=0.0
    ).to(tl.float32)

    # Match the existing BF16 gate kernel + dynamic quantizer: sigmoid rounds
    # to BF16, then the multiply rounds to BF16 before amax/quantization.
    sigmoid = tl.sigmoid(gate).to(tl.bfloat16)
    gated = (x * sigmoid.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.where(mask, tl.abs(gated), 0.0))
    scale = amax / fp8_max
    inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    q = tl.minimum(tl.maximum(gated * inv, -fp8_max), fp8_max)
    tl.store(
        out_ptr + row * stride_om + cols,
        q.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(scale_ptr + row, scale)


@cache_once
def _jit_mla_output_gate_module() -> Module:
    args = make_cpp_args(_THREADS, is_arch_support_pdl())
    return load_jit(
        "kimi_k3_mla_output_gate_" + str(_THREADS),
        *args,
        cuda_files=["kimi_k3/mla_output_gate.cuh"],
        cuda_wrappers=[("run", f"MlaOutputGateKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def covered(x: torch.Tensor, gate: torch.Tensor) -> bool:
    return (
        x.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and x.shape == gate.shape
        and x.is_contiguous()
        and gate.is_contiguous()
        and x.numel() % 8 == 0
        and x.numel() > 0
    )


def kimi_k3_mla_output_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """out = bf16(x * bf16(sigmoid(gate))); double rounding matches the
    unfused torch.sigmoid + mul pair bit-for-bit. Caller checks covered()."""
    out = torch.empty_like(x)
    _jit_mla_output_gate_module().run(x.view(-1), gate.view(-1), out.view(-1))
    return out


def kimi_k3_mla_output_gate_fp8_per_token(
    x: torch.Tensor, gate: torch.Tensor, quant_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the BF16 output gate and PTPC activation quantization."""
    assert covered(x, gate)
    rows, cols = x.reshape(-1, x.shape[-1]).shape
    x_2d = x.reshape(rows, cols)
    gate_2d = gate.reshape(rows, cols)
    out = torch.empty_like(x_2d, dtype=quant_dtype)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _mla_output_gate_fp8_per_token_kernel[(rows,)](
        x_2d,
        gate_2d,
        out,
        scale,
        n_cols=cols,
        stride_xm=x_2d.stride(0),
        stride_gm=gate_2d.stride(0),
        stride_om=out.stride(0),
        fp8_max=float(torch.finfo(quant_dtype).max),
        BLOCK=triton.next_power_of_2(cols),
        num_warps=4,
    )
    return out.reshape(x.shape), scale
