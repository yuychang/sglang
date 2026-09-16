"""CUDA JIT K3 MLA output gate: out = x * sigmoid(gate) in one kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_THREADS: int = 256
_is_npu = is_npu()


@cache_once
def _jit_mla_output_gate_module() -> Module:
    args = make_cpp_args(_THREADS, is_arch_support_pdl())
    return load_jit(
        "kimi_k3_mla_output_gate_fp8quant2_" + str(_THREADS),
        *args,
        cuda_files=["kimi_k3/mla_output_gate.cuh"],
        cuda_wrappers=[
            ("run", f"MlaOutputGateKernel<{args}>::run"),
            ("run_fp8_quant", f"MlaOutputGateFp8QuantKernel<{args}>::run"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def covered(x: torch.Tensor, gate: torch.Tensor) -> bool:
    return (
        not _is_npu
        and x.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and x.shape == gate.shape
        and x.is_contiguous()
        and gate.is_contiguous()
        and x.numel() % 8 == 0
        and x.numel() > 0
    )


def covered_fp8_quant(x: torch.Tensor, gate: torch.Tensor) -> bool:
    return (
        covered(x, gate)
        and x.dim() == 2
        and x.shape[-1] % 8 == 0
        and x.shape[0] > 0
    )


def kimi_k3_mla_output_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """out = bf16(x * bf16(sigmoid(gate))); double rounding matches the
    unfused torch.sigmoid + mul pair bit-for-bit. Caller checks covered()."""
    out = torch.empty_like(x)
    _jit_mla_output_gate_module().run(x.view(-1), gate.view(-1), out.view(-1))
    return out


def kimi_k3_mla_output_gate_fp8_quant(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate multiply + per-token FP8 quant. Returns ``(fp8, [T, 1] scale)``."""
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

    out_q = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
    out_s = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    _jit_mla_output_gate_module().run_fp8_quant(
        x, gate, out_q.view(torch.uint8), out_s
    )
    return out_q, out_s.unsqueeze(1)
