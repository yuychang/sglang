# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused PTPC quant + shared-down FP8 GEMM entry point for gfx950.

The kernel is built per decode CUDA-graph bucket. Which buckets it actually
owns is a measurement result, not a property of the kernel: it has to beat
``per_token_quant_hip`` plus the tuned ``gemm_a8w8_bpreshuffle`` instance for
that exact ``M``, and the tuned GEMM's MFMA tiles amortize the FP8 weight
stream better as ``M`` grows. ``SGLANG_K3_PTPC_FUSED_SHARED_DOWN_TOKENS``
carries the winning set so the sweep can widen it without a code change.
"""

from __future__ import annotations

import functools

import torch
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.utils import is_flydsl_available

from .kernels.kimi_k3_shared_down_ptpc_fp8_gfx950 import (
    build_kimi_k3_shared_down_ptpc_fp8_module,
)

_HIDDEN_SIZE = 7168
_SHARED_INTERMEDIATE_SIZE = 768
# Buckets the kernel can be built for. Membership here only means "compiles
# and is correct"; `enabled_token_buckets` decides what runs.
BUILDABLE_TOKEN_BATCHES = (1, 2, 4, 8, 16)
# (rows_per_wave, weight_cache_modifier) per bucket, filled in from
# bench_shared_down_ptpc_fp8.py. Unmeasured buckets fall back to the default.
_DEFAULT_TUNING = (2, 2)
_LAUNCH_TUNING: dict[int, tuple[int, int]] = {}
_CU_COUNT = 248


def is_available() -> bool:
    return is_flydsl_available() and get_gfx_runtime() == "gfx950"


def enabled_token_buckets() -> tuple[int, ...]:
    """Decode batches the fused kernel is allowed to own."""
    from sglang.srt.environ import envs

    raw = envs.SGLANG_K3_PTPC_FUSED_SHARED_DOWN_TOKENS.get()
    buckets = []
    for field in str(raw).split(","):
        field = field.strip()
        if not field:
            continue
        try:
            value = int(field)
        except ValueError:
            continue
        if value in BUILDABLE_TOKEN_BATCHES:
            buckets.append(value)
    return tuple(sorted(set(buckets)))


def supports_kimi_k3_shared_down_ptpc_fp8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    token_buckets: tuple[int, ...] | None = None,
) -> bool:
    buckets = enabled_token_buckets() if token_buckets is None else token_buckets
    tensors = (activation, weight, weight_scale)
    return (
        activation.is_cuda
        and activation.dtype == torch.bfloat16
        and activation.ndim == 2
        and activation.shape[0] in buckets
        and activation.shape[1] == _SHARED_INTERMEDIATE_SIZE
        and activation.is_contiguous()
        and weight.dtype == torch.float8_e4m3fn
        and tuple(weight.shape) == (_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE)
        and weight.is_contiguous()
        and weight_scale.dtype == torch.float32
        and weight_scale.numel() == _HIDDEN_SIZE
        and weight_scale.is_contiguous()
        and len({tensor.device for tensor in tensors}) == 1
        and is_available()
    )


def quantize_shared_down_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-quantize ``[7168, 768]`` BF16 to natural-layout FP8 plus scales.

    The fused kernel reads the weight row-major rather than bpreshuffled, so
    it cannot share the packed tensor the two-launch path uses.
    """
    if (
        not weight.is_cuda
        or weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE)
    ):
        raise ValueError("shared-down weight must be CUDA BF16 [7168, 768]")
    source = weight.float()
    amax = source.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    packed = (
        (source / scale[:, None])
        .clamp(min=-448.0, max=448.0)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return packed, scale.float().contiguous()


@functools.cache
def _launcher(num_tokens: int):
    rows_per_wave, weight_cache_modifier = _LAUNCH_TUNING.get(
        num_tokens,
        _DEFAULT_TUNING,
    )
    return build_kimi_k3_shared_down_ptpc_fp8_module(
        num_tokens=num_tokens,
        rows_per_wave=rows_per_wave,
        cu_count=_CU_COUNT,
        waves_per_eu=0,
        weight_cache_modifier=weight_cache_modifier,
    )


def kimi_k3_shared_down_ptpc_fp8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    token_buckets: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not supports_kimi_k3_shared_down_ptpc_fp8(
        activation,
        weight,
        weight_scale,
        token_buckets=token_buckets,
    ):
        raise ValueError("unsupported Kimi-K3 fused PTPC shared-down inputs")

    if out is None:
        output = activation.new_empty((activation.shape[0], _HIDDEN_SIZE))
    elif (
        out.device != activation.device
        or out.dtype != torch.bfloat16
        or tuple(out.shape) != (activation.shape[0], _HIDDEN_SIZE)
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous BF16 [M,7168] on the same device")
    else:
        output = out
    _launcher(int(activation.shape[0]))(
        ptr_arg(activation),
        ptr_arg(weight),
        ptr_arg(weight_scale.view(-1)),
        ptr_arg(output),
        stream=torch.cuda.current_stream(activation.device),
    )
    return output


def warmup(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    token_buckets: tuple[int, ...] | None = None,
) -> None:
    """Compile the enabled buckets outside CUDA-graph capture."""
    buckets = enabled_token_buckets() if token_buckets is None else token_buckets
    for num_tokens in buckets:
        activation = torch.zeros(
            (num_tokens, _SHARED_INTERMEDIATE_SIZE),
            dtype=torch.bfloat16,
            device=weight.device,
        )
        if supports_kimi_k3_shared_down_ptpc_fp8(
            activation,
            weight,
            weight_scale,
            token_buckets=buckets,
        ):
            kimi_k3_shared_down_ptpc_fp8(
                activation,
                weight,
                weight_scale,
                token_buckets=buckets,
            )
    torch.cuda.synchronize(weight.device)
