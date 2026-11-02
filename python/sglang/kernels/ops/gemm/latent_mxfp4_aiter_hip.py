"""AITER gfx950 MXFP4 GEMM for Kimi-K3 latent projections."""

from __future__ import annotations

import functools

import torch


@functools.cache
def _ops():
    try:
        from aiter import QuantType, dtypes, gemm_a4w4
        from aiter.ops.quant import get_hip_quant
    except ImportError:
        return None
    return get_hip_quant(QuantType.per_1x32), gemm_a4w4, dtypes.fp4x2


def supported(*, device: torch.device | None = None) -> bool:
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    if device is not None and device.type != "cuda":
        return False
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    return arch == "gfx950" and _ops() is not None


def run(*, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x @ weight.T after per-1x32 MXFP4 activation quantization.

    The weight comes from quantization.mxfp4_preshuffle_aiter_hip.pack on a
    device supported() accepted; this hot path does not re-probe the device.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[0] == 0:
        raise ValueError("latent MXFP4 GEMM needs nonempty 2D BF16 activations")
    if (
        weight.ndim != 2
        or x.shape[1] != weight.shape[1] * 2
        or weight.device != x.device
        or scale.device != x.device
        or scale.numel() < weight.shape[0] * x.shape[1] // 32
        or not weight.is_contiguous()
        or not scale.is_contiguous()
    ):
        raise ValueError("latent MXFP4 weight and scales must match the activations")
    ops = _ops()
    if ops is None:
        raise RuntimeError("latent MXFP4 GEMM requires AITER")
    quant, gemm, fp4x2 = ops
    if weight.dtype != fp4x2:
        raise ValueError("latent MXFP4 GEMM needs packed FP4 weights from pack()")
    rows = x.shape[0]
    xq, x_scale = quant(x.contiguous(), quant_dtype=fp4x2, shuffle=True)
    return gemm(xq, weight, x_scale, scale, dtype=x.dtype, bpreshuffle=True)[:rows]
