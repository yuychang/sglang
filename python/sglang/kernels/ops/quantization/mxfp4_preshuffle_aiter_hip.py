"""AITER MXFP4 quantization of a BF16 weight into the preshuffled gemm_a4w4 layout."""

from __future__ import annotations

import functools

import torch

_PRESHUFFLE_LAYOUT = (16, 16)
# gemm_a4w4 and its preshuffled scales read whole 256x256 tiles.
_TILE = 256


@functools.cache
def _ops():
    try:
        from aiter import QuantType, dtypes
        from aiter.ops.quant import get_hip_quant
        from aiter.ops.shuffle import shuffle_weight
    except ImportError:
        return None
    return get_hip_quant(QuantType.per_1x32), shuffle_weight, dtypes.fp4x2


def available() -> bool:
    return _ops() is not None


def pack(*, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Preshuffled MXFP4 weight and E8M0 scales for a BF16 [out, in] weight."""
    if (
        weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or weight.shape[0] == 0
        or weight.shape[0] % _TILE != 0
        or weight.shape[1] == 0
        or weight.shape[1] % _TILE != 0
    ):
        raise ValueError(
            "MXFP4 needs a nonempty 2D BF16 weight with both dimensions aligned "
            f"to {_TILE}, got {weight.dtype} {tuple(weight.shape)}"
        )
    ops = _ops()
    if ops is None:
        raise RuntimeError("AITER MXFP4 quantization is unavailable")
    quant, shuffle, fp4x2 = ops
    q, scale = quant(weight.contiguous(), quant_dtype=fp4x2, shuffle=True)
    return shuffle(q, layout=_PRESHUFFLE_LAYOUT), scale
