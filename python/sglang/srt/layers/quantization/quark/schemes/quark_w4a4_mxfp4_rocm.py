"""ROCm-only MXFP4 W4A4 dense linears for Quark OCP-MX checkpoints.

Mirrors vLLM's ROCm MXFP4 linear kernels (``kernels/linear/mxfp4``):

* ``fp4`` (default): ``AiterMxfp4LinearKernel``. Triton ``dynamic_mxfp4_quant``
  + ``gemm_afp4wfp4``; with ``SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM=1``,
  ``per_1x32_f4_quant_hip`` + ASM ``gemm_a4w4`` on preshuffled weights, or the
  preshuffled Triton GEMM for M <= 64 where AITER has a tuned config.
* ``emulate``: ``EmulationMxfp4LinearKernel`` (the ``QuarkOCP_MX.emulate``
  setting used for the checkpoint's published accuracy). The activation is
  MXFP4 quantized and dequantized (OCP "even" scale, round-half-even e2m1),
  the weight is dequantized, and the GEMM runs in bf16.

Selected by ``SGLANG_ROCM_QUARK_MXFP4_LINEAR_ACT``.
"""

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MXFP4_LINEAR_MODES = ("fp4", "emulate")
_OCP_MX_BLOCK_SIZE = 32

_ASM_FP4_SCALE_ROW_MULTIPLE = 32
_ASM_FP4_SCALE_COL_MULTIPLE = 8


def resolve_mxfp4_linear_mode() -> str:
    mode = (envs.SGLANG_ROCM_QUARK_MXFP4_LINEAR_ACT.get() or "fp4").lower()
    if mode not in MXFP4_LINEAR_MODES:
        raise ValueError(
            "SGLANG_ROCM_QUARK_MXFP4_LINEAR_ACT must be one of "
            f"{', '.join(MXFP4_LINEAR_MODES)}; got {mode!r}"
        )
    return mode


def dequant_mxfp4(
    packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Packed e2m1 ``(M, K//2)`` uint8 + e8m0 ``(M, K//32)`` uint8 -> ``(M, K)``.

    Every e2m1 value times a power-of-two scale is exact in bf16, so this
    matches Quark's ``dq_mxfp4``.
    """
    rows, k_packed = packed.shape
    lo = (packed & 0xF).to(torch.int32)
    hi = (packed >> 4).to(torch.int32)
    codes = torch.stack((lo, hi), dim=-1).view(rows, k_packed * 2)
    mag = codes & 0x7
    val = torch.where(mag <= 4, mag.to(torch.float32) * 0.5, (mag - 2).to(torch.float32))
    val = torch.where(mag == 7, torch.full_like(val, 6.0), val)
    val = torch.where(codes >= 8, -val, val)
    s = torch.exp2(scale[:rows].to(torch.float32) - 127.0)
    s = torch.where(scale[:rows] == 255, torch.zeros_like(s), s)
    val = val.view(rows, -1, _OCP_MX_BLOCK_SIZE) * s.unsqueeze(-1)
    return val.view(rows, k_packed * 2).to(dtype)


def quant_dequant_mxfp4(x: torch.Tensor) -> torch.Tensor:
    """MXFP4 QDQ of a 2-D activation with AITER's OCP MX quantizer."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    x_q, x_s = dynamic_mxfp4_quant(x)
    return dequant_mxfp4(x_q.view(torch.uint8), x_s.view(torch.uint8), x.dtype)


def _asm_fp4_scale_swizzle_supported(weight_scale: torch.Tensor) -> bool:
    if weight_scale.ndim != 2:
        return False
    sm, sn = weight_scale.shape
    return (
        sm % _ASM_FP4_SCALE_ROW_MULTIPLE == 0
        and sn % _ASM_FP4_SCALE_COL_MULTIPLE == 0
    )


def _presh_gemm_tuned(n: int, k_bytes: int) -> bool:
    try:
        from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

        return bool(get_gemm_config("GEMM-AFP4WFP4_PRESHUFFLED", 128, n, 4 * k_bytes)[1])
    except (AssertionError, ImportError):
        return False


def process_weights_after_loading(layer: torch.nn.Module, mode: str) -> None:
    """Set ``layer.mxfp4_rocm_kernel`` to ``emulate``, ``asm`` or ``triton``."""
    if mode == "emulate":
        w = dequant_mxfp4(
            layer.weight.data.view(torch.uint8),
            layer.weight_scale.data.view(torch.uint8),
            torch.bfloat16,
        )
        layer.weight = torch.nn.Parameter(w.contiguous(), requires_grad=False)
        layer.weight_scale = None
        layer.mxfp4_rocm_kernel = "emulate"
        return

    use_asm = envs.SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM.get()
    if use_asm and not _asm_fp4_scale_swizzle_supported(layer.weight_scale.data):
        logger.warning(
            "AITER ASM FP4 GEMM needs weight_scale dims divisible by (%d, %d), got "
            "%s; this layer uses the Triton FP4 GEMM.",
            _ASM_FP4_SCALE_ROW_MULTIPLE,
            _ASM_FP4_SCALE_COL_MULTIPLE,
            tuple(layer.weight_scale.shape),
        )
        use_asm = False
    if not use_asm:
        layer.mxfp4_rocm_kernel = "triton"
        return

    from aiter.ops.shuffle import shuffle_weight

    weight_scale = layer.weight_scale.data
    sm, sn = weight_scale.shape
    weight_scale = weight_scale.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    weight_scale = weight_scale.permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm, sn)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.weight = torch.nn.Parameter(
        shuffle_weight(layer.weight.data, layout=(16, 16)), requires_grad=False
    )
    layer.mxfp4_rocm_kernel = "asm"


def _gemm_asm(layer, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    from aiter import gemm_a4w4, per_1x32_f4_quant_hip
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle

    weight, weight_scale = layer.weight, layer.weight_scale
    m, n, k_bytes = x.shape[0], weight.shape[0], weight.shape[1]
    if m <= 64 and _presh_gemm_tuned(n, k_bytes):
        x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=m >= 32)
        if m >= 32:
            x_s = x_s.view(torch.uint8).view(x_s.shape[0] // 32, -1)
        else:
            x_s = x_s[:m, ...].view(torch.uint8)
        y = torch.empty(m, n, device=x.device, dtype=out_dtype)
        gemm_afp4wfp4_preshuffle(
            x_q.view(torch.uint8),
            weight.view(torch.uint8).view(n // 16, -1),
            x_s,
            weight_scale.view(torch.uint8).view(weight_scale.shape[0] // 32, -1),
            out_dtype,
            y,
        )
        return y
    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
    y = gemm_a4w4(
        x_q,
        weight.view(x_q.dtype),
        x_s,
        weight_scale.view(x_s.dtype),
        dtype=out_dtype,
        bpreshuffle=True,
    )
    return y[:m]


def apply_weights(
    layer: torch.nn.Module,
    x: torch.Tensor,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """y = x @ W^T for a 2-D bf16 activation; written into ``out`` if given."""
    kernel = layer.mxfp4_rocm_kernel
    if kernel == "emulate":
        qdq_x = quant_dequant_mxfp4(x)
        w = layer.weight.to(x.dtype)
        if out is not None and bias is None:
            return torch.mm(qdq_x, w.t(), out=out)
        y = torch.nn.functional.linear(qdq_x, w, bias)
    elif kernel == "asm":
        y = _gemm_asm(layer, x, out_dtype)
        if bias is not None:
            y = y + bias
    else:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        x_q, x_s = dynamic_mxfp4_quant(x)
        y = out
        if y is None or bias is not None:
            y = torch.empty(x.shape[0], layer.weight.shape[0], device=x.device, dtype=out_dtype)
        gemm_afp4wfp4(x_q, layer.weight, x_s, layer.weight_scale, out_dtype, y)
        if bias is not None:
            y = y + bias
    if out is not None and y.data_ptr() != out.data_ptr():
        out.copy_(y)
        return out
    return y
