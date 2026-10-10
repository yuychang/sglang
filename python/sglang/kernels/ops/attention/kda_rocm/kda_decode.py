# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm entry points for the fused Kimi-K3 KDA decode kernel.

Ported from vLLM ``vllm/models/kimi_k3/amd/ops/kda_decode.py``. The kernel in
``csrc/fused_kda_decode_kernel_rocm.cu`` replaces, for a pure non-speculative
decode batch, the packed causal conv1d update, the recurrent delta-rule step
and the gated output RMSNorm.

The kernel wants a width-major conv weight and an fp32 norm weight. vLLM
stages both in its weight loaders; SGLang stages the same copies once after
loading (:func:`stage_decode_conv1d_weight`, :func:`stage_decode_norm_weight`).
"""

from typing import Optional

import torch

from sglang.kernels.ops.attention.kda_rocm import jit

# Head counts the kernel is instantiated for (Kimi-K3 has 96 KDA heads, so this
# covers TP1/2/4/8).
SUPPORTED_NUM_HEADS = (12, 24, 48, 96)


def is_fused_kda_decode_supported(
    num_heads: int,
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
) -> bool:
    """Whether the fused decode kernel can serve this layer on this device."""
    if (
        num_heads not in SUPPORTED_NUM_HEADS
        or head_dim != 128
        or conv_width != 4
        or num_spec != 0
        or input_dtype != torch.bfloat16
        or conv_state_dtype != torch.bfloat16
    ):
        return False
    # gfx950 (MI355X) and gfx942 (MI325X): both CDNA, sharing the wave64 / DPP /
    # bf16 primitives the kernel relies on.
    return jit.device_arch() in ("gfx942", "gfx950") and jit.has_op(
        "fused_kda_decode"
    )


def stage_decode_conv1d_weight(conv_weight: torch.Tensor) -> torch.Tensor:
    """``[3 * LP, width]`` packed q/k/v conv weight -> fp32 ``[3, width, LP]``.

    Same copy as vLLM's ``make_decode_conv1d_weight_loader``: the fused kernel
    indexes the weight as ``[qkv, width, channel]`` so channels are contiguous.
    """
    w = conv_weight.reshape(conv_weight.shape[0], -1)
    lp = w.shape[0] // 3
    return w.float().view(3, lp, w.shape[1]).transpose(1, 2).contiguous()


def stage_decode_norm_weight(norm_weight: torch.Tensor) -> torch.Tensor:
    return norm_weight.detach().float().contiguous()


def fused_kda_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    lower_bound: Optional[float] = None,
    output_gate: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
) -> None:
    """conv1d update + KDA recurrence (+ gated RMSNorm) for one token per row.

    ``x`` is ``[B, 3 * H * 128]``; ``conv_state`` ``[slots, 3 * H * 128, 3]``
    (SD or DS layout); ``state`` fp32 ``[slots, H, 128, 128]`` updated in place
    at ``state_indices``; ``out`` ``[1, B, H, 128]``.
    """
    jit.load_ops().fused_kda_decode(
        x,
        weight,
        bias,
        conv_state,
        raw_g,
        raw_beta,
        A_log.reshape(-1),
        dt_bias,
        state_indices,
        state,
        out,
        lower_bound,
        output_gate,
        norm_weight,
        norm_eps,
    )
