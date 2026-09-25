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
"""Kimi-K3 ROCm MoE front and latent GEMM dispatch.

The merged BF16 MoE front runs through AITER tuned_gemm inside its tuned token
window. Large batches run the latent down/up projections as AITER MXFP4 GEMMs
and decode batches run the latent up-projection as PTPC FP8; the BF16 weights
stay live as the fallback. ``kimi_k3.py`` calls these behind
``_is_hip``.
"""

from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

from sglang.srt.environ import envs

_FRONT_SHAPE = (6016, 7168)


def k3_router_bias_dtype(mlp: nn.Module) -> torch.dtype:
    """AITER routes on gate-logit dtype bias; cast it once at load."""
    if envs.SGLANG_USE_AITER.get():
        return mlp.gate.weight.dtype
    return torch.float32


def k3_tuned_front_gemm(
    x: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """Merged BF16 front through AITER tuned_gemm, or None if not covered."""
    if not (
        envs.SGLANG_USE_AITER.get()
        and envs.SGLANG_ROCM_K3_AITER_TUNED_MOE_FRONT.get()
        and envs.SGLANG_ROCM_K3_AITER_TUNED_MOE_FRONT_MIN_TOKENS.get()
        <= x.shape[0]
        <= envs.SGLANG_ROCM_K3_AITER_TUNED_MOE_FRONT_MAX_TOKENS.get()
        and tuple(weight.shape) == _FRONT_SHAPE
        and x.dtype == weight.dtype == torch.bfloat16
    ):
        return None
    from aiter.tuned_gemm import tgemm

    return tgemm.mm(x, weight, None, otype=x.dtype)


def k3_prepare_moe_latent_mxfp4(mlp: nn.Module) -> None:
    """Pack MXFP4 copies of the latent down/up projections."""
    mlp._k3_latent_mxfp4 = None
    if (
        not envs.SGLANG_ROCM_K3_MOE_LATENT_MXFP4.get()
        or not mlp._eligible_for_fused_front
        or len(mlp._front_sizes) != 3
    ):
        return
    from sglang.kernels.ops.kimi_k3 import latent_mxfp4_aiter_hip as ops

    if not ops.supported():
        return
    head_rows = mlp._front_sizes[0] + mlp._front_sizes[1]
    down = mlp._front_w[head_rows:]
    up = mlp.routed_expert_up_proj.weight
    if (
        tuple(down.shape) != (3584, 7168)
        or tuple(up.shape) != (7168, 3584)
        or down.dtype != torch.bfloat16
        or up.dtype != torch.bfloat16
    ):
        return
    p = SimpleNamespace(
        head=mlp._front_w[:head_rows],
        min_tokens=envs.SGLANG_ROCM_K3_MOE_LATENT_MXFP4_MIN_TOKENS.get(),
    )
    p.down_w, p.down_s = ops.pack(down, "latent down_proj")
    p.up_w, p.up_s = ops.pack(up, "latent up_proj")
    mlp._k3_latent_mxfp4 = p


def k3_use_latent_mxfp4(mlp: nn.Module, num_tokens: int) -> bool:
    p = getattr(mlp, "_k3_latent_mxfp4", None)
    return p is not None and num_tokens >= p.min_tokens


def k3_run_front_mxfp4(
    mlp: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (gate_up, router_logits, routed_input) with an MXFP4 down."""
    from sglang.kernels.ops.kimi_k3 import latent_mxfp4_aiter_hip as ops

    p = mlp._k3_latent_mxfp4
    head = torch.nn.functional.linear(hidden_states, p.head)
    gate_up, router_logits = torch.split(head, mlp._front_sizes[:2], dim=-1)
    routed_input = ops.run(hidden_states, p.down_w, p.down_s)
    return gate_up, router_logits, routed_input


def k3_run_latent_up_mxfp4(mlp: nn.Module, latent: torch.Tensor) -> torch.Tensor:
    from sglang.kernels.ops.kimi_k3 import latent_mxfp4_aiter_hip as ops

    p = mlp._k3_latent_mxfp4
    return ops.run(latent, p.up_w, p.up_s)


def k3_prepare_latent_up_ptpc_fp8(mlp: nn.Module) -> None:
    """Pack a PTPC FP8 copy of the latent up-projection."""
    mlp._k3_latent_up_ptpc = None
    up = getattr(mlp, "routed_expert_up_proj", None)
    if not envs.SGLANG_ROCM_K3_PTPC_FP8.get() or up is None:
        return
    from sglang.kernels.ops.kimi_k3 import ptpc_fp8_aiter_hip as ops

    weight = up.weight
    if (
        not ops.available()
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
    ):
        return
    w, s, n = ops.pack(weight.contiguous())
    ops.warmup(w, s, n, weight.shape[1])
    mlp._k3_latent_up_ptpc = (w, s, n)


def k3_run_latent_up_ptpc_fp8(
    mlp: nn.Module, latent: torch.Tensor
) -> Optional[torch.Tensor]:
    """Latent up-projection in PTPC FP8, or None if not covered."""
    packed = getattr(mlp, "_k3_latent_up_ptpc", None)
    if packed is None or not (
        envs.SGLANG_ROCM_K3_PTPC_FP8_MIN_TOKENS.get()
        <= latent.shape[0]
        <= envs.SGLANG_ROCM_K3_PTPC_FP8_MAX_TOKENS.get()
    ):
        return None
    from sglang.kernels.ops.kimi_k3 import ptpc_fp8_aiter_hip as ops

    if not ops.covered(latent, packed[0]):
        return None
    return ops.run(latent, *packed)
