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
"""Kimi-K3 ROCm FP8 MoE preroute.

For 1-4 decode tokens, one FlyDSL kernel computes the routed down projection,
the shared gate_up and the router logits from FP8 row-scaled weight copies.
``kimi_k3.py`` calls every entry point behind ``_is_hip``.
"""

from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

from sglang.srt.layers.activation import SituAndMul

_FP8_MAX = 448.0
_MAX_TOKENS = 4


def quantize_fp8_rows(
    weight: torch.Tensor, pow2_scale: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [out, in] weight to FP8 with one FP32 scale per row.

    pow2_scale rounds the scales up to powers of two, which keeps weights
    dequantized from MXFP4 exact on the E4M3 grid.
    """
    weight = weight.float()
    scale = (weight.abs().amax(dim=1) / _FP8_MAX).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    if pow2_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (weight / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def k3_prepare_preroute_fp8(mlp: nn.Module) -> None:
    """Build the FP8 weight copies the preroute kernels read."""
    from sglang.kernels.ops.kimi_k3 import moe_preroute_aiter_hip as ops

    mlp._k3_preroute = None
    shared = mlp.shared_experts
    if not ops.enabled() or not mlp._eligible_for_fused_front:
        return
    act = shared.act_fn
    if not isinstance(act, SituAndMul) or act.linear_beta is None:
        return
    routed = mlp.routed_expert_down_proj.weight
    gate_up = shared.gate_up_proj.weight
    down = shared.down_proj.weight
    if (
        tuple(routed.shape) != (3584, 7168)
        or tuple(gate_up.shape) != (1536, 7168)
        or tuple(down.shape) != (7168, 768)
        or tuple(mlp.gate.weight.shape) != (896, 7168)
        or routed.dtype != torch.bfloat16
    ):
        return

    def is_mxfp4(linear: nn.Module) -> bool:
        return getattr(linear, "dequantized_bf16", False)

    p = SimpleNamespace(beta=act.beta, linear_beta=act.linear_beta)
    p.routed_w, p.routed_s = quantize_fp8_rows(routed)
    p.shared_w, p.shared_s = quantize_fp8_rows(
        gate_up, pow2_scale=is_mxfp4(shared.gate_up_proj)
    )
    p.down_w, p.down_s = quantize_fp8_rows(down, pow2_scale=is_mxfp4(shared.down_proj))
    p.inter_w = p.inter_s = None
    if ops.cooperative_preactivated_enabled():
        # The 2-4 token kernel wants gate and up rows interleaved.
        p.inter_w = p.shared_w.view(2, 768, 7168).transpose(0, 1).reshape(1536, 7168)
        p.inter_s = p.shared_s.view(2, 768).t().reshape(1536)
    ops.warmup(
        p.routed_w,
        p.routed_s,
        p.shared_w,
        p.shared_s,
        mlp.gate.weight,
        p.down_w,
        p.down_s,
        p.inter_w,
        p.inter_s,
        situ_beta=p.beta,
        situ_linear_beta=p.linear_beta,
    )
    mlp._k3_preroute = p


def k3_run_preroute(
    mlp: nn.Module, hidden_states: torch.Tensor
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
    """Return (gate_up, router_logits, routed_input, preactivated), or None.

    When preactivated, gate_up already holds the SiTU output.
    """
    p = getattr(mlp, "_k3_preroute", None)
    if p is None or hidden_states.shape[0] > _MAX_TOKENS:
        return None
    from sglang.kernels.ops.kimi_k3 import moe_preroute_aiter_hip as ops

    router_w = mlp.gate.weight
    if p.inter_w is not None and ops.cooperative_preactivated_tri_covered(
        hidden_states, p.routed_w, p.routed_s, p.inter_w, p.inter_s, router_w
    ):
        routed_input, gate_up, logits = ops.run_tri_cooperative_preactivated(
            hidden_states,
            p.routed_w,
            p.routed_s,
            p.inter_w,
            p.inter_s,
            router_w,
            situ_beta=p.beta,
            situ_linear_beta=p.linear_beta,
        )
        return gate_up, logits, routed_input, True
    if ops.tri_covered(
        hidden_states, p.routed_w, p.routed_s, p.shared_w, p.shared_s, router_w
    ):
        routed_input, gate_up, logits = ops.run_tri(
            hidden_states, p.routed_w, p.routed_s, p.shared_w, p.shared_s, router_w
        )
        return gate_up, logits, routed_input, False
    return None


def k3_run_preroute_shared_down(
    mlp: nn.Module, gate_up: torch.Tensor, shared_output: torch.Tensor
) -> bool:
    """Run SiTU and the FP8 shared down into shared_output if covered."""
    p = getattr(mlp, "_k3_preroute", None)
    if p is None:
        return False
    from sglang.kernels.ops.kimi_k3 import moe_preroute_aiter_hip as ops

    if not ops.shared_down_covered(gate_up, p.down_w, p.down_s):
        return False
    ops.run_shared_down(
        gate_up,
        p.down_w,
        p.down_s,
        situ_beta=p.beta,
        situ_linear_beta=p.linear_beta,
        out=shared_output,
    )
    return True
