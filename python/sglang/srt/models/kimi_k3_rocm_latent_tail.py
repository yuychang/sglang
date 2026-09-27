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
"""Kimi-K3 ROCm FP8 latent MoE tail.

For 1/2/4 decode tokens, one FlyDSL kernel runs the latent RMSNorm, the FP8
routed up-projection, the shared-expert add and the residual-prefix add.
``kimi_k3.py`` calls these behind ``_is_hip``.
"""

from typing import Optional

import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_TOKENS = (1, 2, 4)


def k3_prepare_latent_tail_fp8(mlp: nn.Module) -> None:
    """Pack the routed up-projection to row-scaled FP8."""
    from sglang.kernels.ops.kimi_k3 import latent_tail_aiter_hip as ops

    mlp._k3_latent_tail = None
    up = mlp.routed_expert_up_proj
    if not ops.enabled() or not mlp.fuse_ar_norm or up is None:
        return
    weight = getattr(up, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (7168, 3584)
    ):
        return
    up_w, up_s = ops.pack(weight)
    norm_weight, epsilon = mlp._get_fused_norm_params()
    ops.warmup(norm_weight, up_w, up_s, epsilon)
    mlp._k3_latent_tail = (up_w, up_s)


def k3_latent_tail_eligible(
    mlp: nn.Module, num_tokens: int, forward_batch: Optional[ForwardBatch]
) -> bool:
    return (
        getattr(mlp, "_k3_latent_tail", None) is not None
        and num_tokens in _TOKENS
        and forward_batch is not None
        and forward_batch.forward_mode.is_decode_or_idle()
    )


def k3_run_latent_tail(
    mlp: nn.Module,
    latent: torch.Tensor,
    shared_output: torch.Tensor,
    prefix_sum: Optional[torch.Tensor],
    forward_batch: Optional[ForwardBatch],
    skip_rms: bool,
) -> Optional[torch.Tensor]:
    """Return out + shared_output (+ prefix_sum), or None if not covered."""
    if not k3_latent_tail_eligible(mlp, latent.shape[0], forward_batch):
        return None
    from sglang.kernels.ops.kimi_k3 import latent_tail_aiter_hip as ops

    up_w, up_s = mlp._k3_latent_tail
    norm_weight, epsilon = mlp._get_fused_norm_params()
    if not ops.covered(
        latent, shared_output, norm_weight, up_w, up_s, epsilon, prefix_sum
    ):
        return None
    return ops.run(
        latent,
        shared_output,
        norm_weight,
        up_w,
        up_s,
        epsilon,
        prefix_sum,
        skip_rms=skip_rms,
    )
