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
"""Kimi-K3 ROCm all-reduce of the fused-front [latent | shared] pair.

``kimi_k3.py`` calls this behind ``_is_hip``.
"""

from typing import Optional

import torch
from torch import nn

from sglang.srt.layers import k3_ar_fusion
from sglang.srt.layers.k3_fused_ar_rmsnorm import try_fused_ar_rmsnorm
from sglang.srt.layers.k3_moe_pair_ar import all_reduce_moe_latent_shared
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.kimi_k3_rocm_latent_tail import k3_latent_tail_eligible


def k3_all_reduce_moe_pair(
    mlp: nn.Module,
    buf: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    forward_batch: Optional[ForwardBatch],
) -> tuple[tuple[torch.Tensor, torch.Tensor], bool]:
    """Return ((latent, shared), latent_normed).

    Folds the latent RMSNorm into the AR when the fused kernel wins; the FP8
    latent tail has its own norm, so its token counts keep the plain AR.
    """
    if mlp.fuse_ar_norm and not k3_latent_tail_eligible(mlp, num_tokens, forward_batch):
        weight, eps = mlp._get_fused_norm_params()
        view = buf.view(-1, k3_ar_fusion.NORM_DIM)
        fused = try_fused_ar_rmsnorm(view, weight, eps, num_norm_rows=num_tokens)
        if fused is not None:
            normed, reduced = fused
            if reduced.data_ptr() != view.data_ptr():
                buf.copy_(reduced.reshape(-1))
            shared = buf[num_tokens * mlp.moe_hidden_size :].view(
                num_tokens, hidden_size
            )
            return (normed[:num_tokens], shared), True
    pair = all_reduce_moe_latent_shared(
        buf,
        num_tokens=num_tokens,
        moe_hidden_size=mlp.moe_hidden_size,
        hidden_size=hidden_size,
    )
    return pair, False
