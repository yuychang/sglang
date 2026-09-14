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
"""Kimi-K3 ROCm weight preparation for Quark checkpoints.

Quark serializes K3 as per-output-channel FP8, which is not the layout the
AITER kernels want. The conversions live here; ``kimi_k3.py`` calls every entry
point behind ``_is_hip`` and keeps the checkpoint's own layout otherwise.
"""

import torch
from torch import nn

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import (
    channel_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
)

_is_fp8_fnuz = is_fp8_fnuz()


def _k3_channel_fp8_to_tensor_fp8(
    module: nn.Module, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize a per-output-channel FP8 weight to per-tensor, returning
    (weight, scalar scale).

    The aiter batched absorb GEMM dereferences w_scale as a single scalar, so a
    per-channel vector reaching it applies channel 0's scale to every channel.
    Must run while dim 0 is still the channel axis weight_scale indexes, i.e.
    before the kv_b_proj head split."""
    weight_scale = module.weight_scale
    if _is_fp8_fnuz:
        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
            weight=weight, weight_scale=weight_scale, input_scale=None
        )
    # Per-channel scale is 1D [out]; reshape so it broadcasts against [out, in].
    if weight_scale.dim() == 1:
        weight_scale = weight_scale.view(-1, 1)
    return channel_quant_to_tensor_quant(weight, weight_scale)
