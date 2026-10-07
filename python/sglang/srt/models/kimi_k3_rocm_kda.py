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
"""ROCm KDA input-projection and fused-decode paths for Kimi-K3.

Shared ``kimi_k3.py`` keeps an ``_is_hip`` call. These functions still
take the attention module as ``self``.
"""

from types import SimpleNamespace

import torch

def merge_kda_inproj_weights_hip(self) -> bool:
    """Merge the ROCm KDA input projections while retaining split views."""
    from sglang.srt.models.kimi_k3 import (
        _merge_weights_as_views,
    )
    if not may_fuse_kda_inproj(self):
        return False

    merged, sizes = _merge_weights_as_views(
        [self.fused_qkvg_proj, self.f_a_proj, self.b_proj], pad_rows_to=8
    )
    self._bfa_fa_size, self._bfa_b_size = sizes[-2:]
    self._bfa_w = merged[sizes[0] :]
    # Deliberately not an nn.Module: this is only a layer-shaped carrier for
    # the linear method and must not duplicate `merged` in state_dict.
    self._qkvgbfa_layer = SimpleNamespace(weight=merged)
    self._qkvgbfa_sizes = [
        *self.split_sizes,
        self._bfa_fa_size,
        self._bfa_b_size,
        merged.shape[0] - sum(sizes),
    ]
    return True


def use_qkvgbfa_ptpc_fp8(self, hidden_states) -> bool:
    from sglang.srt.models.kimi_k3_rocm_fusion import (
        _k3_hidden_tensor,
        _k3_ptpc_fp8_batch_ok,
    )
    x = _k3_hidden_tensor(hidden_states)
    if getattr(self, "_qkvgbfa_fp8_w", None) is None or not _k3_ptpc_fp8_batch_ok(
        x.shape[0]
    ):
        return False
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    if isinstance(hidden_states, tuple):
        return ptpc_fp8_aiter_hip.covered_prequant(
            x, hidden_states[1], self._qkvgbfa_fp8_w
        )
    return ptpc_fp8_aiter_hip.covered(x, self._qkvgbfa_fp8_w)


def prepare_qkvgbfa_ptpc_fp8(self) -> None:
    """Quantize the merged KDA input projection for PTPC FP8 decode."""
    from sglang.srt.models.kimi_k3_rocm_fusion import (
        _k3_ptpc_fp8,
    )
    layer = getattr(self, "_qkvgbfa_layer", None)
    if not _k3_ptpc_fp8 or layer is None:
        return
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    weight = layer.weight
    if (
        not ptpc_fp8_aiter_hip.available()
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
    ):
        return
    out_features, in_features = weight.shape
    (
        self._qkvgbfa_fp8_w,
        self._qkvgbfa_fp8_s,
        self._qkvgbfa_fp8_n,
    ) = ptpc_fp8_aiter_hip.pack(weight.contiguous())
    ptpc_fp8_aiter_hip.warmup(
        self._qkvgbfa_fp8_w,
        self._qkvgbfa_fp8_s,
        self._qkvgbfa_fp8_n,
        in_features,
    )


def may_fuse_kda_inproj(self) -> bool:
    """Return whether the KDA weights can safely share one ROCm GEMM."""
    from sglang.srt.models.kimi_k3 import (
        _is_hip,
        _is_unquantized_mergeable,
    )
    from sglang.srt.environ import envs
    if not (_is_hip and envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.get()):
        return False
    if not (self._attn_tp_is_full_tp and self.use_full_rank_gate):
        return False
    weights = [
        module.weight
        for module in (self.fused_qkvg_proj, self.f_a_proj, self.b_proj)
    ]
    if not all(
        type(weight.data) is torch.Tensor and weight.dim() == 2
        for weight in weights
    ):
        return False
    # Whitelist the dtype rather than only require the three to agree: the
    # merged buffer carries only .weight, so quantized weights that happen to
    # match each other still lose their per-channel scales.
    if not _is_unquantized_mergeable(weights):
        return False
    return len({(weight.dtype, weight.shape[1]) for weight in weights}) == 1


def prepare_group64_projection(self) -> None:
    from sglang.srt.models.kimi_k3 import (
        _aiter_kda_group64,
        _is_unquantized_mergeable,
    )
    if (
        not _aiter_kda_group64
        or not self._attn_tp_is_full_tp
        or not self.use_full_rank_gate
    ):
        return
    srcs = [self.fused_qkvg_proj.weight, self.b_proj.weight, self.f_a_proj.weight]
    # The shape check below passes for FP8 too, so pack() would reinterpret
    # quantized bytes as bf16 and drop the per-channel scales.
    if not _is_unquantized_mergeable(srcs):
        return
    from sglang.kernels.ops.gemm import kda_group64_aiter_hip

    merged = torch.cat(
        [*srcs, self.f_a_proj.weight.new_zeros((4, self.hidden_size))],
        dim=0,
    ).contiguous()
    if tuple(merged.shape) != (6288, 7168):
        return
    weight, scale = kda_group64_aiter_hip.pack(merged)
    self._kda_group64_weight = weight
    self._kda_group64_scale = scale
    kda_group64_aiter_hip.warmup(weight, scale)


def prepare_fused_decode_hip(self) -> None:
    from sglang.srt.environ import envs
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip

    layer = self.attn
    w = layer.conv_weights
    f_b_weight = self.f_b_proj.weight
    # Quark ships f_b as PTPC FP8; _merge_bfa_weights already
    # dequantized it into the BF16 tiny-GEMM buffer.
    f_b_dense = getattr(self, "_bfa_f_b_w", None)
    if f_b_weight.dtype != torch.bfloat16 and f_b_dense is not None:
        f_b_weight = f_b_dense
    backend = envs.SGLANG_ROCM_K3_KDA_FUSED_BACKEND.get().lower()
    backend_available = (
        backend == "aiter"
        and kda_fused_decode_aiter_hip.available(f_b_weight.device)
    )
    if (
        backend_available
        and w is not None
        and tuple(w.shape) == (3 * 12 * 128, 4)
        and w.dtype == torch.float32
        and f_b_weight.shape == (12 * 128, 128)
        and f_b_weight.dtype == torch.bfloat16
        and layer.A_log is not None
        and layer.A_log.numel() == 12
        and layer.A_log.dtype == torch.float32
        and layer.dt_bias is not None
        and tuple(layer.dt_bias.shape) == (12 * 128,)
        and layer.dt_bias.dtype == torch.float32
        and layer.lower_bound is not None
    ):
        norm_weight = self.o_norm.weight.data.to(torch.bfloat16).contiguous()
        f_b_weight = f_b_weight.view(12, 128, 128).contiguous()
        a_log = layer.A_log.detach().reshape(-1).contiguous()
        layer._k3_hip_fused_decode_args = (
            f_b_weight,
            norm_weight,
            float(self.o_norm.eps),
            a_log,
        )
        kda_fused_decode_aiter_hip.warmup(
            f_b_weight=f_b_weight,
            conv_weight=w,
            A_log=a_log,
            dt_bias=layer.dt_bias,
            lower_bound=float(layer.lower_bound),
            norm_weight=norm_weight,
            norm_eps=float(self.o_norm.eps),
        )
        layer._k3_hip_fused_decode_backend = backend
        self._kda_hip_fused_decode_ready = True


def try_group64_qkv(self, hidden_states: torch.Tensor):
    if self._kda_group64_weight is None or self._kda_group64_scale is None:
        return None
    from sglang.kernels.ops.gemm import kda_group64_aiter_hip

    if not kda_group64_aiter_hip.covered(
        hidden_states,
        self._kda_group64_weight,
        self._kda_group64_scale,
    ):
        return None
    packed = kda_group64_aiter_hip.run(
        hidden_states,
        self._kda_group64_weight,
        self._kda_group64_scale,
    )
    mixed_qkv, g_proj_states, beta, f_a, _padding = torch.split(
        packed,
        [self.split_sizes[0], self.split_sizes[1], 12, 128, 4],
        dim=-1,
    )
    return mixed_qkv, beta, f_a, g_proj_states
