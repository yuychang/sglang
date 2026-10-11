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
"""ROCm KDA input-projection and fused-decode setup for Kimi-K3.

Shared ``kimi_k3.py`` keeps an ``_is_hip`` call. These functions still
take the attention module as ``self``.
"""

from types import SimpleNamespace

import torch

from sglang.srt.environ import envs


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
    from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable

    if not envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.get():
        return False
    if not (self._attn_tp_is_full_tp and self.use_full_rank_gate):
        return False
    weights = [
        module.weight for module in (self.fused_qkvg_proj, self.f_a_proj, self.b_proj)
    ]
    if not all(
        type(weight.data) is torch.Tensor and weight.dim() == 2 for weight in weights
    ):
        return False
    # Whitelist the dtype rather than only require the three to agree: the
    # merged buffer carries only .weight, so quantized weights that happen to
    # match each other still lose their per-channel scales.
    if not _is_unquantized_mergeable(weights):
        return False
    return len({(weight.dtype, weight.shape[1]) for weight in weights}) == 1


def prepare_fused_decode_hip(self) -> None:
    """Stage the fused HIP decode (conv + recurrence + gated norm) inputs."""
    from sglang.srt.layers.attention.linear.kda_rocm import stage_fused_decode

    if stage_fused_decode(self.attn, self.o_norm.weight.data, self.o_norm.eps):
        self._kda_fused_decode_ready = True


def init_kda_rocm_state(self) -> None:
    """ROCm-only buffers, filled by prepare_kda_rocm after weights load."""
    # The in-proj merges only need full-TP sharding; do_fuse_qkvbfg also
    # requires quant_config is None, which excludes the Quark checkpoints.
    self._attn_tp_is_full_tp = self.attn_tp_size == self.tp_size
    self._qkvgbfa_fp8_w = None
    self._qkvgbfa_fp8_s = None
    self._qkvgbfa_fp8_n = 0


def merge_bfa_weights_hip(self) -> bool:
    """Merge the KDA in-proj into one ROCm GEMM; False leaves the plain merge."""
    from sglang.srt.models.kimi_k3_rocm_quant import _k3_merge_kda_inproj_fp8

    if not self._bfa_uses_block_fp8 and _k3_merge_kda_inproj_fp8(self):
        return True
    if merge_kda_inproj_weights_hip(self):
        # The split-path f_b GEMM still uses this above the token threshold.
        self._bfa_f_b_w = self.f_b_proj.weight
        return True
    return False


def prepare_kda_rocm(self) -> None:
    prepare_qkvgbfa_ptpc_fp8(self)


def try_inproj_hip(self, hidden_states):
    """(qkv, beta, forget_gate, g) from a fused ROCm in-proj, or None."""
    from sglang.srt.models.kimi_k3_rocm_fusion import _k3_hidden_num_tokens
    from sglang.srt.models.kimi_k3_rocm_quant import (
        _k3_apply_f_b,
        _k3_qkvgbfa_inproj,
    )

    token_count = _k3_hidden_num_tokens(hidden_states)
    if self._qkvgbfa_sizes is None or not (0 < token_count <= self._qkvgbfa_bs_limit):
        return None
    # One GEMM for the whole in-proj: the [f_a|b] tail rides the wide
    # projection's bandwidth.
    fused_states = _k3_qkvgbfa_inproj(self, hidden_states)
    if fused_states is None:
        return None
    qkv, g_proj_states, f_a, beta, _padding = torch.split(
        fused_states, self._qkvgbfa_sizes, dim=-1
    )
    return qkv, beta, _k3_apply_f_b(self, f_a), g_proj_states


def try_o_norm_quant(self, core_attn_out, g_proj_states):
    """(fp8, per-token scale) for o_proj from a fused o_norm+quant, or None."""
    from sglang.srt.models.kimi_k3_rocm_fusion import _k3_fuse_kda_o_norm_ptpc

    return _k3_fuse_kda_o_norm_ptpc(
        core_attn_out,
        norm_gate=g_proj_states.unflatten(-1, (-1, self.head_dim)),
        o_norm=self.o_norm,
        o_proj=self.o_proj,
    )
