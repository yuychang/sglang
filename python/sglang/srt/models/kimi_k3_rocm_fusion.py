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
"""Kimi-K3 ROCm producer-quant fusion.

A "producer fusion" folds the activation quant a linear would run into the
kernel that produces its input (RMSNorm, output gate), so decode issues one
launch instead of two and hands the linear ``(fp8, scale)`` directly.

Every gate here is ``_is_hip``-derived: where the AITER kernels are absent
``_k3_ptpc_fp8`` is False and each entry point degrades to the split path.
"""

import logging
from typing import Optional

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.utils_rocm import (
    accepts_group128_fp8_tuple,
    accepts_ptpc_fp8_tuple,
)
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_use_aiter = envs.SGLANG_USE_AITER.get() and _is_hip
_k3_ptpc_fp8 = _is_hip and envs.SGLANG_ROCM_K3_PTPC_FP8.get()
_k3_ptpc_fp8_max_tokens = envs.SGLANG_ROCM_K3_PTPC_FP8_MAX_TOKENS.get()
_k3_ptpc_fp8_min_tokens = envs.SGLANG_ROCM_K3_PTPC_FP8_MIN_TOKENS.get()
_k3_ptpc_fp8_prefill_min_tokens = envs.SGLANG_ROCM_K3_PTPC_FP8_PREFILL_MIN_TOKENS.get()
_k3_ptpc_fp8_shared_down = _is_hip and envs.SGLANG_ROCM_K3_PTPC_FP8_SHARED_DOWN.get()

_logged_fusions: set[str] = set()


def _k3_log_once(key: str, msg: str, *args) -> None:
    """Report an enabled fusion once per process, not once per layer."""
    if key not in _logged_fusions:
        _logged_fusions.add(key)
        logger.info(msg, *args)


def _k3_ptpc_fp8_batch_ok(num_tokens: int) -> bool:
    return _k3_ptpc_fp8_min_tokens <= num_tokens <= _k3_ptpc_fp8_max_tokens


def _k3_ptpc_fp8_moe_gemm_ok(num_tokens: int) -> bool:
    """Batch gate for the latent-up and shared-down PTPC GEMMs, which also win
    again at prefill-chunk sizes."""
    return _k3_ptpc_fp8_batch_ok(num_tokens) or (
        0 < _k3_ptpc_fp8_prefill_min_tokens <= num_tokens
    )


def _k3_linear_accepts_ptpc_tuple(module: nn.Module) -> bool:
    """Whether a linear consumes ``(fp8, per_token_scale)`` without re-quant.

    Quark W8A8-FP8 (K3 AttnFP8 o_proj) uses ``per_token`` + ``per_channel``
    weights, not compressed-tensors ``strategy``. Both land in
    ``apply_fp8_linear`` / ``apply_fp8_ptpc_linear`` on ROCm.
    """
    return _k3_ptpc_fp8 and accepts_ptpc_fp8_tuple(module)


def _k3_attn_inproj(self_attn: nn.Module) -> Optional[nn.Module]:
    return (
        getattr(self_attn, "fused_qkvg_proj", None)
        or getattr(self_attn, "fused_qkv_a_proj_with_mqa", None)
        or getattr(self_attn, "q_a_proj", None)
        or getattr(self_attn, "qkv_proj", None)
    )


def _k3_split_linear_accepts_fp8_tuple(module: nn.Module) -> bool:
    return accepts_group128_fp8_tuple(module) or _k3_linear_accepts_ptpc_tuple(module)


def _k3_inproj_needs_bf16_hidden(self_attn: nn.Module, num_tokens: int) -> bool:
    """True when a BF16 copy of the RMSNormed hidden is still consumed.

    Merged KDA ``qkvgbfa`` decode is a single GEMM, so the producer can emit
    only the FP8 tuple. The split ``[q,k,v,g]`` + ``f_a`` / ``b`` path still
    reads BF16 when those linears cannot consume the tuple. Quark dequantizes
    a per-channel FP8 weight whose N is not a multiple of 16 (KDA ``b_proj``
    is 12-wide at TP8) and clears ``weight_scale``; ``apply_weights`` then
    rejects ``(fp8, scale)``.
    """
    sizes = getattr(self_attn, "_qkvgbfa_sizes", None)
    limit = getattr(self_attn, "_qkvgbfa_bs_limit", 0) or 0
    if sizes is not None and 0 < num_tokens <= limit:
        return False
    if getattr(self_attn, "_bfa_w", None) is not None:
        return True
    # Non-full-rank KDA folds beta into fused_qkvbfg_a_proj. MLA has neither
    # projection. Only the full-rank split path calls these on hidden_states.
    if not getattr(self_attn, "use_full_rank_gate", False):
        return False
    for name in ("b_proj", "f_a_proj"):
        module = getattr(self_attn, name, None)
        if module is None:
            continue
        if not _k3_split_linear_accepts_fp8_tuple(module):
            return True
    return False


def _k3_should_fuse_inproj_quant(
    *, self_attn: nn.Module, num_tokens: int, inproj: Optional[nn.Module]
) -> bool:
    if (
        not _use_aiter
        or inproj is None
        or not _k3_ptpc_fp8_batch_ok(num_tokens)
        or _k3_inproj_needs_bf16_hidden(self_attn, num_tokens)
    ):
        return False
    return accepts_group128_fp8_tuple(inproj) or _k3_linear_accepts_ptpc_tuple(inproj)


def _k3_hidden_tensor(hidden_states):
    """The activation itself; a fired producer fusion makes it ``(fp8, scale)``."""
    return hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states


def _k3_hidden_num_tokens(hidden_states) -> int:
    return _k3_hidden_tensor(hidden_states).shape[0]


def _k3_hidden_rows(hidden_states, num_rows: int):
    """Leading ``num_rows`` tokens, preserving the tuple form."""
    if isinstance(hidden_states, tuple):
        return tuple(t[:num_rows] if torch.is_tensor(t) else t for t in hidden_states)
    return hidden_states[:num_rows]


def _k3_fuse_kda_o_norm_ptpc(
    core_attn_out: torch.Tensor,
    *,
    norm_gate: torch.Tensor,
    o_norm: nn.Module,
    o_proj: nn.Module,
):
    """Fuse KDA's gated output norm with the PTPC quant ``o_proj`` expects.

    Returns the ``(fp8, per-token scale)`` tuple, or None when the split
    bf16 norm plus standalone quant has to run.
    """
    token_count = core_attn_out.shape[-3]
    if not (
        _use_aiter
        and _k3_ptpc_fp8_batch_ok(token_count)
        and _k3_linear_accepts_ptpc_tuple(o_proj)
    ):
        return None
    # The split path materializes gated RMSNorm in bf16 and then launches
    # dynamic per-token quant for o_proj. AITER's sigmoid variant preserves
    # that bf16 rounding boundary and emits the tuple consumed directly by
    # apply_fp8_ptpc_linear.
    from aiter import dtypes as aiter_dtypes
    from aiter.ops.gated_rmsnorm_fp8_per_token_quant import (
        gated_rmsnorm_fp8_per_token_quant,
    )

    norm_input = core_attn_out.squeeze(0)
    quant_out = torch.empty(
        (token_count, norm_input.shape[-2] * norm_input.shape[-1]),
        dtype=aiter_dtypes.fp8,
        device=norm_input.device,
    )
    quant_scale = torch.empty(
        (token_count, 1), dtype=torch.float32, device=norm_input.device
    )
    gated_rmsnorm_fp8_per_token_quant(
        quant_out,
        quant_scale,
        norm_input,
        norm_gate,
        o_norm.weight,
        o_norm.eps,
        use_sigmoid=True,
        round_to_input_dtype=True,
    )
    _k3_log_once(
        "kda_onorm_ptpc",
        "KDA o_norm+PTPC producer fusion enabled (scheme=%s, tokens=%d)",
        type(o_proj.scheme).__name__,
        token_count,
    )
    return quant_out, quant_scale


def _k3_fuse_mla_gate_ptpc(x: torch.Tensor, *, gate: torch.Tensor, o_proj: nn.Module):
    """Fuse the MLA output gate with the PTPC quant ``o_proj`` expects.

    Returns the ``(fp8, per-token scale)`` tuple, or None when the gate has
    to be applied on its own.
    """
    if not _k3_linear_accepts_ptpc_tuple(o_proj):
        return None
    from sglang.kernels.ops.attention import mla_output_gate_fp8_quant_hip

    if not mla_output_gate_fp8_quant_hip.covered(x, gate):
        return None
    _k3_log_once("mla_gate_ptpc", "K3 MLA o_proj gate+PTPC producer fusion enabled")
    return mla_output_gate_fp8_quant_hip.kimi_k3_mla_output_gate_fp8_quant(x, gate)


def _attn_res_can_skip_out_norm(hidden_size: int, nvb: int) -> bool:
    """Whether ``AttnResidual.forward`` can take ``out_norm=None``.

    Only the ROCm ``_agg_kernel`` runs without an out-norm, so the fused path
    applies only where attn_residual picks that kernel (or nvb == 0).
    """
    from sglang.srt.layers.attn_residual import _use_hip_fused

    return nvb == 0 or _use_hip_fused(hidden_size, nvb)


def attn_bank_read_quant_format(
    self_attn: nn.Module,
    *,
    hidden_size: int,
    num_valid_blocks: int,
    num_tokens: int,
    forward_mode: ForwardMode,
) -> str:
    """The ``quant_format`` the attention-residual bank read hands the in-proj
    in: decode fuses the input RMSNorm with the activation quant the in-proj
    expects; "" keeps the bf16 read."""
    inproj = _k3_attn_inproj(self_attn)
    if not (
        forward_mode.is_decode()
        and _attn_res_can_skip_out_norm(hidden_size, num_valid_blocks)
        and _k3_should_fuse_inproj_quant(
            self_attn=self_attn, num_tokens=num_tokens, inproj=inproj
        )
    ):
        return ""
    if accepts_group128_fp8_tuple(inproj):
        _k3_log_once(
            "inproj_group128", "K3 in_proj RMS+group128 producer fusion enabled"
        )
        return "fp8"
    _k3_log_once(
        "inproj_ptpc",
        "K3 in_proj RMS+PTPC producer fusion enabled (scheme=%s)",
        type(inproj.scheme).__name__,
    )
    return "fp8_per_token"
