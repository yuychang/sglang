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
"""ROCm latent-MoE, preroute, and PTPC paths for Kimi-K3.

Shared ``kimi_k3.py`` keeps an ``_is_hip`` call. These functions still
take the MoE module as ``self``.
"""

import torch


def prepare_moe_latent_mxfp4(self) -> None:
    """Pack non-EP latent projections for the large-M MXFP4 path."""
    from sglang.srt.models.kimi_k3 import (
        _moe_latent_mxfp4,
    )

    if (
        not _moe_latent_mxfp4
        or not self.use_latent_moe
        or not (
            self._eligible_for_fused_front or self._eligible_for_partial_fused_front
        )
        or self._front_sizes is None
        or len(self._front_sizes) not in (2, 3)
    ):
        return
    from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip

    if not latent_mxfp4_aiter_hip.supported():
        return
    head_rows = sum(self._front_sizes[:-1])
    front_head = self._front_w[:head_rows]
    down = self._front_w[head_rows:]
    up = self.routed_expert_up_proj.weight
    if (
        tuple(down.shape) != (3584, 7168)
        or tuple(up.shape) != (7168, 3584)
        or down.dtype != torch.bfloat16
        or up.dtype != torch.bfloat16
    ):
        return
    self._front_head = front_head
    self._front_down_w4, self._front_down_scale4 = latent_mxfp4_aiter_hip.pack(
        down, "latent down_proj"
    )
    self._latent_up_w4, self._latent_up_scale4 = latent_mxfp4_aiter_hip.pack(
        up, "latent up_proj"
    )


def use_moe_latent_mxfp4(self, num_tokens: int) -> bool:
    from sglang.srt.models.kimi_k3 import (
        _moe_latent_mxfp4_min_tokens,
    )

    return (
        self._front_down_w4 is not None
        and self._front_down_scale4 is not None
        and self._latent_up_w4 is not None
        and self._latent_up_scale4 is not None
        and num_tokens >= _moe_latent_mxfp4_min_tokens
    )


def preroute_dense_weight(linear: torch.nn.Module) -> torch.Tensor:
    """Materialize a dense weight only while building decode-side caches."""
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16):
        return weight
    # Quark hangs the dequant on the scheme, other quant configs on the
    # quant method; either way only this cache build wants it dense.
    for owner in (getattr(linear, "scheme", None), linear.quant_method):
        materialize = getattr(owner, "materialize_bf16_weight", None)
        if materialize is not None:
            return materialize(linear)
    return weight


def prepare_preroute_fp8(self) -> None:
    from sglang.srt.models.kimi_k3 import (
        _aiter_moe_preroute_fp8,
    )

    if (
        not _aiter_moe_preroute_fp8
        or not self.use_latent_moe
        or self.shared_experts is None
    ):
        return
    from sglang.kernels.ops.moe import moe_preroute_aiter_hip
    from sglang.kernels.ops.quantization.aiter_fusion import (
        quantize_fp8_rows,
    )

    routed = preroute_dense_weight(self.routed_expert_down_proj)
    shared = preroute_dense_weight(self.shared_experts.gate_up_proj)
    shared_down = preroute_dense_weight(self.shared_experts.down_proj)
    if (
        tuple(routed.shape) != (3584, 7168)
        or tuple(shared.shape) != (1536, 7168)
        or tuple(shared_down.shape) != (7168, 768)
        or tuple(self.gate.weight.shape) != (896, 7168)
    ):
        return
    self._preroute_routed_weight, self._preroute_routed_scale = quantize_fp8_rows(
        routed.contiguous()
    )
    self._preroute_shared_weight, self._preroute_shared_scale = quantize_fp8_rows(
        shared.contiguous()
    )
    if moe_preroute_aiter_hip.cooperative_preactivated_enabled():
        self._preroute_shared_interleaved_weight = (
            self._preroute_shared_weight.view(2, 768, 7168)
            .permute(1, 0, 2)
            .contiguous()
            .view(1536, 7168)
        )
        self._preroute_shared_interleaved_scale = (
            self._preroute_shared_scale.view(2, 768).t().contiguous().view(1536)
        )
    (
        self._preroute_shared_down_weight,
        self._preroute_shared_down_scale,
    ) = quantize_fp8_rows(shared_down.contiguous())
    moe_preroute_aiter_hip.warmup(
        self._preroute_routed_weight,
        self._preroute_routed_scale,
        self._preroute_shared_weight,
        self._preroute_shared_scale,
        self.gate.weight,
        self._preroute_shared_down_weight,
        self._preroute_shared_down_scale,
        self._preroute_shared_interleaved_weight,
        self._preroute_shared_interleaved_scale,
        situ_beta=self._situ_beta,
        situ_linear_beta=self._situ_linear_beta,
    )


def use_latent_up_ptpc_fp8(self, latent: torch.Tensor) -> bool:
    from sglang.srt.models.kimi_k3_rocm_fusion import (
        _k3_ptpc_fp8_moe_gemm_ok,
    )

    if self._latent_up_fp8_w is None or not _k3_ptpc_fp8_moe_gemm_ok(
        latent.shape[0]
    ):
        return False
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    return ptpc_fp8_aiter_hip.covered(latent, self._latent_up_fp8_w)


def prepare_latent_up_ptpc_fp8(self) -> None:
    """Quantize the latent up-projection for the PTPC FP8 decode path."""
    from sglang.srt.models.kimi_k3_rocm_fusion import (
        _k3_ptpc_fp8,
    )

    if not _k3_ptpc_fp8 or self.routed_expert_up_proj is None:
        return
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    weight = self.routed_expert_up_proj.weight
    if (
        not ptpc_fp8_aiter_hip.available()
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
    ):
        return
    out_features, in_features = weight.shape
    (
        self._latent_up_fp8_w,
        self._latent_up_fp8_s,
        self._latent_up_fp8_n,
    ) = ptpc_fp8_aiter_hip.pack(weight.contiguous())
    ptpc_fp8_aiter_hip.warmup(
        self._latent_up_fp8_w,
        self._latent_up_fp8_s,
        self._latent_up_fp8_n,
        in_features,
    )


def prepare_shared_down_ptpc_fp8(self) -> None:
    """Quantize the shared-expert down projection for decode."""
    from sglang.srt.models.kimi_k3_rocm_fusion import (
        _k3_ptpc_fp8_shared_down,
    )

    if not _k3_ptpc_fp8_shared_down or self.shared_experts is None:
        return
    # Requantizing Quark's dequantized MXFP4 weight to FP8 stacks two
    # rounding steps; full GSM8K fell to 0.937 (vs 0.949 in BF16).
    if getattr(self.shared_experts.down_proj, "dequantized_bf16", False):
        return
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    weight = preroute_dense_weight(self.shared_experts.down_proj)
    if (
        not ptpc_fp8_aiter_hip.available()
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
    ):
        return
    _, in_features = weight.shape
    (
        self._shared_down_fp8_w,
        self._shared_down_fp8_s,
        self._shared_down_fp8_n,
    ) = ptpc_fp8_aiter_hip.pack(weight.contiguous())
    ptpc_fp8_aiter_hip.warmup(
        self._shared_down_fp8_w,
        self._shared_down_fp8_s,
        self._shared_down_fp8_n,
        in_features,
        token_buckets=(2, 4, 8, 16, 32, 64, 128, 256),
    )


def prepare_latent_tail_fp8(self) -> None:
    from sglang.srt.models.kimi_k3 import (
        _aiter_latent_tail_fp8,
    )

    if (
        not _aiter_latent_tail_fp8
        or not self.fuse_ar_norm
        or self.routed_expert_up_proj is None
        or self.routed_expert_norm is None
    ):
        return
    from sglang.kernels.ops.moe import latent_tail_aiter_hip

    if tuple(self.routed_expert_up_proj.weight.shape) != (7168, 3584):
        return
    self._latent_tail_weight, self._latent_tail_scale = latent_tail_aiter_hip.pack(
        self.routed_expert_up_proj.weight
    )
    norm_weight, epsilon = self._get_fused_norm_params()
    latent_tail_aiter_hip.warmup(
        norm_weight,
        self._latent_tail_weight,
        self._latent_tail_scale,
        epsilon,
    )


def try_shared_down_ptpc(self, x: torch.Tensor, out: torch.Tensor) -> bool:
    from sglang.srt.models.kimi_k3_rocm_fusion import _k3_ptpc_fp8_moe_gemm_ok

    if self._shared_down_fp8_w is None or not _k3_ptpc_fp8_moe_gemm_ok(x.shape[0]):
        return False
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    if not ptpc_fp8_aiter_hip.covered(x, self._shared_down_fp8_w):
        return False
    ptpc_fp8_aiter_hip.run(
        x,
        self._shared_down_fp8_w,
        self._shared_down_fp8_s,
        self._shared_down_fp8_n,
        out=out,
    )
    return True


def try_shared_down_preroute(self, gate_up, shared_output) -> bool:
    if (
        self._preroute_shared_down_weight is None
        or self._preroute_shared_down_scale is None
    ):
        return False
    from sglang.kernels.ops.moe import moe_preroute_aiter_hip

    if not moe_preroute_aiter_hip.shared_down_covered(
        gate_up,
        self._preroute_shared_down_weight,
        self._preroute_shared_down_scale,
    ):
        return False
    moe_preroute_aiter_hip.run_shared_down(
        gate_up,
        self._preroute_shared_down_weight,
        self._preroute_shared_down_scale,
        situ_beta=self._situ_beta,
        situ_linear_beta=self._situ_linear_beta,
        out=shared_output,
    )
    return True


def try_moe_preroute_front(self, hidden_states: torch.Tensor, num_tokens: int):
    if not (
        num_tokens <= 4
        and self._preroute_routed_weight is not None
        and self._preroute_routed_scale is not None
        and self._preroute_shared_weight is not None
        and self._preroute_shared_scale is not None
    ):
        return None
    from sglang.kernels.ops.moe import moe_preroute_aiter_hip

    if (
        self._preroute_shared_interleaved_weight is not None
        and self._preroute_shared_interleaved_scale is not None
        and moe_preroute_aiter_hip.cooperative_preactivated_tri_covered(
            hidden_states,
            self._preroute_routed_weight,
            self._preroute_routed_scale,
            self._preroute_shared_interleaved_weight,
            self._preroute_shared_interleaved_scale,
            self.gate.weight,
        )
    ):
        routed_input, gate_up, router_logits = (
            moe_preroute_aiter_hip.run_tri_cooperative_preactivated(
                hidden_states,
                self._preroute_routed_weight,
                self._preroute_routed_scale,
                self._preroute_shared_interleaved_weight,
                self._preroute_shared_interleaved_scale,
                self.gate.weight,
                situ_beta=self._situ_beta,
                situ_linear_beta=self._situ_linear_beta,
            )
        )
        return routed_input, gate_up, router_logits, True
    if moe_preroute_aiter_hip.tri_covered(
        hidden_states,
        self._preroute_routed_weight,
        self._preroute_routed_scale,
        self._preroute_shared_weight,
        self._preroute_shared_scale,
        self.gate.weight,
    ):
        routed_input, gate_up, router_logits = moe_preroute_aiter_hip.run_tri(
            hidden_states,
            self._preroute_routed_weight,
            self._preroute_routed_scale,
            self._preroute_shared_weight,
            self._preroute_shared_scale,
            self.gate.weight,
        )
        return routed_input, gate_up, router_logits, False
    return None


def run_latent_mxfp4(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
    from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip

    return latent_mxfp4_aiter_hip.run(x, weight, scale)


def run_latent_up_ptpc(self, latent: torch.Tensor) -> torch.Tensor:
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    return ptpc_fp8_aiter_hip.run(
        latent,
        self._latent_up_fp8_w,
        self._latent_up_fp8_s,
        self._latent_up_fp8_n,
    )


def try_latent_tail(self, latent, shared_output, prefix_sum, fused_norm):
    from sglang.kernels.ops.moe import latent_tail_aiter_hip

    norm_weight, epsilon = self._get_fused_norm_params()
    if not latent_tail_aiter_hip.covered(
        latent,
        shared_output,
        norm_weight,
        self._latent_tail_weight,
        self._latent_tail_scale,
        epsilon,
        prefix_sum,
    ):
        return None
    return latent_tail_aiter_hip.run(
        latent,
        shared_output,
        norm_weight,
        self._latent_tail_weight,
        self._latent_tail_scale,
        epsilon,
        prefix_sum,
        skip_rms=fused_norm,
    )
