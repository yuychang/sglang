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
"""ROCm latent-MoE path for Kimi-K3.

Shared ``kimi_k3.py`` only keeps ``_is_hip`` hooks into this module. These
functions take the ``KimiK3MoE`` module as ``self``.

Numerics follow vLLM's ROCm ``KimiMoE``: the router gate and the latent
down/up projections stay BF16 (they are excluded from quantization), the
router logits are fp32 (``GateLinear(out_dtype=float32)``) with an fp32
correction bias, and the shared experts run through their checkpoint
quantization (Quark MXFP4 W4A4).
"""

from typing import Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.communication.k3_moe_pair_ar import all_reduce_moe_latent_shared
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel


def init_moe_rocm_state(self, config) -> None:
    """ROCm-only state, filled by prepare_moe_rocm after weights load."""
    self._partial_fused_front = False


def select_front_modules(self) -> Optional[list]:
    """Modules for the merged MoE front on ROCm, or None to skip the merge."""
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable

    if self.shared_experts is not None and get_moe_a2a_backend().is_none():
        full_front = [
            self.shared_experts.gate_up_proj,
            self.gate,
            self.routed_expert_down_proj,
        ]
        if _is_unquantized_mergeable([m.weight for m in full_front]):
            return full_front
        if envs.SGLANG_K3_FUSED_FRONT.get():
            # Quark quantizes the shared experts but leaves the router and
            # latent projections dense; keep the gate+latent merge and leave
            # the shared branch on its native quantized kernels.
            return [self.gate, self.routed_expert_down_proj]
        return None
    if envs.SGLANG_K3_FUSED_FRONT.get():
        return [self.gate, self.routed_expert_down_proj]
    return None


def prepare_moe_rocm(self) -> None:
    """Post-load ROCm setup; runs after the shared front merge."""
    from sglang.srt.layers.moe import get_moe_a2a_backend

    # Dense gate+latent front with a separately quantized shared branch.
    self._partial_fused_front = (
        self.use_latent_moe
        and self.shared_experts is not None
        and self._front_w is not None
        and self._front_is_ep_pair
        and get_moe_a2a_backend().is_none()
    )


def _front_needs_dense_bf16(self) -> bool:
    # AITER's MoE route indexes rows by input.stride(-2), so it consumes the
    # fused-front split view directly.
    runner = self.experts.runner
    if runner is not None and runner.runner_backend.is_aiter():
        return False
    return self._moe_front_needs_dense_bf16


def _forward_shared(self, gate_up, shared_output) -> None:
    from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

    _k3_bf16_gemm(
        self.shared_experts.act_fn(gate_up),
        self.shared_experts.down_proj.weight,
        out=shared_output,
    )


def _apply_into(linear: torch.nn.Module, x: torch.Tensor, output: torch.Tensor) -> bool:
    """Run a quantized linear into ``output`` when its scheme supports it.

    Quark keeps ``apply_into`` on ``linear.scheme``, not ``quant_method``.
    """
    scheme = getattr(linear, "scheme", None)
    apply_into = getattr(scheme, "apply_into", None) if scheme is not None else None
    if apply_into is None:
        apply_into = getattr(getattr(linear, "quant_method", None), "apply_into", None)
    if apply_into is None:
        return False
    apply_into(linear, x, output)
    return True


def _forward_quantized_shared(
    self, hidden_states: torch.Tensor, shared_output: torch.Tensor
) -> None:
    """Run the quantized shared MLP into the fused collective buffer."""
    shared = self.shared_experts
    gate_up, _ = shared.gate_up_proj(hidden_states)
    activated = shared.act_fn(gate_up)
    if not _apply_into(shared.down_proj, activated, shared_output):
        output, _ = shared.down_proj(activated)
        shared_output.copy_(output)


def _run_front(self, hidden_states: torch.Tensor):
    """Return (gate_up, router_logits, routed_input).

    The merged front GEMM accumulates in fp32 and writes fp32, so the router
    slice carries vLLM's fp32 gate logits; the BF16 slices are rounded back to
    bf16, which equals a separate bf16-output GEMM.
    """
    from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

    fused = _k3_bf16_gemm(hidden_states, self._front_w, out_dtype=torch.float32)
    if self._partial_fused_front:
        router_logits, routed_input = torch.split(fused, self._front_sizes, dim=-1)
        gate_up = None
    else:
        gate_up, router_logits, routed_input = torch.split(
            fused, self._front_sizes, dim=-1
        )
        gate_up = gate_up.to(hidden_states.dtype)
    routed_input = routed_input.to(hidden_states.dtype)
    return gate_up, router_logits, routed_input


def _all_reduce_pair(
    self,
    buf: torch.Tensor,
    *,
    num_tokens: int,
    hidden_size: int,
):
    """Reduce the flat [latent | shared] buffer.

    Returns (latent, shared_output, fused_norm); with fused_norm the latent is
    already RMS-normalized.
    """
    from sglang.srt.layers.communication import k3_ar_fusion
    from sglang.srt.layers.communication.hip_fused_ar_rmsnorm import (
        try_fused_ar_rmsnorm,
    )

    latent_numel = num_tokens * self.moe_hidden_size
    if self.fuse_ar_norm:
        weight, eps = self._get_fused_norm_params()
        view = buf.view(-1, k3_ar_fusion.NORM_DIM)
        fused = try_fused_ar_rmsnorm(view, weight, eps, num_norm_rows=num_tokens)
        if fused is not None:
            normed, reduced = fused
            if reduced.data_ptr() != view.data_ptr():
                buf.copy_(reduced.reshape(-1))
            shared_output = buf[latent_numel:].view(num_tokens, hidden_size)
            return normed[:num_tokens], shared_output, True
    # A 16K concat (~336 MiB) misses the 256 MiB QR cap and becomes NCCL
    # Generic; split so each slice fits QR.
    latent, shared_output = all_reduce_moe_latent_shared(
        buf,
        num_tokens=num_tokens,
        moe_hidden_size=self.moe_hidden_size,
        hidden_size=hidden_size,
    )
    return latent, shared_output, False


def forward_fused_rocm(
    self,
    hidden_states: torch.Tensor,
    *,
    prefix_sum: Optional[torch.Tensor],
    forward_batch: Optional[ForwardBatch],
) -> torch.Tensor:
    """ROCm counterpart of KimiK3MoE._forward_fused.

    Same [latent | shared] single-collective layout; adds the partial (Quark)
    front and the fused AR+RMSNorm.
    """
    from sglang.srt.models.kimi_k3 import _add3, _aiter_k3_opt

    num_tokens, hidden_size = hidden_states.shape
    gate_up, router_logits, routed_input = _run_front(self, hidden_states)
    if num_tokens > 1 and not _aiter_k3_opt:
        router_logits = router_logits.contiguous()
    if _front_needs_dense_bf16(self):
        routed_input = routed_input.contiguous()
    latent_numel = num_tokens * self.moe_hidden_size
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        buf = hidden_states.new_empty(latent_numel + num_tokens * hidden_size)
    latent = buf[:latent_numel].view(num_tokens, self.moe_hidden_size)
    shared_output = buf[latent_numel:].view(num_tokens, hidden_size)

    if gate_up is None:
        _forward_quantized_shared(self, hidden_states, shared_output)
    else:
        _forward_shared(self, gate_up, shared_output)
    self._forward_routed(hidden_states, router_logits, routed_input, latent)

    latent, shared_output, fused_norm = _all_reduce_pair(
        self, buf, num_tokens=num_tokens, hidden_size=hidden_size
    )
    if not fused_norm:
        latent = self._latent_norm(latent)
    out, _ = self.routed_expert_up_proj(latent)
    return _add3(out, shared_output, prefix_sum, prefetch_bc=True)
