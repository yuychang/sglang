# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from sglang.srt.environ import envs

_SHARED_PARTIAL_ATTR = "_sglang_rocm_shared_partial"
_SUPPORTED_TOKEN_COUNTS = frozenset((1, 2, 4, 8, 16, 32))
_KIMI_HIDDEN_SIZE = 7168


def rocm_mxfp4_moe_add_shared(
    routed_output: torch.Tensor,
    shared_output: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    try:
        from sgl_kernel import rocm_mxfp4_moe_add_shared as add_shared

        return add_shared(
            routed_output,
            shared_output,
            output=output,
        )
    except ImportError:
        if output is None:
            return routed_output + shared_output
        return torch.add(routed_output, shared_output, out=output)


def can_defer_shared_partial_to_graph_ar(
    routed_output: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    *,
    should_allreduce_fusion: bool,
    shared_expert_tp1: bool,
    tp_world_size: int,
    is_graph_capture_mode: bool,
    is_gfx950: bool,
) -> bool:
    return bool(
        envs.SGLANG_ROCM_FUSE_SHARED_PARTIAL_AR_RMSNORM.get()
        and should_allreduce_fusion
        and not shared_expert_tp1
        and is_graph_capture_mode
        and is_gfx950
        and tp_world_size == 4
        and shared_output is not None
        and routed_output.device.type == "cuda"
        and shared_output.device == routed_output.device
        and routed_output.dtype == torch.bfloat16
        and shared_output.dtype == torch.bfloat16
        and routed_output.dim() == 2
        and routed_output.shape == shared_output.shape
        and routed_output.shape[0] in _SUPPORTED_TOKEN_COUNTS
        and routed_output.shape[1] == _KIMI_HIDDEN_SIZE
        and routed_output.is_contiguous()
        and shared_output.is_contiguous()
    )


def attach_shared_partial(
    routed_output: torch.Tensor,
    shared_output: torch.Tensor,
) -> torch.Tensor:
    if hasattr(routed_output, _SHARED_PARTIAL_ATTR):
        raise RuntimeError("ROCm shared partial carrier was already populated")
    setattr(routed_output, _SHARED_PARTIAL_ATTR, shared_output)
    return routed_output


def pop_shared_partial(routed_output: torch.Tensor) -> Optional[torch.Tensor]:
    shared_output = getattr(routed_output, _SHARED_PARTIAL_ATTR, None)
    if hasattr(routed_output, _SHARED_PARTIAL_ATTR):
        delattr(routed_output, _SHARED_PARTIAL_ATTR)
    return shared_output
