"""HIP fused all-reduce + residual add for K3 attention o_proj (AR1).

NVIDIA folds the pending attn-res prefix into the fused MNNVL all-reduce
(``k3_ar_fusion.all_reduce(x, prefix)``), then runs aggregation 2 with
``prefix_sum=None``. On HIP that MNNVL path is unavailable, and AITER's
token-parallel fused AR+RMSNorm 1-stage kernel loses to the element-parallel
1-stage AR used for c2/c4 decode.

This helper folds the prefix into AITER custom AR. Decode M in {1, 2, 4}
uses the element-parallel 1-stage kernel. M=8 is 112 KiB and uses the
2-stage kernel with the residual add in the all-gather writeback. M=16
stays on split AR + ``_agg_kernel`` HAS_ADD.

Aggregation 2 (the MLP-side mixer) is *before* MoE, so it cannot be folded
into ``latent_tail``. The adjacent MoE-side pair is AR2 then latent_tail;
fusing those needs IPC inside FlyDSL and is not this path.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

# Steady-state decode M plus the M=1 CUDA-graph drain bucket.
_RESIDUAL_BATCHES = (1, 2, 4, 8)


def enabled() -> bool:
    return is_hip() and envs.SGLANG_ROCM_K3_AR_RESIDUAL.get()


def covers(num_tokens: int) -> bool:
    if num_tokens not in _RESIDUAL_BATCHES:
        return False
    return num_tokens <= envs.SGLANG_ROCM_K3_AR_RESIDUAL_MAX_TOKENS.get()


def fresh_output(like: torch.Tensor) -> torch.Tensor:
    """Allocate the fused AR output. Never cache it by shape.

    M=8 is past the 80 KiB 1-stage cutoff, so the collective is
    ``cross_device_reduce_2stage_res``: stage 1 matches plain 2-stage AR,
    and stage 2 adds the residual after the reduced value has rounded to
    the storage dtype. The residual pointer is the caller's tensor. An
    earlier fusion stashed that pointer (and the output) in a shape-keyed
    cache and replaced the entry on the next capture, which freed storage
    a CUDA graph still replayed. This allocator returns a new buffer every
    call. During capture that buffer comes from the graph pool.
    """
    return torch.empty_like(like)


def try_all_reduce_add(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """``AR(x) + residual`` in one custom AR, or ``None``.

    ``residual`` must be identical on every rank. Residual fusion runs for
    decode M covered by ``SGLANG_ROCM_K3_AR_RESIDUAL_MAX_TOKENS``. When
    ``residual`` is ``None`` this still completes a deferred o_proj
    all-reduce (plain custom AR) at any M.
    """
    if not enabled() or x.numel() == 0:
        return None
    if residual is not None and not covers(x.shape[0]):
        return None
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        ca_comm = group.ca_comm
        if ca_comm is None or getattr(ca_comm, "disabled", True):
            return None
        if residual is None:
            if hasattr(ca_comm, "custom_all_reduce"):
                out = ca_comm.custom_all_reduce(x)
                return out
            return None
        # KDA preserves a leading singleton/head view around the residual while
        # o_proj returns the same token-major storage flattened. Aggregation
        # treats these as the same elementwise tensor; normalize the view so
        # AITER's shape guard does not send the seven KDA layers down split AR.
        if residual.shape != x.shape:
            if residual.numel() != x.numel() or not residual.is_contiguous():
                return None
            residual = residual.view_as(x)
        if not hasattr(ca_comm, "custom_all_reduce_residual"):
            return None
        # Capture allocates here, on the capturing stream, and passes that
        # buffer through. The residual itself is not copied or retained.
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            return ca_comm.custom_all_reduce_residual(x, residual, fresh_output(x))
        return ca_comm.custom_all_reduce_residual(x, residual)
    except Exception as exc:
        logger.debug("K3 HIP AR+residual unavailable: %s", exc)
        return None
