"""HIP fused all-reduce + residual add for K3 attention o_proj (AR1).

NVIDIA folds the pending attn-res prefix into the fused MNNVL all-reduce
(``k3_ar_fusion.all_reduce(x, prefix)``), then runs aggregation 2 with
``prefix_sum=None``. On HIP that MNNVL path is unavailable, and AITER's
token-parallel fused AR+RMSNorm 1-stage kernel loses to the element-parallel
1-stage AR used for c2/c4 decode.

This helper launches that same element-parallel 1-stage kernel with the
prefix add folded into the fp32 writeback. Decode M in {1, 2, 4} sit
inside AITER's TP8 1-stage AR sizes (28 KiB / 56 KiB). M=8 is 112 KiB,
past the 80 KiB 2-stage crossover for *plain* AR, but the residual
kernel can still run 1-stage without flipping other collectives. M=16
is 224 KiB and stays on split AR + ``_agg_kernel`` HAS_ADD.

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

# Steady-state decode M plus the M=1 CUDA-graph drain bucket. The env cap
# defaults to 8 so conc=8 can use residual fusion; set it to 4 to restore
# the previous {1, 2, 4}-only policy.
_RESIDUAL_BATCHES = (1, 2, 4, 8)


def enabled() -> bool:
    return is_hip() and envs.SGLANG_ROCM_K3_AR_RESIDUAL.get()


def covers(num_tokens: int) -> bool:
    if num_tokens not in _RESIDUAL_BATCHES:
        return False
    return num_tokens <= envs.SGLANG_ROCM_K3_AR_RESIDUAL_MAX_TOKENS.get()


def try_all_reduce_add(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """``AR(x) + residual`` in one 1-stage custom AR, or ``None``.

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
        if not hasattr(ca_comm, "custom_all_reduce_residual"):
            return None
        return ca_comm.custom_all_reduce_residual(x, residual)
    except Exception as exc:
        logger.debug("K3 HIP AR+residual unavailable: %s", exc)
        return None
