"""HIP fused all-reduce + residual add for K3 attention o_proj (AR1).

NVIDIA folds the pending attn-res prefix into the fused MNNVL all-reduce
(``k3_ar_fusion.all_reduce(x, prefix)``), then runs aggregation 2 with
``prefix_sum=None``. On HIP that MNNVL path is unavailable, and AITER's
token-parallel fused AR+RMSNorm 1-stage kernel loses to the element-parallel
1-stage AR used for c2/c4 decode.

This helper launches that same element-parallel 1-stage kernel with the
prefix add folded into the fp32 writeback. It is used only for decode
batches M in {1, 2, 4} (conc 2 / conc 4, plus M=1 drain graphs). Those
are the TP8 1-stage AR sizes (28 KiB / 56 KiB). M>=8 is 2-stage and
keeps split AR + ``_agg_kernel`` HAS_ADD.

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

# Steady-state decode M for conc 2 and conc 4, plus the M=1 CUDA-graph drain
# bucket. M=8 is 112 KiB and takes 2-stage AR, so it is not fused.
_RESIDUAL_BATCHES = (1, 2, 4)


def enabled() -> bool:
    return is_hip() and envs.SGLANG_K3_HIP_AR_RESIDUAL.get()


def covers(num_tokens: int) -> bool:
    return num_tokens in _RESIDUAL_BATCHES


def try_all_reduce_add(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """``AR(x) + residual`` in one 1-stage custom AR, or ``None``.

    ``residual`` must be identical on every rank. Residual fusion runs only
    for M in {1, 2, 4}. When ``residual`` is ``None`` this still completes a
    deferred o_proj all-reduce (plain custom AR) at any M.
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
