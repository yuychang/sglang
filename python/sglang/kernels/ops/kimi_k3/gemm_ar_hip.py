"""HIP gfx950 fused o_proj GEMM + TP all-reduce for K3 decode.

Port target for ``gemm_ar.cuh`` (CUDA SM100+). Phase 1 uses a Triton local-GEMM
plus the existing TP custom all-reduce (two launches — correctness baseline).
Phase 2 will fuse the epilogue into one kernel using P2P push like the CUDA path.

Enabled with ``SGLANG_K3_GEMM_AR=1`` on ROCm gfx950; see ``k3_gemm_ar.py``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import is_hip_runtime

N = 7168  # K3 hidden size (must match gemm_ar.N)
MAX_TOKENS = 512
_K_LOCAL = 12288  # total K / TP8; validated at init from weight shape


class _State(NamedTuple):
    world_size: int
    rank: int
    k_local: int


_STATE: Optional[_State] = None


def init(*, world_size: int, rank: int, k: int) -> None:
    global _STATE
    if _STATE is not None:
        return
    assert is_hip_runtime(), "gemm_ar_hip is ROCm-only"
    assert 2 <= world_size <= 8
    assert k % 128 == 0
    _STATE = _State(world_size=world_size, rank=rank, k_local=k)


def initialized() -> bool:
    return _STATE is not None


def fits(x: torch.Tensor) -> bool:
    """Decode-shaped o_proj input eligible for the HIP path."""
    return (
        _STATE is not None
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and 0 < x.shape[0] <= MAX_TOKENS
        and x.stride(1) == 1
        and x.is_contiguous()
    )


def o_proj_gemm_ar(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fully reduced ``sum_r x_r @ W_r^T`` on every rank.

    Phase 1: bf16 GEMM (local shard) + TP all-reduce. Same numerics as the
    unfused RowParallelLinear path; launch count unchanged until the fused
    HIP kernel lands.
    """
    state = _STATE
    assert state is not None
    assert weight.shape[0] == N and weight.shape[1] == state.k_local
    # Local partial — same contract as RowParallelLinear without reduce.
    partial = torch.matmul(x, weight.t())
    from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(partial)
