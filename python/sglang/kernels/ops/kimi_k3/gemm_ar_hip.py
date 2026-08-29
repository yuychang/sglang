"""HIP gfx950 fused o_proj GEMM + TP all-reduce for K3 decode.

Port target for ``gemm_ar.cuh`` (CUDA SM100+).

* Phase 1: bf16 GEMM + generic TP all-reduce (two launches, copies partial).
* Phase 2: GEMM into the pre-registered custom-AR input pool + registered AR
  (two launches, skips the custom-AR D2D copy on the partial).
* Phase 3 (future): single fused HIP kernel with P2P tile push like CUDA SM100.

Enabled with ``SGLANG_K3_GEMM_AR=1`` on ROCm gfx950; see ``k3_gemm_ar.py``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import is_hip_runtime

N = 7168  # K3 hidden size (must match gemm_ar.N)
MAX_TOKENS = 512


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


def _partial_numel(m: int) -> int:
    return m * N


def _get_ca_comm():
    from sglang.srt.distributed.parallel_state import get_tp_group

    ca = get_tp_group().ca_comm
    if ca is None or ca.disabled:
        return None
    return ca


def _registered_input_view(ca, m: int) -> torch.Tensor:
    """View into the IPC-registered custom-AR input staging buffer."""
    numel = _partial_numel(m)
    if hasattr(ca, "_pool"):
        buf = ca._pool["input"]
        if buf._buffer is not None:
            return buf.tensor.view(torch.bfloat16)[:numel].reshape(m, N)
        raise RuntimeError(
            "gemm_ar_hip: custom-AR input pool has no torch tensor backing "
            "(raw_cached IPC); use fallback path"
        )
    if hasattr(ca, "buffer"):
        return ca.buffer.view(torch.bfloat16)[:numel].reshape(m, N)
    raise RuntimeError("gemm_ar_hip: unknown custom-AR communicator layout")


def _can_use_registered_path(ca, m: int) -> bool:
    try:
        _registered_input_view(ca, m)
    except RuntimeError:
        return False
    probe = torch.empty((m, N), dtype=torch.bfloat16, device=ca.device)
    if hasattr(ca, "should_custom_ar_bytes"):
        return ca.should_custom_ar_bytes(probe)
    return ca.should_custom_ar(probe)


def _all_reduce_registered(ca, inp: torch.Tensor, out: torch.Tensor) -> None:
    if hasattr(ca, "all_reduce"):
        ca.all_reduce(inp, out=out, registered_input=True)
        return
    import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops

    inp_flat = inp.reshape(-1)
    out_flat = out.reshape(-1)
    if ca.use_amd_deterministic_impl:
        ops.deterministic_all_reduce_reg(ca._ptr, inp_flat, out_flat)
    else:
        ops.all_reduce_reg(ca._ptr, inp_flat, out_flat)


def _gemm_ar_via_registered_buffer(
    x: torch.Tensor, weight: torch.Tensor, ca, m: int
) -> torch.Tensor:
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )

    reg = _registered_input_view(ca, m)
    torch.mm(x, weight.t(), out=reg)

    if ca._IS_CAPTURING:
        if torch.cuda.is_current_stream_capturing():
            return torch.zeros((m, N), dtype=torch.bfloat16, device=x.device)
        if not is_in_tc_piecewise_cuda_graph():
            return torch.zeros((m, N), dtype=torch.bfloat16, device=x.device)

    out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
    _all_reduce_registered(ca, reg, out)
    return out


def _gemm_ar_fallback(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    partial = torch.matmul(x, weight.t())
    from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(partial)


def o_proj_gemm_ar(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fully reduced ``sum_r x_r @ W_r^T`` on every rank."""
    state = _STATE
    assert state is not None
    assert weight.shape[0] == N and weight.shape[1] == state.k_local
    m = x.shape[0]
    ca = _get_ca_comm()
    if ca is not None and _can_use_registered_path(ca, m):
        return _gemm_ar_via_registered_buffer(x, weight, ca, m)
    return _gemm_ar_fallback(x, weight)
