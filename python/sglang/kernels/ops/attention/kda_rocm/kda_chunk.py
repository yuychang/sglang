# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm entry point for the fused Kimi-K3 KDA chunk kernel.

Ported from vLLM ``vllm/models/kimi_k3/amd/ops/kda_chunk.py``. The kernel in
``csrc/fused_kda_chunk_kernel_rocm.cu`` replaces the chunk-state recurrence and
the output GEMM of the Triton chunk path with a single launch that keeps the
per-chunk state in registers, so the ``[chunks, H, V, K]`` state tensor and the
recomputed values never reach HBM.
"""

from functools import cache

import torch

from sglang.kernels.ops.attention.kda_rocm import jit
from sglang.kernels.ops.attention.kda_rocm.third_party.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

CHUNK_SIZE = 64
HEAD_DIM = 128
KDA_CHECKPOINT_ALIGNMENT = CHUNK_SIZE
# Chunk-group split policy constants; see _chunk_groups.
_BLOCKS_P1 = 2  # (kV + kK) / kBV
_BLOCKS_P2 = 1  # kV / kBV
_MIN_LEN = 4
_DEEP_LEN = 12
_MAX_GROUPS = 32


def _hip_checkpoint_state_indices(state_indices: torch.Tensor) -> torch.Tensor:
    """The HIP kernel skips negative rows (vLLM maps NULL_BLOCK_ID to -1)."""
    return state_indices.to(torch.int32).contiguous()


@cache
def is_fused_kda_chunk_supported() -> bool:
    # vLLM builds and validates the chunk kernel on gfx950 only.
    return jit.device_arch() == "gfx950" and jit.has_op("fused_kda_chunk")


def can_use_fused_kda_chunk(
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
    chunk_size: int,
) -> bool:
    return (
        head_k_dim == HEAD_DIM
        and head_v_dim == HEAD_DIM
        and chunk_size == CHUNK_SIZE
        and dtype == torch.bfloat16
        and is_fused_kda_chunk_supported()
    )


def fused_kda_prologue(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    cu_seqlens: torch.Tensor,
    conv_weight: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    conv_has_initial_state: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run the whole chunk-path prologue in one launch.

    Replaces the two L2 norms, the gate cumsum, both intra-chunk passes and the
    w/u recompute. Returns the operands the fused chunk kernel consumes.
    """
    _, t_total, num_heads, _ = q.shape
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    chunk_indices = chunk_indices.to(torch.int32)
    num_chunks = chunk_indices.shape[0]
    dev = q.device

    def _like(last: int) -> torch.Tensor:
        return torch.empty(1, t_total, num_heads, last, dtype=q.dtype, device=q.device)

    ws = dict(
        qg=_like(HEAD_DIM),
        w=_like(HEAD_DIM),
        u=_like(HEAD_DIM),
        kg_t=torch.empty(
            num_chunks, num_heads, HEAD_DIM, CHUNK_SIZE, dtype=q.dtype, device=dev
        ),
        aqk=torch.empty(1, t_total, num_heads, CHUNK_SIZE, dtype=q.dtype, device=dev),
        decay=torch.empty(
            num_chunks, num_heads, HEAD_DIM, dtype=torch.float32, device=dev
        ),
    )
    jit.load_ops().fused_kda_prologue(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log.reshape(-1),
        dt_bias.reshape(-1),
        ws["qg"],
        ws["w"],
        ws["u"],
        ws["kg_t"],
        ws["aqk"],
        ws["decay"],
        cu_seqlens.to(torch.int32),
        chunk_indices,
        conv_weight,
        conv_state,
        conv_state_indices,
        conv_has_initial_state,
        scale,
        lower_bound,
    )
    return ws


@cache
def _num_cus(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@cache
def _chunk_groups(chunks_per_seq: int, num_seqs: int, num_heads: int) -> int:
    """How many parallel chunk groups to cut each sequence into.

    The scan composes the group operators with the exact transfer
    ``M_g = prod_c (diag(d_c) - w_c^T kg_c)``, so the composition error does
    not grow with group length and the choice is purely a machine fit (see the
    vLLM source for the derivation of the constants).
    """
    cus = _num_cus(torch.cuda.current_device())
    nh = max(num_seqs * num_heads, 1)
    fill1 = cus // (_BLOCKS_P1 * nh)
    fill2 = cus // (_BLOCKS_P2 * nh)
    cand = fill1
    if fill2 > 0 and chunks_per_seq // fill2 >= _DEEP_LEN:
        cand = fill2
    cand = min(cand, chunks_per_seq // _MIN_LEN, _MAX_GROUPS)
    return cand if cand >= 4 else 1


def _kda_group_workspace(
    groups: int, nh: int, device: torch.device
) -> torch.Tensor | None:
    """One fp32 buffer the kernel carves into bg / sin_ / ag / mgT."""
    if groups <= 1:
        return None
    planes = groups * nh
    plane = HEAD_DIM * HEAD_DIM
    floats = 2 * planes * plane + planes * HEAD_DIM  # bg, sin_, ag
    floats += (planes * plane + 1) // 2  # mgT, bf16
    return torch.empty(floats, dtype=torch.float32, device=device)


def fused_kda_chunk(
    qg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    kg_t: torch.Tensor,
    aqk: torch.Tensor,
    decay: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_offsets: torch.Tensor | None = None,
    checkpoint_state: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    checkpoint_state_indices: torch.Tensor | None = None,
    state_cache: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the chunk recurrence and the output projection in one launch.

    ``state_cache`` is the paged fp32 ``[slots, H, 128, 128]`` recurrent state:
    the walk reads sequence ``n``'s initial state from row ``state_indices[n]``
    (zero when ``has_initial_state[n]`` is false) and writes its final state
    back to the same row. It replaces ``initial_state`` / ``output_final_state``.
    """
    if state_cache is not None:
        if initial_state is not None or output_final_state:
            raise ValueError("state_cache replaces initial_state/output_final_state")
        if state_indices is None or has_initial_state is None:
            raise ValueError("state_cache needs state_indices and has_initial_state")
    num_seqs = cu_seqlens.numel() - 1
    final_state = None
    if output_final_state:
        final_state = torch.empty(
            num_seqs,
            u.shape[2],
            u.shape[3],
            qg.shape[3],
            dtype=torch.float32,
            device=u.device,
        )
    cu_seqlens = cu_seqlens.to(torch.int32)
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)

    chunks_per_seq = kg_t.shape[0] // max(num_seqs, 1)
    groups = _chunk_groups(chunks_per_seq, num_seqs, u.shape[2])
    group_state = _kda_group_workspace(groups, num_seqs * u.shape[2], u.device)

    jit.load_ops().fused_kda_chunk(
        qg,
        w,
        u,
        kg_t,
        aqk,
        decay,
        initial_state,
        final_state,
        out,
        cu_seqlens,
        chunk_offsets.to(torch.int32),
        scale,
        group_state,
        groups,
        checkpoint_state,
        None if checkpoint_offsets is None else checkpoint_offsets.to(torch.int32),
        None
        if checkpoint_state_indices is None
        else _hip_checkpoint_state_indices(checkpoint_state_indices),
        state_cache,
        None if state_indices is None else state_indices.to(torch.int32).contiguous(),
        has_initial_state,
    )
    return out, final_state
