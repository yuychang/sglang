"""Whether a DSpark draft stores sharded KV under decode context parallelism.

The draft is Qwen3 MHA. The target KV is MLA, so the two cannot share one
buffer. Joining DCP keeps one sequence shard per rank and merges partial
attention. That layout holds the same bytes as a replicated draft (16 KV
heads on ``seq / dcp_size`` versus 2 heads on the full sequence) and adds an
all-gather plus an all-reduce on every verify step. The c56 hour of
2026-10-23 measured the cost: inter-token latency 49 ms to 77 ms, and
10065.9 to 8839.0 tok/s/GPU. The join stays off; the helpers below remain
for the index rule.
"""

from __future__ import annotations

import torch

# Measured regression on c56 (see the module note). Flip only with a verify
# path that does not add a collective per draft layer.
_DSPARK_DCP_JOIN = False


def dspark_mha_uses_unified_attention(
    *, is_draft_worker: bool, is_dspark: bool, use_mla: bool
) -> bool:
    """Qwen3 DSpark verify should run AITER ``unified_attention``.

    The draft is MHA with head_dim 64. AITER's ASM paged kernels cover
    head_dim 128 and 256; ATOM sends every other MHA head size through
    ``unified_attention`` (GQA-packed, split-KV). Verify is a few query
    tokens against a long prefix, which is the shape that kernel is for.
    A DeepSeek-V4 self-draft is MLA and keeps its own backend.
    """
    return bool(is_draft_worker and is_dspark and not use_mla)


def dspark_draft_joins_dcp() -> bool:
    """HIP Qwen3 DSpark joins the DCP group.

    A DeepSeek-V4 self-draft keeps its own backend and the replicated pool.
    The join is gated off: the sharded verify path is slower than the
    replicated pool at the same KV bytes.
    """
    if not _DSPARK_DCP_JOIN:
        return False
    from sglang.srt.runtime_context import get_parallel, get_spec
    from sglang.srt.speculative.dspark_components.dspark_config import (
        draft_is_deepseek_v4,
    )
    from sglang.srt.utils import is_hip

    if not is_hip() or get_parallel().attn_dcp_size <= 1:
        return False
    algorithm = getattr(get_spec(), "speculative_algorithm", None)
    # The published spec leaf stores the server-arg string. Pool setup later
    # wraps that same value in SpeculativeAlgorithm.
    if isinstance(algorithm, str):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        algorithm = SpeculativeAlgorithm.from_string(algorithm or None)
    if algorithm is None or not algorithm.is_dspark():
        return False
    return not draft_is_deepseek_v4()


def dcp_joined_kv_locs(
    raw,
    *,
    dcp_size: int,
    dcp_rank: int,
    positions=None,
    commit_lens=None,
    row_width=None,
):
    """Physical rows for a joined draft store, plus the tokens this rank keeps.

    Prefill injection and verify commit hand the allocator's widened ids to
    ``set_kv_buffer`` directly. The device pool is one shard, so those ids
    have to be collapsed (`loc // dcp_size`) and masked to this rank. A
    commit window drops columns past ``commit_lens`` the same way.
    """
    safe, owned = dcp_kv_write_loc(
        raw,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        positions=positions,
    )
    if (
        commit_lens is not None
        and row_width is not None
        and owned.numel() == int(commit_lens.numel()) * int(row_width)
    ):
        cols = torch.arange(int(row_width), device=owned.device)
        committed = cols.unsqueeze(0) < commit_lens.to(
            device=owned.device, dtype=torch.long
        ).unsqueeze(1)
        owned = owned & committed.reshape(-1)
    return safe, owned


def dcp_kv_write_loc(
    raw,
    *,
    dcp_size: int,
    dcp_rank: int,
    positions=None,
    preset_mask=None,
):
    """Physical rows and the tokens this rank may store.

    Write ids are widened: the owner is ``loc % dcp_size`` and the row is
    ``loc // dcp_size``. A CUDA-graph pad of ``-1`` must not be divided (floor
    division keeps it negative) and must not be stored. When ``positions``
    does not line up with ``raw``, fall back to the loc's own owner bit
    instead of writing every index.
    """
    n = raw.numel()
    if positions is not None and positions.numel() == n:
        owned = positions % dcp_size == dcp_rank
    elif preset_mask is not None and preset_mask.numel() == n:
        owned = preset_mask
    else:
        owned = raw % dcp_size == dcp_rank
    owned = owned & (raw >= 0)
    safe = torch.where(raw >= 0, raw // dcp_size, torch.zeros_like(raw))
    return safe, owned


def dcp_extend_kv_head_views(
    k,
    v,
    *,
    tp_k_head_num: int,
    dcp_rank: int,
    dcp_world: int,
):
    """Current-token K/V shard, plus the unsplit tensors for prefix attention.

    ``extend_attention_fwd`` sets ``kv_group_num = q_heads / k_heads``. The
    current-token call sees this rank's query heads, so it needs the matching
    KV-head shard. The prefix call all-gathers every query head and indexes
    the full KV buffer, so its K must keep every head even after ``k[:0]``.
    Slicing that tensor first (64 query heads / 2 KV heads) attends the wrong
    buffer heads and the draft stops matching the target.
    """
    if k.numel() == 0 or tp_k_head_num <= 1 or dcp_world <= 1:
        return k, v, k, v
    kv_head_start = dcp_rank * tp_k_head_num // dcp_world
    kv_head_end = max(
        (dcp_rank + 1) * tp_k_head_num // dcp_world,
        kv_head_start + 1,
    )
    return (
        k[:, kv_head_start:kv_head_end],
        v[:, kv_head_start:kv_head_end],
        k,
        v,
    )


def draft_kv_tp_size(*, tp_size: int, dcp_size: int) -> int:
    """KV-head partition width. Joining DCP replicates KV inside the DCP group."""
    if dcp_size <= 1:
        return tp_size
    if tp_size % dcp_size != 0:
        raise ValueError(
            f"DSpark DCP requires tp_size divisible by dcp_size, got "
            f"tp_size={tp_size} dcp_size={dcp_size}."
        )
    return tp_size // dcp_size
