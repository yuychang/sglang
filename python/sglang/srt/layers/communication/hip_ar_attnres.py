"""Fused attention all-reduce, residual add, AttnRes and RMSNorm.

TokenSpeed folds those four steps into one kernel for up to 16 tokens. This is
the ROCm version of that window. Each rank copies its o_proj partial into a
symmetric buffer and publishes a per-row flag from a one-warp kernel. A wide
kernel then sums the ranks in FP32 and runs the same AttnRes epilogue as
``attn_res_hip`` (residual add, bank mix, output RMSNorm) without spinning.
Larger batches return None and stay on the split path.

The symmetric heap is mapped on the first eager call. A call that arrives
while a CUDA graph is already being captured, before that map exists, declines
so capture never rendezvous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attn_residual import get_cw
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_MAX_TOKENS = 16
_SUPPORTED_WORLD = (2, 4, 8)
_STATES: dict[tuple, "_Heap"] = {}


def enabled() -> bool:
    return is_hip() and envs.SGLANG_ROCM_AR_ATTNRES.get()


def covers(num_tokens: int) -> bool:
    cap = envs.SGLANG_ROCM_AR_ATTNRES_MAX_TOKENS.get()
    return 0 < num_tokens <= min(_MAX_TOKENS, cap)


@dataclass
class _Heap:
    buf: torch.Tensor
    handle: object
    ptrs: torch.Tensor
    ready_epochs: torch.Tensor
    consumed_epochs: torch.Tensor
    flag_off: int
    consumed_off: int
    hidden: int
    world: int


def _flag_layout(hidden: int) -> tuple[int, int, int]:
    payload = _MAX_TOKENS * hidden * 2
    flag_off = (payload + 255) & ~255
    consumed_off = flag_off + _MAX_TOKENS * 4
    total = consumed_off + _MAX_TOKENS * 4
    return flag_off, consumed_off, total


def _heap(group: dist.ProcessGroup, device: torch.device, hidden: int, world: int) -> _Heap:
    key = (device, id(group), hidden, world)
    heap = _STATES.get(key)
    if heap is not None:
        return heap
    import torch.distributed._symmetric_memory as symm_mem

    # The default symmetric-memory backend maps these peers. Forcing MORI just
    # warns when that allocation backend is not installed.
    flag_off, consumed_off, total = _flag_layout(hidden)
    buf = symm_mem.empty(total, dtype=torch.uint8, device=device)
    buf.zero_()
    torch.cuda.synchronize()
    handle = symm_mem.rendezvous(buf, group=group)
    peers = [int(ptr) for ptr in handle.buffer_ptrs]
    if len(peers) != world or not all(peers):
        raise RuntimeError("symmetric heap did not map every rank")
    # buffer_ptrs_dev is a raw device address, not a tensor. The kernel loads
    # this copied table.
    ptrs = torch.tensor(peers, dtype=torch.uint64, device=device)
    ready_epochs = torch.zeros(_MAX_TOKENS, dtype=torch.int32, device=device)
    consumed_epochs = torch.zeros(_MAX_TOKENS, dtype=torch.int32, device=device)
    heap = _Heap(
        buf,
        handle,
        ptrs,
        ready_epochs,
        consumed_epochs,
        flag_off,
        consumed_off,
        hidden,
        world,
    )
    _STATES[key] = heap
    return heap


@triton.jit
def _signal_and_wait(
    ptrs,
    epochs_ptr,
    FLAG_OFF: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
):
    """Publish this row's flag and wait until every peer has published it.

    One program per token, one warp. A wide program that spins on a peer flag
    hangs on this stack, and a volatile reload inside the spin is hoisted, so
    the wait is an acquire atomic on a single lane.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, 1)
    old = tl.atomic_add(epochs_ptr + row + lane, 1)
    epoch = tl.sum(old) + 1
    base = tl.load(ptrs + RANK).to(tl.pointer_type(tl.uint8))
    flag = (base + FLAG_OFF + row * 4).to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(flag + lane, epoch, sem="release", scope="sys")
    for peer in tl.static_range(WORLD):
        peer_base = tl.load(ptrs + peer).to(tl.pointer_type(tl.uint8))
        peer_flag = (peer_base + FLAG_OFF + row * 4).to(tl.pointer_type(tl.int32))
        seen = tl.sum(tl.atomic_add(peer_flag + lane, 0, sem="acquire", scope="sys"))
        while seen != epoch:
            seen = tl.sum(
                tl.atomic_add(peer_flag + lane, 0, sem="acquire", scope="sys")
            )


@triton.jit
def _fused_kernel(
    residual_ptr,
    bank_ptr,
    cw_ptr,
    ow_ptr,
    hidden_ptr,
    residual_out_ptr,
    ptrs,
    stride_r,
    stride_bm,
    stride_bb,
    stride_h,
    stride_o,
    score_eps,
    out_eps,
    WORLD: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NVB: tl.constexpr,
    R_PAD: tl.constexpr,
):
    """One token row: sum the TP partials, then AttnRes + RMSNorm.

    Peer payloads are already published. This program does not spin or store
    flags; the one-warp signal kernels on either side of it do.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < HIDDEN

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for peer in tl.static_range(WORLD):
        peer_payload = tl.load(ptrs + peer).to(tl.pointer_type(tl.bfloat16))
        acc += tl.load(
            peer_payload + row * HIDDEN + offs,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.float32)
    # Round the sum to bf16 before the residual add, matching attn_res_hip.
    prefix = (
        acc.to(tl.bfloat16).to(tl.float32)
        + tl.load(residual_ptr + row * stride_r + offs, mask=mask, other=0).to(
            tl.float32
        )
    ).to(tl.bfloat16)
    tl.store(residual_out_ptr + row * stride_o + offs, prefix, mask=mask)
    pv = prefix.to(tl.float32)

    cw = tl.load(cw_ptr + offs, mask=mask, other=0.0)
    p_score = tl.sum(pv * cw) / tl.sqrt(tl.sum(pv * pv) / HIDDEN + score_eps)

    offs_r = tl.arange(0, R_PAD)
    mask_r = offs_r < NVB
    tile = tl.load(
        bank_ptr + row * stride_bm + offs_r[:, None] * stride_bb + offs[None, :],
        mask=mask_r[:, None] & mask[None, :],
        other=0.0,
    ).to(tl.float32)
    b_score = tl.sum(tile * cw[None, :], axis=1) / tl.sqrt(
        tl.sum(tile * tile, axis=1) / HIDDEN + score_eps
    )
    m = tl.maximum(tl.max(tl.where(mask_r, b_score, -float("inf"))), p_score)
    b_w = tl.where(mask_r, tl.exp(b_score - m), 0.0)
    p_w = tl.exp(p_score - m)
    inv = 1.0 / (tl.sum(b_w) + p_w)
    mixed = pv * (p_w * inv) + tl.sum(b_w[:, None] * inv * tile, axis=0)

    scale = 1.0 / tl.sqrt(tl.sum(mixed * mixed) / HIDDEN + out_eps)
    ow = tl.load(ow_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        hidden_ptr + row * stride_h + offs,
        (mixed * scale * ow).to(hidden_ptr.dtype.element_ty),
        mask=mask,
    )


def _launch_config(nvb: int) -> tuple[int, int]:
    from sglang.kernels.ops.attention.attn_res_hip import (
        _LAUNCH_TUNING,
        _USE_TUNED_LAUNCH,
    )

    r_pad = triton.next_power_of_2(nvb)
    if _USE_TUNED_LAUNCH:
        return _LAUNCH_TUNING[r_pad]
    return 4, 0


def allreduce_residual_attnres(
    partial: torch.Tensor,
    residual: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj,
    score_norm,
    out_norm,
    *,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``partial`` across ``group`` and finish the AttnRes epilogue.

    ``partial`` and ``residual`` are contiguous BF16 ``[M, H]``. ``bank`` is
    the replicated snapshot tensor ``[M, NB, H]``; rows ``0 .. nvb-1`` are live.
    Returns ``(normalized mixture, updated residual)``.
    """
    from sglang.kernels.ops.attention.attn_res_hip import supports_attn_res_hip

    if world_size not in _SUPPORTED_WORLD:
        raise RuntimeError(f"fused AR+AttnRes supports world {_SUPPORTED_WORLD}")
    rows = partial.shape[0]
    if rows > _MAX_TOKENS:
        raise RuntimeError(f"fused AR+AttnRes supports at most {_MAX_TOKENS} tokens")
    if nvb < 1 or not supports_attn_res_hip(partial.shape[1], nvb):
        raise RuntimeError(f"AttnRes shape H={partial.shape[1]} nvb={nvb} is unsupported")
    hidden = partial.shape[1]
    heap = _heap(group, partial.device, hidden, world_size)
    cw = get_cw(score_proj, score_norm)
    block_h = triton.next_power_of_2(hidden)
    num_warps, waves_per_eu = _launch_config(nvb)
    out = torch.empty_like(partial)
    residual_out = torch.empty_like(partial)
    # The copy lands before either flag kernel on this stream. The consumed
    # wait from the previous call is what makes the buffer free to overwrite.
    heap.buf[: rows * hidden * 2].view(torch.bfloat16).view(rows, hidden).copy_(partial)
    _signal_and_wait[(rows,)](
        heap.ptrs,
        heap.ready_epochs,
        FLAG_OFF=heap.flag_off,
        RANK=rank,
        WORLD=world_size,
        num_warps=1,
    )
    _fused_kernel[(rows,)](
        residual,
        bank,
        cw,
        out_norm.weight,
        out,
        residual_out,
        heap.ptrs,
        residual.stride(0),
        bank.stride(0),
        bank.stride(1),
        out.stride(0),
        residual_out.stride(0),
        score_norm.variance_epsilon,
        out_norm.variance_epsilon,
        WORLD=world_size,
        HIDDEN=hidden,
        BLOCK_H=block_h,
        NVB=nvb,
        R_PAD=triton.next_power_of_2(nvb),
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    _signal_and_wait[(rows,)](
        heap.ptrs,
        heap.consumed_epochs,
        FLAG_OFF=heap.consumed_off,
        RANK=rank,
        WORLD=world_size,
        num_warps=1,
    )
    return out, residual_out


def try_all_reduce_residual_attnres(
    partial: torch.Tensor,
    residual: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj,
    score_norm,
    out_norm,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(AttnRes RMSNorm output, residual + AR(partial))``, or None.

    None means the caller keeps the split all-reduce and AttnRes. Declines
    before any collective, so a rejected shape cannot deadlock the group.
    """
    if not enabled() or residual is None or out_norm is None:
        return None
    if torch.cuda.is_current_stream_capturing() and not _STATES:
        return None
    num_tokens = partial.shape[0]
    if (
        not covers(num_tokens)
        or nvb < 1
        or partial.ndim != 2
        or residual.shape != partial.shape
        or partial.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or not partial.is_contiguous()
        or not residual.is_contiguous()
        or bank.shape[0] != num_tokens
        or bank.shape[-1] != partial.shape[-1]
    ):
        return None
    from sglang.kernels.ops.attention.attn_res_hip import supports_attn_res_hip

    if not supports_attn_res_hip(partial.shape[1], nvb):
        return None
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = get_tp_group()
    except Exception:
        logger.debug("HIP AR+AttnRes: no tensor-parallel group", exc_info=True)
        return None
    pg = group.device_group
    world_size = dist.get_world_size(group=pg)
    if world_size not in _SUPPORTED_WORLD:
        return None
    # A rejected launch must raise on every rank. Swallowing it on one rank
    # leaves the others spinning on the flag.
    return allreduce_residual_attnres(
        partial,
        residual,
        bank,
        nvb,
        score_proj,
        score_norm,
        out_norm,
        group=pg,
        rank=dist.get_rank(group=pg),
        world_size=world_size,
    )
