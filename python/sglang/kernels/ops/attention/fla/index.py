# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Sequence

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _chunk_indices_from_lens(lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


def seed_chunk_metadata_from_host(
    cu_seqlens: torch.Tensor,
    seq_lens_cpu: Sequence[int],
    chunk_size: int,
    pin_memory: bool,
) -> None:
    """Fill the chunk-index caches for ``cu_seqlens`` from host lengths.

    Computing them from the device tensor costs a D2H ``.tolist()`` and two
    pageable H2D copies, each of which waits for every kernel queued ahead,
    including an overlapped previous step. ``seq_lens_cpu`` must equal
    ``cu_seqlens[1:] - cu_seqlens[:-1]``; later calls with the same
    ``cu_seqlens`` object then hit the cache.
    """
    if len(seq_lens_cpu) == 0 or len(seq_lens_cpu) + 1 != cu_seqlens.shape[0]:
        return
    lens = torch.tensor(list(seq_lens_cpu), dtype=cu_seqlens.dtype)
    indices = _chunk_indices_from_lens(lens, chunk_size).to(cu_seqlens.dtype)
    offsets = torch.cat([lens.new_tensor([0]), triton.cdiv(lens, chunk_size)]).cumsum(
        -1
    )
    if pin_memory:
        indices = indices.pin_memory()
        offsets = offsets.pin_memory()
    device = cu_seqlens.device
    prepare_chunk_indices.seed(
        indices.to(device, non_blocking=pin_memory), cu_seqlens, chunk_size
    )
    prepare_chunk_offsets.seed(
        offsets.to(device, non_blocking=pin_memory), cu_seqlens, chunk_size
    )
