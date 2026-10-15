# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def chunk_indices_from_lens(lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)


def chunk_offsets_from_lens(lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat([lens.new_tensor([0]), triton.cdiv(lens, chunk_size)]).cumsum(-1)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return chunk_indices_from_lens(prepare_lens(cu_seqlens), chunk_size).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return chunk_offsets_from_lens(prepare_lens(cu_seqlens), chunk_size)
