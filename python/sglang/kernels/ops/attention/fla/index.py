# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    num_chunks = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    total_chunks = int(num_chunks.sum().item())
    seq_indices = torch.repeat_interleave(
        torch.arange(num_chunks.numel(), device=cu_seqlens.device),
        num_chunks,
        output_size=total_chunks,
    )
    seq_offsets = torch.cumsum(num_chunks, dim=0) - num_chunks
    chunk_indices = torch.arange(total_chunks, device=cu_seqlens.device)
    chunk_indices -= torch.repeat_interleave(
        seq_offsets, num_chunks, output_size=total_chunks
    )
    return torch.stack([seq_indices, chunk_indices], dim=1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
