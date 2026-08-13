# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Split-KV reduce for aiter's MLA decode, emitting per-(token, head) LSE."""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dcp_kernels import create_mla_kv_page_table_for_dcp
from sglang.kernels.ops.kvcache.kv_indices import (
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)


@triton.jit
def _dcp_ragged_to_block_table_kernel(
    kv_indices_ptr,
    kv_indptr_ptr,
    dest_ptr,
    dest_stride0: tl.int64,
    MAX_COLS: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(kv_indptr_ptr + req)
    n = tl.load(kv_indptr_ptr + req + 1) - start
    cols = tl.arange(0, MAX_COLS)
    mask = cols < n
    vals = tl.load(kv_indices_ptr + start + cols, mask=mask, other=0)
    tl.store(dest_ptr + req * dest_stride0 + cols, vals, mask=mask)


def build_dcp_block_table(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
    max_cols: int,
):
    """Scatter a ragged shard into a block-size-one 2D block table."""
    block_tables = torch.zeros(
        bs, max_cols, dtype=torch.int32, device=kv_indices.device
    )
    _dcp_ragged_to_block_table_kernel[(bs,)](
        kv_indices,
        kv_indptr,
        block_tables,
        block_tables.stride(0),
        MAX_COLS=triton.next_power_of_2(max_cols),
    )
    return block_tables


def build_dcp_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    local_kv_lens: torch.Tensor,
    bs: int,
    max_pages: int,
    page_size: int,
    dcp_size: int,
    dcp_rank: int,
    out: Optional[torch.Tensor] = None,
):
    """Build this rank's page table for AITER MLA decode."""
    if out is None:
        out = torch.zeros(
            bs, max_pages, dtype=torch.int32, device=req_to_token.device
        )
    if max_pages == 0:
        return out
    pages_per_block = get_num_page_per_block_flashmla(page_size)
    create_mla_kv_page_table_for_dcp[
        (bs, get_num_kv_index_blocks_flashmla(max_pages, page_size))
    ](
        req_to_token,
        req_pool_indices,
        local_kv_lens,
        out,
        req_to_token.stride(0),
        out.stride(0),
        PHYSICAL_PAGE_SIZE=page_size,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        PAGES_PER_BLOCK=pages_per_block,
    )
    return out


@triton.jit
def _pack_dcp_kv_pages_kernel(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    dst_idx_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    src = tl.load(src_idx_ptr + row).to(tl.int64)
    dst = tl.load(dst_idx_ptr + row).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    tl.store(
        dst_ptr + dst * D + offs,
        tl.load(src_ptr + src * D + offs, mask=mask),
        mask=mask,
    )


def pack_dcp_kv_into_pages(
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
    page_size: int,
):
    """Repack assembled per-forward KV into per-request page-aligned storage."""
    _, _, dim = kv_buffer.shape[0], kv_buffer.shape[1], kv_buffer.shape[-1]
    device = kv_buffer.device
    lens = (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(torch.int64)
    pages = (lens + page_size - 1) // page_size
    page_start = torch.cumsum(pages, dim=0) - pages
    shift = page_start * page_size - kv_indptr[:bs].to(torch.int64)
    n_tokens = int(kv_indptr[bs].item())
    dst_idx = torch.repeat_interleave(shift, lens) + torch.arange(
        n_tokens, device=device, dtype=torch.int64
    )

    n_pages = int(pages.sum().item())
    paged = torch.zeros(
        (n_pages * page_size, 1, dim), dtype=kv_buffer.dtype, device=device
    )
    if n_tokens > 0:
        _pack_dcp_kv_pages_kernel[(n_tokens,)](
            kv_buffer,
            paged,
            kv_indices,
            dst_idx,
            D=dim,
            BLOCK=triton.next_power_of_2(dim),
        )
    max_pages = int(pages.max().item()) if bs > 0 else 0
    block_tables = page_start[:, None] + torch.arange(
        max_pages, device=device, dtype=torch.int64
    )
    return paged.view(-1, page_size, 1, dim), block_tables.to(torch.int32)


@triton.jit
def _dcp_mla_reduce_kernel(
    out_ptr,
    lse_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_query_heads: tl.constexpr,
    out_stride0: tl.int64,
    out_stride1: tl.int64,
    lse_stride0: tl.int64,
    TILE_SIZE: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + tok)
    tiles_per_segment = tl.cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = tl.cdiv(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < act_num_segments

    seg_off = (
        tok.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + head * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(
        segm_max_ptr + seg_off, mask=segm_mask, other=float("-inf")
    )
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(
        segm_expsum_ptr + seg_off, mask=segm_mask, other=0.0
    )
    segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    out_off = (
        tok.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + head * (NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * KV_LORA_RANK
        + tl.arange(0, KV_LORA_RANK)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + out_off, mask=segm_mask[:, None], other=0.0
    )
    segm_output = segm_output * tl.math.exp2(segm_max - overall_max)[:, None]
    acc = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc / overall_expsum)

    lse = tl.where(
        overall_expsum == 0.0,
        float("-inf"),
        overall_max + tl.log2(overall_expsum),
    )

    tl.store(
        out_ptr
        + tok * out_stride0
        + head * out_stride1
        + tl.arange(0, KV_LORA_RANK),
        acc.to(out_ptr.type.element_ty),
    )
    tl.store(lse_ptr + tok * lse_stride0 + head, lse)


def dcp_mla_reduce(
    segm_output: torch.Tensor,
    segm_max: torch.Tensor,
    segm_expsum: torch.Tensor,
    seq_lens: torch.Tensor,
    tile_size: int,
    out_dtype: torch.dtype,
):
    """Reduce AITER split-KV partials and return output plus base-2 LSE."""
    num_tokens, num_heads, num_segments, kv_lora_rank = segm_output.shape
    out = torch.empty(
        num_tokens,
        num_heads,
        kv_lora_rank,
        dtype=out_dtype,
        device=segm_output.device,
    )
    lse = torch.empty(
        num_tokens,
        num_heads,
        dtype=torch.float32,
        device=segm_output.device,
    )
    _dcp_mla_reduce_kernel[(num_tokens, num_heads)](
        out,
        lse,
        segm_output,
        segm_max,
        segm_expsum,
        seq_lens,
        num_query_heads=num_heads,
        out_stride0=out.stride(0),
        out_stride1=out.stride(1),
        lse_stride0=lse.stride(0),
        TILE_SIZE=tile_size,
        KV_LORA_RANK=kv_lora_rank,
        NUM_SEGMENTS_PER_SEQ=num_segments,
    )
    return out, lse
