"""Numerical check for the fused AR + residual + AttnRes + RMSNorm kernel.

Launch with torchrun so every rank maps the symmetric heap:

    torchrun --nproc_per_node=2 test/manual/kernels/test_hip_ar_attnres.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
from sglang.srt.layers.attn_residual import get_cw
from sglang.srt.layers.communication.hip_ar_attnres import (
    _STATES,
    allreduce_residual_attnres,
)


class _Param:
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.weight = weight
        self.variance_epsilon = eps


def _same(shape, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=g)


def _check(partial, residual, bank, nvb, score_proj, score_norm, out_norm, group) -> None:
    rank = dist.get_rank()
    world = dist.get_world_size()
    print(
        f"rank {rank} launch M={partial.shape[0]} H={partial.shape[1]} nvb={nvb}",
        flush=True,
    )
    out, residual_out = allreduce_residual_attnres(
        partial,
        residual,
        bank,
        nvb,
        score_proj,
        score_norm,
        out_norm,
        group=group,
        rank=rank,
        world_size=world,
    )
    # Second call reuses the heap. A missed consumed-flag would hang here.
    out2, residual_out2 = allreduce_residual_attnres(
        partial,
        residual,
        bank,
        nvb,
        score_proj,
        score_norm,
        out_norm,
        group=group,
        rank=rank,
        world_size=world,
    )
    gathered = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(gathered, partial)
    reduced = torch.stack([p.float() for p in gathered]).sum(0).to(torch.bfloat16)
    cw = get_cw(score_proj, score_norm)
    ref = torch.empty_like(reduced)
    ref_residual = torch.empty_like(reduced)
    attn_res_hip(
        reduced,
        bank,
        cw,
        out_norm.weight,
        ref,
        nvb,
        score_norm.variance_epsilon,
        out_norm.variance_epsilon,
        addend=residual,
        prefix_out=ref_residual,
    )
    for name, got in (
        ("out", out),
        ("residual", residual_out),
        ("out2", out2),
        ("residual2", residual_out2),
    ):
        reference = ref_residual if name.startswith("residual") else ref
        diff = (got.float() - reference.float()).abs()
        max_diff = diff.max().item()
        print(
            f"rank {rank} M={partial.shape[0]} H={partial.shape[1]} nvb={nvb} "
            f"{name} max_diff={max_diff:.5f}",
            flush=True,
        )
        if max_diff != 0:
            raise SystemExit(f"{name} diverged by {max_diff}")


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    group = dist.group.WORLD
    cases = [(1, 128, 1), (7, 128, 4), (16, 128, 8), (4, 7168, 2)]
    if os.environ.get("HIP_AR_ATTNRES_SMALL") == "1":
        cases = cases[:1]
    for m, h, nvb in cases:
        partial = _same((m, h), device, seed=1000 + rank)
        residual = _same((m, h), device, seed=7)
        bank = _same((m, nvb, h), device, seed=11)
        score_proj = _Param(_same((1, h), device, seed=3))
        score_norm = _Param(_same((h,), device, seed=5))
        out_norm = _Param(_same((h,), device, seed=9))
        _check(partial, residual, bank, nvb, score_proj, score_norm, out_norm, group)
        dist.barrier()
    if rank == 0:
        print(f"ok heap_keys={len(_STATES)}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
