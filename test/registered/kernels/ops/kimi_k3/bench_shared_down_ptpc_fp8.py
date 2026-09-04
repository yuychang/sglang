#!/usr/bin/env python3
"""Fused PTPC shared-down (one launch) vs `per_token_quant_hip` + GEMM (two).

Answers, per decode bucket, whether folding the activation quant into the
projection is worth owning that bucket. Both sides are timed inside a CUDA
graph so launch overhead is counted the way the decoder sees it.

    python python/sglang/test/registered/kernels/ops/kimi_k3/\
bench_shared_down_ptpc_fp8.py
"""

from __future__ import annotations

import sys

import torch

HIDDEN = 7168
INTERMEDIATE = 768
BUCKETS = (1, 2, 4, 8, 16, 32)
WARMUP = 20
ITERS = 200


def _time_graph(graph: torch.cuda.CUDAGraph) -> float:
    stream = torch.cuda.current_stream()
    for _ in range(WARMUP):
        graph.replay()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    stream.synchronize()
    return start.elapsed_time(end) * 1e3 / ITERS


def main() -> int:
    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    from sglang.kernels.ops.kimi_k3 import ptpc_fp8_aiter_hip
    from sglang.kernels.ops.kimi_k3.flydsl.shared_down_ptpc_fp8 import (
        BUILDABLE_TOKEN_BATCHES,
        is_available,
        kimi_k3_shared_down_ptpc_fp8,
        quantize_shared_down_weight,
    )

    if not is_available():
        print("fused PTPC shared-down unavailable (needs gfx950 + FlyDSL)")
        return 1
    if not ptpc_fp8_aiter_hip.available():
        print("aiter PTPC two-launch path unavailable")
        return 1

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(17)
    weight = (
        torch.randn((HIDDEN, INTERMEDIATE), generator=generator)
        .mul_(INTERMEDIATE**-0.5)
        .bfloat16()
        .to(device)
    )
    fused_w, fused_s = quantize_shared_down_weight(weight)
    packed_w, packed_s, logical_n = ptpc_fp8_aiter_hip.pack(weight)

    print(f"{'M':>4} {'fused us':>10} {'split us':>10} {'delta':>9} {'rel err':>9}")
    for num_tokens in BUCKETS:
        x = (
            torch.randn((num_tokens, INTERMEDIATE), generator=generator)
            .bfloat16()
            .to(device)
        )
        out_fused = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        out_split = torch.empty_like(out_fused)

        buildable = num_tokens in BUILDABLE_TOKEN_BATCHES
        if buildable:
            kimi_k3_shared_down_ptpc_fp8(
                x,
                fused_w,
                fused_s,
                out=out_fused,
                token_buckets=(num_tokens,),
            )
        ptpc_fp8_aiter_hip.warmup(
            packed_w, packed_s, logical_n, INTERMEDIATE, token_buckets=(num_tokens,)
        )
        ptpc_fp8_aiter_hip.run(x, packed_w, packed_s, logical_n, out=out_split)
        torch.cuda.synchronize()

        reference = out_split.float()
        rel = (
            ((out_fused.float() - reference).norm() / reference.norm()).item()
            if buildable
            else float("nan")
        )

        split_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(split_graph):
            ptpc_fp8_aiter_hip.run(x, packed_w, packed_s, logical_n, out=out_split)
        split_us = _time_graph(split_graph)

        if not buildable:
            print(f"{num_tokens:>4} {'n/a':>10} {split_us:>10.2f} {'-':>9} {'-':>9}")
            continue

        fused_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fused_graph):
            kimi_k3_shared_down_ptpc_fp8(
                x,
                fused_w,
                fused_s,
                out=out_fused,
                token_buckets=(num_tokens,),
            )
        fused_us = _time_graph(fused_graph)

        delta = split_us - fused_us
        print(
            f"{num_tokens:>4} {fused_us:>10.2f} {split_us:>10.2f} "
            f"{delta:>+9.2f} {rel:>9.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
