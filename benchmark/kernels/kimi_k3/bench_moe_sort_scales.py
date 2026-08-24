"""Measure AITER's mxfp4 MoE sort_scales launch at Kimi-K3 decode shapes.

This kernel runs once per MoE layer -- 92 times per decode step -- and the c2
profile puts it at 4.0 us a call. It shuffles the per-token activation scales
into the sorted layout gemm1 wants, which at two tokens is a couple of hundred
kilobytes: nowhere near 4 us of work. This times it in isolation to separate the
work from the launch.

Timed under graph replay, since that is how decode executes it.
"""

import argparse
import sys

import torch

D_HIDDEN = 3584  # Kimi-K3 routed latent width
NUM_EXPERTS = 896
TOPK = 16
BLOCK_M = 32


def sorted_extent(m):
    """max_sorted as aiter.fused_moe._k3_a8w4_fused_sort_quant computes it."""
    active = min(NUM_EXPERTS, m * TOPK)
    return (((m * TOPK + active * (BLOCK_M - 1)) + BLOCK_M - 1) // BLOCK_M) * BLOCK_M


def make_inputs(m, device="cuda"):
    max_sorted = sorted_extent(m)
    cols = D_HIDDEN // 32
    a_scale = torch.randint(0, 255, (m, cols), device=device, dtype=torch.uint8)
    sorted_token_ids = torch.randint(
        0, m, (max_sorted,), device=device, dtype=torch.int32
    )
    cumsum = torch.tensor([m * TOPK, 0], device=device, dtype=torch.int32)
    out = torch.empty((max_sorted, cols), device=device, dtype=torch.uint8)
    return a_scale, sorted_token_ids, cumsum, out, max_sorted


def graph_time(fn, reps=32, iters=50, warmup=20):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(3):
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters / reps * 1000.0)
    del graph
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4, 8, 32, 64])
    args = parser.parse_args()

    import aiter

    torch.cuda.set_device(0)
    print(
        f"{torch.cuda.get_device_name(0)}  D_HIDDEN={D_HIDDEN} "
        f"NE={NUM_EXPERTS} TOPK={TOPK} BM={BLOCK_M}\n"
    )
    print(
        f"{'tokens':>7} {'max_sorted':>11} {'out KB':>8} {'us':>8} "
        f"{'GB/s':>8} {'blocks needed':>14}"
    )

    for m in args.tokens:
        a_scale, sorted_ids, cumsum, out, max_sorted = make_inputs(m)

        def call():
            aiter.mxfp4_moe_sort_scales(
                a_scale=a_scale,
                sorted_token_ids=sorted_ids,
                cumsum_tensor=cumsum,
                a_scale_sorted_shuffled=out,
                NE=NUM_EXPERTS,
                TOPK=TOPK,
                D_HIDDEN=D_HIDDEN,
                MB=BLOCK_M,
                max_sorted=max_sorted,
            )

        us = graph_time(call)
        kb = out.numel() / 1024
        # The kernel writes one dword per work item; 1024 threads per block.
        work_items = out.numel() // 4
        blocks = (work_items + 1023) // 1024
        gbps = out.numel() * 2 / (us * 1e-6) / 1e9  # read + write
        print(
            f"{m:>7} {max_sorted:>11} {kb:8.1f} {us:8.2f} {gbps:8.1f} " f"{blocks:>14}"
        )

    print("\nkNCtasSort is 512 blocks x 1024 threads regardless of token count.")


if __name__ == "__main__":
    sys.exit(main())
