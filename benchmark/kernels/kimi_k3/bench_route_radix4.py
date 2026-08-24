"""Measure the Kimi-K3 router top-k (route_radix4) at decode shapes.

One launch per MoE layer, 92 per decode step, measured at 6.9 us a call in the
c2 profile against a ~1.7 us graph-replay floor. It selects the top 16 of 896
experts for each token, so the data is ~1.8 KB per token: the cost is the
dependent chain of block-wide radix passes, not bandwidth or dispatch.

The kernel is launched one block per token with kRadix4Block threads. That
constant sets both the per-thread value count (VPT = ceil(896/BLOCK)) and the
number of waves that must be merged through LDS each pass (NWAVE = BLOCK/64), so
it trades per-thread work against synchronization depth. At BLOCK=64 there is a
single wave and no __syncthreads() at all.

Timed under graph replay, since that is how decode executes it.
"""

import argparse
import sys

import torch

EXPERTS = 896
TOPK = 16


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

    from sglang.kernels.ops.moe.moe_route_radix4 import route_radix4

    torch.cuda.set_device(0)
    print(f"{torch.cuda.get_device_name(0)}  experts={EXPERTS} topk={TOPK}")
    print(
        "kRadix4Block is a compile-time constant in "
        "jit/csrc/moe/route_radix4_hip.cuh; rebuild to change it.\n"
    )
    print(f"{'tokens':>7} {'us':>8} {'us/token':>9}")

    torch.manual_seed(0)
    for m in args.tokens:
        scores = torch.randn(m, EXPERTS, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(EXPERTS, device="cuda", dtype=torch.bfloat16)

        # Warm the op once so any lazy JIT load is outside the timed region.
        route_radix4(scores, bias, TOPK, True, 1.0)
        torch.cuda.synchronize()

        us = graph_time(lambda: route_radix4(scores, bias, TOPK, True, 1.0))
        print(f"{m:>7} {us:8.2f} {us / m:9.2f}")


if __name__ == "__main__":
    sys.exit(main())
