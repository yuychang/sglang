"""Validate and time the multi-token Kimi-K3 fused FP8 latent tail.

The fused kernel computes, for M in {1,2,4},

    shared + Linear_FP8(RMSNorm(routed))

in one launch.  Its persistent waves reuse each FP8 weight load across all M
token accumulators, so weight traffic is independent of batch size.

Validation compares B2/B4 bitwise against running the established B1 kernel
once per token with the same packed weight.  That isolates the schedule change:
both sides use the same RMS order, FP8 representation, BF16 linear boundary and
shared add.  Timing uses graph replay and compares against the B1-per-token
construction; end-to-end serving still decides whether a bucket ships.
"""

import argparse
import sys

import torch

LATENT = 3584
HIDDEN = 7168
EPS = 1e-5


def graph_time(fn, reps=8, iters=30, warmup=10):
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
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4])
    args = parser.parse_args()

    from sglang.kernels.ops.kimi_k3.flydsl.latent_moe_tail_fp8 import (
        latent_moe_tail_fp8,
        quantize_latent_moe_tail_weight,
        supports_latent_moe_tail_fp8,
    )

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    weight = torch.randn(HIDDEN, LATENT, device="cuda", dtype=torch.bfloat16) * 0.02
    rms_weight = torch.randn(LATENT, device="cuda", dtype=torch.bfloat16) * 0.1 + 1
    packed, scale = quantize_latent_moe_tail_weight(weight)

    print(torch.cuda.get_device_name(0))
    print(
        f"{'M':>3} {'covered':>8} {'bitwise B1':>11} {'max diff':>10} "
        f"{'fused us':>9} {'B1 x M us':>10} {'speedup':>8}"
    )
    for m in args.tokens:
        routed = torch.randn(m, LATENT, device="cuda", dtype=torch.bfloat16)
        shared = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        covered = supports_latent_moe_tail_fp8(
            routed, shared, rms_weight, packed, scale, EPS
        )
        if not covered:
            print(f"{m:>3} {str(covered):>8}")
            continue

        fused = latent_moe_tail_fp8(routed, shared, rms_weight, packed, scale, EPS)
        per_token = torch.empty_like(shared)
        for token in range(m):
            latent_moe_tail_fp8(
                routed[token : token + 1],
                shared[token : token + 1],
                rms_weight,
                packed,
                scale,
                EPS,
                out=per_token[token : token + 1],
            )
        torch.cuda.synchronize()
        identical = torch.equal(fused, per_token)
        max_diff = (fused.float() - per_token.float()).abs().max().item()

        out = torch.empty_like(shared)

        def run_fused():
            latent_moe_tail_fp8(routed, shared, rms_weight, packed, scale, EPS, out=out)

        def run_b1_loop():
            for token in range(m):
                latent_moe_tail_fp8(
                    routed[token : token + 1],
                    shared[token : token + 1],
                    rms_weight,
                    packed,
                    scale,
                    EPS,
                    out=out[token : token + 1],
                )

        fused_us = graph_time(run_fused)
        b1_us = graph_time(run_b1_loop)
        print(
            f"{m:>3} {str(covered):>8} {str(identical):>11} "
            f"{max_diff:10.3e} {fused_us:9.2f} {b1_us:10.2f} "
            f"{b1_us / fused_us:7.2f}x"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
