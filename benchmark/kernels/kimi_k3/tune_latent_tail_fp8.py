"""Tune multi-token Kimi-K3 fused FP8 latent-tail launch parameters on gfx950.

The B2/B4 kernels inherited B1's `(rows_per_wave=1, cu_count=240,
waves_per_eu=2, weight_cache_modifier=2)` schedule.  This sweep times the same
kernel under graph replay while rotating over enough 25.7 MB FP8 weights to
overflow the last-level cache, matching the 92 distinct layer weights used by
decode.

Every candidate is compared bitwise to the shipping schedule before its timing
is accepted.
"""

import argparse
import itertools
import sys

import torch

LATENT = 3584
HIDDEN = 7168
EPS = 1e-5


def graph_time(fn, reps, iters=30, warmup=10):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
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
    parser.add_argument("--tokens", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--rows-per-wave", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--cache-modifier", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--cu-count", nargs="+", type=int, default=[240])
    parser.add_argument("--waves-per-eu", nargs="+", type=int, default=[2])
    parser.add_argument("--weight-copies", type=int, default=12)
    args = parser.parse_args()

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
    from sglang.kernels.ops.kimi_k3.flydsl.kernels.latent_moe_tail_fp8_gfx950 import (
        build_latent_moe_tail_fp8_persistent_module,
    )
    from sglang.kernels.ops.kimi_k3.flydsl.latent_moe_tail_fp8 import (
        quantize_latent_moe_tail_weight,
    )

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    source = torch.randn(HIDDEN, LATENT, device="cuda", dtype=torch.bfloat16) * 0.02
    packed, scale = quantize_latent_moe_tail_weight(source)
    weights = [packed] + [packed.clone() for _ in range(args.weight_copies - 1)]
    rms_weight = torch.randn(LATENT, device="cuda", dtype=torch.bfloat16) * 0.1 + 1

    print(torch.cuda.get_device_name(0))
    print(f"rotating {len(weights)} x {packed.numel()/1e6:.1f} MB FP8 weights\n")

    for tokens in args.tokens:
        routed = torch.randn(tokens, LATENT, device="cuda", dtype=torch.bfloat16)
        shared = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(shared)
        base_out = torch.empty_like(shared)
        scale_arg = ptr_arg(scale)
        routed_arg = ptr_arg(routed)
        shared_arg = ptr_arg(shared)
        rms_arg = ptr_arg(rms_weight)

        def build(rpw, cu, wpe, wcm):
            return build_latent_moe_tail_fp8_persistent_module(
                num_tokens=tokens,
                rows_per_wave=rpw,
                cu_count=cu,
                waves_per_eu=wpe,
                weight_cache_modifier=wcm,
            )

        base = build(1, 240, 2, 2)
        base(
            routed_arg,
            shared_arg,
            rms_arg,
            ptr_arg(weights[0]),
            scale_arg,
            shared_arg,
            ptr_arg(base_out),
            EPS,
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()

        print(f"=== B{tokens} ===")
        print(
            f"{'rpw':>3} {'wcm':>3} {'cu':>3} {'wpe':>3} "
            f"{'us':>8} {'bitwise':>8} {'speedup':>8}"
        )
        results = []
        for rpw, wcm, cu, wpe in itertools.product(
            args.rows_per_wave,
            args.cache_modifier,
            args.cu_count,
            args.waves_per_eu,
        ):
            launcher = build(rpw, cu, wpe, wcm)
            launcher(
                routed_arg,
                shared_arg,
                rms_arg,
                ptr_arg(weights[0]),
                scale_arg,
                shared_arg,
                ptr_arg(out),
                EPS,
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            identical = torch.equal(out, base_out)
            if not identical:
                print(
                    f"{rpw:3d} {wcm:3d} {cu:3d} {wpe:3d} "
                    f"{'--':>8} {str(False):>8} {'--':>8}"
                )
                continue

            weight_args = [ptr_arg(w) for w in weights]
            out_arg = ptr_arg(out)

            def body():
                for weight_arg in weight_args:
                    launcher(
                        routed_arg,
                        shared_arg,
                        rms_arg,
                        weight_arg,
                        scale_arg,
                        shared_arg,
                        out_arg,
                        EPS,
                        stream=torch.cuda.current_stream(),
                    )

            us = graph_time(body, len(weights))
            results.append((us, rpw, wcm, cu, wpe))
            print(f"{rpw:3d} {wcm:3d} {cu:3d} {wpe:3d} " f"{us:8.2f} {str(True):>8}")

        results.sort()
        base_row = next(r for r in results if r[1:] == (1, 2, 240, 2))
        winner = results[0]
        print(
            f"best B{tokens}: {winner[0]:.2f} us "
            f"(rpw={winner[1]}, wcm={winner[2]}, cu={winner[3]}, "
            f"wpe={winner[4]}), shipping {base_row[0]:.2f} us, "
            f"{base_row[0]/winner[0]:.3f}x\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
