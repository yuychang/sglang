"""Tune the B2/B4 Kimi-K3 cooperative preactivated MoE-front kernel.

This kernel is 1.13 ms / 92 calls (7.8% of the final c2 profile). It jointly
computes latent down, shared gate/up + SiTU, and router logits. Sweep its
persistent-grid launch parameters under graph replay while rotating enough
weights to exceed the last-level cache. Only bit-identical candidates are timed.
"""

import argparse
import itertools
import sys

import torch

HIDDEN = 7168
ROUTED = 3584
SHARED = 1536
SHARED_OUT = SHARED // 2
ROUTER = 896
FP8_MAX = 448.0


def pack_rows(weight):
    w = weight.float()
    amax = w.abs().amax(dim=1).clamp(min=1e-8)
    scale = (amax / FP8_MAX).contiguous()
    packed = (
        (w / scale[:, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return packed, scale


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
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--wpb", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--wcm", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--cu", nargs="+", type=int, default=[256])
    parser.add_argument("--wpe", nargs="+", type=int, default=[3])
    parser.add_argument("--weight-copies", type=int, default=6)
    args = parser.parse_args()

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
    from sglang.kernels.ops.kimi_k3.flydsl.kernels.kimi_k3_tri_projection_multitoken_gfx950 import (
        build_kimi_k3_multitoken_tri_projection_module,
    )

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    copies = []
    for _ in range(args.weight_copies):
        rw, rs = pack_rows(
            torch.randn(ROUTED, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        sw, ss = pack_rows(
            torch.randn(SHARED, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        gw = (
            torch.randn(ROUTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
        ).contiguous()
        copies.append((rw, rs, sw, ss, gw))

    print(torch.cuda.get_device_name(0), f"rotating {len(copies)} weight sets")
    for tokens in args.tokens:
        hidden = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        routed_out = torch.empty(tokens, ROUTED, device="cuda", dtype=torch.bfloat16)
        shared_out = torch.empty(
            tokens, SHARED_OUT, device="cuda", dtype=torch.bfloat16
        )
        router_out = torch.empty(tokens, ROUTER, device="cuda", dtype=torch.float32)
        hidden_arg = ptr_arg(hidden)
        out_args = tuple(map(ptr_arg, (routed_out, shared_out, router_out)))

        def build(wpb, wcm, cu, wpe):
            return build_kimi_k3_multitoken_tri_projection_module(
                num_tokens=tokens,
                token_tile=tokens,
                cu_count=cu,
                waves_per_block=wpb,
                waves_per_eu=wpe,
                weight_cache_modifier=wcm,
                interleaved_shared_pairs=True,
                fast_situ=True,
                situ_beta=4.0,
                situ_linear_beta=25.0,
            )

        base = build(8, 3, 256, 3)
        base(
            hidden_arg,
            *map(ptr_arg, copies[0]),
            *out_args,
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        reference = (
            routed_out.clone(),
            shared_out.clone(),
            router_out.clone(),
        )

        print(f"\n=== B{tokens} ===")
        print(
            f"{'wpb':>3} {'wcm':>3} {'cu':>3} {'wpe':>3} " f"{'us':>8} {'bitwise':>8}"
        )
        rows = []
        for wpb, wcm, cu, wpe in itertools.product(
            args.wpb, args.wcm, args.cu, args.wpe
        ):
            launcher = build(wpb, wcm, cu, wpe)
            launcher(
                hidden_arg,
                *map(ptr_arg, copies[0]),
                *out_args,
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            identical = all(
                torch.equal(out, ref)
                for out, ref in zip((routed_out, shared_out, router_out), reference)
            )
            if not identical:
                print(
                    f"{wpb:3d} {wcm:3d} {cu:3d} {wpe:3d} " f"{'--':>8} {str(False):>8}"
                )
                continue

            copy_args = [tuple(map(ptr_arg, c)) for c in copies]

            def body():
                for weight_args in copy_args:
                    launcher(
                        hidden_arg,
                        *weight_args,
                        *out_args,
                        stream=torch.cuda.current_stream(),
                    )

            us = graph_time(body, len(copies))
            rows.append((us, wpb, wcm, cu, wpe))
            print(f"{wpb:3d} {wcm:3d} {cu:3d} {wpe:3d} " f"{us:8.2f} {str(True):>8}")

        rows.sort()
        baseline = next(r for r in rows if r[1:] == (8, 3, 256, 3))
        best = rows[0]
        print(
            f"best B{tokens}: {best[0]:.2f} us "
            f"(wpb={best[1]}, wcm={best[2]}, cu={best[3]}, wpe={best[4]}), "
            f"base {baseline[0]:.2f}, {baseline[0]/best[0]:.3f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
