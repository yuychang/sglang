"""Tune Kimi-K3's single-CTA ROCm attention-residual aggregation.

The final c2 profile spends 1.48 ms across 186 `_agg_kernel` launches.  The
shipping launch fixes `num_warps=4` for every valid-bank depth even though the
register tile grows from 1x8192 to 8x8192.  Sweep launch occupancy under graph
replay and reject any configuration that changes the BF16 result.
"""

import argparse
import sys

import torch
import triton

HIDDEN = 7168
MAX_BANK = 8


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
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--nvb", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-warps", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--waves-per-eu", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    from sglang.kernels.ops.kimi_k3.attn_res_hip import _agg_kernel

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    T, H = args.tokens, HIDDEN
    prefix = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    addend = torch.randn_like(prefix)
    prefix_out = torch.empty_like(prefix)
    bank = torch.randn(T, MAX_BANK, H, device="cuda", dtype=torch.bfloat16)
    cw = torch.randn(H, device="cuda", dtype=torch.float32)
    ow = torch.randn(H, device="cuda", dtype=torch.bfloat16)

    def launch(out, nvb, num_warps, waves_per_eu):
        _agg_kernel[(T,)](
            prefix,
            addend,
            prefix_out,
            bank,
            cw,
            ow,
            out,
            1e-5,
            1e-5,
            prefix.stride(0),
            addend.stride(0),
            prefix_out.stride(0),
            bank.stride(0),
            bank.stride(1),
            out.stride(0),
            H=H,
            BLOCK_H=triton.next_power_of_2(H),
            NVB=nvb,
            R_PAD=triton.next_power_of_2(nvb),
            HAS_ADD=True,
            WRITE_BANK=False,
            APPLY_OUT_NORM=True,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )

    print(torch.cuda.get_device_name(0), f"T={T}, H={H}")
    for nvb in args.nvb:
        ref = torch.empty_like(prefix)
        launch(ref, nvb, 4, 0)
        torch.cuda.synchronize()
        print(f"\n=== nvb={nvb} ===")
        print(f"{'warps':>5} {'wpe':>4} {'us':>8} {'bitwise':>8} {'vs base':>8}")
        rows = []
        for warps in args.num_warps:
            for wpe in args.waves_per_eu:
                out = torch.empty_like(prefix)
                launch(out, nvb, warps, wpe)
                torch.cuda.synchronize()
                identical = torch.equal(out, ref)
                if not identical:
                    print(f"{warps:5d} {wpe:4d} {'--':>8} {str(False):>8}")
                    continue

                us = graph_time(lambda: launch(out, nvb, warps, wpe))
                rows.append((us, warps, wpe))
                print(f"{warps:5d} {wpe:4d} {us:8.2f} {str(True):>8}")
        rows.sort()
        base = next(r for r in rows if r[1:] == (4, 0))
        best = rows[0]
        print(
            f"best nvb={nvb}: {best[0]:.2f} us "
            f"(warps={best[1]}, wpe={best[2]}), "
            f"base {base[0]:.2f}, {base[0]/best[0]:.3f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
