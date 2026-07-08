"""Microbenchmark for the ROCm-native multi-stream MXFP4 MoE overlap used by
``amd/Kimi-K2.5-MXFP4`` (DeepSeek-style MoE with 1 shared expert).

It measures three things at Kimi-K2.5 decode shapes:

1. The fused combine kernels themselves
   (``rocm_mxfp4_moe_add_shared`` = P0, ``rocm_mxfp4_moe_finalize_fuse_shared``
   = P1) across token counts, for whichever backend is active
   (sgl_kernel HIP -> Triton -> torch).

2. The end-to-end MoE-block *schedule* wall time comparing:
     * single-stream : shared -> routed -> combine (serialized)
     * multi-stream  : shared (secondary HIP stream) || routed (main) + combine
   using the real ``RocmMoeStreamState`` manager. The routed/shared experts are
   modeled with BF16 GEMMs sized like the MXFP4 GEMMs so the benchmark runs even
   without AITER; on a real deployment the AITER MXFP4 kernels replace them.

3. Max error of the multi-stream combine vs the single-stream reference.

This is a *scheduling / kernel* microbenchmark. For a full model measurement use
the end-to-end command printed at the bottom + rocprof to confirm overlap.

Kimi-K2.5 shapes:
  hidden_size=7168, moe_intermediate_size=2048, n_routed_experts=384,
  n_shared_experts=1, top_k=8, group_size=32.

Usage:
  GPU_MAX_HW_QUEUES=5 SGLANG_USE_AITER=1 SGLANG_ROCM_USE_MULTI_STREAM=1 \
      python benchmark/kernels/benchmark_rocm_kimi_k25_mxfp4_moe_multistream.py
"""

import argparse
import statistics
import time

import torch

from sglang.srt.layers.moe.rocm_kimi_mxfp4_moe import (
    build_trivial_row_map,
    get_rocm_moe_stream_state,
    rocm_mxfp4_moe_add_shared,
    rocm_mxfp4_moe_finalize_fuse_shared,
)

HIDDEN = 7168
MOE_INTER = 2048
N_ROUTED = 384
N_SHARED = 1
TOP_K = 8
GROUP = 32
TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    _sync()
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        lat.append((time.perf_counter() - t0) * 1e6)  # us
    lat.sort()
    return {
        "avg": statistics.mean(lat),
        "p50": lat[len(lat) // 2],
        "p90": lat[int(len(lat) * 0.9)],
        "p99": lat[int(len(lat) * 0.99)],
    }


def bench_combine_kernels(device, dtype):
    print("\n=== Fused combine kernels (Kimi hidden=%d, top_k=%d) ===" % (HIDDEN, TOP_K))
    print(f"{'tokens':>7} | {'P0 add-shared us':>18} | {'P1 finalize us':>16}")
    print("-" * 50)
    for T in TOKEN_COUNTS:
        routed = torch.randn(T, HIDDEN, dtype=dtype, device=device)
        shared = torch.randn(T, HIDDEN, dtype=dtype, device=device)
        partial = torch.randn(T * TOP_K, HIDDEN, dtype=dtype, device=device)
        row_map = build_trivial_row_map(T, TOP_K, device)
        weights = torch.rand(T, TOP_K, dtype=torch.float32, device=device)

        p0 = _bench(lambda: rocm_mxfp4_moe_add_shared(routed, shared))
        p1 = _bench(
            lambda: rocm_mxfp4_moe_finalize_fuse_shared(
                partial, row_map, weights, shared, 1.0, TOP_K
            )
        )
        print(f"{T:>7} | {p0['avg']:>18.2f} | {p1['avg']:>16.2f}")


class _MoEProxy:
    """BF16 GEMM proxy for the routed + shared experts (stand-in for the AITER
    MXFP4 kernels so the schedule benchmark runs without AITER)."""

    def __init__(self, device, dtype):
        g = torch.Generator(device=device).manual_seed(0)
        # Shared expert: gate_up (hidden->2*inter) + down (inter->hidden).
        self.s_gate_up = torch.randn(
            2 * MOE_INTER, HIDDEN, dtype=dtype, device=device, generator=g
        )
        self.s_down = torch.randn(
            HIDDEN, MOE_INTER, dtype=dtype, device=device, generator=g
        )
        # Routed proxy: activate top_k experts; approximate with a single wide
        # GEMM whose cost matches top_k expert MLPs (memory-bound on decode).
        self.r_gate_up = torch.randn(
            2 * MOE_INTER, HIDDEN, dtype=dtype, device=device, generator=g
        )
        self.r_down = torch.randn(
            HIDDEN, MOE_INTER, dtype=dtype, device=device, generator=g
        )

    def shared(self, x):
        h = torch.nn.functional.silu(x @ self.s_gate_up.t()[:, :MOE_INTER])
        return h @ self.s_down.t()

    def routed(self, x, top_k):
        # top_k expert MLPs on the same token set (proxy for grouped MoE).
        out = torch.zeros_like(x)
        for _ in range(top_k):
            h = torch.nn.functional.silu(x @ self.r_gate_up.t()[:, :MOE_INTER])
            out = out + (h @ self.r_down.t())
        return out


def bench_schedule(device, dtype):
    print("\n=== MoE block schedule wall time (single-stream vs multi-stream) ===")
    print(f"{'tokens':>7} | {'single us':>10} | {'multi us':>10} | {'speedup':>8} | {'max_err':>9}")
    print("-" * 60)
    proxy = _MoEProxy(device, dtype)
    state = get_rocm_moe_stream_state(device) if torch.cuda.is_available() else None

    for T in TOKEN_COUNTS:
        x = torch.randn(T, HIDDEN, dtype=dtype, device=device)

        def single():
            s = proxy.shared(x)
            r = proxy.routed(x, TOP_K)
            return rocm_mxfp4_moe_add_shared(r, s)

        def multi():
            main = torch.cuda.current_stream()
            state.shared_stream.wait_stream(main)
            with torch.cuda.stream(state.shared_stream):
                s = proxy.shared(x)
                state.shared_done_event.record(state.shared_stream)
            r = proxy.routed(x, TOP_K)
            main.wait_event(state.shared_done_event)
            return rocm_mxfp4_moe_add_shared(r, s)

        ref = single()
        single_stats = _bench(single)
        if state is not None:
            multi_out = multi()
            multi_stats = _bench(multi)
            max_err = (multi_out.float() - ref.float()).abs().max().item()
            speedup = single_stats["avg"] / max(multi_stats["avg"], 1e-9)
            print(
                f"{T:>7} | {single_stats['avg']:>10.2f} | "
                f"{multi_stats['avg']:>10.2f} | {speedup:>7.2f}x | {max_err:>9.4f}"
            )
        else:
            print(f"{T:>7} | {single_stats['avg']:>10.2f} | {'n/a (cpu)':>10} | "
                  f"{'n/a':>8} | {'n/a':>9}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    hip = getattr(torch.version, "hip", None)
    print(f"device={device} dtype={args.dtype} torch.version.hip={hip}")
    if device == "cpu":
        print(
            "WARNING: no GPU detected; running the combine-kernel torch fallback "
            "on CPU only (no real overlap). Run on MI350/MI355 for real numbers."
        )

    bench_combine_kernels(device, dtype)
    bench_schedule(device, dtype)

    print(
        "\nEnd-to-end (real model) benchmark:\n"
        "  export SGLANG_USE_AITER=1 SGLANG_ROCM_USE_MULTI_STREAM=1 \\\n"
        "         SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1 GPU_MAX_HW_QUEUES=5\n"
        "  python -m sglang.launch_server --model-path amd/Kimi-K2.5-MXFP4 \\\n"
        "         --tp 4 --attention-backend aiter --trust-remote-code\n"
        "  # then: python -m sglang.bench_serving --backend sglang ... (decode-heavy)\n"
        "  # profile overlap with: rocprof --hip-trace python -m sglang.bench_one_batch ...\n"
    )


if __name__ == "__main__":
    main()
