"""Microbenchmark for the Kimi-K3 MLA prefill per-tensor FP8 Q/K/V quant.

Compares the per-tensor amax + static-quant sequence that the FlyDSL FP8 MLA
prefill used before (``AiterAttnBackend._quantize_mla_prefill_fp8`` once per
operand) with the batched two-launch ``mla_prefill_fp8_quant``. Shapes are
Kimi-K3 TP8: 12 local heads, QK dim 192, V dim 128, V a strided slice of the
fused kv_b_proj output.
"""

import argparse
import functools
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.kernels.ops.quantization.mla_prefill_fp8_quant import (
    mla_prefill_fp8_quant,
)
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.test.ci.ci_register import register_amd_ci
from sglang.utils import is_in_ci

register_amd_ci(est_time=120, stage="jit-kernel-benchmark", runner_config="amd")

HEADS = 12
QK_DIM = 192
V_DIM = 128
NOPE_DIM = 128


def make_operands(num_q: int, num_kv: int, kv_fp8: bool):
    q = torch.randn((num_q, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
    if kv_fp8:
        return [q]
    k = torch.randn((num_kv, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(
        (num_kv, HEADS, NOPE_DIM + V_DIM), device="cuda", dtype=torch.bfloat16
    )
    return [q, k, kv[..., NOPE_DIM:]]


def legacy(tensors):
    return [AiterAttnBackend._quantize_mla_prefill_fp8(x) for x in tensors]


def fused(tensors):
    return mla_prefill_fp8_quant(tensors)


def count_kernels(fn) -> int:
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(
        1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )


def gpu_us(fn, warmup: int, iters: int, trials: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(samples)


def host_us(fn, iters: int, trials: int) -> float:
    """Host time to enqueue one call; the minimum filters CPU contention."""
    samples = []
    for _ in range(trials):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter() - begin) * 1e6 / iters)
        torch.cuda.synchronize()
    return min(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[512, 2048, 4096, 8192]
    )
    parser.add_argument("--prefix", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if is_in_ci():
        args.tokens, args.prefix, args.prefix_tokens = [2048], [8192], [2048]
        args.iters, args.trials = 5, 3

    cases = [(f"extend T={t}", t, t, False) for t in args.tokens]
    cases += [
        (f"prefix T={t} P={p}", t, p, False)
        for t in args.prefix_tokens
        for p in args.prefix
    ]
    cases += [(f"q-only T={t}", t, 0, True) for t in args.tokens]

    header = (
        f"{'case':<24} {'krn old':>7} {'krn new':>7} {'gpu old us':>11} "
        f"{'gpu new us':>11} {'host old us':>11} {'host new us':>11}"
    )
    print(header)
    for name, num_q, num_kv, kv_fp8 in cases:
        torch.manual_seed(0)
        tensors = make_operands(num_q, num_kv, kv_fp8)
        old_fn = functools.partial(legacy, tensors)
        new_fn = functools.partial(fused, tensors)
        row = [
            count_kernels(old_fn),
            count_kernels(new_fn),
            gpu_us(old_fn, 3, args.iters, args.trials),
            gpu_us(new_fn, 3, args.iters, args.trials),
            host_us(old_fn, args.iters, 5 * args.trials),
            host_us(new_fn, args.iters, 5 * args.trials),
        ]
        print(
            f"{name:<24} {row[0]:>7} {row[1]:>7} {row[2]:>11.1f} {row[3]:>11.1f} "
            f"{row[4]:>11.1f} {row[5]:>11.1f}",
            flush=True,
        )
        del tensors, old_fn, new_fn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
