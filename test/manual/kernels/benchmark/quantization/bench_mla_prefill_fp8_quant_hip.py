"""Per-tensor (AiterAttnBackend._quantize_mla_prefill_fp8) vs batched MLA prefill FP8 quant.

Shapes are Kimi-K3 TP8: 12 local heads, QK dim 192, V dim 128, with V a strided
slice of the fused kv_b_proj output. Each timed call reads a different input
set so L2 and MALL stay cold.
"""

import argparse
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.kernels.ops.quantization.mla_prefill_fp8_quant_hip import (
    mla_prefill_fp8_quant,
)
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

HEADS = 12
QK_DIM = 192
V_DIM = 128
NOPE_DIM = 128
# Rotated inputs must exceed MI355's 256 MB MALL plus L2.
COLD_BYTES = 512 << 20
MAX_INPUT_SETS = 64


def make_operands(num_q: int, num_kv: int, kv_fp8: bool):
    q = torch.randn((num_q, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
    if kv_fp8:
        return [q]
    k = torch.randn((num_kv, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(
        (num_kv, HEADS, NOPE_DIM + V_DIM), device="cuda", dtype=torch.bfloat16
    )
    return [q, k, kv[..., NOPE_DIM:]]


def make_input_sets(num_q: int, num_kv: int, kv_fp8: bool):
    first = make_operands(num_q, num_kv, kv_fp8)
    set_bytes = sum(x.untyped_storage().nbytes() for x in first)
    count = min(MAX_INPUT_SETS, max(2, -(-COLD_BYTES // set_bytes)))
    return [first] + [make_operands(num_q, num_kv, kv_fp8) for _ in range(count - 1)]


def per_tensor(tensors):
    return [AiterAttnBackend._quantize_mla_prefill_fp8(x) for x in tensors]


def batched(tensors):
    return mla_prefill_fp8_quant(tensors)


def count_kernels(fn, tensors) -> int:
    fn(tensors)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(tensors)
        torch.cuda.synchronize()
    return sum(
        1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )


def gpu_us(fn, input_sets, trials: int) -> float:
    for tensors in input_sets:
        fn(tensors)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for tensors in input_sets:
            fn(tensors)
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
        samples.append(start.elapsed_time(end) * 1000.0 / len(input_sets))
    return statistics.median(samples)


def host_us(fn, input_sets, trials: int) -> float:
    # Host time to enqueue one call; the minimum filters CPU contention.
    samples = []
    for _ in range(trials):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for tensors in input_sets:
            fn(tensors)
        samples.append((time.perf_counter() - begin) * 1e6 / len(input_sets))
        torch.cuda.synchronize()
    return min(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[512, 2048, 4096, 8192]
    )
    parser.add_argument("--prefix", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    cases = [(f"extend T={t}", t, t, False) for t in args.tokens]
    cases += [
        (f"prefix T={t} P={p}", t, p, False)
        for t in args.prefix_tokens
        for p in args.prefix
    ]
    cases += [(f"q-only T={t}", t, 0, True) for t in args.tokens]

    print(
        f"{'case':<24} {'krn old':>7} {'krn new':>7} {'gpu old us':>11} "
        f"{'gpu new us':>11} {'host old us':>11} {'host new us':>11}"
    )
    for name, num_q, num_kv, kv_fp8 in cases:
        torch.manual_seed(0)
        input_sets = make_input_sets(num_q, num_kv, kv_fp8)
        row = [
            count_kernels(per_tensor, input_sets[0]),
            count_kernels(batched, input_sets[0]),
            gpu_us(per_tensor, input_sets, args.trials),
            gpu_us(batched, input_sets, args.trials),
            host_us(per_tensor, input_sets, 5 * args.trials),
            host_us(batched, input_sets, 5 * args.trials),
        ]
        print(
            f"{name:<24} {row[0]:>7} {row[1]:>7} {row[2]:>11.1f} {row[3]:>11.1f} "
            f"{row[4]:>11.1f} {row[5]:>11.1f}",
            flush=True,
        )
        del input_sets
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
