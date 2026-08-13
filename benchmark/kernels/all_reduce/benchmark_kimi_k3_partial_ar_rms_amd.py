"""Benchmark Kimi-K3 packed all-reduce + latent-only RMSNorm on TP8 AMD."""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist

import aiter
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)

HIDDEN = 3584
EPS = 1e-5


def _measure(fn, warmup: int, iters: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        dist.barrier(group=get_tp_group().device_group)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        local = torch.tensor(
            [start.elapsed_time(end) * 1000.0 / iters],
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(local, op=dist.ReduceOp.MAX, group=get_tp_group().device_group)
        samples.append(float(local.item()))
    return statistics.median(samples)


def _capture(fn):
    with graph_capture() as capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture.stream):
            fn()
    return graph.replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", default="1,2,4,8,16,32,64,128,256", type=str
    )
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--iters", default=100, type=int)
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    group = get_tp_group()
    comm = group.ca_comm
    assert comm is not None and not comm.disabled

    modes = ("eager", "graph") if args.mode == "both" else (args.mode,)
    if rank == 0:
        print("mode,tokens,split_us,fused_1stage_us,fused_2stage_us,best_speedup")

    for tokens in [int(item) for item in args.tokens.split(",") if item]:
        generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
        packed = torch.randn(
            3 * tokens,
            HIDDEN,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            HIDDEN,
            generator=torch.Generator(device="cuda").manual_seed(99),
            device="cuda",
            dtype=torch.bfloat16,
        )

        def split():
            reduced = group.all_reduce(packed)
            return aiter.rms_norm(reduced[:tokens], weight, EPS)

        def fused_1stage():
            return comm.custom_fused_ar_partial_rms(
                packed, weight, tokens, EPS, True
            )

        def fused_2stage():
            return comm.custom_fused_ar_partial_rms(
                packed, weight, tokens, EPS, False
            )

        # Compile before graph capture and remove JIT from eager measurements.
        split()
        fused_1stage()
        fused_2stage()
        torch.cuda.synchronize()

        for mode in modes:
            functions = (split, fused_1stage, fused_2stage)
            if mode == "graph":
                functions = tuple(_capture(fn) for fn in functions)
            values = [
                _measure(fn, args.warmup, args.iters, args.repeats)
                for fn in functions
            ]
            if rank == 0:
                best = min(values[1:])
                print(
                    f"{mode},{tokens},{values[0]:.3f},{values[1]:.3f},"
                    f"{values[2]:.3f},{values[0] / best:.4f}"
                )

    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
