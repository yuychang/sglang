#!/usr/bin/env python3
"""CUDA-graph microbench: fused AR + residual + AttnRes + RMSNorm.

Compares that kernel with the split path serving uses today: AITER custom
all-reduce, then attn_res_hip with the residual add. For M in {1, 2, 4, 8}
it also times custom all-reduce-with-residual plus attn_res_hip without the
add, which is the path SGLANG_ROCM_AR_RESIDUAL turns on.

Launch: torchrun --nproc_per_node=8 test/manual/kernels/bench_hip_ar_attnres.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

H = 7168
WARMUP = int(os.environ.get("BENCH_WARMUP", "20"))
ITERS = int(os.environ.get("BENCH_ITERS", "50"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "3"))
NVB = int(os.environ.get("BENCH_NVB", "4"))
TOKENS = [1, 2, 4, 7, 8, 12, 16]
if os.environ.get("BENCH_TOKENS"):
    TOKENS = [int(t) for t in os.environ["BENCH_TOKENS"].split(",")]
PROFILE = os.environ.get("BENCH_PROFILE", "0") == "1"


class _Param:
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.weight = weight
        self.variance_epsilon = eps


def _init():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.test.test_utils import publish_build_topology

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local,
        distributed_init_method="env://",
        backend="nccl",
    )
    publish_build_topology(tp_size=world, world_rank=rank)
    initialize_model_parallel()


def _time_graph(graph: torch.cuda.CUDAGraph) -> float:
    stream = torch.cuda.current_stream()
    for _ in range(WARMUP):
        graph.replay()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    stream.synchronize()
    return start.elapsed_time(end) * 1e3 / ITERS


def _capture(fn):
    from sglang.srt.distributed.parallel_state import graph_capture

    with graph_capture() as gc:
        stream = gc.stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
    return graph


def _median_max(samples: list[float]) -> float:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, samples)
    # One fusion call is done when the slowest rank finishes.
    per_repeat = []
    for i in range(len(samples)):
        per_repeat.append(max(rank_samples[i] for rank_samples in gathered))
    per_repeat.sort()
    return per_repeat[len(per_repeat) // 2]


def main() -> int:
    _init()
    rank = dist.get_rank()
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.layers.communication.hip_ar_attnres import (
        allreduce_residual_attnres,
    )

    group = get_tp_group()
    pg = group.device_group
    ca = group.ca_comm
    if ca is None or ca.disabled:
        if rank == 0:
            print("custom all-reduce unavailable", file=sys.stderr)
        return 1
    tp_rank = dist.get_rank(group=pg)
    world = dist.get_world_size(group=pg)
    device = torch.device("cuda", torch.cuda.current_device())
    score_proj = _Param(torch.randn(1, H, dtype=torch.bfloat16, device=device))
    score_norm = _Param(torch.ones(H, dtype=torch.bfloat16, device=device))
    out_norm = _Param(torch.ones(H, dtype=torch.bfloat16, device=device))
    if rank == 0:
        print(
            f"TP={world} H={H} default_nvb={NVB} "
            f"warmup={WARMUP} iters={ITERS} repeats={REPEATS}",
            flush=True,
        )

    def report(label: str, fused_us: float, split_us: float, extra: str = "") -> None:
        if rank != 0:
            return
        delta = split_us - fused_us
        print(
            f"{label:<22} fused={fused_us:8.2f} us  split={split_us:8.2f} us  "
            f"delta={delta:+7.2f} us ({100 * delta / split_us:+5.1f}%){extra}",
            flush=True,
        )

    profiled = False
    for tokens in TOKENS:
        nvb = min(NVB, 8)
        g = torch.Generator(device=device)
        g.manual_seed(1000 + rank + tokens)
        partial = torch.randn(tokens, H, dtype=torch.bfloat16, device=device, generator=g)
        residual = torch.randn(tokens, H, dtype=torch.bfloat16, device=device, generator=g)
        dist.broadcast(residual, src=0)
        bank = torch.randn(tokens, nvb, H, dtype=torch.bfloat16, device=device, generator=g)
        dist.broadcast(bank, src=0)
        cw = (score_norm.weight.float() * score_proj.weight.squeeze().float()).contiguous()

        # Compile and map the symmetric heap on every rank before capture.
        allreduce_residual_attnres(
            partial,
            residual,
            bank,
            nvb,
            score_proj,
            score_norm,
            out_norm,
            group=pg,
            rank=tp_rank,
            world_size=world,
        )
        dist.barrier()
        torch.cuda.synchronize()

        split_out = torch.empty_like(partial)
        split_prefix = torch.empty_like(partial)

        def _split(p=partial, r=residual, b=bank, n=nvb, o=split_out, po=split_prefix, w=cw):
            reduced = ca.custom_all_reduce(p)
            attn_res_hip(
                reduced,
                b,
                w,
                out_norm.weight,
                o,
                n,
                score_norm.variance_epsilon,
                out_norm.variance_epsilon,
                addend=r,
                prefix_out=po,
            )

        def _fused(p=partial, r=residual, b=bank, n=nvb):
            allreduce_residual_attnres(
                p,
                r,
                b,
                n,
                score_proj,
                score_norm,
                out_norm,
                group=pg,
                rank=tp_rank,
                world_size=world,
            )

        if rank == 0:
            print(f"capture M={tokens} nvb={nvb}", flush=True)
        g_split = _capture(_split)
        g_fused = _capture(_fused)
        fused_runs = []
        split_runs = []
        for _ in range(REPEATS):
            fused_runs.append(_time_graph(g_fused))
            split_runs.append(_time_graph(g_split))
        report(f"M={tokens} nvb={nvb}", _median_max(fused_runs), _median_max(split_runs))

        if tokens in (1, 2, 4, 8) and hasattr(ca, "custom_all_reduce_residual"):
            res_out = torch.empty_like(partial)

            def _split_res(p=partial, r=residual, b=bank, n=nvb, o=res_out, w=cw):
                summed = ca.custom_all_reduce_residual(p, r)
                attn_res_hip(
                    summed,
                    b,
                    w,
                    out_norm.weight,
                    o,
                    n,
                    score_norm.variance_epsilon,
                    out_norm.variance_epsilon,
                )

            probe = ca.custom_all_reduce_residual(partial, residual)
            if probe is not None:
                g_res = _capture(_split_res)
                res_runs = [_time_graph(g_res) for _ in range(REPEATS)]
                # Reuse the fused samples from this case; retime so the
                # comparison is adjacent to the residual-AR graph.
                fused_again = [_time_graph(g_fused) for _ in range(REPEATS)]
                report(
                    f"  vs AR+res M={tokens}",
                    _median_max(fused_again),
                    _median_max(res_runs),
                    extra="  (split = AR+residual, then AttnRes)",
                )

        if PROFILE and not profiled and tokens == 7:
            profiled = True
            _profile(rank, g_fused, g_split)
        dist.barrier()
    return 0


def _profile(rank: int, g_fused: torch.cuda.CUDAGraph, g_split: torch.cuda.CUDAGraph) -> None:
    from torch.profiler import ProfilerActivity, profile

    def _run(graph: torch.cuda.CUDAGraph) -> None:
        for _ in range(8):
            graph.replay()
        torch.cuda.synchronize()

    for name, graph in (("fused", g_fused), ("split", g_split)):
        _run(graph)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _run(graph)
        if rank == 0:
            print(f"\nprofile {name} M=7 nvb={NVB}", flush=True)
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12),
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
