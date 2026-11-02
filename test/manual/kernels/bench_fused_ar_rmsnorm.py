"""Token sweep on ROCm TP8: fused AR + RMSNorm vs split AR then RMSNorm, on the K3 MoE.

Launch: torchrun --nproc_per_node=8 test/manual/kernels/bench_fused_ar_rmsnorm.py

The [latent | shared] buffer is 3 rows of NORM_DIM per token; only the first
num_tokens rows are normalized. Prints the crossover for
SGLANG_OPT_HIP_K3_FUSED_AR_RMSNORM_MAX_TOKENS.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

NORM_DIM = 3584
ROWS_PER_TOKEN = 3
EPS = 1e-6
WARMUP = 20
ITERS = int(os.environ.get("BENCH_ITERS", "50"))
# Median of interleaved fused/split repeats; single runs are noise-dominated.
REPEATS = int(os.environ.get("BENCH_REPEATS", "7"))
# Dense around the decode/verify cap, sparse beyond it.
TOKENS = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 48, 64, 128, 256, 1024, 4096]
if os.environ.get("BENCH_TOKENS"):
    TOKENS = [int(t) for t in os.environ["BENCH_TOKENS"].split(",")]


def _init():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.test.test_utils import publish_build_topology

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    publish_build_topology(world_rank=rank, tp_size=world_size)
    initialize_model_parallel()
    return rank, local_rank


# Graph replays reuse one input per arm, so it stays L2-resident. That is the
# serving pattern: the MoE writes the buffer just before this all-reduce.
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
        s = gc.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            fn()
    return graph


def main() -> int:
    rank, local_rank = _init()
    device = torch.device(f"cuda:{local_rank}")

    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_all_reduce,
    )
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.layers.layernorm import RMSNorm

    # Same kernel the model runs: routed_expert_norm is an RMSNorm module.
    norm = RMSNorm(NORM_DIM, eps=EPS).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    rows = []

    def _fused_call(buf, n):
        return get_tp_group().fused_allreduce_rmsnorm_rows(
            buf, norm.weight, EPS, num_norm_rows=n
        )

    def _decline_reason(buf):
        group = get_tp_group()
        if group.fused_allreduce_rmsnorm_rows_unsupported is not None:
            return group.fused_allreduce_rmsnorm_rows_unsupported
        if group.ca_comm.uses_1stage_ar(buf):
            return "AITER would run the 1-stage all-reduce"
        return "the plain all-reduce would not use AITER custom all-reduce"

    for tokens in TOKENS:
        n_rows = tokens * ROWS_PER_TOKEN
        g = torch.Generator(device=device)
        g.manual_seed(rank * 1000 + tokens)
        base = torch.randn(
            n_rows, NORM_DIM, dtype=torch.bfloat16, device=device, generator=g
        )

        probe = _fused_call(base.clone(), tokens)
        if probe is None:
            if rank == 0:
                rows.append((tokens, None, None))
                print(f"tokens={tokens}: fused path declined: {_decline_reason(base)}")
            continue

        fused_buf = base.clone()

        def _fused(buf=fused_buf, n=tokens):
            _fused_call(buf, n)

        split_buf = base.clone()

        def _split(buf=split_buf, n=tokens):
            reduced = tensor_model_parallel_all_reduce(buf)
            norm(reduced[:n])

        g_fused = _capture(_fused)
        g_split = _capture(_split)
        fused_runs = []
        split_runs = []
        for _ in range(REPEATS):
            fused_runs.append(_time_graph(g_fused))
            split_runs.append(_time_graph(g_split))
        fused_runs.sort()
        split_runs.sort()
        us_fused = fused_runs[len(fused_runs) // 2]
        us_split = split_runs[len(split_runs) // 2]

        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (us_fused, us_split))
        if rank == 0:
            mean_fused = sum(x[0] for x in gathered) / len(gathered)
            mean_split = sum(x[1] for x in gathered) / len(gathered)
            rows.append((tokens, mean_fused, mean_split))
            delta = mean_split - mean_fused
            verdict = "fused" if delta > 0 else "split"
            print(
                f"tokens={tokens:<4} rows={n_rows:<4} "
                f"fused={mean_fused:8.2f} us  split={mean_split:8.2f} us  "
                f"delta={delta:+7.2f} us ({100 * delta / mean_split:+5.1f}%)  "
                f"-> {verdict}"
            )

    if rank == 0:
        measured = [r for r in rows if r[1] is not None]
        wins = [t for t, f, s in measured if s > f]
        losses = [t for t, f, s in measured if s <= f]
        print("\nfused wins at tokens:", wins or "none")
        print("fused loses at tokens:", losses or "none")
        # The cap admits every batch up to it, so it ends before the first loss.
        first_loss = min(losses, default=None)
        cap = max((t for t in wins if first_loss is None or t < first_loss), default=0)
        if cap == 0:
            print("fused never wins from the smallest fused size; leave it off")
        elif first_loss is None:
            print(f"fused wins up to {cap}, the end of the sweep; extend BENCH_TOKENS")
        else:
            print(f"suggested SGLANG_OPT_HIP_K3_FUSED_AR_RMSNORM_MAX_TOKENS={cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
