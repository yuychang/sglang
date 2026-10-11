#!/usr/bin/env python3
"""TP8 bit-exact and graph test for AITER registered-input copy elision."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

import aiter as ops
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


def _assert_equal(rank: int, name: str, actual: torch.Tensor, expected: torch.Tensor):
    if not torch.equal(actual, expected):
        mismatch = int(torch.count_nonzero(actual != expected).item())
        raise AssertionError(f"rank {rank}: {name} has {mismatch} mismatched elements")


def _rank_input(shape: tuple[int, ...], rank: int, device: torch.device):
    numel = 1
    for dim in shape:
        numel *= dim
    values = (torch.arange(numel, device=device, dtype=torch.int32) % 97) - 48
    return (values.view(shape).to(torch.bfloat16) * 0.03125).add_(rank)


def run(args):
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"this validation requires exactly 8 ranks, got {world_size}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    arch = torch.cuda.get_device_properties(device).gcnArchName
    if "gfx950" not in arch:
        raise RuntimeError(f"expected MI355X/gfx950, got {arch}")

    ca = CustomAllreduce(
        group=dist.group.WORLD,
        device=device,
        max_size=8 * 1024 * 1024,
        enable_register_for_capturing=False,
    )
    if ca.disabled:
        raise RuntimeError("AITER custom all-reduce is disabled")

    try:
        for rows in args.rows:
            shape = (rows, 7168)
            source = _rank_input(shape, rank, device)

            baseline = ca.all_reduce(source, registered_input=False)
            direct = ca.get_registered_input_buffer(shape, torch.bfloat16)
            torch.add(source, 0, out=direct)
            if ops.input_needs_staging(direct, ca._pool["input"].data_ptr):
                raise AssertionError("base registered-input view would still stage")
            direct_out = ca.all_reduce(direct, registered_input=False)
            torch.cuda.synchronize()
            _assert_equal(rank, f"plain M={rows}", direct_out, baseline)

            # K3 MoE's [latent | shared] producer layout: both producers write
            # disjoint spans, followed by one collective over the flat pair.
            latent_numel = rows * 3584
            shared_numel = rows * 7168
            pair_source = _rank_input(
                (latent_numel + shared_numel,), rank, device
            )
            pair_baseline = ca.all_reduce(pair_source, registered_input=False)
            pair = ca.get_registered_input_buffer(pair_source.shape, torch.bfloat16)
            torch.add(pair_source[:latent_numel], 0, out=pair[:latent_numel])
            torch.add(pair_source[latent_numel:], 0, out=pair[latent_numel:])
            pair_out = ca.all_reduce(pair, registered_input=False)
            torch.cuda.synchronize()
            _assert_equal(rank, f"MoE pair M={rows}", pair_out, pair_baseline)

            # Split fallback safety: the base latent span bypasses staging,
            # while the offset shared span stages only after the first AR on
            # the same stream. Their output lifetimes must not alias.
            latent_source = pair_source[:latent_numel].clone()
            shared_source = pair_source[latent_numel:].clone()
            latent_expected = ca.all_reduce(latent_source, registered_input=False)
            shared_expected = ca.all_reduce(shared_source, registered_input=False)
            torch.add(latent_source, 0, out=pair[:latent_numel])
            torch.add(shared_source, 0, out=pair[latent_numel:])
            latent_out = ca.all_reduce(pair[:latent_numel], registered_input=False)
            shared_out = ca.all_reduce(pair[latent_numel:], registered_input=False)
            torch.cuda.synchronize()
            _assert_equal(rank, f"split latent M={rows}", latent_out, latent_expected)
            _assert_equal(rank, f"split shared M={rows}", shared_out, shared_expected)

        # Capture a producer kernel and custom AR against the same fixed pool
        # address, then replay with changing source contents.
        rows = args.rows[0]
        shape = (rows, 7168)
        source = _rank_input(shape, rank, device)
        direct = ca.get_registered_input_buffer(shape, torch.bfloat16)
        graph_out = torch.empty_like(source)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with ca.capture():
            with torch.cuda.graph(graph):
                torch.add(source, rank + 1, out=direct)
                ca.all_reduce(
                    direct,
                    out=graph_out,
                    registered_input=False,
                )
        for iteration in range(args.graph_replays):
            source.copy_(_rank_input(shape, rank + iteration, device))
            eager_input = source + (rank + 1)
            expected = ca.all_reduce(eager_input, registered_input=False)
            graph.replay()
            torch.cuda.synchronize()
            _assert_equal(rank, f"graph replay {iteration}", graph_out, expected)

        dist.barrier()
        if rank == 0:
            print(
                "PASS: TP8 direct-input all-reduce is bit-identical; "
                "MoE pair/split lifetimes and CUDA-graph replay are valid."
            )
    finally:
        ca.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[4, 8, 20])
    parser.add_argument("--graph-replays", type=int, default=3)
    run(parser.parse_args())
