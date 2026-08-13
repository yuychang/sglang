"""TP8 correctness and graph replay for K3 packed AR + partial RMSNorm."""

from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_amd_ci(est_time=240, suite="stage-c-test-large-8-gpu-amd")

HIDDEN = 3584
EPS = 1e-5
_INITIALIZED = False


def _init() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
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
    _INITIALIZED = True


def _cleanup() -> None:
    global _INITIALIZED
    if not _INITIALIZED:
        return
    destroy_model_parallel()
    destroy_distributed_environment()
    _INITIALIZED = False


atexit.register(_cleanup)


def _inputs(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    rank = int(os.environ["RANK"])
    generator = torch.Generator(device="cuda").manual_seed(1234 + rank * 31 + tokens)
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
    return packed, weight


def _reference(packed: torch.Tensor, weight: torch.Tensor, tokens: int) -> torch.Tensor:
    reduced = packed.clone()
    dist.all_reduce(reduced, group=get_tp_group().device_group)
    out = reduced.clone()
    out[:tokens] = F.rms_norm(
        reduced[:tokens].float(), (HIDDEN,), weight.float(), EPS
    ).to(torch.bfloat16)
    return out


def _assert_close(got: torch.Tensor, want: torch.Tensor, tokens: int) -> None:
    torch.testing.assert_close(
        got[tokens:], want[tokens:], rtol=0, atol=0.125
    )
    torch.testing.assert_close(
        got[:tokens], want[:tokens], rtol=2e-2, atol=0.125
    )


@pytest.mark.parametrize(
    ("tokens", "use_1stage"),
    [(1, True), (8, True), (32, False), (128, False)],
)
@torch.inference_mode()
def test_partial_ar_rms_eager(tokens: int, use_1stage: bool) -> None:
    _init()
    packed, weight = _inputs(tokens)
    want = _reference(packed, weight, tokens)
    comm = get_tp_group().ca_comm
    assert comm is not None and not comm.disabled
    got = comm.custom_fused_ar_partial_rms(
        packed, weight, tokens, EPS, use_1stage
    )
    assert got is not None
    torch.cuda.synchronize()
    _assert_close(got, want, tokens)


@torch.inference_mode()
def test_partial_ar_rms_graph_replay_updates() -> None:
    _init()
    tokens = 8
    packed, weight = _inputs(tokens)
    comm = get_tp_group().ca_comm
    assert comm is not None and not comm.disabled

    # Compile and register the op before capture.
    comm.custom_fused_ar_partial_rms(packed, weight, tokens, EPS, True)
    torch.cuda.synchronize()
    with graph_capture() as capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture.stream):
            got = comm.custom_fused_ar_partial_rms(
                packed, weight, tokens, EPS, True
            )
    assert got is not None

    packed.add_(0.25)
    want = _reference(packed, weight, tokens)
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(got, want, tokens)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
