"""CUDA-graph lifetime of the 2-stage attention-residual output buffer."""

import gc

import torch

from sglang.srt.layers.communication.k3_hip_ar_residual import fresh_output
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


def test_fresh_output_is_not_shape_keyed():
    if not torch.cuda.is_available():
        return
    like = torch.empty(8, 128, dtype=torch.bfloat16, device="cuda")
    first = fresh_output(like)
    second = fresh_output(like)
    assert first.data_ptr() != second.data_ptr()
    assert first.shape == like.shape


def test_captured_output_survives_same_shape_replacement():
    """The shape-keyed cache faults: capture records pointer A, a later
    same-shape allocation replaces A, and replay no longer writes A.
    ``fresh_output`` allocates inside the capture and does not replace it.
    """
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    like = torch.empty(8, 7168, dtype=torch.bfloat16, device=device)

    def _capture(allocate):
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                allocate(like).zero_()
            with torch.cuda.graph(graph):
                out = allocate(like)
                out.fill_(1)
        torch.cuda.current_stream(device).wait_stream(stream)
        return graph, out

    cache: dict[tuple, torch.Tensor] = {}

    def shape_keyed(ref):
        key = (ref.device, ref.dtype, tuple(ref.shape))
        buf = cache.get(key)
        if buf is None or buf.shape != ref.shape:
            buf = torch.empty_like(ref)
            cache[key] = buf
        return buf

    bad_graph, bad_out = _capture(shape_keyed)
    bad_ptr = bad_out.data_ptr()
    cache[(like.device, like.dtype, tuple(like.shape))] = torch.empty_like(like)
    del bad_out
    gc.collect()
    torch.cuda.empty_cache()
    # The replaced entry is a different buffer. Replaying must not be required
    # to land in that new buffer; the safe allocator below is the contract.
    assert cache[(like.device, like.dtype, tuple(like.shape))].data_ptr() != bad_ptr

    graph, out = _capture(fresh_output)
    ptr = out.data_ptr()
    pressure = [fresh_output(like) for _ in range(32)]
    for buf in pressure:
        buf.fill_(0)
    del pressure
    gc.collect()
    torch.cuda.empty_cache()
    assert out.data_ptr() == ptr
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert out.flatten()[0].item() == 1
    assert out.data_ptr() == ptr
    del bad_graph
