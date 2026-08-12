#!/usr/bin/env python3
"""How much does one CUDA-graph stream fork/join cost on this GPU?

K3 has 92 MoE layers. If the dual-stream shared/routed overlap is a net loss on
this hardware, the two candidate explanations are (a) the overlap gain is
smaller than we think, or (b) the fork/join itself is expensive. This isolates
(b) with no model in the loop: capture a graph with N fork/join pairs around a
trivially small kernel, capture the same N kernels with no fork, replay both,
and divide the difference by N.

    python fork_cost.py [--layers 92] [--iters 200]

Single GPU, a few seconds. Also reports the cost with reused events vs
Stream.wait_stream, and the wall time of a shared-expert-sized down GEMM so the
two can be compared directly.
"""

import argparse

import torch


def build_graph(layers, fork, reuse_events, x, w, out):
    """A graph of `layers` tiny GEMMs, optionally each wrapped in a fork/join
    onto a second stream. The forked work is deliberately trivial (a copy) --
    we are pricing the synchronization, not the payload."""
    alt = torch.cuda.Stream()
    side_in = torch.empty_like(out)
    fork_ev = torch.cuda.Event() if reuse_events else None
    join_ev = torch.cuda.Event() if reuse_events else None

    # warm the capture stream and let the allocator settle
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            torch.mm(x, w, out=out)
    torch.cuda.current_stream().wait_stream(s)
    alt.wait_stream(torch.cuda.current_stream())

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cur = torch.cuda.current_stream()
        for _ in range(layers):
            if fork:
                if fork_ev is None:
                    alt.wait_stream(cur)
                else:
                    fork_ev.record(cur)
                    alt.wait_event(fork_ev)
            torch.mm(x, w, out=out)
            if fork:
                with torch.cuda.stream(alt):
                    side_in.copy_(out)
                if join_ev is None:
                    cur.wait_stream(alt)
                else:
                    join_ev.record(alt)
                    cur.wait_event(join_ev)
    return g


def time_graph(g, iters):
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters  # ms per replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=92, help="K3 has 92 MoE layers")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--tokens", type=int, default=32, help="decode batch")
    a = p.parse_args()

    dev = "cuda"
    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)}")

    # stand-in for a per-layer main-stream kernel
    x = torch.randn(a.tokens, 1024, device=dev, dtype=torch.bfloat16)
    w = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    out = torch.empty(a.tokens, 1024, device=dev, dtype=torch.bfloat16)

    res = {}
    for name, fork, reuse in [
        ("no fork", False, False),
        ("fork, wait_stream", True, False),
        ("fork, reused events", True, True),
    ]:
        g = build_graph(a.layers, fork, reuse, x, w, out)
        res[name] = time_graph(g, a.iters)
        del g
        torch.cuda.synchronize()

    base = res["no fork"]
    print(f"\n{a.layers} layers, replay time per graph:")
    for name, ms in res.items():
        extra = ms - base
        per = extra / a.layers * 1000.0
        tail = "" if name == "no fork" else f"   (+{extra:.3f} ms = {per:.1f} us/fork)"
        print(f"  {name:22s} {ms:8.3f} ms{tail}")

    # What the fork is trying to hide: K3's shared-expert down GEMM at TP8.
    # 2 shared experts x moe_intermediate_size 3072 / tp 8 = 768 rows in,
    # hidden_size 7168 out.
    sx = torch.randn(a.tokens, 768, device=dev, dtype=torch.bfloat16)
    sw = torch.randn(768, 7168, device=dev, dtype=torch.bfloat16)
    sout = torch.empty(a.tokens, 7168, device=dev, dtype=torch.bfloat16)
    for _ in range(20):
        torch.mm(sx, sw, out=sout)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(500):
        torch.mm(sx, sw, out=sout)
    e1.record()
    torch.cuda.synchronize()
    down_us = e0.elapsed_time(e1) / 500 * 1000.0
    print(
        f"\nshared-expert down GEMM [{a.tokens}, 768] x [768, 7168] bf16: "
        f"{down_us:.1f} us  ({sw.numel() * 2 / 1e6:.1f} MB of weights)"
    )
    print(
        "\nThe fork can only pay for itself if the work it hides costs more "
        "than the fork does."
    )


if __name__ == "__main__":
    main()
