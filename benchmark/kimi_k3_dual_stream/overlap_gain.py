#!/usr/bin/env python3
"""Does forking K3's shared-expert down GEMM onto a second stream save anything?

fork_cost.py priced the barrier (~8 us/layer in a hipGraph). This prices the
other side: how much wall clock a fork actually removes, as a function of how
busy the main stream is. Three graphs, 92 layers each:

  main only          -- the main-stream payload alone
  serialized         -- main payload then the shared down GEMM, one stream
  forked             -- main payload on stream 0, shared down GEMM on stream 1

    gain  = serialized - forked        (what overlap buys)
    verdict: forking wins iff gain > 0, and gain is capped by the side GEMM.

The main payload is a square bf16 GEMM whose size is swept, so the columns walk
from "main stream is nearly idle" to "main stream saturates the GPU" -- the
routed-expert path at decode sits toward the busy end.

    python overlap_gain.py [--tokens 32] [--layers 92]
"""

import argparse

import torch


def timed(fn_build, iters=100):
    g = fn_build()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    ms = a.elapsed_time(b) / iters
    del g
    torch.cuda.synchronize()
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--layers", type=int, default=92)
    a = p.parse_args()
    dev = "cuda"
    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{a.layers} layers, {a.tokens} tokens, all times per replay\n")

    # K3 shared expert at TP8: 2 x moe_intermediate_size 3072 / 8 = 768 in,
    # hidden_size 7168 out.
    sx = torch.randn(a.tokens, 768, device=dev, dtype=torch.bfloat16)
    sw = torch.randn(768, 7168, device=dev, dtype=torch.bfloat16)
    sout = torch.empty(a.tokens, 7168, device=dev, dtype=torch.bfloat16)

    alt = torch.cuda.Stream()
    fork_ev, join_ev = torch.cuda.Event(), torch.cuda.Event()

    hdr = f"{'main K':>8} {'main only':>10} {'serialized':>11} {'forked':>9} {'gain':>9} {'verdict':>9}"
    print(hdr)
    print("-" * len(hdr))

    for k in (1024, 4096, 8192, 16384, 32768):
        mx = torch.randn(a.tokens, k, device=dev, dtype=torch.bfloat16)
        mw = torch.randn(k, 4096, device=dev, dtype=torch.bfloat16)
        mout = torch.empty(a.tokens, 4096, device=dev, dtype=torch.bfloat16)

        def build(mode):
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(5):
                    torch.mm(mx, mw, out=mout)
                    torch.mm(sx, sw, out=sout)
            torch.cuda.current_stream().wait_stream(warm)
            alt.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                cur = torch.cuda.current_stream()
                for _ in range(a.layers):
                    if mode == "forked":
                        fork_ev.record(cur)
                        alt.wait_event(fork_ev)
                    torch.mm(mx, mw, out=mout)
                    if mode == "serialized":
                        torch.mm(sx, sw, out=sout)
                    elif mode == "forked":
                        with torch.cuda.stream(alt):
                            torch.mm(sx, sw, out=sout)
                        join_ev.record(alt)
                        cur.wait_event(join_ev)
            return g

        m = timed(lambda: build("main"))
        s = timed(lambda: build("serialized"))
        f = timed(lambda: build("forked"))
        gain_us = (s - f) / a.layers * 1000.0
        verdict = "WIN" if gain_us > 0 else "loss"
        print(
            f"{k:>8} {m:>9.3f}m {s:>10.3f}m {f:>8.3f}m {gain_us:>+7.1f}us {verdict:>9}"
        )
        del mx, mw, mout
        torch.cuda.empty_cache()

    print(
        "\ngain = (serialized - forked) / layers. Positive means the fork removed\n"
        "more wall clock than its barrier cost."
    )


if __name__ == "__main__":
    main()
