#!/usr/bin/env python3
"""Does the fork lose because the main stream is many small kernels?

Two probes now say forking the shared MLP behind the routed work should win by
9-18 us/layer, in both compute-bound and bandwidth-saturated regimes. The real
model loses 2-7%. So the discrepancy is not payload size and not HBM headroom.

Remaining structural difference: the probes model the routed experts as ONE
large torch.mm. The real path is a long chain of small aiter kernels -- topk,
sort, grouped mxfp4 GEMM, activation, second GEMM, scatter-reduce -- each of
which is a separate node in the hipGraph, and each of which is statically tuned
(tuned_fmoe.csv) on the assumption it owns the whole device.

This holds total main-stream work fixed and sweeps how many kernels it is split
into. If gain decays toward zero (or negative) as the kernel count rises, the
loss is a per-node cost of having a live concurrent stream -- which is a
property of the real MoE's shape and cannot be fixed by resizing the payload.

    python overlap_kernelcount.py [--layers 92]
"""

import argparse

import torch

H = 7168
GU = 1536
SI = 768


def timed(build, iters=100):
    g = build()
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
    p.add_argument("--layers", type=int, default=92)
    a = p.parse_args()
    L = a.layers
    dev = "cuda"
    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{L} layers per graph, total main-stream work held fixed\n")

    T = 32
    hs = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    wgu = torch.randn(H, GU, device=dev, dtype=torch.bfloat16)
    ogu = torch.empty(T, GU, device=dev, dtype=torch.bfloat16)
    sin = torch.empty(T, SI, device=dev, dtype=torch.bfloat16)
    wdn = torch.randn(SI, H, device=dev, dtype=torch.bfloat16)
    odn = torch.empty(T, H, device=dev, dtype=torch.bfloat16)

    alt = torch.cuda.Stream()
    fe, je = torch.cuda.Event(), torch.cuda.Event()

    def side():
        torch.mm(hs, wgu, out=ogu)
        torch.mul(torch.nn.functional.silu(ogu[:, :SI]), ogu[:, SI:], out=sin)
        torch.mm(sin, wdn, out=odn)

    TOTAL_K = 32768
    hdr = (
        f"{'kernels':>8} {'K each':>8} {'main only':>10} {'serial':>9} "
        f"{'forked':>9} {'gain':>9} {'verdict':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for nk in (1, 2, 4, 8, 16, 32, 64):
        k = TOTAL_K // nk
        mx = torch.randn(T, k, device=dev, dtype=torch.bfloat16)
        mws = [
            torch.randn(k, 4096, device=dev, dtype=torch.bfloat16) for _ in range(nk)
        ]
        mo = torch.empty(T, 4096, device=dev, dtype=torch.bfloat16)

        def mainwork():
            for w in mws:
                torch.mm(mx, w, out=mo)

        def build(mode):
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(3):
                    mainwork()
                    side()
            torch.cuda.current_stream().wait_stream(warm)
            alt.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                cur = torch.cuda.current_stream()
                for _ in range(L):
                    if mode == "forked":
                        fe.record(cur)
                        alt.wait_event(fe)
                    mainwork()
                    if mode == "serial":
                        side()
                    elif mode == "forked":
                        with torch.cuda.stream(alt):
                            side()
                        je.record(alt)
                        cur.wait_event(je)
            return g

        m = timed(lambda: build("main"))
        s = timed(lambda: build("serial"))
        f = timed(lambda: build("forked"))
        gain = (s - f) / L * 1000.0
        print(
            f"{nk:>8} {k:>8} {m:>9.3f}m {s:>8.3f}m {f:>8.3f}m "
            f"{gain:>+7.1f}us {'WIN' if gain > 0 else 'loss':>8}"
        )
        del mx, mws, mo
        torch.cuda.empty_cache()

    print(
        "\nTotal main-stream FLOPs and bytes are identical on every row; only the\n"
        "number of hipGraph nodes the fork has to coexist with changes."
    )


if __name__ == "__main__":
    main()
