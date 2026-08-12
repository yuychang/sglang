#!/usr/bin/env python3
"""Why does forking help a square-GEMM main stream but not K3's real decode?

overlap_atom_shape.py predicted the ATOM-shaped split would win (+1.3..+2.8
us/layer). On the real model it recovered ~0.5%. The suspect is the stand-in:
that probe's main-stream payload was a large square bf16 GEMM -- compute bound,
so it leaves HBM headroom a second stream can use. K3's decode main stream is a
skinny GEMV over big weights -- bandwidth bound, so there is no headroom to
give, and a fork buys nothing while still costing its barrier.

This isolates that one variable. Two main-stream payloads tuned to comparable
duration but opposite arithmetic intensity, each with the same shared-MLP
payload forked behind it:

  compute   [1024, 4096] x [4096, 4096]  -- high flop:byte, HBM headroom
  bandwidth [   32, K   ] x [K, 8192]    -- weight-read dominated, no headroom

If gain > 0 for compute and <= 0 for bandwidth, the negative serving result is
a bandwidth-saturation property of decode, not a payload-size problem -- and no
reshaping of the side payload can fix it, because a second stream adds
concurrency, not bandwidth.

    python overlap_intensity.py [--layers 92]
"""

import argparse

import torch

H = 7168
GU = 1536
SI = 768

BW = 8.0e12  # MI355X HBM3E peak, bytes/s -- for the utilization column only


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
    print(f"{L} layers per graph, shared MLP = gate_up + act + down (33 MB)\n")

    alt = torch.cuda.Stream()
    fe, je = torch.cuda.Event(), torch.cuda.Event()

    cases = [
        ("compute", 1024, 4096, 4096),
        ("bandwidth", 32, 16384, 8192),
        ("bandwidth", 32, 32768, 8192),
        ("bandwidth", 32, 65536, 8192),
    ]

    hdr = (
        f"{'kind':>10} {'M':>6} {'K':>7} {'flop:byte':>10} {'main only':>10} "
        f"{'main GB/s':>10} {'%peak':>6} {'serial':>8} {'forked':>8} {'gain':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for kind, M, K, N in cases:
        mx = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        mw = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
        mo = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        # shared MLP, sized per token count M is irrelevant -- it is decode
        # shaped in the model, so keep it at 32 tokens like the real thing.
        T = 32
        hs = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
        wgu = torch.randn(H, GU, device=dev, dtype=torch.bfloat16)
        ogu = torch.empty(T, GU, device=dev, dtype=torch.bfloat16)
        sin = torch.empty(T, SI, device=dev, dtype=torch.bfloat16)
        wdn = torch.randn(SI, H, device=dev, dtype=torch.bfloat16)
        odn = torch.empty(T, H, device=dev, dtype=torch.bfloat16)

        flops = 2.0 * M * K * N
        mbytes = 2.0 * (M * K + K * N + M * N)
        intensity = flops / mbytes

        def side():
            torch.mm(hs, wgu, out=ogu)
            torch.mul(torch.nn.functional.silu(ogu[:, :SI]), ogu[:, SI:], out=sin)
            torch.mm(sin, wdn, out=odn)

        def build(mode):
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(5):
                    torch.mm(mx, mw, out=mo)
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
                    torch.mm(mx, mw, out=mo)
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
        gbs = mbytes * L / (m * 1e-3) / 1e9
        gain = (s - f) / L * 1000.0
        print(
            f"{kind:>10} {M:>6} {K:>7} {intensity:>10.1f} {m:>9.3f}m "
            f"{gbs:>10.0f} {gbs*1e9/BW*100:>5.0f}% {s:>7.3f}m {f:>7.3f}m "
            f"{gain:>+7.1f}us"
        )
        del mx, mw, mo, hs, wgu, ogu, sin, wdn, odn
        torch.cuda.empty_cache()

    print(
        "\nIf gain is positive only where %peak is low, the fork was never buying\n"
        "overlap -- it was borrowing idle bandwidth that K3's decode does not have."
    )


if __name__ == "__main__":
    main()
