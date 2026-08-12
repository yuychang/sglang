#!/usr/bin/env python3
"""Would ATOM's dual-stream split -- whole shared MLP on the alt stream -- flip
the sign that overlap_gain.py measured negative?

overlap_gain.py showed the fork is ~96% efficient but the payload (sglang's
shared *down* GEMM alone, 11 MB) is about the same size as the fork itself
(8.1 us), so it nets out negative. ATOM does not merge the shared gate_up into
a front GEMM, so its alt stream carries gate_up (22 MB) + act + down (11 MB) --
3x the payload against the same fixed fork. This prices that directly.

Three graphs per point, 92 layers each, real K3 TP8 shapes:

  serial   -- front[gu|gate|latent] GEMM, routed payload, shared act+down. one stream.
  fork_dn  -- what sglang does today: front GEMM (incl gate_up) + routed on main,
              shared act+down on alt.
  fork_gu  -- what ATOM does: front GEMM *without* gate_up + routed on main,
              shared gate_up + act + down on alt.

    gain_dn = serial - fork_dn      (measured negative by overlap_gain.py)
    gain_gu = serial - fork_gu      (the question)

Note fork_gu also *shrinks* the main stream (gate_up leaves the front GEMM), so
it is not purely a bigger payload -- it moves work across, which is why it has
to be measured rather than extrapolated.

    python overlap_atom_shape.py [--tokens 32] [--layers 92]
"""

import argparse

import torch

H = 7168  # hidden_size
GU = 1536  # 2 * (moe_intermediate_size 3072 * num_shared_experts 2 / tp 8)
SI = 768  # shared intermediate per rank
GATE = 896  # num_experts
LAT = 3584  # routed_expert_hidden_size


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
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--layers", type=int, default=92)
    a = p.parse_args()
    T, L = a.tokens, a.layers
    dev = "cuda"
    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{L} layers, {T} tokens, TP8 K3 shapes, times per graph replay")
    print(
        f"shared gate_up [{GU}, {H}] = {2*GU*H/1e6:.1f} MB   "
        f"shared down [{H}, {SI}] = {2*H*SI/1e6:.1f} MB\n"
    )

    hs = torch.randn(T, H, device=dev, dtype=torch.bfloat16)

    # front GEMM, with and without the shared gate_up block fused in.
    w_full = torch.randn(H, GU + GATE + LAT, device=dev, dtype=torch.bfloat16)
    o_full = torch.empty(T, GU + GATE + LAT, device=dev, dtype=torch.bfloat16)
    w_nogu = torch.randn(H, GATE + LAT, device=dev, dtype=torch.bfloat16)
    o_nogu = torch.empty(T, GATE + LAT, device=dev, dtype=torch.bfloat16)
    w_gu = torch.randn(H, GU, device=dev, dtype=torch.bfloat16)
    o_gu = torch.empty(T, GU, device=dev, dtype=torch.bfloat16)

    # shared down.
    s_in = torch.empty(T, SI, device=dev, dtype=torch.bfloat16)
    w_dn = torch.randn(SI, H, device=dev, dtype=torch.bfloat16)
    o_dn = torch.empty(T, H, device=dev, dtype=torch.bfloat16)

    alt = torch.cuda.Stream()
    fe, je = torch.cuda.Event(), torch.cuda.Event()

    def act(src):
        # silu_and_mul over the [T, GU] gate_up block -> [T, SI]
        g, u = src[:, :SI], src[:, SI:]
        torch.mul(torch.nn.functional.silu(g), u, out=s_in)

    hdr = (
        f"{'routed K':>9} {'serial':>9} {'fork_dn':>9} {'gain_dn':>9} "
        f"{'fork_gu':>9} {'gain_gu':>9} {'verdict':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for k in (1024, 4096, 8192, 16384, 32768):
        rx = torch.randn(T, k, device=dev, dtype=torch.bfloat16)
        rw = torch.randn(k, 4096, device=dev, dtype=torch.bfloat16)
        ro = torch.empty(T, 4096, device=dev, dtype=torch.bfloat16)

        def routed():
            torch.mm(rx, rw, out=ro)

        def build(mode):
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(5):
                    torch.mm(hs, w_full, out=o_full)
                    torch.mm(hs, w_nogu, out=o_nogu)
                    torch.mm(hs, w_gu, out=o_gu)
                    act(o_gu)
                    torch.mm(s_in, w_dn, out=o_dn)
                    routed()
            torch.cuda.current_stream().wait_stream(warm)
            alt.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                cur = torch.cuda.current_stream()
                for _ in range(L):
                    if mode == "serial":
                        torch.mm(hs, w_full, out=o_full)
                        routed()
                        act(o_full[:, :GU])
                        torch.mm(s_in, w_dn, out=o_dn)
                    elif mode == "fork_dn":
                        torch.mm(hs, w_full, out=o_full)
                        fe.record(cur)
                        alt.wait_event(fe)
                        routed()
                        with torch.cuda.stream(alt):
                            act(o_full[:, :GU])
                            torch.mm(s_in, w_dn, out=o_dn)
                        je.record(alt)
                        cur.wait_event(je)
                    elif mode == "fork_gu":
                        fe.record(cur)
                        alt.wait_event(fe)
                        torch.mm(hs, w_nogu, out=o_nogu)
                        routed()
                        with torch.cuda.stream(alt):
                            torch.mm(hs, w_gu, out=o_gu)
                            act(o_gu)
                            torch.mm(s_in, w_dn, out=o_dn)
                        je.record(alt)
                        cur.wait_event(je)
            return g

        s = timed(lambda: build("serial"))
        fd = timed(lambda: build("fork_dn"))
        fg = timed(lambda: build("fork_gu"))
        gd = (s - fd) / L * 1000.0
        gg = (s - fg) / L * 1000.0
        verdict = "ATOM WIN" if gg > 0 and gg > gd else ("win" if gg > 0 else "loss")
        print(
            f"{k:>9} {s:>8.3f}m {fd:>8.3f}m {gd:>+8.1f}us "
            f"{fg:>8.3f}m {gg:>+8.1f}us {verdict:>8}"
        )
        del rx, rw, ro
        torch.cuda.empty_cache()

    print(
        "\ngain = (serial - fork) / layers, per MoE layer. x92 layers for the\n"
        "per-decode-step effect; an ITL at concurrency 2 is ~18.4 ms."
    )


if __name__ == "__main__":
    main()
