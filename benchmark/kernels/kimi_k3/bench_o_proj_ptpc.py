"""Can the Kimi-K3 o_proj move to PTPC FP8 without an in-kernel quant fusion?

o_proj is [M, 1536] x [1536, 7168] per rank under TP8, run 69 times per decode
step for KDA layers and 24 for MLA. The tuned tables put BF16 at 6.61 us and FP8
at 5.11 us, a 1.50 us margin, and the question is whether a standalone per-token
activation quant fits inside it.

The earlier 2.1-2.5 us quant measurement was for a 7168-wide activation, which is
the MLA input projection's shape, not this one: o_proj's input is 1536 wide, 4.7x
narrower. If the narrow quant fits the margin, generic PTPC is enough and no
custom kernel is needed. If it does not, the quant has to be folded into the
producer -- for KDA at decode that is the recurrence kernel, which already
absorbs the gated RMSNorm.

Timed under graph replay over rotating weights, as decode runs it.
"""

import argparse
import sys

import torch

IN_FEATURES = 1536  # 96 heads x 128 / TP8
OUT_FEATURES = 7168


def graph_time(fn, reps=32, iters=50, warmup=20):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(3):
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters / reps * 1000.0)
    del graph
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--weight-copies", type=int, default=8)
    args = parser.parse_args()

    from aiter import dtypes
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
    from aiter.ops.quant import per_token_quant_hip
    from aiter.ops.shuffle import shuffle_weight
    from aiter.tuned_gemm import tgemm

    torch.cuda.set_device(0)
    print(
        f"{torch.cuda.get_device_name(0)}  "
        f"o_proj [M,{IN_FEATURES}] x [{IN_FEATURES},{OUT_FEATURES}]"
    )
    print(
        f"weights: bf16 {OUT_FEATURES*IN_FEATURES*2/1e6:.1f} MB, "
        f"fp8 {OUT_FEATURES*IN_FEATURES/1e6:.1f} MB\n"
    )

    torch.manual_seed(0)
    n = args.weight_copies
    w_bf16 = [
        torch.randn(OUT_FEATURES, IN_FEATURES, device="cuda", dtype=torch.bfloat16)
        for _ in range(n)
    ]
    w_fp8, w_scale = [], []
    for w in w_bf16:
        q, s = per_token_quant_hip(w.contiguous(), quant_dtype=dtypes.fp8)
        w_fp8.append(shuffle_weight(q, layout=(16, 16)))
        w_scale.append(s.view(OUT_FEATURES, 1).contiguous().float())

    print(
        f"{'M':>4} {'bf16 tuned':>11} {'quant':>7} {'fp8 gemm':>9} "
        f"{'quant+fp8':>10} | {'verdict':>22}"
    )
    for m in args.tokens:
        x = torch.randn(m, IN_FEATURES, device="cuda", dtype=torch.bfloat16)
        xq, xs = per_token_quant_hip(x, quant_dtype=dtypes.fp8)
        xs_v = xs.view(m, 1)
        i = {"k": 0}

        def nxt():
            i["k"] = (i["k"] + 1) % n
            return i["k"]

        # aiter's tuned dispatch, which is what a ROCm linear layer resolves to.
        def do_bf16():
            tgemm.mm(x, w_bf16[nxt()], None, None, None)

        def do_quant():
            per_token_quant_hip(x, quant_dtype=dtypes.fp8)

        def do_fp8():
            k = nxt()
            gemm_a8w8_bpreshuffle(xq, w_fp8[k], xs_v, w_scale[k], dtype=torch.bfloat16)

        def do_both():
            k = nxt()
            q, s = per_token_quant_hip(x, quant_dtype=dtypes.fp8)
            gemm_a8w8_bpreshuffle(
                q, w_fp8[k], s.view(m, 1), w_scale[k], dtype=torch.bfloat16
            )

        t_bf16 = graph_time(do_bf16)
        t_quant = graph_time(do_quant)
        t_fp8 = graph_time(do_fp8)
        t_both = graph_time(do_both)
        if t_both < t_bf16:
            verdict = f"generic wins {t_bf16/t_both:.2f}x"
        elif t_fp8 < t_bf16:
            verdict = f"needs fusion ({t_bf16/t_fp8:.2f}x)"
        else:
            verdict = "no headroom"
        print(
            f"{m:>4} {t_bf16:11.2f} {t_quant:7.2f} {t_fp8:9.2f} {t_both:10.2f} | "
            f"{verdict:>22}"
        )

    print("\nquant+fp8 below bf16 means generic PTPC is enough; only fp8 below")
    print("bf16 means the quant must move into the producing kernel.")


if __name__ == "__main__":
    sys.exit(main())
