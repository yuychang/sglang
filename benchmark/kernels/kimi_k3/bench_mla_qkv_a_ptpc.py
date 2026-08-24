"""Is PTPC FP8 worth wiring into the Kimi-K3 MLA input projection?

`fused_qkv_a_proj_with_mqa` is q_a_proj (1536) fused with kv_a_proj_with_mqa
(512 + 64), so [M, 7168] x [7168, 2112], once per MLA layer and 24 layers per
decode step. In BF16 that is 30.3 MB of weights; in FP8 15.1 MB.

Three candidates, because the answer depends on which one you can actually build:

  bf16              what ships today
  quant + fp8 gemm  what wiring generic PTPC gives you: a standalone
                    per-token quant kernel followed by the FP8 GEMM
  fp8 gemm only     the ceiling, i.e. what a norm+quant fusion that hands the
                    GEMM pre-quantized activations would reach

If "quant + fp8 gemm" loses to bf16 but "fp8 gemm only" wins, the win is real but
only reachable with the fusion, and wiring generic PTPC first is a regression.

Timed under graph replay and over rotating weight buffers, since decode replays a
graph and streams 24 distinct weights per step.
"""

import argparse
import sys

import torch

HIDDEN = 7168
Q_LORA = 1536
KV_LORA_PLUS_ROPE = 512 + 64
FUSED_N = Q_LORA + KV_LORA_PLUS_ROPE  # 2112
BF16_BYTES = FUSED_N * HIDDEN * 2
FP8_BYTES = FUSED_N * HIDDEN * 1


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

    from sglang.kernels.ops.kimi_k3 import ptpc_fp8_aiter_hip

    torch.cuda.set_device(0)
    print(f"{torch.cuda.get_device_name(0)}  shape [M,{HIDDEN}] x [{HIDDEN},{FUSED_N}]")
    print(f"weights: bf16 {BF16_BYTES/1e6:.1f} MB, fp8 {FP8_BYTES/1e6:.1f} MB\n")

    torch.manual_seed(0)
    n = args.weight_copies
    w_bf16 = [
        torch.randn(FUSED_N, HIDDEN, device="cuda", dtype=torch.bfloat16)
        for _ in range(n)
    ]
    # Pack through the adapter the model would actually use, so the layout and
    # scale conventions match production rather than this file's guess.
    packed = [ptpc_fp8_aiter_hip.pack(w) for w in w_bf16]

    print(
        f"{'M':>4} {'bf16':>9} {'quant+fp8':>10} {'fp8 only':>9} "
        f"{'quant':>8} | {'vs bf16':>9} {'ceiling':>9}"
    )
    for m in args.tokens:
        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        out_bf16 = torch.empty(m, FUSED_N, device="cuda", dtype=torch.bfloat16)
        xq, xs = per_token_quant_hip(x, quant_dtype=dtypes.fp8)
        xs_v = xs.view(m, 1)
        i = {"k": 0}

        def nxt():
            i["k"] = (i["k"] + 1) % n
            return i["k"]

        def do_bf16():
            torch.mm(x, w_bf16[nxt()].t(), out=out_bf16)

        def do_quant():
            per_token_quant_hip(x, quant_dtype=dtypes.fp8)

        def do_fp8():
            w, ws, _ = packed[nxt()]
            gemm_a8w8_bpreshuffle(xq, w, xs_v, ws, dtype=torch.bfloat16)

        def do_both():
            w, ws, _ = packed[nxt()]
            q, s = per_token_quant_hip(x, quant_dtype=dtypes.fp8)
            gemm_a8w8_bpreshuffle(q, w, s.view(m, 1), ws, dtype=torch.bfloat16)

        t_bf16 = graph_time(do_bf16)
        t_quant = graph_time(do_quant)
        t_fp8 = graph_time(do_fp8)
        t_both = graph_time(do_both)
        print(
            f"{m:>4} {t_bf16:9.2f} {t_both:10.2f} {t_fp8:9.2f} {t_quant:8.2f} | "
            f"{t_bf16/t_both:8.3f}x {t_bf16/t_fp8:8.3f}x"
        )


if __name__ == "__main__":
    sys.exit(main())
