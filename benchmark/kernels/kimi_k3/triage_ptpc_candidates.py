"""Triage which Kimi-K3 BF16 projections are worth moving to PTPC FP8.

Reads AITER's merged tuned tables and compares, per shape, the best BF16 kernel
against the best FP8 one for this GPU. Both numbers come from the same tuner
under the same conditions, which is the point: a hand-written microbenchmark that
uses torch.mm as the BF16 baseline will overstate the FP8 gain by roughly 2x,
because production does not use torch.mm. For [2112, 7168] the tuned BF16 kernel
is FlyDSL at 8.27 us where torch.mm takes 16.4, and the FP8 entry is 11.2 -- so a
microbenchmark says 1.77x while the truth is 0.74x.

The per-step column is what decides anything: a shape can win handily on the GEMM
and still be worthless if the model only has one of that layer.

    python triage_ptpc_candidates.py
"""

import argparse
import csv
import sys

BF16_TABLE = "/tmp/aiter_configs/bf16_tuned_gemm.csv"
FP8_TABLE = "/tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv"

# (label, N, K, layers-per-decode-step). Layer counts are for Kimi-K3: 93 layers,
# 69 KDA and 24 MLA, 92 MoE, and first_k_dense_replace=1 so exactly one dense MLP.
SHAPES = [
    ("MLA fused_qkv_a", 2112, 7168, 24),
    ("MLA o_proj", 7168, 1536, 24),
    ("KDA o_proj", 7168, 1536, 69),
    ("dense gate_up", 8448, 7168, 1),
    ("dense down", 7168, 4224, 1),
    ("KDA in-proj (fused)", 6288, 7168, 69),
    ("latent up", 7168, 3584, 92),
    ("latent down", 3584, 7168, 92),
]


def best_by_m(path, n, k, gfx, cu):
    out = {}
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        return out
    for r in rows:
        if r.get("gfx") != gfx or r.get("cu_num") != str(cu):
            continue
        if r.get("N") != str(n) or r.get("K") != str(k):
            continue
        try:
            m, us = int(r["M"]), float(r["us"])
        except (KeyError, ValueError):
            continue
        lib = r.get("libtype") or ""
        if m not in out or us < out[m][0]:
            out[m] = (us, lib)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 8, 32])
    parser.add_argument("--gfx", default="gfx950")
    parser.add_argument("--cu", type=int, default=256)
    parser.add_argument(
        "--tpot-us",
        type=float,
        default=13900.0,
        help="decode step time to express the saving against",
    )
    args = parser.parse_args()

    print(
        f"{args.gfx} / {args.cu} CUs, tuned tables only, "
        f"step time {args.tpot_us/1000:.1f} ms\n"
    )
    print(
        f"{'shape':<22} {'lyrs':>4} {'M':>4} {'bf16':>8} {'lib':>8} "
        f"{'fp8':>7} {'ceil':>6} {'per step':>10} {'of step':>8}"
    )
    for label, n, k, layers in SHAPES:
        bf = best_by_m(BF16_TABLE, n, k, args.gfx, args.cu)
        fp = best_by_m(FP8_TABLE, n, k, args.gfx, args.cu)
        if not bf and not fp:
            print(
                f"{label:<22} {layers:>4}  -- neither table covers "
                f"[{n}, {k}] on this GPU"
            )
            continue
        for m in args.tokens:
            b, f = bf.get(m), fp.get(m)
            if not b or not f:
                continue
            saved = (b[0] - f[0]) * layers
            print(
                f"{label:<22} {layers:>4} {m:>4} {b[0]:8.2f} {b[1]:>8} "
                f"{f[0]:7.2f} {b[0]/f[0]:5.2f}x {saved:9.1f}us "
                f"{saved/args.tpot_us*100:7.2f}%"
            )
        print()
    print("A positive ceiling is necessary but not sufficient: generic PTPC also")
    print("pays a standalone per-token activation quant, measured 2.1-2.5 us at")
    print("these token counts, so a shape needs to beat bf16 by more than that")
    print("before wiring it without a norm+quant fusion.")


if __name__ == "__main__":
    sys.exit(main())
