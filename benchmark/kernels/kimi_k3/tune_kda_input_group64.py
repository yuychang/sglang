"""Tune the Kimi-K3 KDA group-64 E4M3 input projection on gfx950.

The shipping launch configuration is fixed in
`sglang/kernels/ops/kimi_k3/flydsl/kimi_k3_kda_input_group64.py`:
rows_per_wave=2, cu_count=256, waves_per_eu=0, weight_cache_modifier=2,
hidden_to_lds=True. This kernel is the largest single weight read in a decode
step -- 47.9 MB per rank per KDA layer, 69 of those layers -- so its achieved
bandwidth sets a floor on low-concurrency TPOT.

Reports microseconds and achieved bandwidth for each candidate, and checks every
one against a dequantized reference before trusting its timing.

    python tune_kda_input_group64.py                # default sweep
    python tune_kda_input_group64.py --tokens 1 2   # both captured buckets
"""

import argparse
import itertools
import sys

import torch

HIDDEN = 7168
PADDED_OUTPUT = 6288
LOGICAL_OUTPUT = 6284
GROUP = 64
GROUPS_PER_ROW = HIDDEN // GROUP
FP8_MAX = 448.0

# FP8 weights plus one fp32 scale per group of 64.
WEIGHT_BYTES = LOGICAL_OUTPUT * HIDDEN + LOGICAL_OUTPUT * GROUPS_PER_ROW * 4


def build(num_tokens, **cfg):
    from sglang.kernels.ops.kimi_k3.flydsl.kernels.kimi_k3_kda_input_group64_gfx950 import (
        build_kimi_k3_kda_input_group64_module,
    )

    return build_kimi_k3_kda_input_group64_module(num_tokens=num_tokens, **cfg)


def make_inputs(num_tokens, device="cuda"):
    torch.manual_seed(0)
    source = torch.randn(LOGICAL_OUTPUT, HIDDEN, device=device, dtype=torch.float32)
    grouped = source.reshape(LOGICAL_OUTPUT, GROUPS_PER_ROW, GROUP)
    amax = grouped.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax)).contiguous()
    weight = (
        (grouped / scale[..., None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(LOGICAL_OUTPUT, HIDDEN)
        .contiguous()
    )
    hidden = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    return hidden, weight, scale


def reference(hidden, weight, scale):
    """Dequantize and matmul in fp32, then round once, like the kernel does."""
    deq = (
        weight.to(torch.float32).reshape(LOGICAL_OUTPUT, GROUPS_PER_ROW, GROUP)
        * scale[..., None]
    ).reshape(LOGICAL_OUTPUT, HIDDEN)
    out = hidden.to(torch.float32) @ deq.t()
    padded = torch.zeros(
        hidden.shape[0], PADDED_OUTPUT, device=hidden.device, dtype=torch.float32
    )
    padded[:, :LOGICAL_OUTPUT] = out
    return padded


def time_launcher(launcher, hidden, weights, scale, out, reps=32, iters=50, warmup=20):
    """Time one launch as the model executes it: inside a captured graph.

    Eager launches cost this kernel ~48 us against ~13 us replayed, so timing it
    eagerly ranks launch overhead rather than the kernel. Decode always replays a
    graph, so that is the only mode worth tuning.

    `weights` rotates over several buffers. One weight is 45 MB and would sit in
    the 256 MB last-level cache across a tight loop, reporting a bandwidth the
    model never sees -- it streams 69 distinct weights per step.
    """
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    hidden_arg, scale_arg, out_arg = ptr_arg(hidden), ptr_arg(scale), ptr_arg(out)
    weight_args = [ptr_arg(w) for w in weights]

    def body():
        for i in range(reps):
            launcher(
                hidden_arg,
                weight_args[i % len(weight_args)],
                scale_arg,
                out_arg,
                stream=torch.cuda.current_stream(hidden.device),
            )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
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
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--rows-per-wave", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--cache-modifier", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--cu-count", nargs="+", type=int, default=[256])
    parser.add_argument("--waves-per-eu", nargs="+", type=int, default=[0])
    parser.add_argument("--hidden-to-lds", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--weight-copies",
        type=int,
        default=8,
        help="distinct weight buffers to rotate over, so the "
        "sweep measures HBM rather than cache",
    )
    args = parser.parse_args()

    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  CUs={props.multi_processor_count}")
    print(f"weight traffic per launch: {WEIGHT_BYTES / 1e6:.2f} MB\n")

    for num_tokens in args.tokens:
        hidden, weight, scale = make_inputs(num_tokens)
        ref = reference(hidden, weight, scale)
        out = hidden.new_empty((num_tokens, PADDED_OUTPUT))
        # 8 x 45 MB overflows the last-level cache, matching the 69 distinct
        # per-layer weights a real decode step streams.
        weights = [weight] + [weight.clone() for _ in range(args.weight_copies - 1)]

        print(f"=== num_tokens={num_tokens} ===")
        print(
            f"{'rpw':>4} {'wcm':>4} {'cu':>4} {'wpe':>4} {'lds':>4} "
            f"{'us':>8} {'TB/s':>7} {'max_err':>9}"
        )
        results = []
        for rpw, wcm, cu, wpe, lds in itertools.product(
            args.rows_per_wave,
            args.cache_modifier,
            args.cu_count,
            args.waves_per_eu,
            args.hidden_to_lds,
        ):
            cfg = dict(
                rows_per_wave=rpw,
                cu_count=cu,
                waves_per_eu=wpe,
                weight_cache_modifier=wcm,
                hidden_to_lds=bool(lds),
            )
            try:
                launcher = build(num_tokens, **cfg)
            except Exception as exc:  # noqa: BLE001 - report and continue sweeping
                print(
                    f"{rpw:>4} {wcm:>4} {cu:>4} {wpe:>4} {lds:>4}  build failed: "
                    f"{type(exc).__name__}"
                )
                continue
            out.zero_()
            us = time_launcher(launcher, hidden, weights, scale, out)
            err = (out.to(torch.float32) - ref).abs().max().item()
            scale_ref = ref.abs().max().item()
            rel = err / max(scale_ref, 1e-6)
            bw = WEIGHT_BYTES / (us * 1e-6) / 1e12
            print(
                f"{rpw:>4} {wcm:>4} {cu:>4} {wpe:>4} {lds:>4} "
                f"{us:8.2f} {bw:7.2f} {rel:9.2e}"
            )
            if rel < 0.02:
                results.append((us, cfg, bw))

        if results:
            results.sort()
            best_us, best_cfg, best_bw = results[0]
            print(f"\nbest: {best_us:.2f} us ({best_bw:.2f} TB/s) {best_cfg}")
            shipped = [
                r
                for r in results
                if r[1]
                == dict(
                    rows_per_wave=2,
                    cu_count=256,
                    waves_per_eu=0,
                    weight_cache_modifier=2,
                    hidden_to_lds=True,
                )
            ]
            if shipped:
                print(
                    f"shipped config: {shipped[0][0]:.2f} us -> "
                    f"{shipped[0][0] / best_us:.3f}x speedup available"
                )
        print()


if __name__ == "__main__":
    sys.exit(main())
