"""Graph microbenchmark for Kimi-K3 preroute projections."""

import argparse
import statistics

import torch

from sglang.kernels.ops.moe.kimi_k3_flydsl.kimi_k3_moe_preroute_fp8 import (
    kimi_k3_moe_tri_projection_cooperative_preactivated_fp8,
    kimi_k3_moe_tri_projection_fp8,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.utils import is_in_ci

# Exits immediately in CI; run by hand on gfx950.
register_amd_ci(est_time=5, stage="jit-kernel-benchmark", runner_config="amd")

# Weight copies each graph replay walks through: serving reads every layer's
# weights once per step, so ~8 x 50 MB keeps them out of the 256 MB MALL.
_COPIES = 8


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = weight.float()
    scale = value.abs().amax(dim=1).clamp_min(1e-12) / 448.0
    return (
        (value / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous(),
        scale.contiguous(),
    )


def bench(fns, warmup: int, iters: int, trials: int) -> tuple[float, float, float]:
    """Per-call latency of ``fns`` (one closure per weight copy), each replay
    running every copy once so no call finds its weights already cached."""

    def step():
        for fn in fns:
            fn()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (iters * len(fns)))
    ordered = sorted(samples)
    return (
        statistics.median(samples),
        ordered[max(0, trials // 10 - 1)],
        ordered[min(trials - 1, 9 * trials // 10)],
    )


def _make_fns(hidden: torch.Tensor, tokens: int, device: torch.device):
    """The FlyDSL tri-projection and its BF16 GEMM baseline over one fresh
    copy of the weights."""
    routed_bf16 = torch.randn((3584, 7168), dtype=torch.bfloat16, device=device)
    shared_bf16 = torch.randn((1536, 7168), dtype=torch.bfloat16, device=device)
    router = torch.randn((896, 7168), dtype=torch.bfloat16, device=device)
    routed, routed_scale = quantize_rows(routed_bf16)
    shared, shared_scale = quantize_rows(shared_bf16)
    merged = torch.cat((shared_bf16, router, routed_bf16), dim=0).contiguous()
    if tokens in (2, 4):
        shared = (
            shared.view(2, 768, 7168).permute(1, 0, 2).contiguous().view(1536, 7168)
        )
        shared_scale = shared_scale.view(2, 768).t().contiguous().view(1536)

        def rowdot_fn():
            return kimi_k3_moe_tri_projection_cooperative_preactivated_fp8(
                hidden,
                routed,
                routed_scale,
                shared,
                shared_scale,
                router,
                situ_beta=4.0,
                situ_linear_beta=25.0,
                fast_situ=True,
            )

    else:

        def rowdot_fn():
            return kimi_k3_moe_tri_projection_fp8(
                hidden, routed, routed_scale, shared, shared_scale, router
            )

    return rowdot_fn, lambda: torch.mm(hidden, merged.t())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--trials", type=int, default=11)
    args = parser.parse_args()
    torch.manual_seed(20260817)
    device = torch.device("cuda")
    hidden = torch.randn((args.tokens, 7168), dtype=torch.bfloat16, device=device)
    rowdot_fns, bf16_fns = [], []
    for _ in range(_COPIES):
        rowdot_fn, bf16_fn = _make_fns(hidden, args.tokens, device)
        rowdot_fns.append(rowdot_fn)
        bf16_fns.append(bf16_fn)
    rowdot = bench(rowdot_fns, args.warmup, args.iters, args.trials)
    bf16 = bench(bf16_fns, args.warmup, args.iters, args.trials)
    print(
        f"M={args.tokens} rowdot_p50_us={rowdot[0]:.3f} "
        f"rowdot_p10_us={rowdot[1]:.3f} rowdot_p90_us={rowdot[2]:.3f} "
        f"bf16_p50_us={bf16[0]:.3f} "
        f"rowdot_speedup={bf16[0] / rowdot[0]:.3f}x"
    )


if __name__ == "__main__":
    if is_in_ci():
        print("Skipping bench_kimi_k3_moe_preroute_fp8.py in CI")
        raise SystemExit(0)
    main()
