"""
Kimi-K3 prefill all-reduce microbenchmark for ROCm (TP8, MI355X).

Times every all-reduce backend the serving stack can pick, at the bf16 shapes
a K3 prefill chunk produces:

    [N, 7168]   attention o_proj / dense-MLP / embedding all-reduce
    [N, 10752]  MoE [latent 3584 | shared 7168] pair all-reduce

Backends (columns):
    rccl         torch.distributed NCCL/RCCL all-reduce (in place)
    aiter_ca     AITER custom AR, ``use_new=True``: 1-stage < 80 KiB, 2-stage above
                 (what serving runs for any input <= AITER_CUSTOM_AR_MAX_SIZE)
    aiter_ca_old AITER custom AR legacy kernels (``use_new=False``)
    sgl_ca_1stage sglang CustomAllreduce with SGLANG_USE_1STAGE_ALLREDUCE=1,
                 i.e. the 1-stage kernel at every size
    qr_int4 / qr_int8 / qr_int6 / qr_fp
                 sglang QuickAllReduce at that ROCM_QUICK_REDUCE_QUANTIZATION
                 level (``fp`` is the unquantized fp16 two-shot; there is no
                 FP8 level). Quick-reduce's own 2 MiB minimum is bypassed so
                 the whole size range is measured.

The aiter and quick-reduce size gates are bypassed on purpose (CA is called
past its 64 MiB serving cutoff) so the crossover points are visible. The
``route`` column replays what GroupCoordinator would pick in serving today:
CA if ``should_custom_ar``, else QR(ROCM_QUICK_REDUCE_QUANTIZATION) if its gate
passes, else RCCL.

Timing is a back-to-back loop of ``--iters`` calls bracketed by HIP events after
a barrier, so it includes the CA copy-in staging (serving inputs are not
registered in eager prefill) and any launch gap, but not cross-rank compute
skew. Latency is the mean over ranks. ``--rel-err`` also reports the relative
RMS error of each backend versus RCCL on random data with a few outlier
channels (real activations have heavier tails, so treat INT4 error as a floor).

Run (8 GPUs, serving env):

    torchrun --nproc_per_node 8 benchmark/kernels/all_reduce/benchmark_k3_prefill_ar.py
"""

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist

MB = 1024 * 1024
DEFAULT_TOKENS = [1, 4, 16, 32, 64, 128, 256, 772, 2502, 4096, 8192]
DEFAULT_WIDTHS = [7168, 10752]
DEFAULT_BACKENDS = [
    "rccl",
    "aiter_ca",
    "aiter_ca_old",
    "sgl_ca_1stage",
    "qr_int4",
    "qr_int8",
    "qr_int6",
    "qr_fp",
]
QR_LEVELS = {"qr_int4": "INT4", "qr_int8": "INT8", "qr_int6": "INT6", "qr_fp": "FP"}
LOSSY = ("qr_int4", "qr_int8", "qr_int6")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    p.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    p.add_argument(
        "--backends",
        type=str,
        default=",".join(DEFAULT_BACKENDS),
        help="Comma-separated subset of: " + ",".join(DEFAULT_BACKENDS),
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30, help="Iterations for >= 1 MiB.")
    p.add_argument("--iters-small", type=int, default=200, help="Iterations < 1 MiB.")
    p.add_argument(
        "--ca-pool-mb",
        type=int,
        default=512,
        help="AITER/sglang CA input pool. AITER prefill needs 2x the message.",
    )
    p.add_argument(
        "--qr-max-mb",
        type=int,
        default=256,
        help="ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB for the QR instances (the "
        "largest message here is 176 MiB). 0 leaves it unset like serving, "
        "which allocates a 4 GiB uncached IPC buffer per rank.",
    )
    p.add_argument(
        "--rel-err",
        action="store_true",
        help="Also report relative RMS error vs RCCL for every backend.",
    )
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args()


def rank0(*a, **k):
    if dist.get_rank() == 0:
        print(*a, **k, flush=True)


def fmt_bytes(n: int) -> str:
    return f"{n / MB:.2f}M" if n >= MB else f"{n / 1024:.0f}K"


def make_input(n_tokens: int, width: int, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(n_tokens, width, device=device, generator=g, dtype=torch.float32)
    # A few outlier channels, like real post-o_proj activations.
    n_out = max(1, width // 1024)
    cols = torch.randperm(width, device=device, generator=g)[:n_out]
    x[:, cols] *= 20.0
    return x.to(torch.bfloat16).contiguous()


class Backends:
    """Owns the communicators; every method must be entered by all ranks."""

    def __init__(self, args, cpu_group, nccl_group, device):
        self.args = args
        self.cpu_group = cpu_group
        self.nccl_group = nccl_group
        self.device = device
        self.names: List[str] = []
        self.aiter = None
        self.sgl_1stage = None
        self.qr: Dict[str, object] = {}
        self.errors: Dict[str, str] = {}
        self.serving_qr = None
        wanted = [b for b in args.backends.split(",") if b]
        unknown = [b for b in wanted if b not in DEFAULT_BACKENDS]
        if unknown:
            raise SystemExit(f"unknown backends: {unknown}")
        self.wanted = wanted

    def _try(self, name: str, ctor: Callable):
        try:
            obj = ctor()
            if obj is None or getattr(obj, "disabled", False):
                self.errors[name] = "disabled"
                return None
            return obj
        except Exception as e:  # noqa: BLE001
            self.errors[name] = f"{type(e).__name__}: {e}"[:120]
            return None

    def setup(self):
        a = self.args
        pool = a.ca_pool_mb * MB
        # Always built: the ``route`` column needs its size gate.
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as AiterCA,
        )

        self.aiter = self._try(
            "aiter_ca",
            lambda: AiterCA(
                group=self.cpu_group,
                device=self.device,
                max_size=pool,
                enable_register_for_capturing=False,
            ),
        )
        if "sgl_ca_1stage" in self.wanted:
            from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce as SglCA,
            )

            old = os.environ.get("SGLANG_USE_1STAGE_ALLREDUCE")
            os.environ["SGLANG_USE_1STAGE_ALLREDUCE"] = "1"
            try:
                self.sgl_1stage = self._try(
                    "sgl_ca_1stage",
                    lambda: SglCA(
                        group=self.cpu_group, device=self.device, max_size=pool
                    ),
                )
            finally:
                if old is None:
                    os.environ.pop("SGLANG_USE_1STAGE_ALLREDUCE", None)
                else:
                    os.environ["SGLANG_USE_1STAGE_ALLREDUCE"] = old
        qr_names = [b for b in self.wanted if b in QR_LEVELS]
        serving_level = os.environ.get("ROCM_QUICK_REDUCE_QUANTIZATION", "NONE")
        if qr_names:
            from sglang.srt.distributed.device_communicators.quick_all_reduce import (
                QuickAllReduce,
            )

            if a.qr_max_mb > 0:
                os.environ["ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB"] = str(a.qr_max_mb)
            # The QR data buffer size is a process-wide static in the kernel
            # library, so the first instance fixes it for all of them.
            for name in qr_names:
                os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = QR_LEVELS[name]
                self.qr[name] = self._try(
                    name,
                    lambda: QuickAllReduce(group=self.cpu_group, device=self.device),
                )
            os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = serving_level
            self.serving_qr = self.qr.get(
                next((n for n, lv in QR_LEVELS.items() if lv == serving_level), ""),
            )

    def make_op(self, name: str, inp: torch.Tensor) -> Optional[Callable]:
        """Return a zero-arg closure doing one all-reduce, or None if unavailable."""
        if name == "rccl":
            buf = torch.zeros_like(inp)  # in place; zeros cannot overflow
            return lambda: dist.all_reduce(buf, group=self.nccl_group)
        if name in ("aiter_ca", "aiter_ca_old"):
            ca = self.aiter
            if ca is None or inp.numel() * inp.element_size() > ca.max_size / 2:
                return None
            use_new = name == "aiter_ca"
            return lambda: ca.all_reduce(inp, use_new=use_new, registered_input=False)
        if name == "sgl_ca_1stage":
            ca = self.sgl_1stage
            if ca is None or inp.numel() * inp.element_size() >= ca.max_size:
                return None
            return lambda: ca.custom_all_reduce(inp)
        if name in QR_LEVELS:
            qr = self.qr.get(name)
            if qr is None or inp.numel() * inp.element_size() > qr.qr_max_size:
                return None
            return lambda: qr.quick_all_reduce(inp)
        return None

    def serving_route(self, inp: torch.Tensor) -> str:
        ca = self.aiter
        if ca is not None and ca.should_custom_ar(inp):
            return "CA1" if ca.uses_1stage_ar(inp) else "CA2"
        qr = self.serving_qr
        if qr is not None and qr.should_quick_allreduce(inp):
            return "QR"
        return "RCCL"

    def close(self):
        for comm in [self.aiter, self.sgl_1stage, *self.qr.values()]:
            if comm is not None and hasattr(comm, "close"):
                try:
                    comm.close()
                except Exception:  # noqa: BLE001
                    pass


def time_op(fn: Callable, warmup: int, iters: int, cpu_group) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


def mean_over_ranks(value: float, device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return float(t.item()) / dist.get_world_size()


def rel_rms_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    o, r = out.float(), ref.float()
    return float(((o - r).pow(2).sum() / r.pow(2).sum().clamp_min(1e-30)).sqrt().item())


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", device_id=device, rank=rank, world_size=world
    )
    if world != 8:
        rank0(f"WARNING: world_size={world}; the K3 recipe is TP8.", file=sys.stderr)
    cpu_group = dist.new_group(backend="gloo")
    nccl_group = dist.group.WORLD

    rank0(
        "env: "
        + " ".join(
            f"{k}={os.environ.get(k)}"
            for k in (
                "ROCM_QUICK_REDUCE_QUANTIZATION",
                "ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB",
                "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16",
                "AITER_CUSTOM_AR_MAX_SIZE",
                "AITER_CUSTOM_AR_MIN_SIZE",
                "NCCL_MIN_NCHANNELS",
                "PYTORCH_HIP_ALLOC_CONF",
            )
        )
    )

    be = Backends(args, cpu_group, nccl_group, device)
    be.setup()
    dist.barrier(group=cpu_group)
    if be.errors:
        rank0("unavailable backends:", be.errors)
    active = [b for b in be.wanted if b == "rccl" or b not in be.errors]

    results = []
    for width in args.widths:
        for n_tokens in args.tokens:
            inp = make_input(n_tokens, width, device, seed=1234 + rank)
            nbytes = inp.numel() * inp.element_size()
            iters = args.iters_small if nbytes < MB else args.iters
            row = {
                "tokens": n_tokens,
                "width": width,
                "bytes": nbytes,
                "route": be.serving_route(inp),
                "us": {},
                "rel_err": {},
            }
            ref = None
            if args.rel_err:
                ref = inp.clone()
                dist.all_reduce(ref, group=nccl_group)
            for name in active:
                op = be.make_op(name, inp)
                has_op = torch.tensor([0 if op is None else 1], device=device)
                dist.all_reduce(has_op, op=dist.ReduceOp.MIN, group=nccl_group)
                if not int(has_op.item()):
                    continue
                try:
                    us = time_op(op, args.warmup, iters, cpu_group)
                    row["us"][name] = mean_over_ranks(us, device)
                    if args.rel_err and name != "rccl":
                        out = be.make_op(name, inp)()
                        torch.cuda.synchronize()
                        row["rel_err"][name] = mean_over_ranks(
                            rel_rms_error(out, ref), device
                        )
                except Exception as e:  # noqa: BLE001
                    rank0(f"[{name}] N={n_tokens} W={width} failed: {e}")
            results.append(row)
            del inp, ref
            torch.cuda.empty_cache()

    if rank == 0:
        print_tables(results, active, args)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(results, f, indent=1)
            print(f"wrote {args.json_out}")

    be.close()
    dist.barrier()
    dist.destroy_process_group()


def print_tables(results, active, args):
    cols = [b for b in active]
    for width in args.widths:
        rows = [r for r in results if r["width"] == width]
        print(f"\n=== bf16 [N x {width}], TP8 all-reduce latency (us, mean over ranks)")
        head = f"{'N':>6} {'size':>8} {'route':>5} " + " ".join(
            f"{c:>13}" for c in cols
        )
        head += f" {'best':>13} {'best GB/s':>9}"
        print(head)
        print("-" * len(head))
        for r in rows:
            cells = []
            for c in cols:
                v = r["us"].get(c)
                cells.append(f"{v:13.1f}" if v is not None else f"{'-':>13}")
            best = min(r["us"], key=r["us"].get) if r["us"] else "-"
            bus = r["us"].get(best)
            gbs = f"{r['bytes'] / (bus * 1e-6) / 1e9:9.0f}" if bus else f"{'-':>9}"
            print(
                f"{r['tokens']:>6} {fmt_bytes(r['bytes']):>8} {r['route']:>5} "
                + " ".join(cells)
                + f" {best:>13} {gbs}"
            )
        print("best GB/s = message bytes / best latency (algorithm bandwidth)")
    if args.rel_err:
        lossy = [c for c in cols if c in LOSSY]
        for width in args.widths:
            print(f"\n=== relative RMS error vs RCCL, bf16 [N x {width}]")
            head = f"{'N':>6} " + " ".join(f"{c:>10}" for c in lossy)
            print(head)
            for r in [r for r in results if r["width"] == width]:
                print(
                    f"{r['tokens']:>6} "
                    + " ".join(
                        f"{r['rel_err'].get(c, float('nan')):10.4f}" for c in lossy
                    )
                )
    print(
        "\nroute = backend serving picks today (CA1/CA2 = AITER custom AR, QR, RCCL)"
        " with the env above."
    )


if __name__ == "__main__":
    main()
