#!/usr/bin/env python3
"""Attribute small "glue" GPU kernels in step[<KIND>] windows to their aten ops.

For every kernel launched inside a ``step[KIND bs=N]`` CPU span of a torch
profiler trace (record_shapes=True), find the chain of enclosing cpu_op events
(aten::to > aten::_to_copy > aten::copy_, ...), take the shapes/dtypes of the
outermost aten op, and group by (kernel class, op chain, shapes). The output
gives per-step kernel counts and GPU time, plus the most common neighbouring
non-glue kernels so a group can be located in the model code.

Usage:
  attribute_glue_kernels.py TRACE.json.gz [--kind VERIFY] [--classes memcpy,elementwise,cat,index,fill]
                            [--max-us 20] [--anchors 2] [--json out.json]

Kernel classes: memcpy (DtoD/HtoD/DtoH), memset, elementwise, cat, index,
fill, reduce, copy (copy_ kernels), other_torch (any other at::native kernel),
and ``nontorch`` (everything else, only with --classes nontorch/all).
"""

import argparse
import bisect
import gzip
import json
import re
import sys
from collections import Counter, defaultdict

STEP_RE = re.compile(r"step\[(\w+)\s+bs=(\d+)(?:\s+toks=(\d+))?\]")
KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver", "hip_runtime"}

# Ops that only wrap the interesting one; they are dropped from the printed chain.
_BORING = {"aten::empty", "aten::empty_like", "aten::empty_strided", "aten::resize_"}


def classify(ev):
    cat, name = ev["cat"], ev["name"]
    if cat == "gpu_memcpy":
        return "memcpy"
    if cat == "gpu_memset":
        return "memset"
    if "CatArrayBatchedCopy" in name:
        return "cat"
    if (
        "index_elementwise_kernel" in name
        or "index_put" in name
        or "IndexKernel" in name
    ):
        return "index"
    if "FillFunctor" in name or "fill_kernel" in name:
        return "fill"
    if "reduce_kernel" in name or "cunn_SoftMax" in name:
        return "reduce"
    if "CopyFunctor" in name or "direct_copy_kernel" in name:
        return "copy"
    if "elementwise_kernel" in name or "vectorized_elementwise" in name:
        return "elementwise"
    if "at::native" in name or "at_cuda_detail" in name or "at::cuda" in name:
        return "other_torch"
    return "nontorch"


def short_kernel(name):
    name = re.sub(r"\bvoid\s+", "", name)
    name = re.sub(r"at::native::", "", name)
    name = re.sub(r"\(.*", "", name)
    name = re.sub(r"<.*", "", name)
    return name.strip()[:60]


def fmt_dims(dims, types):
    out = []
    for d, t in zip(dims or [], types or []):
        if d in ([], None) and t in ("Scalar", "ScalarList", "", None):
            continue
        out.append(f"{list(d)}:{t}" if d else f"{t}")
    return "(" + ", ".join(out) + ")"


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        doc = json.load(fh)
    return doc["traceEvents"] if isinstance(doc, dict) else doc


def build_index(events):
    steps, kernels, launches = [], [], {}
    ops_by_tid = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat")
        if cat == "user_annotation":
            m = STEP_RE.search(ev["name"])
            if m:
                steps.append((ev["ts"], ev["ts"] + ev.get("dur", 0.0), m.groups()))
        elif cat in KERNEL_CATS:
            kernels.append(ev)
        elif cat in LAUNCH_CATS:
            corr = (ev.get("args") or {}).get("correlation")
            if corr is not None:
                launches[corr] = ev
        elif cat == "cpu_op":
            ops_by_tid[ev["tid"]].append(ev)
    steps.sort()
    for ops in ops_by_tid.values():
        ops.sort(key=lambda e: (e["ts"], -e.get("dur", 0.0)))
    return steps, kernels, launches, ops_by_tid


class OpStacks:
    """Enclosing cpu_op chain (outermost first) for a timestamp on a thread."""

    def __init__(self, ops_by_tid):
        self.ops = ops_by_tid
        self.starts = {t: [o["ts"] for o in ops] for t, ops in ops_by_tid.items()}

    def chain(self, tid, ts):
        ops = self.ops.get(tid)
        if not ops:
            return []
        i = bisect.bisect_right(self.starts[tid], ts)
        # Walk back until the span containing ts is found, then climb parents.
        chain, end_limit = [], None
        j = i - 1
        # Arbitrary scan bound; enough for the op nesting seen in K3 traces.
        lo = max(0, i - 400)
        while j >= lo:
            o = ops[j]
            if o["ts"] <= ts <= o["ts"] + o.get("dur", 0.0) and (
                end_limit is None or o["ts"] + o.get("dur", 0.0) >= end_limit
            ):
                chain.append(o)
                end_limit = o["ts"] + o.get("dur", 0.0)
            j -= 1
        chain.reverse()
        return chain


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("trace")
    ap.add_argument(
        "--kind", default="VERIFY", help="step kind (VERIFY, DECODE, DRAFT, EXTEND...)"
    )
    ap.add_argument(
        "--classes",
        default="memcpy,memset,elementwise,cat,index,fill,copy,reduce,other_torch",
        help="comma list of kernel classes, or 'all'",
    )
    ap.add_argument(
        "--max-us", type=float, default=None, help="only kernels shorter than this"
    )
    ap.add_argument(
        "--anchors",
        type=int,
        default=2,
        help="neighbour (prev>next) pairs shown per group",
    )
    ap.add_argument(
        "--min-count", type=float, default=0.0, help="hide groups below this count/step"
    )
    ap.add_argument("--json", help="write the grouped table to this file")
    args = ap.parse_args()

    events = load(args.trace)
    steps, kernels, launches, ops_by_tid = build_index(events)
    stacks = OpStacks(ops_by_tid)
    starts = [s[0] for s in steps]
    want = None if args.classes == "all" else set(args.classes.split(","))

    # Kernels per step in launch order, so neighbours can be reported.
    per_step = defaultdict(list)
    for ev in kernels:
        launch = launches.get((ev.get("args") or {}).get("correlation"))
        if launch is None:
            continue
        i = bisect.bisect_right(starts, launch["ts"]) - 1
        if i < 0 or launch["ts"] > steps[i][1] or steps[i][2][0] != args.kind:
            continue
        per_step[i].append((launch["ts"], launch, ev))
    n_steps = len(per_step)
    if not n_steps:
        sys.exit(f"no {args.kind} steps in trace")
    bs = Counter(steps[i][2][1] for i in per_step)

    groups = defaultdict(
        lambda: {
            "n": 0,
            "us": 0.0,
            "anchors": Counter(),
            "cls": None,
            "kern": Counter(),
        }
    )
    total_k, total_us = 0, 0.0
    glue_k, glue_us = 0, 0.0
    for i, items in per_step.items():
        items.sort(key=lambda x: x[0])
        is_glue = [classify(e) != "nontorch" for _, _, e in items]
        for idx, (ts, launch, ev) in enumerate(items):
            total_k += 1
            total_us += ev.get("dur", 0.0)
            cls = classify(ev)
            if want is not None and cls not in want:
                continue
            if args.max_us is not None and ev.get("dur", 0.0) >= args.max_us:
                continue
            glue_k += 1
            glue_us += ev.get("dur", 0.0)
            chain = stacks.chain(launch["tid"], launch["ts"])
            aten = [o for o in chain if o["name"].startswith("aten::")]
            parent_custom = next(
                (
                    o["name"]
                    for o in reversed(chain)
                    if not o["name"].startswith("aten::")
                ),
                "-",
            )
            shown = [o["name"] for o in aten if o["name"] not in _BORING]
            # Shapes come from the outermost aten op; the innermost is appended when it differs.
            root = aten[0] if aten else None
            a = root["args"] if root else {}
            shapes = (
                fmt_dims(a.get("Input Dims"), a.get("Input type")) if root else "()"
            )
            inner = aten[-1] if aten else None
            if inner is not None and inner is not root:
                ia = inner["args"]
                shapes += (
                    " => "
                    + inner["name"].replace("aten::", "")
                    + fmt_dims(ia.get("Input Dims"), ia.get("Input type"))
                )
            key = (
                cls,
                ">".join(n.replace("aten::", "") for n in shown) or "-",
                shapes,
                parent_custom,
            )
            g = groups[key]
            g["n"] += 1
            g["us"] += ev.get("dur", 0.0)
            g["cls"] = cls
            g["kern"][short_kernel(ev["name"])] += 1
            prev = next(
                (
                    short_kernel(items[j][2]["name"])
                    for j in range(idx - 1, -1, -1)
                    if not is_glue[j]
                ),
                "<start>",
            )
            nxt = next(
                (
                    short_kernel(items[j][2]["name"])
                    for j in range(idx + 1, len(items))
                    if not is_glue[j]
                ),
                "<end>",
            )
            g["anchors"][f"{prev} > {nxt}"] += 1

    print(f"trace: {args.trace}")
    print(
        f"{args.kind} steps: {n_steps} (bs {dict(bs)}); all kernels/step {total_k / n_steps:.0f}, "
        f"GPU {total_us / n_steps:.0f} us/step"
    )
    print(
        f"selected: {glue_k / n_steps:.1f} kernels/step, {glue_us / n_steps:.1f} us/step\n"
    )
    print(
        f"{'n/step':>7} {'us/step':>8} {'us/k':>6}  class        aten chain [shapes]  <- custom-op parent"
    )
    rows = sorted(groups.items(), key=lambda kv: -kv[1]["us"])
    out = []
    for (cls, chain, shapes, parent), g in rows:
        n = g["n"] / n_steps
        if n < args.min_count:
            continue
        print(
            f"{n:7.2f} {g['us'] / n_steps:8.1f} {g['us'] / g['n']:6.2f}  {cls:11s} {chain} {shapes}  <- {parent}"
        )
        print(
            f"{'':27s}kernel: {', '.join(f'{k} x{c / n_steps:.1f}' for k, c in g['kern'].most_common(2))}"
        )
        for anchor, c in g["anchors"].most_common(args.anchors):
            print(f"{'':27s}between: {anchor}  (x{c / n_steps:.1f})")
        out.append(
            {
                "class": cls,
                "aten_chain": chain,
                "shapes": shapes,
                "custom_parent": parent,
                "count_per_step": n,
                "us_per_step": g["us"] / n_steps,
                "kernels": dict(g["kern"]),
                "anchors": dict(g["anchors"].most_common(5)),
            }
        )
    by_cls = defaultdict(lambda: [0, 0.0])
    for g in groups.values():
        by_cls[g["cls"]][0] += g["n"]
        by_cls[g["cls"]][1] += g["us"]
    print(
        "\nper class: "
        + ", ".join(
            f"{c}: {v[0] / n_steps:.1f}/step {v[1] / n_steps:.0f}us"
            for c, v in sorted(by_cls.items())
        )
    )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"steps": n_steps, "groups": out}, fh, indent=1)


if __name__ == "__main__":
    main()
