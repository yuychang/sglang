# ROCm Kimi K2.5 MXFP4 graph-decode optimization

This document summarizes the retained Kimi-K2.5 MXFP4 optimizations validated
on 4× AMD MI355X, TP=4. The SGLang and AITER changes are designed as one
coordinated stack; unsupported models, shapes, dtypes, and execution modes
retain their existing implementations.

## Retained optimization stack

### AITER grouped top-k shared-column append

When shared experts are fused into routed MoE, AITER's grouped top-k kernel
writes the routed top-k columns and constant shared-expert columns in one
launch. This removes the separate append kernel while preserving routed
renormalization and scaling semantics. Separate-shared deployments continue to
produce the normal routed-only top-k output.

### Fused route sort and once-per-token MXFP4 quantization

For gfx950 graph-decode shapes with hidden size 7168, E=384/top-k=8 or
E=385/top-k=9, AITER replaces the generic route-sort plus quant/sort sequence
with one specialized launch:

```text
top-k ids/weights ─┐
hidden BF16 ───────┼─ fused sort + compact MXFP4 quant
                   ├─ sorted token/expert metadata
                   ├─ compact FP4 activation [M, H/2]
                   └─ compact E8M0 scale [M, H/32]
```

The producer quantizes each original token once rather than once per routed
row. It also zeros the routed output buffer in the same launch.

### Compact-scale FlyDSL GEMM1 for M>=8

For graph batches M=8,16,32,64,128, FlyDSL GEMM1 consumes compact token scales
directly. The kernel reuses sorted token ids already staged in LDS, gathers two
compact scale dwords per routed row pair, and repacks the four E8M0 bytes in
registers for the existing scaled-MFMA interface.

This removes the standalone compact-to-sorted scale conversion. Tier-specific
N tiles and wave settings are used for M=8,16,32,64; M=128 keeps its tuned
kernel shape. M<=4 deliberately retains the conventional scale conversion
because direct compact-scale consumption regressed serving TPOT.

The compact consumer follows `SGLANG_AITER_FUSED_DECODE_TOPK_SORT` by default.
It can be disabled independently:

```bash
export SGLANG_AITER_FUSED_DECODE_COMPACT_SCALE=0
```

### Shared/dense MLP activation and quantization fusion

On the separate-shared ROCm path, the shared MLP fuses SiLU×up with dynamic
MXFP4 quantization before the down projection. The down projection consumes the
packed FP4 activation and compact E8M0 scale directly, avoiding a standalone
activation tensor and quantization pass.

### MLA value projection and output quantization fusion

During AITER graph decode, the absorbed MLA value projection can emit the
flattened per-1x32 MXFP4 activation and E8M0 scales consumed by `o_proj`
directly from its GEMM accumulator. This removes the per-layer BF16
`[tokens, heads, v_head_dim]` intermediate and the following flatten/quantize
launch. The epilogue rounds through BF16 before quantization to preserve the
split path's numerical boundary.

The path is limited to non-split AITER GEMM configurations whose output tile is
aligned to a complete MXFP4 quantization group. Unsupported configurations and
active `kv_b_proj` LoRAs retain the split implementation. It can be disabled
for A/B testing:

```bash
export SGLANG_ROCM_FUSE_MLA_VALUE_MXFP4_QUANT=0
```

### Dedicated MoE stream

The routed branch stays on the graph/main stream while the shared MLP runs on a
dedicated MoE stream. The MoE stream is intentionally distinct from the MLA
alternate stream so MLA graph capture can retain its faster gfx950 fused norm
path. Multi-stream execution requires enough ROCm hardware queues to avoid
queue-level serialization.

### Fused shared add, all-reduce, residual, and RMSNorm

For separate-shared deployments, routed and shared TP-local outputs can remain
separate until the next layer. AITER consumes both registered graph buffers in
one fused all-reduce+residual+RMSNorm operation, removing the full-hidden-state
local add from the critical path.

### Tuned one-stage/two-stage fused AR policy

For TP=4 and hidden size 7168, fused AR+RMSNorm uses a measured byte/token
crossover rather than a single global cutoff. The two-input path uses one-stage
through M=8 and the partitioned two-stage path for M=16/32.

## Rejected designs not present in the active path

The following experiments were measured and removed rather than left behind as
runtime flags or alternate source paths:

- true bypass top-k and deferred MoE finalize;
- direct compact-scale GEMM1 for M<=4;
- cooperative-grid sort/quant plus scale materialization;
- block-16 routed GEMM tiles;
- existing one-stage per-token MXFP4 MoE binaries;
- quantize-on-load, which repeats activation quantization for top-k routes.

Block-16 reduced padded MFMA rows but not expert weight traffic and remained
slower. The best available one-stage MXFP4 binary was about 10 microseconds
slower than the tuned two-stage M=8 path.

Matched 1k/1k serving sweeps for the retained M>=8 compact-scale policy:

| Concurrency | Converted scale | Compact scale | Delta |
|---:|---:|---:|---:|
| 8 | 172.50 tok/s/GPU | 188.10 tok/s/GPU | +9.04% |
| 16 | 275.00 tok/s/GPU | 284.51 tok/s/GPU | +3.46% |
| 32 | 390.98 tok/s/GPU | 395.21 tok/s/GPU | +1.08% |
| 64 | 562.60 tok/s/GPU | 566.87 tok/s/GPU | +0.76% |

## Topology

The retained separate-shared implementation runs routed MoE on the graph/main
stream and the shared MLP on a dedicated `moe_alt_stream`. Previously, after
the streams joined, SGLang launched a full-hidden-state BF16 add before the
next layer's fused all-reduce and RMSNorm.

For supported HIP-graph decode shapes, routed and shared outputs now remain
separate until the next layer:

```text
routed_local ─┐
              ├─ AITER fused two-input AR + residual + RMSNorm
shared_local ─┘
```

## AITER registration design

AITER custom AR never treats a local tensor pointer as sufficient. Its kernels
consume a `RankData` table containing peer addresses. The two-input API obtains:

- one `RankData*` for routed input;
- one `RankData*` for shared input.

Both use the existing graph registration lifecycle. During capture, unknown
persistent tensor addresses are recorded. After capture,
`register_graph_buffers()` exchanges IPC handles and writes the peer pointer
tables. Graph replay launches the fused kernel directly. No payload is copied.

## Numerical semantics

The implementation preserves:

```python
local_sum = (routed.float() + shared.float()).to(torch.bfloat16)
global_sum = all_reduce(local_sum)  # BF16 result
residual_out = (global_sum.float() + residual.float()).to(torch.bfloat16)
norm_out = rms_norm(residual_out, weight, eps)
```

Packed BF16x2 addition is used for the local and residual adds. RMSNorm uses
the rounded residual output.

## Gate

Enabled only when all conditions hold:

- HIP/ROCm gfx950;
- Kimi-K2.5 Quark/OCP MXFP4;
- separate shared expert;
- HIP graph capture;
- TP=4;
- BF16 contiguous `[M, 7168]`;
- M=1,2,4,8,16,32;
- next-layer all-reduce fusion is available;
- not the final layer.

Fallback executes the existing local add. One-stage is selected through M=8;
two-stage is selected for M=16/32.

## Validation

Profiler, summed across four ranks for one c4 decode step:

- add-shared: 240 -> 4 launches;
- total fused AR: 484 -> 484;
- two-input fused AR: 236 launches;
- copy-like launches: 40 -> 40;
- total GPU kernels: 5,960 -> 5,724.

Final three-run serving medians:

| Workload | Dedicated before | Two-input | Delta |
|---|---:|---:|---:|
| 1k/1k c4 | 87.51 | 88.70 | +1.36% |
| 1k/1k c16 | 241.67 | 239.29 | -0.98% |
| 8k/1k c4 | 83.63 | 85.67 | +2.44% |

Fused shared remains faster: 95.14, 264.27, and 89.36 tok/s/GPU,
respectively.

Correctness passed for the complete AITER M matrix, repeated SGLang graph
replay, unit fallback/carrier checks, and the final five-prompt graph smoke.

## Deployment

The optimization is enabled by default inside the already opt-in ROCm
multi-stream separate-shared path. It can be disabled for A/B testing:

```bash
export SGLANG_ROCM_FUSE_SHARED_PARTIAL_AR_RMSNORM=0
```

Fused shared experts remain the recommended maximum-throughput default. The
two-input path is useful when separate shared experts are required and the
deployment target is low concurrency.
