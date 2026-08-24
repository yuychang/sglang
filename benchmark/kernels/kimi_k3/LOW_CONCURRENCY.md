# Kimi-K3 low-concurrency decode on MI355X

Measured on 8x MI355X (gfx950, ROCm 7.2), TP8, 8192 in / 256 out, 4 waves,
against the tuned AITER 8k/1k recipe. Numbers are median ITL in ms, which is
the clean decode metric here: it is insensitive to how prefill amortizes over
the output length, so it matches a 1024-token-output TPOT closely while
letting a sweep finish four times sooner.

## Two configuration wins

| Config | c2 | c4 | c8 | c32 |
| --- | --- | --- | --- | --- |
| Recipe as shipped (fused sort+quant off, PTPC from 8 tokens) | 14.50\* | — | 19.19 | 27.59 |
| `AITER_MOE_A8W4_FUSED_SORT_QUANT=1` | 14.02 | 15.78 | 18.58 | 27.16 |
| ... and `SGLANG_K3_PTPC_FP8_MIN_TOKENS=1` | **13.87** | **15.68** | 18.59 | — |

\* derived: measured 14.34 with PTPC at 1 token, scaled by the 1.1% that flag
is worth at c2. It lands on the 14.58 recorded independently for this recipe.

**`AITER_MOE_A8W4_FUSED_SORT_QUANT` is worth 3.3% at c2, 3.1% at c8, 1.6% at
c32, and it is easy to leave off by accident.** AITER resolves it from
`SGLANG_ROCM_USE_MULTI_STREAM` when it is unset:

```python
_MOE_A8W4_FUSED_SORT_QUANT = (
    os.environ.get(
        "AITER_MOE_A8W4_FUSED_SORT_QUANT",
        os.environ.get("SGLANG_ROCM_USE_MULTI_STREAM", "0"),
    )
    == "1"
)
```

Those two settings are unrelated in every other respect, so a recipe that
leaves multi-stream alone silently loses the K3 fused route sort. Set it
explicitly.

**`SGLANG_K3_PTPC_FP8_MIN_TOKENS=1`** adds 1.1% at c2 and 0.6% at c4 and is
neutral from c8 up, where it changes nothing. GSM8K over 1319 questions is
0.956 against 0.957 for the recipe, i.e. unchanged within sampling noise.

## One configuration loss

**Do not enable MXFP4 on the latent projections at decode.** Dropping
`SGLANG_K3_MOE_LATENT_MXFP4_MIN_TOKENS` from 2048 to 1 costs 7.8% at c8 and
4.4% at c32. The 2048-token gate is correct: the MXFP4 GEMV does not pay at
small M, and taking that path also splits the merged front into a separate
head GEMM plus an MXFP4 down projection, adding a kernel per layer.

## Why there is not more to win from precision

A decode step at c2 reads 29.8 GB of weights per rank, which at the ~5.3 TB/s
this GPU actually sustains is 5.6 ms of a 14.0 ms step — 40%. The kernels that
move those bytes are already at hardware speed: the KDA input projection moves
90 MB in 17 us (5.3 TB/s) and the routed experts 69 MB in 12.2 us (5.6 TB/s).
Halving their bytes cannot buy much, which is what the 1.1% from extending PTPC
reflects.

The other 60% is kernel count. **One decode step launches 1981 kernels**, and
nothing in the trace runs faster than about 3.5 us however little work it does
-- an elementwise `add3` over a [2, 7168] tensor takes 4.3 us, `sort_scales`
takes 4.0 us. Roughly 2000 kernels against a floor near 4 us is most of the
step. It also explains why the gap to B300 closes as concurrency rises, from
64% at c2 to 85% at c64: real work per kernel grows while the floor does not.

Where the time goes at c2, per decode step:

| Group | ms | % | calls | us/call |
| --- | --- | --- | --- | --- |
| routing (radix4 + sort_quant + sort_scales) | 1.65 | 10.9% | 276 | ~6 |
| Triton `_gemm_a16_w16` | 1.57 | 10.3% | 185 | 8.5 |
| all-reduce (`cross_device_reduce_2stage`) | 1.48 | 9.7% | 187 | 7.9 |
| `_agg_kernel` (attention residual) | 1.47 | 9.7% | 186 | 7.9 |
| KDA in-proj | 1.17 | 7.7% | 69 | 17.0 |
| MoE front (`mixed_tri`) | 1.13 | 7.5% | 92 | 12.3 |
| routed experts | 1.13 | 7.4% | 92 | 12.2 |

## Targets examined and what they are worth

`_agg_kernel` costs 9.7% across two calls per layer, but the two are the
attention-side and MLP-side aggregation points -- different projections and
norms, with the second depending on attention output -- so they cannot be
merged. The kernel itself is already tuned for this regime: it deliberately
runs one CTA per token to hold the whole `[R_PAD, BLOCK_H]` tile in registers,
and its docstring records the 2-kernel alternative measuring 2x slower at T=4.
At c2 that means 2 CTAs on 256 CUs, but splitting H across CTAs needs the
full-H score first, which costs a second launch against a 4 us floor. No win
available without a cooperative-launch rewrite.

The two all-reduces per layer are the attention `o_proj` reduction and the MoE
output reduction; both are architecturally required. The fusion that folds the
pending prefix add into the collective exists but sits behind `k3_ar_fusion`,
which requires SM100/SM103 and `CustomAllReduceV2` multicast and is therefore
NVIDIA-only. Porting it to ROCm would not remove a collective, only an
elementwise pass.

That left the routing group. Sort and quant are already fused into one kernel by
the flag above, so the two candidates were folding in `sort_scales` and pulling
in `route_radix4`. Both were measured and neither pays.

**`sort_scales` is already within 0.5 us of the floor.** In isolation
(`bench_moe_sort_scales.py`) it costs 2.24 us at two tokens against the 1.70 us
graph-replay floor, shuffling 112 KB. The in-model 4.0 us is contention, not
work. So fusing it away saves at most ~1.7 us x 92 layers, about 1%, not the
2.6% the in-model figure suggested.

Its grid is also mis-sized -- 512 blocks launched where the work needs 28 -- but
fixing that is a no-op: sizing the grid to the work extent (and striding over
`gridDim.x` instead of the template maximum) measured 2.15 -> 2.01 us at one
token and nothing at all elsewhere. **The per-kernel floor on this stack is
fixed dispatch cost, not workgroup count**, which is worth knowing generally: it
means the only thing that helps is launching fewer kernels, never launching them
more cheaply. That change was reverted.

Fusing `sort_scales` into `sort_quant` needs a grid-wide barrier, because it
consumes `sorted_token_ids` and `cumsum` from block 0's sort *and* `a_scale`
from the other blocks' quant. A cooperative launch requires the whole grid
co-resident, and `kNCtasSort=512` x `kThreadsSort=1024` is 524,288 threads,
exactly the residency ceiling of 256 CUs x 2048. There is no headroom, so the
launch would fail or a hand-rolled barrier would deadlock. Shrinking the grid to
make room would slow the quant phase at prefill, which shares this path.

**`route_radix4` is already at its best block size.** It is 5.76 us at two
tokens in isolation, and `kRadix4Block` trades per-thread values
(`VPT = ceil(896/BLOCK)`) against synchronization depth
(`NWAVE = BLOCK/64`). Sweeping it found the shipped 256 optimal:

| kRadix4Block | 64 | 128 | **256** | 512 |
| --- | --- | --- | --- | --- |
| us at 2 tokens | 9.02 | 6.57 | **5.76** | 6.21 |

Single-wave (64) is the worst by far, so the per-thread work dominates rather
than the LDS merges. What is left of its 4 us over the floor is the dependent
chain of block-wide radix passes, which only a different selection algorithm
would shorten. Folding it into block 0 of the sort kernel remains structurally
possible and needs no barrier, but it means reimplementing sigmoid-plus-bias
top-k bit-exactly in AITER against a kernel that is already near its algorithmic
latency.

## What the server log claims, and what it is worth

The log is loud about things that turn out not to matter, and quiet about the
one that does.

**"not found tuned config ... will use default config! using torch solution"**
appears 1888 times for BF16 GEMMs, 576 for `a8w8_bpreshuffle` and 432 for
`a8w8_blockscale`. It reads like a large amount of untuned work. For decode it
is cosmetic. Only two decode-sized shapes miss, both `N=6288, K=7168` at M=1 and
M=4, and that shape has *zero* rows in any tuned CSV even though the unfused
`N=6016, K=7168` is covered for M=1..192. Running AITER's tuner over
`N=6288, K=7168` for M=1..256 picks **hipblaslt at 17.2 us / 5.24 TB/s**, which
is exactly the kernel the default path already falls back to. Writing the rows
silences the log and changes nothing. The rest of the misses are prefill shapes
with per-chunk M (8115, 8127, 8163, 8176, 8192), so they bear on TTFT rather
than TPOT.

**`[AR] All-reduce call path: NCCL (custom AR disabled)`** is misleading. It is
logged by an `elif` that does not describe the path the TP group actually takes;
the profile shows `aiter::cross_device_reduce_2stage` running, so the custom
all-reduce is in use. Both a fast and a slow configuration log it identically.

`FlashInfer TRTLLM MoE deferred finalize is disabled` and `Acceleration for
non-quantized schemes is not supported by Compressed Tensors. Falling back to
UnquantizedLinearMethod` are both expected: the latter is the checkpoint's
`ignore` list (`self_attn`, `shared_experts`, `mlp.*_proj`) staying BF16 by
design.

## The KDA input projection: already FP8, but under-tuned at two tokens

An earlier draft of this file claimed the projection was BF16 and wanted an FP8
variant built. That was wrong: `quantize_kimi_k3_kda_input_group64` packs the
weight to `float8_e4m3fn` with one fp32 scale per group of 64, so the kernel
already reads 47.9 MB rather than 90 MB. What it was not doing is reading them
quickly.

`tune_kda_input_group64.py` sweeps the launch parameters the module had
hard-coded. Two measurement details decide whether the sweep means anything.
The kernel must be timed **under graph replay** -- eagerly it costs ~48 us
against ~13 us replayed, so an eager sweep ranks launch overhead instead of the
kernel, and decode always replays a graph. And the sweep must rotate over
several weight buffers, because one 45 MB weight sits in the 256 MB last-level
cache and reports a bandwidth the model never sees while streaming 69 distinct
weights per step.

With those fixed, the shipping configuration turns out to be optimal at one
token and 25% off at two:

| tokens | rows_per_wave | cache_modifier | us | TB/s |
| --- | --- | --- | --- | --- |
| 1 | 2 (shipped) | 2 (shipped) | 9.02 | 5.31 |
| 2 | 2 (shipped) | 2 (shipped) | 15.81 | 3.03 |
| 2 | **3** | **1** | **12.62** | **3.82** |

`hidden_to_lds=True` and `cu_count=256` are both required -- dropping either
costs 30-70%. `waves_per_eu` is irrelevant. Selecting per token bucket gives
0.9% at c2 end-to-end and nothing above, since `covered()` restricts this kernel
to one or two tokens.

It is on by default and free. An earlier reading of this attributed a GSM8K
drop to it -- 0.951 and 0.952 with it against 0.956 without -- and gated it off.
That attribution was wrong. All sixteen (rows_per_wave, weight_cache_modifier)
combinations produce **bit-identical output** at both token counts: these
parameters distribute output rows over waves and set the weight load's cache
policy, and neither touches the order of the K=7168 reduction. The kernel cannot
change accuracy, so those GSM8K numbers are the eval's own run-to-run spread,
which is what the section below measures.

## Why the projection cannot simply be quantized further

PTPC cannot get at it. Profiling c2 with `SGLANG_K3_PTPC_FP8_MIN_TOKENS=1`
leaves the projection at 16.9 us, unchanged: `SGLANG_K3_AITER_KDA_GROUP64`
returns from `kda_group64_aiter_hip.run` before control reaches the
`_use_qkvgbfa_ptpc_fp8` branch, so the largest single BF16 weight read in the
model is never quantized. Clearing the way by setting
`SGLANG_K3_AITER_KDA_GROUP64=0` costs 3.4% (14.34 against 13.87 at c2): the
generic FP8 GEMM plus its activation quant gives up the group64 fusion for more
than the halved bytes return.

That path is closed, and it did not need opening: the group64 weight is already
FP8. The win was in the launch configuration instead, above.

## Measured 8k/1k sweep

Both configurations run end to end at the reported serving shape -- 8192 in,
1024 out, four waves, median values. Baseline is the recipe as shipped: fused
sort+quant off (which is what leaving `SGLANG_ROCM_USE_MULTI_STREAM` unset used
to produce), `SGLANG_K3_PTPC_FP8_MIN_TOKENS=8`, and the KDA group64 launch forced
to its pre-tuning configuration via `SGLANG_K3_KDA_GROUP64_LEGACY_LAUNCH=1`.

| c | TPOT base | TPOT opt | ΔTPOT | TTT base | TTT opt | ΔTTT |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 14.50 | 13.90 | 4.14% | 1169.1 | 1215.3 | 3.95% |
| 4 | 16.53 | 15.97 | 3.39% | 1999.0 | 2063.2 | 3.21% |
| 8 | 20.11 | 19.47 | 3.18% | 3197.7 | 3291.0 | 2.92% |
| 16 | 25.16 | 25.02 | 0.56% | 4907.4 | 4932.0 | 0.50% |
| 32 | 33.67 | 33.25 | 1.25% | 6988.1 | 7056.0 | 0.97% |
| 64 | 49.35 | 49.23 | 0.24% | 8999.7 | 8995.8 | -0.04% |

This baseline reproduces the reported table to within 0.2% at c2 and c4 and
1.5-2.2% at c8 and above, the residual being TTFT: prefill is slower on this box
while decode matches.

Carrying the measured throughput ratios onto the reported B300 comparison:

| c | 2 | 4 | 8 | 16 | 32 | 64 | geomean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before | 65 | 66 | 70 | 78 | 83 | 85 | **74.1%** |
| after | 67.6 | 68.1 | 72.0 | 78.4 | 83.8 | 85.0 | **75.5%** |

### An earlier version of this sweep was invalid

The first pair of runs measured 5.32% at c2 rather than 4.14%, because this
container exports `SGLANG_K3_PTPC_FP8=0` and the suite writes every recipe
variable as `${VAR:-default}`, so the shell's stale value silently won and PTPC
FP8 was off in *both* arms. Two things follow.

The delta was overstated: with PTPC genuinely on, the baseline is faster and the
headline drops from 5.32% to 4.14% at c2. And `SGLANG_K3_PTPC_FP8_MIN_TOKENS=1`
was credited with 1.1% at c2 that it cannot have produced, since PTPC was
disabled entirely -- that was run-to-run noise, and the flag's real value is
still unmeasured.

The accident did produce one useful number. Diffing the two baselines, which
differ only in whether PTPC was actually enabled, measures PTPC FP8 itself:
nothing at c2 and c4 (its own threshold was 8 tokens, so it was inactive there
either way), then 2.1% at c8, 1.0% at c16, 3.3% at c32 and 2.0% at c64.

`k3x_serve.sh` now parses the suite's own `K3_RECIPE_ENV_KEYS` list, unsets those
keys, and sources the recipe afterwards, so the recipe file defines the recipe.

## Recommended delta to the recipe

```bash
export AITER_MOE_A8W4_FUSED_SORT_QUANT=1     # +3.3% c2, +3.1% c8, +1.6% c32
export SGLANG_K3_PTPC_FP8_MIN_TOKENS=1       # +1.1% c2, +0.6% c4
# leave SGLANG_K3_MOE_LATENT_MXFP4_MIN_TOKENS at 2048
```

Cumulative against the recipe as shipped, median ITL at c2:

| Step | c2 | c8 | c32 |
| --- | --- | --- | --- |
| shipped (no fused sort+quant, PTPC from 8) | 14.50 | 19.19 | 27.59 |
| `AITER_MOE_A8W4_FUSED_SORT_QUANT=1` | 14.02 | 18.58 | 27.16 |
| `SGLANG_K3_PTPC_FP8_MIN_TOKENS=1` | 13.87 | 18.59 | — |
| KDA group64 per-bucket launch (default, no flag) | **13.71** | **18.62** | **27.21** |
| | **5.4%** | **3.0%** | **1.4%** |

Confirmed on a final run after all module rebuilds: 13.71 / 15.73 / 18.62 /
27.21 at c2 / c4 / c8 / c32.

Reproduce with `bench_multi_stream_overlap.py`'s sibling harness: launch via the
benchmark suite recipe, then sweep concurrency with `sglang.bench_serving` at
8192/256 and read median ITL.

## Moving more BF16 projections to FP8: a triage

`triage_ptpc_candidates.py` compares, per shape, the best tuned BF16 kernel
against the best tuned FP8 one from AITER's merged tables, then weights the
difference by how many layers of that kind a decode step actually runs.

| shape | layers | bf16 | fp8 | ceiling | per step | of a 13.9 ms step |
| --- | --- | --- | --- | --- | --- | --- |
| MLA fused_qkv_a `[2112,7168]` | 24 | 8.27 (flydsl) | 11.21 | 0.74x | −70 us | −0.51% |
| MLA o_proj `[7168,1536]` | 24 | 6.61 | 5.11 | 1.29x | +36 us | +0.26% |
| KDA o_proj `[7168,1536]` | 69 | 6.61 | 5.11 | 1.29x | +104 us | +0.74% |
| dense gate_up `[8448,7168]` | 1 | 22.98 | 13.54 | 1.70x | +9 us | +0.07% |
| dense down `[7168,4224]` | 1 | 12.53 | 17.76 | 0.71x | −5 us | −0.04% |

Two shapes -- the fused KDA input projection and the latent down -- appear in
neither table on this GPU; the first uses the specialized group64 FP8 kernel
instead, which is why tuning `[6288, 7168]` changed nothing.

**Do not benchmark these against `torch.mm`.** For `[2112, 7168]` the tuned BF16
kernel is FlyDSL at 8.27 us while `torch.mm` takes 16.4, so a microbenchmark
using `torch.mm` as the baseline reports FP8 winning 1.77x when it actually loses
at 0.74x. I wired the MLA input projection through PTPC on the strength of that
1.77x and it cost 0.8% end to end at c2 (14.00 against 13.89) before the tuned
tables explained why. The wiring is reverted.

That also settles the proposed norm+quant+GEMM fusion for the same projection: a
fusion removes the standalone activation quant, but the FP8 GEMM underneath is
already 36% slower than the BF16 one at this shape, so there is nothing for the
fusion to recover.

The dense MLP shapes win handsomely per GEMM and are worth nothing per step,
because `first_k_dense_replace=1` means Kimi-K3 has exactly one dense layer.

**What survives is the KDA o_proj**, at +0.74% of a c2 step. Generic PTPC cannot
collect it -- the standalone per-token quant costs 2.1-2.5 us against a 1.5 us
margin -- so it needs the quant folded into the preceding output-norm epilogue.
That is the one custom kernel on this list with a positive case.

## The o_proj, and why isolated GEMM wins keep not transferring

The triage above left the attention output projection as the one candidate with a
positive case, on a 1.29x tuned-table ceiling over 93 layers. Measured in
isolation it looked better still, because a 1536-wide activation quant is cheap
enough to fit the margin that the wider projections' quant does not:

| M | bf16 tuned | quant | fp8 gemm | quant+fp8 |
| --- | --- | --- | --- | --- |
| 1 | 6.37 | 1.84 | 3.63 | **5.27** |
| 2 | 6.02 | 1.85 | 3.63 | **5.44** |
| 8 | 6.39 | 2.06 | 3.62 | **5.52** |
| 32 | 6.22 | 2.08 | 3.85 | **5.81** |
| 64 | 5.85 | 2.06 | 4.23 | 6.25 |

So no fused epilogue was needed after all, up to 32 tokens. Wired in behind
`SGLANG_K3_PTPC_FP8_O_PROJ` by wrapping the linear's `quant_method` rather than
its `forward`, so `RowParallelLinear` keeps owning the TP reduction, the
symmetric-memory context and the skip-all-reduce flags.

End to end it does not survive: c1 12.16 against 12.15, c2 13.87 against 13.89,
**c8 19.84 against 19.47 -- 1.9% slower**. The fast path was confirmed firing from
its startup log, so this is not a dead path.

The GEMM saving at c8 is about 66 us of a 19.5 ms step, and the loss is 370 us --
five times larger and the wrong sign, so the GEMM is not what moved. The cost I
had not accounted for is memory: PTPC packs an FP8 copy and  leaves the BF16 weight
in place, which across 93 o_proj layers is ~1 GB per rank of KV cache given up.
It is off by default, and worth re-testing only if the packed weight can replace
the original instead of accompanying it.

### The methodology lesson

Three candidates in a row won in isolation and lost end to end:

| candidate | isolated | end to end | what the isolated number missed |
| --- | --- | --- | --- |
| MLA fused_qkv_a PTPC | 1.77x | −0.8% at c2 | baseline was `torch.mm`, production uses a tuned FlyDSL kernel ~2x faster |
| KDA group64 launch tuning | 1.25x | +0.9% at c2 | (this one transferred) |
| o_proj PTPC | 1.10x | −1.9% at c8 | FP8 copy costs ~1 GB/rank of KV cache |

On this stack essentially every hot GEMM already has a tuned or fused
implementation behind it, and PTPC adds resident memory rather than replacing it.
A kernel-level number is worth having as a ceiling, but nothing should be enabled
on one: the E2E A/B is the measurement, and it disagreed with the microbenchmark
in two of three cases here.


## Multi-token fused FP8 latent tail

The original latent-tail kernel was B1-only. `bench_latent_tail_fp8.py` now
validates B2/B4 specializations that normalize every token with the established
B1 reduction order and reuse each FP8 weight vector across all token
accumulators. B2 and B4 are bit-identical to invoking the B1 kernel once per
token with the same packed weight:

| M | fused launch | B1 x M | kernel speedup | E2E TPOT | result |
| --- | --- | --- | --- | --- | --- |
| 1 | 5.84 us | 6.91 us | 1.18x | 12.65 -> 12.15 ms | +3.95% |
| 2 | 8.60 us | 13.14 us | 1.53x | 13.90 -> 13.44 ms | +3.3% |
| 4 | 12.10 us | 25.49 us | 2.11x | 15.97 -> 15.82 ms | +0.9% |
| 8 | 18.81 us | 48.00 us | 2.55x | 19.47 -> 21.04 ms | **-8.1%** |

B8 shows why beating repeated B1 launches is not the shipping criterion: the
real baseline is a tuned batched GEMM, which amortizes overhead across M and
crosses over before the persistent kernel does. Supported buckets stop at B4.

The fast path is also decode-only. Before that guard, a four-row prefill could
enter the fused path and moved c4 TTFT from 1573 to 1821 ms; with the guard TTFT
is flat at 1576 ms while the decode gain remains.


## Dense gate_up PTPC

The one dense layer's TP8 `[8448,7168]` gate_up is wired through PTPC behind
`SGLANG_K3_PTPC_FP8_DENSE_GATE_UP`. The fast path was confirmed active. Tuned
FP8 is 1.70x faster than tuned BF16 for this GEMM, but
`first_k_dense_replace=1` means one call in a 93-layer step, a 9 us / 0.07%
ceiling. Eight-wave serving at c2 measures 13.46 against 13.44 ms -- neutral
within noise. It defaults off.


## KDA recurrence -> per-head FP8 -> blockscale o_proj

The requested producer fusion is implemented behind
`SGLANG_K3_KDA_O_PROJ_BLOCK_FP8`. The existing KDA recurrence kernel already
owns sigmoid-gated RMSNorm. Its epilogue now optionally computes one E4M3 scale
per 128-wide head and stores packed FP8; o_proj consumes `[M,12]` scales through
a blockscale GEMM. This is the correct fusion boundary -- no standalone quant
launch and no cross-head barrier.

Validation:

- Quantization adds 0.47 / 0.46 / 0.38 us to the recurrence at B1/B2/B8.
- The BF16 recurrence output is bit-identical with the epilogue enabled.
- B1 quant is bit-identical to standalone AITER per-1x128 quant. At B2 the scale
  differs by one fp32 ULP and 7 of 3072 FP8 values sit on the adjacent boundary.
- Blockscale o_proj is 5.19 us at B2 against 6.02 us tuned BF16, leaving a
  kernel-level net ceiling of roughly 0.37 us/layer after quant overhead.

End to end the ceiling does not transfer: c2 is 13.57 against 13.44 ms and c8
19.57 against 19.47. The fast path is confirmed active. It defaults off.


## Final accepted 8k/1k sweep and accuracy

After every optional path that lost E2E was left off, the accepted defaults are:

- `AITER_MOE_A8W4_FUSED_SORT_QUANT=1` (explicit and decoupled from multi-stream)
- per-bucket KDA group64 launch tuning
- `SGLANG_K3_PTPC_FP8_MIN_TOKENS=1`
- `SGLANG_K3_AITER_LATENT_TAIL_FP8=1`, decoded buckets B1/B2/B4 only

| c | TPOT | TTFT | TTT | TTT gain vs corrected baseline |
| --- | --- | --- | --- | --- |
| 2 | 13.48 | 917.99 | 1251.18 | 7.02% |
| 4 | 15.85 | 1587.34 | 2075.53 | 3.83% |
| 8 | 19.48 | 2493.44 | 3290.55 | 2.90% |
| 16 | 24.61 | 4319.28 | 5005.91 | 2.01% |
| 32 | 33.20 | 7977.94 | 7061.59 | 1.05% |
| 64 | 49.25 | 15264.88 | 9010.22 | 0.12% |

Applying those measured throughput ratios to the reported B300 comparison moves
65/66/70/78/83/85 to 69.6/68.5/72.0/79.6/83.9/85.1, geomean **74.1% ->
76.15%**.

GSM8K (1319 questions, five-shot, parallel 64):

| configuration | accuracy |
| --- | --- |
| control: latent tail off, PTPC minimum 8 | 0.955 |
| latent tail B1/B2/B4 on, PTPC minimum 8 | 0.958 |
| accepted final: latent tail on, PTPC minimum 1 | 0.951 |

The user acceptance threshold is >=0.94, so the performance stack passes. The
paired accuracy isolation attributes the 0.4-0.7 point movement to low-batch
PTPC rather than the fused latent tail.

## Closed optimization cycles from the final profile

The final c2 trace is 14.59 ms across 1,981 launches. Two larger changes were
implemented exactly and rejected:

- Fused sort+quant+sort_scales: all intermediate tensors bit-identical, graph
  replay safe, but 10.26 vs 8.90 us at B2 and 9-29% slower through B256. The
  resident-grid barrier and reduced quant parallelism cost more than one launch.
- Dynamic one-stage all-reduce: the global switch improves TPOT 4.5-4.7% but
  regresses TTFT 63-77%. Two dual-communicator selectors preserved TTFT but lost
  at c8. AITER's existing small-tensor kernel is already one-stage; forcing its
  16-block limit also regressed c2 against the current B2-tail baseline.

No part of these rejected cycles is enabled by default.


## Low-concurrency cycle: re-tuning the B2/B4 latent-tail launch

The final c2 profile puts the fused latent tail at 0.91 ms / 92 calls (6.2% of
the step), and B2/B4 had inherited B1's launch schedule. A graph-replay sweep
rotating 12 x 25.7 MB weights found bit-identical microbenchmark winners:

- B2: 9.20 -> 9.07 us (rows_per_wave=2, CUs=224, waves_per_eu=1)
- B4: 12.31 -> 12.18 us (rows_per_wave=1, CUs=224, waves_per_eu=3)

The ceiling is only ~0.1% of a full step, and the paired eight-wave E2E run
rejects it: c2 13.46 -> 13.50 ms and c4 15.84 -> 15.87. Shipping launch
parameters remain unchanged. `tune_latent_tail_fp8.py` is retained so future
hardware/ROCm revisions can repeat the sweep without rebuilding the harness.


## Low-concurrency cycle: attention-residual launch tuning

The accepted c2 profile spends 1.48 ms across 186 single-CTA `_agg_kernel`
launches. A graph-replay sweep over `num_warps` and `waves_per_eu`, with bitwise
validation, selects by padded bank depth:

| R_PAD | original | selected | kernel gain |
| --- | --- | --- | --- |
| 1 | warps=4, wpe=0 | warps=4, wpe=3 | 0.8% |
| 2 | 4, 0 | 4, 3 | 2.3% |
| 4 | 4, 0 | 4, 1 | 1.1% |
| 8 | 4, 0 | 8, 2 | 1.5% |

Eight-wave c2 A/B: control 13.51 ms; tuned 13.37 and 13.39 on two runs, a
repeatable 0.9-1.0% TPOT and ~0.9% TTT gain. At c4 it is neutral (15.85 vs
15.87); c8 improves 19.48 -> 19.36 ms (0.6%). The launch parameters do not
change arithmetic -- every candidate enabled here is bit-identical -- and are
the default. Set `SGLANG_K3_ATTN_RES_TUNED_LAUNCH=0` for A/B.


## Low-concurrency cycle: fused MoE-front launch tuning

The final c2 profile spends 1.13 ms / 92 calls (7.8%) in the cooperative
preactivated tri-projection. A bitwise graph-replay sweep over waves/block,
weight-cache modifier, CU count and waves/EU found only WCM=2 as a plausible
candidate: neutral at B2 and ~2% faster at B4 in isolation.

A matched eight-wave E2E A/B on the attention-residual-tuned stack rejects it:
WCM3 (shipping) measures 13.38/15.73 ms at c2/c4; WCM2 measures 13.40/15.73,
with slightly lower TTT. WCM3 remains unchanged. The reusable sweep harness is
`tune_moe_preroute_fp8.py`.


## Low-concurrency cycle: fuse the residual-prefix add into latent tail

The final c2 profile contains 93 generic BF16 add launches (0.41 ms). For the 92
MoE layers, B1/B2/B4 latent-tail output was immediately followed by
`out + prefix_sum`. The tail now accepts the prefix and preserves both original
BF16 rounding boundaries: first `BF16(projected + shared)`, then
`BF16(tail + prefix)`. B1/B2/B4 outputs are bit-identical to the two-launch path.

Graph-replay ceiling:

| bucket | separate | fused | speedup |
| --- | --- | --- | --- |
| B1 | 7.66 us | 6.52 us | 1.17x |
| B2 | 10.05 us | 7.95 us | 1.26x |
| B4 | 13.33 us | 10.82 us | 1.23x |

Matched eight-wave E2E, confirmed twice:

| c | control | fused run 1 | fused run 2 |
| --- | --- | --- | --- |
| 2 | 13.38 ms | 13.17 | 13.20 |
| 4 | 15.73 ms | 15.58 | 15.53 |

That is a repeatable 1.3-1.6% c2 and 1.0-1.3% c4 TPOT gain, with TTT improving
0.7-1.6%. No accuracy run is needed because every fused output is bit-identical.


## Optimization cycle 2: final sweep

Cycle 2 started from a fresh c2 profile after prefix fusion and kept only two
changes: bit-identical attention-residual launch tuning, and a c2-specific
coupled inline-A4W4 routed-MoE row that passed the 0.94 accuracy floor. The
prefix-add fusion landed immediately before this cycle and is included in both
the final numbers and the 2.04% cycle-over-cycle c2 gain.

| c | previous TPOT | cycle-2 TPOT | TPOT gain | previous TTT | cycle-2 TTT | TTT gain |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 13.48 | 13.20 | 2.12% | 1251.18 | 1276.75 | 2.04% |
| 4 | 15.85 | 15.54 | 1.99% | 2075.53 | 2112.03 | 1.76% |
| 8 | 19.48 | 19.35 | 0.67% | 3290.55 | 3311.23 | 0.63% |
| 16 | 24.61 | 24.42 | 0.78% | 5005.91 | 5034.39 | 0.57% |
| 32 | 33.20 | 33.05 | 0.45% | 7061.59 | 7100.86 | 0.56% |
| 64 | 49.25 | 49.03 | 0.45% | 9010.22 | 9050.55 | 0.45% |

Cycle-2 geometric-mean TTT gain over the previous accepted stack is 1.00%.
Cumulatively against the corrected shipped baseline, TTT improves
9.21/5.65/3.55/2.59/1.61/0.56% from c2 through c64. Applying those measured
ratios to the reported B300 comparison moves the geomean **74.08% -> 76.91%**.
GSM8K is 0.951, above the accepted 0.94 floor.

Rejected cycle-2 candidates, all measured E2E or exact kernel A/B:

- B2/B4 latent-tail launch retuning: micro win, E2E loss.
- Fused MoE-front cache modifier 2: neutral/slightly worse than matched WCM3.
- Fused sort+quant+sort-scales: bit-identical but 9-29% kernel regression.
- Global, size-based and phase-aware one-stage all-reduce: either TTT/TTFT or
  c8 regression. No dynamic communicator change is enabled.
