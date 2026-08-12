# Kimi-K3 on sglang/ROCm — porting ATOM's optimizations, measured one at a time

Target tree: `/sgl-workspace/sglang`, branch `kimi-k3-optimizations`.
Hardware: 8× MI355X (gfx950), ROCm, TP 8.
Workload: ISL 8192, OSL 1024, concurrency 2 / 4 / 8 / 16 / 32.

---

## 0. Status at a glance

| | output tok/s vs baseline | ships |
|---|---|---|
| **sv3** — fused KDA in-projection + fused gated-RMSNorm→FP8 KDA `o_proj` | **+1.4 … +2.7%** (all five concurrencies) | **on** |
| **sv10** — sv3 + `--dcp-size 8` | **+3.4 … +6.6%** on top of sv3, TTFT −20 … −55%, 8× KV capacity per GPU, gsm8k 0.9521 | on for this workload (§13.5) |
| sv4/sv5 — dual-stream MoE overlap (ATOM's headline change) | −2.0 … −6.8% | **off** |
| sv9 — the three aiter knobs (flydsl sort, quick-reduce quant, mxfp4 intermediate) | −1.6 … +0.3% | off |

Sections: §1–§9 are the original measurement campaign. §10 is the consolidated
version ladder, §11 compares against ATOM PR #1752, §12 lists what of ATOM's
work is still missing here, §13 is DCP (three ROCm bugs found and fixed).

---

## 1. Benchmark methodology

### 1.1 Harness

A real server driven by `sglang.bench_serving`, not an in-process latency
harness:

```
python -m sglang.launch_server \
  --model-path /models/Kimi-K3 --trust-remote-code --tp-size 8 \
  --attention-backend aiter --dtype bfloat16 \
  --context-length 10240 \
  --mem-fraction-static 0.93 --mamba-full-memory-ratio 0.3 \
  --max-total-tokens 311296 \
  --disable-radix-cache --chunked-prefill-size 8192 \
  --max-running-requests 32 --cuda-graph-max-bs 64

python -m sglang.bench_serving --backend sglang \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --max-concurrency $C --num-prompts $((C*3)) --warmup-requests $C

env: SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1
     AITER_SITUV2_A8W4=1 AITER_LOG_LEVEL=WARNING
```

Files: `/tmp/k3bench/serve_bench.sh <tag>` (one version, all five concurrency
points), `svall.sh` / `svrest2.sh` / `sv9.sh` (drivers),
`sv_summarize.py` (Δ tables), `acc.py` (gsm8k).

One server serves all five concurrency points — loading K3 costs ~6 min and
the points are independent, so paying it once per version rather than once per
point is most of the wall clock. `--disable-radix-cache` so a later
concurrency point cannot reuse an earlier one's prefix and post a fake TTFT.
`--random-range-ratio 1.0` so every request is exactly 8192 in / 1024 out.

Every version differs from the one before it by exactly one flag, and the Δ%
columns are always against the immediately preceding version.

### 1.2 Why a server and not `sglang.benchmark.one_batch`

`one_batch` prefills the whole batch in a single forward. aiter's
`tuned_fmoe.csv` stops at `token=16384` for K3's MoE shape (`model_dim=3584`,
`inter_dim=384`, `expert=896`, `topk=16`); above it `AITER_FLYDSL_FORCE=1`
takes a heuristic fallback that picks a kernel whose *name* says
`t64x128x256` while the shape forces tile_k/tile_n down to 128, and the run
dies with `Memory access fault by GPU node-N`. (Reproduced on the pristine
tree — pre-existing, not caused by anything here.) Concurrency 32 at ISL 8192
is 262144 tokens in one shot, sixteen times over that ceiling.

The server chunks prefill at `--chunked-prefill-size 8192`, so every forward
stays inside the tuned table.

Two other pre-existing harness limits, for the record: `--dcp-size 8` cannot
be used with `one_batch` (the gluon DCP MLA decode reshapes `q` with the
TP-sharded head count at `aiter_backend.py:1076` and dies during graph
capture), so DCP is off throughout; and `one_batch` writes its jsonl only
after the whole sweep, so a late crash discards everything earlier —
`PYTHONUNBUFFERED=1` plus log parsing works around it.

### 1.3 A sizing bug that made concurrency 32 meaningless

The first baseline reported concurrency 32 at **327 tok/s with a 62 s median
TTFT** — the same throughput and the same 26 ms ITL as concurrency 16. The
server was admitting ~16 requests and queueing the other 16, so the "32" point
was measuring the 16 point plus a queue.

`--mem-fraction-static 0.85` (the value the K3 accuracy test uses) left a KV
pool of 153620 tokens = 16.7 requests of 8192+1024. The KV budget is
`(fraction × total) − weights − slack`, and `slack` is itself
`(1 − fraction) × pre_model_load_memory` (`kv_cache_configurator.py:1660`), so
the fraction is paid twice: on a 288 GB MI355X the last 0.08 is worth ~46 GB
of budget. Neither `--mamba-full-memory-ratio` (the mamba pool was already
capped at 32 slots by `--max-running-requests`, 1.73 GB) nor
`--max-total-tokens` alone fixed it — the latter is clamped to the profiled
value:

```
max_total_tokens=311296 is larger than the profiled value 153620.
Use the profiled value instead.
```

At `--mem-fraction-static 0.93` the profiler offers enough, and
`--max-total-tokens 311296` then pins the pool to exactly 32 requests' worth,
so every version measures against identical KV capacity rather than against
whatever that version's weights happened to leave over.

| baseline, concurrency 32 | out tok/s | median ITL | median TTFT |
|---|---|---|---|
| capacity-limited (0.85) | 327.1 | 26.04 ms | 61.7 s |
| genuinely 32-way (0.93) | 378.6 | 32.43 ms | 25.6 s |

`serve_bench.sh` now asserts `max_total_num_tokens ≥ 32 × (8192+1024)` at
startup and aborts rather than publishing a queued regime as a concurrency
number.

### 1.4 Noise floor

Concurrency 2 / 4 / 8 / 16 reproduced to within **0.1%** across two separate
server launches with *different memory configurations* (0.85 and 0.93 — the
two runs differ only in KV pool size, which does not bind below concurrency
32). Anything above ~0.3% on this harness is real.

---

## 2. Versions

| tag | change (cumulative) |
|-----|---------------------|
| `sv1_baseline` | every K3 optimization off |
| `sv2_inproj` | + fused KDA in-projection GEMM (`SGLANG_ROCM_K3_FUSE_KDA_INPROJ=1`) |
| `sv3_fp8oproj` | + fused gated-RMSNorm → per-token FP8 KDA `o_proj` (`SGLANG_ROCM_K3_KDA_O_PROJ_FP8=1`) |
| `sv4a_ms_decode` | + dual-stream MoE, overlap window 1…1024 tokens (decode only) |
| `sv4b_ms_all` | + dual-stream MoE, overlap window 1…16384 tokens (decode + prefill chunks) |
| `sv5_ms_split` | sv3 + dual-stream MoE (window 1…1024) with the shared `gate_up` un-fused from the merged front, so the side stream runs the whole shared MLP — ATOM's shape (`SGLANG_ROCM_K3_MULTI_STREAM_SPLIT_FRONT=1`) |
| `sv9a_flydsl_sort` | sv3 + `AITER_USE_FLYDSL_MOE_SORTING=1` |
| `sv9b_qr_fp` | sv3 + `ROCM_QUICK_REDUCE_QUANTIZATION=FP` |
| `sv9c_qr_int8` | sv3 + `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` |
| `sv9d_mxfp4_inter` | sv3 + `AITER_MXFP4_INTERMEDIATE=1` |
| `sv10_dcp8` | sv3 + `--dcp-size 8` (decode context parallelism), after the three DCP fixes in §13. Quote the `_r3` sweep — r1/r2 predate the §13.2 address fix |

Shipped defaults are decided in §6 from these numbers.

---

## 3. Code changes

Committed on `kimi-k3-optimizations`, one commit per item below (`git log
--oneline main..kimi-k3-optimizations`). A flat patch of the same change set
(1481 lines, 12 files) is mirrored at `/tmp/k3bench/k3_opts.patch`.

| file | Δ | what |
|---|---|---|
| `python/sglang/srt/models/kimi_k3.py` | +265 | in-proj fusion, FP8 `o_proj`, dual-stream MoE |
| `python/sglang/srt/environ.py` | +41 | 7 new env knobs, plus the measured rationale on the existing ones |
| `python/sglang/kernels/ops/kimi_k3/rmsnorm_gated_quant.py` | new | Triton fused gated-RMSNorm + per-token FP8 quant |
| `python/sglang/kernels/ops/kimi_k3/__init__.py` | +13 | lazy export |
| `test/registered/amd/test_kimi_k3_rmsnorm_gated_fp8.py` | new | 3 tests + 5 subtests |
| `test/registered/amd/test_kimi_k3_kda_inproj_fusion.py` | new | fusion equivalence |
| `test/registered/amd/test_kimi_k3_kda_fused_decode_aiter.py` | new | aiter KDA decode backend |
| `.../attention_forward_methods/forward_mla_rocm.py` | +14 | route the DCP decode/target-verify phase to `attn_mqa_for_dcp_decode` on the aiter path (§13.1) |
| `.../attention_forward_methods/forward_mla.py` | +13 | the same gap on the CUDA path (flashmla), fixed by sweep |
| `python/sglang/srt/mem_cache/memory_pool.py` | +12/−6 | capture-safe DCP KV write: redirect non-owned rows to dummy slot 0 instead of a syncing boolean compaction (§13.2) |
| `test/registered/amd/test_kimi_k3_front_split.py` | new | ATOM-shape front split (§5.6) |

### 3.1 Fused KDA in-projection (`SGLANG_ROCM_K3_FUSE_KDA_INPROJ`)

ATOM issues the KDA input projection as one GEMM. sglang issued it as two:
a `[q|k|v|g]` merged column-parallel linear and a separate `[f_a|b]` tail.
`_merge_bfa_weights()` concatenates the tail's rows onto the first weight at
load time into `_qkvgbfa_w` (6288 rows at TP 8), records `_qkvgbfa_sizes`, and
`forward_qkvbfg_fused` takes the single-GEMM path when

```python
0 < hidden_states.shape[0] <= self._qkvgbfa_bs_limit
```

`_qkvgbfa_bs_limit` is `SGLANG_ROCM_K3_FUSE_KDA_INPROJ_MAX_TOKENS` (default
256). Above that the two GEMMs are already large enough to saturate, and the
extra concat/split is pure overhead — the fusion pays off precisely because at
small token counts both GEMMs are launch- and weight-read-bound, so merging
them halves the fixed cost and reads no extra weights.

This never fires on an 8192-token prefill chunk. It is a decode optimization
and the results say so.

Compatible with `SGLANG_K3_KDA_FUSED_BACKEND=aiter`: the fusion is upstream of
the KDA core, it only changes *how* `q/k/v/g/f_a/b` are produced, and it hands
the backend the same tensors either way.

### 3.2 Fused gated-RMSNorm → per-token FP8 `o_proj` (`SGLANG_ROCM_K3_KDA_O_PROJ_FP8`)

This is the item the earlier gap analysis had written off as
*"inapplicable — exists only because ATOM's recipe runs ptpc_fp8 on attention
linears; sglang keeps K3 attention BF16."* Implemented anyway, and it is the
single biggest win in the series.

Three pieces:

1. **`rmsnorm_gated_fp8_per_token(x, weight, gate, eps, quant_dtype)`** — a
   Triton kernel (ported from ATOM `atom/model_ops/kimi_k3/activations.py`)
   that in one pass computes the gated RMSNorm of the KDA core output and
   emits `(fp8 tensor, per-token fp32 scale)`. Previously this was
   norm → gate → separate quant, i.e. three round trips of the
   `[tokens, 6144]` activation through HBM; now it is one.

2. **`_quantize_o_proj_fp8()`** — one-time, at first forward: per-output-channel
   quantization of the `o_proj` weight to FP8, then
   `shuffle_weight(qw, (16,16))` for `aiter.gemm_a8w8_bpreshuffle`. It
   *declines* (stays BF16) when `self.all_reduce_fusion or
   envs.SGLANG_K3_GEMM_AR.get()`, because those paths consume the BF16 weight
   directly.

3. **`_K3PtpcFp8LinearMethod`** — the ptpc-FP8 linear method, with a no-op
   `process_weights_after_loading` hook. That hook is not decoration:
   `DefaultModelLoader` sweeps every module's `quant_method` after
   `load_weights` returns, and without it the sweep raises on this method.

Numerics: per-token activation scales and per-channel weight scales, which is
the same recipe ATOM validated for K3. gsm8k check in §7.

### 3.3 Dual-stream MoE overlap (`SGLANG_ROCM_USE_MULTI_STREAM`)

`_forward_fused` forks the shared-expert `act_fn` + `down_proj` GEMM
(`_forward_shared`) onto `self.alt_stream` while the routed experts
(`_forward_routed`) run on the main stream, then joins.

Two new knobs bound where it applies —
`SGLANG_ROCM_K3_MULTI_STREAM_MIN_TOKENS` (64) and `..._MAX_TOKENS` (1024) —
materialized at construction into `self._rocm_overlap_tokens`, a set, so the
per-forward test is `num_tokens in self._rocm_overlap_tokens` rather than two
comparisons inside the traced region.

`_ms_events()` returns a long-lived `(fork_event, join_event)` pair, gated on
`SGLANG_ROCM_K3_MULTI_STREAM_REUSE_EVENTS`, so the fork/join uses
`record`/`wait_event` on reused events instead of `Stream.wait_stream`, which
allocates and destroys a `torch.cuda.Event` on every call. That hypothesis was
measured and **refuted** (§5.2) — the edit is behaviour-preserving and
performance-neutral, and is kept only because it is strictly no worse.

A third knob, `SGLANG_ROCM_K3_MULTI_STREAM_SPLIT_FRONT` (on), changes *what* is
overlapped. By default `_forward_fused` reads `hidden_states` once through the
merged `[H, gate_up + E + latent]` front weight, which leaves only
`act_fn + down_proj` for the side stream. Inside the overlap window the shared
`gate_up` is pulled back out — the main stream takes a narrowed
`[H, E + latent]` front and the side stream computes `gate_up` itself from
`hidden_states`, so it carries the whole shared MLP, which is ATOM's shape. The
fork moves *above* the front GEMM so the narrowed front overlaps the side
`gate_up` too. Both weights are dim-0 views of the buffer
`_merge_weights_as_views` already built (`self._front_w_nogu` is
`self._front_w[gate_up_rows:]`; the shared module's own `.weight` is the
leading block), so this costs no memory and no copies. Outside the window the
merged front is kept. Measured in §5.5: it helps, and not nearly enough.
Pinned by `test/registered/amd/test_kimi_k3_front_split.py`.

---

## 4. Results — items 1, 2, 3

### 4.1 Baseline (`sv1_baseline`)

| conc | out tok/s | total tok/s | median ITL (ms) | median TTFT (ms) | median E2E (ms) | duration (s) |
|---|---|---|---|---|---|---|
| 2 | 91.53 | 823.78 | 18.71 | 2927 | 22108 | 67.12 |
| 4 | 158.97 | 1430.74 | 19.44 | 5097 | 25756 | 77.30 |
| 8 | 234.79 | 2113.15 | 22.59 | 8073 | 34884 | 104.67 |
| 16 | 326.56 | 2939.07 | 26.07 | 13918 | 50112 | 150.51 |
| 32 | 378.61 | 3407.50 | 32.43 | 25610 | 78585 | 259.64 |

### 4.2 Output token throughput (tok/s), Δ% vs previous version

| conc | sv1 | sv2 | Δ | sv3 | Δ | **sv1→sv3** |
|---|---|---|---|---|---|---|
| 2 | 91.53 | 92.49 | +1.0% | 94.04 | +1.7% | **+2.7%** |
| 4 | 158.97 | 159.39 | +0.3% | 161.47 | +1.3% | **+1.6%** |
| 8 | 234.79 | 235.12 | +0.1% | 238.03 | +1.2% | **+1.4%** |
| 16 | 326.56 | 329.56 | +0.9% | 332.18 | +0.8% | **+1.7%** |
| 32 | 378.61 | 382.87 | +1.1% | 385.52 | +0.7% | **+1.8%** |

### 4.3 Median ITL (ms), lower is better

| conc | sv1 | sv2 | Δ | sv3 | Δ | **sv1→sv3** |
|---|---|---|---|---|---|---|
| 2 | 18.71 | 18.76 | +0.2% | 18.41 | −1.8% | **−1.6%** |
| 4 | 19.44 | 19.37 | −0.4% | 19.07 | −1.6% | **−1.9%** |
| 8 | 22.59 | 22.55 | −0.2% | 22.16 | −1.7% | **−1.9%** |
| 16 | 26.07 | 25.65 | −1.6% | 25.26 | −1.5% | **−3.1%** |
| 32 | 32.43 | 31.55 | −2.7% | 31.10 | −1.4% | **−4.1%** |

### 4.4 Median TTFT (ms), lower is better

| conc | sv1 | sv2 | Δ | sv3 | Δ |
|---|---|---|---|---|---|
| 2 | 2927 | 2926 | −0.1% | 2918 | −0.3% |
| 4 | 5097 | 5101 | +0.1% | 5080 | −0.4% |
| 8 | 8073 | 8064 | −0.1% | 8041 | −0.3% |
| 16 | 13918 | 13928 | +0.1% | 13884 | −0.3% |
| 32 | 25610 | 25570 | −0.2% | 25467 | −0.4% |

### 4.5 Median E2E latency (ms) and total throughput (tok/s)

| conc | E2E sv1 | E2E sv3 | Δ | total sv1 | total sv3 | Δ |
|---|---|---|---|---|---|---|
| 2 | 22108 | 21771 | −1.5% | 823.78 | 846.39 | +2.7% |
| 4 | 25756 | 25359 | −1.5% | 1430.74 | 1453.20 | +1.6% |
| 8 | 34884 | 34419 | −1.3% | 2113.15 | 2142.23 | +1.4% |
| 16 | 50112 | 49270 | −1.7% | 2939.07 | 2989.66 | +1.7% |
| 32 | 78585 | 76971 | −2.1% | 3407.50 | 3469.71 | +1.8% |

### 4.6 Reading

Both optimizations are decode-side, and TTFT confirms it: it does not move
outside the noise floor at any concurrency, while ITL drops at every one. The
in-projection fusion caps out at 256 tokens per forward so it never touches an
8192-token prefill chunk; the FP8 `o_proj` runs on every KDA layer but
`o_proj` is a small share of a prefill chunk's arithmetic and its saving is
swamped there.

The in-projection fusion's benefit *grows* with concurrency (+0.2% ITL at 2,
−2.7% at 32) — expected: at concurrency 2 a decode forward carries 2 tokens
and the merged GEMM is still entirely launch-bound, so merging two launches
into one saves one launch out of a very deep chain; at 32 the merged GEMM
starts doing real work and the single larger GEMM is meaningfully more
efficient than two thin ones.

The FP8 `o_proj` is nearly flat at −1.4% to −1.8% ITL across the whole range,
which is what a bandwidth saving on a fixed-size per-layer tensor should look
like.

Because E2E is dominated by 1024 decode steps and TTFT by one 8192-token
prefill, the E2E improvement (−1.3% to −2.1%) tracks ITL rather than
throughput.

---

## 5. Item 4 — dual-stream MoE overlap

### 5.1 The stock window never fires on this workload

`SGLANG_ROCM_K3_MULTI_STREAM_MIN_TOKENS=64`, `..._MAX_TOKENS=1024`. At
ISL 8192 / concurrency 2…32 a decode forward carries **2–32 tokens** (below
MIN) and a prefill chunk carries **8192** (above MAX). The overlap as shipped
is dead code here, which is why simply flipping `SGLANG_ROCM_USE_MULTI_STREAM=1`
would have measured nothing. Two retuned windows were measured instead:

- `sv4a_ms_decode` — window 1…1024: decode overlaps, prefill does not.
- `sv4b_ms_all` — window 1…16384: decode *and* prefill chunks overlap.

### 5.2 Results — a clear regression, and opening the window wider changes nothing

Output token throughput (tok/s):

| conc | sv3 (off) | sv4a (decode) | Δ | sv4b (decode+prefill) | Δ vs sv4a |
|---|---|---|---|---|---|
| 2 | 94.04 | 86.98 | **−7.5%** | 86.81 | −0.2% |
| 4 | 161.47 | 150.83 | **−6.6%** | 150.57 | −0.2% |
| 8 | 238.03 | 227.08 | **−4.6%** | 227.21 | +0.1% |
| 16 | 332.18 | 323.53 | **−2.6%** | 323.39 | −0.0% |
| 32 | 385.52 | 377.77 | **−2.0%** | 377.72 | −0.0% |

Median ITL (ms):

| conc | sv3 | sv4a | Δ | sv4b | Δ vs sv4a | sv4a − sv3 |
|---|---|---|---|---|---|---|
| 2 | 18.41 | 20.12 | +9.3% | 20.20 | +0.4% | **+1.71 ms** |
| 4 | 19.07 | 20.81 | +9.1% | 20.86 | +0.2% | **+1.74 ms** |
| 8 | 22.16 | 23.79 | +7.3% | 23.78 | −0.0% | **+1.63 ms** |
| 16 | 25.26 | 26.62 | +5.4% | 26.65 | +0.1% | **+1.36 ms** |
| 32 | 31.10 | 32.14 | +3.3% | 32.11 | −0.1% | **+1.04 ms** |

Median TTFT (ms) — flat throughout: sv3 → sv4a is +0.3 / +0.2 / −0.0 / +0.1 /
+0.5 %, sv4b → +0.1 / +0.1 / −0.0 / +0.0 / +0.1 %.

Two things to read off this:

1. **The cost is a fixed per-decode-step charge, not a percentage.** The last
   column is 1.0–1.7 ms regardless of concurrency, and the *percentage* shrinks
   only because the step itself gets longer. Divided over K3's 92 MoE layers
   (`num_hidden_layers=93`, `first_k_dense_replace=1`, `moe_layer_freq=1`) that
   is **~11–19 µs per fork/join pair**. Note this exceeds the 8.1 µs a bare
   fork/join costs in an empty graph (§5.3) — the excess is the per-node tax
   §5.4 isolates, and the fact that it *is* an excess is the first hint that
   the barrier alone cannot be the explanation.

2. **Overlapping the prefill chunks is worth exactly zero — both ways.** sv4b
   opens the window from 1024 to 16384 so every 8192-token prefill chunk forks
   too, and *every* metric matches sv4a to within ±0.2%. At 8192 tokens both
   branches already saturate the GPU, so there is no idle capacity to overlap
   into; and the same ~15 µs barrier is now amortized over a much longer layer,
   so the cost disappears too. Gain zero, cost zero, net zero.

### 5.3 Why it does not pay — measured directly, off-model

Three off-model probes were run against this. **The first two supported a
mechanism that the third refuted**; the sequence is kept here because the
refutation is the actual finding.

**`/tmp/k3bench/fork_cost.py`** — 92-layer graph, tiny payload, with and
without a fork:

```
device: AMD Instinct MI355X
92 layers, replay time per graph:
  no fork                 0.437 ms
  fork, wait_stream       1.182 ms   (+0.745 ms = 8.1 us/fork)
  fork, reused events     1.174 ms   (+0.737 ms = 8.0 us/fork)

shared-expert down GEMM [32, 768] x [768, 7168] bf16: 12.8 us (11.0 MB)
```

A fork/join pair inside a hipGraph costs **8.1 µs** on MI355X — and reusing
events buys 0.1 µs, independently reproducing the v7 refutation on a harness
with no model in it.

**`/tmp/k3bench/overlap_gain.py`** then swept how busy the main stream is,
modelling the routed experts as one large `torch.mm`:

```
92 layers, 32 tokens, all times per replay
  main K  main only  serialized    forked      gain   verdict
    1024     0.623m      1.162m    1.403m    -2.6us      loss
    4096     1.593m      2.244m    2.383m    -1.5us      loss
    8192     1.534m      2.164m    2.413m    -2.7us      loss
   16384     2.394m      3.028m    3.232m    -2.2us      loss
   32768     4.359m      5.108m    5.138m    -0.3us      loss
```

Negative at every point, which looked like a complete explanation: the barrier
costs 8.1 µs and the overlap returns only 5.4–7.8 µs, so the payload is simply
too small. **That explanation was wrong**, and §5.4 and §5.5 are how it fell
apart.

### 5.4 The payload is not the problem

If the payload were the binding constraint, enlarging it would fix things. It
can be enlarged — see §5.5 — and doing so recovered almost nothing on the real
model. Two further probes show why.

**`/tmp/k3bench/overlap_intensity.py`** — is the main stream out of HBM
headroom at decode, so that a second stream has no bandwidth to win? Same side
payload, main payloads of opposite arithmetic intensity:

```
      kind      M       K  flop:byte  main only  main GB/s  %peak   serial   forked      gain
   compute   1024    4096      682.7     3.636m       1273    16%   6.039m   4.972m   +11.6us
 bandwidth     32   16384       31.8     4.307m       5767    72%   7.330m   6.497m    +9.1us
 bandwidth     32   32768       31.8     8.674m       5722    72%  11.777m  10.182m   +17.3us
 bandwidth     32   65536       31.9    16.329m       6076    76%  19.444m  17.768m   +18.2us
```

Hypothesis refuted. The gain is **strongly positive** (+9 to +18 µs/layer) even
at 76% of peak HBM. Forking is not bandwidth-limited here, and — awkwardly for
§5.3 — it returns *far more* than the 8.1 µs barrier in every one of these
configurations. So the probes now disagree with the model, and the difference
must be something neither probe was modelling.

**`/tmp/k3bench/overlap_kernelcount.py`** finds it. Total main-stream FLOPs and
bytes are held exactly fixed; the only thing that changes down the table is how
many kernels that work is split into:

```
 kernels   K each  main only    serial    forked      gain  verdict
       1    32768     4.340m    7.316m    6.397m   +10.0us      WIN
       2    16384     4.888m    8.094m    6.592m   +16.3us      WIN
       4     8192     6.330m    9.721m    8.126m   +17.3us      WIN
       8     4096    13.094m   19.791m   18.395m   +15.2us      WIN
      16     2048    16.623m   23.595m   22.939m    +7.1us      WIN
      32     1024    24.318m   31.142m   31.839m    -7.6us     loss
      64      512    36.896m   42.545m   47.259m   -51.2us     loss
```

**The dual-stream penalty is a per-node tax on the main stream, levied for the
whole time a second queue is live.** It has nothing to do with the size of the
side payload and nothing to do with the fork barrier itself; it scales with the
number of graph nodes the main stream executes while the other stream is bound.
Below ~16 kernels/layer the overlap wins by 7–17 µs; past ~32 it collapses,
reaching −51 µs at 64.

K3's routed path per MoE layer is a long chain — router sigmoid, bias add,
top-k, sort, `moe_align`, grouped mxfp4 GEMM, `situ`, second GEMM,
scatter-reduce, plus the front GEMM and its quantization — which puts it past
the crossover. The earlier probes modelled it as a *single* `torch.mm`, i.e.
the extreme left of this table, which is exactly why they all predicted a win.

This is the same effect the pre-existing `SGLANG_ROCM_USE_MULTI_STREAM` comment
in `environ.py` records from tracing: the overlap itself is perfect (100% of
the side stream hidden, worth 952 µs per decode step) but binding the second
queue inflates the rest of the graph by 2,071 µs, **94% of it on kernels that
never run beside the side stream**. The kernel-count sweep is the controlled
version of that observation and supplies its scaling law.

### 5.5 Imitating ATOM: the whole shared MLP on the side stream (`sv5_ms_split`)

§5.3 asserted the payload "cannot be made bigger" because the shared `gate_up`
is already inside the merged `_front_w`. That was wrong on both counts — it can
be, and ATOM already does it.

ATOM has **no merged front GEMM at all**. Its `dual_stream_moe_forward`
(`atom/models/kimi_k3.py:497`) puts the entire shared MLP on the alt stream:

```python
with torch.cuda.stream(alt):
    shared_output = self.shared_experts(hidden_states)   # gate_up -> act -> down
```

And in sglang the split is *free*. `_merge_weights_as_views` cats along **dim
0** and re-points each module's `weight.data` at a slice of the merged buffer,
so `shared_experts.gate_up_proj.weight` is still a live contiguous view, and
`_front_w[gate_up_rows:]` is one too. No copy, no extra memory, no restride.
That is now `SGLANG_ROCM_K3_MULTI_STREAM_SPLIT_FRONT` (default on), applied
only inside the overlap window — outside it the merged front is kept, because
an 8192-token prefill activation is 117 MB and reading it three times is real.

It moves the side payload from 11 MB (down GEMM alone) to 33 MB (gate_up 22 MB
+ act + down). The `overlap_atom_shape.py` probe predicted +1.3…+2.8 µs/layer.
On the real model:

| conc | sv3 (no fork) | sv4a (down only) | **sv5 (ATOM shape)** | sv5 vs sv3 | sv5 vs sv4a |
|---|---|---|---|---|---|
| 2 | 94.04 | 86.99 | **87.66** | **−6.8%** | +0.8% |
| 4 | 161.47 | 150.81 | **151.72** | **−6.0%** | +0.6% |
| 8 | 238.03 | 227.08 | **227.66** | **−4.4%** | +0.3% |
| 16 | 332.18 | 323.54 | **323.41** | **−2.6%** | −0.0% |
| 32 | 385.52 | 377.81 | **377.88** | **−2.0%** | +0.0% |

Output token throughput, tok/s. Median ITL moves the same way: sv5 is +8.5%,
+8.4%, +7.4%, +5.6%, +3.3% over sv3, against sv4a's +9.3%, +9.1%, +7.3%, +5.4%,
+3.3%.

**Tripling the payload recovered 0.3–0.8% of a 2.0–6.8% loss, and nothing at
all above concurrency 8.** That is the result that sent §5.4 looking for a
different mechanism, and it is exactly what the kernel-count law predicts: the
split *moves* work between streams, it does not *remove* main-stream nodes, so
it barely touches the term that is actually costing.

The change is kept and defaults on: it is free, it is strictly better wherever
the fork runs, and it is inert while `SGLANG_ROCM_USE_MULTI_STREAM` is off.

### 5.6 Everything that was tried to make it pay

| attempt | version | result |
|---|---|---|
| Enable it as shipped (window 64…1024) | `v4_msmoe` | −2.8% to −7.1% at batch 64–256 |
| Blame HW-queue contention: `GPU_MAX_HW_QUEUES=1` | `v5a_hwq1` | penalty unchanged |
| `GPU_MAX_HW_QUEUES=2` | `v5b_hwq2` | penalty unchanged |
| Blame the number of fork sites: MoE only, no attention | `v6a_ms_nosite` | penalty unchanged |
| Attention site only | `v6b_attnonly` | −9% to −17% at batch 1–32 (that site is not token-windowed, so it fires everywhere) |
| Both sites | `v6c_attn_moe` | no worse than MoE alone — the cost does not scale with fork count |
| Blame `Stream.wait_stream`'s per-call `torch.cuda.Event` alloc: reuse a long-lived pair | `v7a_reuse` vs `v7b_noreuse` | **identical within noise**, in graph mode *and* eager (batch 1 eager prefill 0.15588 vs 0.15584 s) — hypothesis refuted |
| Retune the window down so decode fires | `sv4a_ms_decode` | −2.0% to −7.5% |
| Retune the window up so prefill fires too | `sv4b_ms_all` | zero change vs sv4a |
| **Enlarge the payload 3× — un-fuse shared `gate_up`, ATOM's shape** | **`sv5_ms_split`** | **recovers only +0.0…+0.8%; still −2.0…−6.8% vs no fork** |
| Price the barrier off-model | `fork_cost.py` | 8.1 µs/fork in a hipGraph; reused events save 0.1 µs |
| Price the overlap gain off-model, main stream idle → saturated | `overlap_gain.py` | negative at all five load points — but see below, this probe does not model the real routed path |
| Blame HBM saturation at decode | `overlap_intensity.py` | **refuted** — gain is +9…+18 µs even at 76% of peak bandwidth |
| Sweep main-stream kernel count at fixed total work | `overlap_kernelcount.py` | **the mechanism**: +17 µs at 4 kernels/layer → −8 at 32 → −51 at 64 |

The `_ms_events()` reuse machinery is kept because it is behaviour-preserving
and strictly no worse, not because it helped.

### 5.7 Verdict

Dual-stream MoE overlap is **implemented, plumbed, tunable, measured, and
optimized to ATOM's shape**, and it remains **a net loss for K3 on gfx950 at
every batch size and concurrency tested**. It ships **off**.

The mechanism is *not* the fork barrier and *not* the payload size — both were
hypothesized, both were tested, both were refuted. Off-model the same overlap
returns +9 to +18 µs per layer in every payload and bandwidth regime tried, far
more than the 8.1 µs barrier costs. What actually costs is a **per-node tax on
the main stream for as long as a second HIP queue is bound**, scaling with the
number of graph nodes the main stream runs — +17 µs/layer at 4 kernels, −8 at
32, −51 at 64, with total work held fixed. K3's routed-expert path sits past
that crossover.

That is why nothing on the payload side rescues it: enlarging the overlapped
work 3× to match ATOM recovered under 1%. The lever that would work is
**reducing the main stream's kernel count** — fusing K3's routed-expert chain —
which is an aiter-side change well outside this port.

The optimization is real on hardware, and on model shapes, where a live second
queue does not tax every node. It is not on this one.

---

## 6. Shipped defaults

| env | default | why |
|---|---|---|
| `SGLANG_ROCM_K3_FUSE_KDA_INPROJ` | **on** | +0.1…+1.1% throughput, −0.2…−2.7% ITL, no TTFT cost |
| `SGLANG_ROCM_K3_FUSE_KDA_INPROJ_MAX_TOKENS` | 256 | above this the two unfused GEMMs already saturate |
| `SGLANG_ROCM_K3_KDA_O_PROJ_FP8` | **on** (flipped from `False`) | +0.7…+1.7% throughput, −1.4…−1.8% ITL, uniform across the range |
| `SGLANG_ROCM_USE_MULTI_STREAM` | **off** | §5: −2.0…−7.5% throughput |
| `SGLANG_ROCM_K3_MULTI_STREAM_MOE` | on | only consulted when the master switch is on |
| `SGLANG_ROCM_K3_MULTI_STREAM_ATTN` | off | worse than the MoE site (§5.6) |
| `SGLANG_ROCM_K3_MULTI_STREAM_{MIN,MAX}_TOKENS` | 64 / 1024 | left at stock; §5.4 shows the penalty is per-node, not per-token, so no window is profitable and there is no better value to pick |
| `SGLANG_ROCM_K3_MULTI_STREAM_REUSE_EVENTS` | on | performance-neutral, strictly no worse |
| `SGLANG_ROCM_K3_MULTI_STREAM_SPLIT_FRONT` | **on** | free (both halves are already contiguous views); recovers +0.0…+0.8% of the dual-stream loss when the master switch is on, inert when it is off (§5.5) |

---

## 7. Accuracy

gsm8k, 400 examples, `/tmp/k3bench/accrun.sh` (server args mirror
`test/registered/amd/test_kimi_k3_dcp8_gsm8k.py` minus `--dcp-size`).

The in-projection fusion does not need this — same weights, same dtype, one
GEMM instead of two, so it is bit-exact modulo GEMM tiling. The FP8 `o_proj`
does: it quantizes a layer the checkpoint ships in bf16.

| run | flags | gsm8k |
|---|---|---|
| `acc_ref` | in-proj fusion on, FP8 `o_proj` **off** | **0.9825** |
| `acc_fp8` | in-proj fusion on, FP8 `o_proj` **on** | **0.9850** |

Per-example std is 0.131, so the standard error at n=400 is 0.66 pp. The
difference is +0.25 pp — one extra correct answer out of 400, comfortably
inside noise. **No accuracy cost from the FP8 `o_proj`.** Per-token activation
scales plus per-output-channel weight scales is a fine-grained enough recipe
that a single attention output projection absorbs it.

`ROCM_QUICK_REDUCE_QUANTIZATION=INT8` would also have needed an accuracy
defence; §8.4 shows it buys nothing, so it was not worth spending one on.

### 7.1 `SGLANG_ROCM_K3_FUSE_KDA_INPROJ` × `SGLANG_K3_KDA_FUSED_BACKEND=aiter`

Compatible, and the two are explicitly designed to cooperate. Three things had
to hold, and all do:

1. **Weights.** `_merge_bfa_weights()` *concatenates* into a new `_qkvgbfa_w`;
   `self.f_b_proj.weight` is left in place, which is what
   `_prepare_fused_decode()` reads at `kimi_k3.py:1858` when it stashes the
   aiter kernel's arguments. Load order in `load_weights` is
   `_merge_bfa_weights` → `_quantize_o_proj_fp8` → `_prepare_fused_decode`
   (`kimi_k3.py:3334-3336`), so the aiter setup always sees loaded, unmodified
   `f_b`.

2. **`defer_f_b`.** When the aiter backend is live and the step is a decode,
   `forward()` sets `defer_f_b=True` and `forward_qkvbfg_fused` returns `f_a`
   *un*-projected, because the aiter kernel applies `f_b` itself. The fused
   in-projection path takes an explicit `defer_f_b` argument for exactly this.

3. **No double norm under FP8 `o_proj`.** The aiter decode kernel folds the
   gated RMSNorm in. `forward()` guards the FP8 quant behind
   `if not fused_onorm:`, so when the kernel consumed the gate, `o_proj` gets
   the bf16 activation and `_K3PtpcFp8LinearMethod` quantizes it inside the
   linear instead — still using the FP8 weight, just not the fused-norm kernel.

Verified empirically as well, with `SGLANG_K3_KDA_FUSED_BACKEND=aiter` set:

Verified empirically as well, with `SGLANG_K3_KDA_FUSED_BACKEND=aiter` set
(`/tmp/k3bench/acc_aiter.sh`):

| run | gsm8k |
|---|---|
| `acc_aiter_plain` — aiter backend, both new opts off | **0.9750** |
| `acc_aiter_both` — aiter backend + in-proj fusion + FP8 `o_proj` | **0.9750** |

Identical. Both flags run clean on top of the aiter KDA decode backend, with
no accuracy change. (The aiter backend itself scores 0.75 pp below the default
KDA path — 0.9750 vs 0.9825, about 1.1 standard errors, so noise; and it is
pre-existing either way.)

Summary of all four accuracy runs:

| | default KDA backend | aiter KDA backend |
|---|---|---|
| both new opts off | 0.9825 | 0.9750 |
| in-proj fusion + FP8 `o_proj` | 0.9850 | 0.9750 |

All four inside ±1.5 SEM of each other.

---

## 8. The three aiter knobs

The user asked whether `AITER_USE_FLYDSL_MOE_SORTING`,
`AITER_QUICK_REDUCE_QUANTIZATION` and `AITER_MXFP4_INTERMEDIATE` help.

### 8.1 `AITER_MXFP4_INTERMEDIATE` — cannot fire for K3

`aiter/fused_moe.py:1497` gates it on

```python
BM == 128 and D_HIDDEN in (7168, 6144) and D_INTER == 512 and NE in (257, 385)
```

K3's routed MoE at TP 8 is `D_HIDDEN = 3584` (`routed_expert_hidden_size`),
`D_INTER = 3072/8 = 384`, `NE = 896`. No match on any of the three. The source
comment names the intended models — "7168: DeepSeek-V3-class; 6144: GLM-5.2" —
and warns the transform is a "lossy before-sum 4-bit quant (ok for gsm8k,
degrades other evals)". Measured anyway as `sv9d` so the claim is not just a
source read.

### 8.2 `AITER_USE_FLYDSL_MOE_SORTING` — applicable

`fused_moe.py:377` requires `not output_aux`. `output_aux=True` is only ever
set by the mxfp4 **a4w4** metadata path (`fused_moe.py:2070`); K3 under
`AITER_SITUV2_A8W4=1` is **afp8_wfp4** (`fused_moe.py:666-673`) and takes the
ordinary path. So the FlyDSL sorting backend is genuinely reachable here.
Measured as `sv9a`.

### 8.3 `AITER_QUICK_REDUCE_QUANTIZATION` — wrong variable name for sglang

That name is read only by
`aiter/dist/device_communicators/quick_all_reduce.py`, which sglang never
instantiates. sglang has its own `QuickAllReduce` keyed on
**`ROCM_QUICK_REDUCE_QUANTIZATION`** (default `NONE` = never constructed;
regimes `FP` / `INT8` / `INT6` / `INT4`; gfx94x/gfx95x, world size 2/4/8),
with `ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` (default 1) and
`ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB` alongside.

Dispatch order (`parallel_state.py:896-908`) is custom-AR **first**
(`AiterCustomAllreduce`, confirmed active in the server log), quick-reduce only
on tensors custom-AR declines — i.e. those over its `max_size`
(`custom_all_reduce.py:280`, `inp_size <= self.max_size`). On this workload a
prefill chunk's all-reduce is 8192 × 7168 × 2 B = **117 MB**, far over that, so
today those fall through to RCCL — exactly the band quick-reduce exists for,
and at ISL 8192 prefill is roughly half the wall clock at low concurrency.
Decode's 2–32-token all-reduces are ~0.7 MB and stay on custom-AR either way.

**Prediction: TTFT moves, ITL does not.** Measured as `sv9b` (FP, lossless
enough to ship) and `sv9c` (INT8, lossy — would need gsm8k before it could
ever be a default).

### 8.4 Results — all three measured, none of them helps

All four runs are `sv3_fp8oproj` plus exactly one variable.

**Output token throughput (tok/s), Δ vs sv3:**

| conc | sv3 | sv9a flydsl_sort | sv9b qr=FP | sv9c qr=INT8 | sv9d mxfp4_inter |
|---|---|---|---|---|---|
| 2 | 94.04 | 93.34 (−0.8%) | 93.83 (−0.2%) | 94.16 (+0.1%) | 94.10 (+0.1%) |
| 4 | 161.47 | 159.94 (−0.9%) | 160.57 (−0.6%) | 161.60 (+0.1%) | 161.58 (+0.1%) |
| 8 | 238.03 | 236.44 (−0.7%) | 235.92 (−0.9%) | 238.19 (+0.1%) | 238.03 (+0.0%) |
| 16 | 332.18 | 328.40 (−1.1%) | 328.34 (−1.2%) | 332.37 (+0.1%) | 332.19 (+0.0%) |
| 32 | 385.52 | 379.26 (−1.6%) | 379.68 (−1.5%) | 385.11 (−0.1%) | 385.03 (−0.1%) |

**Median TTFT (ms), Δ vs sv3:**

| conc | sv3 | sv9a | sv9b qr=FP | sv9c qr=INT8 | sv9d |
|---|---|---|---|---|---|
| 2 | 2918 | 2920 (+0.1%) | 2994 (**+2.6%**) | 2920 (+0.1%) | (+0.1%) |
| 4 | 5080 | 5088 (+0.2%) | 5215 (**+2.7%**) | 5082 (+0.0%) | (+0.0%) |
| 8 | 8041 | 8059 (+0.2%) | 8255 (**+2.7%**) | 8048 (+0.1%) | (+0.1%) |
| 16 | 13884 | 13901 (+0.1%) | 14242 (**+2.6%**) | 13890 (+0.0%) | (+0.1%) |
| 32 | 25467 | 25553 (+0.3%) | 26343 (**+3.4%**) | 25551 (+0.3%) | (+0.2%) |

**Median ITL (ms), Δ vs sv3:** sv9a +0.9 / +1.2 / +0.9 / +2.3 / +1.5 %;
sv9b −0.2 / −0.1 / +0.0 / −0.0 / −0.1 %; sv9c −0.2 / −0.1 / −0.1 / −0.0 /
+0.2 %; sv9d −0.1 / −0.1 / −0.1 / +0.2 / +0.2 %.

**`AITER_USE_FLYDSL_MOE_SORTING=1` — small but real regression, −0.7 to −1.6%.**
It is reachable, as the source read said, and it is simply the slower sorting
backend for K3's shape (896 experts, topk 16). The regression grows with
concurrency, which is what you expect if the cost is in the sort itself:
more tokens, more to sort. Leave it off.

**`ROCM_QUICK_REDUCE_QUANTIZATION` — the prediction was right about *where*, wrong
about the sign.** As predicted, it moves TTFT and leaves ITL alone: at `FP`,
TTFT is +2.6 to +3.4% at every concurrency while ITL is flat within ±0.2%.
That is exactly the fingerprint of a change confined to the 117 MB prefill
all-reduce, and it confirms those really were falling through to RCCL. But
quick-reduce is *slower* than RCCL there.

`INT8` makes the mechanism explicit. The `FP` regime casts bf16→fp16 — still
two bytes, so it moves the same traffic as RCCL and pays quick-reduce's
overhead for nothing: +2.7%. `INT8` is one byte, halving the traffic, and
lands back at exactly parity (+0.0 to +0.3%). So quick-reduce's kernel is
~2.7% of TTFT worse than RCCL at equal bytes, and halving the bytes buys back
precisely that and no more. There is no configuration where it wins, and
`INT8` would additionally need an accuracy defence to earn a net-zero result.
Leave `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`.

**`AITER_MXFP4_INTERMEDIATE=1` — an exact no-op, as the gate predicted.** Every
metric within ±0.2% of sv3 at every concurrency. The kernel is never selected
for K3's `(D_HIDDEN 3584, D_INTER 384, NE 896)`, so the flag has nothing to
switch on. Setting it is harmless and pointless.

**Answer to the question as asked: no, none of the three helps this workload.**
Two are measurable regressions and one cannot fire.

---

## 9. Appendix — `one_batch` micro-sweeps

These predate the workload being pinned to ISL 8192 / OSL 1024 and are kept
because they isolate a single decode step far better than a server can, which
is what the dual-stream investigation in §5 rests on. `--context-length 5200`,
input_len 64, `--output-len 65`, each batch size listed three times so the
repetitions interleave; every figure is best-of-three with the min-to-max
spread reported.

**Median decode latency (s), Δ vs previous version:**

| batch | v1_baseline | v1b_patched_off | Δ | v2_inproj | Δ | v3_fp8oproj | Δ |
|---|---|---|---|---|---|---|---|
| 1 | 0.01718 | 0.01717 | −0.1% | 0.01721 | +0.2% | 0.01705 | −0.9% |
| 8 | 0.02170 | 0.02165 | −0.2% | 0.02115 | −2.3% | 0.02093 | −1.0% |
| 32 | 0.02765 | 0.02759 | −0.2% | 0.02607 | −5.5% | 0.02635 | +1.1% |
| 64 | 0.03446 | 0.03396 | −1.5% | 0.03435 | +1.1% | 0.03369 | −1.9% |
| 128 | 0.04441 | 0.04464 | +0.5% | 0.04543 | +1.8% | 0.04500 | −0.9% |
| 256 | 0.06294 | 0.06307 | +0.2% | 0.06265 | −0.7% | 0.06234 | −0.5% |

`v1b_patched_off` is the control: the patch applied with every new flag off.
It matches `v1_baseline` to within the spread at every batch size — the code
changes cost nothing when disabled.

**Median decode throughput (tok/s):**

| batch | v1_baseline | v2_inproj | v3_fp8oproj | v1→v3 |
|---|---|---|---|---|
| 1 | 58.22 | 58.11 | 58.66 | +0.8% |
| 8 | 368.74 | 378.28 | 382.21 | +3.7% |
| 32 | 1157.18 | 1227.25 | 1214.51 | +5.0% |
| 64 | 1857.29 | 1863.11 | 1899.54 | +2.3% |
| 128 | 2881.91 | 2817.52 | 2844.32 | −1.3% |
| 256 | 4067.39 | 4086.06 | 4106.69 | +1.0% |

The in-projection fusion peaks at batch 32 (+5.8% on its own step) and turns
slightly negative at 128 (−1.7%), consistent with `..._MAX_TOKENS=256`: by 128
the two unfused GEMMs are large enough that merging stops paying, and at 256
the guard is about to switch it off anyway. On the serving workload decode
batches are 2–32, squarely in the fusion's best band.

The dual-stream rows (`v4_msmoe`, `v5a/b_hwq*`, `v6a/b/c`, `v7a/b`) are
tabulated in §5.6.

---

## 10. Consolidated version ladder

Every row is output token throughput in tok/s. `Δ` is against the version to
its immediate left; `vs sv1` is against the baseline. sv4a/sv4b/sv5 all branch
off sv3, so their Δ is against sv3, not against each other. sv9a–d likewise
branch off sv3 and are tabulated in §8.

| conc | sv1 baseline | sv2 +inproj | Δ | sv3 +fp8 o_proj | Δ | sv4a +dual-stream | Δ vs sv3 | sv5 +ATOM-shape split | Δ vs sv3 | best vs sv1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 91.53 | 92.49 | +1.0% | **94.04** | +1.7% | 86.99 | −6.8% | 87.66 | −6.8% | **+2.7%** |
| 4 | 158.97 | 159.39 | +0.3% | **161.47** | +1.3% | 150.81 | −6.6% | 151.72 | −6.0% | **+1.6%** |
| 8 | 234.79 | 235.12 | +0.1% | **238.03** | +1.2% | 227.08 | −4.6% | 227.66 | −4.4% | **+1.4%** |
| 16 | 326.56 | 329.56 | +0.9% | **332.18** | +0.8% | 323.54 | −2.6% | 323.41 | −2.6% | **+1.7%** |
| 32 | 378.61 | 382.87 | +1.1% | **385.52** | +0.7% | 377.81 | −2.0% | 377.88 | −2.0% | **+1.8%** |

Median ITL (ms), the metric these decode-side changes actually move:

| conc | sv1 | sv2 | sv3 | sv4a | sv5 | sv1→sv3 |
|---|---|---|---|---|---|---|
| 2 | 18.71 | 18.76 | 18.41 | 19.97 | 19.98 | **−1.6%** |
| 4 | 19.44 | 19.37 | 19.07 | 20.68 | 20.68 | **−1.9%** |
| 8 | 22.59 | 22.55 | 22.16 | 23.79 | 23.80 | **−1.9%** |
| 16 | 26.07 | 25.65 | 25.26 | 26.67 | 26.68 | **−3.1%** |
| 32 | 32.43 | 31.55 | 31.10 | 32.13 | 32.13 | **−4.1%** |

**Reading the ladder.** Two of the four ported changes pay, in the same
direction at every concurrency, and both are decode-side (TTFT never moves
outside ±0.4%, §4.4). The two dual-stream variants lose at every concurrency,
by an amount that shrinks as concurrency grows — the shape of a fixed per-step
tax being amortized over more tokens, which is exactly the per-graph-node
mechanism §5.4 isolates. sv5 (ATOM's larger side payload) recovers between
+0.8% and 0.0% of that loss: real, free, and nowhere near enough.

Shipped configuration is **sv3**: `SGLANG_ROCM_K3_FUSE_KDA_INPROJ=1`,
`SGLANG_ROCM_K3_KDA_O_PROJ_FP8=1`, `SGLANG_ROCM_USE_MULTI_STREAM=0`.

---

## 11. Comparison against ATOM PR #1752

ATOM's [PR #1752](https://github.com/ROCm/ATOM/pull/1752) ("enable dual stream
for shared expert and some fusions", merged 2026-08-06, local commit
`d182e770`, 11 files / +618) is the upstream this port follows. Its own E2E
table is at ISL 8192 / OSL 1024 on MI355 TP8 — nominally the same workload as
this report.

### 11.1 Their numbers

| conc | main out tok/s | PR out tok/s | Δ | main med TTFT | PR med TTFT |
|---|---|---|---|---|---|
| 4 | 142.59 | 147.14 | **+3.2%** | 557 ms | 512 ms |
| 8 | 248.98 | 263.76 | **+5.9%** | 549 ms | 511 ms |
| 16 | 417.48 | 401.82 | **−3.8%** | 566 ms | 523 ms |
| 32 | 624.52 | 656.35 | **+5.1%** | 597 ms | 556 ms |
| 64 | 839.41 | 923.87 | **+10.1%** | 799 ms | 760 ms |

gsm8k 0.9568 ± 0.0056 (5-shot, full set).

### 11.2 Absolute throughput is not comparable, and the reason is visible in TTFT

Their median TTFT is **512–799 ms**; mine is **2918–25467 ms** — 6× at
concurrency 4 rising to 32× at 32. An 8192-token prefill at 64-way concurrency
cannot complete in 799 ms on this hardware; 64 × 8192 = 524288 tokens of
prefill, and my own measured prefill rate puts that in the tens of seconds. A
TTFT that is flat in concurrency (557 → 799 ms from 4 to 64) is the fingerprint
of prompts that share a prefix and hit a cache, not of 64 independent 8192-token
prefills. My harness sets `--disable-radix-cache` and
`--random-range-ratio 1.0` specifically so that cannot happen (§1.1).

So the two stacks are not being asked the same question, and **their absolute
tok/s must not be read against mine.** Only the relative main→PR deltas are
comparable, and even those carry the confound in §11.3.

### 11.3 Their table bundles the two changes; mine separates them

ATOM's `main → PR` delta is the whole PR at once: the RMSNorm→FP8 fusions,
the ptpc_fp8 online quant of every dense linear, *and* the dual-stream MoE
overlap. Nothing in their table isolates the overlap. This report's ladder
does — sv2 and sv3 are the fusions alone, sv4a/sv5 add only the overlap:

| | ATOM (bundled) | this port, fusions only (sv1→sv3) | this port, overlap only (sv3→sv4a) |
|---|---|---|---|
| conc 4 | +3.2% | +1.6% | −6.6% |
| conc 8 | +5.9% | +1.4% | −4.6% |
| conc 16 | −3.8% | +1.7% | −2.6% |
| conc 32 | +5.1% | +1.8% | −2.0% |

### 11.4 Their concurrency-16 anomaly is the same effect this report measures

The PR's own "Remaining issue" reads: *"Overlap failed at CONC=16, caused some
regression instead, which is strange."* At that point their bundle goes **−3.8%
overall** — and that is with the fusions (worth roughly +1.7% here) already
inside it, so the overlap component alone is costing them on the order of
−5%. My isolated measurement of the same overlap at concurrency 16 is **−2.6%**.

Same hardware family, same sign, same order of magnitude, arrived at
independently. Their result is not strange: on gfx950 the fork is a **per-node
tax on the main stream for as long as a second HIP queue is bound** (§5.4), so
it costs wherever the main stream is a long chain of small kernels — which K3's
routed-expert path always is. What is unusual about their table is not the
conc-16 row but the other four, and the most likely explanation is that at
those points the ptpc_fp8 half of the bundle is large enough to cover the
overlap's loss.

### 11.5 What that implies for the gains they report

Their +3.2 / +5.9 / +5.1 / +10.1% cannot be attributed to the overlap. Once the
overlap is subtracted (negative on this hardware) the residue has to come from
the quantization half — the part of the PR this port has only partially
adopted. That is §12.

---

## 12. ATOM optimizations not (or only partly) present in sglang

Item-by-item against `d182e770`, plus the follow-up `dbc49311` (PR #1792).

| # | ATOM change | in sglang? | note |
|---|---|---|---|
| a | Dual-stream shared/routed MoE overlap | **ported, shipped off** | sv4a/sv4b/sv5; measured a net loss at every point (§5) |
| b | `rmsnorm_gated` fused to per-token FP8, feeding KDA `o_proj` | **ported, shipped on** | sv3; `SGLANG_ROCM_K3_KDA_O_PROJ_FP8` |
| c | **ptpc_fp8 on every dense linear**, activation quant folded into `input_layernorm` / `post_attention_layernorm` | **NOT ported** | the big one — see §12.1 |
| d | `compressed-tensors` added to the online-quant allow-list | n/a | an ATOM-internal bug: their `--online_quant_config` was being silently ignored for K3. sglang applies ptpc explicitly per layer, so there is no allow-list to miss |
| e | Skip `process_weights_after_loading` for empty fused shells (`weight.numel()==0`) | n/a | ATOM empties the source modules when it concatenates into `in_proj`. sglang's `_merge_weights_as_views` re-points each module's `weight.data` at a **view** of the merged buffer, so no module is ever left empty |
| f | Pad a8w8 fp8 preshuffle output up to the CK N-tile (128) | **not needed yet, needed for (c)** | the one layer sglang quantizes has N=7168, already 128-aligned. Extending to (c) hits `g_proj`, `f_b_proj` and the KDA in-proj shells, whose N is not |
| g | Both TP all-reduces launched on two streams, launch order reordered | **not ported** | only meaningful with (a) on, which it is not |
| h | `rope_max_position` read from the text config | n/a | ATOM was falling back to `max_model_len` because `max_position_embeddings` lives under `text_config`. sglang's `KimiK3Config` resolves the nested text config at load, so the value is already correct |
| i | Gate dual-stream by runtime graph mode (FULL keeps overlap, PIECEWISE does not) — PR #1792 | **not ported** | ditto (g). Noted because it shows ATOM also found the overlap unsafe/unprofitable under piecewise capture |

### 12.1 The one real gap: ptpc_fp8 across the dense linears

ATOM's recipe turns on

```
--online_quant_config '{"global_quant_config": "ptpc_fp8",
  "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*",
                    "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*",
                    "*vision_tower*", "*mm_projector*"]}'
```

The K3 checkpoint is `mxfp4-pack-quantized` with
`ignore: [".*self_attn.*", ".*shared_experts.*", ".*mlp\.(gate|up|gate_up|down)_proj.*", ".*lm_head.*", ...]`
— i.e. **only the routed experts are quantized on disk; every attention linear
and both shared-expert GEMMs ship bf16.** ATOM's exclude list re-quantizes
essentially all of them online to per-token-activation / per-channel-weight FP8:

* MLA: `fused_qkv_a_proj`, `q_b_proj`, `kv_b_proj`, `g_proj`, `o_proj`
* KDA: the fused `in_proj`, `f_b_proj`, `o_proj`
* MoE: `shared_experts.gate_up_proj`, `shared_experts.down_proj`, and the router `gate`

and folds the activation quant into the preceding RMSNorm so the activation is
written once rather than normed→written→quantized→written.

sglang does **one** of these (KDA `o_proj`, sv3) and that single layer is worth
+0.7…+1.7% throughput and −1.4…−1.8% ITL. The shared-expert `gate_up`
(`[1536, 7168]`, 22 MB) and `down` (`[7168, 768]`, 11 MB) and the MLA
projections are collectively an order of magnitude more weight traffic per
decode step than that one `o_proj`, so this is where ATOM's residual gain
(§11.5) most plausibly lives.

**What porting it needs** (not attempted here — it is a larger change than the
remaining items and wants its own accuracy run):

1. A per-layer online FP8 weight quantization pass at load. sglang already has
   the GEMM (`apply_fp8_ptpc_linear` → `aiter.gemm_a8w8_bpreshuffle`) and sv3
   already does the load-time quantize for one layer; it needs generalizing to
   a layer list.
2. N-tile padding, ATOM item (f) — `g_proj` (N=96 at TP8) and the KDA in-proj
   shells are not 128-aligned and `shuffle_weight(..., (16,16))` will reject
   or mis-tile them.
3. A fused RMSNorm→per-token-FP8 on ROCm for the *un-gated* norms. sglang's
   fused rmsnorm+quant is FlashInfer-only (CUDA); the ROCm side has
   `fused_rms_mxfp4_quant` for the mxfp4 layout but no per-token fp8
   equivalent wired up. sv3's gated variant
   (`kernels/ops/kimi_k3/rmsnorm_gated_quant.py`) is the template.
4. gsm8k before shipping: this quantizes ~10 layer types the checkpoint ships
   in bf16, where sv3 quantized one.

Estimated ceiling, extrapolating from sv3's single layer by weight bytes moved
per decode step: low single-digit percent, which is consistent with the residue
in §11.5 once the overlap's loss is added back.

---

## 13. sv10 — DCP (decode context parallelism) enabled

`--dcp-size 8` on top of sv3. DCP round-robins KV cache slots across the 8
ranks (`loc % dcp_size == rank`), so each rank stores and attends over 1/8 of
the context; decode all-gathers Q over the DCP group and reduce-scatters the
output, merging the per-rank partial attentions through their log-sum-exps.

It was off in v1–v9 because it crashed during decode cuda-graph capture. It now
runs. Three separate bugs, all on the ROCm path, all fixed.

### 13.1 Bug 1 + 2 — the aiter MLA path never used the DCP-sized attention layer

`deepseek_v2.py:1904` builds a second `RadixAttention` when DCP is on:

```python
# use num_local_heads * dcp_world_size because q_nope, q_rope is all gathered from dcp ranks
if get_parallel().dcp_enabled:
    self.attn_mqa_for_dcp_decode = RadixAttention(
        self.num_local_heads * get_parallel().attn_dcp_size, ...
    )
```

Same `layer_id` and same `prefix` as `attn_mqa`, so it shares the KV writes and
the quant scales; the only difference is that its head count is the **gathered**
96 rather than the local 12.

`forward_absorb_*_core` dispatches to it only inside the branch guarded by
`FORWARD_ABSORB_CORE_ATTENTION_BACKENDS` (`deepseek_common/utils.py:60`) — a
list of fa3, fa4, dsa, nsa, flashinfer, cutlass_mla, trtllm_mla, cutedsl_mla,
tokenspeed_mla, ascend, intel_xpu. **`"aiter"` is not in it.** So on ROCm the
`else` branch called plain `self.attn_mqa`, which is still sized for 12 heads,
with a Q that had already been gathered to 96:

```
RuntimeError: shape '[32, 12, 576]' is invalid for input of size 1769472
    aiter_backend.py:1076
```

1769472 / (32·12·576) = **8** = `dcp_size`, exactly. Fixing the head count
inside `aiter_backend.py` clears that crash and immediately produces the next
one — the DCP kernels return `(out, lse)` for the cross-rank merge, and the
`else` branch never unpacked the tuple:

```
AttributeError: 'tuple' object has no attribute 'view'
    forward_mla_rocm.py:736
```

Both are the same root cause, so both are fixed in one place — route the DCP
decode/target-verify phase to `attn_mqa_for_dcp_decode` in the `else` branch
too:

```python
if is_dcp_mla_decode_phase(forward_batch):
    attn_output, lse = self.attn_mqa_for_dcp_decode(...)
else:
    attn_output = self.attn_mqa(...)
```

`forward_mla_rocm.py` (aiter) and, by fix-then-sweep, `forward_mla.py` — the
CUDA sibling has the identical gap, reachable by flashmla, which
`is_mla_dcp_lse_base_on_e` explicitly names as a DCP backend yet which is also
absent from the allow-list.

Fixing it at the model level rather than in the backend matters: it makes the
backend's existing comments (`_gathered_num_head = self.num_head *
self.dcp_world_size`) literally true instead of double-counting. Patching
`aiter_backend.py` *and* passing the DCP layer would have given 12·8·8 = 768
heads.

### 13.2 Bug 3 — the DCP KV write was both uncapturable and wrong

With the dispatch fixed, capture got as far as the KV write and died:

```
torch.AcceleratorError: HIP error: operation not permitted when stream is capturing
    memory_pool.py:4037   if not valid_mask.all():
```

`MLATokenToKVPool.set_kv_buffer` — the single-tensor write the aiter MLA path
uses — filtered the write down to the rows this rank owns by compacting them:

```python
valid_mask = loc % parallel.attn_dcp_size == parallel.attn_dcp_rank
if not valid_mask.all():          # device -> host sync
    loc = loc[valid_mask]         # data-dependent output shape
    cache_k = cache_k[valid_mask]
```

Both lines are illegal under graph capture: `.all()` in a Python `if` forces a
device→host sync, and boolean indexing has a shape only the device knows.

**And the surviving rows went to the wrong place.** Under DCP the allocator is
built over a *widened* loc space — `max_total_num_tokens * dcp_size` slots at
`page_size * dcp_size` per page (`kv_cache_configurator.py:1599`) — so
`out_cache_loc` is a virtual index in `[0, size * dcp_size)`, the owner is `loc
% dcp_size`, and **the physical row in this rank's pool is `loc //
dcp_size`**. Every reader already follows that convention:

* `set_mla_kv_buffer` (the two-tensor path FA3 uses on CUDA) says so in as many
  words — *"loc is widened under DCP; the kernel divides by the world size
  itself"* — and its kernel does exactly
  `safe_loc = tl.where(is_valid, loc, 0) // DCP_WORLD_SIZE`
  (`kernels/ops/kvcache/mla_buffer.py:39-41`);
* the DCP planner builds its read indices as
  `idx[idx % dcp_size == dcp_rank] // dcp_size` (`layers/dcp/planner.py:113`);
* aiter's own decode helper documents `kv_values // dcp_world_size` as *"physical
  rows of this rank's pool"* (`aiter_backend.py:1055`).

`set_kv_buffer` never divided. Owned rows were written `dcp_size` rows too far
into the buffer while the kernels read at `loc // dcp_size`, so the MLA layers
attended over whatever happened to be at the low end of the pool. It is not a
crash — K3 is 93 layers of which only ~23 are MLA and the rest are KDA, so the
model still produces fluent, mostly-wrong text. **gsm8k measured 0.576 against a
0.958 baseline.**

Both problems are fixed by writing the same thing the canonical kernel writes:

```python
valid_mask = loc % parallel.attn_dcp_size == parallel.attn_dcp_rank
loc = torch.where(valid_mask, loc, torch.zeros_like(loc))
loc = loc // parallel.attn_dcp_size
```

Rows this rank does not own land in the reserved dummy slot 0 — which exists for
exactly this, every allocator starting its free list at 1
(`allocator/paged.py:332`, *"The padded slot 0 is used for writing dummy outputs
from padded tokens"*) — and owned rows land on their physical row. The
`maybe_detect_oob` bound is widened by `attn_dcp_size` to match, as
`set_mla_kv_buffer` already does. The form is capture-safe, and it also removes
**one host sync per MLA layer per forward** that the compacting version paid
unconditionally.

Post-fix: `test_kimi_k3_dcp8_gsm8k.py` passes both tests, **gsm8k = 0.9521**
(n=1319) against the 0.958 reference — inside the ±0.6 pt binomial SE and the
0.68 pt run-to-run spread the test file documents.

Not fixed, because nothing on this workload exercises them and a blind edit
would be untested: `move_kv_cache` (spec-decode accept) and
`get_cpu_copy`/`load_cpu_copy` (hicache offload) on the same pool also index
`kv_buffer` with a raw loc and have no DCP division. They are the same
convention violation and will need the same treatment before DCP is combined
with speculative decoding or CPU offloading.

### 13.3 Results — sv10_dcp8 vs sv3

Same harness, same flags, `--dcp-size 8` added. Three full sweeps: r1 and r2
before the §13.2 address fix, r3 after it. r1's concurrency-4 point hit a
scheduling straggler (p99 TTFT 12234 ms against a 3169 ms median, σ(e2e)
3333 ms vs 779 ms in r2) and is dropped. **r3 is the number to quote** — it is
the only one that is also numerically correct — and it lands within 0.2% of r2
at every point, which is the expected result: the address bug changed where
rows were written, not how many.

| conc | sv3 out tok/s | r2 (pre-fix) | **r3 (correct)** | Δ vs sv3 |
|---|---|---|---|---|
| 2 | 94.04 | 97.54 | **97.24** | **+3.4%** |
| 4 | 161.47 | 167.38 | **167.02** | **+3.4%** |
| 8 | 238.03 | 249.64 | **249.25** | **+4.7%** |
| 16 | 332.18 | 353.80 | **353.61** | **+6.4%** |
| 32 | 385.52 | 410.63 | **411.05** | **+6.6%** |

The whole gain is prefill, and it is large:

| conc | median TTFT sv3 | r3 | Δ | median ITL sv3 | r3 | Δ |
|---|---|---|---|---|---|---|
| 2 | 2917.6 ms | 1312.4 | **−55.0%** | 18.41 ms | 19.02 | +3.3% |
| 4 | 5080.4 | 2916.8 | **−42.6%** | 19.07 | 20.09 | +5.4% |
| 8 | 8041.4 | 6000.6 | **−25.4%** | 22.16 | 23.31 | +5.2% |
| 16 | 13884.4 | 10914.8 | **−21.4%** | 25.26 | 26.35 | +4.3% |
| 32 | 25466.5 | 20406.7 | **−19.9%** | 31.10 | 32.09 | +3.2% |

Prefill does **not** all-gather Q — each rank computes attention against its own
1/8 of the KV and the partials are merged — so an 8192-token prefill costs each
rank an eighth of the attention work. Decode does gather Q, and there the
all-gather / reduce-scatter round trip slightly outweighs the sharded
attention: ITL is uniformly 3–5% worse. Since this workload spends most of its
wall clock in prefill (ISL 8192 vs OSL 1024), the net is positive at every
concurrency, and it grows with concurrency as prefill queueing dominates.

Both configurations use the same all-reduce path (`[AR] All-reduce call path:
NCCL (custom AR disabled)` in both logs), so the delta is not a communication
backend change.

### 13.4 The memory benefit is real and is 8×

`--max-total-tokens 311296` is pinned in the harness so every version measures
against the same pool, which makes the two runs *look* identical: both report
`max_total_num_tokens=311296` and both allocate `KV Cache ... 8.02 GB`. The
scheduler's own accounting shows what actually changed — at the same occupancy:

```
sv3        #full token: 27435,  full token usage: 0.09
sv10_dcp8  #full token: 27648,  full token usage: 0.01
```

0.09 = 27435 / 311296. 0.011 = 27648 / **2490368** = 311296 × 8. The pinned
number is the *physical per-rank* pool; DCP makes the allocator's virtual loc
space `dcp_size` times larger (`kv_cache_configurator.py:301`), so the same
8.02 GB per GPU now holds 8× the context. At this pin the extra capacity is
unused headroom; its value is that the same hardware serves 8× the context, or
the same context at 1/8 the KV footprint.

### 13.5 Status

DCP works on the ROCm/aiter K3 path and is a win on this workload:

* **+3.4 … +6.6%** output throughput over sv3, growing with concurrency
* **−20 … −55%** median TTFT
* **8×** KV capacity per GPU at identical bytes
* **−3 … −5%** ITL (the one cost)
* **gsm8k 0.9521** (n=1319) vs the 0.958 reference — `test_kimi_k3_dcp8_gsm8k.py`
  passes both its config assertion and its accuracy assertion

All three fixes are correctness fixes rather than tuning, and two of them
(§13.1 in `forward_mla.py`, §13.2) are not ROCm-specific — the widened-loc bug
in `set_kv_buffer` hits any DCP backend that writes MLA KV through the
single-tensor path.

Remaining before DCP can be a default: the two untouched pool methods listed at
the end of §13.2 (`move_kv_cache`, `get_cpu_copy`/`load_cpu_copy`) block
combining DCP with speculative decoding or hicache offload.

---

## 14. Reproducing

```
/tmp/k3bench/
  serve_bench.sh <tag>     one version, all five concurrency points
  serve_bench_dcp.sh <tag> same, with --dcp-size 8 (sv10, §13)
  svall.sh / svrest2.sh    sv1..sv4b
  sv9.sh                   the three aiter knobs
  sv_summarize.py <tags>   Δ tables, six metrics
  accrun.sh                gsm8k, FP8 o_proj off vs on
  dcp8_gsm8k.log           gsm8k under --dcp-size 8 (the registered CI test)
  acc_aiter.sh             gsm8k, aiter KDA backend compatibility
  fork_cost.py             prices a hipGraph fork/join, off-model
  overlap_gain.py          prices what that fork buys, off-model
  overlap_atom_shape.py    prices ATOM's 3x-larger side payload (predicts a win)
  overlap_intensity.py     tests HBM saturation as the cause (refutes it)
  overlap_kernelcount.py   sweeps main-stream kernel count at fixed total work
                           -- this is the one that finds the real mechanism
                           (all five probes are committed under
                            benchmark/kimi_k3_dual_stream/)
  k3_opts.patch            the whole change set, 7 files
  serving_<tag>.jsonl      raw bench_serving output
  serve_<tag>.log          raw server log
  run.sh / all.sh / summarize.py   the one_batch harness behind §9
```
