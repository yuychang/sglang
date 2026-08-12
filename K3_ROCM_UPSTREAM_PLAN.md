# Upstreaming plan — Kimi-K3 ROCm optimizations

Branch: `kimi-k3-optimizations`. Measurements and rationale behind every number
here: `K3_ROCM_OPT_REPORT.md`.

Nine commits sit on this branch. Excluding the dual-stream MoE work, they go
upstream as **three PRs**. The split follows reviewer and risk, not file count:
PR 1 changes shared code that every DCP user runs, PR 2 changes only a ROCm
weight layout, PR 3 changes numerics.

| # | PR | commits | files | diff |
|---|---|---|---|---|
| 1 | `fix(dcp)`: make DCP work with the non-absorb-core MLA decode | `39716d4d0e`, `e037bec72b` | 3 | +88 / −21 |
| 2 | `feat(kimi-k3)`: fuse the KDA input projection on ROCm | `7897b8df00`, `5e6a60cd72` | 4 | +219 / −8 |
| 3 | `feat(kimi-k3)`: gated RMSNorm fused into a per-token FP8 o_proj | `cc25f65e94`, `6480c391ec` | 6 | +613 / −3 |

Land PR 1 first. PR 2 and PR 3 are functionally independent of it and of each
other and can go up in parallel; they touch adjacent lines in `kimi_k3.py` and
`environ.py`, so whichever lands second needs a trivial rebase.

---

## PR 1 — `fix(dcp)`: make DCP work with the non-absorb-core MLA decode

**Commits**

- `39716d4d0e` fix(dcp): route the non-absorb-core MLA decode through the DCP-sized attention
- `e037bec72b` fix(dcp): divide the widened loc in the MLA single-tensor KV write

**Files**

- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py`
- `python/sglang/srt/mem_cache/memory_pool.py`

**Why it goes first, and alone.** This is the only PR touching shared,
non-ROCm code. `forward_mla.py` is the flashmla path and `memory_pool.py` is
every MLA model's KV write, so the blast radius is every DCP user, and the
reviewers are the DCP/MLA owners rather than the ROCm folks. Nothing else on
the branch depends on it.

**Why both commits stay in one PR.** The dispatch fix alone yields a DCP run
that is fast and numerically wrong — it stops crashing, throughput reads +6%,
and gsm8k reads 0.576. The two have to be reviewed and validated as a unit.

**Evidence for the description**

- Crash signature: `shape '[32, 12, 576]' is invalid for input of size 1769472`
  (ratio is exactly `dcp_size`), then `'tuple' object has no attribute 'view'`.
  Root cause: `"aiter"` is not in `FORWARD_ABSORB_CORE_ATTENTION_BACKENDS`, so
  the fallback branch called `attn_mqa` (`num_local_heads`) with a q the DCP
  path had already all-gathered to `num_local_heads * attn_dcp_size`.
- The widened-loc contract, with `kernels/ops/kvcache/mla_buffer.py:39-41` as
  the reference the single-tensor path now matches, and
  `kv_cache_configurator.py:1599` as where the widening is set up.
- gsm8k 0.576 → **0.9521** (n=1319) under `--dcp-size 8`, TP8, aiter/gluon,
  gated by `test/registered/amd/test_kimi_k3_dcp8_gsm8k.py`.
- Result once fixed: +3.4…+6.6% output throughput, median TTFT −20…−55%,
  ITL +3…+5%, and 8× KV capacity per GPU at identical bytes.

**Call out explicitly in the description:** `move_kv_cache` (spec-decode
accept) and `get_cpu_copy` / `load_cpu_copy` (hicache) index the same pool with
a raw widened loc and still have no DCP division. They are untestable on this
workload and are left alone, so DCP + speculative decoding and DCP + offload
remain broken. A reviewer must not read this PR as "DCP is now correct
everywhere."

---

## PR 2 — `feat(kimi-k3)`: fuse the KDA input projection on ROCm

**Commits**

- `7897b8df00` feat(kimi-k3): fuse the KDA input projection into one GEMM on ROCm
- `5e6a60cd72` style(kimi-k3): reformat a test line with black — drive-by, keeps
  `black --check` green on `test/registered/amd`

**Files**

- `python/sglang/srt/models/kimi_k3.py`
- `python/sglang/srt/environ.py`
- `test/registered/amd/test_kimi_k3_kda_inproj_fusion.py`
- `test/registered/amd/test_kimi_k3_aiter_mla_kernels.py` (formatting only)

**Why it is its own PR.** Self-contained, ROCm-only, env-gated
(`SGLANG_ROCM_K3_FUSE_KDA_INPROJ`), and layout-only — no numerics change beyond
bf16 rounding from a different GEMM kernel. It should not wait behind the
FP8 review.

**Evidence for the description**

- The M-sweep that sets the default threshold at 256: on gfx950 at H=7168/TP8,
  one GEMM is 0.55–0.73× the split at M≤64 and 0.83–0.97× through M=256; past
  that the untuned N=6288 config costs more than the launch it saves.
- Why NVIDIA keeps the split at every M (cublas picks worse kernels for 6288 —
  the existing `fused_qkvg_proj` comment).
- `test_kimi_k3_kda_inproj_fusion.py` pins the slice offsets, the tail view the
  split path still reads, and that the strided f_a slice is a legal input to
  the f_b GEMM.

---

## PR 3 — `feat(kimi-k3)`: gated RMSNorm fused into a per-token FP8 o_proj

**Commits**

- `cc25f65e94` feat(kimi-k3): fuse the gated output RMSNorm into a per-token FP8 o_proj
- `6480c391ec` test(kimi-k3): add a numerical check for the AITER fused KDA decode

**Files**

- `python/sglang/kernels/ops/kimi_k3/rmsnorm_gated_quant.py` (new kernel)
- `python/sglang/kernels/ops/kimi_k3/__init__.py`
- `python/sglang/srt/models/kimi_k3.py`
- `python/sglang/srt/environ.py`
- `test/registered/amd/test_kimi_k3_rmsnorm_gated_fp8.py`
- `test/registered/amd/test_kimi_k3_kda_fused_decode_aiter.py`

**Why it is separate from PR 2.** This is the only change on the branch that
alters numerics: it quantizes online a layer the K3 checkpoint ships in bf16.
It therefore carries an accuracy argument, and it will take the longest review.
Keeping it apart means it cannot hold up the layout fusion.

**Why the fused-decode test rides along.** `test_kimi_k3_kda_fused_decode_aiter.py`
tests a pre-existing kernel path rather than anything this PR adds, but it is
216 lines of pure test against code these reviewers are already reading. It is
not worth its own PR.

**Evidence for the description**

- Accuracy first: gsm8k n=400 scores **0.9850** with the FP8 o_proj and 0.9825
  without (SEM 0.66pp).
- Then performance: 0.7–1.7% output throughput and 1.4–1.8% ITL at ISL 8192 /
  OSL 1024 / TP 8 across concurrency 2…32.
- The parity argument: this is what ATOM's recipe does via
  `--online_quant_config ptpc_fp8`.
- The declines are deliberate and should be pointed at: an already-quantized
  weight, `all_reduce_fusion`, and `SGLANG_K3_GEMM_AR` all keep bf16, because
  their fused GEMM+comm kernels have no fp8 variant and would fall back on
  every call — paying the quant and getting none of the GEMM back.

---

## Not upstreamed

**The dual-stream MoE work** — `b2bc85cc4d` (split front + event pair) and
`d1b26e519f` (the five off-model probes). Excluded by request. The two knobs
default on but the fork itself stays off behind `SGLANG_ROCM_USE_MULTI_STREAM`,
and on gfx950 there is no token window in which enabling it wins (−2.0…−7.5%
output throughput). The probes exist only to explain that result, so they
travel with it whenever it goes.

**`K3_ROCM_OPT_REPORT.md`** — cites `/tmp/k3bench` paths and is largely the
record of the dual-stream negative result. Do not upstream it as a file. Mine
the relevant section into each PR description and leave the document
branch-local.

**`K3_ROCM_UPSTREAM_PLAN.md`** — this file. Branch-local too.
