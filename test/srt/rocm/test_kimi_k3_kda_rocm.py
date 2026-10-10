"""Kimi-K3 KDA on ROCm: the vLLM-ported HIP kernels against references.

Covers ``sglang.kernels.ops.attention.kda_rocm``:

* ``fused_kda_decode`` (conv1d update + gated delta rule + gated sigmoid
  RMSNorm in one launch) against a pure-torch reference, against vLLM's Triton
  fallback chain (``causal_conv1d_update`` + ``fused_recurrent_kda_packed_decode``
  + gated norm) and against SGLang's previous Triton decode
  (``fused_sigmoid_gating_delta_rule_update``);
* ``chunk_kda_prefill`` with the fused HIP chunk kernel and with the Triton
  fallback, against a pure-torch reference and SGLang's previous ``chunk_kda``;
* chunked prefill carrying state through the paged cache, and prefill followed
  by fused decode steps.

Run: ``pytest -q test/srt/rocm/test_kimi_k3_kda_rocm.py``
"""

import pytest
import torch
import torch.nn.functional as F

if not (torch.cuda.is_available() and torch.version.hip is not None):
    pytest.skip("ROCm only", allow_module_level=True)

from sglang.kernels.ops.attention.kda_rocm import jit
from sglang.kernels.ops.attention.kda_rocm.kda_chunk import (
    is_fused_kda_chunk_supported,
)
from sglang.kernels.ops.attention.kda_rocm.kda_decode import (
    fused_kda_decode,
    is_fused_kda_decode_supported,
    stage_decode_conv1d_weight,
    stage_decode_norm_weight,
)
from sglang.kernels.ops.attention.kda_rocm.kda_prefill import chunk_kda_prefill

HEAD_DIM = 128
CONV_WIDTH = 4
LOWER_BOUND = -5.0
NORM_EPS = 1e-5
DTYPE = torch.bfloat16
DEV = "cuda"

needs_decode = pytest.mark.skipif(
    not is_fused_kda_decode_supported(96, HEAD_DIM, CONV_WIDTH, 0, DTYPE, DTYPE),
    reason="fused KDA decode is built for gfx942 / gfx950 only",
)
needs_chunk = pytest.mark.skipif(
    not is_fused_kda_chunk_supported(),
    reason="fused KDA chunk is built for gfx950 only",
)


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, e = actual.float(), expected.float()
    return ((a - e).norm() / e.norm().clamp_min(1e-12)).item()


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------


def _ref_gate(raw_g, A_log, dt_bias, num_heads):
    g = raw_g.float() + dt_bias.float().view(num_heads, HEAD_DIM)
    return LOWER_BOUND * torch.sigmoid(A_log.float().view(num_heads, 1).exp() * g)


def _ref_recurrence(q, k, v, raw_g, raw_beta, A_log, dt_bias, h0):
    """One sequence. q/k/v/raw_g: [T, H, D]; raw_beta: [T, H]; h0: [H, V, K]."""
    num_heads = q.shape[1]
    scale = HEAD_DIM**-0.5
    q = q.float()
    k = k.float()
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    v = v.float()
    gate = _ref_gate(raw_g, A_log, dt_bias, num_heads).exp()
    beta = torch.sigmoid(raw_beta.float())
    state = h0.float().clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        state = state * gate[t].unsqueeze(1)
        v_t = (v[t] - torch.einsum("hvk,hk->hv", state, k[t])) * beta[t, :, None]
        state = state + v_t.unsqueeze(-1) * k[t].unsqueeze(1)
        out[t] = torch.einsum("hvk,hk->hv", state, q[t])
    return out, state


def _ref_conv(x, conv_weight, prefix):
    """x: [T, C]; conv_weight: [C, W]; prefix: [W-1, C] (raw inputs)."""
    xs = torch.cat([prefix.float(), x.float()], 0)
    t = x.shape[0]
    acc = torch.zeros(t, x.shape[1], device=x.device, dtype=torch.float32)
    for w in range(CONV_WIDTH):
        acc += xs[w : w + t] * conv_weight[:, w].float().unsqueeze(0)
    return F.silu(acc), xs[-(CONV_WIDTH - 1) :]


def _gated_rmsnorm(x, gate, weight):
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + NORM_EPS)
    return normed * weight.float() * torch.sigmoid(gate.float())


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


class DecodeInputs:
    """One decode step laid out the way SGLang's KDA layer holds it."""

    def __init__(self, num_tokens, num_heads, num_slots, seed=0):
        torch.manual_seed(seed)
        dim = num_heads * HEAD_DIM
        self.num_heads, self.dim = num_heads, dim
        self.mixed_qkv = torch.randn(num_tokens, 3 * dim, device=DEV, dtype=DTYPE) * 0.5
        # SGLang keeps the conv1d weight as [3 * dim, 1, width] fp32.
        self.conv_weight = (
            torch.randn(3 * dim, 1, CONV_WIDTH, device=DEV, dtype=torch.float32) * 0.3
        )
        # SGLang's conv pool is [slots, width - 1, 3 * dim]; layers pass the
        # transposed [slots, 3 * dim, width - 1] view.
        self.conv_pool = (
            torch.randn(num_slots, CONV_WIDTH - 1, 3 * dim, device=DEV, dtype=DTYPE)
            * 0.5
        )
        self.ssm_pool = (
            torch.randn(
                num_slots, num_heads, HEAD_DIM, HEAD_DIM, device=DEV, dtype=torch.float32
            )
            * 0.1
        )
        self.raw_g = (
            torch.randn(1, num_tokens, num_heads, HEAD_DIM, device=DEV, dtype=DTYPE)
            * 0.5
        )
        self.gate = (
            torch.randn(num_tokens, num_heads, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.5
        )
        self.raw_beta = torch.randn(1, num_tokens, num_heads, device=DEV, dtype=DTYPE)
        self.A_log = torch.randn(num_heads, device=DEV, dtype=torch.float32) * 0.5
        self.dt_bias = torch.randn(dim, device=DEV, dtype=torch.float32) * 0.1
        self.norm_weight = 1 + 0.1 * torch.randn(HEAD_DIM, device=DEV, dtype=DTYPE)
        # SGLang reserves slot 0; live requests use slots >= 1.
        assert num_slots > num_tokens + 1
        self.state_indices = (
            torch.randperm(num_slots - 1, device=DEV)[:num_tokens] + 1
        ).to(torch.int32)


def _run_fused_decode(inp: DecodeInputs, with_norm=True):
    conv_pool = inp.conv_pool.clone()
    ssm_pool = inp.ssm_pool.clone()
    out = torch.empty(
        1, inp.mixed_qkv.shape[0], inp.num_heads, HEAD_DIM, device=DEV, dtype=DTYPE
    )
    fused_kda_decode(
        x=inp.mixed_qkv,
        weight=stage_decode_conv1d_weight(inp.conv_weight),
        bias=None,
        conv_state=conv_pool.transpose(-1, -2),
        raw_g=inp.raw_g,
        raw_beta=inp.raw_beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        state_indices=inp.state_indices,
        state=ssm_pool,
        out=out,
        lower_bound=LOWER_BOUND,
        output_gate=inp.gate if with_norm else None,
        norm_weight=stage_decode_norm_weight(inp.norm_weight) if with_norm else None,
        norm_eps=NORM_EPS,
    )
    return out, conv_pool, ssm_pool


def _run_reference_decode(inp: DecodeInputs, with_norm=True):
    conv_pool = inp.conv_pool.clone()
    ssm_pool = inp.ssm_pool.clone()
    num_tokens = inp.mixed_qkv.shape[0]
    out = torch.zeros(1, num_tokens, inp.num_heads, HEAD_DIM, device=DEV)
    w = inp.conv_weight.squeeze(1)
    for i, slot in enumerate(inp.state_indices.tolist()):
        if slot <= 0:
            continue
        conv, new_prefix = _ref_conv(inp.mixed_qkv[i : i + 1], w, conv_pool[slot])
        conv_pool[slot] = new_prefix.to(DTYPE)
        q, k, v = conv.to(DTYPE).view(1, 3, inp.num_heads, HEAD_DIM).unbind(1)
        o, state = _ref_recurrence(
            q,
            k,
            v,
            inp.raw_g[0, i : i + 1],
            inp.raw_beta[0, i : i + 1],
            inp.A_log,
            inp.dt_bias,
            ssm_pool[slot],
        )
        ssm_pool[slot] = state
        out[0, i] = (
            _gated_rmsnorm(o[0], inp.gate[i], inp.norm_weight) if with_norm else o[0]
        )
    return out, conv_pool, ssm_pool


def _run_vllm_triton_chain(inp: DecodeInputs):
    """The fallback vLLM's AMD layer runs when the fused decode is unavailable."""
    from sglang.kernels.ops.attention.kda_rocm.third_party.kda import (
        fused_recurrent_kda_packed_decode,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    conv_pool = inp.conv_pool.clone()
    ssm_pool = inp.ssm_pool.clone()
    conv_out = causal_conv1d_update(
        inp.mixed_qkv.clone(),
        conv_pool.transpose(-1, -2),
        inp.conv_weight.squeeze(1),
        None,
        activation="silu",
        conv_state_indices=inp.state_indices,
    )
    core, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=conv_out,
        raw_g=inp.raw_g,
        raw_beta=inp.raw_beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=ssm_pool,
        state_indices=inp.state_indices,
    )
    out = _gated_rmsnorm(core, inp.gate, inp.norm_weight).to(DTYPE)
    return out, conv_pool, ssm_pool


def _run_previous_sglang_decode(inp: DecodeInputs):
    """SGLang's KDA decode before the port (TritonKDAKernel.decode)."""
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    conv_pool = inp.conv_pool.clone()
    ssm_pool = inp.ssm_pool.clone()
    num_tokens = inp.mixed_qkv.shape[0]
    conv_out = causal_conv1d_update(
        inp.mixed_qkv.clone(),
        conv_pool.transpose(-1, -2),
        inp.conv_weight.squeeze(1),
        None,
        activation="silu",
        conv_state_indices=inp.state_indices,
    )
    q, k, v = (
        t.unflatten(-1, (inp.num_heads, HEAD_DIM)).unsqueeze(0)
        for t in conv_out.split(inp.dim, dim=-1)
    )
    core = fused_sigmoid_gating_delta_rule_update(
        A_log=inp.A_log.view(1, 1, inp.num_heads, 1),
        a=inp.raw_g[0].reshape(num_tokens, inp.dim),
        dt_bias=inp.dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=inp.raw_beta[0],
        initial_state_source=ssm_pool,
        initial_state_indices=inp.state_indices,
        cu_seqlens=torch.arange(num_tokens + 1, device=DEV, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=LOWER_BOUND,
    )
    out = _gated_rmsnorm(core.view(1, num_tokens, inp.num_heads, HEAD_DIM), inp.gate, inp.norm_weight)
    return out.to(DTYPE), conv_pool, ssm_pool


@needs_decode
@pytest.mark.parametrize("num_heads", [12, 24, 96])
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@torch.inference_mode()
def test_fused_decode_matches_torch_reference(num_heads, num_tokens):
    inp = DecodeInputs(num_tokens, num_heads, num_slots=num_tokens + 4)
    out, conv, state = _run_fused_decode(inp)
    ref_out, ref_conv, ref_state = _run_reference_decode(inp)
    torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
    assert _rel_err(state, ref_state) < 2e-3
    assert _rel_err(out, ref_out) < 1e-2
    torch.testing.assert_close(out.float(), ref_out, atol=3e-2, rtol=3e-2)


@needs_decode
@pytest.mark.parametrize("num_heads", [12, 96])
@pytest.mark.parametrize("num_tokens", [1, 33])
@torch.inference_mode()
def test_fused_decode_matches_vllm_triton_chain(num_heads, num_tokens):
    inp = DecodeInputs(num_tokens, num_heads, num_slots=num_tokens + 4, seed=1)
    out, conv, state = _run_fused_decode(inp)
    ref_out, ref_conv, ref_state = _run_vllm_triton_chain(inp)
    torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
    torch.testing.assert_close(state, ref_state, atol=2e-3, rtol=2e-3)


@needs_decode
@pytest.mark.parametrize("num_tokens", [1, 16])
@torch.inference_mode()
def test_fused_decode_matches_previous_sglang_decode(num_tokens):
    inp = DecodeInputs(num_tokens, 12, num_slots=num_tokens + 4, seed=2)
    out, conv, state = _run_fused_decode(inp)
    ref_out, ref_conv, ref_state = _run_previous_sglang_decode(inp)
    torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
    torch.testing.assert_close(state, ref_state, atol=2e-3, rtol=2e-3)


@needs_decode
@torch.inference_mode()
def test_fused_decode_without_norm_returns_raw_recurrence():
    inp = DecodeInputs(5, 12, num_slots=9, seed=3)
    out, _, state = _run_fused_decode(inp, with_norm=False)
    ref_out, _, ref_state = _run_reference_decode(inp, with_norm=False)
    assert _rel_err(state, ref_state) < 2e-3
    assert _rel_err(out, ref_out) < 1e-2


@needs_decode
@pytest.mark.parametrize("pad_value", [-1, 0])
@torch.inference_mode()
def test_fused_decode_skips_padded_rows(pad_value):
    """CUDA-graph padding in SGLang uses -1; slot 0 is SGLang's reserved slot."""
    num_real, num_pad = 3, 5
    inp = DecodeInputs(num_real + num_pad, 12, num_slots=12, seed=4)
    inp.state_indices[num_real:] = pad_value
    out, conv, state = _run_fused_decode(inp)
    ref_out, ref_conv, ref_state = _run_reference_decode(inp)
    assert not out[0, num_real:].any(), "padded rows must produce zeros"
    live = inp.state_indices[:num_real].long()
    untouched = torch.ones(12, dtype=torch.bool, device=DEV)
    untouched[live] = False
    torch.testing.assert_close(conv[untouched], inp.conv_pool[untouched], atol=0, rtol=0)
    torch.testing.assert_close(state[untouched], inp.ssm_pool[untouched], atol=0, rtol=0)
    torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
    assert _rel_err(state[live], ref_state[live]) < 2e-3
    assert _rel_err(out[0, :num_real], ref_out[0, :num_real]) < 1e-2


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------

NUM_HEADS = 12


def _prefill_inputs(seqlens, seed, strided=False, served_bias=False):
    torch.manual_seed(seed)
    total = sum(seqlens)
    lp = NUM_HEADS * HEAD_DIM
    cu = torch.tensor([0] + torch.tensor(seqlens).cumsum(0).tolist(), device=DEV, dtype=torch.int32)
    shape = (1, total, NUM_HEADS, HEAD_DIM)
    if strided:
        # q/k/v as bands of the packed conv output and beta as a slice of the
        # projection, the way SGLang's layer produces them.
        packed = torch.randn(total, 3 * lp, device=DEV, dtype=DTYPE) * 0.5
        q, k, v = (
            b.unflatten(-1, (NUM_HEADS, HEAD_DIM)).unsqueeze(0)
            for b in packed.split(lp, dim=-1)
        )
        proj = torch.randn(total, lp + NUM_HEADS + 16, device=DEV, dtype=DTYPE)
        raw_beta = proj[:, lp : lp + NUM_HEADS].unsqueeze(0)
        if total > 1:
            assert not raw_beta.is_contiguous() and not q.is_contiguous()
    else:
        q = torch.randn(shape, device=DEV, dtype=DTYPE) * 0.5
        k = torch.randn(shape, device=DEV, dtype=DTYPE) * 0.5
        v = torch.randn(shape, device=DEV, dtype=DTYPE) * 0.5
        raw_beta = torch.randn(1, total, NUM_HEADS, device=DEV, dtype=DTYPE)
    return dict(
        q=q,
        k=k,
        v=v,
        raw_g=torch.randn(shape, device=DEV, dtype=DTYPE),
        raw_beta=raw_beta,
        A_log=torch.randn(NUM_HEADS, device=DEV, dtype=torch.float32) * 0.5,
        # vLLM's kernel tests draw dt_bias from randn; the served checkpoint's
        # sits in [-7.8, -1.4], where bf16 rounding inside a chunk shows more.
        dt_bias=(
            torch.rand(lp, device=DEV, dtype=torch.float32) * 6.4 - 7.8
            if served_bias
            else torch.randn(lp, device=DEV, dtype=torch.float32) * 0.5
        ),
        h0=torch.randn(len(seqlens), NUM_HEADS, HEAD_DIM, HEAD_DIM, device=DEV) * 0.1,
        cu=cu,
    )


def _paged(inp, warm, rows, slots):
    cache = torch.full((slots, NUM_HEADS, HEAD_DIM, HEAD_DIM), -7.0, device=DEV)
    for n, row in enumerate(rows):
        if warm[n]:
            cache[row] = inp["h0"][n]
    idx = torch.tensor(rows, device=DEV, dtype=torch.int32)
    return cache, idx, torch.tensor(warm, device=DEV, dtype=torch.bool)


def _run_prefill(inp, use_fused, cache, idx, warm, **kw):
    return chunk_kda_prefill(
        q=inp["q"].clone(),
        k=inp["k"].clone(),
        v=inp["v"].clone(),
        raw_g=inp["raw_g"],
        raw_beta=inp["raw_beta"],
        A_log=inp["A_log"],
        g_bias=inp["dt_bias"],
        lower_bound=LOWER_BOUND,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu"],
        use_fused_chunk=use_fused,
        state_cache=cache,
        state_indices=idx,
        has_initial_state=warm,
        **kw,
    )[0]


def _reference_prefill(inp, warm):
    outs, states = [], []
    cu = inp["cu"].tolist()
    for n in range(len(cu) - 1):
        b, e = cu[n], cu[n + 1]
        h0 = inp["h0"][n] if warm[n] else torch.zeros_like(inp["h0"][n])
        o, s = _ref_recurrence(
            inp["q"][0, b:e],
            inp["k"][0, b:e],
            inp["v"][0, b:e],
            inp["raw_g"][0, b:e],
            inp["raw_beta"][0, b:e],
            inp["A_log"],
            inp["dt_bias"],
            h0,
        )
        outs.append(o)
        states.append(s)
    return torch.cat(outs, 0).unsqueeze(0), torch.stack(states)


PREFILL_CASES = [[1], [64], [130], [513, 64, 1, 200]]


@pytest.mark.parametrize(
    "use_fused", [pytest.param(True, marks=needs_chunk), False]
)
@pytest.mark.parametrize("seqlens", PREFILL_CASES)
@torch.inference_mode()
def test_prefill_matches_torch_reference(use_fused, seqlens):
    inp = _prefill_inputs(seqlens, seed=len(seqlens), served_bias=True)
    n = len(seqlens)
    warm = [i % 2 == 0 for i in range(n)]
    rows = [2 * i + 1 for i in range(n)]
    cache, idx, warm_t = _paged(inp, warm, rows, slots=2 * n + 3)
    before = cache.clone()
    out = _run_prefill(inp, use_fused, cache, idx, warm_t)
    ref_out, ref_state = _reference_prefill(inp, warm)
    assert _rel_err(out, ref_out) < 1e-2
    assert _rel_err(cache[idx.long()], ref_state) < 5e-3
    untouched = torch.ones(cache.shape[0], dtype=torch.bool, device=DEV)
    untouched[idx.long()] = False
    assert torch.equal(cache[untouched], before[untouched])


@needs_chunk
@pytest.mark.parametrize("seqlens", [[1], [64], [1024], [513, 64, 1, 1200], [512] * 6])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_fused_prefill_matches_triton_fallback(seqlens, strided):
    inp = _prefill_inputs(seqlens, seed=7 + len(seqlens), strided=strided)
    n = len(seqlens)
    rows = [3 * i + 2 for i in range(n)]
    warm = [True] * n
    cache_f, idx, warm_t = _paged(inp, warm, rows, slots=3 * n + 4)
    cache_t = cache_f.clone()
    out_f = _run_prefill(inp, True, cache_f, idx, warm_t)
    out_t = _run_prefill(inp, False, cache_t, idx, warm_t)
    torch.testing.assert_close(out_f.float(), out_t.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cache_f, cache_t, rtol=1e-3, atol=1e-3)


@needs_chunk
@pytest.mark.parametrize("seqlens", [[1024], [513, 64, 1, 1200]])
@torch.inference_mode()
def test_fused_prefill_matches_triton_fallback_served_gate_bias(seqlens):
    inp = _prefill_inputs(seqlens, seed=3, served_bias=True)
    n = len(seqlens)
    rows = [3 * i + 2 for i in range(n)]
    cache_f, idx, warm_t = _paged(inp, [True] * n, rows, slots=3 * n + 4)
    cache_t = cache_f.clone()
    out_f = _run_prefill(inp, True, cache_f, idx, warm_t)
    out_t = _run_prefill(inp, False, cache_t, idx, warm_t)
    torch.testing.assert_close(out_f.float(), out_t.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cache_f, cache_t, rtol=1e-2, atol=1e-2)
    assert _rel_err(cache_f[idx.long()], cache_t[idx.long()]) < 5e-3


@pytest.mark.parametrize(
    "use_fused", [pytest.param(True, marks=needs_chunk), False]
)
@pytest.mark.parametrize("seqlens", [[64], [513, 64, 1, 300]])
@torch.inference_mode()
def test_prefill_matches_previous_sglang_chunk_kda(use_fused, seqlens):
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    inp = _prefill_inputs(seqlens, seed=19)
    n = len(seqlens)
    rows = [2 * i + 1 for i in range(n)]
    cache, idx, warm_t = _paged(inp, [True] * n, rows, slots=2 * n + 3)
    prev_cache = cache.clone()
    out = _run_prefill(inp, use_fused, cache, idx, warm_t)

    res = chunk_kda(
        q=inp["q"].clone(),
        k=inp["k"].clone(),
        v=inp["v"].clone(),
        g=inp["raw_g"].clone(),
        beta=inp["raw_beta"].clone(),
        initial_state=prev_cache,
        initial_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu"],
        A_log=inp["A_log"].view(1, 1, NUM_HEADS, 1),
        dt_bias=inp["dt_bias"],
        lower_bound=LOWER_BOUND,
        beta_is_raw=True,
    )
    prev_out = res[0] if isinstance(res, tuple) else res
    torch.testing.assert_close(out.float(), prev_out.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        cache[idx.long()], prev_cache[idx.long()], rtol=1e-3, atol=1e-3
    )


def _slice_tokens(inp, b, e, h0=None):
    out = dict(inp)
    for key in ("q", "k", "v", "raw_g", "raw_beta"):
        out[key] = inp[key][:, b:e].contiguous()
    out["cu"] = torch.tensor([0, e - b], device=DEV, dtype=torch.int32)
    if h0 is not None:
        out["h0"] = h0
    return out


@pytest.mark.parametrize(
    "use_fused", [pytest.param(True, marks=needs_chunk), False]
)
@pytest.mark.parametrize("split", [64, 300, 511])
@torch.inference_mode()
def test_chunked_prefill_carries_state_through_cache(use_fused, split):
    """Two prefill chunks via the paged cache equal one prefill of the whole."""
    total = 700
    inp = _prefill_inputs([total], seed=split, served_bias=True)
    row = 5
    whole_cache, idx, warm = _paged(inp, [True], [row], slots=8)
    whole_out = _run_prefill(inp, use_fused, whole_cache, idx, warm)

    cache, _, _ = _paged(inp, [True], [row], slots=8)
    first = _slice_tokens(inp, 0, split)
    out1 = _run_prefill(first, use_fused, cache, idx, warm)
    second = _slice_tokens(inp, split, total)
    out2 = _run_prefill(second, use_fused, cache, idx, warm)

    # A split off the 64-token grid re-chunks the tail, so the two runs round
    # differently in bf16; each must still match the fp32 recurrence.
    ref_out, ref_state = _reference_prefill(inp, [True])
    split_out = torch.cat([out1, out2], 1)
    assert _rel_err(split_out, ref_out) < 1e-2
    assert _rel_err(whole_out, ref_out) < 1e-2
    assert _rel_err(cache[row], ref_state[0]) < 5e-3
    assert _rel_err(whole_cache[row], ref_state[0]) < 5e-3
    torch.testing.assert_close(split_out.float(), whole_out.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cache[row], whole_cache[row], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "use_fused", [pytest.param(True, marks=needs_chunk), False]
)
@torch.inference_mode()
def test_mixed_batch_of_cold_and_continuing_chunks(use_fused):
    """One batch with a fresh request and a request on its second chunk."""
    inp = _prefill_inputs([400, 260], seed=31, served_bias=True)
    rows = [3, 6]
    # The continuing request (seq 1) already prefilled 260 tokens of context.
    ctx = _prefill_inputs([260], seed=32)
    cache, _, _ = _paged(inp, [False, False], rows, slots=8)
    cache[rows[1]] = 0.0
    ctx_idx = torch.tensor([rows[1]], device=DEV, dtype=torch.int32)
    ctx["A_log"], ctx["dt_bias"] = inp["A_log"], inp["dt_bias"]
    _run_prefill(ctx, use_fused, cache, ctx_idx, torch.tensor([False], device=DEV))
    carried = cache[rows[1]].clone()

    idx = torch.tensor(rows, device=DEV, dtype=torch.int32)
    warm = torch.tensor([False, True], device=DEV)
    out = _run_prefill(inp, use_fused, cache, idx, warm)

    ref_inp = dict(inp)
    ref_inp["h0"] = torch.stack([torch.zeros_like(carried), carried])
    ref_out, ref_state = _reference_prefill(ref_inp, [True, True])
    assert _rel_err(out, ref_out) < 1e-2
    assert _rel_err(cache[idx.long()], ref_state) < 5e-3


@needs_decode
@pytest.mark.parametrize(
    "use_fused", [pytest.param(True, marks=needs_chunk), False]
)
@torch.inference_mode()
def test_prefill_then_fused_decode_continues_the_sequence(use_fused):
    """conv + prefill, then fused decode steps, against one torch pass."""
    torch.manual_seed(41)
    prompt, steps = 150, 4
    total = prompt + steps
    lp = NUM_HEADS * HEAD_DIM
    x = torch.randn(total, 3 * lp, device=DEV, dtype=DTYPE) * 0.5
    conv_weight = torch.randn(3 * lp, 1, CONV_WIDTH, device=DEV) * 0.3
    raw_g = torch.randn(1, total, NUM_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE)
    raw_beta = torch.randn(1, total, NUM_HEADS, device=DEV, dtype=DTYPE)
    A_log = torch.randn(NUM_HEADS, device=DEV) * 0.5
    dt_bias = torch.rand(lp, device=DEV) * 6.4 - 7.8

    zeros_prefix = torch.zeros(CONV_WIDTH - 1, 3 * lp, device=DEV)
    conv_all, _ = _ref_conv(x, conv_weight.squeeze(1), zeros_prefix)
    q, k, v = conv_all.to(DTYPE).view(total, 3, NUM_HEADS, HEAD_DIM).unbind(1)
    ref_out, ref_state = _ref_recurrence(
        q, k, v, raw_g[0], raw_beta[0], A_log, dt_bias,
        torch.zeros(NUM_HEADS, HEAD_DIM, HEAD_DIM, device=DEV),
    )

    slot = 2
    ssm_pool = torch.full((4, NUM_HEADS, HEAD_DIM, HEAD_DIM), -7.0, device=DEV)
    conv_pool = torch.zeros(4, CONV_WIDTH - 1, 3 * lp, device=DEV, dtype=DTYPE)
    conv_p, prefix = _ref_conv(x[:prompt], conv_weight.squeeze(1), zeros_prefix)
    conv_pool[slot] = prefix.to(DTYPE)
    qp, kp, vp = (
        t.unsqueeze(0)
        for t in conv_p.to(DTYPE).view(prompt, 3, NUM_HEADS, HEAD_DIM).unbind(1)
    )
    idx = torch.tensor([slot], device=DEV, dtype=torch.int32)
    pre_out = chunk_kda_prefill(
        q=qp, k=kp, v=vp,
        raw_g=raw_g[:, :prompt], raw_beta=raw_beta[:, :prompt],
        A_log=A_log, g_bias=dt_bias, lower_bound=LOWER_BOUND,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0, prompt], device=DEV, dtype=torch.int32),
        use_fused_chunk=use_fused,
        state_cache=ssm_pool, state_indices=idx,
        has_initial_state=torch.tensor([False], device=DEV),
    )[0]
    assert _rel_err(pre_out[0], ref_out[:prompt]) < 1e-2

    staged_w = stage_decode_conv1d_weight(conv_weight)
    for t in range(prompt, total):
        out = torch.empty(1, 1, NUM_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE)
        fused_kda_decode(
            x=x[t : t + 1], weight=staged_w, bias=None,
            conv_state=conv_pool.transpose(-1, -2),
            raw_g=raw_g[:, t : t + 1].contiguous(),
            raw_beta=raw_beta[:, t : t + 1].contiguous(),
            A_log=A_log, dt_bias=dt_bias, state_indices=idx, state=ssm_pool,
            out=out, lower_bound=LOWER_BOUND,
        )
        assert _rel_err(out[0, 0], ref_out[t]) < 1e-2, f"decode step {t - prompt}"
    assert _rel_err(ssm_pool[slot], ref_state) < 5e-3


def test_jit_build_registers_ops():
    ops = jit.load_ops()
    assert ops is not None
    assert jit.has_op("fused_kda_decode") == (jit.device_arch() in ("gfx942", "gfx950"))
    assert jit.has_op("fused_kda_chunk") == (jit.device_arch() == "gfx950")
