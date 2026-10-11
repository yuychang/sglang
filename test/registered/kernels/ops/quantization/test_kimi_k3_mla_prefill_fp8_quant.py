"""Bit-equality of the batched MLA prefill FP8 quant against the per-tensor path.

The reference is ``AiterAttnBackend._quantize_mla_prefill_fp8`` (torch amax,
``/ 448`` and ``clamp_``, then AITER's static per-tensor quant), which is what
the gfx950 FlyDSL MLA prefill ran per Q/K/V operand before.
"""

import os

os.environ.setdefault("SGLANG_USE_AITER", "1")

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip()),
    reason="the reference quant is AITER's ROCm static per-tensor kernel",
)

HEADS = 12
QK_DIM = 192
V_DIM = 128
NOPE_DIM = 128


def _imports():
    from sglang.kernels.ops.quantization import fp8_kernel
    from sglang.kernels.ops.quantization.mla_prefill_fp8_quant import (
        mla_prefill_fp8_quant,
    )
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

    assert fp8_kernel._use_aiter, "reference must run AITER's static quant"
    return AiterAttnBackend, mla_prefill_fp8_quant


def _assert_bit_equal(tensors):
    backend_cls, fused = _imports()
    got = fused(tensors)
    assert got is not None
    assert len(got) == len(tensors)
    for i, x in enumerate(tensors):
        ref_q, ref_s = backend_cls._quantize_mla_prefill_fp8(x)
        new_q, new_s = got[i]
        assert new_q.dtype == ref_q.dtype and new_q.shape == ref_q.shape
        assert new_q.is_contiguous()
        assert new_s.dtype == torch.float32 and new_s.shape == (1,)
        assert torch.equal(new_s.view(torch.int32), ref_s.view(torch.int32)), (
            f"operand {i}: scale {new_s.item()!r} != {ref_s.item()!r}"
        )
        mismatch = (new_q.view(torch.uint8) != ref_q.view(torch.uint8)).sum().item()
        assert mismatch == 0, f"operand {i}: {mismatch} FP8 bytes differ"
    return got


def _qkv(num_q, num_kv, dtype=torch.bfloat16, scale=1.0, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((num_q, HEADS, QK_DIM), device="cuda", dtype=dtype, generator=gen)
    k = torch.randn((num_kv, HEADS, QK_DIM), device="cuda", dtype=dtype, generator=gen)
    # V is the trailing slice of the fused kv_b_proj output, as in forward_mha.
    kv = torch.randn(
        (num_kv, HEADS, NOPE_DIM + V_DIM), device="cuda", dtype=dtype, generator=gen
    )
    kv *= 0.1 * scale
    return [q * scale, k * (3 * scale), kv[..., NOPE_DIM:]]


@pytest.mark.parametrize("num_tokens", [1, 7, 512, 2048, 4096, 8192])
def test_extend_qkv(num_tokens):
    _assert_bit_equal(_qkv(num_tokens, num_tokens, seed=num_tokens))


@pytest.mark.parametrize("num_q,num_kv", [(2048, 8192), (8192, 32768), (512, 65536)])
def test_prefix_chunk_qkv(num_q, num_kv):
    tensors = _qkv(num_q, num_kv, seed=num_kv)
    assert not tensors[2].is_contiguous()
    _assert_bit_equal(tensors)


@pytest.mark.parametrize("num_tokens", [1, 333, 8192])
def test_q_only(num_tokens):
    _assert_bit_equal(_qkv(num_tokens, 1, seed=3)[:1])


def test_two_operands():
    _assert_bit_equal(_qkv(100, 300, seed=4)[:2])


def test_fp16_inputs():
    _assert_bit_equal(_qkv(1024, 2048, dtype=torch.float16, seed=5))


def test_random_magnitudes():
    for seed in range(20):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        exponent = torch.randint(-60, 60, (3,), generator=gen, device="cuda").tolist()
        tensors = _qkv(257 + seed, 911 + seed, seed=seed)
        for x, e in zip(tensors, exponent):
            x.mul_(2.0**e)
        assert not tensors[2].is_contiguous()
        _assert_bit_equal(tensors)


def test_all_zeros_and_negative_zeros():
    q = torch.zeros((64, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
    k = -torch.zeros((64, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros((64, HEADS, NOPE_DIM + V_DIM), device="cuda", dtype=torch.bfloat16)
    got = _assert_bit_equal([q, k, v[..., NOPE_DIM:]])
    assert got[0][1].item() == torch.tensor(1e-12, dtype=torch.float32).item()
    assert (got[1][0].view(torch.uint8) == 0x80).all()


def test_large_finite_values():
    finfo = torch.finfo(torch.bfloat16)
    tensors = _qkv(300, 300, seed=6)
    tensors[0] = tensors[0].clamp(-1, 1) * finfo.max
    tensors[1][17, 3, 100] = finfo.max
    tensors[1][18, 4, 101] = -finfo.max
    tensors[2].mul_(1e30)
    for x in tensors:
        assert torch.isfinite(x).all()
    _assert_bit_equal(tensors)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_values(value):
    tensors = _qkv(64, 64, seed=12)
    tensors[1][3, 4, 5] = value
    _assert_bit_equal(tensors)


def test_tiny_and_subnormal_values():
    tensors = _qkv(300, 300, seed=7)
    tensors[0].mul_(1e-38)
    tensors[1].mul_(1e-12)
    tensors[2].mul_(1e-40)
    _assert_bit_equal(tensors)


def test_single_outlier_anywhere():
    for position in [0, 4095, 4096, 123_457, 300 * HEADS * QK_DIM - 1]:
        q = torch.randn((300, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
        q.view(-1)[position] = -1234.0
        _assert_bit_equal([q])


def test_every_bf16_pattern_under_many_scales():
    bits = torch.arange(0, 1 << 16, dtype=torch.int32, device="cuda").to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    gen = torch.Generator(device="cpu").manual_seed(8)
    anchors = [1.0, 448.0, 1e-3, 1e-20, 3e5, 1e30, torch.finfo(torch.bfloat16).max]
    anchors += (10.0 ** (torch.rand(64, generator=gen) * 70 - 35)).tolist()
    for anchor in anchors:
        anchor = torch.tensor(anchor, dtype=torch.bfloat16).item()
        x = values[values.float().abs() <= anchor]
        x = torch.cat([x, torch.tensor([anchor], device="cuda", dtype=x.dtype)])
        x = torch.cat([x, x.new_zeros((-x.numel()) % QK_DIM)])
        _assert_bit_equal([x.view(-1, 1, QK_DIM)])


def test_strided_and_misaligned_views():
    # A head-sliced Q (stride 16 * 192) and an operand whose base pointer is
    # not 16-byte aligned exercise each kernel specialization; repeat calls
    # take the cached-launch path.
    padded = torch.randn((500, 16, QK_DIM), device="cuda", dtype=torch.bfloat16)
    q = padded[:, :HEADS, :]
    flat = torch.randn((1 + 500 * HEADS * QK_DIM,), device="cuda", dtype=torch.bfloat16)
    misaligned = flat[1:].view(500, HEADS, QK_DIM)
    assert misaligned.data_ptr() % 16 != 0
    aligned = flat[:-1].view(500, HEADS, QK_DIM)
    for _ in range(2):
        _assert_bit_equal([q, aligned])
        _assert_bit_equal([q, misaligned])
        _assert_bit_equal([misaligned, q, aligned])


def test_regular_triton_launch(monkeypatch):
    from sglang.kernels.ops.quantization import mla_prefill_fp8_quant as module

    monkeypatch.setattr(module, "_DIRECT_LAUNCH", False)
    monkeypatch.setattr(module, "_compiled", {})
    for _ in range(2):
        _assert_bit_equal(_qkv(700, 1300, seed=11))
    assert module._compiled == {}


def test_unsupported_inputs_decline():
    _, fused = _imports()
    x = torch.randn((8, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
    assert fused([x.float()]) is None
    assert fused([x.view(8, -1)]) is None
    assert fused([x[..., :100]]) is None
    assert fused([x.transpose(1, 2)]) is None
    assert fused([x[:0]]) is None
    assert fused([x, x.half()]) is None
    assert fused([x] * 4) is None


@pytest.mark.parametrize("fused_enabled", [False, True])
def test_backend_helper(fused_enabled):
    backend_cls, fused = _imports()
    backend = backend_cls.__new__(backend_cls)
    backend._mla_prefill_fused_fp8_quant = fused if fused_enabled else None
    tensors = _qkv(1000, 3000, seed=9)
    got = backend._quantize_mla_prefill_fp8_operands(*tensors)
    for x, (x8, scale) in zip(tensors, got):
        ref8, ref_scale = backend_cls._quantize_mla_prefill_fp8(x)
        assert torch.equal(scale.view(torch.int32), ref_scale.view(torch.int32))
        assert torch.equal(x8.view(torch.uint8), ref8.view(torch.uint8))
    # A shape the fused kernels decline falls back to the per-tensor path.
    odd = torch.randn((10, 3, 100), device="cuda", dtype=torch.bfloat16)
    ((odd8, odd_scale),) = backend._quantize_mla_prefill_fp8_operands(odd)
    ref8, ref_scale = backend_cls._quantize_mla_prefill_fp8(odd)
    assert torch.equal(odd_scale, ref_scale)
    assert torch.equal(odd8.view(torch.uint8), ref8.view(torch.uint8))


def _flydsl_fp8_available():
    try:
        from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
            flydsl_flash_attn_fp8_supported,
        )

        return flydsl_flash_attn_fp8_supported(
            torch.device("cuda"), HEADS, HEADS, QK_DIM, V_DIM
        )
    except Exception:
        return False


@pytest.mark.parametrize("causal", [True, False])
def test_flydsl_prefill_output_unchanged(causal):
    if not _flydsl_fp8_available():
        pytest.skip("FlyDSL FP8 varlen attention is unavailable")
    backend_cls, fused = _imports()
    seq_q = [300, 1, 1024, 77]
    seq_k = seq_q if causal else [1500, 129, 4096, 200]
    q, k, v = _qkv(sum(seq_q), sum(seq_k), seed=10)
    if causal:
        q, k, v = q, k[: q.shape[0]], v[: q.shape[0]]
    cu_q = torch.tensor([0] + seq_q, device="cuda").cumsum(0).to(torch.int32)
    cu_k = torch.tensor([0] + seq_k, device="cuda").cumsum(0).to(torch.int32)

    outputs = []
    for enabled in (False, True):
        backend = backend_cls.__new__(backend_cls)
        backend.use_mla_flydsl_fp8_prefill = True
        backend.require_mla_flydsl_fp8_prefill = True
        backend._mla_flydsl_unit_descale = None
        backend._mla_flydsl_fp8_fallback_logged = False
        backend._mla_prefill_fused_fp8_quant = fused if enabled else None
        out, lse = backend._mla_prefill_varlen_attn(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max(seq_q),
            max(seq_k),
            softmax_scale=QK_DIM**-0.5,
            causal=causal,
            return_lse=True,
        )[:2]
        outputs.append((out, lse))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
