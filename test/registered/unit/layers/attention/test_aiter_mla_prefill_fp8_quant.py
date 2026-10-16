"""AiterAttnBackend's MLA prefill FP8 operand quant: the batched kernel and the
per-tensor path give the same operands and the same FlyDSL prefill output."""

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip()),
    reason="the reference quant is AITER's ROCm static per-tensor kernel",
)

HEADS = 12
QK_DIM = 192
V_DIM = 128
NOPE_DIM = 128


def _imports():
    # fp8_kernel and aiter_backend read SGLANG_USE_AITER once, at import.
    with envs.SGLANG_USE_AITER.override(True):
        from sglang.kernels.ops.quantization.mla_prefill_fp8_quant import (
            mla_prefill_fp8_quant,
        )
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

    return AiterAttnBackend, mla_prefill_fp8_quant


def _qkv(num_q, num_kv, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((num_q, HEADS, QK_DIM), device="cuda", dtype=dtype, generator=gen)
    k = torch.randn((num_kv, HEADS, QK_DIM), device="cuda", dtype=dtype, generator=gen)
    # V is the trailing slice of the fused kv_b_proj output, as in forward_mha.
    kv = torch.randn(
        (num_kv, HEADS, NOPE_DIM + V_DIM), device="cuda", dtype=dtype, generator=gen
    )
    kv *= 0.1
    return [q, k * 3, kv[..., NOPE_DIM:]]


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
