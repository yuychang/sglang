"""Routing of the gfx950 FlyDSL FP8 MLA prefill back to bf16 FMHA when FlyDSL declines."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend, fp8_dtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEADS = 12
QK_DIM = 192
V_DIM = 128


def _fake_quant(x):
    return x.to(fp8_dtype), torch.ones(1)


def _backend(flydsl_fn):
    backend = AiterAttnBackend.__new__(AiterAttnBackend)
    backend.use_mla_flydsl_fp8_prefill = True
    backend._mla_flydsl_varlen_attn = flydsl_fn
    backend._mla_flydsl_unit_descale = torch.ones(1)
    backend._mla_flydsl_fallback_logged = False
    # The real quant runs AITER's GPU kernel.
    backend._quantize_mla_prefill_fp8 = _fake_quant
    return backend


def _run(backend, q, k, v):
    cu = torch.tensor([0, q.shape[0]], dtype=torch.int32)
    return backend._mla_prefill_varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        q.shape[0],
        k.shape[0],
        softmax_scale=QK_DIM**-0.5,
        causal=True,
    )


class TestFlydslMlaPrefillFallback(CustomTestCase):
    def setUp(self):
        self.bf16_fmha = mock.MagicMock(return_value="bf16")
        patcher = mock.patch.object(
            aiter_backend, "flash_attn_varlen_func", self.bf16_fmha, create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_declined_fp8_kv_reaches_bf16_fmha_as_bf16(self):
        flydsl = mock.MagicMock(return_value=None)
        q = torch.zeros((8, HEADS, QK_DIM), dtype=torch.bfloat16)
        k = torch.zeros((8, HEADS, QK_DIM), dtype=fp8_dtype)
        v = torch.zeros((8, HEADS, V_DIM), dtype=fp8_dtype)

        self.assertEqual(_run(_backend(flydsl), q, k, v), "bf16")

        flydsl.assert_called_once()
        _, fmha_k, fmha_v = self.bf16_fmha.call_args.args
        self.assertEqual(fmha_k.dtype, torch.bfloat16)
        self.assertEqual(fmha_v.dtype, torch.bfloat16)

    def test_int32_overflowing_operand_skips_flydsl(self):
        flydsl = mock.MagicMock()
        num_tokens = 2**31 // (HEADS * QK_DIM) + 1
        q = torch.empty(
            (num_tokens, HEADS, QK_DIM), dtype=torch.bfloat16, device="meta"
        )
        v = torch.empty((num_tokens, HEADS, V_DIM), dtype=torch.bfloat16, device="meta")

        self.assertEqual(_run(_backend(flydsl), q, q, v), "bf16")

        flydsl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
