"""Unit tests for FlyDSL FP8 MLA prefill operand routing."""

import unittest

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend, fp8_dtype
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAiterFlyDSLPrefill(unittest.TestCase):
    def setUp(self):
        self.backend = AiterAttnBackend.__new__(AiterAttnBackend)
        self.backend.use_mla_flydsl_fp8_prefill = True
        self.q = torch.empty((8, 12, 192), dtype=torch.bfloat16)

    def test_accepts_bf16_operands(self):
        k = torch.empty((16, 12, 192), dtype=torch.bfloat16)
        v = torch.empty((16, 12, 128), dtype=torch.bfloat16)
        self.assertTrue(
            self.backend._mla_flydsl_fp8_prefill_applicable(self.q, k, v)
        )

    def test_accepts_direct_fp8_kv(self):
        k = torch.empty((16, 12, 192), dtype=fp8_dtype)
        v = torch.empty((16, 12, 128), dtype=fp8_dtype)
        self.assertTrue(
            self.backend._mla_flydsl_fp8_prefill_applicable(self.q, k, v)
        )

    def test_rejects_mixed_kv_dtypes(self):
        k = torch.empty((16, 12, 192), dtype=fp8_dtype)
        v = torch.empty((16, 12, 128), dtype=torch.bfloat16)
        self.assertFalse(
            self.backend._mla_flydsl_fp8_prefill_applicable(self.q, k, v)
        )


if __name__ == "__main__":
    unittest.main()
