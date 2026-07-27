"""Unit tests for BF16 MLA-absorb weight prescaling and output layout."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    _bmm_mla_absorb_bf16_to_contiguous,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _prescale_mla_absorb_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMlaAbsorbBf16(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.num_heads = 4
        self.num_tokens = 7
        self.kv_lora_rank = 12
        self.qk_nope_dim = 10
        self.v_dim = 8

    def test_prescale_weights_matches_per_step_dequant(self):
        w_kc = torch.randn(
            self.num_heads,
            self.kv_lora_rank,
            self.qk_nope_dim,
            dtype=torch.bfloat16,
        )
        w_vc = torch.randn(
            self.num_heads,
            self.kv_lora_rank,
            self.v_dim,
            dtype=torch.bfloat16,
        )
        scale = torch.tensor(1.7)
        attn = SimpleNamespace(
            w_kc=w_kc.clone(),
            w_vc=w_vc.clone(),
            w_scale=scale,
            mla_absorb_weights_prescaled=False,
        )

        expected_kc = w_kc.to(torch.bfloat16) * scale
        expected_vc = w_vc.to(torch.bfloat16) * scale
        _prescale_mla_absorb_weights(attn)

        self.assertTrue(attn.mla_absorb_weights_prescaled)
        self.assertTrue(torch.equal(attn.w_kc, expected_kc))
        self.assertTrue(torch.equal(attn.w_vc, expected_vc))
        # Keep the scale available so repeated weight loading can prescale a
        # freshly loaded raw tensor again instead of relying on a None sentinel.
        self.assertIs(attn.w_scale, scale)

    def test_v_absorb_bmm_matches_reference_and_returns_contiguous_storage(self):
        attn_output = torch.randn(
            self.num_tokens,
            self.num_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
        )
        w_vc = torch.randn(
            self.num_heads,
            self.kv_lora_rank,
            self.v_dim,
            dtype=torch.bfloat16,
        )

        reference = torch.bmm(attn_output.transpose(0, 1), w_vc)
        reference = reference.transpose(0, 1).flatten(1, 2)
        output_buffer = _bmm_mla_absorb_bf16_to_contiguous(attn_output, w_vc)
        output = output_buffer.flatten(1, 2)

        self.assertTrue(torch.equal(output, reference))
        self.assertTrue(output_buffer.is_contiguous())
        self.assertEqual(
            output.untyped_storage().data_ptr(),
            output_buffer.untyped_storage().data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
