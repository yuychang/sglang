import unittest

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    _get_bf16_mla_bmm_weight,
)


class TestMlaBmmWeightScale(unittest.TestCase):
    def test_scalar_identity_scale_avoids_weight_multiply(self):
        weight = torch.randn(4, 8, 16, dtype=torch.bfloat16)

        actual = _get_bf16_mla_bmm_weight(weight, 1.0)

        self.assertEqual(actual.data_ptr(), weight.data_ptr())
        torch.testing.assert_close(actual, weight, rtol=0, atol=0)

    def test_non_identity_scale_preserves_bmm_weight_semantics(self):
        weight = torch.randn(4, 8, 16, dtype=torch.bfloat16)

        actual = _get_bf16_mla_bmm_weight(weight, 0.5)
        expected = weight * 0.5

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertNotEqual(actual.data_ptr(), weight.data_ptr())

    def test_tensor_scale_remains_applied(self):
        weight = torch.randn(4, 8, 16, dtype=torch.bfloat16)
        scale = torch.tensor(1.0, dtype=torch.bfloat16)

        actual = _get_bf16_mla_bmm_weight(weight, scale)

        torch.testing.assert_close(actual, weight * scale, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
