"""Unit tests for srt/models/kimi_k3_rocm_quant and the K3 weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    QuarkW4A4MXFP4,
)
from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable
from sglang.srt.models.kimi_k3_rocm_quant import (
    _k3_channel_fp8_to_bf16,
    _k3_densify_quark_shared_experts,
)
from sglang.test.test_utils import CustomTestCase


def _per_channel_fp8(out_features: int, in_features: int):
    torch.manual_seed(0)
    dense = torch.randn(out_features, in_features, dtype=torch.float32)
    # Spread the per-channel magnitudes so a wrongly-broadcast scale cannot
    # coincidentally reproduce the right answer.
    dense *= torch.logspace(-1, 1, out_features).unsqueeze(1)
    scale = dense.abs().amax(dim=1, keepdim=True) / 448.0
    weight = (dense / scale).to(torch.float8_e4m3fn)
    return weight, scale, dense


class TestChannelFp8ToBf16(CustomTestCase):
    """aiter's batched absorb GEMM takes ``w_scale`` as a scalar, so a
    per-channel vector cannot reach it. The scale axis is the GEMM's
    contraction axis, so requantizing to one per-tensor scale is lossy --
    ~2.3% relative error on real K3 weights. Dequantizing to bf16 costs ~0.12%
    and lets the absorb use its bf16 path with w_scale at the 1.0 default."""

    def test_dequantizes_every_channel_with_its_own_scale(self):
        weight, scale, dense = _per_channel_fp8(out_features=8, in_features=16)
        module = SimpleNamespace(weight_scale=scale.squeeze(1))

        out = _k3_channel_fp8_to_bf16(module, weight)

        self.assertEqual(out.dtype, torch.bfloat16)
        # Only the original fp8 rounding plus the bf16 cast separate this from
        # the checkpoint's own values; no second quantization.
        exact = weight.to(torch.float32) * scale
        torch.testing.assert_close(out.float(), exact, rtol=0.01, atol=0.0)
        # Guard the failure this replaced: one shared scale would leave the
        # low-magnitude channels far from their true values.
        worst = ((out.float() - dense).abs().sum(1) / dense.abs().sum(1)).max()
        self.assertLess(worst.item(), 0.05)

    def test_accepts_a_column_vector_scale(self):
        """Quark serializes the scale as [out]; callers may hold [out, 1]."""
        weight, scale, _ = _per_channel_fp8(out_features=8, in_features=16)

        flat = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.squeeze(1)), weight
        )
        column = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.clone()), weight
        )

        torch.testing.assert_close(flat, column)


class TestMergeDtypeGuard(CustomTestCase):
    """``_merge_weights_as_views`` cats ``.weight`` alone, so fusing a quantized
    projection silently drops the scale tensors that give its integer payload
    meaning. Only dtypes that carry their full value in ``.weight`` may fuse."""

    def test_rejects_weights_that_carry_a_separate_scale(self):
        for dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4).to(dtype) for _ in range(2)]
                self.assertFalse(_is_unquantized_mergeable(weights))

    def test_accepts_plain_float_weights(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4, dtype=dtype) for _ in range(2)]
                self.assertTrue(_is_unquantized_mergeable(weights))

    def test_rejects_mixed_dtypes(self):
        weights = [
            torch.zeros(4, 4, dtype=torch.bfloat16),
            torch.zeros(4, 4, dtype=torch.float16),
        ]
        self.assertFalse(_is_unquantized_mergeable(weights))

    def test_rejects_float32(self):
        """The merged buffer is only built for the two half dtypes the fused
        KDA/MoE kernels read; fp32 must not slip in through a dtype-agreement
        check that only asks whether the operands match each other."""
        weights = [torch.zeros(4, 4, dtype=torch.float32) for _ in range(2)]
        self.assertFalse(_is_unquantized_mergeable(weights))


class TestDensifyQuarkSharedExperts(CustomTestCase):
    """Quark MXFP4 shared experts become BF16 before the MoE front merge."""

    @staticmethod
    def _mlp():
        scheme = object.__new__(QuarkW4A4MXFP4)
        scheme.is_checkpoint_mxfp4_serialized = True

        def linear():
            # Packed nibbles 1/2 decode to 0.5/1.0 with a unit e8m0 scale.
            return SimpleNamespace(
                weight=torch.full((2, 16), 0x21, dtype=torch.uint8),
                weight_scale=torch.full((2, 1), 127, dtype=torch.uint8),
                scheme=scheme,
            )

        return SimpleNamespace(
            shared_experts=SimpleNamespace(gate_up_proj=linear(), down_proj=linear())
        )

    def test_dequantizes_shared_experts_once(self):
        mlp = self._mlp()
        path = (
            "sglang.srt.layers.quantization.quark.schemes."
            "quark_w4a4_mxfp4._dequant_linear_to_bf16"
        )
        with mock.patch(path, True):
            _k3_densify_quark_shared_experts(mlp)
            weight = mlp.shared_experts.gate_up_proj.weight
            # The loader's own pass must reuse the BF16 weight.
            _k3_densify_quark_shared_experts(mlp)

        expected = torch.tensor([0.5, 1.0], dtype=torch.bfloat16).repeat(2, 16)
        for linear in (mlp.shared_experts.gate_up_proj, mlp.shared_experts.down_proj):
            torch.testing.assert_close(linear.weight.data, expected)
            self.assertIsNone(linear.weight_scale)
        self.assertEqual(
            mlp.shared_experts.gate_up_proj.weight.data_ptr(), weight.data_ptr()
        )

    def test_fp4_activation_keeps_packed_weights(self):
        mlp = self._mlp()
        path = (
            "sglang.srt.layers.quantization.quark.schemes."
            "quark_w4a4_mxfp4._dequant_linear_to_bf16"
        )
        with mock.patch(path, False):
            _k3_densify_quark_shared_experts(mlp)

        self.assertEqual(mlp.shared_experts.down_proj.weight.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
