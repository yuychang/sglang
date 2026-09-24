"""Unit tests for srt/models/kimi_k3_rocm_preroute."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.kimi_k3 import moe_preroute_aiter_hip
from sglang.srt.layers.activation import SituAndMul
from sglang.srt.models.kimi_k3_rocm_preroute import (
    k3_prepare_preroute_fp8,
    k3_run_preroute,
    quantize_fp8_rows,
)
from sglang.test.test_utils import CustomTestCase


class TestQuantizeFp8Rows(CustomTestCase):
    def test_amax_scale_round_trips(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 256) * torch.logspace(-2, 1, 64).unsqueeze(1)
        q, scale = quantize_fp8_rows(weight)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (64,))
        torch.testing.assert_close(
            scale, weight.abs().amax(dim=1) / 448.0, rtol=0, atol=0
        )
        rel = (q.float() * scale[:, None] - weight).norm() / weight.norm()
        self.assertLess(rel.item(), 0.05)

    def test_pow2_scale_keeps_mxfp4_values_exact(self):
        # FP4 E2M1 values times per-32 E8M0 block scales, like a dequantized
        # Quark MXFP4 weight.
        torch.manual_seed(0)
        fp4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        values = fp4[torch.randint(0, 8, (16, 256))]
        values *= torch.where(torch.rand(16, 256) < 0.5, -1.0, 1.0)
        block_exp = torch.randint(-6, 2, (16, 8)).repeat_interleave(32, dim=1)
        weight = values * torch.exp2(block_exp.float())
        q, scale = quantize_fp8_rows(weight, pow2_scale=True)
        self.assertTrue(torch.equal(scale, torch.exp2(torch.log2(scale).round())))
        torch.testing.assert_close(q.float() * scale[:, None], weight, rtol=0, atol=0)


def _fake_mlp(routed_rows=3584, beta=4.0, linear_beta=25.0):
    def linear(rows, cols):
        return SimpleNamespace(weight=torch.randn(rows, cols, dtype=torch.bfloat16))

    shared = SimpleNamespace(
        gate_up_proj=linear(1536, 7168),
        down_proj=linear(7168, 768),
        act_fn=SituAndMul(beta=beta, linear_beta=linear_beta),
    )
    return SimpleNamespace(
        shared_experts=shared,
        routed_expert_down_proj=linear(routed_rows, 7168),
        gate=linear(896, 7168),
        _eligible_for_fused_front=True,
    )


class TestPreparePreroute(CustomTestCase):
    def _prepare(self, mlp):
        with (
            mock.patch.object(moe_preroute_aiter_hip, "enabled", return_value=True),
            mock.patch.object(
                moe_preroute_aiter_hip,
                "cooperative_preactivated_enabled",
                return_value=True,
            ),
            mock.patch.object(moe_preroute_aiter_hip, "warmup"),
        ):
            k3_prepare_preroute_fp8(mlp)

    def test_builds_interleaved_copy(self):
        mlp = _fake_mlp()
        self._prepare(mlp)
        p = mlp._k3_preroute
        self.assertIsNotNone(p)
        self.assertEqual(p.routed_w.shape, (3584, 7168))
        self.assertEqual((p.beta, p.linear_beta), (4.0, 25.0))
        # Row 2i is gate row i, row 2i+1 is up row i.
        self.assertTrue(
            torch.equal(
                p.inter_w[0::2].view(torch.uint8), p.shared_w[:768].view(torch.uint8)
            )
        )
        self.assertTrue(
            torch.equal(
                p.inter_w[1::2].view(torch.uint8), p.shared_w[768:].view(torch.uint8)
            )
        )
        self.assertTrue(torch.equal(p.inter_s[1::2], p.shared_s[768:]))

    def test_skips_unexpected_shapes(self):
        mlp = _fake_mlp(routed_rows=1792)
        self._prepare(mlp)
        self.assertIsNone(mlp._k3_preroute)

    def test_skips_without_linear_beta(self):
        mlp = _fake_mlp(linear_beta=None)
        self._prepare(mlp)
        self.assertIsNone(mlp._k3_preroute)

    def test_disabled_leaves_front_alone(self):
        mlp = _fake_mlp()
        with mock.patch.object(moe_preroute_aiter_hip, "enabled", return_value=False):
            k3_prepare_preroute_fp8(mlp)
        self.assertIsNone(mlp._k3_preroute)
        self.assertIsNone(k3_run_preroute(mlp, torch.zeros(1, 7168)))


if __name__ == "__main__":
    unittest.main()
