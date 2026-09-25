"""Tests for Kimi-K3 non-EP tuned front and latent MXFP4 adapters."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.kimi_k3 import latent_mxfp4_aiter_hip
from sglang.srt.environ import envs
from sglang.srt.models.kimi_k3_rocm_moe_front import (
    k3_tuned_front_gemm,
    k3_use_latent_mxfp4,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3LatentMXFP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not latent_mxfp4_aiter_hip.supported():
            raise unittest.SkipTest("Kimi-K3 latent MXFP4 needs gfx950")
        torch.manual_seed(0)
        cls.device = torch.device("cuda", 0)
        cls.weight = (
            torch.randn(3584, 7168, dtype=torch.bfloat16, device=cls.device) * 0.02
        )
        cls.packed, cls.scale = latent_mxfp4_aiter_hip.pack(
            cls.weight, "latent down_proj"
        )

    def test_pack_footprint(self):
        expected = latent_mxfp4_aiter_hip.packed_bytes(tuple(self.weight.shape))
        actual = self.packed.numel() * self.packed.element_size()
        actual += self.scale.numel() * self.scale.element_size()
        self.assertEqual(actual, expected)

    def test_run_matches_bf16_reference(self):
        x = torch.randn(8, 7168, dtype=torch.bfloat16, device=self.device) * 0.1
        actual = latent_mxfp4_aiter_hip.run(x, self.packed, self.scale)
        expected = torch.nn.functional.linear(x, self.weight)
        rel_l2 = (
            (actual.float() - expected.float()).norm() / expected.float().norm()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        # Per-1x32 MXFP4 has expected quantization error on random normal
        # inputs; these bounds catch layout/scale corruption while leaving
        # model-level quality to the GSM8K gate.
        self.assertLess(rel_l2, 0.20)
        self.assertGreater(cosine, 0.98)

    def test_graph_replay_uses_changed_inputs(self):
        x = torch.randn(8, 7168, dtype=torch.bfloat16, device=self.device) * 0.1
        latent_mxfp4_aiter_hip.run(x, self.packed, self.scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = latent_mxfp4_aiter_hip.run(x, self.packed, self.scale)
        graph.replay()
        first = out.clone()
        x.copy_(torch.randn_like(x) * 0.1)
        graph.replay()
        second = out.clone()
        self.assertFalse(torch.equal(first, second))

    def test_threshold_is_fail_closed(self):
        mlp = SimpleNamespace(_k3_latent_mxfp4=SimpleNamespace(min_tokens=2048))
        self.assertFalse(k3_use_latent_mxfp4(mlp, 2047))
        self.assertTrue(k3_use_latent_mxfp4(mlp, 2048))
        self.assertFalse(k3_use_latent_mxfp4(SimpleNamespace(), 4096))

    def test_tuned_front_token_window(self):
        x = torch.randn(48, 7168, dtype=torch.bfloat16, device=self.device)
        weight = torch.empty(6016, 7168, dtype=torch.bfloat16, device=self.device)
        expected = torch.randn(48, 6016, dtype=torch.bfloat16, device=self.device)
        with (
            envs.SGLANG_USE_AITER.override(True),
            envs.SGLANG_ROCM_K3_AITER_TUNED_MOE_FRONT.override(True),
            envs.SGLANG_ROCM_K3_AITER_TUNED_MOE_FRONT_MIN_TOKENS.override(48),
            envs.SGLANG_ROCM_K3_AITER_TUNED_MOE_FRONT_MAX_TOKENS.override(192),
            patch("aiter.tuned_gemm.tgemm.mm", return_value=expected) as tuned_mm,
        ):
            self.assertIsNone(k3_tuned_front_gemm(x[:47], weight))
            big = torch.empty(193, 7168, dtype=torch.bfloat16, device=self.device)
            self.assertIsNone(k3_tuned_front_gemm(big, weight))
            self.assertIsNone(k3_tuned_front_gemm(x, weight[:6000]))
            tuned_mm.assert_not_called()
            self.assertIs(k3_tuned_front_gemm(x, weight), expected)
            tuned_mm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
