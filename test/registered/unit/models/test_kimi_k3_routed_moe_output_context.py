import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers import zero_copy_context
from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRoutedMoeOutputContext(unittest.TestCase):
    def test_topk_and_experts_both_see_the_latent_destination(self):
        latent = torch.zeros(4, 8)
        seen = {}

        def topk(hidden, logits):
            seen["topk"] = zero_copy_context.get_moe_output_spec(
                latent.shape, latent.dtype, latent.device
            )
            return "topk_output"

        def experts(routed, topk_output):
            seen["experts"] = zero_copy_context.get_moe_output_spec(
                latent.shape, latent.dtype, latent.device
            )
            return latent

        fake = SimpleNamespace(
            _route_quant_fuse_eligible=False, topk=topk, experts=experts
        )
        KimiK3MoE._forward_routed(fake, None, None, None, latent)

        self.assertIs(seen["topk"], latent)
        self.assertIs(seen["experts"], latent)
        self.assertIsNone(zero_copy_context.ctx.moe_output)

    def test_context_is_restored_when_topk_raises(self):
        latent = torch.zeros(2, 4)

        def topk(hidden, logits):
            raise RuntimeError("boom")

        fake = SimpleNamespace(
            _route_quant_fuse_eligible=False, topk=topk, experts=None
        )
        with self.assertRaises(RuntimeError):
            KimiK3MoE._forward_routed(fake, None, None, None, latent)
        self.assertIsNone(zero_copy_context.ctx.moe_output)


if __name__ == "__main__":
    unittest.main()
