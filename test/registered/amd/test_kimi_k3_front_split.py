"""Layout check for the dual-stream split of the Kimi-K3 fused MoE front.

Inside the ROCm overlap window _forward_fused stops using the merged
[gate_up | gate | latent] front weight and instead runs the shared gate_up on
the side stream (ATOM's shape) while the main stream takes a narrowed
[gate | latent] front. Both weights are dim-0 views over the one buffer
_merge_weights_as_views built, so the split path is only correct if the row
offsets line up exactly: the narrowed front must start where the shared
gate_up block ends, and the shared module's own .weight must still address
that block.

Getting this wrong is silent -- the shapes stay valid and the model still
produces plausible tokens off the wrong rows -- so it is pinned here rather
than left to an end-to-end accuracy run.
"""

import unittest

import torch

from sglang.srt.models.kimi_k3 import _merge_weights_as_views
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

HIDDEN = 7168
# 2 * (moe_intermediate_size 3072 * num_shared_experts 2 / tp 8)
GATE_UP = 1536
NUM_EXPERTS = 896  # router gate rows
LATENT = 3584  # routed_expert_hidden_size
TOL = 8e-3  # bf16 ULP is ~4e-3 relative


class _Fake(torch.nn.Module):
    """Minimal stand-in for a linear layer: _merge_weights_as_views only
    touches .weight.data."""

    def __init__(self, rows, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02,
            requires_grad=False,
        )


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3FrontSplit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dev = torch.device("cuda", 0)
        cls.gate_up = _Fake(GATE_UP, dev)
        cls.gate = _Fake(NUM_EXPERTS, dev)
        cls.latent = _Fake(LATENT, dev)
        cls.pre = [m.weight.data.clone() for m in (cls.gate_up, cls.gate, cls.latent)]
        cls.front_w, cls.sizes = _merge_weights_as_views(
            [cls.gate_up, cls.gate, cls.latent]
        )
        # exactly what _merge_front_weights stores as self._front_w_nogu
        cls.front_nogu = cls.front_w[cls.sizes[0] :]
        cls.x = torch.randn(32, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02

    def test_sizes(self):
        self.assertEqual(self.sizes, [GATE_UP, NUM_EXPERTS, LATENT])
        self.assertEqual(self.front_w.shape[0], GATE_UP + NUM_EXPERTS + LATENT)
        self.assertEqual(self.front_nogu.shape[0], NUM_EXPERTS + LATENT)

    def test_narrowed_front_aliases_merged(self):
        """_front_w_nogu must be a view, not a copy -- the split is supposed to
        cost no extra memory."""
        self.assertEqual(
            self.front_nogu.data_ptr(),
            self.front_w[GATE_UP].data_ptr(),
        )
        self.assertTrue(self.front_nogu.is_contiguous())

    def test_shared_weight_still_addresses_its_block(self):
        """The side stream calls shared_experts.gate_up_proj.weight directly
        rather than slicing _front_w, so that module's weight has to be the
        leading block and has to still hold the loaded values."""
        self.assertEqual(self.gate_up.weight.data.data_ptr(), self.front_w.data_ptr())
        torch.testing.assert_close(self.gate_up.weight.data, self.pre[0])
        torch.testing.assert_close(self.front_w[:GATE_UP], self.pre[0])

    def test_split_matches_merged(self):
        """The two paths must produce the same three tensors: merged front then
        split, versus shared gate_up alone plus the narrowed front."""
        merged = torch.mm(self.x, self.front_w.t())
        m_gu, m_gate, m_lat = torch.split(merged, self.sizes, dim=-1)

        s_gu = torch.mm(self.x, self.gate_up.weight.data.t())
        narrowed = torch.mm(self.x, self.front_nogu.t())
        s_gate, s_lat = torch.split(narrowed, self.sizes[1:], dim=-1)

        for a, b, name in (
            (m_gu, s_gu, "gate_up"),
            (m_gate, s_gate, "gate"),
            (m_lat, s_lat, "latent"),
        ):
            with self.subTest(tensor=name):
                torch.testing.assert_close(a, b, rtol=TOL, atol=TOL)


if __name__ == "__main__":
    unittest.main()
