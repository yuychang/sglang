"""ROCm parity for K3 route/quant preparation and the AITER handoff."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

NUM_EXPERTS = 896
HIDDEN = 3584
TOPK = 16


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3RouteQuantAiter(CustomTestCase):
    def test_route_quant_and_prequant_sort_parity(self):
        import aiter
        from aiter import dtypes
        from aiter.fused_moe import moe_sorting
        from aiter.ops.quant import (
            per_1x32_mx_quant_hip,
            sort_prequantized_mxfp8_for_moe,
        )
        from sglang.kernels.ops.moe import (
            moe_route_radix4,
            moe_route_quant_fused,
        )

        self.assertTrue(moe_route_quant_fused.available())
        torch.manual_seed(11)
        for tokens in (1, 4, 32):
            scores = torch.randn(
                tokens, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16
            )
            bias = (
                torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
                * 0.01
            )
            x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)

            weights, ids, _, x_q, x_s = (
                moe_route_quant_fused.route_quant_fused(
                    scores,
                    bias,
                    x,
                    TOPK,
                    renormalize=True,
                    routed_scaling_factor=1.0,
                    apply_scale=False,
                )
            )
            ref_weights, ref_ids = moe_route_radix4.route_radix4(
                scores, bias, TOPK, True, 1.0
            )
            ids_sorted, order = ids.sort(dim=1)
            ref_ids_sorted, ref_order = ref_ids.sort(dim=1)
            self.assertTrue(torch.equal(ids_sorted, ref_ids_sorted))
            torch.testing.assert_close(
                weights.gather(1, order),
                ref_weights.gather(1, ref_order),
                rtol=0,
                atol=1.5e-8,
            )

            ref_q, ref_s = per_1x32_mx_quant_hip(
                x,
                quant_dtype=dtypes.fp8,
                scale_type=dtypes.fp8_e8m0,
                shuffle=False,
            )
            self.assertTrue(torch.equal(x_q.view(torch.uint8), ref_q.view(torch.uint8)))
            self.assertTrue(torch.equal(x_s.view(torch.uint8), ref_s.view(torch.uint8)))

            sorted_ids, _, _, num_valid_ids, _ = moe_sorting(
                ids, weights, NUM_EXPERTS, HIDDEN, torch.bfloat16
            )
            expected_q, expected_s = aiter.fused_dynamic_mxfp8_quant_moe_sort(
                x, sorted_ids, num_valid_ids[0], tokens, TOPK, 64
            )
            actual_q, actual_s = sort_prequantized_mxfp8_for_moe(
                x_q,
                x_s.view(dtypes.fp8_e8m0).reshape(tokens, -1),
                sorted_ids,
                num_valid_ids[0],
                tokens,
            )
            self.assertTrue(
                torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
            )
            # Padding rows are not consumed by GEMM and may remain unwritten.
            valid = int(num_valid_ids[0].item())
            mask = expected_s[:valid].view(torch.uint8) != 0
            self.assertTrue(
                torch.equal(
                    actual_s[:valid].view(torch.uint8)[mask],
                    expected_s[:valid].view(torch.uint8)[mask],
                )
            )


if __name__ == "__main__":
    unittest.main()
