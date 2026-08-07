"""Hardware dispatch for the Kimi-K3 attention-residual aggregation.

The SM100 TMA kernel is selected by a raw device-capability probe, and HIP
answers that probe with the gfx architecture rather than an NVIDIA compute
capability: gfx1030 reports (10, 3), gfx1100 reports (11, 0), gfx1250 reports
(12, 5). Those parts must not reach an NVIDIA-only kernel, and — since the fast
gate is tested first — must still reach the ROCm aggregation kernel.
"""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.srt.layers import attn_residual
from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

_H = 7168  # the only hidden size the fast kernel is instantiated for


@contextmanager
def _pretend_gpu(*, cuda, hip, capability):
    """Run the gates as if this were one specific GPU.

    Both gates memoize into module globals, so they are cleared on the way in
    and again on the way out — otherwise the first case decides the rest.
    """
    attn_residual._FAST_SUPPORTED = None
    attn_residual._HIP_SHAPE_GATE = None
    try:
        with patch.object(attn_residual, "is_cuda", return_value=cuda), patch.object(
            attn_residual, "is_hip", return_value=hip
        ), patch("torch.cuda.get_device_capability", return_value=capability):
            yield
    finally:
        attn_residual._FAST_SUPPORTED = None
        attn_residual._HIP_SHAPE_GATE = None


# gfx parts whose HIP-reported major clears the fast kernel's `>= 10` test.
_HIP_CAPABILITIES = {"gfx1030": (10, 3), "gfx1100": (11, 0), "gfx1250": (12, 5)}


class TestAttnResidualFastGate(CustomTestCase):
    def test_hip_never_claims_the_sm100_kernel(self):
        for arch, capability in _HIP_CAPABILITIES.items():
            with self.subTest(arch=arch):
                with _pretend_gpu(cuda=False, hip=True, capability=capability):
                    self.assertFalse(attn_residual._use_fast(_H))

    def test_cuda_blackwell_takes_the_fast_kernel(self):
        with _pretend_gpu(cuda=True, hip=False, capability=(10, 0)):
            self.assertTrue(attn_residual._use_fast(_H))

    def test_cuda_hopper_does_not(self):
        with _pretend_gpu(cuda=True, hip=False, capability=(9, 0)):
            self.assertFalse(attn_residual._use_fast(_H))

    def test_fast_kernel_is_shape_bound(self):
        with _pretend_gpu(cuda=True, hip=False, capability=(10, 0)):
            self.assertFalse(attn_residual._use_fast(4096))


class TestAttnResidualDispatch(CustomTestCase):
    """Which aggregation implementation _aggregate hands the work to."""

    NVB = 4

    def _dispatch(self, *, cuda, hip, capability, hidden_size=_H):
        prefix_sum = torch.empty(1, hidden_size)
        bank = torch.empty(1, 8, hidden_size)
        with _pretend_gpu(cuda=cuda, hip=hip, capability=capability), patch.object(
            attn_residual, "_aggregate_fast"
        ) as fast, patch.object(
            attn_residual, "_aggregate_hip", return_value=(prefix_sum, prefix_sum)
        ) as hip_agg, patch.object(
            attn_residual, "_aggregate_fused"
        ) as fused:
            attn_residual._aggregate(
                prefix_sum, bank, self.NVB, None, None, None, write_bank_row=False
            )
        return fast, hip_agg, fused

    def test_hip_reaches_the_rocm_kernel(self):
        for arch, capability in _HIP_CAPABILITIES.items():
            with self.subTest(arch=arch):
                fast, hip_agg, fused = self._dispatch(
                    cuda=False, hip=True, capability=capability
                )
                fast.assert_not_called()
                hip_agg.assert_called_once()
                fused.assert_not_called()

    def test_cuda_blackwell_reaches_the_tma_kernel(self):
        fast, hip_agg, fused = self._dispatch(cuda=True, hip=False, capability=(10, 0))
        fast.assert_called_once()
        hip_agg.assert_not_called()
        fused.assert_not_called()

    def test_cuda_hopper_reaches_the_triton_pipeline(self):
        fast, hip_agg, fused = self._dispatch(cuda=True, hip=False, capability=(9, 0))
        fast.assert_not_called()
        hip_agg.assert_not_called()
        fused.assert_called_once()


if __name__ == "__main__":
    unittest.main()
