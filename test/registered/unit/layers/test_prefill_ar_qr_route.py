import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    prefers_quick_allreduce,
)
from sglang.srt.layers.communication.hip_ar_residual import try_all_reduce_add
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MIB = 1024 * 1024


def _qr(accepts=True, disabled=False):
    qr = Mock()
    qr.disabled = disabled
    qr.should_quick_allreduce.return_value = accepts
    return qr


class TestPrefillArQrRoute(CustomTestCase):
    def test_default_off(self):
        x = torch.empty(2502, 7168, dtype=torch.bfloat16)
        self.assertFalse(prefers_quick_allreduce(_qr(), x, 0))

    def test_threshold_gates_by_bytes(self):
        qr = _qr()
        attn_2502 = torch.empty(2502, 7168, dtype=torch.bfloat16)
        small = torch.empty(772, 7168, dtype=torch.bfloat16)
        self.assertTrue(prefers_quick_allreduce(qr, attn_2502, 32 * _MIB))
        self.assertFalse(prefers_quick_allreduce(qr, small, 32 * _MIB))

    def test_qr_own_gates_still_apply(self):
        x = torch.empty(2502, 7168, dtype=torch.bfloat16)
        self.assertFalse(prefers_quick_allreduce(_qr(accepts=False), x, _MIB))
        self.assertFalse(prefers_quick_allreduce(_qr(disabled=True), x, _MIB))
        self.assertFalse(prefers_quick_allreduce(None, x, _MIB))

    def test_residual_none_defers_to_dispatcher_when_qr_preferred(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        ca_comm = Mock()
        ca_comm.disabled = False
        group = Mock()
        group.ca_comm = ca_comm
        group.prefers_quick_allreduce.return_value = True
        with (
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.hip_ar_residual.envs.SGLANG_ROCM_AR_RESIDUAL.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=group,
            ),
        ):
            self.assertIsNone(try_all_reduce_add(x, None))
        ca_comm.custom_all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
