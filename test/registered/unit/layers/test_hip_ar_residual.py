import contextlib
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.communication.hip_ar_residual import (
    enabled,
    try_all_reduce_add,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextlib.contextmanager
def _hip_env(*, is_hip=True, flag=True, max_tokens=8, ca_comm=None):
    group = Mock()
    group.ca_comm = ca_comm
    with (
        patch(
            "sglang.srt.layers.communication.hip_ar_residual.is_hip",
            return_value=is_hip,
        ),
        envs.SGLANG_ROCM_AR_RESIDUAL.override(flag),
        envs.SGLANG_ROCM_AR_RESIDUAL_MAX_TOKENS.override(max_tokens),
        patch(
            "sglang.srt.distributed.parallel_state.get_tp_group",
            return_value=group,
        ),
    ):
        yield


def _ca_comm(out=None):
    ca_comm = Mock()
    ca_comm.disabled = False
    ca_comm.custom_all_reduce_residual.return_value = out
    return ca_comm


class TestHipArResidual(CustomTestCase):
    def test_helper_stays_off_when_flag_disabled(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        with _hip_env(flag=False, ca_comm=_ca_comm()):
            self.assertFalse(enabled())
            self.assertIsNone(try_all_reduce_add(x, torch.ones_like(x)))

    def test_helper_stays_off_when_not_hip(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        with _hip_env(is_hip=False, ca_comm=_ca_comm()):
            self.assertFalse(enabled())
            self.assertIsNone(try_all_reduce_add(x, torch.ones_like(x)))

    def test_helper_fail_closed_without_communicator(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        with _hip_env(ca_comm=None):
            self.assertTrue(enabled())
            self.assertIsNone(try_all_reduce_add(x, torch.ones_like(x)))

    def test_helper_dispatches_residual_kernel(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        out = torch.full_like(x, 3)
        ca_comm = _ca_comm(out)
        with _hip_env(ca_comm=ca_comm):
            got = try_all_reduce_add(x, residual)
        self.assertIs(got, out)
        ca_comm.custom_all_reduce_residual.assert_called_once_with(x, residual)

    def test_helper_skips_m8_when_capped_at_4(self):
        x = torch.zeros(8, 7168, dtype=torch.bfloat16)
        ca_comm = _ca_comm()
        with _hip_env(max_tokens=4, ca_comm=ca_comm):
            self.assertIsNone(try_all_reduce_add(x, torch.ones_like(x)))
        ca_comm.custom_all_reduce_residual.assert_not_called()

    def test_helper_skips_m_outside_decode_buckets(self):
        """Only M in {1, 2, 4, 8} has a residual kernel; M=3 under the cap must
        stay on split AR + add."""
        x = torch.zeros(3, 7168, dtype=torch.bfloat16)
        ca_comm = _ca_comm()
        with _hip_env(ca_comm=ca_comm):
            self.assertIsNone(try_all_reduce_add(x, torch.ones_like(x)))
        ca_comm.custom_all_reduce_residual.assert_not_called()

    def test_helper_covers_m8(self):
        x = torch.zeros(8, 7168, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        out = torch.full_like(x, 3)
        ca_comm = _ca_comm(out)
        with _hip_env(ca_comm=ca_comm):
            self.assertIs(try_all_reduce_add(x, residual), out)
        ca_comm.custom_all_reduce_residual.assert_called_once_with(x, residual)

    def test_helper_normalizes_kda_residual_view(self):
        x = torch.zeros(8, 7168, dtype=torch.bfloat16)
        residual = torch.ones(1, 8, 7168, dtype=torch.bfloat16)
        out = torch.full_like(x, 3)
        ca_comm = _ca_comm(out)
        with _hip_env(ca_comm=ca_comm):
            self.assertIs(try_all_reduce_add(x, residual), out)
        actual_residual = ca_comm.custom_all_reduce_residual.call_args.args[1]
        self.assertEqual(actual_residual.shape, x.shape)
        self.assertEqual(actual_residual.data_ptr(), residual.data_ptr())

    def test_none_residual_uses_plain_custom_ar(self):
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)
        out = torch.ones_like(x)
        ca_comm = _ca_comm()
        ca_comm.custom_all_reduce.return_value = out
        with _hip_env(ca_comm=ca_comm):
            got = try_all_reduce_add(x, None)
        self.assertIs(got, out)
        ca_comm.custom_all_reduce.assert_called_once_with(x)


if __name__ == "__main__":
    unittest.main()
