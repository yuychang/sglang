import contextlib
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.communication.hip_ar_attnres import (
    covers,
    enabled,
    try_all_reduce_residual_attnres,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextlib.contextmanager
def _hip_env(*, is_hip=True, flag=True, max_tokens=16):
    with (
        patch(
            "sglang.srt.layers.communication.hip_ar_attnres.is_hip",
            return_value=is_hip,
        ),
        envs.SGLANG_ROCM_AR_ATTNRES.override(flag),
        envs.SGLANG_ROCM_AR_ATTNRES_MAX_TOKENS.override(max_tokens),
    ):
        yield


class TestHipArAttnRes(CustomTestCase):
    def test_covers_token_speed_window(self):
        with _hip_env():
            self.assertTrue(covers(1))
            self.assertTrue(covers(16))
            self.assertFalse(covers(17))
            self.assertFalse(covers(0))

    def test_cap_can_narrow_the_window(self):
        with _hip_env(max_tokens=8):
            self.assertTrue(covers(8))
            self.assertFalse(covers(16))

    def test_helper_stays_off_without_the_flag(self):
        partial = torch.zeros(4, 128, dtype=torch.bfloat16)
        with _hip_env(flag=False):
            self.assertFalse(enabled())
            self.assertIsNone(
                try_all_reduce_residual_attnres(
                    partial, partial, partial[:, None, :], 1, None, None, object()
                )
            )

    def test_helper_rejects_a_missing_residual(self):
        partial = torch.zeros(4, 128, dtype=torch.bfloat16)
        with _hip_env():
            self.assertIsNone(
                try_all_reduce_residual_attnres(
                    partial, None, partial[:, None, :], 1, None, None, object()
                )
            )


if __name__ == "__main__":
    unittest.main()
