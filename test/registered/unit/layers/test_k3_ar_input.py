import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.communication.k3_ar_input import get_k3_ar_input
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestK3ArInput(CustomTestCase):
    def test_returns_registered_pool_view_when_custom_ar_covers_shape(self):
        view = torch.empty(4, 16, dtype=torch.bfloat16)
        ca_comm = Mock()
        ca_comm.disabled = False
        ca_comm.get_registered_input_buffer.return_value = view
        ca_comm.should_custom_ar.return_value = True
        with (
            patch(
                "sglang.srt.layers.communication.k3_ar_input.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.k3_ar_input.envs."
                "SGLANG_K3_AR_INPLACE_INPUT.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.distributed.get_tp_group",
                return_value=Mock(ca_comm=ca_comm),
            ),
        ):
            got = get_k3_ar_input(view.shape, view.dtype, view.device)
        self.assertIs(got, view)
        ca_comm.get_registered_input_buffer.assert_called_once_with(
            view.shape, view.dtype
        )

    def test_kill_switch_prevents_pool_access(self):
        with (
            patch(
                "sglang.srt.layers.communication.k3_ar_input.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.k3_ar_input.envs."
                "SGLANG_K3_AR_INPLACE_INPUT.get",
                return_value=False,
            ),
            patch("sglang.srt.distributed.get_tp_group") as get_group,
        ):
            self.assertIsNone(
                get_k3_ar_input((4, 16), torch.bfloat16, torch.device("cpu"))
            )
        get_group.assert_not_called()

    def test_falls_back_when_custom_ar_does_not_cover_shape(self):
        view = torch.empty(4, 16, dtype=torch.bfloat16)
        ca_comm = Mock()
        ca_comm.disabled = False
        ca_comm.get_registered_input_buffer.return_value = view
        ca_comm.should_custom_ar.return_value = False
        with (
            patch(
                "sglang.srt.layers.communication.k3_ar_input.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.k3_ar_input.envs."
                "SGLANG_K3_AR_INPLACE_INPUT.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.distributed.get_tp_group",
                return_value=Mock(ca_comm=ca_comm),
            ),
        ):
            self.assertIsNone(get_k3_ar_input(view.shape, view.dtype, view.device))


if __name__ == "__main__":
    unittest.main()
