import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear import utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSelectVerifyIntermediateStateIndices(unittest.TestCase):
    def setUp(self):
        self.default = torch.arange(8, dtype=torch.int32)
        self.req_pool = torch.tensor([5, 6, 7], dtype=torch.int64)
        self.cache = torch.tensor([3, -1, 4], dtype=torch.int32)

    def test_disabled_never_evaluates_mask(self):
        calls = []

        def valid():
            calls.append(1)
            return self.cache >= 0

        with patch.object(utils, "pp_spec_stable_rows_enabled", return_value=False):
            out = utils.select_verify_intermediate_state_indices(
                self.default, self.req_pool, valid, 8
            )
        self.assertIs(out, self.default)
        self.assertEqual(calls, [])

    def test_enabled_matches_eager_mask(self):
        with patch.object(utils, "pp_spec_stable_rows_enabled", return_value=True):
            eager = utils.select_verify_intermediate_state_indices(
                self.default, self.req_pool, self.cache >= 0, 8
            )
            lazy = utils.select_verify_intermediate_state_indices(
                self.default, self.req_pool, lambda: self.cache >= 0, 8
            )
        self.assertTrue(torch.equal(eager, lazy))
        self.assertEqual(lazy.tolist(), [5, 8, 7])
        self.assertEqual(lazy.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
