"""SGLANG_USE_GLUON_DSA_PREFILL_LOGITS opts the ROCm prefill indexer into aiter's Triton
(Gluon) fp8_mqa_logits; unset keeps FlyDSL."""

import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillLogitsSwitch(CustomTestCase):
    def _kernel(self, use_gluon):
        """Which aiter module the resolver returns; None leaves the flag unset."""
        flydsl = mock.Mock(flydsl_fp8_mqa_logits="flydsl")
        triton = mock.Mock(fp8_mqa_logits="gluon")
        flag = envs.SGLANG_USE_GLUON_DSA_PREFILL_LOGITS
        modules = {
            "aiter": mock.Mock(),
            "aiter.ops": mock.Mock(),
            "aiter.ops.triton": mock.Mock(),
            "aiter.ops.flydsl": flydsl,
            "aiter.ops.triton.fp8_mqa_logits": triton,
        }
        dsa_indexer._resolve_hip_prefill_mqa_logits.cache_clear()
        try:
            with (
                mock.patch.dict("sys.modules", modules),
                flag.override(bool(use_gluon)),
            ):
                if use_gluon is None:
                    flag.clear()
                return dsa_indexer._resolve_hip_prefill_mqa_logits()
        finally:
            dsa_indexer._resolve_hip_prefill_mqa_logits.cache_clear()

    def test_default_is_flydsl(self):
        self.assertEqual(self._kernel(None), "flydsl")

    def test_flag_opts_into_gluon(self):
        self.assertEqual(self._kernel(True), "gluon")
        self.assertEqual(self._kernel(False), "flydsl")


if __name__ == "__main__":
    unittest.main()
