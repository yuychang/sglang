import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRocmFusedArMxfp4QuantArg(CustomTestCase):
    def test_flag_defaults_off_and_parses_on(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        defaults = parser.parse_args(["--model", "dummy"])
        enabled = parser.parse_args(
            ["--model", "dummy", "--enable-rocm-fused-ar-mxfp4-quant"]
        )

        self.assertFalse(defaults.enable_rocm_fused_ar_mxfp4_quant)
        self.assertTrue(enabled.enable_rocm_fused_ar_mxfp4_quant)


if __name__ == "__main__":
    unittest.main()
