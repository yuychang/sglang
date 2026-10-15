import unittest

import torch

from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")


def _run(hidden_size: int, nvb: int, tuned_launch: bool) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(nvb)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda")

    num_tokens = 2
    prefix = randn(num_tokens, hidden_size).to(torch.bfloat16)
    bank = randn(num_tokens, nvb, hidden_size).to(torch.bfloat16)
    cw = randn(hidden_size) * hidden_size**-0.5
    ow = (1 + 0.1 * randn(hidden_size)).to(torch.bfloat16)
    out = torch.empty_like(prefix)
    attn_res_hip(
        prefix,
        bank,
        cw,
        ow,
        out,
        nvb,
        1e-6,
        1e-6,
        tuned_launch=tuned_launch,
    )
    return out


class TestAttnResHipTunedLaunch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise unittest.SkipTest("ROCm is not available")

    def test_tuned_launch_matches_untuned(self):
        # Depths 1, 2 and 4 only change waves_per_eu, an occupancy hint,
        # so they stay bit-identical; depth 8 also changes the warp split.
        for nvb, exact in ((1, True), (2, True), (4, True), (8, False)):
            with self.subTest(nvb=nvb):
                tuned = _run(hidden_size=7168, nvb=nvb, tuned_launch=True)
                untuned = _run(hidden_size=7168, nvb=nvb, tuned_launch=False)
                if exact:
                    torch.testing.assert_close(tuned, untuned, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(tuned, untuned, rtol=1e-2, atol=1e-2)

    def test_tuned_launch_outside_table_falls_back(self):
        # supports_attn_res_hip admits depths past the tuned table when H is small;
        # those must launch untuned instead of raising KeyError.
        tuned = _run(hidden_size=1024, nvb=9, tuned_launch=True)
        untuned = _run(hidden_size=1024, nvb=9, tuned_launch=False)
        torch.testing.assert_close(tuned, untuned, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
