# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.kernels.ops.kimi_k3 import mla_output_gate_fp8_quant as ops
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip()),
    reason="K3 MLA gate+FP8 quant runs on ROCm",
)


@pytest.mark.parametrize("tokens", [1, 4, 64, 256])
@pytest.mark.parametrize("hidden", [1024, 2048])
def test_matches_gate_then_per_token_quant(tokens, hidden):
    torch.manual_seed(tokens)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    assert ops.covered(x, gate)
    q, scale = ops.kimi_k3_mla_output_gate_fp8_quant(x, gate)
    assert scale.shape == (tokens, 1)

    ref = (x * torch.sigmoid(gate)).float()
    fp8_max = torch.finfo(q.dtype).max
    ref_scale = ref.abs().amax(dim=1, keepdim=True) / fp8_max
    torch.testing.assert_close(scale, ref_scale, rtol=1e-2, atol=1e-6)
    deq = q.float() * scale
    rel = (deq - ref).norm() / ref.norm()
    assert rel.item() < 0.05
