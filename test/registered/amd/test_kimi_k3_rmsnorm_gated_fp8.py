"""Numerics check for the fused gated-RMSNorm + per-token FP8 quant kernel.

Under SGLANG_ROCM_K3_KDA_O_PROJ_FP8 the KDA output norm writes o_proj's
``(fp8, per-token scale)`` pair directly instead of a bf16 tensor that a
separate quant kernel then re-reads. This pins the fused result against the
unfused reference (gated norm in fp32, then per-token quant) and checks the
pieces the consuming GEMM relies on: the scale is per token across *all* heads,
the norm is per head, and a strided gate is read in place.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

HEADS = 12  # num_heads / tp8
HEAD_DIM = 128
EPS = 1e-6


def _ref(x, weight, gate, quant_dtype):
    """Gated RMSNorm then per-token quant, all in fp32."""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = (
        (xf * torch.rsqrt(var + EPS)) * weight.float() * torch.sigmoid(gate.float())
    )
    flat = normed.flatten(-2)  # [tokens, heads * head_dim]
    fp8_max = torch.finfo(quant_dtype).max
    scale = flat.abs().amax(dim=-1, keepdim=True) / fp8_max
    inv = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    return (flat * inv).clamp(-fp8_max, fp8_max).to(quant_dtype), scale


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3RMSNormGatedFp8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.fp8_utils import is_fp8_fnuz

        torch.manual_seed(0)
        cls.dev = torch.device("cuda", 0)
        cls.dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
        cls.weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=cls.dev)

    def _run(self, x, gate):
        from sglang.kernels.ops.kimi_k3 import rmsnorm_gated_fp8_per_token

        return rmsnorm_gated_fp8_per_token(x, self.weight, gate, EPS, self.dtype)

    def test_matches_unfused_reference(self):
        for tokens in (1, 3, 16, 129, 1024):
            with self.subTest(tokens=tokens):
                shape = (tokens, HEADS, HEAD_DIM)
                x = torch.randn(shape, dtype=torch.bfloat16, device=self.dev)
                gate = torch.randn(shape, dtype=torch.bfloat16, device=self.dev) * 2
                got, got_scale = self._run(x, gate)
                want, want_scale = _ref(x, self.weight, gate, self.dtype)

                self.assertEqual(got.dtype, self.dtype)
                self.assertEqual(tuple(got.shape), (tokens, HEADS * HEAD_DIM))
                self.assertEqual(tuple(got_scale.shape), (tokens, 1))
                self.assertEqual(got_scale.dtype, torch.float32)
                torch.testing.assert_close(got_scale, want_scale, rtol=2e-2, atol=0)
                # Compare dequantized: an fp8 code differing by one step is a
                # rounding tie, not an error, but the reconstructed value must
                # track the reference.
                deq = got.float() * got_scale
                ref = want.float() * want_scale
                denom = ref.abs().amax().clamp_min(1e-6)
                self.assertLess(((deq - ref).abs().amax() / denom).item(), 5e-2)

    def test_scale_spans_all_heads(self):
        """One head carrying a much larger magnitude must set the token scale;
        a per-head scale would clip it."""
        x = torch.full(
            (2, HEADS, HEAD_DIM), 0.01, dtype=torch.bfloat16, device=self.dev
        )
        x[:, HEADS - 1, :] = 4.0
        gate = torch.zeros_like(x)
        _, scale = self._run(x, gate)
        # RMSNorm is per head, so every head normalizes to ~1 * weight * 0.5;
        # the amax is therefore driven by the gain, identically for both tokens.
        _, want_scale = _ref(x, self.weight, gate, self.dtype)
        torch.testing.assert_close(scale, want_scale, rtol=2e-2, atol=0)

    def test_strided_gate_read_in_place(self):
        """The gate is a column slice of the fused in-proj output in the model,
        so it arrives with a row stride wider than heads * head_dim."""
        tokens = 33
        x = torch.randn(
            (tokens, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=self.dev
        )
        wide = torch.randn(
            (tokens, 4 * HEADS * HEAD_DIM), dtype=torch.bfloat16, device=self.dev
        )
        gate = wide[:, : HEADS * HEAD_DIM].unflatten(-1, (HEADS, HEAD_DIM))
        self.assertNotEqual(gate.stride(0), HEADS * HEAD_DIM)
        got, got_scale = self._run(x, gate)
        want, want_scale = _ref(x, self.weight, gate, self.dtype)
        torch.testing.assert_close(got_scale, want_scale, rtol=2e-2, atol=0)
        denom = (want.float() * want_scale).abs().amax().clamp_min(1e-6)
        err = ((got.float() * got_scale) - (want.float() * want_scale)).abs().amax()
        self.assertLess((err / denom).item(), 5e-2)


if __name__ == "__main__":
    unittest.main()
