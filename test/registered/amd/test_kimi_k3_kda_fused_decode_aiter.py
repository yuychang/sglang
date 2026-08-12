"""Numerical check for the AITER gfx950 fused Kimi-K3 KDA decode.

The fused kernel replaces a four-step decode chain (f_b projection, width-4
causal conv1d update, delta-rule recurrence, gated output RMSNorm) with one
launch. This drives both paths from identical inputs and compares the output
*and* both mutated caches, since the kernel updates conv/SSM state in place.

Requires gfx950 and an AITER build carrying
``aiter.ops.flydsl.kimi_k3_kda_decode``. Skips otherwise.
"""

import os
import unittest

import torch

os.environ.setdefault("SGLANG_K3_KDA_FUSED_BACKEND", "aiter")

from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip as aiter_kda
from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.kimi_k3 import kimi_k3_tiny_gemm
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

HEADS = 12
DIM = 128
CHANNELS = 3 * HEADS * DIM
NORM_EPS = 1e-5
LOWER_BOUND = -5.0
# Slot 0 is the reserved padding slot; the kernel skips non-positive indices.
SLOTS = 8


def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    if not aiter_kda.available(torch.device("cuda", 0)):
        return "AITER fused KDA decode unavailable on this device/build"
    return None


class TestKimiK3KDAFusedDecodeAiter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reason = _skip_reason()
        if reason:
            raise unittest.SkipTest(reason)
        cls.device = torch.device("cuda", 0)

    def _inputs(self, batch, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)
        dev, bf16, f32 = self.device, torch.bfloat16, torch.float32

        def rn(*shape, dtype=bf16, scale=1.0):
            return (torch.randn(*shape, generator=g, device=dev, dtype=f32) * scale).to(
                dtype
            )

        return dict(
            f_a=rn(batch, DIM),
            # The model reshapes f_b_proj.weight [1536, 128] -> [12, 128, 128].
            f_b_weight=rn(HEADS * DIM, DIM, scale=0.05).view(HEADS, DIM, DIM),
            mixed_qkv=rn(batch, CHANNELS),
            conv_weight=rn(CHANNELS, 4, dtype=f32, scale=0.2),
            # Pool layout is [slots, 3, CHANNELS]; both paths take the
            # transposed [slots, CHANNELS, 3] view of it.
            conv_states=rn(SLOTS, 3, CHANNELS, scale=0.3),
            raw_beta=rn(1, batch, HEADS),
            A_log=rn(HEADS, dtype=f32, scale=0.5),
            dt_bias=rn(HEADS * DIM, dtype=f32, scale=0.5),
            ssm_states=rn(SLOTS, HEADS, DIM, DIM, dtype=f32, scale=0.1),
            # Distinct non-zero slots: slot 0 is reserved padding.
            state_indices=torch.arange(1, batch + 1, device=dev, dtype=torch.int32),
            output_gate=rn(batch, HEADS, DIM),
            norm_weight=rn(DIM),
        )

    def _reference(self, t):
        """f_b GEMM -> conv1d update -> packed KDA recurrence -> gated RMSNorm."""
        batch = t["f_a"].shape[0]
        conv_states = t["conv_states"].clone()
        ssm_states = t["ssm_states"].clone()

        a = kimi_k3_tiny_gemm(t["f_a"], t["f_b_weight"].reshape(HEADS * DIM, DIM))
        qkv = causal_conv1d_update(
            t["mixed_qkv"],
            conv_states.transpose(-1, -2),
            t["conv_weight"],
            None,
            activation="silu",
            conv_state_indices=t["state_indices"],
        )

        out = qkv.new_empty(batch, 1, HEADS, DIM)
        fused_recurrent_kda_packed_decode(
            mixed_qkv=qkv,
            a=a,
            b=t["raw_beta"].reshape(batch, HEADS),
            A_log=t["A_log"],
            dt_bias=t["dt_bias"],
            scale=DIM**-0.5,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=t["state_indices"],
            use_qk_l2norm_in_kernel=True,
            lower_bound=LOWER_BOUND,
        )
        core = out.transpose(0, 1)  # [1, B, H, D]

        norm = FusedRMSNormGated(
            DIM, eps=NORM_EPS, activation="sigmoid", device=self.device
        )
        norm.weight.data.copy_(t["norm_weight"].to(norm.weight.dtype))
        gated = norm(core, t["output_gate"].unsqueeze(0))
        return gated, conv_states, ssm_states

    def _fused(self, t):
        conv_states = t["conv_states"].clone()
        ssm_states = t["ssm_states"].clone()
        conv_view = conv_states.transpose(-1, -2)

        self.assertTrue(
            aiter_kda.covered(
                t["f_a"],
                t["f_b_weight"],
                t["mixed_qkv"],
                t["raw_beta"],
                conv_view,
                ssm_states,
                t["state_indices"],
                t["output_gate"],
                t["norm_weight"],
            ),
            "covered() rejected inputs that match the K3 decode layout",
        )
        out = aiter_kda.run(
            f_a=t["f_a"],
            f_b_weight=t["f_b_weight"],
            mixed_qkv=t["mixed_qkv"],
            conv_weight=t["conv_weight"],
            conv_state=conv_view,
            raw_beta=t["raw_beta"],
            A_log=t["A_log"],
            dt_bias=t["dt_bias"],
            lower_bound=LOWER_BOUND,
            state=ssm_states,
            state_indices=t["state_indices"],
            output_gate=t["output_gate"],
            norm_weight=t["norm_weight"],
            norm_eps=NORM_EPS,
        )
        return out, conv_states, ssm_states

    def _assert_close(self, got, want, name, rtol, atol):
        got, want = got.float(), want.float()
        err = (got - want).abs()
        denom = want.abs().clamp_min(1e-3)
        self.assertTrue(
            torch.allclose(got, want, rtol=rtol, atol=atol),
            f"{name} mismatch: max abs {err.max():.5f}, "
            f"max rel {(err / denom).max():.5f}, mean abs {err.mean():.6f}",
        )

    def test_matches_reference_chain(self):
        for batch in (1, 2, 4, 7):
            with self.subTest(batch=batch):
                t = self._inputs(batch, seed=1000 + batch)
                ref_out, ref_conv, ref_ssm = self._reference(t)
                got_out, got_conv, got_ssm = self._fused(t)

                self.assertEqual(tuple(got_out.shape), (1, batch, HEADS, DIM))
                self.assertEqual(got_out.dtype, torch.bfloat16)
                # bf16 output of a long fp32 recurrence: 2e-2 is ~1 ulp at
                # magnitude 1, which is the scale these activations live at.
                self._assert_close(got_out, ref_out, "output", 2e-2, 2e-2)
                self._assert_close(got_conv, ref_conv, "conv_state", 1e-2, 1e-2)
                self._assert_close(got_ssm, ref_ssm, "ssm_state", 2e-2, 2e-2)

    def test_reserved_slot_zero_is_inert(self):
        """Non-positive state_indices must emit zeros and touch no cache."""
        t = self._inputs(2, seed=77)
        t["state_indices"] = torch.zeros(2, device=self.device, dtype=torch.int32)
        out, conv_states, ssm_states = self._fused(t)

        self.assertTrue(torch.all(out == 0), "padding rows produced nonzero output")
        self.assertTrue(torch.equal(conv_states, t["conv_states"]))
        self.assertTrue(torch.equal(ssm_states, t["ssm_states"]))

    def test_out_kwarg_writes_in_place(self):
        t = self._inputs(3, seed=42)
        dest = torch.empty((1, 3, HEADS, DIM), dtype=torch.bfloat16, device=self.device)
        conv_view = t["conv_states"].clone().transpose(-1, -2)
        got = aiter_kda.run(
            f_a=t["f_a"],
            f_b_weight=t["f_b_weight"],
            mixed_qkv=t["mixed_qkv"],
            conv_weight=t["conv_weight"],
            conv_state=conv_view,
            raw_beta=t["raw_beta"],
            A_log=t["A_log"],
            dt_bias=t["dt_bias"],
            lower_bound=LOWER_BOUND,
            state=t["ssm_states"].clone(),
            state_indices=t["state_indices"],
            output_gate=t["output_gate"],
            norm_weight=t["norm_weight"],
            norm_eps=NORM_EPS,
            out=dest,
        )
        self.assertIs(got, dest)


if __name__ == "__main__":
    unittest.main()
