"""Correctness tests for Kimi-K3 ATOM-aligned PTPC FP8."""

import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-small-amd-mi35x")


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3PtpcFp8(CustomTestCase):
    def test_quantize_linear_n_padding(self):
        from sglang.kernels.ops.kimi_k3.ptpc_fp8 import (
            K3PtpcFp8LinearMethod,
            quantize_linear_weight_ptpc,
        )

        mod = torch.nn.Linear(7168, 96, bias=False).cuda().bfloat16()
        assert quantize_linear_weight_ptpc(mod)
        assert mod.weight.dtype == torch.float8_e4m3fn
        assert mod.weight.shape[0] == 128
        assert mod._k3_ptpc_orig_out_features == 96
        assert isinstance(mod.quant_method, K3PtpcFp8LinearMethod)
        # Padding rows encode exact zero with a benign scale.
        torch.testing.assert_close(
            mod.weight_scale[96:], torch.ones_like(mod.weight_scale[96:])
        )

    def test_padded_apply_returns_packed_tensor(self):
        """A padded layer must hand back a packed tensor, not a strided view.

        Kimi-K3 splits fused_qkv_a_proj_with_mqa's output into q/kv/rope and
        passes the pieces to kernels that index by row width. When apply()
        returned out[..., :orig_n] the row stride was still the padded width, so
        those kernels read across the pad and the model emitted fluent garbage
        rather than failing.
        """
        from sglang.kernels.ops.kimi_k3.ptpc_fp8 import quantize_linear_weight_ptpc

        torch.manual_seed(11)
        # 2112 is K3's real fused_qkv_a width and pads to 2176.
        mod = torch.nn.Linear(7168, 2112, bias=False).cuda().bfloat16()
        assert quantize_linear_weight_ptpc(mod)
        assert mod._k3_ptpc_orig_out_features == 2112

        x = torch.randn(8, 7168, device="cuda", dtype=torch.bfloat16)
        out = mod.quant_method.apply(mod, x)
        assert out.shape == (8, 2112)
        assert out.is_contiguous(), "padded PTPC output must be repacked"
        assert out.stride(0) == 2112

    def test_weight_quant_matches_aiter_per_token(self):
        """PTPC weights must be byte-identical to ATOM's AITER quantizer."""
        from aiter import QuantType, get_hip_quant

        from sglang.kernels.ops.kimi_k3.ptpc_fp8 import (
            _quantize_weight_rows,
            k3_ptpc_fp8_dtype,
        )

        torch.manual_seed(19)
        weight = (
            torch.randn(96, 7168, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        dtype = k3_ptpc_fp8_dtype()
        expected_q, expected_scale = get_hip_quant(QuantType.per_Token)(
            weight, quant_dtype=dtype
        )
        actual_q, actual_scale = _quantize_weight_rows(weight, dtype, pad_n=96)
        assert torch.equal(actual_q, expected_q)
        assert torch.equal(actual_scale, expected_scale)

    def test_mla_input_quant_is_shared_by_both_ptpc_consumers(self):
        from aiter import dtypes, per_token_quant_hip

        from sglang.srt.models.kimi_k3 import KimiK3MLAAttention

        attn = KimiK3MLAAttention.__new__(KimiK3MLAAttention)
        torch.nn.Module.__init__(attn)
        attn.use_output_gate = True
        attn.fused_qkv_a_proj_with_mqa = types.SimpleNamespace(
            _k3_ptpc_per_token=True
        )
        attn.g_proj = types.SimpleNamespace(_k3_ptpc_per_token=True)

        torch.manual_seed(23)
        x = torch.randn(8, 7168, device="cuda", dtype=torch.bfloat16)
        actual = attn.maybe_quantize_ptpc_input(x)
        expected = per_token_quant_hip(x, quant_dtype=dtypes.fp8)
        assert isinstance(actual, tuple)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])

        # A pre-quantized tuple is handed through, not quantized twice.
        assert attn.maybe_quantize_ptpc_input(actual) is actual

    def test_ptpc_linear_bf16_and_prequantized_inputs(self):
        import aiter

        from sglang.kernels.ops.kimi_k3.ptpc_fp8 import (
            quantize_linear_weight_ptpc,
        )

        torch.manual_seed(3)
        # Use a production K3 shape; tiny synthetic N/K combinations are not
        # necessarily instantiated in AITER's CK kernel library.
        mod = torch.nn.Linear(7168, 896, bias=False).cuda().bfloat16()
        weight = mod.weight.detach().clone()
        x = torch.randn(16, 7168, device="cuda", dtype=torch.bfloat16)
        reference = F.linear(x.float(), weight.float()).bfloat16()

        self.assertTrue(quantize_linear_weight_ptpc(mod))
        out = mod.quant_method.apply(mod, x)
        self.assertEqual(tuple(out.shape), (16, 896))
        torch.testing.assert_close(out, reference, rtol=0.12, atol=0.35)

        qx, sx = aiter.per_token_quant_hip(x, quant_dtype=aiter.dtypes.fp8)
        tuple_out = mod.quant_method.apply(mod, (qx, sx))
        torch.testing.assert_close(tuple_out, out, rtol=0, atol=0)

    def test_misaligned_k_fails_closed(self):
        from sglang.kernels.ops.kimi_k3.ptpc_fp8 import (
            quantize_linear_weight_ptpc,
        )

        mod = torch.nn.Linear(33, 64, bias=False).cuda().bfloat16()
        original = mod.weight.detach().clone()
        self.assertFalse(quantize_linear_weight_ptpc(mod))
        self.assertEqual(mod.weight.dtype, torch.bfloat16)
        torch.testing.assert_close(mod.weight, original)

    def test_full_row_scale_differs_from_local_shard_scale(self):
        from sglang.kernels.ops.kimi_k3.ptpc_fp8 import (
            _quantize_weight_rows,
            k3_ptpc_fp8_dtype,
        )

        # Emulates a row-parallel matrix whose second TP shard has the global
        # amax. Every rank must receive the full-row scale, not a local one.
        left = torch.full((8, 32), 0.25, device="cuda", dtype=torch.bfloat16)
        right = torch.full((8, 32), 8.0, device="cuda", dtype=torch.bfloat16)
        full = torch.cat((left, right), dim=1)
        _, global_scale = _quantize_weight_rows(
            full, k3_ptpc_fp8_dtype(), pad_n=128
        )
        _, local_scale = _quantize_weight_rows(
            left, k3_ptpc_fp8_dtype(), pad_n=128
        )
        self.assertTrue(torch.all(global_scale[:8] > local_scale[:8]))

    def test_layer_coverage_matches_atom_exclude_list(self):
        from sglang.kernels.ops.kimi_k3 import ptpc_fp8

        def named(name):
            return types.SimpleNamespace(name=name)

        kda = types.SimpleNamespace(
            fused_qkvg_proj=named("qkvg"),
            _qkvgbfa_layer=named("fused_inproj"),
            f_a_proj=named("f_a"),
            b_proj=named("b"),
            f_b_proj=named("f_b"),
            o_proj=named("o"),
        )
        moe = types.SimpleNamespace(
            experts=named("experts"),
            shared_experts=types.SimpleNamespace(
                gate_up_proj=named("shared_up"),
                down_proj=named("shared_down"),
            ),
            gate=named("router"),
        )
        layer = types.SimpleNamespace(self_attn=kda, mlp=moe)
        calls = []
        with mock.patch.object(
            ptpc_fp8,
            "quantize_linear_weight_ptpc",
            side_effect=lambda module: calls.append(module.name) or True,
        ):
            count = ptpc_fp8.quantize_k3_dense_linears_in_layer(layer)
        self.assertEqual(count, 5)
        self.assertEqual(
            calls, ["fused_inproj", "f_b", "o", "shared_up", "shared_down"]
        )

        split_kda = types.SimpleNamespace(
            fused_qkvg_proj=named("qkvg"),
            f_a_proj=named("f_a"),
            b_proj=named("b"),
            f_b_proj=named("f_b"),
            o_proj=named("o"),
        )
        layer = types.SimpleNamespace(self_attn=split_kda, mlp=moe)
        calls = []
        with mock.patch.object(
            ptpc_fp8,
            "quantize_linear_weight_ptpc",
            side_effect=lambda module: calls.append(module.name) or True,
        ):
            count = ptpc_fp8.quantize_k3_dense_linears_in_layer(layer)
        self.assertEqual(count, 7)
        self.assertEqual(
            calls, ["qkvg", "f_a", "b", "f_b", "o", "shared_up", "shared_down"]
        )

        mla = types.SimpleNamespace(
            fused_qkv_a_proj_with_mqa=named("qkv_a"),
            q_b_proj=named("q_b"),
            o_proj=named("mla_o"),
            g_proj=named("g"),
            use_output_gate=True,
            w_kc=object(),
        )
        dense = types.SimpleNamespace(
            gate_up_proj=named("dense_up"),
            down_proj=named("dense_down"),
        )
        layer = types.SimpleNamespace(self_attn=mla, mlp=dense)
        calls = []
        with mock.patch.object(
            ptpc_fp8,
            "quantize_linear_weight_ptpc",
            side_effect=lambda module: calls.append(module.name) or True,
        ):
            count = ptpc_fp8.quantize_k3_dense_linears_in_layer(layer)
        self.assertEqual(count, 6)
        self.assertEqual(
            calls, ["qkv_a", "q_b", "mla_o", "g", "dense_up", "dense_down"]
        )

    def test_rmsnorm_fp8_per_token(self):
        from sglang.kernels.ops.kimi_k3.rmsnorm_fp8_quant import rmsnorm_fp8_per_token

        torch.manual_seed(7)
        x = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
        out, scale = rmsnorm_fp8_per_token(x, w, 1e-5)
        assert out.shape == (4, 7168)
        assert out.dtype == torch.float8_e4m3fn
        assert scale.shape == (4, 1)
        reference = F.rms_norm(x.float(), (7168,), w.float(), 1e-5)
        dequantized = out.float() * scale
        torch.testing.assert_close(dequantized, reference, rtol=0.08, atol=0.08)

    def test_rmsnorm_fp8_residual(self):
        from sglang.kernels.ops.kimi_k3.rmsnorm_fp8_quant import rmsnorm_fp8_per_token

        x = torch.randn(3, 7168, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        w = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
        (out, scale), residual_out = rmsnorm_fp8_per_token(
            x, w, 1e-5, residual=residual
        )
        reference_residual = (x.float() + residual.float()).bfloat16()
        reference = F.rms_norm(
            reference_residual.float(), (7168,), w.float(), 1e-5
        )
        torch.testing.assert_close(
            residual_out, reference_residual, rtol=0, atol=0
        )
        torch.testing.assert_close(
            out.float() * scale, reference, rtol=0.08, atol=0.08
        )
