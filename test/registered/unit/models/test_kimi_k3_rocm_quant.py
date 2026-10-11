"""Unit tests for srt/models/kimi_k3_rocm_quant and the K3 weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8
from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable
from sglang.srt.models.kimi_k3_rocm_quant import (
    _k3_channel_fp8_to_bf16,
    _k3_merge_kda_inproj_fp8,
    k3_absorb_kv_b_rocm,
    k3_per_batched_tensor_fp8,
)
from sglang.test.test_utils import CustomTestCase


def _per_channel_fp8(out_features: int, in_features: int):
    torch.manual_seed(0)
    dense = torch.randn(out_features, in_features, dtype=torch.float32)
    # Spread the per-channel magnitudes so a wrongly-broadcast scale cannot
    # coincidentally reproduce the right answer.
    dense *= torch.logspace(-1, 1, out_features).unsqueeze(1)
    scale = dense.abs().amax(dim=1, keepdim=True) / 448.0
    weight = (dense / scale).to(torch.float8_e4m3fn)
    return weight, scale, dense


class TestChannelFp8ToBf16(CustomTestCase):
    """aiter's batched absorb GEMM takes ``w_scale`` as a scalar, so a
    per-channel vector cannot reach it. The scale axis is the GEMM's
    contraction axis, so requantizing to one per-tensor scale is lossy --
    ~2.3% relative error on real K3 weights. Dequantizing to bf16 costs ~0.12%
    and lets the absorb use its bf16 path with w_scale at the 1.0 default."""

    def test_dequantizes_every_channel_with_its_own_scale(self):
        weight, scale, dense = _per_channel_fp8(out_features=8, in_features=16)
        module = SimpleNamespace(weight_scale=scale.squeeze(1))

        out = _k3_channel_fp8_to_bf16(module, weight)

        self.assertEqual(out.dtype, torch.bfloat16)
        # Only the original fp8 rounding plus the bf16 cast separate this from
        # the checkpoint's own values; no second quantization.
        exact = weight.to(torch.float32) * scale
        torch.testing.assert_close(out.float(), exact, rtol=0.01, atol=0.0)
        # Guard the failure this replaced: one shared scale would leave the
        # low-magnitude channels far from their true values.
        worst = ((out.float() - dense).abs().sum(1) / dense.abs().sum(1)).max()
        self.assertLess(worst.item(), 0.05)

    def test_accepts_a_column_vector_scale(self):
        """Quark serializes the scale as [out]; callers may hold [out, 1]."""
        weight, scale, _ = _per_channel_fp8(out_features=8, in_features=16)

        flat = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.squeeze(1)), weight
        )
        column = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.clone()), weight
        )

        torch.testing.assert_close(flat, column)


def _vllm_per_batched_tensor_quant(x: torch.Tensor, dtype: torch.dtype):
    # vllm/model_executor/layers/attention/mla_attention.py
    dtype_max = torch.finfo(dtype).max
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-10)
    scale = dtype_max / amax
    x_scl_sat = (x * scale).clamp(min=-dtype_max, max=dtype_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def _fp8_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class TestPerBatchedTensorFp8(CustomTestCase):
    """The FP8BMM absorb quantizes W_K and W_V like vLLM ROCm."""

    def test_matches_vllm_formula_bitwise(self):
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

        torch.manual_seed(0)
        x = (torch.randn(4, 32, 16) * 0.03).to(torch.bfloat16)
        q, s = k3_per_batched_tensor_fp8(x)
        q_ref, s_ref = _vllm_per_batched_tensor_quant(x, fp8_dtype)
        self.assertEqual(q.dtype, fp8_dtype)
        self.assertTrue(_fp8_equal(q, q_ref))
        self.assertTrue(torch.equal(s, s_ref))
        self.assertEqual(s.dtype, torch.float32)
        self.assertEqual(s.numel(), 1)


    @staticmethod
    def _mla(heads=2, nope=4, v=3, lora=8):
        weight, scale, _ = _per_channel_fp8(
            out_features=heads * (nope + v), in_features=lora
        )
        attn = SimpleNamespace(
            kv_b_proj=SimpleNamespace(weight_scale=scale.squeeze(1)),
            qk_nope_head_dim=nope,
            v_head_dim=v,
        )
        return attn, weight

    def test_absorb_splits_with_separate_tensor_scales(self):
        attn, weight = self._mla()
        with envs.SGLANG_USE_AITER.override(True):
            with envs.SGLANG_ROCM_USE_AITER_FP8BMM.override(True):
                self.assertTrue(k3_absorb_kv_b_rocm(attn, weight))

        dense = _k3_channel_fp8_to_bf16(attn.kv_b_proj, weight).unflatten(0, (2, 7))
        k_ref, ks_ref = k3_per_batched_tensor_fp8(dense[:, :4].transpose(1, 2))
        v_ref, vs_ref = k3_per_batched_tensor_fp8(dense[:, 4:])
        # w_kc is an [N, P, L] view of W_K [N, L, P]; w_vc an [N, L, V] view of
        # W_V [N, V, L], the layouts AITER's batched GEMM reads.
        self.assertEqual(tuple(attn.w_kc.shape), (2, 4, 8))
        self.assertEqual(tuple(attn.w_vc.shape), (2, 8, 3))
        self.assertTrue(attn.w_kc.transpose(1, 2).is_contiguous())
        self.assertTrue(attn.w_vc.transpose(1, 2).is_contiguous())
        self.assertTrue(_fp8_equal(attn.w_kc.transpose(1, 2), k_ref))
        self.assertTrue(_fp8_equal(attn.w_vc.transpose(1, 2), v_ref))
        self.assertTrue(torch.equal(attn.w_kc_tensor_scale, ks_ref))
        self.assertTrue(torch.equal(attn.w_vc_tensor_scale, vs_ref))

    def test_fp8bmm_off_keeps_bf16_absorb(self):
        attn, weight = self._mla()
        with envs.SGLANG_ROCM_USE_AITER_FP8BMM.override(False):
            self.assertTrue(k3_absorb_kv_b_rocm(attn, weight))
        self.assertEqual(attn.w_kc.dtype, torch.bfloat16)
        self.assertFalse(hasattr(attn, "w_kc_tensor_scale"))


class TestMergeDtypeGuard(CustomTestCase):
    """``_merge_weights_as_views`` cats ``.weight`` alone, so fusing a quantized
    projection silently drops the scale tensors that give its integer payload
    meaning. Only dtypes that carry their full value in ``.weight`` may fuse."""

    def test_rejects_weights_that_carry_a_separate_scale(self):
        for dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4).to(dtype) for _ in range(2)]
                self.assertFalse(_is_unquantized_mergeable(weights))

    def test_accepts_plain_float_weights(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4, dtype=dtype) for _ in range(2)]
                self.assertTrue(_is_unquantized_mergeable(weights))

    def test_rejects_mixed_dtypes(self):
        weights = [
            torch.zeros(4, 4, dtype=torch.bfloat16),
            torch.zeros(4, 4, dtype=torch.float16),
        ]
        self.assertFalse(_is_unquantized_mergeable(weights))

    def test_rejects_float32(self):
        """The merged buffer is only built for the two half dtypes the fused
        KDA/MoE kernels read; fp32 must not slip in through a dtype-agreement
        check that only asks whether the operands match each other."""
        weights = [torch.zeros(4, 4, dtype=torch.float32) for _ in range(2)]
        self.assertFalse(_is_unquantized_mergeable(weights))


def _kda_attn(bf16_b_proj: bool = False):
    scheme = QuarkW8A8Fp8(
        {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": True}
    )
    scheme.out_dtype = torch.bfloat16

    def linear(out_features, in_features=16):
        weight, scale, _ = _per_channel_fp8(out_features, in_features)
        return SimpleNamespace(
            weight=weight, weight_scale=scale.squeeze(1), scheme=scheme
        )

    b_proj = linear(3)
    if bf16_b_proj:
        b_proj = SimpleNamespace(weight=torch.zeros(3, 16, dtype=torch.bfloat16))
    return SimpleNamespace(
        use_full_rank_gate=True,
        split_sizes=[24, 8],
        fused_qkvg_proj=linear(32),
        f_a_proj=linear(8),
        b_proj=b_proj,
        f_b_proj=linear(16, in_features=8),
    )


def _dequant(module):
    return module.weight.float() * module.weight_scale.view(-1, 1)


@mock.patch(
    "sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8."
    "use_aiter_bpreshuffle_gemm",
    return_value=False,
)
class TestMergeKdaInprojFp8(CustomTestCase):
    """Quark FP8 KDA in-proj merge: one per-channel FP8 linear for [qkvg|f_a|b]."""

    def test_merged_weight_keeps_every_channel_scale(self, _):
        attn = _kda_attn()
        expected = torch.cat(
            [_dequant(m) for m in (attn.fused_qkvg_proj, attn.f_a_proj, attn.b_proj)]
        )

        self.assertTrue(_k3_merge_kda_inproj_fp8(attn))

        layer = attn._qkvgbfa_layer
        # Processed like the Quark linear: [in, out] weight, [out, 1] scale.
        merged = layer.weight.t().float() * layer.weight_scale.view(-1, 1)
        self.assertEqual(merged.shape[0] % 64, 0)
        torch.testing.assert_close(merged[: expected.shape[0]], expected)
        self.assertEqual(merged[expected.shape[0] :].abs().max().item(), 0.0)
        self.assertEqual(attn._qkvgbfa_sizes, [24, 8, 8, 3, merged.shape[0] - 43])

    def test_f_b_stays_on_its_fp8_linear(self, _):
        # As in vLLM, f_b keeps its own FP8 linear rather than a BF16 copy.
        attn = _kda_attn()

        self.assertTrue(_k3_merge_kda_inproj_fp8(attn))

        self.assertIsNone(getattr(attn, "_bfa_f_b_w", None))

    def test_skips_non_fp8_projection(self, _):
        self.assertFalse(_k3_merge_kda_inproj_fp8(_kda_attn(bf16_b_proj=True)))

    def test_env_disables_merge(self, _):
        with envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.override(False):
            self.assertFalse(_k3_merge_kda_inproj_fp8(_kda_attn()))


if __name__ == "__main__":
    unittest.main()
