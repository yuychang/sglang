"""Unit tests for srt/models/kimi_k3_rocm_{quant,moe} and the K3 weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import contextlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip
from sglang.srt.environ import envs
from sglang.srt.layers.quantization import unquant
from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.models import kimi_k3, kimi_k3_rocm_quant
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention, _is_unquantized_mergeable
from sglang.srt.models.kimi_k3_rocm_moe import (
    _first_untuned_tokens,
    build_tuned_front_gemm,
)
from sglang.srt.models.kimi_k3_rocm_quant import (
    _k3_channel_fp8_to_bf16,
    k3_fused_decode_f_b_weight,
    k3_kda_inproj_fp8_tail_rows,
    k3_kv_b_for_absorb,
    k3_merge_kda_inproj_fp8,
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
    """The BF16 kv_b of the MLA absorb (off gfx950 + AITER, or with
    SGLANG_OPT_HIP_K3_MLA_ABSORB_FP8=0) and the merged in-proj's f_b: each channel
    dequantized with its own scale, ~0.12% error on real K3 weights."""

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


class TestFusedDecodeFbWeight(CustomTestCase):
    """The fused KDA decode takes a dense f_b with no scale argument,
    so a Quark per-channel FP8 f_b must reach it with every channel's scale applied."""

    def test_quark_fp8_f_b_is_dequantized_and_bf16_passes_through(self):
        weight, scale, _ = _per_channel_fp8(out_features=16, in_features=8)
        quark_f_b = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale=scale.squeeze(1),
            quant_method=mock.Mock(spec=QuarkLinearMethod),
            scheme=QuarkW8A8Fp8(
                {"qscheme": "per_channel"},
                {"qscheme": "per_channel", "is_dynamic": True},
            ),
        )

        out = k3_fused_decode_f_b_weight(quark_f_b)

        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(
            out.float(), weight.float() * scale, rtol=0.01, atol=0.0
        )
        # The FP8 layer keeps serving the unfused path.
        self.assertEqual(quark_f_b.weight.dtype, torch.float8_e4m3fn)

        bf16 = torch.nn.Parameter(
            torch.randn(16, 8, dtype=torch.bfloat16), requires_grad=False
        )
        plain_f_b = SimpleNamespace(weight=bf16, quant_method=mock.Mock())
        self.assertEqual(
            k3_fused_decode_f_b_weight(plain_f_b).data_ptr(), bf16.data_ptr()
        )

    def test_post_processed_quark_f_b_disables_fused_decode_instead_of_crashing(self):
        """Sharded and remote loaders transpose the FP8 f_b before post_load_weights;
        that weight must not crash the load."""
        weight, scale, _ = _per_channel_fp8(out_features=32, in_features=16)
        scheme = QuarkW8A8Fp8(
            {"qscheme": "per_channel"},
            {"qscheme": "per_channel", "is_dynamic": True},
        )
        f_b = torch.nn.Module()
        f_b.weight = torch.nn.Parameter(weight, requires_grad=False)
        f_b.weight_scale = torch.nn.Parameter(scale.squeeze(1), requires_grad=False)
        f_b.quant_method = mock.Mock(spec=QuarkLinearMethod)
        f_b.scheme = scheme
        scheme.process_weights_after_loading(f_b)

        with mock.patch.object(kimi_k3_rocm_quant, "print_warning_once") as warn:
            self.assertIsNone(k3_fused_decode_f_b_weight(f_b))
        warn.assert_called_once()


class TestHipFusedDecodeArming(CustomTestCase):
    """_prepare_fused_decode on ROCm, on a stand-in attention module."""

    def test_failed_rearm_drops_the_previous_arming(self):
        """A re-run that cannot arm (here the backend reports unavailable) must not
        leave the previous f_b serving the fused decode."""
        layer = SimpleNamespace(
            conv_weights=None,
            _k3_hip_fused_decode_args=("stale f_b", None, 1e-5, None),
            _k3_hip_fused_decode_backend="aiter",
            _k3_hip_fused_decode_ops=kda_fused_decode_aiter_hip,
        )
        attention = SimpleNamespace(
            attn=layer,
            _kda_hip_fused_decode_ready=True,
            f_b_proj=SimpleNamespace(weight=torch.empty(12 * 128, 128)),
            config=SimpleNamespace(
                mamba2_cache_params=SimpleNamespace(
                    dtype=SimpleNamespace(temporal=torch.bfloat16)
                )
            ),
        )

        with (
            mock.patch.object(kimi_k3, "_is_hip", True),
            envs.SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
            mock.patch.object(
                kda_fused_decode_aiter_hip, "available", return_value=False
            ),
        ):
            KimiK3DeltaAttention._prepare_fused_decode(attention)

        self.assertFalse(attention._kda_hip_fused_decode_ready)
        self.assertIsNone(layer._k3_hip_fused_decode_args)
        self.assertIsNone(layer._k3_hip_fused_decode_ops)

    def test_block_fp8_f_b_arms_from_the_merged_bf16_copy(self):
        """A block-FP8 f_b_proj holds FP8 with no Quark scheme; the fused decode must
        take the BF16 copy _merge_bfa_weights() dequantized, and warn when none exists."""
        merged = torch.randn(12 * 128, 128, dtype=torch.bfloat16)
        attention = SimpleNamespace(
            use_full_rank_gate=True,
            _bfa_f_b_w=merged,
            f_b_proj=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.zeros(12 * 128, 128, dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                ),
                quant_method=mock.Mock(),
            ),
        )
        self.assertIs(KimiK3DeltaAttention._fused_decode_f_b_weight(attention), merged)

        attention._bfa_f_b_w = None
        with mock.patch.object(kimi_k3, "print_warning_once") as warn:
            self.assertIsNone(KimiK3DeltaAttention._fused_decode_f_b_weight(attention))
        warn.assert_called_once()


def _shaped(rows: int, cols: int, dtype=torch.bfloat16) -> torch.Tensor:
    # Only shape and dtype are read; expand keeps the 6016x7168 front at one element.
    return torch.zeros(1, dtype=dtype).expand(rows, cols)


def _fake_aiter(*, gfx: str = "gfx950", cu_num: int = 256, rows_up_to=192) -> dict:
    tuned_gemm = types.ModuleType("aiter.tuned_gemm")
    tuned_gemm.tgemm = SimpleNamespace(mm=lambda **kwargs: None)
    tuned_gemm.get_GEMM_A16W16_tuned_config = lambda *, M, **kwargs: (
        {"libtype": "flydsl"} if M <= rows_up_to else None
    )
    chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx = lambda: gfx
    chip_info.get_cu_num = lambda: cu_num
    return {
        "aiter": types.ModuleType("aiter"),
        "aiter.tuned_gemm": tuned_gemm,
        "aiter.jit": types.ModuleType("aiter.jit"),
        "aiter.jit.utils": types.ModuleType("aiter.jit.utils"),
        "aiter.jit.utils.chip_info": chip_info,
    }


class TestTunedFrontGemm(CustomTestCase):
    """AITER has tuned rows for one bf16 front shape (TP8) on gfx950 only;
    the builder declines every other target, and a broken AITER fails the load."""

    def _build(
        self,
        *,
        use_aiter=True,
        enabled=True,
        weight=None,
        aiter=None,
        window=(48, 192),
        deterministic=False,
    ):
        _first_untuned_tokens.cache_clear()
        with (
            patch.dict(sys.modules, _fake_aiter() if aiter is None else aiter),
            patch.object(unquant, "_skip_aiter_unquant", deterministic),
            envs.SGLANG_USE_AITER.override(use_aiter),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT.override(enabled),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MIN_TOKENS.override(window[0]),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MAX_TOKENS.override(window[1]),
        ):
            return build_tuned_front_gemm(
                weight=_shaped(6016, 7168) if weight is None else weight
            )

    def test_covers_the_inclusive_env_window(self):
        tuned = self._build()
        for num_tokens, covered in ((47, False), (48, True), (192, True), (193, False)):
            with self.subTest(num_tokens=num_tokens):
                self.assertIs(tuned.covers(num_tokens), covered)

    def test_declines_outside_the_tuned_target(self):
        no_aiter = {"aiter": None, "aiter.tuned_gemm": None}
        for name, kwargs in (
            ("SGLANG_USE_AITER off", dict(use_aiter=False, aiter=no_aiter)),
            ("switch off", dict(enabled=False, aiter=no_aiter)),
            # Shape and dtype declines need no AITER, so they hold even without tgemm.
            ("TP4 front shape", dict(weight=_shaped(7552, 7168), aiter=no_aiter)),
            (
                "fp16 front",
                dict(weight=_shaped(6016, 7168, torch.float16), aiter=no_aiter),
            ),
            ("gfx942", dict(aiter=_fake_aiter(gfx="gfx942"))),
            # A partitioned gfx950 has fewer CUs, and AITER keys rows on the CU count.
            ("gfx950 with 32 CUs", dict(aiter=_fake_aiter(cu_num=32))),
            # AITER picks the kernel and split-K by batch size.
            ("deterministic inference", dict(deterministic=True, aiter=no_aiter)),
        ):
            with self.subTest(name):
                self.assertIsNone(self._build(**kwargs))

    def test_fails_at_load_on_an_aiter_without_tgemm_or_its_rows(self):
        without_tgemm = {**_fake_aiter(), "aiter.tuned_gemm": None}
        with self.assertRaises(ImportError):
            self._build(aiter=without_tgemm)
        without_probe = _fake_aiter()
        del without_probe["aiter.tuned_gemm"].get_GEMM_A16W16_tuned_config
        with self.assertRaisesRegex(ImportError, "get_GEMM_A16W16_tuned_config"):
            self._build(aiter=without_probe)
        with self.assertRaises(RuntimeError):
            self._build(aiter=_fake_aiter(rows_up_to=0))

    def test_fails_at_load_when_the_window_runs_past_the_tuned_rows(self):
        """tgemm runs an untuned default kernel for a batch without a row, so a
        window that only its smallest batch probes would pass a partial AITER."""
        with self.assertRaisesRegex(RuntimeError, "M=193"):
            self._build(window=(1, 256))
        with self.assertRaisesRegex(RuntimeError, "M=65"):
            self._build(aiter=_fake_aiter(rows_up_to=64), window=(1, 192))

    def test_rejects_an_invalid_window_on_every_target(self):
        """A bad MIN/MAX fails the load even where the target then declines the tuned front."""
        no_aiter = {"aiter": None, "aiter.tuned_gemm": None}
        for name, kwargs in (
            ("tuned target", {}),
            ("TP4 front shape", dict(weight=_shaped(7552, 7168), aiter=no_aiter)),
            ("gfx942", dict(aiter=_fake_aiter(gfx="gfx942"))),
            ("deterministic inference", dict(deterministic=True, aiter=no_aiter)),
        ):
            for window in ((64, 32), (0, 128)):
                with self.subTest(name, window=window), self.assertRaises(ValueError):
                    self._build(window=window, **kwargs)


@mock.patch("sglang.srt.models.deepseek_common.utils._use_aiter_gfx95", True)
class TestKvBForAbsorb(CustomTestCase):
    """The MLA absorb BMMs take one scalar w_scale, so on gfx950 + AITER kv_b is
    requantized to per-tensor FP8; the kill switch keeps the BF16 dequant."""

    def test_default_requantizes_to_one_tensor_scale(self):
        module = _quark_fp8_linear(8)
        weight, scale = module.weight, module.weight_scale.view(-1, 1)

        q, tensor_scale = k3_kv_b_for_absorb(module=module)

        self.assertEqual((q.dtype, q.shape), (weight.dtype, weight.shape))
        self.assertEqual(tensor_scale.numel(), 1)
        # The largest channel spans the full FP8 range, so nothing clips.
        self.assertEqual(q.float().abs().max().item(), torch.finfo(q.dtype).max)
        # Error is bounded by one FP8 step at the tensor-wide scale.
        exact = weight.float() * scale
        err = (q.float() * tensor_scale - exact).abs().max()
        self.assertLess(err.item(), 0.07 * exact.abs().max().item())

    def test_kill_switch_keeps_the_bf16_dequant(self):
        module = _quark_fp8_linear(8)

        with envs.SGLANG_OPT_HIP_K3_MLA_ABSORB_FP8.override(False):
            out, tensor_scale = k3_kv_b_for_absorb(module=module)

        self.assertIsNone(tensor_scale)
        torch.testing.assert_close(out, _k3_channel_fp8_to_bf16(module, module.weight))

    def test_no_requant_where_the_absorb_has_no_fp8_kernel(self):
        """Off gfx950 + AITER the absorb dequantizes an FP8 w_kc/w_vc to BF16,
        so a requant would add its error for no speedup."""
        module = _quark_fp8_linear(8)

        with mock.patch(
            "sglang.srt.models.deepseek_common.utils._use_aiter_gfx95", False
        ):
            out, tensor_scale = k3_kv_b_for_absorb(module=module)

        self.assertIsNone(tensor_scale)
        torch.testing.assert_close(out, _k3_channel_fp8_to_bf16(module, module.weight))

    def test_restored_processed_state_matches_the_checkpoint(self):
        """Loaders that restore a saved, already-processed state run post_load_weights
        on kv_b's transposed, maybe AITER-preshuffled view; the absorb weights must
        match the ones built from the checkpoint layout."""
        for preshuffle in (False, True):
            module = _quark_fp8_linear(32, in_features=64)
            want_q, want_scale = k3_kv_b_for_absorb(module=module)
            with (
                self.subTest(preshuffle=preshuffle),
                _quark_processing(fnuz=False, preshuffle=preshuffle),
            ):
                module.scheme.process_weights_after_loading(module)
                got_q, got_scale = k3_kv_b_for_absorb(module=module)

                self.assertEqual(got_q.shape, want_q.shape)
                self.assertTrue(
                    torch.equal(got_q.view(torch.uint8), want_q.view(torch.uint8))
                )
                torch.testing.assert_close(got_scale, want_scale, rtol=0, atol=0)


_QUARK_FP8 = "sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8"


def _aiter_shuffle_16x16(weight: torch.Tensor) -> torch.Tensor:
    """AITER's shuffle_weight(layout=(16, 16)) for a one-byte [N, K] weight."""
    n, k = weight.shape
    return (
        weight.view(n // 16, 16, k // 32, 2, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(n, k)
    )


@contextlib.contextmanager
def _quark_processing(*, fnuz: bool, preshuffle: bool = False):
    """QuarkW8A8Fp8.process_weights_after_loading as an fnuz or OCP GPU runs it;
    narrow partitions stay FP8, as K3's 128-row f_a does."""
    with (
        mock.patch(f"{_QUARK_FP8}._is_hip", False),
        mock.patch(f"{_QUARK_FP8}._is_fp8_fnuz", fnuz),
        # Both copies of the predicate, so the unshuffle sees what was shuffled.
        mock.patch(f"{_QUARK_FP8}.use_aiter_bpreshuffle_gemm", return_value=preshuffle),
        mock.patch(
            "sglang.srt.layers.quantization.fp8_utils.use_aiter_bpreshuffle_gemm",
            return_value=preshuffle,
        ),
        mock.patch(
            f"{_QUARK_FP8}.shuffle_weight",
            lambda weight, layout: _aiter_shuffle_16x16(weight),
            create=True,
        ),
    ):
        yield


def _quark_fp8_linear(out_features: int, in_features: int = 16):
    scheme = QuarkW8A8Fp8(
        {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": True}
    )
    scheme.out_dtype = torch.bfloat16
    weight, scale, _ = _per_channel_fp8(out_features, in_features)
    return SimpleNamespace(
        weight=weight,
        weight_scale=scale.squeeze(1),
        scheme=scheme,
        quant_method=mock.Mock(spec=QuarkLinearMethod),
    )


def _dequant(linear) -> torch.Tensor:
    return linear.weight.float() * linear.weight_scale.view(-1, 1)


class _QuarkFp8Config:
    """Resolves every linear to Quark per-channel FP8 unless its prefix or class
    name is excluded; Quark's layer_type_quant_config matches the class name."""

    def __init__(self, excluded=()):
        self.excluded = excluded

    def get_quant_method(self, layer, prefix):
        if prefix in self.excluded or type(layer).__name__ in self.excluded:
            return UnquantizedLinearMethod()
        layer.scheme = _quark_fp8_linear(1).scheme
        return QuarkLinearMethod(None)


def _kda_inproj_linears(b_proj=None, quant_config=None) -> dict:
    """[q,k,v,g] = 32 rows plus the tail k3_kda_inproj_fp8_tail_rows reserves."""
    mods = dict(
        f_a_proj=_quark_fp8_linear(8),
        b_proj=_quark_fp8_linear(3) if b_proj is None else b_proj,
        f_b_proj=_quark_fp8_linear(16, in_features=8),
    )
    tail = k3_kda_inproj_fp8_tail_rows(
        wide_rows=32,
        quant_config=quant_config or _QuarkFp8Config(),
        qkvg_prefix="fused_qkvg_proj",
        **mods,
    )
    return dict(qkvg_proj=_quark_fp8_linear(32 + tail), qkvg_sizes=[24, 8], **mods)


class TestMergeKdaInprojFp8(CustomTestCase):
    """The merged buffer has to carry every projection's channel scales; a raw
    row-cat of FP8 payloads would silently drop them."""

    def test_tail_pads_the_merged_width_to_64(self):
        # 32 + 8 + 3 = 43 rows; N % 64 == 0 keeps AITER's preshuffled FP8 GEMM.
        mods = _kda_inproj_linears()
        self.assertEqual(mods["qkvg_proj"].weight.shape[0], 64)

    def test_merged_rows_keep_every_channel_scale(self):
        mods = _kda_inproj_linears()
        expected = torch.cat(
            [
                _dequant(mods["qkvg_proj"])[:32],
                _dequant(mods["f_a_proj"]),
                _dequant(mods["b_proj"]),
            ]
        )

        sizes, f_b = k3_merge_kda_inproj_fp8(**mods)

        # Checkpoint layout, [out, in] and [out], for the loader to process.
        merged = _dequant(mods["qkvg_proj"])
        torch.testing.assert_close(merged[:43], expected)
        self.assertEqual(merged[43:].abs().max().item(), 0.0)
        self.assertEqual(sizes, [24, 8, 8, 3, 21])
        torch.testing.assert_close(f_b, _dequant(mods["f_b_proj"]).to(torch.bfloat16))

    def test_merge_fills_the_reserved_rows_in_place(self):
        """Swapping in a new, slightly wider buffer would free the wide weight
        and strand caching-allocator segments on every KDA layer."""
        mods = _kda_inproj_linears()
        weight, scale = mods["qkvg_proj"].weight, mods["qkvg_proj"].weight_scale
        ptrs = (weight.data_ptr(), scale.data_ptr())

        k3_merge_kda_inproj_fp8(**mods)

        self.assertIs(mods["qkvg_proj"].weight, weight)
        self.assertEqual((weight.data_ptr(), scale.data_ptr()), ptrs)

    def test_no_tail_for_a_projection_that_is_not_quark_fp8(self):
        bf16_b_proj = SimpleNamespace(
            weight=torch.zeros(3, 16, dtype=torch.bfloat16), quant_method=None
        )
        mods = _kda_inproj_linears(b_proj=bf16_b_proj)
        self.assertEqual(mods["qkvg_proj"].weight.shape[0], 32)
        self.assertIsNone(k3_merge_kda_inproj_fp8(**mods))

    def test_projection_without_a_channel_scale_skips_the_merge(self):
        """A BF16 or block-FP8 Kimi-K3 checkpoint has no per-channel weight_scale;
        loading it must skip the FP8 merge instead of raising."""
        mods = {
            name: SimpleNamespace(
                weight=torch.zeros(rows, cols, dtype=torch.bfloat16), quant_method=None
            )
            for name, rows, cols in (
                ("qkvg_proj", 32, 16),
                ("f_a_proj", 8, 16),
                ("b_proj", 3, 16),
                ("f_b_proj", 16, 8),
            )
        }
        self.assertIsNone(k3_merge_kda_inproj_fp8(qkvg_sizes=[24, 8], **mods))

    def test_no_tail_when_qkvg_is_excluded_from_quantization(self):
        """[q,k,v,g] kept BF16 while f_a/b/f_b are FP8: loading must keep the split
        projection instead of failing on a tail the merge cannot fill."""
        quant_config = _QuarkFp8Config(excluded=("fused_qkvg_proj",))
        mods = _kda_inproj_linears(quant_config=quant_config)
        self.assertEqual(mods["qkvg_proj"].weight.shape[0], 32)

    def test_tail_follows_the_quant_config_of_the_real_qkvg_class(self):
        """A config that resolves [q,k,v,g] by class name must size the tail for
        fused_qkvg_proj's own class, or the merge fails on a tail it cannot fill."""
        quant_config = _QuarkFp8Config(excluded=("MergedColumnParallelLinear",))
        mods = _kda_inproj_linears(quant_config=quant_config)
        self.assertEqual(mods["qkvg_proj"].weight.shape[0], 32)

    def test_env_disables_the_tail(self):
        with envs.SGLANG_OPT_HIP_K3_FUSE_KDA_INPROJ.override(False):
            mods = _kda_inproj_linears()
        self.assertEqual(mods["qkvg_proj"].weight.shape[0], 32)
        self.assertIsNone(k3_merge_kda_inproj_fp8(**mods))

    @staticmethod
    def _process(linear) -> None:
        """What QuarkW8A8Fp8.process_weights_after_loading leaves: an [in, out] view."""
        linear.weight = linear.weight.t()
        linear.weight_scale = linear.weight_scale.view(-1, 1)

    def _restored_processed_state(self) -> dict:
        """A ShardedStateLoader restore: processed modules holding a merged tail."""
        mods = _kda_inproj_linears()
        k3_merge_kda_inproj_fp8(**mods)
        for name in ("qkvg_proj", "f_a_proj", "b_proj", "f_b_proj"):
            self._process(mods[name])
        return mods

    def test_processed_state_keeps_the_restored_tail(self):
        mods = self._restored_processed_state()
        weight = mods["qkvg_proj"].weight
        before = weight.float().clone()

        sizes, f_b = k3_merge_kda_inproj_fp8(**mods)

        self.assertEqual(sizes, [24, 8, 8, 3, 21])
        self.assertIsNone(f_b)
        self.assertIs(mods["qkvg_proj"].weight, weight)
        torch.testing.assert_close(weight.float(), before, rtol=0, atol=0)

    def test_processed_state_from_an_fnuz_gpu_keeps_the_restored_tail(self):
        """On e4m3fnuz GPUs (MI300) processing doubles every channel scale, the
        merged pad rows' included; a correctly saved state must still restore."""
        mods = _kda_inproj_linears()
        k3_merge_kda_inproj_fp8(**mods)
        with _quark_processing(fnuz=True):
            for name in ("qkvg_proj", "f_a_proj", "b_proj", "f_b_proj"):
                mods[name].scheme.process_weights_after_loading(mods[name])

        sizes, f_b = k3_merge_kda_inproj_fp8(**mods)

        self.assertEqual(sizes, [24, 8, 8, 3, 21])
        self.assertIsNone(f_b)

    def test_processed_tail_that_was_never_merged_raises(self):
        """A state saved without the tail loads into a narrowed view and leaves the
        reserved rows uninitialized; serving them would corrupt f_a and beta."""
        mods = self._restored_processed_state()
        mods["qkvg_proj"].weight_scale[32:].normal_()

        with self.assertRaisesRegex(RuntimeError, "same"):
            k3_merge_kda_inproj_fp8(**mods)

    def test_processed_narrow_b_proj_dequantized_to_bf16(self):
        mods = self._restored_processed_state()
        b_proj = mods["b_proj"]
        b_proj.weight = b_proj.weight.t().to(torch.bfloat16).t()
        b_proj.weight_scale = None

        sizes, _ = k3_merge_kda_inproj_fp8(**mods)

        self.assertEqual(sizes, [24, 8, 8, 3, 21])

    def test_mixed_processed_and_checkpoint_layouts_raise(self):
        mods = _kda_inproj_linears()
        self._process(mods["qkvg_proj"])
        with self.assertRaises(RuntimeError):
            k3_merge_kda_inproj_fp8(**mods)

    def test_processed_state_without_a_tail_skips_the_merge(self):
        with envs.SGLANG_OPT_HIP_K3_FUSE_KDA_INPROJ.override(False):
            mods = _kda_inproj_linears()
        for name in ("qkvg_proj", "f_a_proj", "b_proj", "f_b_proj"):
            self._process(mods[name])
        self.assertIsNone(k3_merge_kda_inproj_fp8(**mods))

    @staticmethod
    def _kda_attention() -> torch.nn.Module:
        mods = _kda_inproj_linears()
        attn = torch.nn.Module()
        attn.__dict__.update(
            fused_qkvg_proj=mods["qkvg_proj"],
            split_sizes=mods["qkvg_sizes"],
            f_a_proj=mods["f_a_proj"],
            b_proj=mods["b_proj"],
            f_b_proj=mods["f_b_proj"],
            _qkvgbfa_in_qkvg_proj=False,
            _qkvgbfa_sizes=None,
            _bfa_f_b_w=None,
            _derived_weight_cache_error=None,
        )
        return attn

    def test_merged_layer_rejects_online_weight_updates(self):
        """f_a/b rows moved into fused_qkvg_proj and the cached BF16 f_b would
        not follow a reload, so the updater must refuse before writing."""
        attn = self._kda_attention()
        model = torch.nn.Sequential(attn)
        self.assertIsNone(_unsupported_derived_weight_cache_error(model))

        self.assertTrue(KimiK3DeltaAttention._merge_kda_inproj_fp8_hip(attn))

        self.assertEqual(attn.fused_qkvg_proj.weight.shape[0], 64)
        self.assertIn(
            "SGLANG_OPT_HIP_K3_FUSE_KDA_INPROJ=0",
            _unsupported_derived_weight_cache_error(model),
        )

    def test_processed_state_runs_f_b_on_its_projection(self):
        attn = self._kda_attention()
        mods = self._restored_processed_state()
        attn.__dict__.update(
            fused_qkvg_proj=mods["qkvg_proj"],
            f_a_proj=mods["f_a_proj"],
            b_proj=mods["b_proj"],
            f_b_proj=mods["f_b_proj"],
        )

        self.assertTrue(KimiK3DeltaAttention._merge_kda_inproj_fp8_hip(attn))

        self.assertEqual(attn._qkvgbfa_sizes, [24, 8, 8, 3, 21])
        self.assertIsNone(attn._bfa_f_b_w)

    def test_rerun_keeps_the_merge(self):
        """A second post_load_weights must keep the FP8 merge, not fall back to the
        BF16 merge over weights that no longer have the checkpoint layout."""
        attn = self._kda_attention()
        self.assertTrue(KimiK3DeltaAttention._merge_kda_inproj_fp8_hip(attn))
        weight, sizes = attn.fused_qkvg_proj.weight, attn._qkvgbfa_sizes

        self.assertTrue(KimiK3DeltaAttention._merge_kda_inproj_fp8_hip(attn))

        self.assertIs(attn.fused_qkvg_proj.weight, weight)
        self.assertEqual(attn._qkvgbfa_sizes, sizes)


if __name__ == "__main__":
    unittest.main()
