"""Hermetic unit tests for DeepSeek MLA attention-method dispatch on ROCm.

`_dispatch_mla_subtype` picks the forward method for MLA attention. On ROCm the
fused-decode-MLA + fused-RoPE fast path (`MLA_FUSED_ROPE_ROCM`) is only correct
for the aiter attention backend; taking it under the triton backend GPU-faults
on gfx95 (MI355). This test pins the dispatch table so the triton MLA path stays
on the plain `MLA` method.

Pure Python (no GPU, no model weights): `_is_hip` is patched and `attn` /
`forward_batch` are lightweight fakes. Runs on any PR-CI lane.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.models.deepseek_common import attention_backend_handler as abh
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fake_forward_batch(is_decode: bool):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: is_decode))


def _fake_attn(backend: str, rocm_fused_decode_mla: bool = True):
    return SimpleNamespace(
        current_attention_backend=backend,
        rocm_fused_decode_mla=rocm_fused_decode_mla,
    )


class TestDispatchMLASubtype(CustomTestCase):
    def test_hip_aiter_decode_takes_fused_rope(self):
        # aiter + fused-decode + decode -> fused ROPE fast path (unchanged).
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("aiter"), _fake_forward_batch(is_decode=True)
            )
        self.assertEqual(method, AttnForwardMethod.MLA_FUSED_ROPE_ROCM)

    def test_hip_triton_decode_stays_plain_mla(self):
        # The fix: triton backend must NOT take the aiter-only fused path even
        # with rocm_fused_decode_mla set -- that path GPU-faults on gfx95.
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("triton"), _fake_forward_batch(is_decode=True)
            )
        self.assertEqual(method, AttnForwardMethod.MLA)

    def test_hip_aiter_extend_stays_plain_mla(self):
        # Fused path is decode-only; extend/prefill uses plain MLA.
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("aiter"), _fake_forward_batch(is_decode=False)
            )
        self.assertEqual(method, AttnForwardMethod.MLA)


class TestResolveRocmForwardMethod(CustomTestCase):
    """The generic MHA/MLA methods must never reach the CUDA forward paths on
    ROCm: those were stripped of their AMD branches when the AITER kernels moved
    into forward_mha_rocm.py / forward_mla_rocm.py."""

    def test_hip_routes_shared_methods_to_rocm(self):
        with mock.patch.object(abh, "_is_hip", True):
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MHA),
                AttnForwardMethod.MHA_ROCM,
            )
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MHA_ONE_SHOT),
                AttnForwardMethod.MHA_ONE_SHOT_ROCM,
            )
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MLA),
                AttnForwardMethod.MLA_ROCM,
            )

    def test_hip_leaves_platform_specific_methods_alone(self):
        with mock.patch.object(abh, "_is_hip", True):
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MLA_FUSED_ROPE_ROCM),
                AttnForwardMethod.MLA_FUSED_ROPE_ROCM,
            )

    def test_non_hip_is_identity(self):
        with mock.patch.object(abh, "_is_hip", False):
            for method in AttnForwardMethod:
                self.assertEqual(abh.resolve_rocm_forward_method(method), method)


class TestAiterKimiK3ChunkedPrefixDispatch(CustomTestCase):
    @staticmethod
    def _attn():
        return SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            num_local_heads=12,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            kv_lora_rank=512,
        )

    @staticmethod
    def _batch(prefix_len):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend_without_speculative=lambda: True,
            ),
            extend_prefix_lens_cpu=[prefix_len],
            get_max_chunk_capacity=lambda: 131072,
        )

    def test_long_kimi_k3_prefix_uses_chunked_mha(self):
        parallel = SimpleNamespace(dcp_enabled=False)
        with (
            mock.patch.object(abh, "_is_hip", True),
            mock.patch.object(abh, "is_gfx95_supported", return_value=True),
            mock.patch.object(abh, "get_parallel", return_value=parallel),
            mock.patch.object(abh, "is_in_tc_piecewise_cuda_graph", return_value=False),
            mock.patch.object(abh, "is_in_breakable_cuda_graph", return_value=False),
        ):
            method = abh.handle_attention_aiter(
                self._attn(), self._batch(prefix_len=254000)
            )
        self.assertEqual(method, AttnForwardMethod.MHA_CHUNKED_KV)

    def test_prefix_within_capacity_keeps_fast_mha(self):
        parallel = SimpleNamespace(dcp_enabled=False)
        with (
            mock.patch.object(abh, "_is_hip", True),
            mock.patch.object(abh, "is_gfx95_supported", return_value=True),
            mock.patch.object(abh, "get_parallel", return_value=parallel),
            mock.patch.object(abh, "is_in_tc_piecewise_cuda_graph", return_value=False),
            mock.patch.object(abh, "is_in_breakable_cuda_graph", return_value=False),
        ):
            method = abh.handle_attention_aiter(
                self._attn(), self._batch(prefix_len=131072)
            )
        self.assertEqual(method, AttnForwardMethod.MHA)


if __name__ == "__main__":
    unittest.main()
