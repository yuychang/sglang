"""Kimi-K3 fused MLA Q/cache producer: FP8 padded Q for the fp8-Q asm backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import mla_q_cache_aiter_hip
from sglang.srt.models import kimi_k3_rocm_mla as mla
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TOKENS, _HEADS = 4, 12
_FP8 = torch.float8_e4m3fn


def _module(k_scale=None):
    return SimpleNamespace(
        _k3_mla_q_cache_fusion=True,
        attn_mqa=SimpleNamespace(layer_id=0, k_scale=k_scale),
        _k3_mla_q_cache_scale=torch.ones(1),
        _k3_identity_rope_cos=torch.ones(1, 32, dtype=torch.bfloat16),
        _k3_identity_rope_sin=torch.zeros(1, 32, dtype=torch.bfloat16),
        current_attention_backend="aiter",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )


def _inputs():
    return (
        torch.zeros(_TOKENS, _HEADS, 512, dtype=torch.bfloat16),
        torch.zeros(_TOKENS, _HEADS, 64, dtype=torch.bfloat16),
        torch.zeros(_TOKENS, 1, 512, dtype=torch.bfloat16),
        torch.zeros(_TOKENS, 1, 64, dtype=torch.bfloat16),
        torch.zeros(_TOKENS, dtype=torch.int64),
        torch.zeros(_TOKENS, dtype=torch.int64),
    )


class TestA8W8FusedQ(unittest.TestCase):
    def _produce(self, *, enabled, a8w8_backend, k_scale=None, gluon=True):
        captured = {}

        def fake_run(**kwargs):
            captured["out"] = kwargs["out"]
            return kwargs["out"]

        pool = SimpleNamespace(
            get_key_buffer=lambda layer_id: torch.zeros(4, 1, 576, dtype=_FP8)
        )
        with (
            patch.object(mla, "_a8w8_fused_q", enabled),
            patch.object(mla, "_backend_uses_a8w8_asm", return_value=a8w8_backend),
            patch(
                "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
                return_value=pool,
            ),
            patch(
                "sglang.srt.layers.attention.aiter_mla_gluon.prefer_mla_gluon_decode",
                return_value=gluon,
            ),
            patch.object(mla_q_cache_aiter_hip, "covered", return_value=True),
            patch.object(mla_q_cache_aiter_hip, "run", fake_run),
        ):
            q, _ = mla.try_fused_mla_q_cache(_module(k_scale), *_inputs())
        return q, captured["out"]

    def test_default_keeps_bf16_unpadded_q_for_gluon(self):
        q, out = self._produce(enabled=False, a8w8_backend=True)
        self.assertEqual((q.dtype, tuple(q.shape)), (torch.bfloat16, (_TOKENS, 12, 576)))

    def test_enabled_with_asm_backend_emits_zero_padded_fp8_q(self):
        q, out = self._produce(enabled=True, a8w8_backend=True)
        self.assertEqual((q.dtype, tuple(q.shape)), (_FP8, (_TOKENS, 16, 576)))
        self.assertEqual(q[:, _HEADS:].float().abs().sum().item(), 0.0)

    def test_enabled_without_asm_backend_is_unchanged(self):
        q, _ = self._produce(enabled=True, a8w8_backend=False)
        self.assertEqual((q.dtype, tuple(q.shape)), (torch.bfloat16, (_TOKENS, 12, 576)))

    def test_layer_k_scale_disables_the_fused_a8w8_q(self):
        q, _ = self._produce(enabled=True, a8w8_backend=True, k_scale=torch.full((1,), 2.0))
        self.assertEqual(q.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
