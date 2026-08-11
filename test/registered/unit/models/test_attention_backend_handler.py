"""Unit tests for model-specific attention forward routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.models.deepseek_common import attention_backend_handler as handler
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ForwardMode:
    def __init__(self, *, prefill: bool):
        self.prefill = prefill

    def is_extend_without_speculative(self):
        return self.prefill


class TestKimiK3AiterMlaRouting(CustomTestCase):
    def _route(self, *, local_heads: int, declared_heads=(), prefill=True):
        attn = SimpleNamespace(
            num_local_heads=local_heads,
            aiter_mla_prefill_head_counts=declared_heads,
        )
        forward_batch = SimpleNamespace(
            forward_mode=_ForwardMode(prefill=prefill),
        )
        with (
            patch.object(
                handler, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
            ),
            patch.object(handler, "is_in_tc_piecewise_cuda_graph", return_value=False),
            patch.object(handler, "is_in_breakable_cuda_graph", return_value=False),
        ):
            return handler.handle_attention_aiter(attn, forward_batch)

    def test_kimi_k3_tp8_prefill_uses_absorbed_mla(self):
        self.assertEqual(
            self._route(local_heads=12, declared_heads=(12,)),
            AttnForwardMethod.MLA,
        )

    def test_other_kimi_k3_tp_sizes_keep_default_mha_prefill(self):
        self.assertEqual(
            self._route(local_heads=24, declared_heads=(12,)),
            AttnForwardMethod.MHA,
        )

    def test_unmarked_twelve_head_model_keeps_default_mha_prefill(self):
        self.assertEqual(
            self._route(local_heads=12),
            AttnForwardMethod.MHA,
        )

    def test_decode_uses_absorbed_mla(self):
        self.assertEqual(
            self._route(local_heads=12, declared_heads=(12,), prefill=False),
            AttnForwardMethod.MLA,
        )

    def test_gluon_runner_dispatches_decode_to_gluon_mla(self):
        backend = AiterAttnBackend.__new__(AiterAttnBackend)
        backend.kv_cache_dtype = torch.bfloat16
        backend.use_mla = True
        backend.use_aiter_gluon_mla = True
        backend.forward_metadata = SimpleNamespace(dcp_cp_world_size=1)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _layer_id: object()
        )
        layer = SimpleNamespace(tp_q_head_num=12, qk_head_dim=576, layer_id=0)
        q = torch.empty(2, 12, 576)
        expected = object()

        with patch.object(
            backend, "_mla_decode_fwd_gluon", return_value=expected
        ) as gluon_decode:
            result = backend.forward_decode(
                q,
                k=None,
                v=None,
                layer=layer,
                forward_batch=SimpleNamespace(),
                save_kv_cache=False,
            )

        self.assertIs(result, expected)
        gluon_decode.assert_called_once()
        called_q, called_layer = gluon_decode.call_args.args
        self.assertEqual(called_q.shape, (2, 12 * 576))
        self.assertIs(called_layer, layer)


if __name__ == "__main__":
    unittest.main()
