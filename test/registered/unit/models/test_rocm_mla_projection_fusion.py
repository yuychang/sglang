import types
import unittest
from unittest import mock

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRocmMlaProjectionFusion(CustomTestCase):
    def test_projection_fusion_gate_targets_aiter_graph_decode(self):
        attn = types.SimpleNamespace(
            current_attention_backend="aiter",
            use_dsa=False,
            w_kc=torch.empty(1, dtype=torch.uint8),
            w_scale_k=torch.empty(1),
            rotary_emb=object(),
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        forward_batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_decode_or_idle=mock.Mock(return_value=True)
            )
        )

        with (
            mock.patch.object(forward_mla, "_is_hip", True),
            mock.patch.object(forward_mla, "_use_aiter_gfx95", True),
            mock.patch.object(
                forward_mla.envs.SGLANG_ROCM_USE_MULTI_STREAM,
                "get",
                side_effect=AssertionError(
                    "MLA projection fusion must not depend on MoE multi-stream mode"
                ),
            ),
            mock.patch.object(
                forward_mla.envs.SGLANG_ROCM_FUSE_MLA_PROJECTION_ROPE_CACHE,
                "get",
                return_value=True,
            ),
            mock.patch.object(forward_mla, "is_kv_b_lora_active", return_value=False),
            mock.patch.object(
                forward_mla,
                "get_parallel",
                return_value=types.SimpleNamespace(dcp_enabled=False),
            ),
        ):
            self.assertTrue(
                forward_mla.DeepseekMLAForwardMixin._can_fuse_rocm_mla_projection_rope_cache(
                    attn, forward_batch, is_capture_mode=True
                )
            )

            attn.current_attention_backend = "flashinfer"
            self.assertFalse(
                forward_mla.DeepseekMLAForwardMixin._can_fuse_rocm_mla_projection_rope_cache(
                    attn, forward_batch, is_capture_mode=True
                )
            )

    def test_value_quant_fusion_gate_targets_non_lora_aiter_decode(self):
        attn = types.SimpleNamespace(
            current_attention_backend="aiter",
            use_dsa=False,
            w_vc=torch.empty(1, dtype=torch.uint8),
            w_scale_v=torch.empty(1, dtype=torch.uint8),
            o_proj=types.SimpleNamespace(weight=torch.empty(1, dtype=torch.uint8)),
            kv_lora_rank=512,
            v_head_dim=128,
        )
        forward_batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_decode_or_idle=mock.Mock(return_value=True)
            )
        )

        with (
            mock.patch.object(forward_mla, "_is_hip", True),
            mock.patch.object(forward_mla, "_use_aiter_gfx95", True),
            mock.patch.object(
                forward_mla.envs.SGLANG_ROCM_FUSE_MLA_VALUE_MXFP4_QUANT,
                "get",
                return_value=True,
            ),
            mock.patch.object(
                forward_mla, "is_kv_b_lora_active", return_value=False
            ) as lora_active,
        ):
            self.assertTrue(
                forward_mla.DeepseekMLAForwardMixin._can_fuse_rocm_mla_value_mxfp4_quant(
                    attn, forward_batch
                )
            )
            lora_active.return_value = True
            self.assertFalse(
                forward_mla.DeepseekMLAForwardMixin._can_fuse_rocm_mla_value_mxfp4_quant(
                    attn, forward_batch
                )
            )


if __name__ == "__main__":
    unittest.main()
