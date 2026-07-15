import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers import layernorm
from sglang.srt.layers.moe import rocm_kimi_shared


def _fake_cuda_tensor(shape=(4, 7168), dtype=torch.bfloat16):
    tensor = mock.Mock()
    tensor.device = types.SimpleNamespace(type="cuda")
    tensor.dtype = dtype
    tensor.shape = shape
    tensor.dim.return_value = len(shape)
    tensor.is_contiguous.return_value = True
    return tensor


class TestRocmKimiSharedArFusion(unittest.TestCase):
    def test_supported_graph_shape_selects_carrier(self):
        routed = _fake_cuda_tensor()
        shared = _fake_cuda_tensor()
        shared.device = routed.device
        with mock.patch.object(
            rocm_kimi_shared.envs.SGLANG_ROCM_FUSE_SHARED_PARTIAL_AR_RMSNORM,
            "get",
            return_value=True,
        ):
            self.assertTrue(
                rocm_kimi_shared.can_defer_shared_partial_to_graph_ar(
                    routed,
                    shared,
                    should_allreduce_fusion=True,
                    shared_expert_tp1=False,
                    tp_world_size=4,
                    is_graph_capture_mode=True,
                    is_gfx950=True,
                )
            )

    def test_final_layer_and_unsupported_shapes_keep_local_add(self):
        routed = _fake_cuda_tensor()
        shared = _fake_cuda_tensor()
        shared.device = routed.device
        common = dict(
            routed_output=routed,
            shared_output=shared,
            shared_expert_tp1=False,
            tp_world_size=4,
            is_graph_capture_mode=True,
            is_gfx950=True,
        )
        self.assertFalse(
            rocm_kimi_shared.can_defer_shared_partial_to_graph_ar(
                should_allreduce_fusion=False,
                **common,
            )
        )
        shared.shape = (3, 7168)
        self.assertFalse(
            rocm_kimi_shared.can_defer_shared_partial_to_graph_ar(
                should_allreduce_fusion=True,
                **common,
            )
        )

    def test_shared_partial_is_consumed_once(self):
        routed = torch.empty(1)
        shared = torch.empty(1)
        rocm_kimi_shared.attach_shared_partial(routed, shared)
        self.assertIs(rocm_kimi_shared.pop_shared_partial(routed), shared)
        self.assertIsNone(rocm_kimi_shared.pop_shared_partial(routed))

    def test_layernorm_selects_two_input_api(self):
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        shared = torch.randn_like(x)
        residual = torch.randn_like(x)
        weight = torch.ones(8, dtype=torch.bfloat16)
        expected = (torch.empty_like(x), torch.empty_like(x))
        norm = types.SimpleNamespace(
            variance_epsilon=1e-6,
            forward=mock.Mock(),
        )

        import sglang.srt.distributed as distributed

        with (
            mock.patch.object(layernorm, "_use_aiter", True),
            mock.patch.object(
                layernorm,
                "get_parallel",
                return_value=types.SimpleNamespace(
                    attn_tp_size=4,
                    moe_ep_size=1,
                    moe_tp_size=4,
                ),
            ),
            mock.patch.object(
                distributed,
                "tensor_model_parallel_fused_allreduce_rmsnorm_two_input",
                return_value=expected,
            ) as fused_two_input,
        ):
            result = layernorm._forward_with_allreduce_fusion(
                norm,
                x,
                residual,
                None,
                weight,
                use_attn_tp_group=False,
                shared_input=shared,
            )
        self.assertIs(result, expected)
        fused_two_input.assert_called_once_with(
            x,
            shared,
            residual,
            weight,
            1e-6,
        )

    def test_unavailable_two_input_api_runs_existing_add(self):
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        shared = torch.randn_like(x)
        residual = torch.randn_like(x)
        weight = torch.ones(8, dtype=torch.bfloat16)
        combined = torch.empty_like(x)
        expected = (torch.empty_like(x), torch.empty_like(x))
        norm = types.SimpleNamespace(
            variance_epsilon=1e-6,
            forward=mock.Mock(),
        )

        import sglang.srt.distributed as distributed

        with (
            mock.patch.object(layernorm, "_use_aiter", True),
            mock.patch.object(
                layernorm,
                "get_parallel",
                return_value=types.SimpleNamespace(
                    attn_tp_size=4,
                    moe_ep_size=1,
                    moe_tp_size=4,
                ),
            ),
            mock.patch.object(
                distributed,
                "tensor_model_parallel_fused_allreduce_rmsnorm_two_input",
                return_value=None,
            ),
            mock.patch.object(
                distributed,
                "tensor_model_parallel_fused_allreduce_rmsnorm",
                return_value=expected,
            ) as fused_existing,
            mock.patch.object(
                rocm_kimi_shared,
                "rocm_mxfp4_moe_add_shared",
                return_value=combined,
            ) as add_shared,
        ):
            result = layernorm._forward_with_allreduce_fusion(
                norm,
                x,
                residual,
                None,
                weight,
                use_attn_tp_group=False,
                shared_input=shared,
            )
        self.assertIs(result, expected)
        add_shared.assert_called_once_with(x, shared, output=shared)
        fused_existing.assert_called_once_with(combined, residual, weight, 1e-6)


if __name__ == "__main__":
    unittest.main()
