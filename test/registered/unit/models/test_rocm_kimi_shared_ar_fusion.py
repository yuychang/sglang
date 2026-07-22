import types
import unittest
from unittest import mock

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.layers import layernorm
from sglang.srt.layers.moe import rocm_kimi_shared
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_cuda_tensor(shape=(4, 7168), dtype=torch.bfloat16):
    tensor = mock.Mock()
    tensor.device = types.SimpleNamespace(type="cuda")
    tensor.dtype = dtype
    tensor.shape = shape
    tensor.dim.return_value = len(shape)
    tensor.is_contiguous.return_value = True
    return tensor


class _CarrierTensor:
    def __init__(self, shape, dtype, ptr, device=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device or torch.device("cuda")
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr

    def dim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def is_contiguous(self):
        return True


class TestRocmKimiSharedArFusion(CustomTestCase):
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

    def test_fused_ar_mxfp4_quant_mode_is_default_off(self):
        with mock.patch(
            "sglang.srt.runtime_context.get_server_args",
            return_value=types.SimpleNamespace(
                enable_rocm_fused_ar_mxfp4_quant=False
            ),
        ):
            self.assertEqual(
                rocm_kimi_shared.get_rocm_fused_ar_mxfp4_quant_mode(),
                "off",
            )

    def test_fused_ar_mxfp4_quant_mode_rejects_invalid_internal_value(self):
        with (
            mock.patch(
                "sglang.srt.runtime_context.get_server_args",
                return_value=types.SimpleNamespace(
                    enable_rocm_fused_ar_mxfp4_quant=True
                ),
            ),
            mock.patch.object(
                rocm_kimi_shared.envs.SGLANG_ROCM_FUSED_AR_MXFP4_QUANT_MODE,
                "get",
                return_value="invalid",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "must be one of"):
                rocm_kimi_shared.get_rocm_fused_ar_mxfp4_quant_mode()

    def test_fused_ar_mxfp4_quant_eligibility_concurrency_matrix(self):
        kwargs = dict(
            is_target_layer=True,
            mode="optimized",
            is_graph_capture_mode=True,
            is_gfx950=True,
            tp_world_size=4,
            ep_world_size=1,
            hip_version=(7, 2),
        )
        for token_count in (4, 8, 16, 32, 64, 128):
            with self.subTest(token_count=token_count):
                hidden = _CarrierTensor(
                    (token_count, 7168), torch.bfloat16, token_count
                )
                residual = _CarrierTensor(
                    (token_count, 7168),
                    torch.bfloat16,
                    token_count + 1,
                    device=hidden.device,
                )
                weight = _CarrierTensor(
                    (7168,), torch.bfloat16, token_count + 2, device=hidden.device
                )
                self.assertTrue(
                    rocm_kimi_shared.can_fuse_rocm_mxfp4_activation(
                        hidden,
                        residual,
                        weight,
                        **kwargs,
                    )
                )

        for token_count in (1, 2, 3, 5, 256):
            with self.subTest(unsupported_token_count=token_count):
                hidden = _CarrierTensor(
                    (token_count, 7168), torch.bfloat16, token_count
                )
                residual = _CarrierTensor(
                    (token_count, 7168),
                    torch.bfloat16,
                    token_count + 1,
                    device=hidden.device,
                )
                weight = _CarrierTensor(
                    (7168,), torch.bfloat16, token_count + 2, device=hidden.device
                )
                self.assertFalse(
                    rocm_kimi_shared.can_fuse_rocm_mxfp4_activation(
                        hidden,
                        residual,
                        weight,
                        **kwargs,
                    )
                )

    def test_mxfp4_activation_carrier_concurrency_matrix(self):
        for token_count in (4, 8, 16, 32, 64, 128):
            with self.subTest(token_count=token_count):
                hidden = _CarrierTensor(
                    (token_count, 7168), torch.bfloat16, token_count * 10 + 1
                )
                packed = _CarrierTensor(
                    (token_count, 3584),
                    torch.uint8,
                    token_count * 10 + 2,
                    device=hidden.device,
                )
                scale = _CarrierTensor(
                    (token_count, 224),
                    torch.uint8,
                    token_count * 10 + 3,
                    device=hidden.device,
                )
                rocm_kimi_shared.attach_rocm_mxfp4_activation(
                    hidden, packed, scale
                )
                carrier = rocm_kimi_shared.pop_rocm_mxfp4_activation(hidden)
                self.assertIs(carrier.packed, packed)
                self.assertIs(carrier.scale, scale)
                self.assertIsNone(
                    rocm_kimi_shared.pop_rocm_mxfp4_activation(hidden)
                )

    def test_mxfp4_activation_carrier_rejects_unsupported_concurrency(self):
        hidden = _CarrierTensor((2, 7168), torch.bfloat16, 1)
        packed = _CarrierTensor((2, 3584), torch.uint8, 2, device=hidden.device)
        scale = _CarrierTensor((2, 224), torch.uint8, 3, device=hidden.device)
        with self.assertRaisesRegex(RuntimeError, "invalid ROCm fused AR"):
            rocm_kimi_shared.attach_rocm_mxfp4_activation(
                hidden,
                packed,
                scale,
            )

    def test_mxfp4_activation_carrier_rejects_stale_source(self):
        hidden = _CarrierTensor((32, 7168), torch.bfloat16, 21)
        packed = _CarrierTensor((32, 3584), torch.uint8, 22, device=hidden.device)
        scale = _CarrierTensor((32, 224), torch.uint8, 23, device=hidden.device)
        rocm_kimi_shared.attach_rocm_mxfp4_activation(hidden, packed, scale)
        hidden._ptr += 1
        with self.assertRaisesRegex(RuntimeError, "stale or mismatched"):
            rocm_kimi_shared.pop_rocm_mxfp4_activation(hidden)
        self.assertIsNone(rocm_kimi_shared.pop_rocm_mxfp4_activation(hidden))

    def test_mxfp4_carrier_matches_exact_shared_fc1_concurrency_matrix(self):
        for token_count in (4, 8, 16, 32, 64, 128):
            with self.subTest(token_count=token_count):
                hidden = _CarrierTensor(
                    (token_count, 7168), torch.bfloat16, token_count * 10 + 1
                )
                packed = _CarrierTensor(
                    (token_count, 3584),
                    torch.uint8,
                    token_count * 10 + 2,
                    device=hidden.device,
                )
                scale = _CarrierTensor(
                    (token_count, 224),
                    torch.uint8,
                    token_count * 10 + 3,
                    device=hidden.device,
                )
                rocm_kimi_shared.attach_rocm_mxfp4_activation(
                    hidden, packed, scale
                )
                carrier = rocm_kimi_shared.pop_rocm_mxfp4_activation(hidden)
                weight = _CarrierTensor(
                    (1024, 3584),
                    torch.uint8,
                    token_count * 10 + 4,
                    device=hidden.device,
                )
                weight_scale = _CarrierTensor(
                    (1024, 224),
                    torch.uint8,
                    token_count * 10 + 5,
                    device=hidden.device,
                )
                self.assertTrue(
                    rocm_kimi_shared.validate_rocm_mxfp4_shared_fc1(
                        carrier,
                        weight,
                        weight_scale,
                    )
                )

                weight.shape = (2048, 3584)
                self.assertFalse(
                    rocm_kimi_shared.validate_rocm_mxfp4_shared_fc1(
                        carrier,
                        weight,
                        weight_scale,
                    )
                )

    def test_layernorm_selects_mxfp4_producer_fusion(self):
        x = torch.randn(32, 8, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.ones(8, dtype=torch.bfloat16)
        packed = torch.empty(32, 4, dtype=torch.uint8)
        scale = torch.empty(32, 1, dtype=torch.uint8)
        residual_out = torch.empty_like(x)
        bf16_out = torch.empty_like(x)
        norm = types.SimpleNamespace(
            variance_epsilon=1e-6,
            _rocm_fused_ar_mxfp4_quant_target=True,
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
                rocm_kimi_shared,
                "can_fuse_rocm_mxfp4_activation",
                return_value=True,
            ),
            mock.patch.object(
                rocm_kimi_shared,
                "get_rocm_fused_ar_mxfp4_quant_mode",
                return_value="optimized",
            ),
            mock.patch(
                "sglang.srt.model_executor.runner.get_is_capture_mode",
                return_value=True,
            ),
            mock.patch("sglang.srt.utils.get_hip_version", return_value=(7, 2)),
            mock.patch("sglang.srt.utils.is_gfx95_supported", return_value=True),
            mock.patch.object(
                distributed,
                "tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant",
                return_value=(packed, residual_out, scale, bf16_out),
            ) as fused_quant,
            mock.patch.object(
                rocm_kimi_shared,
                "attach_rocm_mxfp4_activation",
            ) as attach,
            mock.patch.object(layernorm.logger, "info_once"),
        ):
            result = layernorm._forward_with_allreduce_fusion(
                norm,
                x,
                residual,
                None,
                weight,
                use_attn_tp_group=True,
            )
        self.assertIs(result[0], bf16_out)
        self.assertIs(result[1], residual_out)
        fused_quant.assert_called_once_with(x, residual, weight, 1e-6)
        attach.assert_called_once_with(bf16_out, packed, scale)

    def test_group_coordinator_uses_direct_mxfp4_api_through_m32(self):
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        for token_count in (4, 8, 16, 32):
            with self.subTest(token_count=token_count):
                coordinator = object.__new__(GroupCoordinator)
                coordinator.world_size = 4
                coordinator.ca_comm = types.SimpleNamespace(
                    disabled=False,
                    custom_fused_ar_rms_mxfp4_quant=mock.Mock(
                        return_value="result"
                    ),
                )
                x = _CarrierTensor((token_count, 7168), torch.bfloat16, 1)
                residual = _CarrierTensor(
                    (token_count, 7168), torch.bfloat16, 2, device=x.device
                )
                weight = _CarrierTensor(
                    (7168,), torch.bfloat16, 3, device=x.device
                )

                result = coordinator.fused_allreduce_rmsnorm_mxfp4_quant(
                    x,
                    residual,
                    weight,
                    1e-6,
                )

                self.assertEqual(result, "result")
                coordinator.ca_comm.custom_fused_ar_rms_mxfp4_quant.assert_called_once_with(
                    x,
                    residual,
                    weight,
                    1e-6,
                    use_1stage=token_count == 4,
                    emit_bf16=True,
                )

    def test_group_coordinator_composes_safe_m64_m128_fallback(self):
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        for token_count in (64, 128):
            with self.subTest(token_count=token_count):
                coordinator = object.__new__(GroupCoordinator)
                coordinator.world_size = 4
                direct = mock.Mock(return_value="unexpected")
                coordinator.ca_comm = types.SimpleNamespace(
                    disabled=False,
                    custom_fused_ar_rms_mxfp4_quant=direct,
                )
                x = _CarrierTensor((token_count, 7168), torch.bfloat16, 11)
                residual = _CarrierTensor(
                    (token_count, 7168), torch.bfloat16, 12, device=x.device
                )
                weight = _CarrierTensor(
                    (7168,), torch.bfloat16, 13, device=x.device
                )
                bf16_out = _CarrierTensor(
                    (token_count, 7168), torch.bfloat16, 14, device=x.device
                )
                residual_out = _CarrierTensor(
                    (token_count, 7168), torch.bfloat16, 15, device=x.device
                )
                packed = _CarrierTensor(
                    (token_count, 3584), torch.uint8, 16, device=x.device
                )
                scale = _CarrierTensor(
                    (token_count, 224), torch.uint8, 17, device=x.device
                )
                coordinator.fused_allreduce_rmsnorm = mock.Mock(
                    return_value=(bf16_out, residual_out)
                )

                with mock.patch.object(
                    parallel_state,
                    "_dynamic_mxfp4_quant",
                    return_value=(packed, scale),
                ) as dynamic_quant:
                    result = coordinator.fused_allreduce_rmsnorm_mxfp4_quant(
                        x,
                        residual,
                        weight,
                        1e-6,
                    )

                self.assertEqual(result, (packed, residual_out, scale, bf16_out))
                direct.assert_not_called()
                coordinator.fused_allreduce_rmsnorm.assert_called_once_with(
                    x, residual, weight, 1e-6
                )
                dynamic_quant.assert_called_once_with(bf16_out)

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
