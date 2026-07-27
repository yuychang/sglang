# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.environ import envs

_SHARED_PARTIAL_ATTR = "_sglang_rocm_shared_partial"
_MXFP4_ACTIVATION_ATTR = "_sglang_rocm_mxfp4_activation"
_SUPPORTED_TOKEN_COUNTS = frozenset((1, 2, 4, 8, 16, 32))
_FUSED_AR_MXFP4_TOKEN_COUNTS = frozenset((4, 8, 16, 32, 64, 128))
_KIMI_HIDDEN_SIZE = 7168
_TARGET_FC1_N = 1024
_MXFP4_PACKED_K = _KIMI_HIDDEN_SIZE // 2
_MXFP4_SCALE_K = _KIMI_HIDDEN_SIZE // 32
_VALID_FUSED_AR_MXFP4_QUANT_MODES = frozenset(("off", "event", "optimized"))

RocmFusedArMxfp4QuantMode = Literal["off", "event", "optimized"]


@dataclass(frozen=True)
class RocmMxfp4ActivationCarrier:
    source_data_ptr: int
    source_shape: tuple[int, ...]
    source_dtype: torch.dtype
    source_device: torch.device
    packed: torch.Tensor
    scale: torch.Tensor


def get_rocm_fused_ar_mxfp4_quant_mode() -> RocmFusedArMxfp4QuantMode:
    from sglang.srt.runtime_context import get_server_args

    if not getattr(
        get_server_args(),
        "enable_rocm_fused_ar_mxfp4_quant",
        False,
    ):
        return "off"
    mode = envs.SGLANG_ROCM_FUSED_AR_MXFP4_QUANT_MODE.get().lower()
    if mode not in _VALID_FUSED_AR_MXFP4_QUANT_MODES:
        raise ValueError(
            "SGLANG_ROCM_FUSED_AR_MXFP4_QUANT_MODE must be one of "
            f"{sorted(_VALID_FUSED_AR_MXFP4_QUANT_MODES)}, got {mode!r}"
        )
    return mode  # type: ignore[return-value]


def can_fuse_rocm_mxfp4_activation(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    *,
    is_target_layer: bool,
    mode: RocmFusedArMxfp4QuantMode,
    is_graph_capture_mode: bool,
    is_gfx950: bool,
    tp_world_size: int,
    ep_world_size: int,
    hip_version: tuple[int, ...],
) -> bool:
    return bool(
        is_target_layer
        and mode == "optimized"
        and is_graph_capture_mode
        and is_gfx950
        and hip_version >= (7, 2)
        and tp_world_size == 4
        and ep_world_size == 1
        and residual is not None
        and hidden_states.device.type == "cuda"
        and residual.device == hidden_states.device
        and weight.device == hidden_states.device
        and hidden_states.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.dim() == 2
        and hidden_states.shape == residual.shape
        and hidden_states.shape[0] in _FUSED_AR_MXFP4_TOKEN_COUNTS
        and hidden_states.shape[1] == _KIMI_HIDDEN_SIZE
        and weight.numel() == _KIMI_HIDDEN_SIZE
        and hidden_states.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
    )


def attach_rocm_mxfp4_activation(
    hidden_states: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    if hasattr(hidden_states, _MXFP4_ACTIVATION_ATTR):
        raise RuntimeError("ROCm MXFP4 activation carrier was already populated")
    token_count = hidden_states.shape[0] if hidden_states.dim() == 2 else -1
    if (
        token_count not in _FUSED_AR_MXFP4_TOKEN_COUNTS
        or hidden_states.shape != (token_count, _KIMI_HIDDEN_SIZE)
        or packed.shape != (token_count, _MXFP4_PACKED_K)
        or scale.shape != (token_count, _MXFP4_SCALE_K)
        or hidden_states.dtype != torch.bfloat16
        or packed.dtype != torch.uint8
        or scale.dtype != torch.uint8
        or packed.device != hidden_states.device
        or scale.device != hidden_states.device
        or not packed.is_contiguous()
        or not scale.is_contiguous()
    ):
        raise RuntimeError(
            "invalid ROCm fused AR MXFP4 activation carrier: "
            f"hidden={tuple(hidden_states.shape)}/{hidden_states.dtype}, "
            f"packed={tuple(packed.shape)}/{packed.dtype}, "
            f"scale={tuple(scale.shape)}/{scale.dtype}"
        )
    setattr(
        hidden_states,
        _MXFP4_ACTIVATION_ATTR,
        RocmMxfp4ActivationCarrier(
            source_data_ptr=hidden_states.data_ptr(),
            source_shape=tuple(hidden_states.shape),
            source_dtype=hidden_states.dtype,
            source_device=hidden_states.device,
            packed=packed,
            scale=scale,
        ),
    )


def pop_rocm_mxfp4_activation(
    hidden_states: torch.Tensor,
) -> Optional[RocmMxfp4ActivationCarrier]:
    carrier = getattr(hidden_states, _MXFP4_ACTIVATION_ATTR, None)
    if hasattr(hidden_states, _MXFP4_ACTIVATION_ATTR):
        delattr(hidden_states, _MXFP4_ACTIVATION_ATTR)
    if carrier is None:
        return None
    if (
        carrier.source_data_ptr != hidden_states.data_ptr()
        or carrier.source_shape != tuple(hidden_states.shape)
        or carrier.source_dtype != hidden_states.dtype
        or carrier.source_device != hidden_states.device
    ):
        raise RuntimeError("stale or mismatched ROCm MXFP4 activation carrier")
    return carrier


def validate_rocm_mxfp4_shared_fc1(
    carrier: RocmMxfp4ActivationCarrier,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> bool:
    token_count = carrier.source_shape[0] if len(carrier.source_shape) == 2 else -1
    return bool(
        token_count in _FUSED_AR_MXFP4_TOKEN_COUNTS
        and carrier.source_shape == (token_count, _KIMI_HIDDEN_SIZE)
        and carrier.packed.shape == (token_count, _MXFP4_PACKED_K)
        and carrier.scale.shape == (token_count, _MXFP4_SCALE_K)
        and weight.shape == (_TARGET_FC1_N, _MXFP4_PACKED_K)
        and weight_scale.shape == (_TARGET_FC1_N, _MXFP4_SCALE_K)
        and carrier.packed.dtype == torch.uint8
        and carrier.scale.dtype == torch.uint8
        and weight.dtype == torch.uint8
        and weight_scale.dtype == torch.uint8
        and carrier.packed.device == weight.device == weight_scale.device
    )


def rocm_mxfp4_moe_add_shared(
    routed_output: torch.Tensor,
    shared_output: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    try:
        from sgl_kernel import rocm_mxfp4_moe_add_shared as add_shared

        return add_shared(
            routed_output,
            shared_output,
            output=output,
        )
    except ImportError:
        if output is None:
            return routed_output + shared_output
        return torch.add(routed_output, shared_output, out=output)


def can_defer_shared_partial_to_graph_ar(
    routed_output: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    *,
    should_allreduce_fusion: bool,
    shared_expert_tp1: bool,
    tp_world_size: int,
    is_graph_capture_mode: bool,
    is_gfx950: bool,
) -> bool:
    return bool(
        envs.SGLANG_ROCM_FUSE_SHARED_PARTIAL_AR_RMSNORM.get()
        and should_allreduce_fusion
        and not shared_expert_tp1
        and is_graph_capture_mode
        and is_gfx950
        and tp_world_size == 4
        and shared_output is not None
        and routed_output.device.type == "cuda"
        and shared_output.device == routed_output.device
        and routed_output.dtype == torch.bfloat16
        and shared_output.dtype == torch.bfloat16
        and routed_output.dim() == 2
        and routed_output.shape == shared_output.shape
        and routed_output.shape[0] in _SUPPORTED_TOKEN_COUNTS
        and routed_output.shape[1] == _KIMI_HIDDEN_SIZE
        and routed_output.is_contiguous()
        and shared_output.is_contiguous()
    )


def attach_shared_partial(
    routed_output: torch.Tensor,
    shared_output: torch.Tensor,
) -> torch.Tensor:
    if hasattr(routed_output, _SHARED_PARTIAL_ATTR):
        raise RuntimeError("ROCm shared partial carrier was already populated")
    setattr(routed_output, _SHARED_PARTIAL_ATTR, shared_output)
    return routed_output


def pop_shared_partial(routed_output: torch.Tensor) -> Optional[torch.Tensor]:
    shared_output = getattr(routed_output, _SHARED_PARTIAL_ATTR, None)
    if hasattr(routed_output, _SHARED_PARTIAL_ATTR):
        delattr(routed_output, _SHARED_PARTIAL_ATTR)
    return shared_output
