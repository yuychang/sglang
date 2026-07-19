# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import functools
import logging
import math
from typing import Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    get_hip_version,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_musa,
    is_npu,
    is_nvidia_cublas_version_ge_12_9,
    is_xpu,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_musa = is_musa()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
_use_aiter_bpreshuffle_gfx95 = _use_aiter_gfx95 and get_hip_version() >= (7, 2, 0)


_is_cublas_ge_129 = is_nvidia_cublas_version_ge_12_9()

logger = logging.getLogger(__name__)

# Attention-projection stems that may be quantized to fp8 for nvfp4 checkpoints
# when SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN is set. Only standalone dense decode
# projections with a clean 1:1 checkpoint tensor belong here.
_DEFAULT_FP8_ATTN_QUANT_MODULES = ("q_b_proj",)
_ALLOWED_FP8_ATTN_QUANT_MODULES = ("q_b_proj", "o_proj")
# Stems that must never be fp8-quantized on this path:
# - kv_b_proj: under MLA weight absorption it becomes the w_kc/w_vc batched GEMM;
#   its weights are already fp4 on gfx950, so fp8 would double the bytes on a
#   memory-bound op and regress decode.
# - fused_qkv_a_proj_with_mqa: fused at load from separate q_a_proj +
#   kv_a_proj_with_mqa checkpoint tensors, so it has no 1:1 checkpoint weight for
#   the load-time quant step.
_FORBIDDEN_FP8_ATTN_QUANT_MODULES = ("kv_b_proj", "fused_qkv_a_proj_with_mqa")

# Back-compat alias for external importers.
NVFP4_CKPT_FP8_ATTN_QUANT_MODULES = list(_DEFAULT_FP8_ATTN_QUANT_MODULES)


@functools.lru_cache(maxsize=None)
def _resolve_fp8_attn_quant_modules(requested: tuple) -> tuple:
    """Pure resolver, cached on ``requested`` so warnings fire once per value.

    Keyed on the requested tuple (not on nothing), so a test that overrides
    ``SGLANG_NVFP4_CKPT_FP8_ATTN_MODULES`` still re-resolves under the new value
    while the 60 per-layer construction calls that share one value are deduped.
    """
    if not requested:
        return _DEFAULT_FP8_ATTN_QUANT_MODULES

    resolved = []
    for stem in requested:
        if stem in _FORBIDDEN_FP8_ATTN_QUANT_MODULES:
            logger.warning(
                "SGLANG_NVFP4_CKPT_FP8_ATTN_MODULES: refusing to fp8-quantize "
                "'%s' (absorbed or load-time-fused projection); skipping.",
                stem,
            )
        elif stem not in _ALLOWED_FP8_ATTN_QUANT_MODULES:
            logger.warning(
                "SGLANG_NVFP4_CKPT_FP8_ATTN_MODULES: unknown attention "
                "projection '%s'; allowed stems are %s. Skipping.",
                stem,
                ", ".join(_ALLOWED_FP8_ATTN_QUANT_MODULES),
            )
        elif stem not in resolved:
            resolved.append(stem)

    if not resolved:
        logger.warning(
            "SGLANG_NVFP4_CKPT_FP8_ATTN_MODULES had no usable stems; "
            "falling back to default %s.",
            _DEFAULT_FP8_ATTN_QUANT_MODULES,
        )
        return _DEFAULT_FP8_ATTN_QUANT_MODULES
    return tuple(resolved)


def resolve_fp8_attn_quant_modules() -> list[str]:
    """Resolve which attention projections to fp8-quantize for nvfp4 checkpoints.

    Reads the ``SGLANG_NVFP4_CKPT_FP8_ATTN_MODULES`` override (comma-separated
    stems); an empty override falls back to the default ``("q_b_proj",)`` so the
    behavior is unchanged unless the operator opts in. Requested stems that are
    forbidden (absorbed / load-time-fused) or unknown are dropped with a warning
    so a bad override degrades to the safe default instead of a silent regression
    or a load-time crash.
    """
    return list(_resolve_fp8_attn_quant_modules(envs.SGLANG_NVFP4_CKPT_FP8_ATTN_MODULES.get()))

FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "cutedsl_mla",
    "tokenspeed_mla",
    "ascend",
    "intel_xpu",
]


def awq_dequantize_func():
    """
    Get the AWQ dequantize function for the current device

    Return:
        - The AWQ dequantize function for the current device.
        - None if the current device is not supported.
    """
    if _is_cuda:
        from sgl_kernel import awq_dequantize

        return awq_dequantize
    elif _is_hip:
        from sglang.kernel_api_logging import debug_kernel_api
        from sglang.kernels.ops.quantization.awq_triton import (
            awq_dequantize_triton as awq_dequantize,
        )

        return debug_kernel_api(awq_dequantize, op_name="DeepseekCommon.awq_dequantize")
    elif _is_npu:
        from sglang.kernel_api_logging import debug_kernel_api
        from sglang.kernels.ops.quantization.awq_triton import (
            awq_dequantize_decomposition as awq_dequantize,
        )

        return debug_kernel_api(awq_dequantize, op_name="DeepseekCommon.awq_dequantize")
    else:
        return None


def enable_nextn_moe_bf16_cast_to_fp8(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    return (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
        and get_moe_runner_backend().is_deep_gemm()
    )


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    return scaling[..., None, None]
