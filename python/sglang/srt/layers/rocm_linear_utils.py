import torch
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat
from aiter.tuned_gemm import tgemm

__all__ = ["fused_qk_rope_cat", "fused_qk_rope_cat_and_cache_mla"]


def aiter_dsv3_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """Use aiter tuned GEMM dispatcher (tgemm.mm) to automatically select the GEMM kernel."""
    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)


def get_dsv3_gemm_output_zero_allocator_size(
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int
):
    # DeepSeek-V3/R1 (256) and Kimi-K2.5 (384) both use the gfx95 AITER MXFP4
    # decode shape with hidden 7168 and a single shared expert, so both can reuse
    # the pre-zeroed shared-expert gate_up output buffer.
    if embedding_dim != 7168 or n_routed_experts not in (256, 384):
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size
