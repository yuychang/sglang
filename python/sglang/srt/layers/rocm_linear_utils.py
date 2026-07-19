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
    # 256 = DeepSeek-V3; 384 = Kimi K2.5. Both share embedding_dim 7168 and the
    # same gfx95 pre-zeroed shared-expert GEMM path.
    if embedding_dim != 7168 or n_routed_experts not in (256, 384):
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size
