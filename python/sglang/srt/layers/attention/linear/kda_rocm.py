"""Kimi-K3 KDA on ROCm: vLLM's fused HIP decode and chunk prefill.

``KDAAttnBackend`` calls into here only under ``is_hip()``. Decode runs
``fused_kda_decode`` (conv1d update, gated delta rule and the gated sigmoid
RMSNorm in one launch) once the model has staged its static weights on the
attention layer; prefill runs ``chunk_kda_prefill`` (the gfx950 fused chunk
kernel, else the Triton chunk path) on the raw gate and beta, reading and
writing the paged recurrent state in place, as vLLM's ROCm KDA layer does.
"""

from typing import Optional

import torch

from sglang.kernels.ops.attention.kda_rocm.kda_decode import (
    fused_kda_decode,
    is_fused_kda_decode_supported,
    stage_decode_conv1d_weight,
    stage_decode_norm_weight,
)
from sglang.kernels.ops.attention.kda_rocm.kda_prefill import chunk_kda_prefill


def stage_fused_decode(layer, norm_weight: torch.Tensor, norm_eps: float) -> bool:
    """Stash the fused decode's static inputs on ``layer``; False if uncovered."""
    w = layer.conv_weights
    if (
        w is None
        or layer.bias is not None
        or layer.A_log is None
        or layer.dt_bias is None
        or layer.lower_bound is None
        or not is_fused_kda_decode_supported(
            layer.num_v_heads,
            layer.head_v_dim,
            w.shape[-1],
            0,
            torch.bfloat16,
            torch.bfloat16,
        )
    ):
        return False
    layer._k3_rocm_decode_args = (
        stage_decode_conv1d_weight(w),
        stage_decode_norm_weight(norm_weight),
        float(norm_eps),
        layer.A_log.detach().reshape(-1).float().contiguous(),
        layer.dt_bias.detach().reshape(-1).float().contiguous(),
    )
    return True


def forward_decode(
    layer,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Fused decode with the output norm folded in, or None if not covered.

    ``a`` is the raw forget gate ``[B, H * D]``, ``b`` the raw beta
    ``[1, B, H]`` and ``conv_states`` the ``[slots, W - 1, 3 * H * D]`` pool.
    Rows whose ``cache_indices`` entry is ``<= 0`` (CUDA-graph padding and the
    reserved slot 0) produce zeros and touch no state.
    """
    args = getattr(layer, "_k3_rocm_decode_args", None)
    gate = getattr(layer, "_k3_onorm_gate", None)
    if args is None or gate is None:
        return None
    num_tokens = mixed_qkv.shape[0]
    if (
        num_tokens != cache_indices.shape[0]
        or mixed_qkv.dtype != torch.bfloat16
        or mixed_qkv.stride(-1) != 1
        or conv_states.dtype != torch.bfloat16
        or ssm_states.dtype != torch.float32
        or b.ndim != 3
    ):
        return None
    conv_weight, norm_weight, norm_eps, A_log, dt_bias = args
    heads, head_dim = layer.num_v_heads, layer.head_v_dim
    out = mixed_qkv.new_empty((1, num_tokens, heads, head_dim))
    fused_kda_decode(
        x=mixed_qkv,
        weight=conv_weight,
        bias=None,
        conv_state=conv_states.transpose(-1, -2),
        raw_g=a.reshape(1, num_tokens, heads, head_dim).contiguous(),
        raw_beta=b if b.stride(-1) == 1 else b.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        state_indices=cache_indices.to(torch.int32).contiguous(),
        state=ssm_states,
        out=out,
        lower_bound=float(layer.lower_bound),
        output_gate=gate.unflatten(-1, (heads, head_dim)),
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    layer._k3_onorm_consumed = True
    return out


def forward_extend(
    layer,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Chunk prefill from the raw gate ``a [1, T, H, D]`` and beta ``b [1, T, H]``."""
    out, _ = chunk_kda_prefill(
        q=q,
        k=k,
        v=v,
        raw_g=a,
        raw_beta=b,
        A_log=layer.A_log.reshape(-1),
        g_bias=layer.dt_bias,
        lower_bound=layer.lower_bound,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc.to(torch.int32),
        use_fused_chunk=True,
        state_cache=ssm_states,
        state_indices=cache_indices,
        has_initial_state=has_initial_state,
    )
    return out
