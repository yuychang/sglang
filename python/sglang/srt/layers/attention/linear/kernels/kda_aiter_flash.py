"""ROCm gfx95x KDA prefill through AITER FlashKDA, with a Triton chunk_kda fallback.

FlashKDA writes the final state into the V-first pool in place and never materializes
per-chunk states, so interior track snapshots and draft-extend batches run Triton.
"""

from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils.common import rank0_log

# AITER flash_kda_supported(): K == V == 128 and no GVA.
_HEAD_DIM = 128
# AITER safe_gate rejects lower_bound outside [-5, 0); Kimi checkpoints use -5.0.
_LOWER_BOUND_MIN = -5.0
_LOWER_BOUND_MAX = 0.0
# AITER FLASH_KDA_CHUNK; any other chunk size takes the default pipeline,
# which rejects the paged state_cache.
_FLASHKDA_CHUNK_SIZE = 32


def _load_chunk_kimi_delta_attn():
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    return chunk_kimi_delta_attn


def _state_pool_ok(ssm_states: torch.Tensor, num_heads: int) -> bool:
    # AITER paged state_cache: fp32 [slots, H, V, K], dense per slot, stride(0) may be padded.
    if ssm_states.dtype != torch.float32 or ssm_states.dim() != 4:
        return False
    _, h, v, k = ssm_states.shape
    return (
        h == num_heads
        and v == _HEAD_DIM
        and k == _HEAD_DIM
        and ssm_states.stride(3) == 1
        and ssm_states.stride(2) == k
        and ssm_states.stride(1) == v * k
        and ssm_states.stride(0) >= h * v * k
    )


def _inputs_ok(
    *, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta
) -> bool:
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape:
        return False
    _, num_tokens, num_heads, head_dim = q.shape
    return (
        head_dim == _HEAD_DIM
        and v.shape == (1, num_tokens, num_heads, _HEAD_DIM)
        and g.shape == (1, num_tokens, num_heads, _HEAD_DIM)
        and beta.shape == (1, num_tokens, num_heads)
        and q.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
    )


def _fallback_reason(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    A_log: Optional[torch.Tensor],
    lower_bound: Optional[float],
    beta_is_raw: bool,
    return_intermediate_states: bool,
    is_spec_decode: bool,
    extend_prefix_lens: Optional[torch.Tensor],
    track_ssm_h_src: Optional[torch.Tensor],
) -> Optional[str]:
    """Why this batch must run Triton, or None; host-side checks only (no GPU sync)."""
    # FlashKDA commits the slot in place; draft-extend must stay rollback-able.
    if is_spec_decode:
        return "spec"
    # An empty src means every tracked row only needs the final state.
    if return_intermediate_states and (
        track_ssm_h_src is None or track_ssm_h_src.numel() > 0
    ):
        return "track"
    # The kernel sigmoids beta itself (use_beta_sigmoid_in_kernel).
    if not beta_is_raw:
        return "beta"
    if A_log is None or lower_bound is None:
        return "gate"
    if not _LOWER_BOUND_MIN <= float(lower_bound) < _LOWER_BOUND_MAX:
        return "gate"
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        return "varlen"
    if not _inputs_ok(q=q, k=k, v=v, g=g, beta=beta):
        return "shape"
    if not _state_pool_ok(ssm_states, num_heads=q.shape[2]):
        return "state"
    num_seqs = query_start_loc.shape[0] - 1
    if cache_indices.numel() != num_seqs:
        return "indices"
    if extend_prefix_lens is not None and extend_prefix_lens.numel() != num_seqs:
        return "prefix"
    return None


def _flash_state_rows(
    *,
    cache_indices: torch.Tensor,
    extend_prefix_lens: Optional[torch.Tensor],
    pad_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Int32 pool row per sequence and its has_initial_state mask."""
    rows = cache_indices.to(torch.int32)
    valid = rows >= 0
    has_initial = valid
    if extend_prefix_lens is not None:
        has_initial = valid & (extend_prefix_lens.to(device=rows.device) > 0)
    # Padded rows carry the -1 sentinel; AITER trusts indices unchecked, so send
    # them to the pool's spare last row (MambaPool allocates size + 1) unread.
    rows = torch.where(valid, rows, torch.full_like(rows, pad_slot))
    return rows, has_initial


class AiterFlashKDAKernel(LinearAttnKernelBase):
    """gfx95x KDA extend through AITER FlashKDA; unsupported batches run Triton chunk_kda."""

    # Interior track snapshots route to Triton, which writes them; aligned ones
    # only need the final state, which this kernel writes into the slot.
    supports_track_state_snapshot: bool = True

    def __init__(self, triton_fallback: TritonKDAKernel):
        self._fwd = _load_chunk_kimi_delta_attn()
        self._triton = triton_fallback
        self._logged_fallbacks: set[str] = set()
        self._logged_hit = False

    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("AiterFlashKDAKernel only supports extend")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        is_spec_decode: bool = False,
        **kwargs,
    ):
        extend_prefix_lens = kwargs.get("extend_prefix_lens")
        reason = _fallback_reason(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
            return_intermediate_states=return_intermediate_states,
            is_spec_decode=is_spec_decode,
            extend_prefix_lens=extend_prefix_lens,
            track_ssm_h_src=kwargs.get("track_ssm_h_src"),
        )
        if reason is not None:
            self._log_fallback(reason)
            return self._triton.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                **kwargs,
            )

        out = self._flash_extend(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            extend_prefix_lens=extend_prefix_lens,
        )
        if return_intermediate_states:
            # Aligned track rows copy the final state from the slot; there is no per-chunk h.
            return out, None
        return out

    def _flash_extend(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: Optional[torch.Tensor],
        lower_bound: float,
        extend_prefix_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        rows, has_initial = _flash_state_rows(
            cache_indices=cache_indices,
            extend_prefix_lens=extend_prefix_lens,
            pad_slot=ssm_states.shape[0] - 1,
        )
        out = torch.empty_like(v)
        self._fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta.float(),
            A_log=A_log.reshape(-1).float(),
            dt_bias=None if dt_bias is None else dt_bias.reshape(-1).float(),
            scale=q.shape[-1] ** -0.5,
            chunk_size=_FLASHKDA_CHUNK_SIZE,
            safe_gate=True,
            lower_bound=float(lower_bound),
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
            cu_seqlens=query_start_loc,
            out=out,
            state_cache=ssm_states,
            state_indices=rows,
            has_initial_state=has_initial,
        )
        if not self._logged_hit:
            self._logged_hit = True
            rank0_log(
                f"KDA prefill running AITER FlashKDA (H={q.shape[2]} K={q.shape[-1]})"
            )
        return out

    def _log_fallback(self, reason: str) -> None:
        if reason in self._logged_fallbacks:
            return
        self._logged_fallbacks.add(reason)
        rank0_log(f"KDA AITER FlashKDA skipped ({reason}); this batch uses Triton.")
