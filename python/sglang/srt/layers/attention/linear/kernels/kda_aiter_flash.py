"""ROCm prefill KDA via AITER FlashKDA, with a Triton fallback.

FlashKDA keeps the recurrent state in registers and writes the V-first pool
in place. It matches SGLang's chunk_kda to bf16 on a nonzero state, including
a bf16 or fp32 paged pool. Interior radix snapshots are the state entering a
64-token Triton chunk, which is flash chunk ``index * 2``. Speculative
draft-extend stays on Triton: this kernel commits the slot before returning.
"""

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils.common import is_gfx95_supported, rank0_log

_K = 128
_FLASH_CHUNK = 32
_TRACK_CHUNK = 64
_LOWER_MIN = -5.0
_LOWER_MAX = 0.0


def _triton_extend(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log,
    dt_bias,
    lower_bound,
    beta_is_raw,
    return_intermediate_states,
    kwargs,
):
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
        output_intermediate_states=return_intermediate_states,
        track_state=kwargs.get("track_state"),
        track_chunk_idx=kwargs.get("track_chunk_idx"),
    )


def _state_plane_ok(state: torch.Tensor) -> bool:
    """Each slot is a dense fp32 or bf16 ``[H, V, K]``; ``stride(0)`` may be padded."""
    if state.dtype not in (torch.float32, torch.bfloat16) or state.dim() != 4:
        return False
    _slots, h, v, k = state.shape
    if k != _K or v != _K:
        return False
    return (
        state.stride(-1) == 1
        and state.stride(2) == k
        and state.stride(1) == v * k
        and state.stride(0) >= h * v * k
    )


class AiterFlashKDAKernel(LinearAttnKernelBase):
    """gfx95 KDA extend through ``aiter`` FlashKDA.

    The fast path passes raw beta and lets the kernel sigmoid it, and it
    addresses the live V-first pool (``ssm_states``) by slot. An interior radix
    snapshot is written as the state entering that 64-token chunk. Speculative
    draft-extend falls back to Triton ``chunk_kda``. Aligned snapshots only
    need the final state, which this path already wrote into the slot.
    """

    # Tracked batches are either served (aligned: final state is the slot) or
    # routed to Triton, which writes the fp32 snapshot. Never leave the buffer
    # unwritten.
    supports_track_state_snapshot: bool = True

    def __init__(self):
        from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

        self._fwd = chunk_kimi_delta_attn
        self._falls: dict[str, int] = {}
        self._hits = 0

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
        snap_chunk, snap_state, track_reason = self._snapshot_args(
            query_start_loc,
            return_intermediate_states=return_intermediate_states,
            track_h_src=kwargs.get("track_ssm_h_src"),
            track_chunk_idx=kwargs.get("track_chunk_idx"),
            track_state=kwargs.get("track_state"),
        )
        reason = track_reason or self._fallback_reason(
            q,
            k,
            v,
            g,
            beta,
            ssm_states,
            cache_indices,
            query_start_loc,
            A_log=A_log,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
            is_spec_decode=is_spec_decode,
            prefix=kwargs.get("extend_prefix_lens"),
        )
        if reason is not None:
            self._log_fallback(reason, q, query_start_loc.shape[0] - 1)
            return _triton_extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
                lower_bound,
                beta_is_raw,
                return_intermediate_states,
                kwargs,
            )

        n = query_start_loc.shape[0] - 1
        idx = cache_indices.to(torch.int32)
        prefix = kwargs.get("extend_prefix_lens")
        if prefix is None:
            # Same contract as chunk_kda: a cleared slot reads as zero.
            has_initial = idx >= 0
        else:
            has_initial = (idx >= 0) & (prefix.to(device=idx.device) > 0)
        # Padded rows use the -1 sentinel. Point them at the pool's extra slot
        # and do not read it; the store lands on that discard row.
        pad_slot = ssm_states.shape[0] - 1
        idx = torch.where(idx >= 0, idx, torch.full_like(idx, pad_slot))
        saved = ssm_states.index_select(0, idx)
        try:
            out = torch.empty_like(v)
            self._fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta if beta.dtype == torch.float32 else beta.float(),
                A_log=A_log.reshape(-1).float(),
                dt_bias=None if dt_bias is None else dt_bias.reshape(-1).float(),
                scale=q.shape[-1] ** -0.5,
                chunk_size=32,
                safe_gate=True,
                lower_bound=float(lower_bound),
                use_gate_in_kernel=True,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                state_v_first=True,
                cu_seqlens=query_start_loc,
                out=out,
                state_cache=ssm_states,
                state_indices=idx,
                has_initial_state=has_initial,
                snapshot_chunk=snap_chunk,
                snapshot_state=snap_state,
            )
        except Exception:
            ssm_states.index_copy_(0, idx, saved)
            self._log_fallback("launch", q, n)
            return _triton_extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
                lower_bound,
                beta_is_raw,
                return_intermediate_states,
                kwargs,
            )

        self._note("hit", q, n)
        if return_intermediate_states:
            # Aligned rows copy the slot below. Interior rows were written to
            # track_state inside the kernel, so there is no per-chunk h.
            return out, None
        return out

    def _note(self, reason: str, q: torch.Tensor, n: int) -> None:
        if reason == "hit":
            self._hits += 1
            count = self._hits
        else:
            self._falls[reason] = self._falls.get(reason, 0) + 1
            count = self._falls[reason]
        if count != 1 and count % 256 != 0:
            return
        rank0_log(
            "KDA prefill extend "
            f"{'running' if reason == 'hit' else 'skipped'} AITER FlashKDA "
            f"({reason}) H={q.shape[2]} K={q.shape[-1]} sequences={n} "
            f"hits={self._hits} falls={self._falls}"
        )

    def _log_fallback(self, reason: str, q: torch.Tensor, n: int) -> None:
        self._note(reason, q, n)

    @staticmethod
    def _snapshot_args(
        query_start_loc: torch.Tensor,
        *,
        return_intermediate_states: bool,
        track_h_src,
        track_chunk_idx,
        track_state,
    ):
        """Flash-chunk index of each interior radix snapshot, or a fallback reason.

        The radix grid is the Triton 64-token chunk. FlashKDA steps by 32, and
        the tracked value is the state entering that 64-token chunk.
        """
        if not return_intermediate_states:
            return None, None, None
        if track_h_src is None:
            return None, None, "track"
        if track_h_src.numel() == 0:
            return None, None, None
        n = query_start_loc.shape[0] - 1
        if (
            track_chunk_idx is None
            or track_state is None
            or track_chunk_idx.shape[0] != n
            or track_state.shape[0] != n
            or _TRACK_CHUNK % _FLASH_CHUNK != 0
        ):
            return None, None, "track"
        scale = _TRACK_CHUNK // _FLASH_CHUNK
        snap_chunk = torch.where(
            track_chunk_idx < 0, track_chunk_idx, track_chunk_idx * scale
        ).to(torch.int32)
        return snap_chunk, track_state, None

    @staticmethod
    def _fallback_reason(
        q,
        k,
        v,
        g,
        beta,
        ssm_states,
        cache_indices,
        query_start_loc,
        *,
        A_log,
        lower_bound,
        beta_is_raw,
        is_spec_decode,
        prefix,
    ) -> Optional[str]:
        if is_spec_decode:
            return "spec"
        if not beta_is_raw:
            return "beta"
        if A_log is None or lower_bound is None:
            return "gate"
        if not (_LOWER_MIN <= float(lower_bound) < _LOWER_MAX):
            return "gate"
        if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
            return "varlen"
        n = query_start_loc.shape[0] - 1
        if (
            q.ndim != 4
            or q.shape[0] != 1
            or k.shape != q.shape
            or v.ndim != 4
            or v.shape[0] != 1
            or v.shape[1] != q.shape[1]
            or v.shape[2] != q.shape[2]
            or q.shape[-1] != _K
            or v.shape[-1] != _K
            or q.dtype != torch.bfloat16
            or v.dtype != torch.bfloat16
            or g.dtype != torch.bfloat16
            or g.shape != (1, q.shape[1], q.shape[2], _K)
            or beta.shape[:3] != (1, q.shape[1], q.shape[2])
        ):
            return "shape"
        if not _state_plane_ok(ssm_states):
            return "state"
        if ssm_states.shape[1] != q.shape[2]:
            return "state"
        if cache_indices.numel() != n:
            return "indices"
        if prefix is not None and prefix.numel() != n:
            return "prefix"
        return None


def maybe_create_aiter_flash_kda_kernel() -> Optional["AiterFlashKDAKernel"]:
    """gfx95 opt-in replacement for the Triton extend kernel, or None.

    Decode and verify stay where the flags put them; an AITER import failure
    leaves extend on Triton.
    """
    if not (envs.SGLANG_AITER_KDA_FLASH_PREFILL.get() and is_gfx95_supported()):
        return None
    try:
        return AiterFlashKDAKernel()
    except Exception as exc:
        rank0_log(
            f"AITER FlashKDA prefill unavailable ({exc}); KDA extend stays on Triton."
        )
        return None
