import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu, is_xpu

logger = logging.getLogger(__name__)

# gfx950 Gluon chunk_kda is the measured prefill path. Empty string means the
# arch probe already failed and must not be retried on every extend.
_AITER_KDA_ARCH: Optional[str] = None

if not is_cpu():
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm import (
        fused_recurrent_linear_replayssm_decode,
    )
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )


def _aiter_kda_arch() -> str:
    global _AITER_KDA_ARCH
    if _AITER_KDA_ARCH is None:
        try:
            from aiter.ops.triton.utils._triton import arch_info

            _AITER_KDA_ARCH = arch_info.get_arch()
        except Exception:
            _AITER_KDA_ARCH = ""
    return _AITER_KDA_ARCH


def _dense_head(x: torch.Tensor, head_dim: int) -> bool:
    return (
        x.ndim == 4
        and x.shape[0] == 1
        and x.shape[-1] == head_dim
        and x.stride()[-2:] == (head_dim, 1)
        and x.stride(1) % 8 == 0
    )


def _aiter_chunk_kda_applies(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    lower_bound: Optional[float],
    beta_is_raw: bool,
    return_intermediate_states: bool,
    track_state: Optional[torch.Tensor],
) -> bool:
    """True when the gfx950 Gluon prefill matches this extend.

    The Gluon kernel fuses L2-norm, the safe gate, and beta sigmoid, and it
    writes one final state per sequence. Interior chunk snapshots and an
    already-activated beta stay on Triton chunk_kda.
    """
    if _aiter_kda_arch() != "gfx950":
        return False
    if (
        not beta_is_raw
        or return_intermediate_states
        or track_state is not None
        or lower_bound is None
        or A_log is None
        or dt_bias is None
    ):
        return False
    if not (-5.5 <= float(lower_bound) < 0):
        return False
    if query_start_loc is None or query_start_loc.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False
    head_dim = 128
    if not _dense_head(q, head_dim):
        return False
    if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        return False
    if not (
        _dense_head(k, head_dim)
        and _dense_head(v, head_dim)
        and _dense_head(g, head_dim)
    ):
        return False
    num_heads = q.shape[2]
    if beta.shape != (1, q.shape[1], num_heads) or beta.stride(-1) != 1:
        return False
    if A_log.numel() != num_heads or dt_bias.numel() != num_heads * head_dim:
        return False
    if ssm_states.ndim != 4 or tuple(ssm_states.shape[1:]) != (
        num_heads,
        head_dim,
        head_dim,
    ):
        return False
    return True


def _aiter_chunk_kda_extend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    """Gather the indexed bf16/fp32 slots, run Gluon, scatter the final state."""
    from aiter.ops.triton.attention.chunk_kda import chunk_kda as aiter_chunk_kda

    slots = cache_indices
    if slots.dtype not in (torch.int32, torch.int64):
        slots = slots.to(torch.int64)
    # A negative slot is padding. Clamp only for the gather address; the
    # stored value for that row is the original slot 0, so the copy is a no-op
    # unless a real sequence also owns slot 0.
    dest = torch.where(slots >= 0, slots, torch.zeros_like(slots)).to(torch.long)
    gathered = ssm_states.index_select(0, dest)
    valid = (slots >= 0).view(-1, 1, 1, 1)
    initial = torch.where(
        valid, gathered.float(), torch.zeros_like(gathered, dtype=torch.float32)
    )
    out, final = aiter_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log.reshape(-1).contiguous(),
        dt_bias.reshape(-1).contiguous(),
        float(lower_bound),
        query_start_loc.contiguous(),
        initial_state=initial,
        output_final_state=True,
    )
    current = ssm_states.index_select(0, dest)
    written = torch.where(valid, final.to(dtype=ssm_states.dtype), current)
    ssm_states.index_copy_(0, dest, written)
    return out


class TritonKDAKernel(LinearAttnKernelBase):
    """Triton-based kernel for KDA (Kimi Delta Attention) linear attention."""

    # XPU has no tvm_ffi CUDA JIT kernel for KDA packed decode; route XPU to the
    # non-packed Triton decode() path (fused_sigmoid_gating_delta_rule_update),
    # the same fallback CPU/NPU use. Batched decode is handled via query_start_loc.
    supports_packed_decode: bool = not is_cpu() and not is_npu() and not is_xpu()
    supports_fused_chain_verify: bool = not is_cpu() and not is_npu()
    supports_track_state_snapshot: bool = True
    _logged_aiter_extend: bool = False

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: feed the conv-1d output ``mixed_qkv``
        straight into a single fused Triton kernel that does Q/K/V extraction,
        gate/beta computation, l2-norm, and the recurrent state update.

        Returns output tensor of shape [1, B, HV, V] to match the existing
        decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        # KDA ReplaySSM buffered decode: drop-in for the packed decode, same
        # args plus the three per-layer ring caches + the per-row write cursor
        # (and optional radix-track force-flush). Uses the gate-generic kernel
        # with is_kda=True (per-K gate); g_cache is [num_slots, HV, L, K].
        # When any ring tensor / cursor is None (flag off) we fall through to
        # the byte-identical legacy path below.
        replayssm_d = kwargs.get("replayssm_d")
        replayssm_k = kwargs.get("replayssm_k")
        replayssm_g = kwargs.get("replayssm_g")
        replayssm_write_pos = kwargs.get("replayssm_write_pos")
        replayssm_force_flush = kwargs.get("replayssm_force_flush")
        if (
            lower_bound is None
            and replayssm_d is not None
            and replayssm_k is not None
            and replayssm_g is not None
            and replayssm_write_pos is not None
        ):
            if lower_bound is not None:
                raise NotImplementedError(
                    "KDA safe gate (lower_bound) is not implemented in the "
                    "ReplaySSM decode kernel; disable --enable-linear-replayssm."
                )
            K = ssm_states.shape[-1]  # ssm_states: [num_slots, HV, V, K]
            fused_recurrent_linear_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a.reshape(B, num_v_heads, K).contiguous(),
                b=b.reshape(B, num_v_heads).contiguous(),
                A_log=A_log.reshape(-1),
                dt_bias=dt_bias.reshape(num_v_heads, K).contiguous(),
                scale=scale,
                initial_state=ssm_states,
                d_cache=replayssm_d,
                k_cache=replayssm_k,
                g_cache=replayssm_g,
                out=out,
                ssm_state_indices=cache_indices,
                write_pos=replayssm_write_pos,
                force_flush=replayssm_force_flush,
                use_qk_l2norm_in_kernel=True,
                is_kda=True,
            )
            return out.transpose(0, 1)

        # a may come in as [B, HV, K] (or [B, 1, HV*K]); b may come in as
        # [B, 1, HV]. Flatten both to the 2D shapes the kernel expects.
        if a.dim() != 2:
            a = a.reshape(B, -1)
        if b.dim() != 2:
            b = b.reshape(B, -1)
        fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            lower_bound=lower_bound,
        )
        # [B, 1, HV, V] -> [1, B, HV, V] view to match existing decode layout.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            lower_bound=lower_bound,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: Optional[torch.Tensor],
        lower_bound: Optional[float] = None,
        # fused ReplaySSM ring-write (dense verify only; off elsewhere).
        cache_ring: bool = False,
        replayssm_rawv: Optional[torch.Tensor] = None,
        replayssm_rawk: Optional[torch.Tensor] = None,
        replayssm_g: Optional[torch.Tensor] = None,
        replayssm_beta: Optional[torch.Tensor] = None,
        ring_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # KDA MTP / speculative-decode verify via the fused KDA kernel (IS_KDA=True),
        # mirroring the GDN triton verify path. Reads the committed state, writes
        # per-draft-token intermediate states to the scratch buffer, does NOT mutate
        # the committed pool (disable_state_update=True), and handles chain + tree
        # (retrieve_parent_token). The verify kernel for the Triton / CuTe DSL KDA
        # decode backends, and the reference the KDA correctness tests assert against.
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            lower_bound=lower_bound,
            cache_ring=cache_ring,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g,
            replayssm_beta=replayssm_beta,
            ring_indices=ring_indices,
        )

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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _aiter_chunk_kda_applies(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
            return_intermediate_states=return_intermediate_states,
            track_state=kwargs.get("track_state"),
        ):
            if not TritonKDAKernel._logged_aiter_extend:
                TritonKDAKernel._logged_aiter_extend = True
                logger.info("KDA prefill extend: aiter chunk_kda")
            return _aiter_chunk_kda_extend(
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
            )
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
            chunk_indices=kwargs.get("chunk_indices"),
            chunk_offsets=kwargs.get("chunk_offsets"),
        )
