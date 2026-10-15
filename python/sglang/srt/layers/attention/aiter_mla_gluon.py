"""Gluon MLA decode wrapper for low head-count MLA (e.g. Kimi K3 TP8: 12 heads/GPU).

Uses aiter ``mla_gluon`` when import succeeds and Triton Gluon exposes ``cga_layout``
(needs Triton >= 3.7). Falls back to the caller (zero-pad + ``mla_decode_fwd``) when
Gluon is unavailable.

Requires aiter ``main`` with ROCm/aiter #4480 (batch>1 ``bh16bn128``) and #4555
(decode CUDA graph KV splits). SGLang probes import + Triton API only; aiter version
is not pinned at build time.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _gluon_fn():
    """aiter mla gluon entry point return None if disabled"""
    if not envs.SGLANG_AITER_MLA_GLUON.get():
        logger.info("aiter mla gluon is disabled manually.")
        return None
    try:
        import triton
        import triton.experimental.gluon.language as gl
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon
    except ImportError as exc:
        logger.info("aiter mla gluon import error message: %s", exc)
        return None
    # mla_gluon builds its shared layouts with cga_layout, added in Triton 3.7;
    # older Triton fails at compile time with an opaque error instead.
    if "cga_layout" not in inspect.signature(gl.PaddedSharedLayout).parameters:
        logger.info(
            "aiter mla gluon is disabled due to triton %s has no Gluon cga_layout (need >= 3.7)",
            getattr(triton, "__version__", "unknown"),
        )
        return None
    return mla_gluon


def log_mla_gluon_capability(log: logging.Logger | None = None) -> None:
    """Report whether Gluon MLA decode is valid; the reason is logged by _gluon_fn."""
    ready = _gluon_fn() is not None
    (log or logger).info(
        "aiter mla gluon is %s",
        "enabled" if ready else "disabled; falling back to zero-pad mla_decode_fwd",
    )


def prefer_mla_gluon_decode(
    *,
    head_pad_mode: str,
    num_head: int,
    kv_cache_dtype: torch.dtype,
    q_dtype: torch.dtype | None = None,
) -> bool:
    return (
        head_pad_mode == "zero"
        and num_head == 12
        and kv_cache_dtype == fp8_dtype
        # Gluon decode takes BF16 Q; an FP8 Q stays on the ASM decode path.
        and (q_dtype is None or q_dtype == torch.bfloat16)
        and _gluon_fn() is not None
    )


# Native 16-head fp8-query persistent ASM covers this query length. A longer
# DSpark verify window is remapped onto a slower kernel, so those servers
# stay on Gluon. Same bound as AiterAttnBackend._MLA_ASM_PS_MAX_QLEN.
FP8_Q_ASM_MAX_QLEN = 4


def fp8_q_asm_mla_selected(
    *,
    decode_backend: str,
    verify_backend: str,
    dcp_world_size: int,
    verify_qlen: int,
    gfx95: bool,
) -> bool:
    """Whether h12 fp8 MLA should run the persistent fp8-query ASM kernel.

    ``SGLANG_AITER_MLA_PERSIST=0`` does not block this. The backend turns
    persistent metadata back on when the verify window fits.
    """
    return (
        decode_backend == "asm"
        and verify_backend == "asm"
        and dcp_world_size <= 1
        and gfx95
        and 1 <= verify_qlen <= FP8_Q_ASM_MAX_QLEN
    )


def fp8_q_asm_mla_enabled() -> bool:
    """Runtime form of :func:`fp8_q_asm_mla_selected` for the Q producer."""
    from sglang.srt.runtime_context import get_parallel, get_spec
    from sglang.srt.utils import is_gfx95_supported

    verify_qlen = int(get_spec().speculative_num_draft_tokens or 1)
    return fp8_q_asm_mla_selected(
        decode_backend=envs.SGLANG_AITER_MLA_DECODE_BACKEND.get().lower(),
        verify_backend=envs.SGLANG_AITER_MLA_VERIFY_BACKEND.get().lower(),
        dcp_world_size=int(get_parallel().attn_dcp_size),
        verify_qlen=verify_qlen,
        gfx95=bool(is_gfx95_supported()),
    )


def mla_gluon_decode(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    layer: RadixAttention,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float,
    min_kv_seq_len: int,
    kv_scale: float = 1.0,
    qlen: int = 1,
    use_2d_view: bool = False,
    return_lse: bool = False,
):
    """Run Gluon MLA decode for fused Q [num_tokens, H, 576] and MLA KV pool.
    Returns [num_tokens, H, v_head_dim] (or ``(out, lse)`` when ``return_lse``),
    or None when Gluon is unavailable.
    """
    mla_gluon = _gluon_fn()
    if mla_gluon is None:
        return None

    num_head = layer.tp_q_head_num
    kv_lora_rank = layer.v_head_dim
    qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank
    batch_size = q.shape[0] // qlen

    q_nope, q_pe = torch.split(q, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    if qlen > 1:
        # Splitting the leading dim is a stride change, so these stay views of
        # the non-contiguous torch.split outputs.
        q_nope = q_nope.view(batch_size, qlen, num_head, kv_lora_rank)
        q_pe = q_pe.view(batch_size, qlen, num_head, qk_rope_head_dim)
        o = q.new_empty((batch_size, qlen, num_head, kv_lora_rank))
    else:
        o = q.new_empty((batch_size, num_head, kv_lora_rank))

    extra_kwargs = {}
    if return_lse:
        extra_kwargs["return_lse"] = True

    result = mla_gluon(
        q_nope,
        q_pe,
        k_buffer.view(-1, layer.qk_head_dim),
        o,
        kv_indices,
        kv_indptr,
        sm_scale,
        k_pe=None,
        kv_pe_offset=kv_lora_rank,
        use_2d_view=use_2d_view,
        kv_scale=kv_scale,
        min_kv_seq_len=min_kv_seq_len,
        **extra_kwargs,
    )
    # Hand back the caller's flat [num_tokens, H, v] layout either way.
    out = o.flatten(0, 1) if qlen > 1 else o
    if not return_lse:
        return out
    # mla_gluon writes the output into `o` and returns it alongside the lse.
    _, lse = result
    return out, lse
