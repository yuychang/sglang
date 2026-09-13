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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MlaGluonCapability:
    """Runtime probe of aiter/Triton Gluon prerequisites for h12 + FP8 decode."""

    enabled_by_env: bool
    import_ok: bool
    triton_version: str
    triton_cga_layout_ok: bool
    ready: bool
    summary: str

    def missing_for_ready(self) -> list[str]:
        missing = []
        if not self.enabled_by_env:
            missing.append("SGLANG_AITER_MLA_GLUON=0")
        if not self.import_ok:
            missing.append("aiter.ops.triton.gluon.mla_gluon import")
        if not self.triton_cga_layout_ok:
            missing.append(
                f"Triton Gluon cga_layout (have {self.triton_version or 'unknown'}, need >= 3.7)"
            )
        return missing


@functools.lru_cache(maxsize=1)
def _gluon_fn():
    """Return the aiter mla_gluon entry point when the runtime requirements hold."""
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


def _triton_version() -> str:
    try:
        import triton

        return getattr(triton, "__version__", "unknown")
    except Exception:
        return "missing"


def _triton_cga_layout_ok() -> bool:
    try:
        import triton.experimental.gluon.language as gl

        return "cga_layout" in inspect.signature(gl.PaddedSharedLayout).parameters
    except Exception:
        return False


def _in_cuda_graph_capture() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def probe_mla_gluon_capability(*, force_refresh: bool = False) -> MlaGluonCapability:
    if force_refresh:
        _gluon_fn.cache_clear()

    enabled = envs.SGLANG_AITER_MLA_GLUON.get()
    triton_ver = _triton_version()
    cga_ok = _triton_cga_layout_ok()
    gluon = _gluon_fn()
    import_ok = gluon is not None or (enabled and cga_ok)
    ready = gluon is not None

    if ready:
        summary = f"Gluon MLA h12+fp8 ready (Triton={triton_ver})"
    else:
        cap = MlaGluonCapability(
            enabled_by_env=enabled,
            import_ok=import_ok,
            triton_version=triton_ver,
            triton_cga_layout_ok=cga_ok,
            ready=False,
            summary="",
        )
        summary = (
            "Gluon MLA h12+fp8 disabled; fallback to zero-pad mla_decode_fwd "
            f"({'; '.join(cap.missing_for_ready())})"
        )

    return MlaGluonCapability(
        enabled_by_env=enabled,
        import_ok=import_ok,
        triton_version=triton_ver,
        triton_cga_layout_ok=cga_ok,
        ready=ready,
        summary=summary,
    )


def prefer_mla_gluon_decode(
    *,
    head_pad_mode: str,
    num_head: int,
    kv_cache_dtype: torch.dtype,
    q_dtype: Optional[torch.dtype] = None,
) -> bool:
    if q_dtype is not None and q_dtype != torch.bfloat16:
        return False
    return (
        head_pad_mode == "zero"
        and num_head == 12
        and kv_cache_dtype == fp8_dtype
        and _gluon_fn() is not None
    )


def log_mla_gluon_capability(
    log: logging.Logger | None = None,
) -> MlaGluonCapability:
    cap = probe_mla_gluon_capability()
    (log or logger).info(cap.summary)
    if not cap.ready:
        for item in cap.missing_for_ready():
            (log or logger).info("  missing: %s", item)
    return cap


def reset_mla_gluon_state_for_test() -> None:
    _gluon_fn.cache_clear()


def mla_gluon_decode(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    layer: RadixAttention,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    seq_lens: Optional[torch.Tensor] = None,
    sm_scale: float,
    kv_scale: float = 1.0,
    min_kv_seq_len: Optional[int] = None,
    qlen: int = 1,
    use_2d_view: bool = False,
    return_lse: bool = False,
):
    """Run Gluon MLA decode for fused Q [num_tokens, H, 576] and MLA KV pool.

    Returns output [tokens, H, v_head_dim], or (output, LSE) for DCP.
    Non-DCP failures return None to fall back; DCP failures propagate because
    the fallback kernel cannot return the LSE needed for the cross-rank merge.

    ``min_kv_seq_len`` must be supplied by the caller during CUDA graph capture
    (no GPU->CPU sync from ``seq_lens``). For eager decode, omit it to derive
    from ``seq_lens`` when safe.
    """
    mla_gluon = _gluon_fn()
    if mla_gluon is None:
        if return_lse:
            raise RuntimeError("AITER Gluon MLA is required for DCP")
        return None

    batch_size = q.shape[0] // qlen
    num_head = q.shape[1]
    kv_lora_rank = layer.v_head_dim
    qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank

    q_nope, q_pe = torch.split(q, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    if qlen > 1:
        # Splitting the leading dim is a stride change, so these stay views of
        # the non-contiguous torch.split outputs.
        q_nope = q_nope.view(batch_size, qlen, num_head, kv_lora_rank)
        q_pe = q_pe.view(batch_size, qlen, num_head, qk_rope_head_dim)
        o = q.new_empty((batch_size, qlen, num_head, kv_lora_rank))
    else:
        o = q.new_empty((batch_size, num_head, kv_lora_rank))

    if min_kv_seq_len is None:
        if _in_cuda_graph_capture():
            logger.warning(
                "mla_gluon_decode: min_kv_seq_len missing during CUDA graph capture"
            )
            min_kv_seq_len = 1
        elif seq_lens is not None and seq_lens.numel():
            min_kv_seq_len = int(seq_lens.max().item())
        else:
            min_kv_seq_len = 1

    try:
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
            **({"return_lse": True} if return_lse else {}),
        )
        out = o.flatten(0, 1) if qlen > 1 else o
        if return_lse:
            _, lse = result
            return out, lse
        return out
    except Exception as exc:
        if return_lse:
            raise  # DCP cannot fall back to a kernel without LSE.
        logger.warning(
            "mla_gluon decode failed (num_head=%s, kv_dtype=%s, batch=%s): %s; "
            "falling back to zero-pad mla_decode_fwd",
            layer.tp_q_head_num,
            k_buffer.dtype,
            batch_size,
            exc,
        )
        return None
