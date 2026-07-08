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
"""ROCm-native multi-stream MXFP4 MoE overlap for Kimi-K2.5 / DeepSeek-style MoE.

This module implements the FlashInfer/TRT-LLM-inspired high-level MoE decode
schedule on ROCm/HIP for offline Quark/OCP MXFP4 checkpoints (e.g.
``amd/Kimi-K2.5-MXFP4``)::

    main HIP stream:                 secondary HIP stream:
        route tokens                     run MXFP4 shared expert MLP
        compute top-k metadata           record shared_expert_done event
        launch MXFP4 routed MoE
        wait for shared_expert_done
        launch fused finalize/add-shared

The routed experts and the shared expert are both executed with the AITER MXFP4
kernels that SGLang already wires up for the Quark ``W4A4 MXFP4`` schemes; this
module only adds the *scheduling* (secondary stream + event) and the *combine*
(fused add-shared / deferred finalize) on top of the existing kernels.

Two phases are implemented:

* **P0** -- overlap the MXFP4 shared expert with the MXFP4 routed experts and
  combine with a fused ``output = routed_final + shared_output`` kernel. The
  routed output already carries ``routed_scaling_factor`` (it is folded into the
  AITER top-k weights, see :mod:`sglang.srt.layers.moe.topk`), so the combine
  must *not* re-apply it.
* **P1** -- additionally defer the routed finalize: the AITER MoE returns
  pre-finalize per-expert partials (``no_combine=True``) and a fused kernel does
  the weighted top-k reduction + shared add in one launch.

The feature defaults OFF. It activates only when the existing opt-in ROCm flag
``SGLANG_ROCM_USE_MULTI_STREAM=1`` (or the dedicated
``SGLANG_ROCM_KIMI_MXFP4_MOE_MULTI_STREAM=1``) is set *and* every gating
condition in :func:`rocm_kimi_mxfp4_multistream_enabled` holds. Anything else
falls back cleanly to the single-stream MXFP4 baseline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_hip

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

OCP_MX_BLOCK_SIZE = 32

# One-time log guards (keyed by a short reason string so each distinct message
# is only emitted once per process). We do not use logger.info_once here because
# we want a single consolidated summary line, not per-call-site dedup.
_logged_keys: set[str] = set()


def _log_once(key: str, msg: str, *, level: int = logging.INFO) -> None:
    if key in _logged_keys:
        return
    _logged_keys.add(key)
    logger.log(level, msg)


# ---------------------------------------------------------------------------
# Hardware / capability helpers
# ---------------------------------------------------------------------------


def native_mxfp4_supported() -> bool:
    """Whether the current GPU has native OCP MXFP4 matrix support (CDNA4 /
    gfx95x, e.g. MI350/MI355). Non-native HIP devices (e.g. gfx942/MI300) still
    run MXFP4 through the AITER Triton emulation kernels, so this is only used
    for logging and for the simulation-fallback policy, not to gate loading."""
    if not _is_hip:
        return False
    try:
        from sglang.srt.utils.common import mxfp_supported

        return bool(mxfp_supported())
    except Exception:
        return False


def aiter_moe_supports_no_combine() -> bool:
    """Probe whether the installed ``aiter.fused_moe`` exposes the ``no_combine``
    kwarg required to return pre-finalize routed partials (P1)."""
    if not _use_aiter:
        return False
    try:
        from sglang.srt.layers.moe.moe_runner.aiter import (
            _aiter_fused_moe_supports_no_combine,
        )

        return bool(_aiter_fused_moe_supports_no_combine())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ROCm MoE stream manager
# ---------------------------------------------------------------------------


class RocmMoeStreamState:
    """Per-device secondary-stream + event state for the multi-stream MoE.

    One instance is created lazily per device (see
    :func:`get_rocm_moe_stream_state`). It owns a single non-blocking secondary
    HIP stream and a reusable, non-timing HIP event used to signal completion of
    the shared-expert work back to the main (routed) stream.

    Stream / event objects are created once (outside the decode hot path and
    outside CUDA/HIP graph capture) and reused for every forward. The main
    stream stays the routed stream; the secondary stream is used only for the
    shared expert.
    """

    __slots__ = ("device", "shared_stream", "shared_done_event", "_workspaces")

    def __init__(self, device: torch.device):
        self.device = device
        # torch.cuda.Stream / Event work for ROCm/HIP builds of PyTorch.
        self.shared_stream = torch.cuda.Stream(device=device)
        self.shared_done_event = torch.cuda.Event(blocking=False, enable_timing=False)
        # Optional reusable scratch keyed by (name, shape, dtype). Kept small:
        # the MXFP4 GEMM / activation-quant kernels allocate their own outputs
        # from the stream-aware caching allocator, so we only reuse the final
        # combine output buffer here.
        self._workspaces: Dict[tuple, torch.Tensor] = {}

    def get_workspace(
        self, name: str, shape: tuple, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (name, tuple(shape), dtype)
        buf = self._workspaces.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._workspaces[key] = buf
        return buf


# device index -> state
_stream_states: Dict[int, RocmMoeStreamState] = {}


def _resolve_device_index(device) -> int:
    if isinstance(device, int):
        return device
    idx = getattr(device, "index", None)
    if idx is not None:
        return idx
    return torch.cuda.current_device()


def get_rocm_moe_stream_state(device: torch.device) -> RocmMoeStreamState:
    """Return the (lazily-created) :class:`RocmMoeStreamState` for ``device``.

    Safe to call during model warmup / first use, but MUST NOT be called for the
    first time inside a CUDA/HIP graph capture region (stream/event creation is
    illegal there). Callers gate this via ``get_is_capture_mode()`` +
    warmup-time initialization.
    """
    idx = _resolve_device_index(device)
    state = _stream_states.get(idx)
    if state is None:
        state = RocmMoeStreamState(torch.device(f"cuda:{idx}"))
        _stream_states[idx] = state
        _log_once(
            "stream_state_init",
            f"[ROCm Kimi MXFP4 MoE] initialized secondary HIP stream + event "
            f"for device cuda:{idx}",
        )
    return state


def maybe_init_rocm_moe_stream_state(device: torch.device) -> None:
    """Eagerly create the stream state (call during warmup so no stream/event is
    created later in the captured hot path)."""
    if not _is_hip:
        return
    try:
        get_rocm_moe_stream_state(device)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "[ROCm Kimi MXFP4 MoE] failed to pre-initialize stream state: %s", e
        )


def rocm_moe_stream_state_exists(device: torch.device) -> bool:
    """Whether the per-device stream state is already initialized. Used to keep
    stream/event creation out of the CUDA/HIP graph capture region."""
    try:
        return _resolve_device_index(device) in _stream_states
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Feature gating
# ---------------------------------------------------------------------------


def _multistream_flag_enabled() -> bool:
    return (
        envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
        or envs.SGLANG_ROCM_KIMI_MXFP4_MOE_MULTI_STREAM.get()
    )


def _experts_are_mxfp4(experts: nn.Module) -> bool:
    scheme = getattr(experts, "scheme", None)
    if scheme is None:
        return False
    try:
        from sglang.srt.layers.quantization.quark.schemes import QuarkW4A4MXFp4MoE
    except Exception:
        return False
    return isinstance(scheme, QuarkW4A4MXFp4MoE)


def _shared_expert_is_mxfp4(shared_experts: nn.Module) -> bool:
    gate_up = getattr(shared_experts, "gate_up_proj", None)
    down = getattr(shared_experts, "down_proj", None)
    if gate_up is None or down is None:
        return False
    try:
        from sglang.srt.layers.quantization.quark.schemes import QuarkW4A4MXFP4
    except Exception:
        return False
    return isinstance(getattr(gate_up, "scheme", None), QuarkW4A4MXFP4) and isinstance(
        getattr(down, "scheme", None), QuarkW4A4MXFP4
    )


def rocm_kimi_mxfp4_multistream_enabled(moe_module: nn.Module) -> bool:
    """Return whether the ROCm MXFP4 multi-stream overlap should run for this
    MoE layer. The result is invariant after construction, so it is cached on
    the module as ``_rocm_kimi_mxfp4_ms``.

    All of the following must hold (otherwise we fall back to the existing
    single-stream / non-fused paths):

    * ROCm/HIP runtime with ``SGLANG_USE_AITER=1``
    * a multi-stream opt-in flag is set
    * the layer has a *separate* (non-fused) single shared expert
    * both the routed experts and the shared expert use the Quark MXFP4 schemes
    * the layer is not on a DeepEP/all-to-all path, EPLB, hash-routing or the
      SBO shared-expert-fusion path
    """
    cached = getattr(moe_module, "_rocm_kimi_mxfp4_ms", None)
    if cached is not None:
        return cached

    enabled = _compute_gate(moe_module)
    moe_module._rocm_kimi_mxfp4_ms = enabled
    if enabled:
        _log_feature_summary(moe_module)
    return enabled


def _compute_gate(moe_module: nn.Module) -> bool:
    if not (_is_hip and _use_aiter):
        return False
    if not _multistream_flag_enabled():
        return False
    # Separate (non-fused) shared expert required.
    if getattr(moe_module, "num_fused_shared_experts", 0) != 0:
        return False
    if not hasattr(moe_module, "shared_experts"):
        return False
    # n_shared_experts == 1 unless generalized support is added.
    if getattr(moe_module, "n_shared_experts", None) != 1:
        return False
    # Not on DeepEP / all-to-all MoE.
    if getattr(moe_module, "_enable_a2a_moe", False):
        return False
    # Not SBO shared-expert fusion.
    if getattr(moe_module, "_fuse_shared_experts_inside_sbo", False):
        return False
    # TP1-replicated shared experts are added after all-reduce; keep the
    # existing path so we don't sum the shared output once per rank.
    if getattr(moe_module, "_shared_expert_tp1", False):
        return False
    # Not hash routing.
    if getattr(moe_module, "is_hash", False):
        return False
    # EPLB rebalancing changes routing; keep the simple single-stream path.
    try:
        from sglang.srt.server_args import get_global_server_args

        if get_global_server_args().enable_eplb:
            return False
    except Exception:
        pass

    experts = getattr(moe_module, "experts", None)
    shared = getattr(moe_module, "shared_experts", None)
    if experts is None or shared is None:
        return False
    if not _experts_are_mxfp4(experts):
        return False
    if not _shared_expert_is_mxfp4(shared):
        return False
    return True


def _log_feature_summary(moe_module: nn.Module) -> None:
    native = native_mxfp4_supported()
    p1 = (
        envs.SGLANG_ENABLE_MOE_DEFERRED_FINALIZE.get()
        and aiter_moe_supports_no_combine()
    )
    _log_once(
        "feature_summary",
        "[ROCm Kimi MXFP4 MoE] offline Quark/OCP MXFP4 multi-stream overlap "
        "ENABLED\n"
        "  - routed experts MXFP4: yes\n"
        "  - shared_experts MXFP4: yes\n"
        "  - activation quantization: dynamic MXFP4 (group_size="
        f"{OCP_MX_BLOCK_SIZE}, scale_format=e8m0)\n"
        f"  - multi-stream: enabled (secondary HIP stream + event)\n"
        f"  - deferred finalize (P1): {'enabled' if p1 else 'disabled -> P0'}\n"
        f"  - phase selected: {'P1' if p1 else 'P0'}\n"
        f"  - native MXFP4 hardware (CDNA4/gfx95x): "
        f"{'yes' if native else 'no (AITER emulation)'}\n"
        f"  - fused combine op: "
        f"{'sgl_kernel HIP' if _has_hip_combine_ops() else 'triton/torch fallback'}",
    )
    if not native and not envs.SGLANG_ROCM_KIMI_MXFP4_ALLOW_SIMULATION_FALLBACK.get():
        # Not fatal: AITER still runs MXFP4 on gfx942 via Triton. Just inform.
        _log_once(
            "non_native_info",
            "[ROCm Kimi MXFP4 MoE] current GPU lacks native MXFP4 matrix "
            "instructions; running via the AITER MXFP4 Triton path. Set "
            "SGLANG_ROCM_KIMI_MXFP4_ALLOW_SIMULATION_FALLBACK=1 only to allow a "
            "slow dequant reference for debugging.",
        )


# ---------------------------------------------------------------------------
# Fused combine ops (HIP -> Triton -> torch reference)
# ---------------------------------------------------------------------------


def _has_hip_combine_ops() -> bool:
    if not _is_hip:
        return False
    try:
        import sgl_kernel  # noqa: F401

        return hasattr(torch.ops.sgl_kernel, "rocm_mxfp4_moe_add_shared")
    except Exception:
        return False


def rocm_mxfp4_moe_add_shared(
    routed_final: torch.Tensor,
    shared_output: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """P0 combine: ``out = routed_final + shared_output``.

    ``routed_final`` already carries ``routed_scaling_factor`` (folded into the
    AITER top-k weights), so this is a pure element-wise add. Both inputs are
    BF16/FP16 of shape ``(num_tokens, hidden)``.
    """
    if routed_final.shape != shared_output.shape:
        raise ValueError(
            "rocm_mxfp4_moe_add_shared: shape mismatch "
            f"routed_final={tuple(routed_final.shape)} "
            f"shared_output={tuple(shared_output.shape)}"
        )

    # Native HIP kernel.
    if out is None and _has_hip_combine_ops():
        try:
            out = torch.empty_like(routed_final)
            torch.ops.sgl_kernel.rocm_mxfp4_moe_add_shared(
                routed_final, shared_output, out
            )
            return out
        except Exception as e:  # pragma: no cover - defensive
            _log_once(
                "hip_add_shared_fallback",
                f"[ROCm Kimi MXFP4 MoE] HIP add_shared op failed ({e}); "
                "using torch fallback.",
                level=logging.WARNING,
            )

    # torch fallback (single fused element-wise kernel on GPU).
    if out is None:
        return shared_output.add(routed_final)
    torch.add(routed_final, shared_output, out=out)
    return out


def rocm_mxfp4_moe_finalize_fuse_shared(
    routed_partial: torch.Tensor,
    row_map: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    routed_scaling_factor: float,
    top_k: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """P1 combine (deferred finalize + shared add), computing::

        out[t, h] = shared_output[t, h]
            + routed_scaling_factor
              * sum_{k=0}^{top_k-1} topk_weights[t, k]
                                    * routed_partial[row_map[t, k], h]

    Parameters
    ----------
    routed_partial : (num_rows, hidden) BF16/FP16 pre-finalize per-expert output.
    row_map        : (num_tokens, top_k) int, maps (token, k) -> expanded row.
    topk_weights   : (num_tokens, top_k) float, the per-expert combine weights.
    shared_output  : (num_tokens, hidden) BF16/FP16 shared-expert output, or None.
    routed_scaling_factor : extra scale on the routed sum. **Pass 1.0 when the
        scale is already folded into ``topk_weights`` (the SGLang AITER path).**
    top_k          : number of selected experts per token.
    """
    num_tokens = row_map.shape[0]
    hidden = routed_partial.shape[-1]
    if row_map.shape[1] != top_k or topk_weights.shape != row_map.shape:
        raise ValueError(
            "rocm_mxfp4_moe_finalize_fuse_shared: row_map/topk_weights must be "
            f"(num_tokens, top_k)=({num_tokens}, {top_k}); got "
            f"row_map={tuple(row_map.shape)} topk_weights={tuple(topk_weights.shape)}"
        )
    if shared_output is not None and shared_output.shape != (num_tokens, hidden):
        raise ValueError(
            "rocm_mxfp4_moe_finalize_fuse_shared: shared_output must be "
            f"({num_tokens}, {hidden}); got {tuple(shared_output.shape)}"
        )

    out_dtype = (
        out.dtype
        if out is not None
        else (
            shared_output.dtype if shared_output is not None else routed_partial.dtype
        )
    )

    if out is None and _has_hip_combine_ops() and routed_partial.is_cuda:
        try:
            out = torch.empty(
                (num_tokens, hidden), dtype=out_dtype, device=routed_partial.device
            )
            torch.ops.sgl_kernel.rocm_mxfp4_moe_finalize_fuse_shared(
                routed_partial.contiguous(),
                row_map.to(torch.int64).contiguous(),
                topk_weights.to(torch.float32).contiguous(),
                shared_output,
                float(routed_scaling_factor),
                int(top_k),
                out,
            )
            return out
        except Exception as e:  # pragma: no cover - defensive
            _log_once(
                "hip_finalize_fallback",
                f"[ROCm Kimi MXFP4 MoE] HIP finalize op failed ({e}); using "
                "triton/torch fallback.",
                level=logging.WARNING,
            )

    # Triton fallback (GPU only).
    if routed_partial.is_cuda:
        triton_out = _finalize_fuse_shared_triton(
            routed_partial,
            row_map,
            topk_weights,
            shared_output,
            routed_scaling_factor,
            top_k,
            out,
        )
        if triton_out is not None:
            return triton_out

    # torch reference (works on CPU/CUDA/ROCm).
    return _finalize_fuse_shared_torch(
        routed_partial,
        row_map,
        topk_weights,
        shared_output,
        routed_scaling_factor,
        top_k,
        out,
    )


def _finalize_fuse_shared_torch(
    routed_partial: torch.Tensor,
    row_map: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    routed_scaling_factor: float,
    top_k: int,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    num_tokens = row_map.shape[0]
    hidden = routed_partial.shape[-1]
    out_dtype = (
        out.dtype
        if out is not None
        else (
            shared_output.dtype if shared_output is not None else routed_partial.dtype
        )
    )

    flat_rows = row_map.reshape(-1).to(torch.long)
    gathered = routed_partial.index_select(0, flat_rows).reshape(
        num_tokens, top_k, hidden
    )
    w = topk_weights.to(torch.float32).reshape(num_tokens, top_k, 1)
    acc = (gathered.to(torch.float32) * w).sum(dim=1)
    acc = acc * float(routed_scaling_factor)
    if shared_output is not None:
        acc = acc + shared_output.to(torch.float32)
    result = acc.to(out_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _finalize_fuse_shared_triton(
    routed_partial: torch.Tensor,
    row_map: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    routed_scaling_factor: float,
    top_k: int,
    out: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    try:
        import triton
    except Exception:
        return None

    num_tokens = row_map.shape[0]
    hidden = routed_partial.shape[-1]
    out_dtype = (
        out.dtype
        if out is not None
        else (
            shared_output.dtype if shared_output is not None else routed_partial.dtype
        )
    )
    if out is None:
        out = torch.empty(
            (num_tokens, hidden), dtype=out_dtype, device=routed_partial.device
        )

    routed_partial = routed_partial.contiguous()
    row_map = row_map.to(torch.int64).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    has_shared = shared_output is not None
    shared_c = shared_output.contiguous() if has_shared else routed_partial

    kernel = _get_finalize_triton_kernel()
    BLOCK = 256 if hidden >= 256 else triton.next_power_of_2(hidden)
    grid = (num_tokens, triton.cdiv(hidden, BLOCK))
    kernel[grid](
        routed_partial,
        row_map,
        topk_weights,
        shared_c,
        out,
        num_tokens,
        hidden,
        top_k,
        float(routed_scaling_factor),
        has_shared,
        routed_partial.stride(0),
        BLOCK=BLOCK,
    )
    return out


_finalize_triton_kernel = None


def _get_finalize_triton_kernel():
    global _finalize_triton_kernel
    if _finalize_triton_kernel is not None:
        return _finalize_triton_kernel

    import triton
    import triton.language as tl

    @triton.jit
    def _finalize_kernel(
        routed_partial_ptr,
        row_map_ptr,
        topk_weights_ptr,
        shared_ptr,
        out_ptr,
        num_tokens,
        hidden,
        top_k: tl.constexpr,
        routed_scaling_factor,
        has_shared: tl.constexpr,
        row_stride,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK + tl.arange(0, BLOCK)
        mask = cols < hidden
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(top_k):
            row = tl.load(row_map_ptr + token * top_k + k)
            w = tl.load(topk_weights_ptr + token * top_k + k)
            vals = tl.load(
                routed_partial_ptr + row * row_stride + cols, mask=mask, other=0.0
            ).to(tl.float32)
            acc += w * vals
        acc = acc * routed_scaling_factor
        if has_shared:
            s = tl.load(shared_ptr + token * hidden + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            acc += s
        tl.store(out_ptr + token * hidden + cols, acc, mask=mask)

    _finalize_triton_kernel = _finalize_kernel
    return _finalize_triton_kernel


def build_trivial_row_map(
    num_tokens: int, top_k: int, device: torch.device
) -> torch.Tensor:
    """Row map for AITER ``no_combine`` output of shape ``(num_tokens, top_k,
    hidden)`` flattened to ``(num_tokens * top_k, hidden)``: row(t, k) = t*top_k + k."""
    return (torch.arange(num_tokens * top_k, device=device, dtype=torch.int64)).reshape(
        num_tokens, top_k
    )


# ---------------------------------------------------------------------------
# P1: expose pre-finalize routed partials from the AITER MoE (no_combine)
# ---------------------------------------------------------------------------


def rocm_aiter_routed_no_combine(experts, hidden_states, topk_output):
    """Run the routed AITER MXFP4 MoE with ``no_combine=True`` and return the
    per-(token, expert) partials of shape ``(num_tokens, top_k, hidden)``.

    The partials are the raw per-expert down-projection outputs *before* the
    top-k weighted reduction (finalize). The reduction is deferred to the fused
    :func:`rocm_mxfp4_moe_finalize_fuse_shared` kernel.

    Returns ``None`` if the installed AITER build cannot produce no-combine
    output (caller falls back to P0).
    """
    scheme = getattr(experts, "scheme", None)
    runner = getattr(scheme, "runner", None) if scheme is not None else None
    if runner is None or not aiter_moe_supports_no_combine():
        return None

    cfg = runner.config
    orig_no_combine = cfg.no_combine
    cfg.no_combine = True
    try:
        dispatch_output = experts.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = scheme.apply_weights(experts, dispatch_output)
    finally:
        cfg.no_combine = orig_no_combine

    partial = combine_input.hidden_states
    if partial.dim() != 3:
        return None
    return partial


def rocm_kimi_mxfp4_p1_enabled() -> bool:
    """Whether the deferred-finalize (P1) phase may be attempted. The final
    decision is made per-layer via a one-time self-check against the P0 combine
    (see the model integration), so this only reflects the opt-in flags."""
    return (
        envs.SGLANG_ENABLE_MOE_DEFERRED_FINALIZE.get()
        and aiter_moe_supports_no_combine()
    )


def p1_self_check_matches(p1_routed: torch.Tensor, ref_routed: torch.Tensor) -> bool:
    """One-time correctness guard for P1: the deferred finalize of the AITER
    ``no_combine`` partials (with ``routed_scaling_factor`` folded into the
    top-k weights) must reproduce the standard combined routed output. If the
    installed AITER build already applies the top-k weight inside ``no_combine``
    (which would double-apply here), this returns False and the caller disables
    P1 and uses the always-correct P0 path.
    """
    if p1_routed.shape != ref_routed.shape:
        return False
    a = p1_routed.to(torch.float32)
    b = ref_routed.to(torch.float32)
    denom = b.abs().max().clamp_min(1e-3)
    max_abs = (a - b).abs().max()
    ok = bool((max_abs / denom).item() < 2e-2)
    if not ok:
        _log_once(
            "p1_self_check_fail",
            "[ROCm Kimi MXFP4 MoE] P1 deferred-finalize self-check failed "
            f"(max_abs_err={max_abs.item():.4f}, ref_scale={denom.item():.4f}); "
            "the AITER no_combine partials do not match the P0 combine. "
            "Disabling P1 and using the P0 add-shared path.",
            level=logging.WARNING,
        )
    return ok
