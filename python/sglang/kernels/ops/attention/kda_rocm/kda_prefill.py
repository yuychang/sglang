# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill backend selection for ROCm.

Ported from vLLM ``vllm/models/kimi_k3/amd/ops/kda_prefill.py``. The Kimi-K3
KDA layer calls :func:`chunk_kda_prefill`, which either runs the fused HIP
kernels in ``kda_chunk`` or falls back to the vendored Triton chunk path.
"""

import logging

import torch

from sglang.kernels.ops.attention.kda_rocm.kda_chunk import (
    can_use_fused_kda_chunk,
    fused_kda_chunk,
    fused_kda_prologue,
)
from sglang.kernels.ops.attention.kda_rocm.third_party.fla.utils import FLA_CHUNK_SIZE
from sglang.kernels.ops.attention.kda_rocm.third_party.kda import (
    chunk_kda_with_fused_gate,
)

logger = logging.getLogger(__name__)
_logged: set[str] = set()


def _info_once(msg: str) -> None:
    if msg not in _logged:
        _logged.add(msg)
        logger.info(msg)


def gather_initial_states(
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Rows of ``state_cache`` per sequence, zero where there is no prefix."""
    states = state_cache[state_indices.long()]
    mask = has_initial_state.view(-1, *([1] * (states.dim() - 1)))
    return torch.where(mask, states, torch.zeros_like(states))


def chunk_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_fused_chunk: bool = False,
    out: torch.Tensor | None = None,
    checkpoint_state: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    checkpoint_state_indices: torch.Tensor | None = None,
    state_cache: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run chunk KDA from raw gate and beta projections.

    ``q``/``k``/``v`` are ``[1, T, H, 128]``; ``raw_g`` / ``raw_beta`` are the
    pre-activation gate and beta. With ``use_fused_chunk`` the two-kernel ROCm
    path runs when all of its preconditions hold, otherwise the Triton path.
    ``state_cache`` is the paged fp32 recurrent state, read and written in
    place by either backend (the Triton path gathers and scatters around the
    call); the returned final state is then ``None``.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # The fused prologue folds the q/k L2 norm and the gate activation in, so it
    # needs the raw projections and a bounded gate rather than the
    # pre-normalized tensors the Triton path takes.
    fused = (
        use_fused_chunk
        and use_qk_l2norm_in_kernel
        and cu_seqlens is not None
        and lower_bound is not None
        and g_bias is not None
        and can_use_fused_kda_chunk(k.shape[-1], v.shape[-1], k.dtype, FLA_CHUNK_SIZE)
    )

    if checkpoint_offsets is not None and not fused:
        raise NotImplementedError(
            "The KDA prefill checkpoint export needs the fused ROCm chunk backend"
        )
    if state_cache is not None:
        if initial_state is not None or output_final_state:
            raise ValueError("state_cache replaces initial_state/output_final_state")
        if state_indices is None or has_initial_state is None:
            raise ValueError("state_cache needs state_indices and has_initial_state")

    scatter_to: torch.Tensor | None = None
    if state_cache is not None and not fused:
        initial_state = gather_initial_states(
            state_cache, state_indices, has_initial_state
        )
        output_final_state = True
        scatter_to, state_cache = state_cache, None

    if fused:
        _info_once("Kimi-K3 KDA prefill: dispatching the fused ROCm chunk kernel.")
        ws = fused_kda_prologue(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=g_bias,
            scale=scale,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return fused_kda_chunk(
            qg=ws["qg"],
            w=ws["w"],
            u=ws["u"],
            kg_t=ws["kg_t"],
            aqk=ws["aqk"],
            decay=ws["decay"],
            out=out
            if out is not None
            else (
                v
                if v.is_contiguous()
                else torch.empty_like(v, memory_format=torch.contiguous_format)
            ),
            scale=scale,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_offsets=chunk_offsets,
            checkpoint_state=checkpoint_state,
            checkpoint_offsets=checkpoint_offsets,
            checkpoint_state_indices=checkpoint_state_indices,
            state_cache=state_cache,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
        )

    _info_once("Kimi-K3 KDA prefill: dispatching the Triton chunk kernels.")
    o, final_state = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    if out is not None and o.data_ptr() != out.data_ptr():
        out.copy_(o)
        o = out
    if scatter_to is not None:
        scatter_to[state_indices.long()] = final_state.to(scatter_to.dtype)
        return o, None
    return o, final_state
