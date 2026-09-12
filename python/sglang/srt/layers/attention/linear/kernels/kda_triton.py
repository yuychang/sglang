import os
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu, is_xpu

if not is_cpu():
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
        fused_recurrent_linear_replayssm_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.attention.fla.kda import chunk_kda


class TritonKDAKernel(LinearAttnKernelBase):
    """Triton-based kernel for KDA (Kimi Delta Attention) linear attention."""

    _prefill_window_size = int(os.getenv("SGLANG_KDA_PREFILL_WINDOW_SIZE", "0"))

    # XPU has no tvm_ffi CUDA JIT kernel for KDA packed decode; route XPU to the
    # non-packed Triton decode() path (fused_sigmoid_gating_delta_rule_update),
    # the same fallback CPU/NPU use. Batched decode is handled via query_start_loc.
    supports_packed_decode: bool = not is_cpu() and not is_npu() and not is_xpu()
    supports_fused_chain_verify: bool = not is_cpu() and not is_npu()

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
        def run_chunk(
            q_chunk,
            k_chunk,
            v_chunk,
            g_chunk,
            beta_chunk,
            state_index,
            chunk_start_loc,
        ):
            return chunk_kda(
                q=q_chunk,
                k=k_chunk,
                v=v_chunk,
                g=g_chunk,
                beta=beta_chunk,
                initial_state=ssm_states,
                initial_state_indices=state_index,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=chunk_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                output_intermediate_states=return_intermediate_states,
            )

        total_tokens = q.shape[1]
        window_size = self._prefill_window_size
        if window_size <= 0 or total_tokens <= window_size:
            return run_chunk(q, k, v, g, beta, cache_indices, query_start_loc)

        # A scheduler prefill chunk does not bound KDA work when a radix-cache
        # restore reconstructs a long linear-attention suffix. chunk_kda then
        # materializes O(T) gate/intra/recurrent workspaces (including fp32
        # tensors) for the full suffix. Process each sequence in chunk-aligned
        # windows instead. chunk_kda commits the final recurrent state in place,
        # so the next window starts from the exact state left by the preceding
        # one. Keeping non-final windows aligned to KDA's 64-token chunk size
        # also preserves the global ordering of intermediate radix snapshots.
        window_size = max(64, window_size // 64 * 64)
        seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
        if seq_lens_cpu is None:
            seq_lens_cpu = (query_start_loc[1:] - query_start_loc[:-1]).cpu()
        seq_lens = [int(length) for length in seq_lens_cpu]

        def token_slice(tensor, start, end):
            if tensor.shape[0] == 1 and tensor.ndim > 1:
                return tensor[:, start:end]
            return tensor[start:end]

        outputs = []
        intermediate_states = []
        token_start = 0
        for seq_idx, seq_len in enumerate(seq_lens):
            seq_end = token_start + seq_len
            for start in range(token_start, seq_end, window_size):
                end = min(start + window_size, seq_end)
                result = run_chunk(
                    q[:, start:end],
                    k[:, start:end],
                    v[:, start:end],
                    token_slice(g, start, end),
                    token_slice(beta, start, end),
                    cache_indices[seq_idx : seq_idx + 1],
                    # This call contains one dense sequence. Fixed-length mode
                    # avoids rebuilding varlen chunk indices (and synchronizing
                    # them to the host) for every KDA layer and window.
                    None,
                )
                if return_intermediate_states:
                    output, states = result
                    intermediate_states.append(states)
                else:
                    output = result
                outputs.append(output)
            token_start = seq_end

        output = torch.cat(outputs, dim=1)
        if return_intermediate_states:
            return output, torch.cat(intermediate_states, dim=1)
        return output
