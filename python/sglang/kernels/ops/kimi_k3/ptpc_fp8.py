"""Kimi-K3 ROCm per-token per-channel FP8 (ptpc_fp8) helpers.

Online weight quantization and linear apply for dense bf16 layers the K3
checkpoint ships unquantized. Uses aiter ``gemm_a8w8_bpreshuffle``; no external
runtime dependency beyond SGLang + aiter.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

_CK_N_TILE = 128
_CK_K_TILE = 32


def k3_ptpc_fp8_dtype() -> torch.dtype:
    from sglang.srt.layers.quantization.fp8_utils import is_fp8_fnuz

    return torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def _round_up(n: int, tile: int = _CK_N_TILE) -> int:
    return ((n + tile - 1) // tile) * tile


def _is_row_parallel_weight(module: torch.nn.Module, weight: torch.Tensor) -> bool:
    """Whether ``weight`` is a K-shard of one logical row-parallel matrix."""
    tp_size = int(getattr(module, "tp_size", 1))
    input_size = int(getattr(module, "input_size", weight.shape[1]))
    input_size_per_partition = getattr(module, "input_size_per_partition", None)
    return (
        tp_size > 1
        and input_size_per_partition is not None
        and int(input_size_per_partition) == weight.shape[1]
        and int(input_size_per_partition) * tp_size == input_size
    )


def _quantize_weight_rows(
    weight: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    pad_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PTPC weight quantization before backend-specific sharding/shuffling."""
    orig_n = weight.shape[0]
    fp8_max = torch.finfo(fp8_dtype).max
    wf = weight.float()
    if pad_n > orig_n:
        wf = torch.nn.functional.pad(wf, (0, 0, 0, pad_n - orig_n))
    amax = wf.abs().amax(dim=1, keepdim=True)
    # A zero row (including output padding) is represented exactly. A scale of
    # one avoids division by zero and matches ATOM's padded-scale convention.
    scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
    qw = (wf / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return qw, scale


def _linear_has_plain_bf16_weight(module: torch.nn.Module) -> bool:
    w = getattr(module, "weight", None)
    return (
        w is not None
        and type(w.data) is torch.Tensor
        and w.dim() == 2
        and w.dtype in (torch.bfloat16, torch.float16)
        and getattr(module, "weight_scale", None) is None
    )


def quantize_linear_weight_ptpc(
    module: torch.nn.Module,
    *,
    fp8_dtype: Optional[torch.dtype] = None,
    n_tile: int = _CK_N_TILE,
) -> bool:
    """Quantize ``module.weight`` to per-output-channel FP8 in place.

    Pads the output dimension (N) up to ``n_tile`` when needed for aiter's
    preshuffled B layout. The logical output width is stored on the module as
    ``_k3_ptpc_orig_out_features`` and trimmed after GEMM.
    """
    if not _linear_has_plain_bf16_weight(module):
        return False

    if fp8_dtype is None:
        fp8_dtype = k3_ptpc_fp8_dtype()

    w = module.weight.data
    orig_n, k = w.shape
    if k % _CK_K_TILE != 0:
        logger.warning(
            "K3 PTPC skipped %s: AITER bpreshuffle requires K %% %d == 0, got K=%d",
            type(module).__name__,
            _CK_K_TILE,
            k,
        )
        return False

    pad_n = _round_up(orig_n, n_tile)
    row_parallel = _is_row_parallel_weight(module, w)
    if row_parallel:
        # PTPC is per *logical* output row. Quantizing each K shard separately
        # gives each TP rank a different scale and diverges from ATOM/offline
        # PTPC. Gather once at load time, quantize the full row, then return this
        # rank's K shard while retaining the replicated per-row scale.
        from sglang.srt.distributed.parallel_state import get_tp_group

        full_w = get_tp_group().all_gather(w.contiguous(), dim=1)
        qw, scale = _quantize_weight_rows(full_w, fp8_dtype, pad_n=pad_n)
        tp_rank = int(getattr(module, "tp_rank", get_tp_group().rank_in_group))
        qw = qw.narrow(1, tp_rank * k, k).contiguous()
        del full_w
    else:
        qw, scale = _quantize_weight_rows(w, fp8_dtype, pad_n=pad_n)

    from aiter.ops.shuffle import shuffle_weight

    module.weight = torch.nn.Parameter(
        shuffle_weight(qw, (16, 16)), requires_grad=False
    )
    module.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    module._k3_ptpc_global_row_scale = row_parallel  # type: ignore[attr-defined]
    if pad_n != orig_n:
        module._k3_ptpc_orig_out_features = orig_n  # type: ignore[attr-defined]
    module.quant_method = K3PtpcFp8LinearMethod()
    return True


class K3PtpcFp8LinearMethod:
    """Per-token per-channel FP8 apply for weights quantized after load."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear

        out = apply_fp8_ptpc_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            bias=bias,
            use_per_token_if_dynamic=True,
        )
        orig_n = getattr(layer, "_k3_ptpc_orig_out_features", None)
        if orig_n is not None:
            out = out[..., :orig_n]
        return out

    def apply_into(
        self,
        layer: torch.nn.Module,
        x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        output: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output.copy_(self.apply(layer, x, bias))
        return output


def _gate_is_quantizable(gate: torch.nn.Module) -> bool:
    w = getattr(gate, "weight", None)
    return (
        w is not None
        and type(w.data) is torch.Tensor
        and w.dim() == 2
        and w.dtype in (torch.bfloat16, torch.float16)
        and getattr(gate, "weight_scale", None) is None
    )


def quantize_moe_gate_ptpc(gate: torch.nn.Module) -> bool:
    """Quantize the MoE router gate weight (``[n_experts, hidden]``)."""
    if not _gate_is_quantizable(gate):
        return False
    return quantize_linear_weight_ptpc(gate)


def quantize_k3_dense_linears_in_layer(
    layer: torch.nn.Module,
    *,
    skip_o_proj: bool = False,
) -> int:
    """Online-quantize every eligible dense bf16 linear in one decoder layer."""
    count = 0
    attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)

    def _try(mod: Optional[torch.nn.Module]) -> None:
        nonlocal count
        if mod is not None and quantize_linear_weight_ptpc(mod):
            count += 1

    if attn is not None:
        # MLA-style attention (fused qkv_a present).
        if hasattr(attn, "fused_qkv_a_proj_with_mqa"):
            for name in (
                "fused_qkv_a_proj_with_mqa",
                "q_b_proj",
                "o_proj",
            ):
                if name == "o_proj" and skip_o_proj:
                    continue
                _try(getattr(attn, name, None))
            if getattr(attn, "use_output_gate", False):
                _try(getattr(attn, "g_proj", None))
        # KDA-style attention (fused qkvg present).
        elif hasattr(attn, "fused_qkvg_proj"):
            # Keep the KDA input stack BF16. Its packed [q,k,v,g|f_a|b] GEMM is
            # faster than separate PTPC launches and the AITER one-launch decode
            # requires BF16 f_b. Quantizing the packed stand-in as well as its
            # constituent views also retained ~5.8 GiB/rank of duplicate weights.
            if not skip_o_proj:
                _try(getattr(attn, "o_proj", None))

    if mlp is not None:
        if hasattr(mlp, "gate_up_proj") and hasattr(mlp, "down_proj"):
            if not hasattr(mlp, "experts"):
                _try(mlp.gate_up_proj)
                _try(mlp.down_proj)
        if hasattr(mlp, "experts"):
            # K3's merged BF16 MoE front and packed latent/shared tail save two
            # GEMM launches and one TP collective. Quantizing the router/shared
            # views disables that path and changes routing logits, so leave the
            # complete fused-front contract in BF16.
            pass

    return count


def quantize_k3_model_dense_linears(model: torch.nn.Module) -> int:
    """Walk ``model.layers`` and quantize all eligible dense bf16 linears."""
    total = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer in layers:
        if not hasattr(layer, "self_attn"):
            continue
        skip_o = bool(getattr(layer, "all_reduce_fusion", False))
        total += quantize_k3_dense_linears_in_layer(layer, skip_o_proj=skip_o)
    return total
