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
    """ATOM-compatible PTPC weight quantization.

    Quantize the real rows with AITER's canonical per-token/per-output-channel
    kernel, then append exact-zero output rows with scale one. ATOM follows this
    order in ``LinearBase.process_weights_after_loading``; using a handwritten
    float32 divide differed at FP8 rounding ties and made SGLang's online format
    subtly different from the validated ATOM recipe.
    """
    from aiter import QuantType, get_hip_quant

    orig_n = weight.shape[0]
    qw, scale = get_hip_quant(QuantType.per_Token)(
        weight.contiguous(), quant_dtype=fp8_dtype
    )
    if pad_n > orig_n:
        qw = torch.nn.functional.pad(qw, (0, 0, 0, pad_n - orig_n))
        scale = torch.cat(
            [scale, scale.new_ones((pad_n - orig_n, *scale.shape[1:]))], dim=0
        )
    return qw.contiguous(), scale.contiguous()


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
        # Match ATOM: PTPC's scale is max(abs(full logical output row)), not the
        # max of each K shard. Gather once at load, quantize once, then return
        # this rank's K shard while retaining the replicated full-row scale.
        #
        # K3 may run attention TP inside a larger global world under DP
        # attention. Select the group whose size matches the layer contract
        # instead of assuming the global TP group.
        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        group = (
            parallel.attn_tp_group
            if parallel.attn_tp_group.world_size == int(getattr(module, "tp_size", 1))
            else get_tp_group()
        )
        full_w = group.all_gather(w.contiguous(), dim=1)
        qw, scale = _quantize_weight_rows(full_w, fp8_dtype, pad_n=pad_n)
        tp_rank = int(getattr(module, "tp_rank", group.rank_in_group))
        qw = qw.narrow(1, tp_rank * k, k).contiguous()
        del full_w
    else:
        qw, scale = _quantize_weight_rows(w, fp8_dtype, pad_n=pad_n)

    from aiter.ops.shuffle import shuffle_weight

    module.weight = torch.nn.Parameter(
        shuffle_weight(qw, (16, 16)), requires_grad=False
    )
    module.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    # Generic ROCm MLA has dtype-only FP8 shortcuts for 128-group blockscale
    # weights. PTPC is per-output-channel and must not enter those paths: q_b
    # would receive group-quantized activations, and o_proj would be quantized
    # before K3's output gate. Keep the scheme explicit on the layer.
    module._k3_ptpc_per_token = True  # type: ignore[attr-defined]
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
            # Narrowing the padded output gives a view with the padded row
            # stride. A BF16 linear here returns a packed tensor, so consumers
            # are entitled to assume one: MLA splits this into q/kv/rope and
            # hands the pieces to kernels that index by row width, and they read
            # across the pad instead of failing. Repack rather than propagate a
            # tensor that only looks right.
            out = out[..., :orig_n].contiguous()
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


def _release_kda_bf16_inproj_views(attn) -> None:
    """Drop the BF16 views that kept the merged in-proj buffer alive.

    ATOM empties ``b_proj`` / ``f_a_proj`` after growing the fused in-proj.
    SGLang's merge instead leaves those modules as views of the BF16 cat, so
    replacing ``_qkvgbfa_layer.weight`` with shuffled FP8 would otherwise
    pin the original allocation for the rest of the process.
    """
    for name in ("fused_qkvg_proj", "f_a_proj", "b_proj"):
        mod = getattr(attn, name, None)
        weight = getattr(mod, "weight", None) if mod is not None else None
        if weight is None:
            continue
        data = weight.data if hasattr(weight, "data") else weight
        if data.numel() == 0:
            continue
        empty = data.new_empty((0, 0) if data.dim() == 2 else (0,))
        if isinstance(weight, torch.nn.Parameter):
            mod.weight = torch.nn.Parameter(empty, requires_grad=False)
        else:
            mod.weight = empty
    if getattr(attn, "_bfa_w", None) is not None:
        attn._bfa_w = None


def quantize_k3_dense_linears_in_layer(
    layer: torch.nn.Module,
    *,
    skip_o_proj: bool = False,
) -> int:
    """Online-quantize K3 dense linears to match ATOM's ptpc_fp8 recipe.

    ATOM ``--online_quant_config`` uses ``global_quant_config=ptpc_fp8`` and
    excludes embeddings, lm_head, KDA conv1d, routed MXFP4 experts, routed
    latent projections, and the vision tower. Shared experts, KDA in_proj /
    f_b / o_proj, and every MLA projection are quantized.
    """
    count = 0
    attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)

    def _try(mod) -> None:
        nonlocal count
        if mod is not None and quantize_linear_weight_ptpc(mod):
            count += 1

    if attn is not None:
        if hasattr(attn, "fused_qkv_a_proj_with_mqa"):
            for name in (
                "fused_qkv_a_proj_with_mqa",
                "q_b_proj",
                "kv_b_proj",
                "o_proj",
            ):
                if name == "o_proj" and skip_o_proj:
                    continue
                # Decode absorbs kv_b into w_kc/w_vc before this hook runs.
                # Quantizing the leftover BF16 copy would keep both layouts
                # resident and is unused on the hot path.
                if name == "kv_b_proj" and getattr(attn, "w_kc", None) is not None:
                    continue
                _try(getattr(attn, name, None))
            if getattr(attn, "use_output_gate", False):
                _try(getattr(attn, "g_proj", None))
        elif hasattr(attn, "fused_qkvg_proj"):
            # ATOM fuses [q,k,v,g|f_a|b] then online-quantizes that one matrix.
            fused_inproj = getattr(attn, "_qkvgbfa_layer", None)
            if fused_inproj is not None:
                if quantize_linear_weight_ptpc(fused_inproj):
                    count += 1
                    _release_kda_bf16_inproj_views(attn)
            else:
                _try(getattr(attn, "fused_qkvg_proj", None))
                _try(getattr(attn, "f_a_proj", None))
                _try(getattr(attn, "b_proj", None))
            _try(getattr(attn, "f_b_proj", None))
            if not skip_o_proj:
                _try(getattr(attn, "o_proj", None))

    if mlp is not None:
        if hasattr(mlp, "gate_up_proj") and hasattr(mlp, "down_proj"):
            if not hasattr(mlp, "experts"):
                _try(mlp.gate_up_proj)
                _try(mlp.down_proj)
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            _try(getattr(shared, "gate_up_proj", None))
            _try(getattr(shared, "down_proj", None))

    return count


def quantize_k3_model_dense_linears(model: torch.nn.Module) -> int:
    """Walk ``model.layers`` and quantize all eligible dense bf16 linears."""
    total = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer in layers:
        if not hasattr(layer, "self_attn"):
            continue
        # ATOM always online-quantizes o_proj. The fused all-reduce path can
        # still write a caller-owned BF16 buffer through apply_into.
        total += quantize_k3_dense_linears_in_layer(layer, skip_o_proj=False)
    return total
