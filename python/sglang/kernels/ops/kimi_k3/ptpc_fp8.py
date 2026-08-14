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
    # A row-parallel weight is K-sharded and each rank's partial product is
    # summed by the output all-reduce, so a per-rank row scale reconstructs the
    # same sum: rank r contributes s_r * (Q_r @ x_r) == W_r @ x_r whatever s_r
    # is. Gathering the full row first only buys agreement with ATOM's offline
    # scale, and it costs a load-time collective plus a rank-indexing contract:
    # the gather runs on the global TP group while the narrow uses the module's
    # own tp_rank, which for MLA is the *attention* TP rank. Quantize the local
    # shard instead -- no collective, no cross-group assumption, and a tighter
    # scale because the amax is taken over this rank's K only.
    qw, scale = _quantize_weight_rows(w, fp8_dtype, pad_n=pad_n)

    from aiter.ops.shuffle import shuffle_weight

    module.weight = torch.nn.Parameter(
        shuffle_weight(qw, (16, 16)), requires_grad=False
    )
    module.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    module._k3_ptpc_row_sharded_scale = row_parallel  # type: ignore[attr-defined]
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


def ptpc_scope() -> frozenset[str]:
    """Which linear families PTPC is allowed to quantize.

    ``SGLANG_ROCM_K3_PTPC_SCOPE`` takes a comma-separated subset of
    ``mla_qkv_a``, ``mla_q_b``, ``mla_o_proj``, ``mla_g_proj``, ``kda_o_proj``,
    ``dense_mlp``, or ``all``. It exists because the families are not equally
    safe: they are consumed by different downstream code, and a family whose
    consumer reads the BF16 weight directly has to be excluded rather than
    quantized and hoped for.
    """
    import os

    raw = os.environ.get("SGLANG_ROCM_K3_PTPC_SCOPE", "").strip().lower()
    if raw == "":
        return frozenset(_SAFE_SCOPES)
    if raw == "all":
        return frozenset(_ALL_SCOPES)
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


_ALL_SCOPES = (
    "mla_qkv_a",
    "mla_q_b",
    "mla_o_proj",
    "mla_g_proj",
    "kda_o_proj",
    "dense_mlp",
)

# Quantizing any of the MLA linears makes Kimi-K3 emit fluent garbage on this
# tree: a greedy gsm8k question that answers "72" comes back as " Natal" padded
# with spaces. Bisected one family at a time -- mla_qkv_a+mla_q_b and
# mla_o_proj+mla_g_proj each reproduce it on their own, while
# kda_o_proj+dense_mlp is clean and scores normally -- so the fault is the MLA
# attention path rather than one layer, and the quant math itself is fine
# (test_kimi_k3_ptpc_fp8.py passes). Root cause is still open, so the default
# scope is the subset that is known good; pass scope=all to reproduce.
_SAFE_SCOPES = (
    "kda_o_proj",
    "dense_mlp",
)


def quantize_k3_dense_linears_in_layer(
    layer: torch.nn.Module,
    *,
    skip_o_proj: bool = False,
    scope: Optional[frozenset[str]] = None,
) -> int:
    """Online-quantize every eligible dense bf16 linear in one decoder layer."""
    count = 0
    attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    if scope is None:
        scope = ptpc_scope()

    def _try(mod: Optional[torch.nn.Module]) -> None:
        nonlocal count
        if mod is not None and quantize_linear_weight_ptpc(mod):
            count += 1

    if attn is not None:
        # MLA-style attention (fused qkv_a present).
        if hasattr(attn, "fused_qkv_a_proj_with_mqa"):
            for name, tag in (
                ("fused_qkv_a_proj_with_mqa", "mla_qkv_a"),
                ("q_b_proj", "mla_q_b"),
                ("o_proj", "mla_o_proj"),
            ):
                if name == "o_proj" and skip_o_proj:
                    continue
                if tag not in scope:
                    continue
                _try(getattr(attn, name, None))
            if getattr(attn, "use_output_gate", False) and "mla_g_proj" in scope:
                _try(getattr(attn, "g_proj", None))
        # KDA-style attention (fused qkvg present).
        elif hasattr(attn, "fused_qkvg_proj"):
            # Keep the KDA input stack BF16. Its packed [q,k,v,g|f_a|b] GEMM is
            # faster than separate PTPC launches and the AITER one-launch decode
            # requires BF16 f_b. Quantizing the packed stand-in as well as its
            # constituent views also retained ~5.8 GiB/rank of duplicate weights.
            if not skip_o_proj and "kda_o_proj" in scope:
                _try(getattr(attn, "o_proj", None))

    if mlp is not None:
        if hasattr(mlp, "gate_up_proj") and hasattr(mlp, "down_proj"):
            if not hasattr(mlp, "experts") and "dense_mlp" in scope:
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
    scope = ptpc_scope()
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer in layers:
        if not hasattr(layer, "self_attn"):
            continue
        skip_o = bool(getattr(layer, "all_reduce_fusion", False))
        total += quantize_k3_dense_linears_in_layer(
            layer, skip_o_proj=skip_o, scope=scope
        )
    return total
