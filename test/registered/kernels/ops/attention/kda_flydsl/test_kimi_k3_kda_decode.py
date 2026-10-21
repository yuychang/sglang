# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the fused FlyDSL Kimi-K3 KDA decode path."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from sglang.kernels.ops.attention.kda_flydsl.kimi_k3_kda_decode import (
    _fb_build_options,
    flydsl_kimi_k3_kda_decode_with_f_b,
    is_flydsl_kimi_k3_kda_decode_supported,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950_flydsl_available() -> bool:
    if importlib.util.find_spec("flydsl") is None:
        return False
    return is_flydsl_kimi_k3_kda_decode_supported()


pytestmark = pytest.mark.skipif(
    not _gfx950_flydsl_available(),
    reason="gfx950 FlyDSL required",
)


def _aiter_kda_state_dtypes() -> tuple[torch.dtype, ...]:
    """State dtypes the ROCm/aiter#6026 decode accepts; empty without it."""
    try:
        from aiter.ops.flydsl.linear_attention_kernels import (
            flydsl_kda_decode_with_f_b,
        )
    except ImportError:
        return ()
    try:
        return tuple(flydsl_kda_decode_with_f_b.state_dtypes)
    except AttributeError:
        return (torch.float32,)


_requires_aiter_kda = pytest.mark.skipif(
    not _aiter_kda_state_dtypes(), reason="aiter without ROCm/aiter#6026"
)
_requires_aiter_bf16_state = pytest.mark.skipif(
    torch.bfloat16 not in _aiter_kda_state_dtypes(),
    reason="aiter without BF16 KDA decode state",
)

_DEVICE = torch.device("cuda")
_HEADS = 12
_DIM = 128
_CHANNELS = 3 * _HEADS * _DIM
_CONV_WIDTH = 4
_LOWER_BOUND = -5.0
_NORM_EPS = 1e-5


@dataclass
class Inputs:
    x: torch.Tensor
    conv_weight: torch.Tensor
    conv_state: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    output_gate: torch.Tensor
    norm_weight: torch.Tensor


def _make_inputs(batch: int, seed: int = 20260728) -> Inputs:
    generator = torch.Generator(device=_DEVICE).manual_seed(seed + batch)
    slots = batch + 2

    x_storage = torch.randn(
        (batch, _CHANNELS + 17),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    x = x_storage[:, :_CHANNELS]
    conv_weight = 0.1 * torch.randn(
        (_CHANNELS, _CONV_WIDTH),
        dtype=torch.float32,
        device=_DEVICE,
        generator=generator,
    )

    # Kimi's hybrid cache can pad the slot stride. Exercise that layout
    # explicitly while keeping each slot's inner dimensions contiguous.
    conv_storage = torch.randn(
        (slots, _CHANNELS * (_CONV_WIDTH - 1) + 19),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    conv_state = conv_storage[:, : _CHANNELS * (_CONV_WIDTH - 1)].view(
        slots, _CHANNELS, _CONV_WIDTH - 1
    )
    state_storage = 0.01 * torch.randn(
        (slots, _HEADS * _DIM * _DIM + 23),
        dtype=torch.float32,
        device=_DEVICE,
        generator=generator,
    )
    state = state_storage[:, : _HEADS * _DIM * _DIM].view(slots, _HEADS, _DIM, _DIM)

    raw_beta_storage = torch.randn(
        (1, batch, _HEADS + 1),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    output_gate_storage = torch.randn(
        (batch, _HEADS * _DIM + 7),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    return Inputs(
        x=x,
        conv_weight=conv_weight,
        conv_state=conv_state,
        raw_g=torch.randn(
            (1, batch, _HEADS, _DIM),
            dtype=torch.bfloat16,
            device=_DEVICE,
            generator=generator,
        ),
        raw_beta=raw_beta_storage[:, :, :_HEADS],
        A_log=0.5
        * torch.randn(
            (_HEADS,),
            dtype=torch.float32,
            device=_DEVICE,
            generator=generator,
        ),
        dt_bias=0.1
        * torch.randn(
            (_HEADS * _DIM,),
            dtype=torch.float32,
            device=_DEVICE,
            generator=generator,
        ),
        state=state,
        state_indices=torch.arange(
            1,
            batch + 1,
            dtype=torch.int32,
            device=_DEVICE,
        ),
        output_gate=output_gate_storage[:, : _HEADS * _DIM].view(batch, _HEADS, _DIM),
        norm_weight=torch.randn(
            (_DIM,),
            dtype=torch.bfloat16,
            device=_DEVICE,
            generator=generator,
        ),
    )


def _copy_inputs(inputs: Inputs) -> Inputs:
    def clone_preserving_strides(tensor: torch.Tensor) -> torch.Tensor:
        clone = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        clone.copy_(tensor)
        return clone

    return Inputs(
        **{
            name: clone_preserving_strides(getattr(inputs, name))
            for name in Inputs.__dataclass_fields__
        }
    )


def _reference(inputs: Inputs) -> torch.Tensor:
    if inputs.state.dtype != torch.float32:
        # The kernel's rounding boundary for a narrow state: widen on load,
        # update in FP32, round to nearest-even on store.
        stored = inputs.state
        inputs.state = stored.float()
        output = _reference(inputs)
        stored.copy_(inputs.state)
        inputs.state = stored
        return output

    batch = inputs.x.shape[0]
    output = torch.zeros(
        (1, batch, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device=_DEVICE,
    )
    dt_bias = inputs.dt_bias.view(_HEADS, _DIM)

    for batch_idx in range(batch):
        state_idx = int(inputs.state_indices[batch_idx])
        if state_idx <= 0:
            continue

        history = inputs.conv_state[state_idx]
        conv_values = torch.cat(
            (
                history.float(),
                inputs.x[batch_idx, :, None].float(),
            ),
            dim=-1,
        )
        packed_qkv = F.silu((conv_values * inputs.conv_weight).sum(dim=-1)).to(
            torch.bfloat16
        )
        history[:, 0].copy_(history[:, 1])
        history[:, 1].copy_(history[:, 2])
        history[:, 2].copy_(inputs.x[batch_idx])

        q, k, v = packed_qkv.view(
            3,
            _HEADS,
            _DIM,
        ).unbind(0)
        for head_idx in range(_HEADS):
            q_head = q[head_idx].float()
            k_head = k[head_idx].float()
            q_head = q_head * torch.rsqrt(q_head.square().sum() + 1e-6)
            q_head = q_head * (_DIM**-0.5)
            k_head = k_head * torch.rsqrt(k_head.square().sum() + 1e-6)

            a = inputs.A_log[head_idx].exp()
            decay = (
                _LOWER_BOUND
                * torch.sigmoid(
                    (
                        inputs.raw_g[
                            0,
                            batch_idx,
                            head_idx,
                        ].float()
                        + dt_bias[head_idx]
                    )
                    * a
                )
            ).exp()
            decayed_state = inputs.state[state_idx, head_idx] * decay[None, :]
            state_dot_k = decayed_state @ k_head
            state_dot_q = decayed_state @ q_head
            k_dot_q = torch.dot(k_head, q_head)
            beta = torch.sigmoid(
                inputs.raw_beta[
                    0,
                    batch_idx,
                    head_idx,
                ].float()
            )
            v_new = (v[head_idx].float() - state_dot_k) * beta
            inputs.state[state_idx, head_idx].copy_(
                decayed_state + v_new[:, None] * k_head[None, :]
            )

            # The model materializes recurrent output in BF16 before the
            # normalization/gating operation.
            recurrent = (state_dot_q + v_new * k_dot_q).to(torch.bfloat16)
            recurrent_f32 = recurrent.float()
            inv_rms = torch.rsqrt(recurrent_f32.square().mean() + _NORM_EPS)
            output[0, batch_idx, head_idx] = (
                recurrent_f32
                * inv_rms
                * inputs.norm_weight.float()
                * torch.sigmoid(
                    inputs.output_gate[
                        batch_idx,
                        head_idx,
                    ].float()
                )
            ).to(torch.bfloat16)
    return output


def _relative_rmse(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> float:
    delta = actual.float() - reference.float()
    return float(
        delta.square().mean().sqrt() / (reference.float().square().mean().sqrt() + 1e-8)
    )


def _make_fb_inputs(
    batch: int,
    seed: int = 20260728,
) -> tuple[torch.Tensor, torch.Tensor, Inputs]:
    generator = torch.Generator(device=_DEVICE).manual_seed(seed + 10_000 + batch)
    f_a_storage = torch.randn(
        (batch, _DIM + 5),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    f_a = f_a_storage[:, :_DIM]
    f_b_weight = (
        0.05
        * torch.randn(
            (_HEADS, _DIM, _DIM),
            dtype=torch.bfloat16,
            device=_DEVICE,
            generator=generator,
        )
    ).to(torch.bfloat16)
    inputs = _make_inputs(batch, seed)
    projected = F.linear(
        f_a.float(),
        f_b_weight.view(_HEADS * _DIM, _DIM).float(),
    ).to(torch.bfloat16)
    inputs.raw_g.copy_(projected.view(1, batch, _HEADS, _DIM))
    return f_a, f_b_weight, inputs


def _run_with_f_b(
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    inputs: Inputs,
) -> torch.Tensor:
    return flydsl_kimi_k3_kda_decode_with_f_b(
        f_a=f_a,
        f_b_weight=f_b_weight,
        x=inputs.x,
        conv_weight=inputs.conv_weight,
        conv_bias=None,
        conv_state=inputs.conv_state,
        raw_beta=inputs.raw_beta,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=_LOWER_BOUND,
        state=inputs.state,
        state_indices=inputs.state_indices,
        output_gate=inputs.output_gate,
        norm_weight=inputs.norm_weight,
        norm_eps=_NORM_EPS,
    )


def test_public_api_and_support_predicate() -> None:
    import sglang.kernels.ops.attention.kda_flydsl as flydsl_ops

    assert (
        flydsl_ops.is_flydsl_kimi_k3_kda_decode_supported
        is is_flydsl_kimi_k3_kda_decode_supported
    )
    assert is_flydsl_kimi_k3_kda_decode_supported(0)
    assert not is_flydsl_kimi_k3_kda_decode_supported("cpu")


def test_f_b_public_api() -> None:
    import sglang.kernels.ops.attention.kda_flydsl as flydsl_ops

    assert (
        flydsl_ops.flydsl_kimi_k3_kda_decode_with_f_b
        is flydsl_kimi_k3_kda_decode_with_f_b
    )


def test_f_b_batch_two_dispatch_is_guarded() -> None:
    assert _fb_build_options(1) == {}
    assert _fb_build_options(4) == {}
    options = _fb_build_options(2)
    assert options == {
        "waves_per_eu": 3,
        "cooperative_f_a": True,
        "parallel_front": True,
        "fused_norm_reduce": True,
        "projection_fdot2": True,
    }


@pytest.mark.parametrize("batch", [1, 8, 16])
def test_kimi_k3_kda_decode_with_f_b_matches_reference(batch: int) -> None:
    f_a, f_b_weight, seed = _make_fb_inputs(batch)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _copy_inputs(seed)

    reference = _reference(reference_inputs)
    actual = _run_with_f_b(f_a, f_b_weight, actual_inputs)
    torch.cuda.synchronize()

    assert not torch.isnan(actual).any()
    assert _relative_rmse(reference, actual) < 1e-3
    assert _relative_rmse(reference_inputs.state, actual_inputs.state) < 1e-3
    assert torch.equal(reference_inputs.conv_state, actual_inputs.conv_state)


def test_kimi_k3_kda_decode_with_f_b_recurrent_sequence() -> None:
    f_a, f_b_weight, seed = _make_fb_inputs(batch=2)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _copy_inputs(seed)
    for _ in range(32):
        reference = _reference(reference_inputs)
        actual = _run_with_f_b(f_a, f_b_weight, actual_inputs)
    torch.cuda.synchronize()
    assert _relative_rmse(reference, actual) < 1e-3
    assert _relative_rmse(reference_inputs.state, actual_inputs.state) < 1e-3
    assert torch.equal(reference_inputs.conv_state, actual_inputs.conv_state)


def test_kimi_k3_kda_decode_with_f_b_graph_replay() -> None:
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=2)
    out = torch.empty((1, 2, _HEADS, _DIM), dtype=torch.bfloat16, device=_DEVICE)
    _run_with_f_b(f_a, f_b_weight, inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_kimi_k3_kda_decode_with_f_b(
            f_a=f_a,
            f_b_weight=f_b_weight,
            x=inputs.x,
            conv_weight=inputs.conv_weight,
            conv_bias=None,
            conv_state=inputs.conv_state,
            raw_beta=inputs.raw_beta,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            lower_bound=_LOWER_BOUND,
            state=inputs.state,
            state_indices=inputs.state_indices,
            output_gate=inputs.output_gate,
            norm_weight=inputs.norm_weight,
            norm_eps=_NORM_EPS,
            out=out,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.isnan(out).any()


def test_f_b_non_positive_slots_do_not_modify_caches() -> None:
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=2)
    inputs.state_indices.copy_(torch.tensor([0, -1], dtype=torch.int32, device=_DEVICE))
    conv_before = inputs.conv_state.clone()
    state_before = inputs.state.clone()

    actual = _run_with_f_b(f_a, f_b_weight, inputs)
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(inputs.conv_state, conv_before)
    assert torch.equal(inputs.state, state_before)


def test_f_b_api_rejects_invalid_projection_inputs() -> None:
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=1)

    with pytest.raises(ValueError, match="`f_a` must have rank 2"):
        _run_with_f_b(f_a.unsqueeze(0), f_b_weight, inputs)
    with pytest.raises(ValueError, match="`f_a` must have dtype"):
        _run_with_f_b(f_a.float(), f_b_weight, inputs)
    with pytest.raises(ValueError, match="`f_b_weight` must have shape"):
        _run_with_f_b(f_a, f_b_weight[:, :, :-1], inputs)
    with pytest.raises(ValueError, match="`f_b_weight` must have inner strides"):
        _run_with_f_b(f_a, f_b_weight.transpose(1, 2), inputs)


def test_fused_kda_backend_is_opt_in():
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip
    from sglang.srt.environ import envs

    with envs.SGLANG_OPT_HIP_K3_KDA_FUSED_BACKEND.override(""):
        assert not kda_fused_decode_aiter_hip.enabled()
    with envs.SGLANG_OPT_HIP_K3_KDA_FUSED_BACKEND.override("aiter"):
        assert kda_fused_decode_aiter_hip.enabled()


@_requires_aiter_kda
def test_aiter_source_binds_pr6026_kda_decode():
    """SGLANG_OPT_HIP_K3_FLYDSL_SOURCE=aiter must call ROCm/aiter#6026."""
    import aiter.ops.flydsl.linear_attention_kernels as aiter_kda

    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip
    from sglang.srt.environ import envs

    with envs.SGLANG_OPT_HIP_K3_FLYDSL_SOURCE.override("aiter"):
        run, supported = kda_fused_decode_aiter_hip._ops()
    assert run is aiter_kda.flydsl_kda_decode_with_f_b
    assert supported is aiter_kda.is_flydsl_kda_decode_supported


def test_sglang_source_binds_vendored_kda_decode():
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip
    from sglang.srt.environ import envs

    with envs.SGLANG_OPT_HIP_K3_FLYDSL_SOURCE.override("sglang"):
        run, supported = kda_fused_decode_aiter_hip._ops()
    assert run is flydsl_kimi_k3_kda_decode_with_f_b
    assert supported is is_flydsl_kimi_k3_kda_decode_supported


@_requires_aiter_kda
def test_aiter_fused_kda_decode_matches_reference():
    """The adapter must call ROCm/aiter#6026 and match the unfused reference."""
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip
    from sglang.srt.environ import envs

    f_a, f_b_weight, seed = _make_fb_inputs(batch=2)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _copy_inputs(seed)
    with (
        envs.SGLANG_OPT_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
        envs.SGLANG_OPT_HIP_K3_FLYDSL_SOURCE.override("aiter"),
    ):
        assert kda_fused_decode_aiter_hip.available(f_a.device)
        assert kda_fused_decode_aiter_hip.covered(
            f_a,
            f_b_weight,
            actual_inputs.x,
            actual_inputs.raw_beta,
            actual_inputs.conv_state,
            actual_inputs.state,
            actual_inputs.state_indices,
            actual_inputs.output_gate,
            actual_inputs.norm_weight,
        )
        out = kda_fused_decode_aiter_hip.run(
            f_a=f_a,
            f_b_weight=f_b_weight,
            mixed_qkv=actual_inputs.x,
            conv_weight=actual_inputs.conv_weight,
            conv_state=actual_inputs.conv_state,
            raw_beta=actual_inputs.raw_beta,
            A_log=actual_inputs.A_log,
            dt_bias=actual_inputs.dt_bias,
            lower_bound=_LOWER_BOUND,
            state=actual_inputs.state,
            state_indices=actual_inputs.state_indices,
            output_gate=actual_inputs.output_gate,
            norm_weight=actual_inputs.norm_weight,
            norm_eps=_NORM_EPS,
        )
    torch.cuda.synchronize()
    reference = _reference(reference_inputs)
    assert out.shape == reference.shape
    assert out.dtype == torch.bfloat16
    assert _relative_rmse(reference, out) < 1e-3
    assert _relative_rmse(reference_inputs.state, actual_inputs.state) < 1e-3
    assert torch.equal(reference_inputs.conv_state, actual_inputs.conv_state)


def _bf16_dense_state(inputs: Inputs) -> None:
    # SGLang's --mamba-ssm-dtype bfloat16 pool: [cache, 12, 128, 128], dense.
    inputs.state = inputs.state.to(torch.bfloat16).contiguous()
    assert inputs.state.stride() == (_HEADS * _DIM * _DIM, _DIM * _DIM, _DIM, 1)


def _adapter_covers(adapter, f_a, f_b_weight, inputs: Inputs) -> bool:
    return adapter.covered(
        f_a,
        f_b_weight,
        inputs.x,
        inputs.raw_beta,
        inputs.conv_state,
        inputs.state,
        inputs.state_indices,
        inputs.output_gate,
        inputs.norm_weight,
    )


@_requires_aiter_bf16_state
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_aiter_fused_kda_decode_bf16_state_matches_reference(batch: int) -> None:
    """A BF16 recurrent state loads as BF16, updates in FP32, stores as BF16."""
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip
    from sglang.srt.environ import envs

    f_a, f_b_weight, seed = _make_fb_inputs(batch)
    _bf16_dense_state(seed)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _copy_inputs(seed)
    with (
        envs.SGLANG_OPT_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
        envs.SGLANG_OPT_HIP_K3_FLYDSL_SOURCE.override("aiter"),
    ):
        assert _adapter_covers(
            kda_fused_decode_aiter_hip, f_a, f_b_weight, actual_inputs
        )
        out = kda_fused_decode_aiter_hip.run(
            f_a=f_a,
            f_b_weight=f_b_weight,
            mixed_qkv=actual_inputs.x,
            conv_weight=actual_inputs.conv_weight,
            conv_state=actual_inputs.conv_state,
            raw_beta=actual_inputs.raw_beta,
            A_log=actual_inputs.A_log,
            dt_bias=actual_inputs.dt_bias,
            lower_bound=_LOWER_BOUND,
            state=actual_inputs.state,
            state_indices=actual_inputs.state_indices,
            output_gate=actual_inputs.output_gate,
            norm_weight=actual_inputs.norm_weight,
            norm_eps=_NORM_EPS,
        )
    torch.cuda.synchronize()
    reference = _reference(reference_inputs)
    assert actual_inputs.state.dtype == torch.bfloat16
    assert _relative_rmse(reference, out) < 1e-3
    assert _relative_rmse(reference_inputs.state, actual_inputs.state) < 1e-3
    # Only elements whose FP32 update sits on a BF16 rounding tie may differ;
    # a truncating store or a misread load would flip far more bits.
    reference_bits = reference_inputs.state.view(torch.int16)
    actual_bits = actual_inputs.state.view(torch.int16)
    assert (reference_bits != actual_bits).float().mean().item() <= 1e-3
    assert torch.equal(reference_inputs.conv_state, actual_inputs.conv_state)


@_requires_aiter_bf16_state
def test_aiter_fused_kda_decode_covers_c48_bf16_state_layout() -> None:
    """The c48 rejection: state (673, 12, 128, 128) bf16, conv (673, 4608, 3)."""
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip
    from sglang.srt.environ import envs

    cache = 673
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=2)
    inputs.state = torch.empty(
        (cache, _HEADS, _DIM, _DIM), dtype=torch.bfloat16, device=_DEVICE
    )
    assert inputs.state.stride() == (196608, 16384, 128, 1)
    # SGLang's [cache, 3, 4608] conv cache viewed through transpose(-1, -2).
    inputs.conv_state = torch.empty(
        (cache, _CONV_WIDTH - 1, _CHANNELS), dtype=torch.bfloat16, device=_DEVICE
    ).transpose(-1, -2)
    assert inputs.conv_state.stride() == (13824, 1, 4608)
    with (
        envs.SGLANG_OPT_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
        envs.SGLANG_OPT_HIP_K3_FLYDSL_SOURCE.override("aiter"),
    ):
        assert _adapter_covers(kda_fused_decode_aiter_hip, f_a, f_b_weight, inputs)
        fp16_inputs = replace(inputs, state=inputs.state.to(torch.float16))
        assert not _adapter_covers(
            kda_fused_decode_aiter_hip, f_a, f_b_weight, fp16_inputs
        )


def test_sglang_source_does_not_cover_bf16_state() -> None:
    """The vendored kernel is FP32-state only, so dispatch must not route BF16 to it."""
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip
    from sglang.srt.environ import envs

    f_a, f_b_weight, inputs = _make_fb_inputs(batch=2)
    with (
        envs.SGLANG_OPT_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
        envs.SGLANG_OPT_HIP_K3_FLYDSL_SOURCE.override("sglang"),
    ):
        assert _adapter_covers(kda_fused_decode_aiter_hip, f_a, f_b_weight, inputs)
        _bf16_dense_state(inputs)
        assert not _adapter_covers(kda_fused_decode_aiter_hip, f_a, f_b_weight, inputs)
        with pytest.raises(ValueError, match="`state` must have dtype"):
            _run_with_f_b(f_a, f_b_weight, inputs)


@_requires_aiter_bf16_state
def test_aiter_fused_kda_warmup_builds_every_state_dtype(monkeypatch) -> None:
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip
    from sglang.srt.environ import envs

    f_a, f_b_weight, inputs = _make_fb_inputs(batch=1)
    warmed_dtypes = []
    real_run = kda_fused_decode_aiter_hip.run

    def recording_run(**kwargs):
        warmed_dtypes.append(kwargs["state"].dtype)
        return real_run(**kwargs)

    monkeypatch.setattr(kda_fused_decode_aiter_hip, "run", recording_run)
    monkeypatch.setattr(kda_fused_decode_aiter_hip, "_WARMED", set())
    with (
        envs.SGLANG_OPT_HIP_K3_KDA_FUSED_BACKEND.override("aiter"),
        envs.SGLANG_OPT_HIP_K3_FLYDSL_SOURCE.override("aiter"),
    ):
        kda_fused_decode_aiter_hip.warmup(
            f_b_weight=f_b_weight,
            conv_weight=inputs.conv_weight,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            lower_bound=_LOWER_BOUND,
            norm_weight=inputs.norm_weight,
            norm_eps=_NORM_EPS,
        )
    assert warmed_dtypes == [torch.float32, torch.bfloat16]


@_requires_aiter_kda
@pytest.mark.parametrize("batch", [1, 8, 16])
def test_vendored_and_aiter_f_b_decode_agree(batch: int) -> None:
    from aiter.ops.flydsl.linear_attention_kernels import (
        flydsl_kda_decode_with_f_b as aiter_decode,
    )

    f_a, f_b_weight, seed = _make_fb_inputs(batch)
    vendored_inputs = _copy_inputs(seed)
    aiter_inputs = _copy_inputs(seed)

    vendored = _run_with_f_b(f_a, f_b_weight, vendored_inputs)
    aiter_out = aiter_decode(
        f_a=f_a,
        f_b_weight=f_b_weight,
        x=aiter_inputs.x,
        conv_weight=aiter_inputs.conv_weight,
        conv_bias=None,
        conv_state=aiter_inputs.conv_state,
        raw_beta=aiter_inputs.raw_beta,
        A_log=aiter_inputs.A_log,
        dt_bias=aiter_inputs.dt_bias,
        lower_bound=_LOWER_BOUND,
        state=aiter_inputs.state,
        state_indices=aiter_inputs.state_indices,
        output_gate=aiter_inputs.output_gate,
        norm_weight=aiter_inputs.norm_weight,
        norm_eps=_NORM_EPS,
    )
    torch.cuda.synchronize()

    assert _relative_rmse(vendored, aiter_out) < 1e-3
    assert _relative_rmse(vendored_inputs.state, aiter_inputs.state) < 1e-3
    assert torch.equal(vendored_inputs.conv_state, aiter_inputs.conv_state)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
