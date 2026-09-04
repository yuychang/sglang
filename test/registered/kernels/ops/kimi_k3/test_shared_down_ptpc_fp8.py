import pytest
import torch
from aiter.jit.utils.chip_info import get_gfx_runtime

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="fused Kimi-K3 PTPC shared-down requires gfx950",
)

HIDDEN = 7168
INTERMEDIATE = 768
BUCKETS = (1, 2, 4)


def _fused():
    from sglang.kernels.ops.kimi_k3.flydsl import shared_down_ptpc_fp8

    return shared_down_ptpc_fp8


def _weights(seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = (
        torch.randn((HIDDEN, INTERMEDIATE), generator=generator)
        .mul_(INTERMEDIATE**-0.5)
        .bfloat16()
        .cuda()
    )
    return weight, generator


@pytest.mark.parametrize("num_tokens", BUCKETS)
@torch.inference_mode()
def test_fused_matches_split_ptpc_path(num_tokens):
    """The fusion must not change PTPC numerics, only the launch count."""
    from sglang.kernels.ops.kimi_k3 import ptpc_fp8_aiter_hip

    fused = _fused()
    if not fused.is_available() or not ptpc_fp8_aiter_hip.available():
        pytest.skip("fused or split PTPC shared-down unavailable")
    weight, generator = _weights(23 + num_tokens)
    x = torch.randn((num_tokens, INTERMEDIATE), generator=generator).bfloat16().cuda()

    packed_w, packed_s, logical_n = ptpc_fp8_aiter_hip.pack(weight)
    split = ptpc_fp8_aiter_hip.run(x, packed_w, packed_s, logical_n)

    fused_w, fused_s = fused.quantize_shared_down_weight(weight)
    got = fused.kimi_k3_shared_down_ptpc_fp8(
        x, fused_w, fused_s, token_buckets=(num_tokens,)
    )
    torch.cuda.synchronize()

    assert got.shape == (num_tokens, HIDDEN)
    assert got.dtype == torch.bfloat16
    # Both sides quantize identically; only the FP32 accumulation order
    # differs, so they agree far more tightly than FP8 error itself.
    reference = split.float()
    rel = ((got.float() - reference).norm() / reference.norm()).item()
    assert rel < 5e-3, rel


@torch.inference_mode()
def test_fused_replays_under_graph_capture():
    fused = _fused()
    if not fused.is_available():
        pytest.skip("fused PTPC shared-down unavailable")
    weight, generator = _weights(31)
    x = torch.randn((2, INTERMEDIATE), generator=generator).bfloat16().cuda()
    fused_w, fused_s = fused.quantize_shared_down_weight(weight)
    out = torch.empty((2, HIDDEN), dtype=torch.bfloat16, device=x.device)

    fused.warmup(fused_w, fused_s, token_buckets=(2,))
    eager = fused.kimi_k3_shared_down_ptpc_fp8(
        x, fused_w, fused_s, token_buckets=(2,)
    ).clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused.kimi_k3_shared_down_ptpc_fp8(
            x, fused_w, fused_s, out=out, token_buckets=(2,)
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, rtol=0, atol=0)


@torch.inference_mode()
def test_all_zero_activation_stays_finite():
    """A zero row has no amax; the scale must not become a division by zero."""
    fused = _fused()
    if not fused.is_available():
        pytest.skip("fused PTPC shared-down unavailable")
    weight, _ = _weights(37)
    fused_w, fused_s = fused.quantize_shared_down_weight(weight)
    x = torch.zeros((2, INTERMEDIATE), dtype=torch.bfloat16, device=weight.device)
    got = fused.kimi_k3_shared_down_ptpc_fp8(
        x, fused_w, fused_s, token_buckets=(2,)
    )
    torch.cuda.synchronize()
    assert torch.isfinite(got.float()).all()
    assert got.abs().max().item() == 0.0


@torch.inference_mode()
def test_support_predicate_fails_closed():
    fused = _fused()
    if not fused.is_available():
        pytest.skip("fused PTPC shared-down unavailable")
    weight, generator = _weights(41)
    fused_w, fused_s = fused.quantize_shared_down_weight(weight)
    x = torch.randn((2, INTERMEDIATE), generator=generator).bfloat16().cuda()

    # Batch outside the enabled buckets.
    assert not fused.supports_kimi_k3_shared_down_ptpc_fp8(
        x, fused_w, fused_s, token_buckets=(4,)
    )
    # Wrong activation width.
    assert not fused.supports_kimi_k3_shared_down_ptpc_fp8(
        x[:, :512].contiguous(), fused_w, fused_s, token_buckets=(2,)
    )
    # Unquantized weight.
    assert not fused.supports_kimi_k3_shared_down_ptpc_fp8(
        x, weight, fused_s, token_buckets=(2,)
    )
    with pytest.raises(ValueError):
        fused.kimi_k3_shared_down_ptpc_fp8(
            x, fused_w, fused_s, token_buckets=(4,)
        )


@torch.inference_mode()
def test_quantize_rejects_wrong_shape():
    fused = _fused()
    if not fused.is_available():
        pytest.skip("fused PTPC shared-down unavailable")
    with pytest.raises(ValueError):
        fused.quantize_shared_down_weight(
            torch.zeros((HIDDEN, 512), dtype=torch.bfloat16, device="cuda")
        )
