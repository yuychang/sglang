import torch

from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
    _absorb_weight_bf16,
)


def test_unit_scale_preserves_bf16_weight():
    """A BF16 absorb weight at unit scale must survive loading unchanged.

    The `* w_scale` pass on an identity scale silently perturbed the weight;
    the no-op must return the caller's tensor, not a rescaled copy.
    """
    weight = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    assert _absorb_weight_bf16(weight, 1.0) is weight
    assert _absorb_weight_bf16(weight, 1) is weight


def test_non_unit_scale_dequantizes():
    weight = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    assert torch.equal(_absorb_weight_bf16(weight, 2.0), weight * 2)


def test_non_bf16_weight_is_converted_even_at_unit_scale():
    weight = torch.randn(2, 4, 8).to(torch.float8_e4m3fn)
    out = _absorb_weight_bf16(weight, 1.0)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, weight.to(torch.bfloat16))
