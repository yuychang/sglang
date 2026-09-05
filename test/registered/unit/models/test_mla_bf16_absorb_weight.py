import torch

from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
    _bf16_absorb_weight,
    _is_unit_host_scale,
)


def test_unit_host_scale_skips_multiply():
    weight = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    assert _is_unit_host_scale(1.0)
    assert _is_unit_host_scale(1)
    assert _is_unit_host_scale(None)
    assert not _is_unit_host_scale(2.0)
    assert not _is_unit_host_scale(torch.ones((), dtype=torch.float32))
    assert _bf16_absorb_weight(weight, 1.0) is weight
    scaled = _bf16_absorb_weight(weight, 2.0)
    assert torch.equal(scaled, weight * 2)
