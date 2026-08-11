import inspect
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)


def test_aiter_fused_rope_path_requires_rotary_embedding():
    should_fuse_rope = (
        forward_mla_rocm.DeepseekMLARocmForwardMixin._skip_rope_for_aiter_fused_mla
    )

    with patch.object(forward_mla_rocm, "_use_aiter_gfx95", True):
        kimi_nope = SimpleNamespace(
            current_attention_backend="aiter",
            rotary_emb=None,
        )
        standard_mla = SimpleNamespace(
            current_attention_backend="aiter",
            rotary_emb=object(),
        )

        assert not should_fuse_rope(kimi_nope)
        assert should_fuse_rope(standard_mla)


def test_twelve_head_prefill_fallback_precedes_empty_prefix_fast_path():
    source = inspect.getsource(AiterAttnBackend.forward_extend)

    fallback = source.index("if layer.tp_q_head_num not in (16, 128):")
    empty_prefix = source.index(
        "extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)"
    )

    assert fallback < empty_prefix
