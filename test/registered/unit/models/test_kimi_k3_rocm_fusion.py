"""CPU tests for the KDA in-proj producer-quant gate.

With ``SGLANG_ROCM_K3_FUSE_KDA_INPROJ=0`` the merged GEMM is absent. Quark
then dequantizes the 12-wide ``b_proj`` to BF16, and that linear rejects the
``(fp8, scale)`` tuple the attention-residual producer would otherwise emit.
"""

from types import SimpleNamespace

import torch

from sglang.srt.models.kimi_k3_rocm_fusion import _k3_inproj_needs_bf16_hidden
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _bf16_linear():
    """Quark narrow per-channel FP8 after the N%16 dequant."""
    return SimpleNamespace(
        weight=torch.empty(32, 12, dtype=torch.bfloat16),
        weight_scale=None,
        scheme=SimpleNamespace(per_token=True, weight_qscheme="per_channel"),
    )


def _group128_linear():
    return SimpleNamespace(
        weight=torch.empty(128, 512, dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(128, 4, dtype=torch.float32),
    )


class TestInprojNeedsBf16Hidden(CustomTestCase):
    def test_merged_gemm_emits_the_tuple_only(self):
        attn = SimpleNamespace(
            _qkvgbfa_sizes=[1, 2, 3],
            _qkvgbfa_bs_limit=64,
            _bfa_w=None,
            use_full_rank_gate=True,
            b_proj=_bf16_linear(),
            f_a_proj=_bf16_linear(),
        )
        self.assertFalse(_k3_inproj_needs_bf16_hidden(attn, 8))

    def test_bf16_tiny_gemm_keeps_hidden(self):
        attn = SimpleNamespace(
            _qkvgbfa_sizes=None,
            _bfa_w=torch.empty(1),
            use_full_rank_gate=True,
        )
        self.assertTrue(_k3_inproj_needs_bf16_hidden(attn, 8))

    def test_dequantized_b_proj_keeps_bf16_hidden(self):
        attn = SimpleNamespace(
            _qkvgbfa_sizes=None,
            _bfa_w=None,
            use_full_rank_gate=True,
            b_proj=_bf16_linear(),
            f_a_proj=_group128_linear(),
        )
        self.assertTrue(_k3_inproj_needs_bf16_hidden(attn, 8))

    def test_split_path_above_the_token_cap_keeps_bf16_hidden(self):
        attn = SimpleNamespace(
            _qkvgbfa_sizes=[1, 2, 3],
            _qkvgbfa_bs_limit=4,
            _bfa_w=None,
            use_full_rank_gate=True,
            b_proj=_bf16_linear(),
            f_a_proj=_group128_linear(),
        )
        self.assertTrue(_k3_inproj_needs_bf16_hidden(attn, 8))

    def test_both_split_linears_can_consume_the_tuple(self):
        attn = SimpleNamespace(
            _qkvgbfa_sizes=None,
            _bfa_w=None,
            use_full_rank_gate=True,
            b_proj=_group128_linear(),
            f_a_proj=_group128_linear(),
        )
        self.assertFalse(_k3_inproj_needs_bf16_hidden(attn, 8))

    def test_mla_and_low_rank_kda_keep_the_old_gate(self):
        mla = SimpleNamespace(_qkvgbfa_sizes=None, _bfa_w=None)
        low_rank = SimpleNamespace(
            _qkvgbfa_sizes=None,
            _bfa_w=None,
            use_full_rank_gate=False,
            b_proj=_bf16_linear(),
            f_a_proj=_bf16_linear(),
        )
        self.assertFalse(_k3_inproj_needs_bf16_hidden(mla, 8))
        self.assertFalse(_k3_inproj_needs_bf16_hidden(low_rank, 8))
