"""fp8-Q asm MLA decode: a producer-padded 16-head Q skips the backend re-pad."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import aiter_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TOKENS, _HEADS, _PADDED, _DIM = 4, 12, 16, 576
_FP8 = aiter_backend.fp8_dtype


def _backend():
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.num_head_padded = _PADDED
    backend.input_dtype = torch.bfloat16
    backend.k_scale = torch.ones(1)
    return backend


def _layer(pad_zero):
    return SimpleNamespace(
        tp_q_head_num=_HEADS,
        qk_head_dim=_DIM,
        v_head_dim=512,
        mla_q_pad_heads_zero=pad_zero,
    )


class TestA8W8PrepaddedQ(unittest.TestCase):
    def _run(self, q, pad_zero):
        seen = {}

        def fake_decode(q_in, k_buf, o, **kwargs):
            seen["q_in"] = q_in.clone()
            seen["q_in_ptr"] = q_in.data_ptr()

        def fake_quant(x, scale):
            seen["quantized"] = True
            return x.to(_FP8), scale

        k_buf = torch.ones(8, 1, 1, _DIM).to(_FP8)
        with (
            patch.object(aiter_backend, "mla_decode_fwd", fake_decode),
            patch.object(aiter_backend, "scaled_fp8_quant", fake_quant),
        ):
            out = _backend()._mla_decode_fwd_a8w8(q, k_buf, _layer(pad_zero))
        self.assertEqual(tuple(out.shape), (_TOKENS, _HEADS, 512))
        self.assertEqual(k_buf[0].float().abs().sum().item(), 0.0)
        return seen

    def test_bf16_q_is_quantized_and_padded(self):
        q = torch.randn(_TOKENS, _HEADS, _DIM).bfloat16()
        seen = self._run(q, pad_zero=True)
        self.assertTrue(seen["quantized"])
        self.assertEqual(tuple(seen["q_in"].shape), (_TOKENS, _PADDED, _DIM))
        self.assertTrue(torch.equal(seen["q_in"][:, :_HEADS].float(), q.to(_FP8).float()))
        self.assertEqual(seen["q_in"][:, _HEADS:].float().abs().sum().item(), 0.0)

    def test_prepadded_fp8_q_is_passed_through(self):
        q = torch.zeros(_TOKENS, _PADDED, _DIM, dtype=_FP8)
        q[:, :_HEADS] = torch.randn(_TOKENS, _HEADS, _DIM).to(_FP8)
        seen = self._run(q, pad_zero=True)
        self.assertNotIn("quantized", seen)
        self.assertEqual(seen["q_in_ptr"], q.data_ptr())
        self.assertTrue(torch.equal(seen["q_in"].float(), q.float()))

    def test_unflagged_prepadded_q_still_zeroes_pad_heads(self):
        q = torch.ones(_TOKENS, _PADDED, _DIM).to(_FP8)
        seen = self._run(q, pad_zero=False)
        self.assertNotEqual(seen["q_in_ptr"], q.data_ptr())
        self.assertEqual(seen["q_in"][:, _HEADS:].float().abs().sum().item(), 0.0)
        self.assertTrue(
            torch.equal(seen["q_in"][:, :_HEADS].float(), q[:, :_HEADS].float())
        )


if __name__ == "__main__":
    unittest.main()
