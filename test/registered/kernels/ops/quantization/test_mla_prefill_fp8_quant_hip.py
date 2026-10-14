"""Batched MLA prefill FP8 quant: bit-identical to AiterAttnBackend's per-tensor quant."""

import os

# The reference is AITER's static per-tensor quant, picked at fp8_kernel import.
os.environ.setdefault("SGLANG_USE_AITER", "1")

import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

HEADS = 12
QK_DIM = 192
V_DIM = 128
NOPE_DIM = 128


def _qkv(num_q, num_kv, dtype=torch.bfloat16, scale=1.0, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((num_q, HEADS, QK_DIM), device="cuda", dtype=dtype, generator=gen)
    k = torch.randn((num_kv, HEADS, QK_DIM), device="cuda", dtype=dtype, generator=gen)
    # V is the trailing slice of the fused kv_b_proj output, as in forward_mha.
    kv = torch.randn(
        (num_kv, HEADS, NOPE_DIM + V_DIM), device="cuda", dtype=dtype, generator=gen
    )
    kv *= 0.1 * scale
    return [q * scale, k * (3 * scale), kv[..., NOPE_DIM:]]


def _flydsl_fp8_available():
    try:
        from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
            flydsl_flash_attn_fp8_supported,
        )
    except ImportError:
        return False
    return flydsl_flash_attn_fp8_supported(
        torch.device("cuda"), HEADS, HEADS, QK_DIM, V_DIM
    )


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "e4m3fn reference quant runs on gfx950"
)
class TestMlaPrefillFp8QuantHip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.quantization import fp8_kernel
        from sglang.kernels.ops.quantization import mla_prefill_fp8_quant_hip as module
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        assert fp8_kernel._use_aiter, "reference must run AITER's static quant"
        cls.module = module
        cls.fused = staticmethod(module.mla_prefill_fp8_quant)
        cls.backend_cls = AiterAttnBackend
        cls.reference = staticmethod(AiterAttnBackend._quantize_mla_prefill_fp8)

    def _assert_bit_equal(self, tensors):
        got = self.fused(tensors)
        self.assertIsNotNone(got)
        self.assertEqual(len(got), len(tensors))
        for i, x in enumerate(tensors):
            ref_q, ref_s = self.reference(x)
            new_q, new_s = got[i]
            self.assertEqual(new_q.dtype, ref_q.dtype)
            self.assertEqual(new_q.shape, ref_q.shape)
            self.assertTrue(new_q.is_contiguous())
            self.assertEqual(new_s.dtype, torch.float32)
            self.assertEqual(new_s.shape, (1,))
            self.assertTrue(
                torch.equal(new_s.view(torch.int32), ref_s.view(torch.int32)),
                f"operand {i}: scale {new_s.item()!r} != {ref_s.item()!r}",
            )
            mismatch = (new_q.view(torch.uint8) != ref_q.view(torch.uint8)).sum()
            self.assertEqual(mismatch.item(), 0, f"operand {i}: FP8 bytes differ")
        return got

    def test_extend_qkv(self):
        for num_tokens in (1, 7, 512, 2048, 4096, 8192):
            with self.subTest(num_tokens=num_tokens):
                self._assert_bit_equal(_qkv(num_tokens, num_tokens, seed=num_tokens))

    def test_prefix_chunk_qkv(self):
        for num_q, num_kv in ((2048, 8192), (8192, 32768), (512, 65536)):
            with self.subTest(num_q=num_q, num_kv=num_kv):
                tensors = _qkv(num_q, num_kv, seed=num_kv)
                self.assertFalse(tensors[2].is_contiguous())
                self._assert_bit_equal(tensors)

    def test_fewer_operands_and_fp16(self):
        for num_tokens in (1, 333, 8192):
            with self.subTest(q_only=num_tokens):
                self._assert_bit_equal(_qkv(num_tokens, 1, seed=3)[:1])
        self._assert_bit_equal(_qkv(100, 300, seed=4)[:2])
        self._assert_bit_equal(_qkv(1024, 2048, dtype=torch.float16, seed=5))

    def test_random_magnitudes(self):
        for seed in range(20):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            exponent = torch.randint(-60, 60, (3,), generator=gen, device="cuda")
            tensors = _qkv(257 + seed, 911 + seed, seed=seed)
            for x, e in zip(tensors, exponent.tolist()):
                x.mul_(2.0**e)
            with self.subTest(seed=seed):
                self._assert_bit_equal(tensors)

    def test_all_zeros_and_negative_zeros(self):
        q = torch.zeros((64, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
        k = -torch.zeros((64, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
        v = torch.zeros(
            (64, HEADS, NOPE_DIM + V_DIM), device="cuda", dtype=torch.bfloat16
        )
        got = self._assert_bit_equal([q, k, v[..., NOPE_DIM:]])
        self.assertEqual(
            got[0][1].item(), torch.tensor(1e-12, dtype=torch.float32).item()
        )
        self.assertTrue((got[1][0].view(torch.uint8) == 0x80).all())

    def test_extreme_values(self):
        finfo = torch.finfo(torch.bfloat16)
        tensors = _qkv(300, 300, seed=6)
        tensors[0] = tensors[0].clamp(-1, 1) * finfo.max
        tensors[1][17, 3, 100] = finfo.max
        tensors[1][18, 4, 101] = -finfo.max
        tensors[2].mul_(1e30)
        self._assert_bit_equal(tensors)

        tensors = _qkv(300, 300, seed=7)
        tensors[0].mul_(1e-38)
        tensors[1].mul_(1e-12)
        tensors[2].mul_(1e-40)
        self._assert_bit_equal(tensors)

    def test_non_finite_values(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            tensors = _qkv(64, 64, seed=12)
            tensors[1][3, 4, 5] = value
            with self.subTest(value=value):
                self._assert_bit_equal(tensors)

    def test_single_outlier_anywhere(self):
        for position in (0, 4095, 4096, 123_457, 300 * HEADS * QK_DIM - 1):
            q = torch.randn((300, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
            q.view(-1)[position] = -1234.0
            with self.subTest(position=position):
                self._assert_bit_equal([q])

    def test_every_bf16_pattern_under_many_scales(self):
        bits = torch.arange(0, 1 << 16, dtype=torch.int32, device="cuda")
        values = bits.to(torch.int16).view(torch.bfloat16)
        values = values[torch.isfinite(values)]
        gen = torch.Generator(device="cpu").manual_seed(8)
        anchors = [1.0, 448.0, 1e-3, 1e-20, 3e5, 1e30, torch.finfo(torch.bfloat16).max]
        anchors += (10.0 ** (torch.rand(64, generator=gen) * 70 - 35)).tolist()
        for anchor in anchors:
            anchor = torch.tensor(anchor, dtype=torch.bfloat16).item()
            x = values[values.float().abs() <= anchor]
            x = torch.cat([x, torch.tensor([anchor], device="cuda", dtype=x.dtype)])
            x = torch.cat([x, x.new_zeros((-x.numel()) % QK_DIM)])
            with self.subTest(anchor=anchor):
                self._assert_bit_equal([x.view(-1, 1, QK_DIM)])

    def test_strided_and_misaligned_views(self):
        # Head-sliced Q and a non-16-byte-aligned base pointer each change the
        # kernel specialization; the repeat calls take the cached launch.
        padded = torch.randn((500, 16, QK_DIM), device="cuda", dtype=torch.bfloat16)
        q = padded[:, :HEADS, :]
        flat = torch.randn(
            (1 + 500 * HEADS * QK_DIM,), device="cuda", dtype=torch.bfloat16
        )
        misaligned = flat[1:].view(500, HEADS, QK_DIM)
        self.assertNotEqual(misaligned.data_ptr() % 16, 0)
        aligned = flat[:-1].view(500, HEADS, QK_DIM)
        for _ in range(2):
            self._assert_bit_equal([q, aligned])
            self._assert_bit_equal([q, misaligned])
            self._assert_bit_equal([misaligned, q, aligned])

    def test_regular_triton_launch(self):
        with (
            mock.patch.object(self.module, "_DIRECT_LAUNCH", False),
            mock.patch.dict(self.module._compiled, clear=True),
        ):
            for _ in range(2):
                self._assert_bit_equal(_qkv(700, 1300, seed=11))
            self.assertEqual(self.module._compiled, {})

    def test_unsupported_inputs_decline(self):
        x = torch.randn((8, HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16)
        for name, tensors in (
            ("fp32", [x.float()]),
            ("2d", [x.view(8, -1)]),
            ("dim not multiple of 64", [x[..., :100]]),
            ("strided last dim", [x.transpose(1, 2)]),
            ("empty", [x[:0]]),
            ("mixed dtypes", [x, x.half()]),
            ("four operands", [x] * 4),
        ):
            with self.subTest(name):
                self.assertIsNone(self.fused(tensors))

    def _backend(self, batched):
        backend = self.backend_cls.__new__(self.backend_cls)
        backend._mla_prefill_batched_fp8_quant = self.fused if batched else None
        return backend

    def test_backend_helper_falls_back_for_declined_shapes(self):
        backend = self._backend(batched=True)
        tensors = _qkv(1000, 3000, seed=9)
        for x, (x8, scale) in zip(
            tensors, backend._quantize_mla_prefill_fp8_operands(*tensors)
        ):
            ref8, ref_scale = self.reference(x)
            self.assertTrue(
                torch.equal(scale.view(torch.int32), ref_scale.view(torch.int32))
            )
            self.assertTrue(torch.equal(x8.view(torch.uint8), ref8.view(torch.uint8)))
        odd = torch.randn((10, 3, 100), device="cuda", dtype=torch.bfloat16)
        ((odd8, odd_scale),) = backend._quantize_mla_prefill_fp8_operands(odd)
        ref8, ref_scale = self.reference(odd)
        self.assertTrue(torch.equal(odd_scale, ref_scale))
        self.assertTrue(torch.equal(odd8.view(torch.uint8), ref8.view(torch.uint8)))

    def test_flydsl_prefill_output_unchanged(self):
        if not _flydsl_fp8_available():
            self.skipTest("FlyDSL FP8 varlen FMHA is unavailable")
        from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_func

        for causal in (True, False):
            seq_q = [300, 1, 1024, 77]
            seq_k = seq_q if causal else [1500, 129, 4096, 200]
            q, k, v = _qkv(sum(seq_q), sum(seq_k), seed=10)
            cu_q = torch.tensor([0] + seq_q, device="cuda").cumsum(0).to(torch.int32)
            cu_k = torch.tensor([0] + seq_k, device="cuda").cumsum(0).to(torch.int32)
            outputs = []
            for batched in (False, True):
                backend = self._backend(batched)
                backend.use_mla_flydsl_fp8_prefill = True
                backend._mla_flydsl_varlen_attn = flydsl_flash_attn_varlen_func
                backend._mla_flydsl_unit_descale = None
                backend._mla_flydsl_fallback_logged = False
                out, lse = backend._mla_flydsl_fp8_varlen_attn(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=max(seq_q),
                    max_seqlen_k=max(seq_k),
                    softmax_scale=QK_DIM**-0.5,
                    causal=causal,
                    return_lse=True,
                )[:2]
                outputs.append((out, lse))
            with self.subTest(causal=causal):
                self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
                self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))


if __name__ == "__main__":
    unittest.main()
