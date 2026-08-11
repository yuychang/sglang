"""Kimi-K3 TP8 correctness tests for AITER MLA prefill and decode."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestKimiK3AiterMlaKernels(CustomTestCase):
    NUM_HEADS = 12
    KV_LORA_RANK = 512
    ROPE_HEAD_DIM = 64
    QK_HEAD_DIM = KV_LORA_RANK + ROPE_HEAD_DIM
    SCALE = QK_HEAD_DIM**-0.5

    def test_prefill_supports_twelve_local_heads(self):
        torch.manual_seed(1)
        q_len, kv_len, pool_rows = 4, 20, 47
        physical = torch.randperm(pool_rows, device="cuda")[:kv_len].to(torch.int32)
        pool = torch.randn(
            pool_rows,
            1,
            self.QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q = torch.randn(
            q_len,
            self.NUM_HEADS,
            self.QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")
        qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
        layer = self._layer()
        backend = self._backend(pool)

        out = backend._mla_prefill_fwd_triton_paged(
            q,
            layer,
            None,
            pool,
            kv_indptr,
            physical,
            qo_indptr,
        ).view(q_len, self.NUM_HEADS, self.KV_LORA_RANK)

        kv = pool[physical.long(), 0]
        logits = self._logits(q, kv)
        prefix_len = kv_len - q_len
        key_pos = torch.arange(kv_len, device="cuda")
        query_pos = prefix_len + torch.arange(q_len, device="cuda")
        logits.masked_fill_(
            key_pos[None, None, :] > query_pos[:, None, None],
            float("-inf"),
        )
        ref = torch.einsum(
            "thk,kd->thd", logits.softmax(-1), kv[:, : self.KV_LORA_RANK].float()
        ).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)

    def test_decode_calls_native_twelve_head_gluon_mla(self):
        torch.manual_seed(2)
        kv_len = 128
        pool = torch.randn(
            kv_len,
            1,
            self.QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q = torch.randn(
            1,
            self.NUM_HEADS,
            self.QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        backend = self._backend(pool)
        backend.max_context_len = kv_len
        backend.forward_metadata = SimpleNamespace(
            kv_indices=torch.arange(kv_len, dtype=torch.int32, device="cuda"),
            kv_indptr=torch.tensor([0, kv_len], dtype=torch.int32, device="cuda"),
        )

        out = backend._mla_decode_fwd_gluon(q, self._layer())
        kv = pool[:, 0]
        logits = self._logits(q, kv)
        ref = torch.einsum(
            "bhk,kd->bhd", logits.softmax(-1), kv[:, : self.KV_LORA_RANK].float()
        ).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)

    def test_decode_supports_batched_fp8_kv(self):
        torch.manual_seed(3)
        seq_lens = (1, 127, 128, 129)
        batch_size = len(seq_lens)
        total_kv = sum(seq_lens)
        pool_rows = total_kv + 37
        kv_scale = 0.375
        physical = torch.randperm(pool_rows, device="cuda")[:total_kv].to(
            torch.int32
        )
        pool_bf16 = torch.randn(
            pool_rows,
            1,
            self.QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        pool = (pool_bf16 / kv_scale).to(fp8_dtype)
        q = torch.randn(
            batch_size,
            self.NUM_HEADS,
            self.QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        kv_indptr = torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        backend = self._backend(pool)
        backend.max_context_len = max(seq_lens)
        backend.forward_metadata = SimpleNamespace(
            kv_indices=physical,
            kv_indptr=kv_indptr,
        )
        layer = self._layer()
        layer.k_scale_float = kv_scale

        out = backend._mla_decode_fwd_gluon(q, layer)
        refs = []
        offset = 0
        for batch_idx, seq_len in enumerate(seq_lens):
            indices = physical[offset : offset + seq_len].long()
            kv = pool[indices, 0].float() * kv_scale
            logits = self._logits(q[batch_idx : batch_idx + 1], kv)
            refs.append(
                torch.einsum(
                    "bhk,kd->bhd",
                    logits.softmax(-1),
                    kv[:, : self.KV_LORA_RANK],
                )
            )
            offset += seq_len
        ref = torch.cat(refs).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)

    def _backend(self, pool):
        backend = AiterAttnBackend.__new__(AiterAttnBackend)
        backend.input_dtype = torch.bfloat16
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _layer_id: pool
        )
        return backend

    def _layer(self):
        return SimpleNamespace(
            tp_q_head_num=self.NUM_HEADS,
            v_head_dim=self.KV_LORA_RANK,
            qk_head_dim=self.QK_HEAD_DIM,
            scaling=self.SCALE,
            layer_id=0,
        )

    def _logits(self, q, kv):
        return (
            q[..., : self.KV_LORA_RANK].float() @ kv[:, : self.KV_LORA_RANK].float().T
            + q[..., self.KV_LORA_RANK :].float() @ kv[:, self.KV_LORA_RANK :].float().T
        ) * self.SCALE


if __name__ == "__main__":
    unittest.main()
