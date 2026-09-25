"""Unit tests for srt/models/kimi_k3_rocm_mla."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.models import kimi_k3_rocm_mla
from sglang.srt.models.kimi_k3_rocm_mla import (
    _cached_buffer,
    k3_init_mla_q_cache,
    k3_try_fused_mla_q_cache,
)
from sglang.test.test_utils import CustomTestCase


def _init(enabled: bool) -> nn.Module:
    attn = nn.Module()
    env = kimi_k3_rocm_mla.envs.SGLANG_ROCM_K3_AITER_MLA_Q_CACHE_FUSION
    with mock.patch.object(type(env), "get", return_value=enabled):
        k3_init_mla_q_cache(attn)
    return attn


class TestInit(CustomTestCase):
    def test_enabled_registers_identity_rope(self):
        attn = _init(True)
        self.assertTrue(torch.equal(attn._k3_identity_rope_cos, torch.ones(1, 32)))
        self.assertTrue(torch.equal(attn._k3_identity_rope_sin, torch.zeros(1, 32)))
        self.assertEqual(attn._k3_mla_q_cache_scale.item(), 1.0)
        # Non-persistent: nothing leaks into the checkpoint.
        self.assertEqual(attn.state_dict(), {})

    def test_disabled_skips_fusion(self):
        attn = _init(False)
        self.assertIsNone(attn._k3_identity_rope_cos)
        self.assertIsNone(k3_try_fused_mla_q_cache(attn, *([None] * 6)))


class TestCachedBuffer(CustomTestCase):
    def test_reuses_and_grows(self):
        attn = nn.Module()
        a = _cached_buffer(attn, "_buf", (8, 16, 4), torch.float32, torch.device("cpu"))
        b = _cached_buffer(attn, "_buf", (4, 16, 4), torch.float32, torch.device("cpu"))
        self.assertEqual(b.shape, (4, 16, 4))
        self.assertEqual(a.data_ptr(), b.data_ptr())
        c = _cached_buffer(attn, "_buf", (9, 16, 4), torch.float32, torch.device("cpu"))
        self.assertEqual(c.shape, (9, 16, 4))

    def test_zero_init_pads(self):
        attn = nn.Module()
        buf = _cached_buffer(
            attn, "_buf", (2, 16, 4), torch.bfloat16, torch.device("cpu"), True
        )
        self.assertTrue(torch.equal(buf, torch.zeros_like(buf)))


if __name__ == "__main__":
    unittest.main()
