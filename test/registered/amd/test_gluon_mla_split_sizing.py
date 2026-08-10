"""Unit test for the KV split count the non-DCP Gluon MLA decode path asks for.

``AiterAttnBackend._mla_decode_fwd_gluon`` cannot hand mla_gluon a NUM_KV_SPLITS.
The kernel derives it internally from ``min_kv_seq_len`` and passes it as a
constexpr, so it is baked into both the launch grid and the compiled kernel.
``gluon_mla_min_kv_seq_len`` therefore works backwards: it picks the
``min_kv_seq_len`` that yields the split count we want.

That makes the helper coupled to arithmetic living in aiter, which is what this
test pins. If an aiter bump changes how mla_gluon derives NUM_KV_SPLITS, the
expectations below stop matching and the coupling has to be re-checked --
otherwise the failure mode is silent: a split count that varies with the batch's
KV length reshapes the launch grid between cuda-graph capture and replay.

The properties that matter:

* stable per (bs, nhead) -- nothing about the batch's actual KV lengths may
  enter, or captured graphs replay against a grid they were not captured with
* a power of two -- NUM_KV_SPLITS is a constexpr, so every distinct value is a
  separate kernel compilation
* never over the workgroup budget -- more splits than one wave of workgroups
  buys nothing and costs a stage-2 reduce over empty splits
"""

import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# mla_gluon's bh16bn64 regime, mirrored from aiter/ops/triton/gluon/mla_gluon.py.
BLOCK_N = 64
BLOCK_H = 16
TARGET_WGS = 256

MAX_CONTEXT_LEN = 128 * 1024


def _aiter_num_kv_splits(bs, nhead, min_kv_seq_len, qlen=1):
    """What mla_gluon derives for NUM_KV_SPLITS, transcribed from its wrapper."""
    num_m_blocks = -(-nhead // BLOCK_H)
    return max(
        1,
        min(
            TARGET_WGS // (bs * qlen * num_m_blocks),
            -(-min_kv_seq_len // BLOCK_N),
        ),
    )


class TestGluonMlaSplitSizing(CustomTestCase):
    # Kimi-K3 has 96 attention heads; these are the local counts its supported
    # tp sizes produce, 12 (tp8) being the one that motivated the Gluon path.
    HEAD_COUNTS = (6, 12, 16, 24, 48, 96)
    BATCH_SIZES = (1, 2, 3, 5, 8, 16, 33, 64, 128, 256, 512)

    @staticmethod
    def _helper(bs, nhead):
        from sglang.srt.layers.attention.aiter_backend import gluon_mla_min_kv_seq_len

        return gluon_mla_min_kv_seq_len(bs, nhead, MAX_CONTEXT_LEN)

    def _splits(self, bs, nhead):
        return _aiter_num_kv_splits(bs, nhead, self._helper(bs, nhead))

    def test_split_count_is_a_power_of_two(self):
        for nhead in self.HEAD_COUNTS:
            for bs in self.BATCH_SIZES:
                splits = self._splits(bs, nhead)
                with self.subTest(bs=bs, nhead=nhead):
                    self.assertGreaterEqual(splits, 1)
                    self.assertEqual(splits & (splits - 1), 0, f"{splits} not 2^k")

    def test_split_count_stays_within_one_wave(self):
        for nhead in self.HEAD_COUNTS:
            for bs in self.BATCH_SIZES:
                splits = self._splits(bs, nhead)
                head_blocks = -(-nhead // BLOCK_H)
                with self.subTest(bs=bs, nhead=nhead):
                    # splits == 1 is the floor: a batch can be wide enough on its
                    # own to exceed the budget, and then there is nothing to trim.
                    if splits > 1:
                        self.assertLessEqual(bs * head_blocks * splits, TARGET_WGS)

    def test_split_count_does_not_depend_on_kv_length(self):
        # The helper takes only (bs, nhead, max_context_len); this pins that no
        # per-step quantity can leak in, which is what keeps a captured graph's
        # grid valid on replay.
        for nhead in self.HEAD_COUNTS:
            for bs in self.BATCH_SIZES:
                with self.subTest(bs=bs, nhead=nhead):
                    self.assertEqual(self._helper(bs, nhead), self._helper(bs, nhead))

    def test_small_context_caps_the_split_count(self):
        from sglang.srt.layers.attention.aiter_backend import gluon_mla_min_kv_seq_len

        # A model whose whole context is 4 KV tiles cannot use 256 splits.
        min_kv = gluon_mla_min_kv_seq_len(
            bs=1, num_heads=12, max_context_len=4 * BLOCK_N
        )
        self.assertEqual(_aiter_num_kv_splits(1, 12, min_kv), 4)

    def test_single_request_decode_fills_the_machine(self):
        # bs=1 is where a fixed NUM_KV_SPLITS=1 would leave the GPU idle: one
        # workgroup for the whole sequence.
        self.assertEqual(self._splits(bs=1, nhead=12), 256)
        self.assertEqual(self._splits(bs=1, nhead=96), 32)

    def test_wide_batch_falls_back_to_one_split(self):
        # 256 requests x 1 head block already exceeds the wave on their own.
        self.assertEqual(self._splits(bs=256, nhead=12), 1)
        self.assertEqual(self._splits(bs=512, nhead=96), 1)


if __name__ == "__main__":
    unittest.main()
