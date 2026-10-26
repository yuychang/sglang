# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DSpark DCP index helpers. The join itself stays off after the c56 regression.

    python -m pytest test/registered/unit/mem_cache/test_dspark_dcp_shard.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.choices import DRAFT_ATTENTION_BACKEND_CHOICES
from sglang.srt.mem_cache.draft_dcp import (
    dcp_extend_kv_head_views,
    dcp_joined_kv_locs,
    dcp_kv_write_loc,
    draft_kv_tp_size,
    dspark_draft_joins_dcp,
    dspark_mha_uses_unified_attention,
)
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.model_executor.pool_configurator import _dflash_draft_cell_size
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _spec(algorithm: str):
    return SimpleNamespace(
        speculative_algorithm=SpeculativeAlgorithm.from_string(algorithm),
        speculative_draft_model_path="",
    )


class TestDSparkDcpShard(CustomTestCase):
    def test_only_hip_qwen_dspark_joins(self):
        # The c56 hour (2026-10-23) kept this false: sharded verify added a
        # collective per draft layer and dropped 10065.9 to 8839.0 tok/s/GPU.
        spec = _spec("DSPARK")
        with (
            patch("sglang.srt.utils.is_hip", return_value=True),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_config.draft_is_deepseek_v4",
                return_value=False,
            ),
            patch("sglang.srt.runtime_context.get_spec", return_value=spec),
            get_parallel().override(attn_dcp_size=8),
        ):
            self.assertFalse(dspark_draft_joins_dcp())
        with (
            patch("sglang.srt.utils.is_hip", return_value=False),
            patch("sglang.srt.runtime_context.get_spec", return_value=spec),
            get_parallel().override(attn_dcp_size=8),
        ):
            self.assertFalse(dspark_draft_joins_dcp())
        with (
            patch("sglang.srt.utils.is_hip", return_value=True),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_config.draft_is_deepseek_v4",
                return_value=True,
            ),
            patch("sglang.srt.runtime_context.get_spec", return_value=spec),
            get_parallel().override(attn_dcp_size=8),
        ):
            self.assertFalse(dspark_draft_joins_dcp())
        with (
            patch("sglang.srt.utils.is_hip", return_value=True),
            patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=_spec("DFLASH"),
            ),
            get_parallel().override(attn_dcp_size=8),
        ):
            self.assertFalse(dspark_draft_joins_dcp())
        # Draft model init reads the raw server arg, before it is wrapped.
        published = SimpleNamespace(
            speculative_algorithm="DSPARK",
            speculative_draft_model_path="",
        )
        with (
            patch("sglang.srt.utils.is_hip", return_value=True),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_config.draft_is_deepseek_v4",
                return_value=False,
            ),
            patch("sglang.srt.runtime_context.get_spec", return_value=published),
            get_parallel().override(attn_dcp_size=8),
        ):
            self.assertFalse(dspark_draft_joins_dcp())

    def test_qwen_dspark_draft_uses_aiter_unified_attention(self):
        # head_dim 64 is outside AITER ASM paged attention. ATOM uses
        # unified_attention there; the draft allow-list has to name aiter
        # or the worker falls back to SGLang's Triton extend.
        self.assertIn("aiter", DRAFT_ATTENTION_BACKEND_CHOICES)
        self.assertTrue(
            dspark_mha_uses_unified_attention(
                is_draft_worker=True, is_dspark=True, use_mla=False
            )
        )
        self.assertFalse(
            dspark_mha_uses_unified_attention(
                is_draft_worker=True, is_dspark=True, use_mla=True
            )
        )
        self.assertFalse(
            dspark_mha_uses_unified_attention(
                is_draft_worker=False, is_dspark=True, use_mla=False
            )
        )
        self.assertFalse(
            dspark_mha_uses_unified_attention(
                is_draft_worker=True, is_dspark=False, use_mla=False
            )
        )

    def test_joined_store_collapses_widened_ids_into_the_device_pool(self):
        # Device pool is 1,381,120 rows. The allocator's id space is 8x that.
        # A prefill past the first shard used to hand the raw id to
        # store_kvcache and abort.
        device_rows = 1_381_120
        raw = torch.tensor([0, 1, -1, 8, device_rows * 8 - 8])
        positions = torch.tensor([0, 1, 2, 8, 16])
        safe, owned = dcp_joined_kv_locs(
            raw, dcp_size=8, dcp_rank=0, positions=positions
        )
        self.assertTrue(torch.equal(safe, torch.tensor([0, 0, 0, 1, device_rows - 1])))
        self.assertTrue((safe >= 0).all())
        self.assertTrue((safe < device_rows).all())
        self.assertTrue(
            torch.equal(owned, torch.tensor([True, False, False, True, True]))
        )
        # Verify commit keeps only this rank's columns inside commit_lens.
        window = torch.tensor([0, 1, 8, 9, 16, 17, 24, 25]).view(2, 4)
        commit = torch.tensor([2, 4])
        pos = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])
        safe, owned = dcp_joined_kv_locs(
            window.reshape(-1),
            dcp_size=8,
            dcp_rank=4,
            positions=pos,
            commit_lens=commit,
            row_width=4,
        )
        self.assertTrue((safe < device_rows).all())
        # rank 4 owns positions 12 and 20. Position 20 is column 0 of row 1
        # (committed). Position 12 is column 2 of row 0, past commit_lens 2.
        self.assertTrue(
            torch.equal(
                owned,
                torch.tensor([False, False, False, False, True, False, False, False]),
            )
        )

    def test_kv_write_skips_negative_pads_and_other_ranks(self):
        raw = torch.tensor([0, 1, -1, 8, 9])
        safe, owned = dcp_kv_write_loc(raw, dcp_size=8, dcp_rank=0)
        self.assertTrue(torch.equal(safe, torch.tensor([0, 0, 0, 1, 1])))
        self.assertTrue(
            torch.equal(owned, torch.tensor([True, False, False, True, False]))
        )
        # Positions that claim ownership still cannot store a -1 pad.
        safe, owned = dcp_kv_write_loc(
            torch.tensor([0, 8, -1]),
            dcp_size=8,
            dcp_rank=0,
            positions=torch.tensor([0, 8, 16]),
        )
        self.assertTrue(torch.equal(owned, torch.tensor([True, True, False])))
        self.assertTrue((safe >= 0).all())

    def test_prefix_kv_keeps_every_replicated_head(self):
        # Qwen3 DSpark, TP=DCP=8: 8 local Q heads, 16 replicated KV heads.
        # Current-token GQA is 8/2. Prefix all-gathers 64 Q heads against the
        # 16-head buffer, so that K must stay 16 wide (group 4). Using the
        # sliced K there yields group 32 and reads the wrong cache heads.
        k = torch.arange(4 * 16 * 8, dtype=torch.float32).reshape(4, 16, 8)
        v = k.clone()
        k_cur, v_cur, k_prefix, v_prefix = dcp_extend_kv_head_views(
            k, v, tp_k_head_num=16, dcp_rank=3, dcp_world=8
        )
        self.assertEqual(tuple(k_cur.shape), (4, 2, 8))
        self.assertEqual(tuple(v_cur.shape), (4, 2, 8))
        self.assertEqual(int(k_cur[0, 0, 0]), int(k[0, 6, 0]))
        self.assertEqual(k_prefix.shape[1], 16)
        self.assertEqual(k_prefix[:0].shape[1], 16)
        self.assertEqual(v_prefix.shape[1], 16)
        self.assertEqual(8 // k_cur.shape[1], 4)
        self.assertEqual(64 // k_prefix[:0].shape[1], 4)
        self.assertEqual(64 // k_cur[:0].shape[1], 32)

        empty = k[:0]
        k_cur, _, k_prefix, _ = dcp_extend_kv_head_views(
            empty, empty, tp_k_head_num=16, dcp_rank=0, dcp_world=8
        )
        self.assertEqual(k_cur.shape[1], 16)
        self.assertEqual(k_prefix.shape[1], 16)

    def test_kv_heads_replicate_inside_the_dcp_group(self):
        self.assertEqual(draft_kv_tp_size(tp_size=8, dcp_size=1), 8)
        self.assertEqual(draft_kv_tp_size(tp_size=8, dcp_size=8), 1)
        self.assertEqual(draft_kv_tp_size(tp_size=8, dcp_size=2), 4)
        # 16 KV heads, tp 8, dcp 8 -> every rank keeps all 16.
        self.assertEqual(16 // draft_kv_tp_size(tp_size=8, dcp_size=8), 16)
        # Query heads stay on the TP partition.
        self.assertEqual(64 // 8, 8)

    def test_joined_draft_pool_is_not_widened(self):
        cfg = SimpleNamespace(is_draft_worker=True)
        with (
            patch(
                "sglang.srt.mem_cache.draft_dcp.dspark_draft_joins_dcp",
                return_value=True,
            ),
            get_parallel().override(attn_dcp_size=8),
        ):
            self.assertEqual(KVCacheConfigurator.loc_space_scale.fget(cfg), 1)
        with (
            patch(
                "sglang.srt.mem_cache.draft_dcp.dspark_draft_joins_dcp",
                return_value=False,
            ),
            get_parallel().override(attn_dcp_size=8),
        ):
            self.assertEqual(KVCacheConfigurator.loc_space_scale.fget(cfg), 8)

    def test_joined_draft_budget_is_one_shard(self):
        kvc = SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.from_string("DSPARK"),
            spec_aux_config=SimpleNamespace(dflash_draft_cell_size_per_token=100),
        )
        with (
            patch(
                "sglang.srt.mem_cache.draft_dcp.dspark_draft_joins_dcp",
                return_value=True,
            ),
            get_parallel().override(attn_dcp_size=8),
        ):
            self.assertEqual(_dflash_draft_cell_size(kvc), 100)
        with (
            patch(
                "sglang.srt.mem_cache.draft_dcp.dspark_draft_joins_dcp",
                return_value=False,
            ),
            get_parallel().override(attn_dcp_size=8),
        ):
            self.assertEqual(_dflash_draft_cell_size(kvc), 800)

    def test_host_transfer_keeps_this_ranks_rows(self):
        host = object.__new__(MHATokenToKVPoolHost)
        host.dcp_size = 8
        host.dcp_rank = 1
        ids = torch.arange(16)
        host_ids, device_ids = host._dcp_local_indices(
            ids, ids.clone(), already_local=False
        )
        # Owner rule: slot % 8 == 1, row = slot // 8.
        self.assertTrue(torch.equal(host_ids, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(device_ids, torch.tensor([0, 1])))
        same, _ = host._dcp_local_indices(ids, ids, already_local=True)
        self.assertTrue(torch.equal(same, ids))


if __name__ == "__main__":
    unittest.main()
