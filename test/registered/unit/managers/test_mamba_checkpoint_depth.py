"""The donated mamba checkpoint depth must land on the tree page.

DCP widens the tree page past the mamba chunk grid. A checkpoint picked on the
finer grid names a depth no radix node can carry, so it gets attached to the
preceding node and a later request resumes from a state that already covers
tokens past that node. The same happens when the depth is measured from a
prefix that a chunked prefill left on the scheduler page but off the widened
tree page: 64 + 512 = 576 is no 512-page boundary, and the finished request
then frees two kv-row segments that share the page holding 576.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder, mamba_prefill_chunk_grid
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, mamba_track_grid
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CHUNK = 64


def _track_seqlen(
    *,
    tree_page: int,
    prefix_len: int,
    extend_len: int,
    dcp_enabled: bool = False,
) -> int | None:
    """Run one extend through the tracker and report the donated depth, or
    None when the extend donates no checkpoint."""
    server_args = ServerArgs(model_path="dummy", page_size=CHUNK)
    # The property would otherwise load the HF config for the dummy model.
    server_args._mamba_cache_chunk_size = CHUNK
    set_global_server_args_for_scheduler(server_args)

    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1] * (prefix_len + extend_len)),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.prefix_len = prefix_len
    req.extend_end = prefix_len + extend_len
    req.kv.mamba_ping_pong_track_buffer = torch.tensor([0, 1], dtype=torch.int64)
    req.kv.mamba_next_track_idx = 0
    req.mamba_branching_seqlen = None

    batch = ScheduleBatch(reqs=[req])
    batch.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(mamba_chunk_size=CHUNK)
    )
    batch.tree_cache = SimpleNamespace(page_size=tree_page)
    batch.req_to_token_pool = MagicMock()
    batch.req_to_token_pool.get_mamba_ping_pong_other_idx.return_value = 1

    with patch(
        "sglang.srt.managers.schedule_batch.get_parallel",
        return_value=SimpleNamespace(dcp_enabled=dcp_enabled),
    ):
        entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
    if not entry.track_mask:
        assert req.kv.mamba_last_track_seqlen is None
        return None
    return req.kv.mamba_last_track_seqlen


class TestMambaCheckpointDepth(unittest.TestCase):
    def test_widened_tree_page_moves_the_donated_depth_onto_it(self):
        # 4066 tokens past a 16384 prefix: the chunk grid would stop at 20416,
        # which a 256-token page cannot name.
        depth = _track_seqlen(
            tree_page=256,
            prefix_len=16384,
            extend_len=4066,
            dcp_enabled=True,
        )
        self.assertEqual(depth % 256, 0)
        self.assertEqual(depth, 20224)

    def test_unwidened_tree_page_keeps_the_chunk_grid(self):
        depth = _track_seqlen(tree_page=CHUNK, prefix_len=16384, extend_len=4066)
        self.assertEqual(depth, 20416)

    def test_dcp_depth_satisfies_tree_page_and_snapshot_grids(self):
        # (tree_page, prefix_len, extend_len) -> donated depth, or None when no
        # prefix-relative kernel snapshot lands on an absolute tree page.
        for tree_page, prefix_len, extend_len, expected in (
            (512, 64, 540, 512),
            (512, 448, 128, 512),
            (512, 64, 400, None),
            (512, 37, 540, None),
        ):
            with self.subTest(prefix_len=prefix_len, extend_len=extend_len):
                self.assertEqual(
                    _track_seqlen(
                        tree_page=tree_page,
                        prefix_len=prefix_len,
                        extend_len=extend_len,
                        dcp_enabled=True,
                    ),
                    expected,
                )

    def test_non_dcp_depth_remains_relative_to_chunked_prefill_prefix(self):
        # Active requests can stop off the absolute grid between prefill chunks.
        # Keep donating their prefix-relative snapshots when DCP is disabled.
        self.assertEqual(
            _track_seqlen(tree_page=16, prefix_len=4253, extend_len=16384),
            20637,
        )


class TestChunkedPrefillStaysOnTheChunkGrid(unittest.TestCase):
    def _floor(self, trunc_len: int, remaining: int, grid: int = CHUNK) -> int:
        adder = object.__new__(PrefillAdder)
        adder.mamba_chunk_grid = grid
        return adder.floor_chunked_tokens(trunc_len, remaining)

    def test_cut_is_rounded_down_onto_the_grid(self):
        self.assertEqual(self._floor(100, 8192), 64)
        self.assertEqual(self._floor(8010, 100000), 8000)

    def test_aligned_cut_is_untouched(self):
        self.assertEqual(self._floor(8192, 100000), 8192)

    def test_final_chunk_may_stop_anywhere(self):
        self.assertEqual(self._floor(37, 37), 37)

    def test_cut_shorter_than_grid_still_makes_progress(self):
        self.assertEqual(self._floor(40, 8192), 40)

    def _published_grid(self, *, dcp_size: int) -> int:
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            dcp_size=dcp_size,
            _mamba_cache_chunk_size=CHUNK,
        ):
            return mamba_prefill_chunk_grid()

    def test_dcp_extra_buffer_uses_the_chunk_grid(self):
        self.assertEqual(self._published_grid(dcp_size=2), CHUNK)

    def test_non_dcp_extra_buffer_leaves_the_cut(self):
        self.assertEqual(self._published_grid(dcp_size=1), 1)

    def test_non_dcp_does_not_change_truncation_alignment(self):
        scheduler = object.__new__(Scheduler)
        scheduler.truncation_align_size = None
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            dcp_size=1,
            _mamba_cache_chunk_size=CHUNK,
        ):
            scheduler.init_mamba_truncation_align()
        self.assertIsNone(scheduler.truncation_align_size)

    def test_dcp_folds_chunk_into_truncation_alignment(self):
        scheduler = object.__new__(Scheduler)
        scheduler.truncation_align_size = 48
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            dcp_size=2,
            _mamba_cache_chunk_size=CHUNK,
        ):
            scheduler.init_mamba_truncation_align()
        self.assertEqual(scheduler.truncation_align_size, 192)

    def test_each_cut_up_chunk_can_donate(self):
        prefix_len, total = 0, 100_000
        for budget in (8192, 5000, 8192, 3333, 8192, 777):
            trunc_len = self._floor(budget, total - prefix_len)
            self.assertIsNotNone(
                _track_seqlen(
                    tree_page=256,
                    prefix_len=prefix_len,
                    extend_len=trunc_len,
                    dcp_enabled=True,
                ),
                f"no donation at {prefix_len=} {trunc_len=}",
            )
            prefix_len += trunc_len


class TestMambaTrackGrid(unittest.TestCase):
    def _grid(self, *, interval: int, tree_page: int, chunk: int = CHUNK) -> int:
        with get_context().override_server_args(
            mamba_track_interval=interval, _mamba_cache_chunk_size=chunk
        ):
            return mamba_track_grid(tree_page)

    def test_widened_tree_page_rounds_the_interval_up(self):
        self.assertEqual(self._grid(interval=256, tree_page=512), 512)

    def test_interval_already_on_the_tree_page_is_untouched(self):
        self.assertEqual(self._grid(interval=256, tree_page=256), 256)
        self.assertEqual(self._grid(interval=256, tree_page=128), 256)
        self.assertEqual(self._grid(interval=256, tree_page=64), 256)

    def test_grid_stays_on_the_chunk_size(self):
        self.assertEqual(self._grid(interval=192, tree_page=64, chunk=128) % 128, 0)


if __name__ == "__main__":
    unittest.main()
