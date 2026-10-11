"""CPU unit test for the DCP prefix-KV all-gather in ``layers/dcp/comm.py``.

Simulates every rank of a DCP group: each rank holds the prefix tokens whose
absolute position p satisfies ``p % dcp_size == rank``, and the mocked
all-gather supplies the other ranks' padded buffers from that ownership rule.
Both the original and the ``SGLANG_ROCM_K3_DCP_FAST_KV_GATHER`` implementations must
return every request's prefix in token order.

Usage:
    python -m pytest test_dcp_kv_gather_unit.py -v
"""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.dcp.comm as comm
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

KV_A_DIM, K_PE_DIM = 4, 2


def _value(req: int, pos: int) -> float:
    return float(req * 100000 + pos)


def _rows(values) -> torch.Tensor:
    t = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)
    return torch.cat([t.expand(-1, 1, KV_A_DIM), (-t).expand(-1, 1, K_PE_DIM)], -1)


def _local_kv(lens, starts, world, rank):
    rows = _rows(
        [
            _value(i, p)
            for i, (n, s) in enumerate(zip(lens, starts))
            for p in range(s, s + n)
            if p % world == rank
        ]
    )
    return rows[..., :KV_A_DIM].contiguous(), rows[..., KV_A_DIM:].contiguous()


def _padded_for_rank(lens, starts, world, rank):
    values = []
    for i, (n, s) in enumerate(zip(lens, starts)):
        off = s % world
        for j in range((n + off + world - 1) // world):
            p = s - off + j * world + rank
            values.append(_value(i, p) if s <= p < s + n else float("nan"))
    return _rows(values)


def _expected(lens, starts):
    return _rows(
        [
            _value(i, p)
            for i, (n, s) in enumerate(zip(lens, starts))
            for p in range(s, s + n)
        ]
    )


class _MockGroup:
    def __init__(self, lens, starts, world, rank):
        self.lens, self.starts, self.world, self.rank = lens, starts, world, rank

    def all_gather_into_tensor(self, out, inp):
        parts = [
            (
                inp
                if r == self.rank
                else _padded_for_rank(self.lens, self.starts, self.world, r)
            )
            for r in range(self.world)
        ]
        for part in parts:
            assert part.shape == inp.shape
        out.copy_(torch.cat(parts, 0))


class TestDcpKvGather(CustomTestCase):
    def _cases(self):
        rng = random.Random(0)
        cases = [([3544], [7]), ([1], [5]), ([0], [3]), ([5, 0, 17], [0, 3, 9])]
        for _ in range(30):
            bs = rng.randint(1, 6)
            cases.append(
                (
                    [rng.randint(0, 300) for _ in range(bs)],
                    [rng.randint(0, 50) for _ in range(bs)],
                )
            )
        return cases

    def _check(self, lens, starts, world, rank, fast, pass_starts):
        parallel = SimpleNamespace(
            dcp_enabled=True,
            dcp_size=world,
            dcp_rank=rank,
            dcp_group=_MockGroup(lens, starts, world, rank),
        )
        kv_a, k_pe = _local_kv(lens, starts, world, rank)
        lens_t = torch.tensor(lens)
        starts_t = torch.tensor(starts) if pass_starts else None
        expected = _expected(lens, starts)
        with (
            patch.object(comm, "get_parallel", return_value=parallel),
            envs.SGLANG_ROCM_K3_DCP_FAST_KV_GATHER.override(fast),
        ):
            full = comm.all_gather_kv_cache_for_dcp(kv_a, k_pe, lens_t, starts_t)
            out_a, out_pe = comm.all_gather_kv_cache_for_mha_chunk_extend(
                kv_a[:, 0], k_pe, lens_t, starts_t
            )
        msg = f"{lens=} {starts=} {world=} {rank=} {fast=}"
        self.assertTrue(torch.equal(full, expected), msg)
        self.assertTrue(torch.equal(out_a, expected[:, 0, :KV_A_DIM]), msg)
        self.assertTrue(torch.equal(out_pe, expected[..., KV_A_DIM:]), msg)
        self.assertTrue(out_a.is_contiguous() and out_pe.is_contiguous(), msg)

    def test_matches_token_order(self):
        for lens, starts in self._cases():
            for world in (1, 2, 4, 8):
                for rank in range(world):
                    for fast in (False, True):
                        self._check(lens, starts, world, rank, fast, True)
                        self._check(lens, [0] * len(lens), world, rank, fast, False)


if __name__ == "__main__":
    unittest.main()
