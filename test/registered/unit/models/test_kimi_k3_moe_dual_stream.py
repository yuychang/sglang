"""K3 plain-TP MoE dual-stream tail: the single-collective branch of
_forward_fused must produce the same result whether the shared experts run on
the side stream or ahead of the routed branch, and the fork must engage only on
the shapes and graph modes it is safe for."""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models.kimi_k3 import KimiK3MoE, _build_alt_streams
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

_H = 256  # hidden_size
_LATENT = 128  # routed_expert_hidden_size (moe_hidden_size)
_INTER = 64  # shared-expert intermediate, per rank


def _gemm(x, weight, out=None):
    """Stand-in for _k3_bf16_gemm: the torch fallback the real one lands on
    when the cutedsl bf16 backend is not selected, so the test needs no
    runtime context."""
    if out is None:
        return torch.nn.functional.linear(x, weight)
    return torch.mm(x, weight.t(), out=out)


@contextmanager
def _noop_symmetric_memory(*args, **kwargs):
    yield


class _Linear:
    """ReplicatedLinear stand-in returning the (output, bias) pair the model
    unpacks."""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _Moe:
    """Minimal owner carrying the real methods under test."""

    _forward_fused = KimiK3MoE._forward_fused
    _dual_stream_single_collective = KimiK3MoE._dual_stream_single_collective
    _forward_shared = KimiK3MoE._forward_shared
    _latent_norm = KimiK3MoE._latent_norm


def _make_moe(alt_stream, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(0)

    def _randn(*shape):
        return (
            torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)
            .mul(0.05)
            .to(torch.bfloat16)
        )

    moe = _Moe()
    moe.alt_stream = alt_stream
    moe.moe_hidden_size = _LATENT
    # Merged front weight: [shared gate_up | router | latent down].
    moe._front_sizes = [2 * _INTER, 8, _LATENT]
    moe._front_w = _randn(sum(moe._front_sizes), _H).contiguous()

    moe.shared_experts = SimpleNamespace(
        # A plain gated activation stands in for SituAndMul: what matters is
        # that real work lands on the side stream, not which nonlinearity.
        act_fn=lambda t: torch.nn.functional.silu(t[..., :_INTER]) * t[..., _INTER:],
        down_proj=SimpleNamespace(weight=_randn(_H, _INTER).contiguous()),
    )

    # The routed branch is stubbed to a deterministic function of the routed
    # input: the real one is top-k plus the MoE runner, and the fork/join
    # topology does not depend on either.
    routed_w = _randn(_LATENT, _LATENT).contiguous()

    def _forward_routed(hidden_states, router_logits, routed_input, latent):
        torch.mm(routed_input, routed_w.t(), out=latent)

    moe._forward_routed = _forward_routed
    moe.routed_expert_up_proj = _Linear(_randn(_H, _LATENT).contiguous())
    moe.routed_expert_norm = None
    moe._moe_front_needs_contiguous = False
    moe._defer_moe_finalize = False
    moe.fuse_ar_norm = False
    moe._gemm_ag_up_eligible = False
    return moe


@contextmanager
def _plain_tp_env():
    """Neutralize everything _forward_fused reaches outside the branch under
    test: no AR fusion (the plain-TP shape this branch serves), no
    symmetric-memory pool, and a single-rank all-reduce."""
    with ExitStack() as stack:
        for p in (
            patch("sglang.srt.models.kimi_k3._k3_bf16_gemm", _gemm),
            patch("sglang.srt.models.kimi_k3.k3_ar_fusion.enabled", return_value=False),
            patch(
                "sglang.srt.models.kimi_k3.use_symmetric_memory", _noop_symmetric_memory
            ),
            patch(
                "sglang.srt.models.kimi_k3.is_allocation_symmetric", return_value=False
            ),
            patch("sglang.srt.models.kimi_k3.get_tp_group", return_value=None),
            patch(
                "sglang.srt.models.kimi_k3.tensor_model_parallel_all_reduce",
                side_effect=lambda t: t,
            ),
        ):
            stack.enter_context(p)
        yield


class TestKimiK3MoeDualStreamGate(CustomTestCase):
    """Eligibility only — no device work, so this runs anywhere."""

    def setUp(self):
        # The predicate only tests the slot for None, so a sentinel stands in
        # for a stream and keeps these cases off the GPU.
        self.moe = SimpleNamespace(alt_stream=object())
        self.gate = lambda n: KimiK3MoE._dual_stream_single_collective(self.moe, n)
        self.no_ar_fusion = patch(
            "sglang.srt.models.kimi_k3.k3_ar_fusion.enabled", return_value=False
        )

    def test_requires_alt_stream(self):
        self.moe.alt_stream = None
        with self.no_ar_fusion:
            self.assertFalse(self.gate(4))

    def test_token_ceiling(self):
        with self.no_ar_fusion:
            with envs.SGLANG_K3_DUAL_STREAM_MOE_MAX_TOKENS.override(16):
                self.assertTrue(self.gate(16))
                self.assertFalse(self.gate(17))
                self.assertFalse(self.gate(0))  # nothing to overlap

    def test_zero_ceiling_is_the_kill_switch(self):
        with self.no_ar_fusion:
            with envs.SGLANG_K3_DUAL_STREAM_MOE_MAX_TOKENS.override(0):
                self.assertFalse(self.gate(1))

    def test_skips_segmented_capture(self):
        """Per-segment capture would split the fork from its join."""
        for name in ("is_in_breakable_cuda_graph", "is_in_tc_piecewise_cuda_graph"):
            with self.subTest(mode=name):
                with self.no_ar_fusion, patch(
                    f"sglang.srt.models.kimi_k3.{name}", return_value=True
                ):
                    self.assertFalse(self.gate(4))

    def test_skips_ar_fusion_path(self):
        """The AR-fusion branch runs its own dual-stream tail."""
        with patch("sglang.srt.models.kimi_k3.k3_ar_fusion.enabled", return_value=True):
            self.assertFalse(self.gate(4))


class TestKimiK3AltStreamPool(CustomTestCase):
    """Pool construction only — Stream() is stubbed, so this runs anywhere."""

    @contextmanager
    def _pool(self, *, hip):
        streams = []

        def _new_stream():
            streams.append(SimpleNamespace(id=len(streams)))
            return streams[-1]

        with patch("sglang.srt.models.kimi_k3._is_hip", hip), patch(
            "torch.cuda.Stream", _new_stream
        ):
            yield lambda: (_build_alt_streams(), streams)

    def test_cuda_gets_one_stream_per_slot(self):
        with self._pool(hip=False) as build:
            pool, streams = build()
            self.assertEqual(len(pool), 3)
            self.assertEqual(len({id(s) for s in pool}), 3)
            self.assertEqual(len(streams), 3)

    def test_hip_is_opt_in(self):
        with self._pool(hip=True) as build:
            with envs.SGLANG_ROCM_USE_MULTI_STREAM.override(False):
                pool, streams = build()
            self.assertIsNone(pool)
            self.assertEqual(streams, [])

    def test_hip_folds_moe_and_mla_onto_one_stream(self):
        """Two physical streams cover the three slots: [0] and [1] never
        overlap each other, so they share, while [2] must stay separate."""
        with self._pool(hip=True) as build:
            with envs.SGLANG_ROCM_USE_MULTI_STREAM.override(True):
                with temp_set_env(GPU_MAX_HW_QUEUES="5"):  # keeps the warning quiet
                    pool, streams = build()
        self.assertEqual(len(streams), 2)
        self.assertEqual(len(pool), 3)
        self.assertIs(pool[0], pool[1])
        self.assertIsNot(pool[0], pool[2])

    def test_hip_warns_on_insufficient_hw_queues(self):
        with self._pool(hip=True) as build:
            with envs.SGLANG_ROCM_USE_MULTI_STREAM.override(True):
                with temp_set_env(GPU_MAX_HW_QUEUES="4"):
                    with self.assertLogs(
                        "sglang.srt.models.kimi_k3", level="WARNING"
                    ) as logs:
                        build()
                self.assertIn("GPU_MAX_HW_QUEUES", logs.output[0])

                with temp_set_env(GPU_MAX_HW_QUEUES="5"):
                    with patch("sglang.srt.models.kimi_k3.logger.warning") as warn:
                        build()
                    warn.assert_not_called()


class TestKimiK3MoeDualStreamEquivalence(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def _run(self, moe, x):
        with _plain_tp_env():
            return KimiK3MoE._forward_fused(moe, x, prefix_sum=None).clone()

    def _input(self, tokens):
        return (
            torch.randn(tokens, _H, device="cuda", dtype=torch.float32)
            .mul(0.05)
            .to(torch.bfloat16)
        )

    def test_dual_stream_matches_serial(self):
        for tokens in (1, 4, 32):
            with self.subTest(tokens=tokens):
                x = self._input(tokens)
                # Both owners seed the same generator, so the weights match.
                serial = self._run(_make_moe(alt_stream=None), x)
                overlap = self._run(_make_moe(alt_stream=torch.cuda.Stream()), x)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(overlap, serial))

    def test_dual_stream_matches_serial_under_graph_capture(self):
        x = self._input(4)
        serial = self._run(_make_moe(alt_stream=None), x)

        moe = _make_moe(alt_stream=torch.cuda.Stream())
        self._run(moe, x)  # warm up allocations outside capture
        graph = torch.cuda.CUDAGraph()
        with _plain_tp_env():
            with torch.cuda.graph(graph):
                captured = KimiK3MoE._forward_fused(moe, x, prefix_sum=None)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(captured, serial))


if __name__ == "__main__":
    unittest.main()
