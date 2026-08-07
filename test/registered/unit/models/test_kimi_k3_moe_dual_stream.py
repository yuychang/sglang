"""K3 plain-TP MoE dual-stream tail: the single-collective branch of
_forward_fused must produce the same result whether the shared experts run on
the side stream or ahead of the routed branch, the fork must engage only on the
shapes and graph modes it is safe for, and it must be issued early enough that
the routed-only repacking falls inside the overlap."""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
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


def _make_moe(alt_stream, device="cuda", needs_contiguous=False):
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
    moe._moe_front_needs_contiguous = needs_contiguous
    moe._defer_moe_finalize = False
    moe.fuse_ar_norm = False
    moe._gemm_ag_up_eligible = False
    return moe


@contextmanager
def _plain_tp_env(gemm=_gemm):
    """Neutralize everything _forward_fused reaches outside the branch under
    test: no AR fusion (the plain-TP shape this branch serves), no
    symmetric-memory pool, and a single-rank all-reduce."""
    with ExitStack() as stack:
        for p in (
            patch("sglang.srt.models.kimi_k3._k3_bf16_gemm", gemm),
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


class _FakeStream:
    """Stand-in for torch.cuda.Stream that records the host-side fork, so the
    ordering tests below need no GPU."""

    def __init__(self, log, name):
        self._log = log
        self._name = name

    def wait_stream(self, other):
        self._log.append(f"{self._name}.wait_stream")


def _repack_logging_gemm(log):
    """_k3_bf16_gemm stand-in whose front-GEMM result records .contiguous().

    The front GEMM's output is the only tensor _forward_fused repacks, and
    torch keeps the subclass across torch.split, so the split views report the
    copies. contiguous() hands back a plain tensor: the repacked halves flow
    into out= GEMMs, where a subclass buys nothing and only widens what the
    test depends on."""

    class _Logged(torch.Tensor):
        def contiguous(self, *args, **kwargs):
            log.append("repack")
            return torch.Tensor.contiguous(self, *args, **kwargs).as_subclass(
                torch.Tensor
            )

    def _wrapped(x, weight, out=None):
        result = _gemm(x, weight, out)
        # out= is the shared-expert down GEMM writing into the collective
        # buffer, not the front; leave it alone.
        return result if out is not None else result.as_subclass(_Logged)

    return _wrapped


class TestKimiK3MoeForkOrder(CustomTestCase):
    """The fork has to be issued before the routed-only repacking.

    wait_stream orders the side stream against the main stream's tail *at the
    call*, so anything enqueued before it is work the shared experts wait on.
    Only the trtllm-gen runner takes the strided split views as they are — on
    ROCm/aiter and on marlin every multi-token MoE layer copies them out — so a
    fork placed after those copies hands the side stream a needless dependency.
    """

    _TOKENS = 4  # > 1, which is what arms the repacking

    def _input(self):
        return (
            torch.randn(self._TOKENS, _H, dtype=torch.float32)
            .mul(0.05)
            .to(torch.bfloat16)
        )

    def _run(self, *, alt_stream, log, x=None):
        moe = _make_moe(alt_stream, device="cpu", needs_contiguous=True)
        x = self._input() if x is None else x
        main = _FakeStream(log, "main")
        with _plain_tp_env(gemm=_repack_logging_gemm(log)), patch(
            "torch.cuda.current_stream", return_value=main
        ), patch("torch.cuda.stream", lambda _stream: nullcontext()), patch(
            # The HIP shape repacks the router logits too, so both copies are
            # covered.
            "sglang.srt.models.kimi_k3._is_hip",
            True,
        ), patch(
            "sglang.srt.models.kimi_k3._aiter_k3_opt", False
        ):
            return KimiK3MoE._forward_fused(moe, x, prefix_sum=None)

    def test_fork_precedes_the_repacking(self):
        log = []
        self._run(alt_stream=_FakeStream(log, "alt"), log=log)
        self.assertEqual(
            log.count("repack"),
            2,  # router logits + routed input
            f"expected both split views to be repacked, got {log}",
        )
        self.assertEqual(log[0], "alt.wait_stream", f"fork is not first: {log}")

    def test_no_fork_without_a_side_stream(self):
        log = []
        self._run(alt_stream=None, log=log)
        self.assertEqual(log, ["repack", "repack"])

    def test_reordering_leaves_the_result_alone(self):
        x = self._input()
        serial_log, overlap_log = [], []
        # Both owners seed the same generator, so the weights match too.
        serial = self._run(alt_stream=None, log=serial_log, x=x)
        overlap = self._run(
            alt_stream=_FakeStream(overlap_log, "alt"), log=overlap_log, x=x
        )
        self.assertTrue(torch.equal(overlap, serial))


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
        # needs_contiguous is the runner-dependent repacking the fork now spans
        # (True for every runner but trtllm-gen), so both settings are covered.
        for tokens in (1, 4, 32):
            for needs_contiguous in (False, True):
                with self.subTest(tokens=tokens, needs_contiguous=needs_contiguous):
                    x = self._input(tokens)
                    # Both owners seed the same generator, so weights match.
                    serial = self._run(
                        _make_moe(alt_stream=None, needs_contiguous=needs_contiguous), x
                    )
                    overlap = self._run(
                        _make_moe(
                            alt_stream=torch.cuda.Stream(),
                            needs_contiguous=needs_contiguous,
                        ),
                        x,
                    )
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
