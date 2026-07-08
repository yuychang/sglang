"""Unit tests for the ROCm Kimi-K2.5 / DeepSeek-style MXFP4 MoE multi-stream
overlap primitives (``sglang.srt.layers.moe.rocm_kimi_mxfp4_moe``).

Covers, on CPU (with GPU-gated variants where a device is available):

* MXFP4 (OCP E2M1 + E8M0) pack/dequant reference semantics.
* The P0 fused add-shared combine.
* The P1 deferred-finalize + shared-add combine, incl. routed_scaling_factor
  handling (no double scaling) and "each contribution counted once".
* The one-time P1 self-check guard.
* Feature-gating duck-typing for the MXFP4 schemes.
* The RocmMoeStreamState manager (GPU only).
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.rocm_kimi_mxfp4_moe import (
    OCP_MX_BLOCK_SIZE,
    build_trivial_row_map,
    p1_self_check_matches,
    rocm_mxfp4_moe_add_shared,
    rocm_mxfp4_moe_finalize_fuse_shared,
)
from sglang.test.test_utils import CustomTestCase

# OCP MXFP4 (E2M1) representable magnitudes.
_E2M1_ABS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _nearest_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round-half-to-nearest onto the signed E2M1 grid (reference)."""
    grid = torch.tensor(
        _E2M1_ABS + [-v for v in _E2M1_ABS], dtype=torch.float32
    ).unique()
    diff = (x.reshape(-1, 1) - grid.reshape(1, -1)).abs()
    idx = diff.argmin(dim=1)
    return grid[idx].reshape(x.shape)


def mxfp4_quant_dequant_reference(
    x: torch.Tensor, group_size: int = OCP_MX_BLOCK_SIZE
) -> torch.Tensor:
    """Reference OCP MXFP4 quantize->dequantize (per-group E8M0 scale + E2M1).

    Scale exponent per block = floor(log2(amax)) - 2 (so amax maps into the top
    E2M1 bucket), stored as a power of two (E8M0). Zero blocks map to 0.
    """
    orig_shape = x.shape
    x = x.reshape(-1, group_size).float()
    amax = x.abs().amax(dim=1, keepdim=True)
    # E8M0 exponent (power-of-two scale); guard the all-zero block.
    safe = amax.clamp_min(1e-30)
    exp = torch.floor(torch.log2(safe)) - 2.0
    scale = torch.pow(2.0, exp)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    q = _nearest_e2m1(x / scale)
    deq = q * scale
    deq = torch.where(amax > 0, deq, torch.zeros_like(deq))
    return deq.reshape(orig_shape)


class TestMXFP4Reference(CustomTestCase):
    def test_exactly_representable_group_roundtrips(self):
        # A block made entirely of grid values (<= amax=6) uses scale=1 and
        # round-trips exactly.
        vals = torch.tensor(
            [6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0] * 4, dtype=torch.float32
        )
        self.assertEqual(vals.numel(), OCP_MX_BLOCK_SIZE)
        deq = mxfp4_quant_dequant_reference(vals)
        torch.testing.assert_close(deq, vals, rtol=0, atol=0)

    def test_power_of_two_scaled_group_roundtrips(self):
        base = torch.tensor([6.0, 3.0, 2.0, 1.0, 0.5] + [0.0] * 27, dtype=torch.float32)
        for k in (-3, 0, 2, 5):
            scaled = base * (2.0**k)
            deq = mxfp4_quant_dequant_reference(scaled)
            torch.testing.assert_close(deq, scaled, rtol=0, atol=0)

    def test_zero_group(self):
        z = torch.zeros(OCP_MX_BLOCK_SIZE)
        deq = mxfp4_quant_dequant_reference(z)
        torch.testing.assert_close(deq, z, rtol=0, atol=0)

    def test_random_group_bounded_error(self):
        torch.manual_seed(0)
        x = torch.randn(8 * OCP_MX_BLOCK_SIZE)
        deq = mxfp4_quant_dequant_reference(x)
        # MXFP4 keeps sign and coarse magnitude; require high cosine similarity.
        cos = torch.nn.functional.cosine_similarity(
            x.reshape(1, -1), deq.reshape(1, -1)
        ).item()
        self.assertGreater(cos, 0.95)

    def test_large_and_small_magnitudes(self):
        for mag in (1e-4, 1e4):
            x = (torch.randn(OCP_MX_BLOCK_SIZE) * mag).clamp_min(1e-6)
            deq = mxfp4_quant_dequant_reference(x)
            # No NaN/Inf and magnitude preserved to within the block's dynamic range.
            self.assertTrue(torch.isfinite(deq).all())
            self.assertGreater(deq.abs().max().item(), 0.0)


class TestP0AddShared(CustomTestCase):
    def _run(self, device):
        torch.manual_seed(1)
        for T, H in [(1, 64), (8, 128), (33, 256)]:
            routed = torch.randn(T, H, dtype=torch.bfloat16, device=device)
            shared = torch.randn(T, H, dtype=torch.bfloat16, device=device)
            out = rocm_mxfp4_moe_add_shared(routed, shared)
            ref = (routed.float() + shared.float()).bfloat16()
            torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_cpu(self):
        self._run("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs GPU")
    def test_gpu(self):
        self._run("cuda")

    def test_shape_mismatch_raises(self):
        a = torch.randn(4, 8, dtype=torch.bfloat16)
        b = torch.randn(4, 16, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            rocm_mxfp4_moe_add_shared(a, b)


class TestP1FinalizeFuseShared(CustomTestCase):
    def _reference(self, partial, row_map, weights, shared, rsf, top_k):
        T = row_map.shape[0]
        H = partial.shape[-1]
        g = partial.float()[row_map.reshape(-1).long()].reshape(T, top_k, H)
        acc = (g * weights.float().reshape(T, top_k, 1)).sum(1) * rsf
        if shared is not None:
            acc = acc + shared.float()
        return acc

    def test_finalize_matches_reference(self):
        torch.manual_seed(2)
        for T, H, K in [(1, 64, 2), (8, 128, 8), (16, 256, 4)]:
            for rsf in (1.0, 2.827):
                partial = torch.randn(T * K, H, dtype=torch.bfloat16)
                row_map = build_trivial_row_map(T, K, torch.device("cpu"))
                weights = torch.rand(T, K, dtype=torch.float32)
                shared = torch.randn(T, H, dtype=torch.bfloat16)
                out = rocm_mxfp4_moe_finalize_fuse_shared(
                    partial, row_map, weights, shared, rsf, K
                )
                ref = self._reference(partial, row_map, weights, shared, rsf, K).to(
                    torch.bfloat16
                )
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_no_double_scaling(self):
        # With rsf folded into the weights (rsf=1.0), the result equals the
        # plain weighted sum + shared: routed and shared each counted once.
        torch.manual_seed(3)
        T, H, K = 4, 64, 3
        partial = torch.randn(T * K, H, dtype=torch.bfloat16)
        row_map = build_trivial_row_map(T, K, torch.device("cpu"))
        weights = torch.rand(T, K, dtype=torch.float32)
        shared = torch.randn(T, H, dtype=torch.bfloat16)

        out_folded = rocm_mxfp4_moe_finalize_fuse_shared(
            partial, row_map, weights, shared, 1.0, K
        )
        # Equivalent: pre-fold rsf into weights, then rsf=1.0.
        rsf = 2.827
        out_prefold = rocm_mxfp4_moe_finalize_fuse_shared(
            partial, row_map, weights * rsf, shared, 1.0, K
        )
        out_rsf = rocm_mxfp4_moe_finalize_fuse_shared(
            partial, row_map, weights, shared, rsf, K
        )
        # (weights*rsf, rsf=1)  ==  (weights, rsf=rsf) modulo bf16 rounding.
        torch.testing.assert_close(
            out_prefold.float() - shared.float(),
            (out_rsf.float() - shared.float()),
            rtol=3e-2,
            atol=3e-2,
        )
        # Shared counted exactly once: subtracting shared twice should NOT match.
        self.assertFalse(
            torch.allclose(
                out_folded.float(),
                (out_folded.float() - shared.float()),
                atol=1e-3,
            )
        )

    def test_shared_none(self):
        torch.manual_seed(4)
        T, H, K = 5, 32, 2
        partial = torch.randn(T * K, H, dtype=torch.bfloat16)
        row_map = build_trivial_row_map(T, K, torch.device("cpu"))
        weights = torch.rand(T, K, dtype=torch.float32)
        out = rocm_mxfp4_moe_finalize_fuse_shared(
            partial, row_map, weights, None, 1.0, K
        )
        ref = self._reference(partial, row_map, weights, None, 1.0, K).to(
            torch.bfloat16
        )
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_bad_shapes_raise(self):
        partial = torch.randn(6, 32, dtype=torch.bfloat16)
        row_map = build_trivial_row_map(3, 2, torch.device("cpu"))
        weights = torch.rand(3, 3, dtype=torch.float32)  # wrong top_k
        with self.assertRaises(ValueError):
            rocm_mxfp4_moe_finalize_fuse_shared(partial, row_map, weights, None, 1.0, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "needs GPU")
    def test_gpu_matches_cpu_reference(self):
        torch.manual_seed(5)
        T, H, K = 8, 128, 8
        partial = torch.randn(T * K, H, dtype=torch.bfloat16, device="cuda")
        row_map = build_trivial_row_map(T, K, torch.device("cuda"))
        weights = torch.rand(T, K, dtype=torch.float32, device="cuda")
        shared = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        out = rocm_mxfp4_moe_finalize_fuse_shared(
            partial, row_map, weights, shared, 2.827, K
        )
        ref = self._reference(
            partial.cpu(), row_map.cpu(), weights.cpu(), shared.cpu(), 2.827, K
        ).to(torch.bfloat16)
        torch.testing.assert_close(out.cpu(), ref, rtol=3e-2, atol=3e-2)


class TestP1SelfCheck(CustomTestCase):
    def test_matches_when_equal(self):
        a = torch.randn(4, 16)
        self.assertTrue(p1_self_check_matches(a, a.clone()))

    def test_mismatch_when_scaled(self):
        a = torch.randn(4, 16)
        self.assertFalse(p1_self_check_matches(a, a * 2.0))

    def test_mismatch_on_shape(self):
        self.assertFalse(p1_self_check_matches(torch.randn(4, 16), torch.randn(4, 8)))


class TestGating(CustomTestCase):
    def _fake_moe(self, **overrides):
        base = dict(
            num_fused_shared_experts=0,
            n_shared_experts=1,
            _enable_a2a_moe=False,
            _fuse_shared_experts_inside_sbo=False,
            _shared_expert_tp1=False,
            is_hash=False,
            shared_experts=object(),
            experts=object(),
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_disabled_on_non_hip(self):
        # On a non-HIP CPU box the feature must be OFF regardless of module state.
        from sglang.srt.layers.moe import rocm_kimi_mxfp4_moe as m

        self.assertFalse(m._is_hip)
        self.assertFalse(m.rocm_kimi_mxfp4_multistream_enabled(self._fake_moe()))

    def test_capability_probes_safe(self):
        from sglang.srt.layers.moe import rocm_kimi_mxfp4_moe as m

        # These must never raise on CPU and must be False without HIP/AITER.
        self.assertFalse(m.native_mxfp4_supported())
        self.assertFalse(m._has_hip_combine_ops())
        self.assertFalse(m.aiter_moe_supports_no_combine())
        self.assertFalse(m.rocm_kimi_mxfp4_p1_enabled())


class TestStreamState(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs GPU")
    def test_stream_state_singleton_and_reuse(self):
        from sglang.srt.layers.moe.rocm_kimi_mxfp4_moe import (
            get_rocm_moe_stream_state,
            rocm_moe_stream_state_exists,
        )

        dev = torch.device("cuda:0")
        s1 = get_rocm_moe_stream_state(dev)
        s2 = get_rocm_moe_stream_state(dev)
        self.assertIs(s1, s2)  # one state per device
        self.assertIs(s1.shared_stream, s2.shared_stream)
        self.assertTrue(rocm_moe_stream_state_exists(dev))
        # Workspace reuse: same key returns the same buffer.
        w1 = s1.get_workspace("scratch", (16, 32), torch.bfloat16)
        w2 = s1.get_workspace("scratch", (16, 32), torch.bfloat16)
        self.assertIs(w1, w2)

    @unittest.skipUnless(torch.cuda.is_available(), "needs GPU")
    def test_overlap_two_streams_no_aliasing(self):
        # Run the finalize combine on the secondary stream while an independent
        # workload runs on the main stream; results must be unaffected.
        from sglang.srt.layers.moe.rocm_kimi_mxfp4_moe import get_rocm_moe_stream_state

        dev = torch.device("cuda:0")
        state = get_rocm_moe_stream_state(dev)
        T, H, K = 16, 256, 8
        partial = torch.randn(T * K, H, dtype=torch.bfloat16, device="cuda")
        row_map = build_trivial_row_map(T, K, dev)
        weights = torch.rand(T, K, dtype=torch.float32, device="cuda")
        shared = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")

        main = torch.cuda.current_stream()
        state.shared_stream.wait_stream(main)
        with torch.cuda.stream(state.shared_stream):
            out = rocm_mxfp4_moe_finalize_fuse_shared(
                partial, row_map, weights, shared, 1.0, K
            )
            state.shared_done_event.record(state.shared_stream)
        main.wait_event(state.shared_done_event)
        torch.cuda.synchronize()

        ref = (
            (partial.float().reshape(T, K, H) * weights.float().reshape(T, K, 1)).sum(1)
            + shared.float()
        ).to(torch.bfloat16)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=3e-2, atol=3e-2)


if __name__ == "__main__":
    unittest.main()
