"""Numerical check of the AITER FlashKDA KDA prefill backend against Triton chunk_kda."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

_ON_GFX95 = is_hip() and torch.cuda.is_available() and is_gfx95_supported()

LOWER_BOUND = -5.0
H, D = 8, 128
NUM_SLOTS = 6  # Last row is the spare pad slot (MambaPool allocates size + 1).


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _make_inputs(seq_lens, *, device="cuda"):
    total = sum(seq_lens)
    cu = torch.zeros(len(seq_lens) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lens, device=device).cumsum(0)

    def act(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16) * 0.5

    return dict(
        q=act(1, total, H, D),
        k=act(1, total, H, D),
        v=act(1, total, H, D),
        g=act(1, total, H, D),
        # Raw beta logits; both paths apply the sigmoid in-kernel.
        beta=torch.randn(1, total, H, device=device, dtype=torch.bfloat16),
        A_log=torch.randn(1, 1, H, 1, device=device, dtype=torch.float32) * 0.5,
        dt_bias=torch.randn(H * D, device=device, dtype=torch.float32) * 0.1,
        query_start_loc=cu,
        # V-first fp32 pool [slots, H, V, K].
        pool=torch.randn(NUM_SLOTS, H, D, D, device=device, dtype=torch.float32) * 0.1,
    )


def _extend(kernel, d, pool, cache_indices, prefix_lens):
    return kernel.extend(
        d["q"].clone(),
        d["k"].clone(),
        d["v"].clone(),
        d["g"].clone(),
        d["beta"].clone(),
        ssm_states=pool,
        cache_indices=cache_indices,
        query_start_loc=d["query_start_loc"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=LOWER_BOUND,
        beta_is_raw=True,
        extend_prefix_lens=prefix_lens,
    )


@unittest.skipUnless(_ON_GFX95, "AITER FlashKDA needs a ROCm gfx95x GPU")
class TestAiterFlashKDAPrefill(CustomTestCase):
    def test_matches_chunk_kda_with_prefix_and_pad_rows(self):
        """Prefixed rows resume their slot, fresh rows ignore stale slot contents,
        and -1 pad rows never touch a live slot."""
        from sglang.srt.layers.attention.linear.kernels.kda_aiter_flash import (
            AiterFlashKDAKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.kda_triton import (
            TritonKDAKernel,
        )

        torch.manual_seed(0)
        seq_lens = [200, 64, 33, 517]
        d = _make_inputs(seq_lens)
        cache_indices = torch.tensor([3, 1, -1, 0], device="cuda", dtype=torch.int32)
        prefix_lens = torch.tensor([17, 0, 0, 96], device="cuda", dtype=torch.int32)
        fresh_slot = 1

        # chunk_kda always reads the slot, so the reference sees a cleared slot
        # where FlashKDA must skip the stale state via has_initial_state.
        ref_pool = d["pool"].clone()
        ref_pool[fresh_slot] = 0
        triton = TritonKDAKernel()
        ref_out = _extend(triton, d, ref_pool, cache_indices, prefix_lens)

        pool = d["pool"].clone()
        out = _extend(
            AiterFlashKDAKernel(triton_fallback=triton),
            d,
            pool,
            cache_indices,
            prefix_lens,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.isfinite(out).all())
        # bf16 cross-implementation noise (chunk 32 FlashKDA vs chunk 64 Triton).
        self.assertGreater(_cos(ref_out, out), 0.995)
        live = [0, 1, 3]
        self.assertGreater(_cos(ref_pool[live], pool[live]), 0.999)
        # Slots no sequence owns stay bit-identical.
        for slot in (2, 4):
            self.assertTrue(torch.equal(pool[slot], d["pool"][slot]))


if __name__ == "__main__":
    unittest.main()
