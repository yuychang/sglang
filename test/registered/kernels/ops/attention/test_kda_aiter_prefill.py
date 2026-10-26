import unittest

import torch

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel


def _gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return arch.startswith("gfx950")


def _err_ratio(ref: torch.Tensor, tri: torch.Tensor) -> float:
    ref = ref.detach().double()
    tri = tri.detach().double()
    return (
        (ref - tri).square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-12)
    ).item()


class TestKdaAiterPrefill(unittest.TestCase):
    def test_extend_matches_triton_chunk_kda(self):
        if not _gfx950():
            self.skipTest("aiter chunk_kda prefill is measured on gfx950")
        torch.manual_seed(0)
        heads, dim = 12, 128
        seqlens = [64, 96]
        total = sum(seqlens)
        q = torch.randn(1, total, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, total, heads, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, total, heads, dim, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(1, total, heads, dim, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(1, total, heads, device="cuda", dtype=torch.bfloat16)
        a_log = torch.log(torch.empty(heads, device="cuda").uniform_(1, 16))
        dt_bias = torch.randn(heads * dim, device="cuda")
        cu = torch.tensor([0, 64, total], dtype=torch.int32, device="cuda")
        pool = torch.randn(8, heads, dim, dim, device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
        lower = -5.0

        TritonKDAKernel._logged_aiter_extend = False
        out = TritonKDAKernel().extend(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            ssm_states=pool,
            cache_indices=slots,
            query_start_loc=cu,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower,
            beta_is_raw=True,
        )
        self.assertTrue(TritonKDAKernel._logged_aiter_extend)
        self.assertEqual(tuple(out.shape), (1, total, heads, dim))

        ref_pool = pool.clone()
        # The aiter call already wrote pool. Rebuild the reference from a copy
        # taken before extend by re-reading is impossible; compare a second
        # pair of fresh pools instead.
        del ref_pool
        fresh = torch.randn(8, heads, dim, dim, device="cuda", dtype=torch.bfloat16)
        triton_pool = fresh.clone()
        triton_out = chunk_kda(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            initial_state=triton_pool,
            initial_state_indices=slots,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower,
            beta_is_raw=True,
        )
        aiter_pool = fresh.clone()
        aiter_out = TritonKDAKernel().extend(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            ssm_states=aiter_pool,
            cache_indices=slots,
            query_start_loc=cu,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower,
            beta_is_raw=True,
        )
        self.assertLess(_err_ratio(triton_out, aiter_out), 0.008)
        self.assertLess(
            _err_ratio(triton_pool[slots.long()].float(), aiter_pool[slots.long()].float()),
            0.008,
        )
        untouched = [i for i in range(8) if i not in (3, 5)]
        self.assertTrue(
            torch.equal(aiter_pool[untouched], fresh[untouched])
        )


if __name__ == "__main__":
    unittest.main()
