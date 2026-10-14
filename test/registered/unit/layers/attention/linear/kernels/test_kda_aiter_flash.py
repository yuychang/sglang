"""Unit tests for the AITER FlashKDA prefill routing wrapper and its backend wiring."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.srt.layers.attention.linear.kernels.kda_aiter_flash import (
    AiterFlashKDAKernel,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_MODULE = "sglang.srt.layers.attention.linear.kernels.kda_aiter_flash"
_H, _D, _SLOTS = 2, 128, 8


def _make_kernel():
    triton = Mock(spec=TritonKDAKernel)
    triton.extend.return_value = "triton-out"
    fwd = Mock()
    with patch(f"{_MODULE}._load_chunk_kimi_delta_attn", return_value=fwd):
        kernel = AiterFlashKDAKernel(triton_fallback=triton)
    return kernel, fwd, triton


def _inputs(seq_lens=(64, 100)):
    total = sum(seq_lens)

    def act():
        return torch.zeros(1, total, _H, _D, dtype=torch.bfloat16)

    return dict(
        q=act(),
        k=act(),
        v=act(),
        g=act(),
        beta=torch.zeros(1, total, _H, dtype=torch.bfloat16),
        ssm_states=torch.zeros(_SLOTS, _H, _D, _D, dtype=torch.float32),
        cache_indices=torch.tensor([1, 3], dtype=torch.int32),
        query_start_loc=torch.tensor(
            [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32
        ),
        A_log=torch.zeros(1, 1, _H, 1, dtype=torch.float32),
        dt_bias=torch.zeros(_H * _D, dtype=torch.float32),
        lower_bound=-5.0,
        beta_is_raw=True,
    )


class TestAiterFlashKDARouting(CustomTestCase):
    def _assert_routed_to_triton(self, fwd, triton, out):
        self.assertEqual(out, "triton-out")
        fwd.assert_not_called()
        triton.extend.assert_called_once()

    def test_interior_track_snapshot_routes_to_triton(self):
        """FlashKDA cannot write the fp32 track snapshot; leaving it unwritten
        silently corrupts later prefix-cache restores."""
        kernel, fwd, triton = _make_kernel()
        track_state = torch.zeros(2, _H, _D, _D, dtype=torch.float32)
        track_chunk_idx = torch.tensor([1, -1], dtype=torch.int32)

        out = kernel.extend(
            **_inputs(),
            return_intermediate_states=True,
            track_ssm_h_src=torch.tensor([1], dtype=torch.long),
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )

        self._assert_routed_to_triton(fwd, triton, out)
        forwarded = triton.extend.call_args.kwargs
        self.assertIs(forwarded["track_state"], track_state)
        self.assertIs(forwarded["track_chunk_idx"], track_chunk_idx)
        self.assertTrue(forwarded["return_intermediate_states"])

    def test_draft_extend_routes_to_triton(self):
        """FlashKDA commits the slot in place, which breaks draft-extend rollback."""
        kernel, fwd, triton = _make_kernel()
        out = kernel.extend(**_inputs(), is_spec_decode=True)
        self._assert_routed_to_triton(fwd, triton, out)

    def test_activated_beta_routes_to_triton(self):
        """The kernel sigmoids beta itself; pre-activated beta would be squashed twice."""
        kernel, fwd, triton = _make_kernel()
        out = kernel.extend(**{**_inputs(), "beta_is_raw": False})
        self._assert_routed_to_triton(fwd, triton, out)
        self.assertFalse(triton.extend.call_args.kwargs["beta_is_raw"])

    def test_out_of_range_lower_bound_routes_to_triton(self):
        """AITER raises outside [-5, 0); the screen must reject before launch."""
        for lower_bound in (None, -5.5, 0.0):
            with self.subTest(lower_bound=lower_bound):
                kernel, fwd, triton = _make_kernel()
                out = kernel.extend(**{**_inputs(), "lower_bound": lower_bound})
                self._assert_routed_to_triton(fwd, triton, out)

    def test_aligned_track_runs_flashkda_with_pad_rows_on_spare_slot(self):
        """Final-state-only tracking stays native; -1 pad rows land on the spare
        last slot unread, and rows without a prefix start from zero."""
        kernel, fwd, triton = _make_kernel()
        x = _inputs(seq_lens=(64, 100, 7))
        x["cache_indices"] = torch.tensor([2, -1, 5], dtype=torch.int32)

        out, h = kernel.extend(
            **x,
            extend_prefix_lens=torch.tensor([10, 0, 0], dtype=torch.int32),
            return_intermediate_states=True,
            track_ssm_h_src=torch.empty(0, dtype=torch.long),
        )

        triton.extend.assert_not_called()
        fwd.assert_called_once()
        self.assertIsNone(h)
        self.assertEqual(out.shape, x["v"].shape)
        kwargs = fwd.call_args.kwargs
        self.assertIs(kwargs["state_cache"], x["ssm_states"])
        self.assertEqual(kwargs["state_indices"].tolist(), [2, _SLOTS - 1, 5])
        self.assertEqual(kwargs["has_initial_state"].tolist(), [True, False, False])
        self.assertTrue(kwargs["use_beta_sigmoid_in_kernel"])
        self.assertEqual(kwargs["chunk_size"], 32)


class TestAiterFlashKDABackendWiring(CustomTestCase):
    def _dispatcher(self, *, hip: bool, gfx95: bool):
        kernel, _, _ = _make_kernel()
        backend = "sglang.srt.layers.attention.linear.kda_backend"
        with (
            patch(f"{backend}.is_hip", return_value=hip),
            patch(f"{backend}.is_gfx95_supported", return_value=gfx95),
            patch(f"{_MODULE}.AiterFlashKDAKernel", return_value=kernel),
        ):
            dispatcher = KDAKernelDispatcher(
                decode_backend=LinearAttnKernelBackend.TRITON,
                prefill_backend=LinearAttnKernelBackend.AITER_FLASHKDA,
                verify_backend=LinearAttnKernelBackend.TRITON,
            )
        return dispatcher, kernel

    def test_choice_selects_kernel_only_on_gfx95(self):
        dispatcher, kernel = self._dispatcher(hip=True, gfx95=True)
        self.assertIs(dispatcher.extend_kernel, kernel)
        self.assertIsInstance(dispatcher.decode_kernel, TritonKDAKernel)
        dispatcher, _ = self._dispatcher(hip=True, gfx95=False)
        self.assertIsInstance(dispatcher.extend_kernel, TritonKDAKernel)
        with self.assertRaisesRegex(ValueError, "requires ROCm"):
            self._dispatcher(hip=False, gfx95=False)

    def test_inherited_base_keeps_triton_decode(self):
        """A prefill-only base backend must not reach the decode dispatcher."""
        args = ServerArgs(model_path="dummy", linear_attn_backend="aiter_flashkda")
        with override_platform(is_sm100=False), override_platform(is_cuda=False):
            handle_linear_attn_backend(args)
        self.assertEqual(
            resolution_result(args, "linear_attn_decode_backend"), "triton"
        )

        explicit = ServerArgs(
            model_path="dummy", linear_attn_decode_backend="aiter_flashkda"
        )
        with override_platform(is_sm100=False), override_platform(is_cuda=False):
            with self.assertRaisesRegex(ValueError, "prefill-only"):
                handle_linear_attn_backend(explicit)


if __name__ == "__main__":
    unittest.main()
