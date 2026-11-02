"""Optional MXFP4 copies of Kimi-K3's replicated latent MoE projections."""

import logging

import torch
from torch import nn

from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip
from sglang.kernels.ops.quantization import mxfp4_preshuffle_aiter_hip
from sglang.srt.utils import print_warning_once

logger = logging.getLogger(__name__)


class LatentMXFP4Projections(nn.Module):
    down_weight: torch.Tensor | None
    down_scale: torch.Tensor | None
    up_weight: torch.Tensor | None
    up_scale: torch.Tensor | None

    def __init__(self, *, min_tokens: int):
        super().__init__()
        if min_tokens < 1:
            raise ValueError(
                "SGLANG_OPT_HIP_K3_MOE_LATENT_MXFP4_MIN_TOKENS must be positive, "
                f"got {min_tokens}"
            )
        self.min_tokens = min_tokens
        self.register_buffer("down_weight", None, persistent=False)
        self.register_buffer("down_scale", None, persistent=False)
        self.register_buffer("up_weight", None, persistent=False)
        self.register_buffer("up_scale", None, persistent=False)

    def mark_unavailable(self) -> None:
        """Keep the original latent projections, e.g. for quantized ones."""
        self.down_weight = self.down_scale = self.up_weight = self.up_scale = None
        print_warning_once(
            "Kimi-K3 latent MXFP4 is unavailable for these weights/device; "
            "keeping the original latent projections"
        )

    def prepare(self, *, down_weight: torch.Tensor, up_weight: torch.Tensor) -> None:
        if (
            down_weight.shape != (3584, 7168)
            or up_weight.shape != (7168, 3584)
            or down_weight.dtype != torch.bfloat16
            or up_weight.dtype != torch.bfloat16
            or down_weight.device != up_weight.device
            or not latent_mxfp4_aiter_hip.supported(device=down_weight.device)
            or not mxfp4_preshuffle_aiter_hip.available()
        ):
            self.mark_unavailable()
            return
        down, down_scale = mxfp4_preshuffle_aiter_hip.pack(weight=down_weight)
        up, up_scale = mxfp4_preshuffle_aiter_hip.pack(weight=up_weight)
        if self.down_weight is None:
            self.down_weight, self.down_scale = down, down_scale
            self.up_weight, self.up_scale = up, up_scale
            return
        # A reload refreshes the copies in place: captured CUDA graphs keep
        # reading these buffers.
        for old, new in (
            (self.down_weight, down),
            (self.down_scale, down_scale),
            (self.up_weight, up),
            (self.up_scale, up_scale),
        ):
            old.copy_(new)

    def can_run(self, *, num_tokens: int) -> bool:
        return self.down_weight is not None and num_tokens >= self.min_tokens

    def down(self, *, x: torch.Tensor) -> torch.Tensor:
        if self.down_weight is None or self.down_scale is None:
            raise RuntimeError("latent MXFP4 down projection has not been prepared")
        return latent_mxfp4_aiter_hip.run(
            x=x, weight=self.down_weight, scale=self.down_scale
        )

    def up(self, *, x: torch.Tensor) -> torch.Tensor:
        if self.up_weight is None or self.up_scale is None:
            raise RuntimeError("latent MXFP4 up projection has not been prepared")
        return latent_mxfp4_aiter_hip.run(
            x=x, weight=self.up_weight, scale=self.up_scale
        )
