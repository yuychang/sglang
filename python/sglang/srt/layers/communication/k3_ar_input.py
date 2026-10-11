"""Direct producer access to AITER custom all-reduce's registered input pool."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)


def get_k3_ar_input(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return the custom-AR input-pool base view, or ``None`` to fall back.

    Only exact base-address views are returned. AITER still receives the tensor
    through its normal unregistered-input API; pointer equality suppresses the
    staging copy without selecting the registered-output kernel mode.
    """
    if not is_hip() or not envs.SGLANG_K3_AR_INPLACE_INPUT.get():
        return None
    try:
        from sglang.srt.distributed import get_tp_group

        ca_comm = get_tp_group().ca_comm
        if (
            ca_comm is None
            or getattr(ca_comm, "disabled", True)
            or not hasattr(ca_comm, "get_registered_input_buffer")
        ):
            return None
        out = ca_comm.get_registered_input_buffer(shape, dtype)
        if out.device != device or not ca_comm.should_custom_ar(out):
            return None
        return out
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("K3 direct custom-AR input unavailable: %s", exc)
        return None
