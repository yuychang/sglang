"""ROCm-only: AITER SiTU-v2 MoE activation selection.

Mirrors vLLM's ``_resolve_situv2_activation`` / ``_sync_aiter_situv2_moe_env``
(``vllm/_aiter_ops.py``), driven by ``SGLANG_ROCM_USE_AITER_MOE_SITUV2``.
AITER reads ``AITER_SITUV2_A8W4`` / ``AITER_SITUV2_A4W4`` from the process
environment and defaults to a16w4, so exactly one is set for a8w4/a4w4 and
both are cleared for a16w4.
"""

import logging
import os

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_AITER_SITUV2_ACT_ENV = {
    "a8w4": "AITER_SITUV2_A8W4",
    "a4w4": "AITER_SITUV2_A4W4",
}
_VALID_ACTIVATIONS = ("a4w4", "a8w4", "a16w4")

_synced = False


def resolve_situv2_activation() -> str:
    """SGLANG_ROCM_USE_AITER_MOE_SITUV2 -> a4w4 | a8w4 | a16w4.

    auto (default) and the legacy 1 mean a4w4; legacy 0 means a16w4.
    """
    value = (envs.SGLANG_ROCM_USE_AITER_MOE_SITUV2.get() or "auto").lower()
    if value in ("auto", "1"):
        return "a4w4"
    if value == "0":
        return "a16w4"
    if value not in _VALID_ACTIVATIONS:
        raise ValueError(
            f"SGLANG_ROCM_USE_AITER_MOE_SITUV2={value!r}; expected one of "
            f"auto, 0, 1, {', '.join(_VALID_ACTIVATIONS)}"
        )
    return value


def sync_aiter_situv2_moe_env() -> str:
    """Export the AITER_SITUV2_* pair for the resolved activation; idempotent."""
    global _synced
    activation = resolve_situv2_activation()
    selected = _AITER_SITUV2_ACT_ENV.get(activation)
    for name in _AITER_SITUV2_ACT_ENV.values():
        if name == selected:
            os.environ[name] = "1"
        else:
            os.environ.pop(name, None)
    if not _synced:
        _synced = True
        logger.info(
            "AITER SiTU-v2 MoE activation: %s (SGLANG_ROCM_USE_AITER_MOE_SITUV2=%s)",
            activation,
            envs.SGLANG_ROCM_USE_AITER_MOE_SITUV2.get(),
        )
    return activation
