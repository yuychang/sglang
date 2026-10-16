"""ROCm-only server-arg resolution for Kimi-K3."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def disable_kimi_k3_rocm_symm_mem(server_args: ServerArgs) -> None:
    """Turn `--enable-symm-mem` back off unless every phase runs eager.

    Symm-mem allocations are per-forward, so an address captured into a graph is
    not stable across replays or ranks; captured prefill and spec decode both
    corrupt. Gates on the arch itself because it runs before the model-specific
    hook block.
    """
    if not get_platform().is_hip:
        return
    from sglang.srt.connector import ConnectorType
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.utils import parse_connector_type

    cfg = resolving_view(server_args)
    if not cfg.enable_symm_mem:
        return
    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        return
    if model_config_of(server_args).hf_config.architectures[0] not in (
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    ):
        return
    graph = cfg.cuda_graph_config
    if (
        graph.decode.backend == Backend.DISABLED
        and graph.prefill.backend == Backend.DISABLED
    ):
        return
    declare_resolution(
        server_args,
        "disable_kimi_k3_rocm_symm_mem",
        enable_symm_mem=False,
    )
    logger.warning(
        "Kimi hybrid model: ignoring --enable-symm-mem because CUDA graphs are on. "
        "The symmetric-memory pool's per-forward allocations are not valid for the "
        "lifetime of a captured graph, which corrupts spec decode and can "
        "silently produce wrong output. The auto-probed K3 fused all-reduce is faster "
        "anyway. Disable capture on every phase "
        "(--cuda-graph-backend-decode=disabled --cuda-graph-backend-prefill=disabled) "
        "if you genuinely need symmetric memory."
    )
