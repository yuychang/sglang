"""The AITER KDA adapter must import the module ROCm/aiter actually ships."""

from __future__ import annotations

import sys
import types

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SHIPPED = "aiter.ops.flydsl.linear_attention_kernels"
_RETIRED = "aiter.ops.flydsl.kda_decode"


def test_aiter_source_resolves_the_shipped_kda_module(monkeypatch):
    """A missing aiter.ops.flydsl.kda_decode must not disable the AITER backend.

    ROCm/aiter#6026 exports KDA decode from linear_attention_kernels. The
    retired module name raises ModuleNotFoundError, and the adapter then
    reports the backend unavailable.
    """
    from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip

    monkeypatch.setenv("SGLANG_ROCM_K3_FLYDSL_SOURCE", "aiter")
    decode = object()
    supported = object()
    shipped = types.ModuleType(_SHIPPED)
    shipped.flydsl_kda_decode_with_f_b = decode
    shipped.is_flydsl_kda_decode_supported = supported
    retired = types.ModuleType(_RETIRED)
    retired.flydsl_kda_decode_with_f_b = object()
    retired.is_flydsl_kda_decode_supported = object()
    monkeypatch.setitem(sys.modules, _SHIPPED, shipped)
    monkeypatch.setitem(sys.modules, _RETIRED, retired)

    assert kda_fused_decode_aiter_hip._ops() == (decode, supported)
