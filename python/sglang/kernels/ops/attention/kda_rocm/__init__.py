# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 KDA kernels for ROCm, ported from vLLM's AMD Kimi-K3 path.

* ``jit``: JIT build of vLLM's ``fused_kda_{decode,chunk}_kernel_rocm.cu``.
* ``kda_chunk`` / ``kda_prefill``: fused HIP chunk prefill (gfx950) with the
  vendored Triton chunk path as fallback.
* ``kda_decode``: fused HIP decode (conv1d + recurrence + gated RMSNorm).
* ``third_party``: vLLM's vendored Triton KDA kernels and the FLA helpers they
  import, unchanged apart from import paths.
"""
