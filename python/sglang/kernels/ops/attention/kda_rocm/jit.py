# SPDX-License-Identifier: Apache-2.0
"""JIT build of the vLLM Kimi-K3 KDA HIP kernels.

vLLM compiles ``fused_kda_decode_kernel_rocm.cu`` for gfx942/gfx950 and
``fused_kda_chunk_kernel_rocm.cu`` for gfx950 only (CMakeLists.txt, the
``FUSED_KDA_{DECODE,CHUNK}_HIP_ARCHS`` blocks). This module builds the same
sources with the same arch split as a stable-ABI torch library and registers
the ops under ``torch.ops.sgl_kimi_k3_rocm``.
"""

import functools
import hashlib
import logging
import os
import shutil
from typing import Optional

import torch

logger = logging.getLogger(__name__)

OPS_NAMESPACE = "sgl_kimi_k3_rocm"
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_DECODE_SRC = "fused_kda_decode_kernel_rocm.cu"
_CHUNK_SRC = "fused_kda_chunk_kernel_rocm.cu"
_BINDINGS_SRC = "kda_rocm_bindings.cpp"
_HEADER = "torch_utils.h"
_DECODE_ARCHS = ("gfx942", "gfx950")
_CHUNK_ARCHS = ("gfx950",)
# vLLM's _C_stable_libtorch target version (CMakeLists.txt).
_TORCH_TARGET_VERSION = "0x020B000000000000ULL"


def device_arch() -> str:
    if torch.version.hip is None or not torch.cuda.is_available():
        return ""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.gcnArchName.split(":")[0]


def _build_dir(arch: str, sources: list[str]) -> str:
    h = hashlib.sha256()
    for src in sources + [_HEADER]:
        with open(os.path.join(_CSRC, src), "rb") as f:
            h.update(f.read())
    h.update(torch.__version__.encode())
    root = os.environ.get(
        "SGLANG_KDA_ROCM_JIT_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "sglang", "kda_rocm"),
    )
    path = os.path.join(root, f"{arch}-{h.hexdigest()[:16]}")
    os.makedirs(path, exist_ok=True)
    return path


@functools.cache
def load_ops() -> Optional[object]:
    """Build (once per source hash) and load the ops; None if unsupported."""
    arch = device_arch()
    if arch not in _DECODE_ARCHS:
        return None
    with_chunk = arch in _CHUNK_ARCHS
    sources = [_BINDINGS_SRC, _DECODE_SRC] + ([_CHUNK_SRC] if with_chunk else [])
    defines = ["-DUSE_ROCM", f"-DTORCH_TARGET_VERSION={_TORCH_TARGET_VERSION}"]
    if with_chunk:
        defines.append("-DSGL_KDA_ROCM_ENABLE_CHUNK")
    from torch.utils.cpp_extension import load

    build_dir = _build_dir(arch, sources)
    # torch hipifies the sources it is given in place; build from copies so
    # nothing is generated next to the package sources.
    src_dir = os.path.join(build_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    for name in sources + [_HEADER]:
        shutil.copyfile(os.path.join(_CSRC, name), os.path.join(src_dir, name))
    saved_arch = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = arch
    try:
        load(
            name=f"sgl_kimi_k3_kda_rocm_{arch}",
            sources=[os.path.join(src_dir, s) for s in sources],
            extra_include_paths=[src_dir],
            extra_cflags=["-O3", *defines],
            extra_cuda_cflags=["-O3", *defines],
            build_directory=build_dir,
            is_python_module=False,
            verbose=False,
        )
    except Exception:
        logger.exception("Kimi-K3 KDA HIP kernels failed to build for %s", arch)
        return None
    finally:
        if saved_arch is None:
            os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = saved_arch
    return getattr(torch.ops, OPS_NAMESPACE)


def has_op(name: str) -> bool:
    ops = load_ops()
    return ops is not None and hasattr(ops, name)
