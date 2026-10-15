"""Batched per-tensor e4m3fn quantization of the MLA prefill Q/K/V operands (ROCm).

Two Triton launches quantize up to three strided (tokens, heads, dim) tensors,
bit-identical per tensor to

    amax = torch.linalg.vector_norm(x, ord=inf, dtype=torch.float32)
    scale = (amax / 448.0).clamp_(min=1e-12)
    out, scale = scaled_fp8_quant(x.view(rows, -1), scale.view(1))

with AITER's static per-tensor quant. The amax pass max-reduces sign-masked
16-bit patterns, whose order is the magnitude order for non-NaN values, so the
max is exact and NaN propagates like torch. The quant pass rebuilds torch's
scale (a CPU-scalar divisor becomes a multiply by its fp32 reciprocal) and
replays AITER's cast: v_rcp_f32, fp32 multiply, clamp to +-448, RNE conversion.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver

from sglang.srt.utils import is_hip

_is_hip = is_hip()

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
# torch evaluates ``amax / 448.0`` as ``amax * fl32(1 / 448)``.
_INV_FP8_MAX = float(torch.tensor(1.0, dtype=torch.float32) / _FP8_MAX)
_MIN_SCALE = 1e-12

# Contiguous elements per load; a head dim must be a multiple of it.
_CHUNK = 64
_BLOCK_C = 64
# No recorded sweep. 512 int32 partials make each quant program's amax reduce one 2 KB load.
_NUM_PARTIALS = 512
# No recorded sweep. Caps the grid-stride quant grid at 8 programs per CU on 256-CU MI355.
_QUANT_PROGRAMS = 2048
_NUM_WARPS = 4
_MAX_TENSORS = 3


@triton.jit
def _rcp_f32(x):
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        "=v,v",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _chunk_offsets(
    tile,
    num_chunks,
    H: tl.constexpr,
    D: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    chunks = tile * BLOCK_C + tl.arange(0, BLOCK_C)
    row = chunks // (D // CHUNK)
    col = (chunks % (D // CHUNK)) * CHUNK
    src = (row // H).to(tl.int64) * ST + (row % H) * SH + col
    lane = tl.arange(0, CHUNK)
    src = src[:, None] + lane[None, :]
    dst = chunks.to(tl.int64)[:, None] * CHUNK + lane[None, :]
    return src, dst, (chunks < num_chunks)[:, None]


@triton.jit
def _partial_amax_one(
    x_ptr,
    num_tokens,
    partial_ptr,
    pid,
    H: tl.constexpr,
    D: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    num_chunks = num_tokens * (H * D // CHUNK)
    acc = tl.zeros((BLOCK_C, CHUNK), dtype=tl.int32)
    for tile in range(pid, tl.cdiv(num_chunks, BLOCK_C), NUM_PARTIALS):
        src, _, mask = _chunk_offsets(tile, num_chunks, H, D, ST, SH, CHUNK, BLOCK_C)
        bits = tl.load(x_ptr + src, mask=mask, other=0.0).to(tl.int16, bitcast=True)
        acc = tl.maximum(acc, bits.to(tl.int32) & 0x7FFF)
    tl.store(partial_ptr + pid, tl.max(tl.max(acc, axis=1), axis=0))


@triton.jit(do_not_specialize=["n0", "n1", "n2"])
def _mla_prefill_partial_amax_kernel(
    buf_ptr,
    x0_ptr,
    n0,
    x1_ptr,
    n1,
    x2_ptr,
    n2,
    H0: tl.constexpr,
    D0: tl.constexpr,
    ST0: tl.constexpr,
    SH0: tl.constexpr,
    H1: tl.constexpr,
    D1: tl.constexpr,
    ST1: tl.constexpr,
    SH1: tl.constexpr,
    H2: tl.constexpr,
    D2: tl.constexpr,
    ST2: tl.constexpr,
    SH2: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    which = tl.program_id(1)
    partial_ptr = (buf_ptr + 3).to(tl.pointer_type(tl.int32)) + which * NUM_PARTIALS
    if which == 0:
        _partial_amax_one(
            x0_ptr, n0, partial_ptr, pid,
            H0, D0, ST0, SH0, NUM_PARTIALS, CHUNK, BLOCK_C,
        )  # fmt: skip
    elif which == 1:
        _partial_amax_one(
            x1_ptr, n1, partial_ptr, pid,
            H1, D1, ST1, SH1, NUM_PARTIALS, CHUNK, BLOCK_C,
        )  # fmt: skip
    else:
        _partial_amax_one(
            x2_ptr, n2, partial_ptr, pid,
            H2, D2, ST2, SH2, NUM_PARTIALS, CHUNK, BLOCK_C,
        )  # fmt: skip


@triton.jit
def _quant_one(
    x_ptr,
    num_tokens,
    out_ptr,
    partial_ptr,
    scale_ptr,
    pid,
    num_programs,
    inv_fp8_max,
    min_scale,
    H: tl.constexpr,
    D: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    IS_BF16: tl.constexpr,
    FP8_MAX: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    amax_bits = tl.max(tl.load(partial_ptr + tl.arange(0, NUM_PARTIALS)), axis=0)
    if IS_BF16:
        amax = (amax_bits << 16).to(tl.float32, bitcast=True)
    else:
        amax = amax_bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    scale = tl.maximum(amax * inv_fp8_max, min_scale, propagate_nan=tl.PropagateNan.ALL)
    if pid == 0:
        tl.store(scale_ptr, scale)
    inv_scale = _rcp_f32(scale)

    num_chunks = num_tokens * (H * D // CHUNK)
    for tile in range(pid, tl.cdiv(num_chunks, BLOCK_C), num_programs):
        src, dst, mask = _chunk_offsets(tile, num_chunks, H, D, ST, SH, CHUNK, BLOCK_C)
        x = tl.load(x_ptr + src, mask=mask, other=0.0).to(tl.float32)
        y = tl.clamp(x * inv_scale, -FP8_MAX, FP8_MAX)
        tl.store(out_ptr + dst, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["n0", "n1", "n2", "inv_fp8_max", "min_scale"])
def _mla_prefill_quant_kernel(
    buf_ptr,
    inv_fp8_max,
    min_scale,
    x0_ptr,
    n0,
    o0_ptr,
    x1_ptr,
    n1,
    o1_ptr,
    x2_ptr,
    n2,
    o2_ptr,
    H0: tl.constexpr,
    D0: tl.constexpr,
    ST0: tl.constexpr,
    SH0: tl.constexpr,
    H1: tl.constexpr,
    D1: tl.constexpr,
    ST1: tl.constexpr,
    SH1: tl.constexpr,
    H2: tl.constexpr,
    D2: tl.constexpr,
    ST2: tl.constexpr,
    SH2: tl.constexpr,
    IS_BF16: tl.constexpr,
    FP8_MAX: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    which = tl.program_id(1)
    partial_ptr = (buf_ptr + 3).to(tl.pointer_type(tl.int32)) + which * NUM_PARTIALS
    scale_ptr = buf_ptr + which
    if which == 0:
        _quant_one(
            x0_ptr, n0, o0_ptr, partial_ptr, scale_ptr, pid, num_programs,
            inv_fp8_max, min_scale, H0, D0, ST0, SH0,
            IS_BF16, FP8_MAX, NUM_PARTIALS, CHUNK, BLOCK_C,
        )  # fmt: skip
    elif which == 1:
        _quant_one(
            x1_ptr, n1, o1_ptr, partial_ptr, scale_ptr, pid, num_programs,
            inv_fp8_max, min_scale, H1, D1, ST1, SH1,
            IS_BF16, FP8_MAX, NUM_PARTIALS, CHUNK, BLOCK_C,
        )  # fmt: skip
    else:
        _quant_one(
            x2_ptr, n2, o2_ptr, partial_ptr, scale_ptr, pid, num_programs,
            inv_fp8_max, min_scale, H2, D2, ST2, SH2,
            IS_BF16, FP8_MAX, NUM_PARTIALS, CHUNK, BLOCK_C,
        )  # fmt: skip


_MAX_INT32 = 2**31 - 1

# CompiledKernel per (kernel, constexprs, device, input specialization), so
# repeat calls skip Triton's per-call argument binder on this host-bound path.
# Token counts and float scalars are never specialized; per pointer Triton 3.7
# derives 16-byte alignment and (AMD buffer ops) whether storage spans <= 2 GB.
# Scratch and outputs are allocated here, so only the inputs vary.
_compiled: dict = {}
# The key mirrors Triton 3.7's specialization only; other versions launch through
# Triton. Drop the direct launch once Triton's binder no longer dominates host time.
_DIRECT_LAUNCH = triton.__version__.split(".")[:2] == ["3", "7"]


def _launch(kernel, grid, args, constexprs, device_index, specialization):
    key = (kernel, constexprs, device_index, specialization)
    compiled = _compiled.get(key)
    if compiled is None:
        compiled = kernel[grid](*args, *constexprs, num_warps=_NUM_WARPS)
        if _DIRECT_LAUNCH and all(
            hasattr(compiled, attr)
            for attr in ("run", "function", "packed_metadata", "launch_metadata")
        ):
            _compiled[key] = compiled
        return
    args = (*args, *constexprs)
    stream = driver.active.get_current_stream(device_index)
    compiled.run(
        grid[0],
        grid[1],
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        compiled.launch_metadata(grid, stream, *args),
        knobs.runtime.launch_enter_hook,
        knobs.runtime.launch_exit_hook,
        *args,
    )


def _mla_prefill_fp8_quant_supported(tensors: Sequence[torch.Tensor]) -> bool:
    # _rcp_f32 inlines AMDGCN asm, so the kernels only build for ROCm.
    if not _is_hip or not 1 <= len(tensors) <= _MAX_TENSORS:
        return False
    first = tensors[0]
    if first.dtype not in (torch.bfloat16, torch.float16) or not first.is_cuda:
        return False
    for x in tensors:
        if (
            x.dtype != first.dtype
            or x.device != first.device
            or x.dim() != 3
            or x.shape[0] == 0
            or x.shape[2] % _CHUNK != 0
            or x.stride(2) != 1
            or x.numel() > _MAX_INT32
        ):
            return False
    return True


def mla_prefill_fp8_quant(
    tensors: Sequence[torch.Tensor],
) -> Optional[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Quantize up to three (tokens, heads, dim) tensors to per-tensor e4m3fn.

    Returns ``(quantized, scale)`` per input: a contiguous FP8 tensor of the
    input's shape and a (1,) float32 descale (amax / 448, floored at 1e-12).
    Returns None if the inputs are outside what the kernels handle.
    """
    if not _mla_prefill_fp8_quant_supported(tensors):
        return None
    num = len(tensors)
    device = tensors[0].device
    # [3 scales | 3 x NUM_PARTIALS int32 partial amax bit patterns]
    buf = torch.empty(
        _MAX_TENSORS * (_NUM_PARTIALS + 1), dtype=torch.float32, device=device
    )
    outs = [
        torch.empty_like(x, dtype=_FP8_DTYPE, memory_format=torch.contiguous_format)
        for x in tensors
    ]
    padded = list(tensors) + [tensors[0]] * (_MAX_TENSORS - num)
    out_padded = outs + [outs[0]] * (_MAX_TENSORS - num)

    layout = []
    specialization = []
    amax_args = [buf]
    quant_args = [buf, _INV_FP8_MAX, _MIN_SCALE]
    max_chunks = 0
    for x, o in zip(padded, out_padded):
        shape, stride = x.shape, x.stride()
        layout += (shape[1], shape[2], stride[0], stride[1])
        specialization.append(
            (x.data_ptr() % 16 == 0, x.untyped_storage().nbytes() <= _MAX_INT32)
        )
        amax_args += (x, shape[0])
        quant_args += (x, shape[0], o)
        max_chunks = max(max_chunks, shape[0] * shape[1] * shape[2] // _CHUNK)
    layout = tuple(layout)
    specialization = (tensors[0].dtype, *specialization)

    amax_constexprs = layout + (_NUM_PARTIALS, _CHUNK, _BLOCK_C)
    _launch(
        _mla_prefill_partial_amax_kernel,
        (_NUM_PARTIALS, num),
        amax_args,
        amax_constexprs,
        device.index,
        specialization,
    )
    quant_constexprs = layout + (
        tensors[0].dtype == torch.bfloat16,
        _FP8_MAX,
        _NUM_PARTIALS,
        _CHUNK,
        _BLOCK_C,
    )
    num_tiles = -(-max_chunks // _BLOCK_C)
    _launch(
        _mla_prefill_quant_kernel,
        (min(num_tiles, _QUANT_PROGRAMS), num),
        quant_args,
        quant_constexprs,
        device.index,
        specialization,
    )
    scales = buf.split([1, 1, 1, _MAX_TENSORS * _NUM_PARTIALS])
    return [(outs[i], scales[i]) for i in range(num)]
