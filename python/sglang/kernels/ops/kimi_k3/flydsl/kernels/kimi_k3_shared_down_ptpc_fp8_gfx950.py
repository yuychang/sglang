# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused per-token FP8 quant + row-scaled FP8 shared-down GEMM for gfx950.

Replaces the two-launch PTPC shared-down path (``per_token_quant_hip`` then
``gemm_a8w8_bpreshuffle``) with one launch. The activation block ``[M, 768]``
is small enough to stay in LDS, so the per-token amax, the FP8 rounding and
the projection all fit in a single kernel and the row scales never round-trip
through HBM.

Numerics follow AITER PTPC exactly: ``xs = amax(x) / 448``,
``xq = round_e4m3(x / xs)`` and ``out = xs * ws * (xq . wq)``. FP8 E4M3 values
are exactly representable in BF16, so the rounded activation is written back
into the same LDS buffer and the projection loop stays BF16-in/FP32-accumulate.

All ``M`` tokens are accumulated against a single weight load, so the 5.5 MiB
FP8 weight is streamed once per launch rather than once per token. Even so the
per-token arithmetic is plain FP32 lane work with no MFMA, so the kernel's cost
grows with ``M`` while a tuned MFMA GEMM's does not. That is what bounds how
far up the batch range this form stays competitive, and it is why
``num_tokens`` is a build parameter over the decode CUDA-graph buckets rather
than a fixed 1: see ``bench_shared_down_ptpc_fp8.py`` for the crossover.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops, vector
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)
from aiter.ops.flydsl.kernels.vector import ReductionOp
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_dialect
from flydsl._mlir.dialects import math as math_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.rocdl import cvt_pk_f32_fp8, cvt_pk_fp8_f32
from flydsl.expr.typing import T

_SHARED_INTERMEDIATE_SIZE = 768
_HIDDEN_SIZE = 7168
_WAVE_SIZE = 64
_ELEMENTS_PER_LOAD = 4
_FP8_MAX = 448.0
# K=768 is 64 lanes x 12, so one wave covers a whole row with a single
# three-dword weight load. Splitting it into three one-dword loads measured the
# same, so this is not the bottleneck; it is kept because one load per row is
# the simpler indexing.
_K_ELEMENTS_PER_LANE = _SHARED_INTERMEDIATE_SIZE // _WAVE_SIZE
_K_DWORDS_PER_LANE = _K_ELEMENTS_PER_LANE // 4
# One thread per four activation elements, so a token's quant prologue is
# covered by 192 threads (three waves) of the projection block.
_VECS_PER_TOKEN = _SHARED_INTERMEDIATE_SIZE // _ELEMENTS_PER_LOAD
_QUANT_WAVES = _VECS_PER_TOKEN // _WAVE_SIZE
_TOKEN_BATCHES = (1, 2, 4, 8, 16)


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_kimi_k3_shared_down_ptpc_fp8_module(
    num_tokens: int = 2,
    rows_per_wave: int = 2,
    cu_count: int = 248,
    waves_per_eu: int = 0,
    weight_cache_modifier: int = 2,
):
    """Build the fused PTPC quant + FP8-weight down projection."""

    if num_tokens not in _TOKEN_BATCHES:
        raise ValueError(f"num_tokens must be one of {_TOKEN_BATCHES}")
    if rows_per_wave not in (1, 2, 3, 4, 5, 6, 8):
        raise ValueError("rows_per_wave must be 1, 2, 3, 4, 5, 6, or 8")
    if not 1 <= cu_count <= 256:
        raise ValueError("cu_count must be between 1 and 256")
    if waves_per_eu < 0:
        raise ValueError("waves_per_eu must be non-negative")
    if weight_cache_modifier not in (0, 1, 2, 3):
        raise ValueError("weight_cache_modifier must be between 0 and 3")

    output_groups = (_HIDDEN_SIZE + rows_per_wave - 1) // rows_per_wave
    waves_per_block = min(16, (output_groups + cu_count - 1) // cu_count)
    block_threads = waves_per_block * _WAVE_SIZE
    if block_threads < _VECS_PER_TOKEN:
        raise ValueError(
            f"rows_per_wave={rows_per_wave} yields {block_threads} threads, "
            f"which cannot cover the {_VECS_PER_TOKEN}-thread quant prologue"
        )
    groups_per_grid = cu_count * waves_per_block
    persistent_iterations = (output_groups + groups_per_grid - 1) // groups_per_grid
    activation_elements = num_tokens * _SHARED_INTERMEDIATE_SIZE
    activation_vectors = num_tokens * _VECS_PER_TOKEN
    load_iterations = (activation_vectors + block_threads - 1) // block_threads

    @fx.struct
    class SharedStorage:
        activated: fx.Array[fx.BFloat16, activation_elements, 16]
        amax: fx.Array[fx.Float32, num_tokens * _QUANT_WAVES, 4]
        scales: fx.Array[fx.Float32, num_tokens, 4]

    kernel_name = (
        f"kimi_k3_m{num_tokens}_shared_down_ptpc_fp8_gfx950"
        f"_rpw{rows_per_wave}_cu{cu_count}_wpb{waves_per_block}"
        f"_wpe{waves_per_eu}_wcm{weight_cache_modifier}"
    )

    @flyc.kernel(
        name=kernel_name,
        known_block_size=[block_threads, 1, 1],
    )
    def shared_down_ptpc_kernel(
        activation: fx.Pointer,
        weight: fx.Pointer,
        weight_scale: fx.Pointer,
        output: fx.Pointer,
    ):
        i32 = T.i32
        f32 = T.f32
        fm_fast = arith.FastMathFlags.fast
        tid = ArithValue(gpu.thread_idx.x)
        lane = tid % arith.constant(_WAVE_SIZE, type=i32)
        wave = tid // arith.constant(_WAVE_SIZE, type=i32)

        activation_rsrc = ptr_rsrc(activation)
        weight_rsrc = ptr_rsrc(weight)
        scale_rsrc = ptr_rsrc(weight_scale)
        output_rsrc = ptr_rsrc(output)
        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        activated_lds = storage.activated.ptr
        amax_lds = storage.amax.ptr
        scales_lds = storage.scales.ptr

        vec2_f32 = T.vec(2, f32)
        vec4_bf16 = T.vec(_ELEMENTS_PER_LOAD, T.bf16)
        vec4_f32 = T.vec(_ELEMENTS_PER_LOAD, f32)
        zero_f32 = arith.constant(0.0, type=f32)
        one_f32 = arith.constant(1.0, type=f32)
        zero_i32 = arith.constant(0, type=i32)
        is_lane_zero = arith.cmpi(CmpIPredicate.eq, lane, zero_i32)

        def load_bf16x4(resource, element_index):
            dwords = buffer_ops.buffer_load(
                resource,
                element_index // arith.constant(2, type=i32),
                vec_width=2,
                dtype=i32,
            )
            return vector.bitcast(vec4_bf16, dwords)

        def load_row_fp8(resource, element_index):
            """One lane's whole slice of a weight row, as three FP8 dwords."""
            return buffer_ops.buffer_load(
                resource,
                element_index // arith.constant(4, type=i32),
                vec_width=_K_DWORDS_PER_LANE,
                dtype=i32,
                cache_modifier=weight_cache_modifier,
            )

        def unpack_fp8x4(packed, dword_index):
            dword = vector.extract(
                _raw(packed),
                static_position=[dword_index],
                dynamic_position=[],
            )
            weight_lo = cvt_pk_f32_fp8(
                res=vec2_f32,
                src=dword,
                word_sel=False,
            )
            weight_hi = cvt_pk_f32_fp8(
                res=vec2_f32,
                src=dword,
                word_sel=True,
            )
            return weight_lo.shuffle(weight_hi, [0, 1, 2, 3])

        def wave_reduce_add(value):
            reduced = _raw(value)
            for offset in (32, 16, 8, 4, 2, 1):
                peer = _raw(
                    ArithValue(reduced).shuffle_xor(
                        arith.constant(offset, type=i32),
                        arith.constant(_WAVE_SIZE, type=i32),
                    )
                )
                reduced = arith_dialect.AddFOp(
                    reduced,
                    peer,
                    fastmath=fm_fast,
                ).result
            return reduced

        def wave_reduce_max(value):
            reduced = _raw(value)
            for offset in (32, 16, 8, 4, 2, 1):
                peer = _raw(
                    ArithValue(reduced).shuffle_xor(
                        arith.constant(offset, type=i32),
                        arith.constant(_WAVE_SIZE, type=i32),
                    )
                )
                reduced = arith_dialect.MaximumFOp(reduced, peer).result
            return reduced

        def round_to_fp8(scaled):
            """Round a vector of four FP32 lanes through E4M3 and back."""
            lanes = [
                vector.extract(
                    _raw(scaled),
                    static_position=[index],
                    dynamic_position=[],
                )
                for index in range_constexpr(_ELEMENTS_PER_LOAD)
            ]
            packed = cvt_pk_fp8_f32(
                res=i32,
                src_a=lanes[0],
                src_b=lanes[1],
                old=zero_i32,
                word_sel=False,
            )
            packed = cvt_pk_fp8_f32(
                res=i32,
                src_a=lanes[2],
                src_b=lanes[3],
                old=packed,
                word_sel=True,
            )
            rounded_lo = cvt_pk_f32_fp8(
                res=vec2_f32,
                src=packed,
                word_sel=False,
            )
            rounded_hi = cvt_pk_f32_fp8(
                res=vec2_f32,
                src=packed,
                word_sel=True,
            )
            return rounded_lo.shuffle(rounded_hi, [0, 1, 2, 3])

        # ── Stage the whole [M, 768] activation block in LDS ──
        for load_iteration in range_constexpr(load_iterations):
            vector_index = tid + arith.constant(
                load_iteration * block_threads, type=i32
            )
            in_range = arith.cmpi(
                CmpIPredicate.ult,
                vector_index,
                arith.constant(activation_vectors, type=i32),
            )
            load_if = scf.IfOp(in_range)
            with ir.InsertionPoint(load_if.then_block):
                element_index = vector_index * arith.constant(
                    _ELEMENTS_PER_LOAD, type=i32
                )
                fx.ptr_store(
                    load_bf16x4(activation_rsrc, element_index),
                    activated_lds + element_index,
                )
                scf.YieldOp([])
        gpu.barrier()

        # ── Per-token amax, one token per pass over the first three waves ──
        quant_thread = arith.cmpi(
            CmpIPredicate.ult,
            tid,
            arith.constant(_VECS_PER_TOKEN, type=i32),
        )
        for token_index in range_constexpr(num_tokens):
            token_base = arith.constant(
                token_index * _SHARED_INTERMEDIATE_SIZE, type=i32
            )
            local_if = scf.IfOp(quant_thread, results_=[f32], has_else=True)
            with ir.InsertionPoint(local_if.then_block):
                element_index = token_base + tid * arith.constant(
                    _ELEMENTS_PER_LOAD, type=i32
                )
                staged = fx.ptr_load(
                    activated_lds + element_index,
                    result_type=vec4_bf16,
                )
                widened = ArithValue(staged).extf(vec4_f32)
                local_amax = None
                for index in range_constexpr(_ELEMENTS_PER_LOAD):
                    magnitude = math_dialect.absf(
                        vector.extract(
                            _raw(widened),
                            static_position=[index],
                            dynamic_position=[],
                        )
                    )
                    local_amax = (
                        magnitude
                        if local_amax is None
                        else arith_dialect.MaximumFOp(local_amax, magnitude).result
                    )
                scf.YieldOp([local_amax])
            with ir.InsertionPoint(local_if.else_block):
                scf.YieldOp([zero_f32])
            wave_amax = wave_reduce_max(local_if.results[0])
            store_if = scf.IfOp(
                arith.andi(
                    is_lane_zero,
                    arith.cmpi(
                        CmpIPredicate.ult,
                        wave,
                        arith.constant(_QUANT_WAVES, type=i32),
                    ),
                )
            )
            with ir.InsertionPoint(store_if.then_block):
                fx.ptr_store(
                    wave_amax,
                    amax_lds
                    + arith.constant(token_index * _QUANT_WAVES, type=i32)
                    + wave,
                )
                scf.YieldOp([])
        gpu.barrier()

        # ── One thread per token turns its wave maxima into the FP8 scale ──
        scale_if = scf.IfOp(
            arith.cmpi(
                CmpIPredicate.ult,
                tid,
                arith.constant(num_tokens, type=i32),
            )
        )
        with ir.InsertionPoint(scale_if.then_block):
            amax_base = tid * arith.constant(_QUANT_WAVES, type=i32)
            token_amax = fx.ptr_load(amax_lds + amax_base, result_type=f32)
            for wave_index in range_constexpr(1, _QUANT_WAVES):
                peer = fx.ptr_load(
                    amax_lds + amax_base + arith.constant(wave_index, type=i32),
                    result_type=f32,
                )
                token_amax = arith_dialect.MaximumFOp(
                    _raw(token_amax),
                    _raw(peer),
                ).result
            scaled = arith_dialect.MulFOp(
                _raw(token_amax),
                arith.constant(1.0 / _FP8_MAX, type=f32),
                fastmath=fm_fast,
            ).result
            # An all-zero activation row would otherwise divide by zero; a
            # unit scale keeps its quantized form zero.
            fx.ptr_store(
                arith.select(
                    arith.cmpf(CmpFPredicate.OGT, _raw(token_amax), zero_f32),
                    scaled,
                    one_f32,
                ),
                scales_lds + tid,
            )
            scf.YieldOp([])
        gpu.barrier()

        # ── Round the staged activation in place; E4M3 is exact in BF16 ──
        for token_index in range_constexpr(num_tokens):
            token_base = arith.constant(
                token_index * _SHARED_INTERMEDIATE_SIZE, type=i32
            )
            quant_if = scf.IfOp(quant_thread)
            with ir.InsertionPoint(quant_if.then_block):
                token_scale = fx.ptr_load(
                    scales_lds + arith.constant(token_index, type=i32),
                    result_type=f32,
                )
                inverse = arith_dialect.DivFOp(
                    one_f32,
                    _raw(token_scale),
                    fastmath=fm_fast,
                ).result
                element_index = token_base + tid * arith.constant(
                    _ELEMENTS_PER_LOAD, type=i32
                )
                staged = fx.ptr_load(
                    activated_lds + element_index,
                    result_type=vec4_bf16,
                )
                scaled = ArithValue(staged).extf(vec4_f32) * ArithValue(inverse)
                fx.ptr_store(
                    arith.trunc_f(vec4_bf16, _raw(round_to_fp8(scaled))),
                    activated_lds + element_index,
                )
                scf.YieldOp([])
        gpu.barrier()

        # ── Projection: one weight load feeds every token's accumulator ──
        first_group = (
            ArithValue(gpu.block_idx.x) * arith.constant(waves_per_block, type=i32)
            + wave
        )
        for persistent_index in range_constexpr(persistent_iterations):
            group = first_group + arith.constant(
                persistent_index * groups_per_grid,
                type=i32,
            )
            row_base = group * arith.constant(rows_per_wave, type=i32)
            for row_offset in range_constexpr(rows_per_wave):
                row = row_base + arith.constant(row_offset, type=i32)
                row_in_range = arith.cmpi(
                    CmpIPredicate.ult,
                    row,
                    arith.constant(_HIDDEN_SIZE, type=i32),
                )
                row_if = scf.IfOp(row_in_range)
                with ir.InsertionPoint(row_if.then_block):
                    row_scale = buffer_ops.buffer_load(
                        scale_rsrc,
                        row,
                        vec_width=1,
                        dtype=f32,
                    )
                    accumulators = [
                        ArithValue(zero_f32) for _ in range_constexpr(num_tokens)
                    ]
                    # A lane owns a contiguous 12-element slice of the row, so
                    # the whole wave covers K in one coalesced 768-byte read.
                    lane_base = lane * arith.constant(
                        _K_ELEMENTS_PER_LANE, type=i32
                    )
                    packed_weight = load_row_fp8(
                        weight_rsrc,
                        row * arith.constant(_SHARED_INTERMEDIATE_SIZE, type=i32)
                        + lane_base,
                    )
                    for dword_index in range_constexpr(_K_DWORDS_PER_LANE):
                        weight_f32 = unpack_fp8x4(packed_weight, dword_index)
                        k_element = lane_base + arith.constant(
                            dword_index * _ELEMENTS_PER_LOAD, type=i32
                        )
                        for token_index in range_constexpr(num_tokens):
                            quantized_bf16 = fx.ptr_load(
                                activated_lds
                                + arith.constant(
                                    token_index * _SHARED_INTERMEDIATE_SIZE,
                                    type=i32,
                                )
                                + k_element,
                                result_type=vec4_bf16,
                            )
                            quantized_f32 = ArithValue(quantized_bf16).extf(vec4_f32)
                            accumulators[token_index] = accumulators[
                                token_index
                            ] + (quantized_f32 * weight_f32).reduce(
                                ReductionOp.ADD, fastmath=fm_fast
                            )

                    for token_index in range_constexpr(num_tokens):
                        token_scale = fx.ptr_load(
                            scales_lds + arith.constant(token_index, type=i32),
                            result_type=f32,
                        )
                        reduced = (
                            ArithValue(wave_reduce_add(accumulators[token_index]))
                            * ArithValue(row_scale)
                            * ArithValue(token_scale)
                        )
                        write_if = scf.IfOp(is_lane_zero)
                        with ir.InsertionPoint(write_if.then_block):
                            buffer_ops.buffer_store(
                                arith.trunc_f(T.bf16, _raw(reduced)),
                                output_rsrc,
                                arith.constant(token_index * _HIDDEN_SIZE, type=i32)
                                + row,
                            )
                            scf.YieldOp([])
                    scf.YieldOp([])

    @flyc.jit
    def launch_shared_down_ptpc(
        activation: fx.Pointer,
        weight: fx.Pointer,
        weight_scale: fx.Pointer,
        output: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ctx = CompilationContext.get_current()
        if const_expr(waves_per_eu > 0):
            for operation in ctx.gpu_module_body.operations:
                if (
                    hasattr(operation, "attributes")
                    and operation.OPERATION_NAME == "gpu.func"
                ):
                    operation.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32,
                        int(waves_per_eu),
                    )
        shared_down_ptpc_kernel(
            activation,
            weight,
            weight_scale,
            output,
        ).launch(
            grid=(cu_count, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_shared_down_ptpc.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_shared_down_ptpc
