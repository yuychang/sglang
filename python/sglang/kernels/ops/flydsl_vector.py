# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Vector helpers for the Kimi-K3 FlyDSL kernels.

ROCm/aiter removed ``aiter.ops.flydsl.kernels.vector`` in #5303, and FlyDSL
removed ``flydsl.expr.vector``. Callers now use
``flydsl._mlir.dialects.vector`` and unwrap DSL values with ``as_ir_value``.
``bitcast``, ``extract``, and ``from_elements`` do that unwrap. ``shuffle`` and
``CombiningKind`` are the raw dialect.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector
from flydsl._mlir.dialects.vector import CombiningKind as CombiningKind
from flydsl._mlir.dialects.vector import shuffle as shuffle
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import as_ir_value

__all__ = [
    "CombiningKind",
    "bitcast",
    "extract",
    "from_elements",
    "shuffle",
]


def _as_index_ir_value(value):
    if isinstance(value, int):
        from flydsl.expr import arith as arith_ext

        return arith_ext.constant(value, index=True)
    converted = as_ir_value(value)
    if isinstance(converted.type, ir.IntegerType):
        from flydsl._mlir.dialects import arith as std_arith

        converted = std_arith.IndexCastOp(ir.IndexType.get(), converted).result
    return converted


@dsl_loc_tracing
def from_elements(*args, **kwargs):
    """Build a vector from scalars, unwrapping DSL values."""
    if len(args) >= 2:
        args = list(args)
        elements = args[1]
        if isinstance(elements, (list, tuple)):
            args[1] = [as_ir_value(value) for value in elements]
        return _vector.from_elements(*args, **kwargs)
    return _vector.from_elements(*args, **kwargs)


@dsl_loc_tracing
def extract(source, static_position=None, dynamic_position=None):
    """Extract vector elements, filling missing dynamic-index sentinels."""
    if static_position is None:
        static_position = []
    if dynamic_position is None:
        dynamic_position = []
    dynamic_position = [_as_index_ir_value(index) for index in dynamic_position]
    if dynamic_position and len(static_position) < len(dynamic_position):
        dynamic_size = ir.ShapedType.get_dynamic_size()
        missing = len(dynamic_position) - len(static_position)
        static_position = list(static_position) + [dynamic_size] * missing
    return _vector.extract(
        as_ir_value(source),
        dynamic_position=dynamic_position,
        static_position=static_position,
    )


@dsl_loc_tracing
def bitcast(result_type, source):
    """Bitcast a vector, unwrapping its DSL value."""
    return _vector.bitcast(result_type, as_ir_value(source))
