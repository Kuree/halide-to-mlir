#ifndef HALIDE_CONVERSION_PASSES
#define HALIDE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertHalideToArith : Pass<"convert-halide-to-arith"> {
  let summary = "Convert Halide dialect operations to Arith dialect";
  let description = [{
    This pass converts Halide dialect operations to equivalent operations in the
    Arith dialect.  It handles arithmetic operations, comparison operations, and
    logical operations.

    Supported conversions:
    - Arithmetic: add, sub, mul, div, mod, min, max
    - Comparisons: eq, ne, lt, le, gt, ge
    - Logical: and, or, not
    - Other: select, cast
  }];

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def ConvertHalideToMath : Pass<"convert-halide-to-math"> {
  let summary = "Convert Halide math intrinsic calls to Math dialect operations";
  let description = [{
    This pass converts Halide CallOp operations that represent mathematical
    intrinsics (sqrt, sin, cos, exp, log, etc.) to their corresponding
    operations in MLIR's Math dialect.

    Only PureIntrinsic and Intrinsic call types are considered for conversion.
    The pass handles common mathematical functions like:
    - Trigonometric:  sin, cos, tan, asin, acos, atan, atan2
    - Hyperbolic: sinh, cosh, tanh
    - Exponential/Logarithmic: exp, exp2, log, log2, log10
    - Power: pow, sqrt, rsqrt
    - Rounding: floor, ceil, round, trunc
    - Other: abs, erf, fma, copysign
  }];
  let dependentDialects = ["mlir::math::MathDialect"];
}

def ConvertHalideToSCF : Pass<"convert-halide-to-scf"> {
  let summary = "Convert Halide control flow operations to SCF dialect";
  let description = [{
    This pass converts Halide control flow operations to their corresponding
    operations in MLIR's SCF (Structured Control Flow) dialect.

    Note: halide.let operations are not converted by this pass and should
    be handled separately by `let2reg`.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::arith::ArithDialect"];
}

def ConvertHalideToMemRef : Pass<"convert-halide-to-memref"> {
  let summary = "Convert Halide handle types and load/store operations to MemRef dialect";
  let description = [{
    This pass converts Halide handle types and memory operations to their
    corresponding operations in MLIR's MemRef dialect.

    Supported conversions:
    - halide.handle type -> memref<...> type with dimension information
    - halide. load with linear index -> memref. load with multi-dimensional indices
    - halide.store with linear index -> memref.store with multi-dimensional indices
    - halide.allocate -> memref.alloc/memref.alloca
  }];
  let dependentDialects = ["mlir::memref::MemRefDialect", "mlir::arith::ArithDialect"];
}

#endif // HALIDE_CONVERSION_PASSES
