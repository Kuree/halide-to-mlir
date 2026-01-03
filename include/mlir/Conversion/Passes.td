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

#endif // HALIDE_CONVERSION_PASSES
