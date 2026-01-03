#ifndef MLIR_CONVERSION_HALIDETOARITH_HALIDETOARITH_H
#define MLIR_CONVERSION_HALIDETOARITH_HALIDETOARITH_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTHALIDETOARITH
#include "mlir/Conversion/Conversions.h.inc"

namespace halide {

/// Populate patterns to convert Halide operations to Arith operations.
void populateHalideToArithConversionPatterns(RewritePatternSet &patterns);

} // namespace halide

} // namespace mlir

#endif // MLIR_CONVERSION_HALIDETOARITH_HALIDETOARITH_H