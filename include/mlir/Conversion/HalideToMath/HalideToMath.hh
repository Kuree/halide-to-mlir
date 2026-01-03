#ifndef MLIR_CONVERSION_HALIDETOMTH_HALIDETOMTH_H
#define MLIR_CONVERSION_HALIDETOMTH_HALIDETOMTH_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTHALIDETOMATH
#include "mlir/Conversion/Conversions.h.inc"

namespace halide {

/// Populate patterns to convert Halide CallOp operations to Math dialect
/// operations.
void populateHalideCallToMathConversionPatterns(RewritePatternSet &patterns);

} // namespace halide

} // namespace mlir

#endif // MLIR_CONVERSION_HALIDETOMTH_HALIDETOMTH_H
