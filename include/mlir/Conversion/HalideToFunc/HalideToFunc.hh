#ifndef MLIR_CONVERSION_HALIDETOFUNC_HALIDETOFUNC_H
#define MLIR_CONVERSION_HALIDETOFUNC_HALIDETOFUNC_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTHALIDETOFUNC
#include "mlir/Conversion/Conversions.h.inc"

namespace halide {

/// Populate patterns to convert Halide AssertOp to func. call operations.
void populateHalideToFuncConversionPatterns(RewritePatternSet &patterns);

} // namespace halide

} // namespace mlir

#endif // MLIR_CONVERSION_HALIDETOFUNC_HALIDETOFUNC_H
