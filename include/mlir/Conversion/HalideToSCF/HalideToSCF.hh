#ifndef MLIR_CONVERSION_HALIDETOSCF_HALIDETOSCF_H
#define MLIR_CONVERSION_HALIDETOSCF_HALIDETOSCF_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTHALIDETOSCF
#include "mlir/Conversion/Conversions.h.inc"

namespace halide {

/// Populate patterns to convert Halide control flow operations to SCF dialect
/// operations.
void populateHalideToSCFConversionPatterns(RewritePatternSet &patterns);

} // namespace halide

} // namespace mlir

#endif // MLIR_CONVERSION_HALIDETOSCF_HALIDETOSCF_H
