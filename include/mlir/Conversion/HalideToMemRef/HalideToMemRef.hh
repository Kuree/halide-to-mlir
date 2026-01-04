#ifndef MLIR_CONVERSION_HALIDETOMEMREF_HALIDETOMEMREF_H
#define MLIR_CONVERSION_HALIDETOMEMREF_HALIDETOMEMREF_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTHALIDETOMEMREF
#include "mlir/Conversion/Conversions.h.inc"

namespace halide {

/// Populate patterns for converting Halide handle types and operations to
/// MemRef.
void populateHalideToMemRefConversionPatterns(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns);

/// Populate patterns for converting Halide buffer helper intrinsic calls to
/// MemRef ops.
void populateHalideBufferHelperConversionPatterns(TypeConverter &typeConverter,
                                                  RewritePatternSet &patterns);

} // namespace halide
} // namespace mlir

#endif // MLIR_CONVERSION_HALIDETOMEMREF_HALIDETOMEMREF_H
