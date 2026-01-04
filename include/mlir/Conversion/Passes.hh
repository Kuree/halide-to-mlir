#ifndef HALIDE_TO_MLIR_CONVERSION_PASSES_HH
#define HALIDE_TO_MLIR_CONVERSION_PASSES_HH

#include "mlir/Conversion/HalideToArith/HalideToArith.hh"
#include "mlir/Conversion/HalideToMath/HalideToMath.hh"
#include "mlir/Conversion/HalideToSCF/HalideToSCF.hh"

namespace mlir::halide {
/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/Conversions.h.inc"
} // namespace mlir::halide

#endif // HALIDE_TO_MLIR_CONVERSION_PASSES_HH
