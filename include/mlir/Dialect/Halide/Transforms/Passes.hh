#ifndef HALIDE_TO_MLIR_PASSES_HH
#define HALIDE_TO_MLIR_PASSES_HH

#include "mlir/Pass/Pass.h"

namespace mlir::halide {
#define GEN_PASS_DECL
#include "mlir/Dialect/Halide/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Halide/Transforms/Passes.h.inc"

} // namespace mlir::bf


#endif // HALIDE_TO_MLIR_PASSES_HH
