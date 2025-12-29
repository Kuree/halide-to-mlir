#ifndef MLIR_TARGET_HALIDE_IMPORT_HALIDE_HH
#define MLIR_TARGET_HALIDE_IMPORT_HALIDE_HH

#include "mlir/IR/BuiltinOps.h"

#include "Halide.h"

namespace mlir::halide {
OwningOpRef<ModuleOp>
importHalide(Halide::Func func, MLIRContext *context,
             const Halide::Target &target = Halide::get_host_target());
}

#endif // MLIR_TARGET_HALIDE_IMPORT_HALIDE_HH
