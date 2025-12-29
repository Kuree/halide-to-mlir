#include "mlir/Dialect/Halide/IR/HalideOps.hh"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

// Include TableGen generated enum definitions
#include "mlir/Dialect/Halide/IR/HalideOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::halide;

//===----------------------------------------------------------------------===//
// HalideDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Halide/IR/HalideOpsDialect.cpp.inc"

void HalideDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Halide/IR/HalideOps.cpp.inc"

        >();

    // Register types/attributes if any (enums are attributes)
}

//===----------------------------------------------------------------------===//
// Halide Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Halide/IR/HalideOps.cpp.inc"
