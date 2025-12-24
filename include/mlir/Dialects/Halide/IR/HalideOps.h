#ifndef HALIDE_OPS_H
#define HALIDE_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialects/Halide/IR/HalideOpsEnums.h.inc"

#include "mlir/Dialects/Halide/IR/HalideOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialects/Halide/IR/HalideOps.h.inc"

#endif // HALIDE_OPS_H
