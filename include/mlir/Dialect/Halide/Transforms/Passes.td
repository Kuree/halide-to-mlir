#ifndef HALIDE_PASSES
#define HALIDE_PASSES

include "mlir/Pass/PassBase.td"

def Let2Reg : Pass<"let2reg"> {
  let summary = "Promote let statement to registers";
}

#endif // HALIDE_PASSES