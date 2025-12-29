#include "HalideToMLIRDriver.hh"
#include "mlir/Target/Halide/ImportHalide.hh"

void convertAndPrintHalideToMLIR(Halide::Func func) {
    mlir::MLIRContext context;
    if (auto mod = mlir::halide::importHalide(std::move(func), &context)) {
        mod->print(llvm::outs());
    }
}
