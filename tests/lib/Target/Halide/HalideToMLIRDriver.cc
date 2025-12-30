#include "HalideToMLIRDriver.hh"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/Halide/ImportHalide.hh"

void convertAndPrintHalideToMLIR(Halide::Func func) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::halide::HalideDialect, mlir::arith::ArithDialect,
                        mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
    if (auto mod = mlir::halide::importHalide(std::move(func), &context)) {
        (void)mlir::verify(*mod);
        mod->print(llvm::outs());
    }
}
