#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/Dialect/Halide/Transforms/Passes.hh"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char *argv[]) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::halide::HalideDialect,
                    mlir::scf::SCFDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect>();
    mlir::MlirOptMainConfig config;
    mlir::registerTransformsPasses();
    mlir::halide::registerPasses();
    return mlir::failed(
        mlir::MlirOptMain(argc, argv, "Halide Optimization Tool", registry));
}
