#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Halide/IR/HalideOps.hh"
#include "mlir/Dialect/Halide/Transforms/Passes.hh"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/Passes.hh"

int main(int argc, char *argv[]) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::halide::HalideDialect,
                    mlir::scf::SCFDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect, mlir::math::MathDialect>();
    mlir::MlirOptMainConfig config;
    mlir::registerTransformsPasses();
    mlir::halide::registerConversionPasses();
    mlir::halide::registerPasses();
    return mlir::failed(
        mlir::MlirOptMain(argc, argv, "Halide Optimization Tool", registry));
}
