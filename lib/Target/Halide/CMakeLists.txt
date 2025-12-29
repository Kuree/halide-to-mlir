add_mlir_library(MLIRHalideToMLIR ImportHalide.cc
        LINK_LIBS MLIRHalideDialect Halide::Halide MLIRArithDialect
)
