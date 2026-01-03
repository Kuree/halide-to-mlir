add_mlir_conversion_library(MLIRHalideToArith
        HalideToArith.cc
        DEPENDS
        MLIRHalideConversionPassIncGen
        LINK_LIBS PUBLIC
        MLIRArithDialect
        MLIRHalideDialect
        MLIRIR
        MLIRPass
        MLIRTransforms
)
