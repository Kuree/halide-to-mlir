add_mlir_conversion_library(MLIRHalideToMemRef
        HalideToMemRef.cc
        DEPENDS
        MLIRHalideConversionPassIncGen
        LINK_LIBS PUBLIC
        MLIRArithDialect
        MLIRFuncDialect
        MLIRHalideDialect
        MLIRMemRefDialect
        MLIRIR
        MLIRPass
        MLIRTransforms
)
