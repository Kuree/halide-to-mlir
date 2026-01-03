add_mlir_conversion_library(MLIRHalideToMath
        HalideToMath.cc
        DEPENDS
        MLIRHalideConversionPassIncGen
        LINK_LIBS PUBLIC
        MLIRMathDialect
        MLIRHalideDialect
        MLIRIR
        MLIRPass
        MLIRTransforms
)
