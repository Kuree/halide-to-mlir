add_mlir_conversion_library(MLIRHalideToSCF
        HalideToSCF.cc
        DEPENDS
        MLIRHalideConversionPassIncGen
        LINK_LIBS PUBLIC
        MLIRHalideDialect
        MLIRSCFDialect
        MLIRArithDialect
        MLIRIR
        MLIRPass
        MLIRTransforms
        MLIRSupport
)
