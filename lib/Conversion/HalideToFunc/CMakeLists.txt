add_mlir_conversion_library(MLIRHalideToFunc
        HalideToFunc.cc
        DEPENDS
        MLIRHalideConversionPassIncGen
        LINK_LIBS PUBLIC
        MLIRSCFDialect
        MLIRFuncDialect
        MLIRHalideDialect
        MLIRLLVMDialect
        MLIRIR
        MLIRPass
        MLIRTransforms
)
