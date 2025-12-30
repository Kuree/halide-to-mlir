add_mlir_dialect_library(MLIRHalideTransforms
        Let2Reg.cc
        DEPENDS
        HalidePassesIncGen
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransforms
        MLIRHalideDialect
)
