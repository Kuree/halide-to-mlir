set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Conversions.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(MLIRHalideConversionPassIncGen)
