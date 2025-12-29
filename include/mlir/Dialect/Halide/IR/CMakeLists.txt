set(LLVM_TARGET_DEFINITIONS HalideOps.td)

mlir_tablegen(HalideOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(HalideOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(HalideOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(HalideOpsDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(HalideOps.h.inc -gen-op-decls)
mlir_tablegen(HalideOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(HalideMLIRIncGen)
