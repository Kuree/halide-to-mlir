add_llvm_executable(halide-opt halide-opt.cc)
llvm_update_compile_flags(halide-opt)

target_link_libraries(
        halide-opt
        PRIVATE MLIROptLib
        MLIRHalideDialect
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLLVMDialect
        MLIRSCFDialect
        MLIRHalideTransforms
        MLIRHalideToArith
        MLIRHalideToMath
        MLIRHalideToSCF)
