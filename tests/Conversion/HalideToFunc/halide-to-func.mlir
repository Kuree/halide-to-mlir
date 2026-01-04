// RUN:  halide-opt %s -convert-halide-to-func -split-input-file | FileCheck %s

// CHECK: llvm.mlir.global private constant @__assert_msg("halide_error_buffer_argument_is_null")
// CHECK: func.func private @puts(!llvm.ptr) -> i32
// CHECK: func.func private @abort()

// CHECK-LABEL: func.func @test_assert_with_call
func.func @test_assert_with_call(%arg0: i1, %arg1: !llvm.ptr) {
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[NOT_COND:.*]] = arith.xori %arg0, %[[TRUE]] : i1
    // CHECK: scf.if %[[NOT_COND]] {
    // CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @__assert_msg : !llvm.ptr
    // CHECK:   func.call @puts(%[[ADDR]]) : (!llvm.ptr) -> i32
    // CHECK:   func.call @abort() : () -> ()
    // CHECK: }
    halide.assert %arg0 {
        %0 = halide.call Extern "halide_error_buffer_argument_is_null"(%arg1) : (!llvm.ptr) -> i32
    }
    return
}

// -----

// CHECK-LABEL: func.func @test_duplicated_message
func.func @test_duplicated_message(%arg0: i1, %arg1: !llvm.ptr) {
    // CHECK: llvm.mlir.addressof @__assert_msg : !llvm.ptr
    // CHECK: llvm.mlir.addressof @__assert_msg_0 : !llvm.ptr
    halide.assert %arg0 {
        %0 = halide.call Extern "halide_error_buffer_argument_is_null"(%arg1) : (!llvm.ptr) -> i32
    }
    halide.assert %arg0 {
        %0 = halide.call Extern "halide_error_buffer_argument_is_null"(%arg1) : (!llvm.ptr) -> i32
    }
    return
}
