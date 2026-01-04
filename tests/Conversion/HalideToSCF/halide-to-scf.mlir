// RUN: halide-opt %s -convert-halide-to-scf --allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// ForOp Conversion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_for_loop
func.func @test_for_loop() {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C10:.*]] = arith.constant 10 : i32
  // CHECK-DAG: %[[UB:.*]] = arith.addi %[[C0]], %[[C10]] : i32
  // CHECK-DAG: %[[C0_IDX:.*]] = arith.index_cast %[[C0]] :  i32 to index
  // CHECK-DAG: %[[UB_IDX:.*]] = arith.index_cast %[[UB]] : i32 to index
  // CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[IV:.*]] = %[[C0_IDX]] to %[[UB_IDX]] step %[[STEP]]
  // CHECK:   %[[IV_CAST:.*]] = arith.index_cast %[[IV]] : index to i32
  // CHECK:   "op.op"(%[[IV_CAST]]) : (i32) -> ()
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  halide.for "i" = %c0 extent %c10 : i32
    type Serial device Host partition Auto {
    %0 = halide.variable "i" : i32
    "op.op"(%0) : (i32) -> ()
  }
  return
}

// CHECK-LABEL: @test_nested_for_loops
func.func @test_nested_for_loops() {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
  // CHECK-DAG: %[[C10:.*]] = arith.constant 10 : i32
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
  // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
  // CHECK:     "op.op"() : () -> ()
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %c10 = arith.constant 10 : i32

  halide.for "i" = %c0 extent %c10 : i32
    type Serial device Host partition Auto {
    halide.for "j" = %c0 extent %c5 : i32
      type Serial device Host partition Auto {
      "op.op"() : () -> ()
    }
  }
  return
}

// CHECK-LABEL: @test_for_i64_type
func.func @test_for_i64_type() {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
  // CHECK-DAG: %[[C100:.*]] = arith.constant 100 : i64
  // CHECK: %[[UB:.*]] = arith.addi %[[C0]], %[[C100]] : i64
  // CHECK-DAG: %[[LB_IDX:.*]] = arith.index_cast %[[C0]] : i64 to index
  // CHECK-DAG: %[[UB_IDX:.*]] = arith.index_cast %[[UB]] : i64 to index
  // CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
  // CHECK: scf.for %{{.*}} = %[[LB_IDX]] to %[[UB_IDX]] step %[[STEP]]
  %c0 = arith.constant 0 : i64
  %c100 = arith.constant 100 : i64
  halide.for "i" = %c0 extent %c100 : i64
    type Serial device Host partition Auto {
  }
  return
}


//===----------------------------------------------------------------------===//
// IfOp Conversion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_if_then
func.func @test_if_then(%cond: i1) {
  // CHECK: scf.if %arg0 {
  // CHECK:   "op.op"() : () -> ()
  // CHECK: }
  halide.if %cond {
    "op.op"() : () -> ()
  }
  return
}

// CHECK-LABEL: @test_if_then_else
func.func @test_if_then_else(%cond: i1) {
  // CHECK: scf.if %arg0 {
  // CHECK: } else {
  // CHECK: }
  // CHECK: return

  halide.if %cond {
    "op.op"() : () -> ()
  } else {
    "op.op"() : () -> ()
  }
  return
}

// CHECK-LABEL: @test_nested_if
func.func @test_nested_if(%cond1: i1, %cond2: i1) {
  // CHECK: scf.if %arg0 {
  // CHECK:    scf.if %arg1 {
  // CHECK:      "op.op"() : () -> ()
  // CHECK:   }
  // CHECK: }
  halide.if %cond1 {
    halide.if %cond2 {
      "op.op"() : () -> ()
    }
  }
  return
}

// CHECK-LABEL: @test_let_not_converted
func.func @test_let_not_converted(%val: i32) {
  // CHECK: halide.let "x" = %arg0
  halide.let "x" = %val : i32 {
    %c1 = arith.constant 1 : i32
    %add = arith.addi %val, %c1 : i32
  }
  return
}
