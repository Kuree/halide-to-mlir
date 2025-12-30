// RUN: halide-opt --allow-unregistered-dialect --split-input-file --let2reg %s | FileCheck %s

func.func @var() {
  %c0 = arith.constant 0: i32
  halide.let "a" = %c0 : i32 {
    %0 = halide.variable "a" : i32
    "op.op"(%0) : (i32) -> ()
  }
  return
}

// CHECK-LABEL: @var
// CHECK: %[[ZERO:.*]] = arith.constant 0
// CHECK: "op.op"(%[[ZERO]])


// -----

func.func @shadow() {
  %c0 = arith.constant 0: i32
  %c1 = arith.constant 1: i32
  halide.let "a" = %c0 : i32 {
    %0 = halide.variable "a" : i32
    "op.op0"(%0) : (i32) -> ()
    halide.let "a" = %c1 : i32 {
      %1 = halide.variable "a" : i32
      "op.op1"(%1) : (i32) -> ()
    }
  }
  return
}

// CHECK-LABEL: @shadow
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1
// CHECK: "op.op0"(%[[ZERO]])
// CHECK-NEXT: "op.op1"(%[[ONE]])
