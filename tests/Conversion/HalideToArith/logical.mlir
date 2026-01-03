// RUN: halide-opt --convert-halide-to-arith %s | FileCheck %s

// CHECK-LABEL:  @test_logical_ops
func.func @test_logical_ops(%arg0: i1, %arg1: i1) -> i1 {
  // CHECK: %[[AND:.*]] = arith.andi %arg0, %arg1 : i1
  %0 = halide.and %arg0, %arg1 : i1

  // CHECK: %[[OR:.*]] = arith.ori %[[AND]], %arg1 : i1
  %1 = halide.or %0, %arg1 : i1

  // CHECK: %[[C:.*]] = arith.constant true
  // CHECK: %[[NOT:.*]] = arith.xori %[[OR]], %[[C]] : i1
  %2 = halide.not %1 : i1

  // CHECK: return %[[NOT]] : i1
  return %2 : i1
}

// CHECK-LABEL: @test_bitwise_int
func.func @test_bitwise_int(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[AND:.*]] = arith.andi %arg0, %arg1 : i32
  %0 = halide.and %arg0, %arg1 : i32

  // CHECK: %[[OR:.*]] = arith.ori %[[AND]], %arg1 : i32
  %1 = halide.or %0, %arg1 : i32

  // CHECK: %[[C:.*]] = arith.constant -1 : i32
  // CHECK: %[[NOT:.*]] = arith.xori %[[OR]], %[[C]] : i32
  %2 = halide.not %1 : i32

  // CHECK: return %[[NOT]] : i32
  return %2 : i32
}
