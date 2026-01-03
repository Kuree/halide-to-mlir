// RUN: halide-opt --convert-halide-to-arith %s | FileCheck %s

// CHECK-LABEL: @test_int_arithmetic
func.func @test_int_arithmetic(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[ADD:.*]] = arith.addi %arg0, %arg1 : i32
  %0 = halide.add %arg0, %arg1 : i32

  // CHECK: %[[SUB:.*]] = arith.subi %[[ADD]], %arg1 : i32
  %1 = halide.sub %0, %arg1 : i32

  // CHECK: %[[MUL:.*]] = arith.muli %[[SUB]], %arg0 : i32
  %2 = halide.mul %1, %arg0 : i32

  // CHECK: %[[DIV:.*]] = arith.divsi %[[MUL]], %arg1 : i32
  %3 = halide.div %2, %arg1 : i32

  // CHECK: %[[MOD:.*]] = arith.remsi %[[DIV]], %arg0 : i32
  %4 = halide.mod %3, %arg0 : i32

  // CHECK: return %[[MOD]] : i32
  return %4 : i32
}

// CHECK-LABEL: @test_float_arithmetic
func.func @test_float_arithmetic(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: %[[ADD:.*]] = arith.addf %arg0, %arg1 : f32
  %0 = halide.add %arg0, %arg1 : f32

  // CHECK: %[[SUB:.*]] = arith.subf %[[ADD]], %arg1 : f32
  %1 = halide.sub %0, %arg1 : f32

  // CHECK: %[[MUL:.*]] = arith.mulf %[[SUB]], %arg0 : f32
  %2 = halide.mul %1, %arg0 : f32

  // CHECK: %[[DIV:.*]] = arith.divf %[[MUL]], %arg1 :  f32
  %3 = halide.div %2, %arg1 : f32

  // CHECK: %[[MOD:.*]] = arith.remf %[[DIV]], %arg0 : f32
  %4 = halide.mod %3, %arg0 : f32

  // CHECK: return %[[MOD]] : f32
  return %4 : f32
}

// CHECK-LABEL: @test_min_max_int
func.func @test_min_max_int(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[MIN:.*]] = arith.minsi %arg0, %arg1 : i32
  %0 = halide.min %arg0, %arg1 : i32

  // CHECK: %[[MAX:.*]] = arith.maxsi %[[MIN]], %arg1 : i32
  %1 = halide.max %0, %arg1 : i32

  // CHECK: return %[[MAX]] : i32
  return %1 : i32
}

// CHECK-LABEL: @test_min_max_float
func.func @test_min_max_float(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: %[[MIN:.*]] = arith.minimumf %arg0, %arg1 : f32
  %0 = halide.min %arg0, %arg1 :  f32

  // CHECK: %[[MAX:.*]] = arith.maximumf %[[MIN]], %arg1 : f32
  %1 = halide.max %0, %arg1 : f32

  // CHECK: return %[[MAX]] : f32
  return %1 : f32
}
