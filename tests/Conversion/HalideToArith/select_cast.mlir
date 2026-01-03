// RUN:  halide-opt --convert-halide-to-arith %s | FileCheck %s

// CHECK-LABEL: @test_select
func.func @test_select(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK: %[[SEL:.*]] = arith.select %arg0, %arg1, %arg2 : i32
  %0 = halide.select %arg0, %arg1, %arg2 : i32

  // CHECK: return %[[SEL]] : i32
  return %0 : i32
}

// CHECK-LABEL: @test_select_float
func.func @test_select_float(%arg0: i1, %arg1: f32, %arg2: f32) -> f32 {
  // CHECK: %[[SEL:.*]] = arith.select %arg0, %arg1, %arg2 : f32
  %0 = halide.select %arg0, %arg1, %arg2 : f32

  // CHECK: return %[[SEL]] : f32
  return %0 : f32
}

// CHECK-LABEL: @test_cast_int_extend
func.func @test_cast_int_extend(%arg0: i16) -> i32 {
  // CHECK: %[[EXT:.*]] = arith.extsi %arg0 : i16 to i32
  %0 = halide.cast %arg0 : i16 to i32

  // CHECK: return %[[EXT]] :  i32
  return %0 : i32
}

// CHECK-LABEL: @test_cast_int_trunc
func.func @test_cast_int_trunc(%arg0: i32) -> i16 {
  // CHECK: %[[TRUNC:.*]] = arith.trunci %arg0 : i32 to i16
  %0 = halide.cast %arg0 : i32 to i16

  // CHECK: return %[[TRUNC]] : i16
  return %0 : i16
}

// CHECK-LABEL:  @test_cast_float_extend
func.func @test_cast_float_extend(%arg0: f16) -> f32 {
  // CHECK: %[[EXT:.*]] = arith.extf %arg0 : f16 to f32
  %0 = halide.cast %arg0 : f16 to f32

  // CHECK: return %[[EXT]] :  f32
  return %0 : f32
}

// CHECK-LABEL: @test_cast_float_trunc
func.func @test_cast_float_trunc(%arg0: f32) -> f16 {
  // CHECK: %[[TRUNC:.*]] = arith.truncf %arg0 : f32 to f16
  %0 = halide.cast %arg0 : f32 to f16

  // CHECK: return %[[TRUNC]] :  f16
  return %0 : f16
}

// CHECK-LABEL: @test_cast_int_to_float
func.func @test_cast_int_to_float(%arg0: i32) -> f32 {
  // CHECK: %[[CONV:.*]] = arith.sitofp %arg0 : i32 to f32
  %0 = halide.cast %arg0 : i32 to f32

  // CHECK: return %[[CONV]] : f32
  return %0 : f32
}

// CHECK-LABEL: @test_cast_float_to_int
func.func @test_cast_float_to_int(%arg0: f32) -> i32 {
  // CHECK: %[[CONV:.*]] = arith.fptosi %arg0 : f32 to i32
  %0 = halide.cast %arg0 : f32 to i32

  // CHECK: return %[[CONV]] : i32
  return %0 : i32
}
