// RUN:  halide-opt --convert-halide-to-arith %s | FileCheck %s

// CHECK-LABEL: @test_int_comparison
func.func @test_int_comparison(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK: %[[EQ:.*]] = arith.cmpi eq, %arg0, %arg1 : i32
  %0 = halide.eq %arg0, %arg1 : i32

  // CHECK: %[[NE:.*]] = arith.cmpi ne, %arg0, %arg1 : i32
  %1 = halide.ne %arg0, %arg1 : i32

  // CHECK: %[[LT:.*]] = arith.cmpi slt, %arg0, %arg1 : i32
  %2 = halide.lt %arg0, %arg1 : i32

  // CHECK: %[[LE:.*]] = arith.cmpi sle, %arg0, %arg1 :  i32
  %3 = halide.le %arg0, %arg1 : i32

  // CHECK: %[[GT:.*]] = arith.cmpi sgt, %arg0, %arg1 : i32
  %4 = halide.gt %arg0, %arg1 : i32

  // CHECK: %[[GE:.*]] = arith.cmpi sge, %arg0, %arg1 : i32
  %5 = halide.ge %arg0, %arg1 : i32

  // CHECK: return %[[GE]] : i1
  return %5 : i1
}

// CHECK-LABEL: @test_float_comparison
func.func @test_float_comparison(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK: %[[EQ:.*]] = arith.cmpf oeq, %arg0, %arg1 : f32
  %0 = halide.eq %arg0, %arg1 : f32

  // CHECK: %[[NE:.*]] = arith.cmpf one, %arg0, %arg1 : f32
  %1 = halide.ne %arg0, %arg1 : f32

  // CHECK: %[[LT:.*]] = arith.cmpf olt, %arg0, %arg1 : f32
  %2 = halide.lt %arg0, %arg1 : f32

  // CHECK: %[[LE:.*]] = arith.cmpf ole, %arg0, %arg1 : f32
  %3 = halide.le %arg0, %arg1 : f32

  // CHECK: %[[GT:.*]] = arith.cmpf ogt, %arg0, %arg1 :  f32
  %4 = halide.gt %arg0, %arg1 : f32

  // CHECK: %[[GE:.*]] = arith.cmpf oge, %arg0, %arg1 :  f32
  %5 = halide.ge %arg0, %arg1 : f32

  // CHECK: return %[[GE]] : i1
  return %5 : i1
}
