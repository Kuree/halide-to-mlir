// RUN: halide-opt %s -convert-halide-to-math -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_abs_float
func.func @test_abs_float(%arg0: f32) -> f32 {
    // CHECK: math.absf %arg0 : f32
    %0 = halide.call PureIntrinsic "abs"(%arg0) : (f32) -> f32
    return %0 : f32
}

// -----

// CHECK-LABEL: func @test_abs_int
func.func @test_abs_int(%arg0: i32) -> i32 {
    // CHECK: math.absi %arg0 : i32
    %0 = halide.call PureIntrinsic "abs"(%arg0) : (i32) -> i32
    return %0 : i32
}

// -----

// CHECK-LABEL: func @test_absd_int
func.func @test_absd_int(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: %[[SUB:.*]] = arith.subi %arg0, %arg1 : i32
    // CHECK: %[[ABS:.*]] = math.absi %[[SUB]] : i32
    %0 = halide.call PureIntrinsic "absd"(%arg0, %arg1) : (i32, i32) -> i32
    return %0 : i32
}

// -----

// CHECK-LABEL: func @test_absd_float
func.func @test_absd_float(%arg0: f32, %arg1: f32) -> f32 {
    // CHECK: %[[SUB:.*]] = arith.subf %arg0, %arg1 : f32
    // CHECK: %[[ABS:.*]] = math.absf %[[SUB]] : f32
    %0 = halide.call PureIntrinsic "absd"(%arg0, %arg1) : (f32, f32) -> f32
    return %0 : f32
}

// -----

// CHECK-LABEL: func @test_count_ops
func.func @test_count_ops(%arg0: i32) -> (i32, i32) {
    // CHECK: %[[CLZ:.*]] = math.ctlz %arg0 : i32
    %0 = halide.call PureIntrinsic "count_leading_zeros"(%arg0) : (i32) -> i32
    // CHECK: %[[CTZ:.*]] = math.cttz %arg0 : i32
    %1 = halide.call PureIntrinsic "count_trailing_zeros"(%arg0) : (i32) -> i32
    return %0, %1 : i32, i32
}

// -----

// CHECK-LABEL: func @test_round
func.func @test_round(%arg0: f32) -> f32 {
    // CHECK: math.roundeven %arg0 : f32
    %0 = halide.call PureIntrinsic "round"(%arg0) : (f32) -> f32
    return %0 : f32
}
