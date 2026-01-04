// RUN: halide-opt %s -convert-halide-to-arith -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_bitwise_ops
func.func @test_bitwise_ops(%arg0: i32, %arg1: i32) -> (i32, i32, i32, i32) {
    // CHECK: %[[AND:.*]] = arith.andi %arg0, %arg1 : i32
    %0 = halide.call PureIntrinsic "bitwise_and"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK: %[[OR:.*]] = arith.ori %arg0, %arg1 : i32
    %1 = halide.call PureIntrinsic "bitwise_or"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK: %[[XOR:.*]] = arith.xori %arg0, %arg1 : i32
    %2 = halide.call PureIntrinsic "bitwise_xor"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK: %[[C:.*]] = arith.constant -1 : i32
    // CHECK: %[[NOT:.*]] = arith.xori %arg0, %[[C]] : i32
    %3 = halide.call PureIntrinsic "bitwise_not"(%arg0) : (i32) -> i32
    return %0, %1, %2, %3 : i32, i32, i32, i32
}

// -----

// CHECK-LABEL: func @test_shift_ops
func.func @test_shift_ops(%arg0: i32, %arg1: i32) -> (i32, i32) {
    // CHECK: %[[SHL:.*]] = arith.shli %arg0, %arg1 : i32
    %0 = halide.call PureIntrinsic "shift_left"(%arg0, %arg1) : (i32, i32) -> i32
    // CHECK: %[[SHR:.*]] = arith.shrsi %arg0, %arg1 : i32
    %1 = halide.call PureIntrinsic "shift_right"(%arg0, %arg1) : (i32, i32) -> i32
    return %0, %1 : i32, i32
}

// -----

// CHECK-LABEL: func @test_mul_shift_right
func.func @test_mul_shift_right(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    // CHECK: %[[MUL:.*]] = arith.muli %arg0, %arg1 : i32
    // CHECK: arith.shrsi %[[MUL]], %arg2 : i32
    %0 = halide.call PureIntrinsic "mul_shift_right"(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
    return %0 : i32
}

// -----

// CHECK-LABEL: func @test_halving_add
func.func @test_halving_add(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: %[[ADD:.*]] = arith.addi %arg0, %arg1 : i32
    // CHECK: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK: arith.shrsi %[[ADD]], %[[ONE]] : i32
    %0 = halide.call PureIntrinsic "halving_add"(%arg0, %arg1) : (i32, i32) -> i32
    return %0 : i32
}

// -----

// CHECK-LABEL: func @test_halving_sub
func.func @test_halving_sub(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: %[[SUB:.*]] = arith.subi %arg0, %arg1 : i32
    // CHECK: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK: arith.shrsi %[[SUB]], %[[ONE]] : i32
    %0 = halide.call PureIntrinsic "halving_sub"(%arg0, %arg1) : (i32, i32) -> i32
    return %0 : i32
}

// -----

// CHECK-LABEL: func @test_rounding_halving_add
func.func @test_rounding_halving_add(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: %[[ADD:.*]] = arith.addi %arg0, %arg1 : i32
    // CHECK: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK: %[[ADDONE:.*]] = arith.addi %[[ADD]], %[[ONE]] : i32
    // CHECK: arith.shrsi %[[ADDONE]], %[[ONE]] : i32
    %0 = halide.call PureIntrinsic "rounding_halving_add"(%arg0, %arg1) : (i32, i32) -> i32
    return %0 : i32
}
