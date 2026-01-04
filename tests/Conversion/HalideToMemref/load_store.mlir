// RUN: halide-opt %s --convert-halide-to-memref | FileCheck %s

// CHECK-LABEL: @load
func.func @load(%arg0: !halide.buffer<2, i32> {halide.name = "buf"}) -> i32 {
  %c0 = arith.constant 0 : i32
  %true = arith.constant true
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: %[[IF:.*]] = scf.if %[[TRUE]]
  // CHECK: %[[SHAPE:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1]] : memref<?x?xi32> into memref<?xi32>
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[LOAD:.*]] = memref.load %[[SHAPE]][%[[ZERO]]] : memref<?xi32>
  // CHECK: scf.yield %[[LOAD]]
  // CHECK: else
  // CHECK: %[[UNDEF:.*]] = arith.constant 0
  // CHECK: scf.yield %[[UNDEF]]
  %0 = halide.load "buf"[%c0: i32] if %true : i32

  // CHECK: return %[[IF]]
  return %0: i32
}

func.func @store(%arg0: !halide.buffer<2, i32> {halide.name = "buf"}) {
  %c0 = arith.constant 0: i32
  %true = arith.constant true
  // CHECK: %[[ZERO_INT:.*]] = arith.constant 0 : i32
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: scf.if %[[TRUE]]
  // CHECK: %[[SHAPE:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1]] : memref<?x?xi32> into memref<?xi32>
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[ZERO_INT]], %[[SHAPE]][%[[ZERO]]] : memref<?xi32>
  // CHECK-NOT: else
  halide.store "buf"[%c0: i32] = %c0 if %true : i32
  return
}

