// RUN: halide-opt %s -convert-halide-to-memref --allow-unregistered-dialect | FileCheck %s

// Test _halide_buffer_get_dimensions
// CHECK-LABEL: @test_get_dimensions
func.func @test_get_dimensions(%arg0: !halide.buffer<2, f32>) -> i32 {
  // CHECK: %[[RANK:.*]] = memref.rank %arg0 : memref<?x?xf32>
  // CHECK: %[[RESULT:.*]] = arith.index_cast %[[RANK]] : index to i32
  // CHECK: return %[[RESULT]]
  
  %result = halide.call Intrinsic "_halide_buffer_get_dimensions"(%arg0) : 
    (!halide.buffer<2, f32>) -> i32
  return %result : i32
}

// Test _halide_buffer_get_extent
// CHECK-LABEL: @test_get_extent
func.func @test_get_extent(%arg0: !halide.buffer<2, f32>) -> i32 {
  %c0 = arith.constant 0 : i32

  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[EXTENT:.*]] = memref.dim %arg0, %[[ZERO]] : memref<?x?xf32>
  // CHECK: %[[RESULT:.*]] = arith.index_cast %[[EXTENT]] : index to i32

  %result = halide.call Intrinsic "_halide_buffer_get_extent"(%arg0, %c0) :
    (!halide.buffer<2, f32>, i32) -> i32

  // CHECK: return %[[RESULT]]
  return %result : i32
}

// Test _halide_buffer_get_min
// CHECK-LABEL: @test_get_min
func.func @test_get_min(%arg0: !halide.buffer<2, f32>) -> i32 {
  %c1 = arith.constant 1 : i32

  // CHECK: %[[RESULT:.*]] = arith.constant 0 : i32

  %result = halide.call Intrinsic "_halide_buffer_get_min"(%arg0, %c1) {
    buffer_dims = [
      {min = 0 : i32, extent = 10 : i32, stride = 20 : i32},
      {min = 0 : i32, extent = 20 : i32, stride = 1 : i32}
    ]
  } : (!halide.buffer<2, f32>, i32) -> i32
  // CHECK: return %[[RESULT]]
  return %result : i32
}

// Test _halide_buffer_get_stride
// CHECK-LABEL: @test_get_stride
func.func @test_get_stride(%arg0: !halide.buffer<2, f32>) -> i32 {
  %c0 = arith.constant 0 : i32

  // CHECK: %{{.*}}, %{{.*}}, %{{.*}}:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %arg0 : memref<?x?xf32>
  // CHECK: %[[RESULT:.*]] = arith.index_cast %[[STRIDES]]#0 : index to i32
  %result = halide.call Intrinsic "_halide_buffer_get_stride"(%arg0, %c0) :
    (!halide.buffer<2, f32>, i32) -> i32
  // CHECK: return %[[RESULT]]
  return %result : i32
}

// Test _halide_buffer_get_max
// CHECK-LABEL: @test_get_max
func.func @test_get_max(%arg0: !halide.buffer<2, f32>) -> i32 {
  %c0 = arith.constant 0 : i32

  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %arg0, %[[ZERO]] : memref<?x?xf32>
  // CHECK: %[[DIM_CAST:.*]] = arith.index_cast %[[DIM]] : index to i32
  // CHECK: %[[ONE:.*]] = arith.constant 1 : i32
  // CHECK: %[[ADD:.*]] = arith.addi %[[DIM_CAST]], %c0
  // CHECK: %[[RESULT:.*]] = arith.subi %[[ADD]], %[[ONE]]


  %result = halide.call Intrinsic "_halide_buffer_get_max"(%arg0, %c0) :
    (!halide.buffer<2, f32>, i32) -> i32
  // CHECK: return %[[RESULT]]
  return %result : i32
}

// Test _halide_buffer_get_type
// CHECK-LABEL: @test_get_type
func.func @test_get_type(%arg0: !halide.buffer<2, f32>) -> i32 {
  // CHECK: %[[TYPE:.*]] = arith.constant {{.*}} : i32
  // CHECK: return %[[TYPE]]

  %result = halide.call Intrinsic "_halide_buffer_get_type"(%arg0) :
    (!halide.buffer<2, f32>) -> i32
  return %result : i32
}

// Test _halide_buffer_is_bounds_query
// CHECK-LABEL: @test_is_bounds_query
func.func @test_is_bounds_query(%arg0: !halide.buffer<2, f32>) -> i1 {
  // CHECK: %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<?x?xf32> -> index
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = arith.cmpi eq, %[[PTR]], %[[ZERO]] : index

  %result = halide.call Intrinsic "_halide_buffer_is_bounds_query"(%arg0) :
    (!halide.buffer<2, f32>) -> i1

  // CHECK: return %[[RESULT]]
  return %result : i1
}

// Test dirty flag operations (should become no-ops or return constants)
// CHECK-LABEL: @test_dirty_flags
func.func @test_dirty_flags(%arg0: !halide.buffer<2, f32>) {
  %true = arith.constant true

  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK: "op.op"(%[[ZERO]])
  %r1 = halide.call Intrinsic "_halide_buffer_set_host_dirty"(%arg0, %true) :
    (!halide.buffer<2, f32>, i1) -> i32
  "op.op"(%r1) : (i32) -> ()

  // CHECK: %[[FALSE:.*]] = arith.constant false
  // CHECK: "op.op"(%[[FALSE]])
  %r2 = halide.call Intrinsic "_halide_buffer_get_host_dirty"(%arg0) :
    (!halide.buffer<2, f32>) -> i1
  "op.op"(%r2) : (i1) -> ()

  return
}

// Test _halide_buffer_set_bounds
// CHECK-LABEL: @test_set_bounds
func.func @test_set_bounds(%arg0: !halide.buffer<2, f32>) -> i32 {
  %dim = arith.constant 0 : i32
  %min = arith.constant 5 : i32
  %extent = arith.constant 10 : i32

  // CHECK: return %c0

  %result = halide.call Intrinsic "_halide_buffer_set_bounds"(%arg0, %dim, %min, %extent) :
    (!halide.buffer<2, f32>, i32, i32, i32) -> i32
  return %result : i32
}

// Test _halide_buffer_get_host
// CHECK-LABEL: @test_buffer_get_host
func.func @test_buffer_get_host(%arg0: !halide.buffer<2, i32>) -> i64 {
  // CHECK: %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<?x?xi32> -> index
  // CHECK: %[[IDX:.*]] = arith.index_cast %[[PTR]] : index to i64
  %0 = halide.call Extern "_halide_buffer_get_host"(%arg0) : (!halide.buffer<2, i32>) -> !halide.handle
  %1 = halide.cast %0 : !halide.handle to i64
  // CHECK: return %[[IDX]]
  return %1: i64
}
