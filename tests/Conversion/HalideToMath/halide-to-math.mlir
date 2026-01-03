// RUN: halide-opt %s -convert-halide-to-math | FileCheck %s

//===----------------------------------------------------------------------===//
// Unary Operations - f32 with type suffix
//===----------------------------------------------------------------------===//

func.func @test_sqrt_f32(%arg0: f32) -> f32 {
  // CHECK: math.sqrt %arg0 : f32
  %0 = halide.call Intrinsic "sqrt_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_sin_f32(%arg0: f32) -> f32 {
  // CHECK: math.sin %arg0 : f32
  %0 = halide.call PureIntrinsic "sin_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_cos_f32(%arg0: f32) -> f32 {
  // CHECK: math.cos %arg0 : f32
  %0 = halide.call PureIntrinsic "cos_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_tan_f32(%arg0: f32) -> f32 {
  // CHECK: math.tan %arg0 : f32
  %0 = halide.call PureIntrinsic "tan_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_exp_f32(%arg0: f32) -> f32 {
  // CHECK: math.exp %arg0 : f32
  %0 = halide.call PureIntrinsic "exp_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_exp2_f32(%arg0: f32) -> f32 {
  // CHECK: math.exp2 %arg0 : f32
  %0 = halide.call PureIntrinsic "exp2_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_log_f32(%arg0: f32) -> f32 {
  // CHECK: math.log %arg0 : f32
  %0 = halide.call PureIntrinsic "log_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_log2_f32(%arg0: f32) -> f32 {
  // CHECK: math.log2 %arg0 : f32
  %0 = halide.call PureIntrinsic "log2_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_log10_f32(%arg0: f32) -> f32 {
  // CHECK: math.log10 %arg0 : f32
  %0 = halide.call PureIntrinsic "log10_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_abs_f32(%arg0: f32) -> f32 {
  // CHECK: math.absf %arg0 : f32
  %0 = halide.call PureIntrinsic "abs_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_fabs(%arg0: f32) -> f32 {
  // CHECK: math.absf %arg0 : f32
  %0 = halide.call PureIntrinsic "fabs"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_floor_f32(%arg0: f32) -> f32 {
  // CHECK: math.floor %arg0 : f32
  %0 = halide.call PureIntrinsic "floor_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_ceil_f32(%arg0: f32) -> f32 {
  // CHECK: math.ceil %arg0 : f32
  %0 = halide.call PureIntrinsic "ceil_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_round_f32(%arg0: f32) -> f32 {
  // CHECK: math.round %arg0 : f32
  %0 = halide.call PureIntrinsic "round_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_trunc_f32(%arg0: f32) -> f32 {
  // CHECK: math.trunc %arg0 : f32
  %0 = halide.call PureIntrinsic "trunc_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_asin_f32(%arg0: f32) -> f32 {
  // CHECK: math.asin %arg0 : f32
  %0 = halide.call PureIntrinsic "asin_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_acos_f32(%arg0: f32) -> f32 {
  // CHECK: math.acos %arg0 : f32
  %0 = halide.call PureIntrinsic "acos_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_atan_f32(%arg0: f32) -> f32 {
  // CHECK: math.atan %arg0 : f32
  %0 = halide.call PureIntrinsic "atan_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_sinh_f32(%arg0: f32) -> f32 {
  // CHECK: math.sinh %arg0 : f32
  %0 = halide.call PureIntrinsic "sinh_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_cosh_f32(%arg0: f32) -> f32 {
  // CHECK: math.cosh %arg0 : f32
  %0 = halide.call PureIntrinsic "cosh_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_tanh_f32(%arg0: f32) -> f32 {
  // CHECK: math.tanh %arg0 : f32
  %0 = halide.call PureIntrinsic "tanh_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_erf_f32(%arg0: f32) -> f32 {
  // CHECK: math.erf %arg0 : f32
  %0 = halide.call PureIntrinsic "erf_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_rsqrt_f32(%arg0: f32) -> f32 {
  // CHECK: math.rsqrt %arg0 : f32
  %0 = halide.call PureIntrinsic "rsqrt_f32"(%arg0) : (f32) -> f32
  return %0 : f32
}

//===----------------------------------------------------------------------===//
// Unary Operations - f64 with type suffix
//===----------------------------------------------------------------------===//

func.func @test_sqrt_f64(%arg0: f64) -> f64 {
  // CHECK: math.sqrt %arg0 : f64
  %0 = halide.call Intrinsic "sqrt_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_sin_f64(%arg0: f64) -> f64 {
  // CHECK: math.sin %arg0 : f64
  %0 = halide.call PureIntrinsic "sin_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_cos_f64(%arg0: f64) -> f64 {
  // CHECK: math.cos %arg0 : f64
  %0 = halide.call PureIntrinsic "cos_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_tan_f64(%arg0: f64) -> f64 {
  // CHECK: math.tan %arg0 : f64
  %0 = halide.call PureIntrinsic "tan_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_exp_f64(%arg0: f64) -> f64 {
  // CHECK: math.exp %arg0 : f64
  %0 = halide.call PureIntrinsic "exp_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_exp2_f64(%arg0: f64) -> f64 {
  // CHECK: math.exp2 %arg0 : f64
  %0 = halide.call PureIntrinsic "exp2_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_log_f64(%arg0: f64) -> f64 {
  // CHECK: math.log %arg0 : f64
  %0 = halide.call PureIntrinsic "log_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_log2_f64(%arg0: f64) -> f64 {
  // CHECK: math.log2 %arg0 : f64
  %0 = halide.call PureIntrinsic "log2_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_log10_f64(%arg0: f64) -> f64 {
  // CHECK: math.log10 %arg0 : f64
  %0 = halide.call PureIntrinsic "log10_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_abs_f64(%arg0: f64) -> f64 {
  // CHECK: math.absf %arg0 : f64
  %0 = halide.call PureIntrinsic "abs_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_floor_f64(%arg0: f64) -> f64 {
  // CHECK: math.floor %arg0 : f64
  %0 = halide.call PureIntrinsic "floor_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_ceil_f64(%arg0: f64) -> f64 {
  // CHECK: math.ceil %arg0 : f64
  %0 = halide.call PureIntrinsic "ceil_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_round_f64(%arg0: f64) -> f64 {
  // CHECK: math.round %arg0 : f64
  %0 = halide.call PureIntrinsic "round_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_trunc_f64(%arg0: f64) -> f64 {
  // CHECK: math.trunc %arg0 : f64
  %0 = halide.call PureIntrinsic "trunc_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_asin_f64(%arg0: f64) -> f64 {
  // CHECK: math.asin %arg0 : f64
  %0 = halide.call PureIntrinsic "asin_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_acos_f64(%arg0: f64) -> f64 {
  // CHECK: math.acos %arg0 : f64
  %0 = halide.call PureIntrinsic "acos_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_atan_f64(%arg0: f64) -> f64 {
  // CHECK: math.atan %arg0 : f64
  %0 = halide.call PureIntrinsic "atan_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_sinh_f64(%arg0: f64) -> f64 {
  // CHECK: math.sinh %arg0 : f64
  %0 = halide.call PureIntrinsic "sinh_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_cosh_f64(%arg0: f64) -> f64 {
  // CHECK: math.cosh %arg0 : f64
  %0 = halide.call PureIntrinsic "cosh_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_tanh_f64(%arg0: f64) -> f64 {
  // CHECK: math.tanh %arg0 : f64
  %0 = halide.call PureIntrinsic "tanh_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_erf_f64(%arg0: f64) -> f64 {
  // CHECK: math.erf %arg0 : f64
  %0 = halide.call PureIntrinsic "erf_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func @test_rsqrt_f64(%arg0: f64) -> f64 {
  // CHECK: math.rsqrt %arg0 : f64
  %0 = halide.call PureIntrinsic "rsqrt_f64"(%arg0) : (f64) -> f64
  return %0 : f64
}

//===----------------------------------------------------------------------===//
// Unary Operations - Generic names (no type suffix)
//===----------------------------------------------------------------------===//

func.func @test_sqrt_generic(%arg0: f32) -> f32 {
  // CHECK: math.sqrt %arg0 : f32
  %0 = halide.call PureIntrinsic "sqrt"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_sin_generic(%arg0: f32) -> f32 {
  // CHECK: math.sin %arg0 : f32
  %0 = halide.call PureIntrinsic "sin"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_cos_generic(%arg0: f32) -> f32 {
  // CHECK: math.cos %arg0 : f32
  %0 = halide.call PureIntrinsic "cos"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_tan_generic(%arg0: f32) -> f32 {
  // CHECK: math.tan %arg0 : f32
  %0 = halide.call PureIntrinsic "tan"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_exp_generic(%arg0: f32) -> f32 {
  // CHECK: math.exp %arg0 : f32
  %0 = halide.call PureIntrinsic "exp"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_exp2_generic(%arg0: f32) -> f32 {
  // CHECK: math.exp2 %arg0 : f32
  %0 = halide.call PureIntrinsic "exp2"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_log_generic(%arg0: f32) -> f32 {
  // CHECK: math.log %arg0 : f32
  %0 = halide.call PureIntrinsic "log"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_log2_generic(%arg0: f32) -> f32 {
  // CHECK: math.log2 %arg0 : f32
  %0 = halide.call PureIntrinsic "log2"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_log10_generic(%arg0: f32) -> f32 {
  // CHECK: math.log10 %arg0 : f32
  %0 = halide.call PureIntrinsic "log10"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_floor_generic(%arg0: f32) -> f32 {
  // CHECK: math.floor %arg0 : f32
  %0 = halide.call PureIntrinsic "floor"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_ceil_generic(%arg0: f32) -> f32 {
  // CHECK: math.ceil %arg0 : f32
  %0 = halide.call PureIntrinsic "ceil"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_round_generic(%arg0: f32) -> f32 {
  // CHECK: math.round %arg0 : f32
  %0 = halide.call PureIntrinsic "round"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_trunc_generic(%arg0: f32) -> f32 {
  // CHECK: math.trunc %arg0 : f32
  %0 = halide.call PureIntrinsic "trunc"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_asin_generic(%arg0: f32) -> f32 {
  // CHECK: math.asin %arg0 : f32
  %0 = halide.call PureIntrinsic "asin"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_acos_generic(%arg0: f32) -> f32 {
  // CHECK: math.acos %arg0 : f32
  %0 = halide.call PureIntrinsic "acos"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_atan_generic(%arg0: f32) -> f32 {
  // CHECK: math.atan %arg0 : f32
  %0 = halide.call PureIntrinsic "atan"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_sinh_generic(%arg0: f32) -> f32 {
  // CHECK: math.sinh %arg0 : f32
  %0 = halide.call PureIntrinsic "sinh"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_cosh_generic(%arg0: f32) -> f32 {
  // CHECK: math.cosh %arg0 : f32
  %0 = halide.call PureIntrinsic "cosh"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_tanh_generic(%arg0: f32) -> f32 {
  // CHECK: math.tanh %arg0 : f32
  %0 = halide.call PureIntrinsic "tanh"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_erf_generic(%arg0: f32) -> f32 {
  // CHECK: math.erf %arg0 : f32
  %0 = halide.call PureIntrinsic "erf"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_fast_inverse_sqrt(%arg0: f32) -> f32 {
  // CHECK: math.rsqrt %arg0 : f32
  %0 = halide.call PureIntrinsic "fast_inverse_sqrt"(%arg0) : (f32) -> f32
  return %0 : f32
}

//===----------------------------------------------------------------------===//
// Binary Operations - f32 with type suffix
//===----------------------------------------------------------------------===//

func.func @test_pow_f32(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: math.powf %arg0, %arg1 : f32
  %0 = halide.call PureIntrinsic "pow_f32"(%arg0, %arg1) : (f32, f32) -> f32
  return %0 : f32
}

func.func @test_atan2_f32(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: math.atan2 %arg0, %arg1 : f32
  %0 = halide.call PureIntrinsic "atan2_f32"(%arg0, %arg1) : (f32, f32) -> f32
  return %0 : f32
}

func.func @test_copysign_f32(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: math.copysign %arg0, %arg1 : f32
  %0 = halide.call PureIntrinsic "copysign_f32"(%arg0, %arg1) : (f32, f32) -> f32
  return %0 : f32
}

//===----------------------------------------------------------------------===//
// Binary Operations - f64 with type suffix
//===----------------------------------------------------------------------===//

func.func @test_pow_f64(%arg0: f64, %arg1: f64) -> f64 {
  // CHECK: math.powf %arg0, %arg1 : f64
  %0 = halide.call PureIntrinsic "pow_f64"(%arg0, %arg1) : (f64, f64) -> f64
  return %0 : f64
}

func.func @test_atan2_f64(%arg0: f64, %arg1: f64) -> f64 {
  // CHECK: math.atan2 %arg0, %arg1 : f64
  %0 = halide.call PureIntrinsic "atan2_f64"(%arg0, %arg1) : (f64, f64) -> f64
  return %0 : f64
}

func.func @test_copysign_f64(%arg0: f64, %arg1: f64) -> f64 {
  // CHECK: math.copysign %arg0, %arg1 : f64
  %0 = halide.call PureIntrinsic "copysign_f64"(%arg0, %arg1) : (f64, f64) -> f64
  return %0 : f64
}

//===----------------------------------------------------------------------===//
// Binary Operations - Generic names (no type suffix)
//===----------------------------------------------------------------------===//

func.func @test_pow_generic(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: math.powf %arg0, %arg1 : f32
  %0 = halide.call PureIntrinsic "pow"(%arg0, %arg1) : (f32, f32) -> f32
  return %0 : f32
}

func.func @test_atan2_generic(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: math.atan2 %arg0, %arg1 : f32
  %0 = halide.call PureIntrinsic "atan2"(%arg0, %arg1) : (f32, f32) -> f32
  return %0 : f32
}

func.func @test_copysign_generic(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: math.copysign %arg0, %arg1 : f32
  %0 = halide.call PureIntrinsic "copysign"(%arg0, %arg1) : (f32, f32) -> f32
  return %0 : f32
}

//===----------------------------------------------------------------------===//
// Ternary Operations - f32 with type suffix
//===----------------------------------------------------------------------===//

func.func @test_fma_f32(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
  // CHECK: math.fma %arg0, %arg1, %arg2 : f32
  %0 = halide.call PureIntrinsic "fma_f32"(%arg0, %arg1, %arg2) : (f32, f32, f32) -> f32
  return %0 : f32
}

//===----------------------------------------------------------------------===//
// Ternary Operations - f64 with type suffix
//===----------------------------------------------------------------------===//

func.func @test_fma_f64(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 {
  // CHECK: math.fma %arg0, %arg1, %arg2 : f64
  %0 = halide.call PureIntrinsic "fma_f64"(%arg0, %arg1, %arg2) : (f64, f64, f64) -> f64
  return %0 : f64
}

//===----------------------------------------------------------------------===//
// Ternary Operations - Generic names (no type suffix)
//===----------------------------------------------------------------------===//

func.func @test_fma_generic(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
  // CHECK: math.fma %arg0, %arg1, %arg2 : f32
  %0 = halide.call PureIntrinsic "fma"(%arg0, %arg1, %arg2) : (f32, f32, f32) -> f32
  return %0 : f32
}

//===----------------------------------------------------------------------===//
// Mixed CallType tests
//===----------------------------------------------------------------------===//

func.func @test_intrinsic_call_type(%arg0: f32) -> f32 {
  // CHECK: math.sqrt %arg0 : f32
  %0 = halide.call Intrinsic "sqrt"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_pure_intrinsic_call_type(%arg0: f32) -> f32 {
  // CHECK: math.exp %arg0 : f32
  %0 = halide.call PureIntrinsic "exp"(%arg0) : (f32) -> f32
  return %0 : f32
}

//===----------------------------------------------------------------------===//
// Negative tests - operations that should NOT be converted
//===----------------------------------------------------------------------===//

func.func @test_extern_call_not_converted(%arg0: f32) -> f32 {
  // CHECK: halide.call Extern "custom_function"
  %0 = halide.call Extern "custom_function"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_halide_call_not_converted(%arg0: f32) -> f32 {
  // CHECK: halide.call Halide "my_func"
  %0 = halide.call Halide "my_func"(%arg0) : (f32) -> f32
  return %0 : f32
}

func.func @test_unknown_intrinsic_not_converted(%arg0: f32) -> f32 {
  // CHECK: halide.call PureIntrinsic "unknown_func"
  %0 = halide.call PureIntrinsic "unknown_func"(%arg0) : (f32) -> f32
  return %0 : f32
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

func.func @test_multiple_calls(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: [[V0:%.+]] = math.sin %arg0 : f32
  // CHECK: [[V1:%.+]] = math.cos %arg1 : f32
  // CHECK: math.powf [[V0]], [[V1]] : f32
  %0 = halide.call PureIntrinsic "sin"(%arg0) : (f32) -> f32
  %1 = halide.call PureIntrinsic "cos"(%arg1) : (f32) -> f32
  %2 = halide.call PureIntrinsic "pow"(%0, %1) : (f32, f32) -> f32
  return %2 : f32
}

func.func @test_nested_calls(%arg0: f32) -> f32 {
  // CHECK: [[V0:%.+]] = math.sqrt %arg0 : f32
  // CHECK: math.log [[V0]] : f32
  %0 = halide.call PureIntrinsic "sqrt"(%arg0) : (f32) -> f32
  %1 = halide.call PureIntrinsic "log"(%0) : (f32) -> f32
  return %1 : f32
}
