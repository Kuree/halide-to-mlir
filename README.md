# Halide to MLIR

# Overview
This project provides a compiler infrastructure to convert [Halide](https://halide-lang.org/) programs into
[MLIR](https://mlir.llvm.org/).

## How is Halide program imported
This project provides a simple function that takes any scheduled Halide function and produces an MLIR module
in `halide` dialect.

```C++
namespace mlir::halide {
OwningOpRef<ModuleOp>
importHalide(Halide::Func func, MLIRContext *context,
             const Halide::Target &target = Halide::get_host_target());
}
```

All the semantics of Halide IR is preserved, albert in a very verbose way. Then an optimization pass called
`let2reg`, similar to LLVM's `mem2reg` is run to convert `halide.let` statements to SSA values, while obeying
proper name shadowing rules.

Then we run a series of conversion passes that convert `halide` to dialect to MLIR's standard dialects:
- `arith`
- `memref`
- `scf`
- `func`

# Usage example

```C++
// Halide code
int main() {
    Halide::Func func("func");
    Halide::Var x("x"), y("y");
    func(x, y) = x + y;
    // scheduling for `func`
    return 0;
}
```

We can then call the importing function after the scheduling, and dumping it to stdout.
```C++
    // this function is defined in the test dir for simple testing:
    //   tests/lib/Target/Halide/HalideToMLIRDriver.cc
    // it calls `importHalide` internally after setting up the MLIRContext
    convertAndPrintHalideToMLIR(func);
```

The `halide` dialect looks something like this:

```mlir
func.func @func(%arg0: !halide.buffer<2, i32> {halide.name = "func"}) {
    halide.let "func.buffer" = %arg0 : !halide.buffer<2, i32> {
      %0 = halide.variable "func.buffer" : !halide.handle
      %c0_i64 = arith.constant 0 : i64
      %1 = halide.cast %0 : !halide.handle to i64
      %2 = halide.ne %1, %c0_i64 : i64
      halide.assert %2 {
        %5 = llvm.mlir.addressof @str : !llvm.ptr
        %6 = halide.call Extern "halide_error_buffer_argument_is_null"(%5) : (!llvm.ptr) -> i32
      }
      // ...
      // this is the innermost loop below
      halide.for "func.s0.x.rebased" = %c0_i32_12 extent %79 : i32 type Serial device None partition Auto {
        %80 = halide.variable "func.s0.x.rebased" : i32
        %81 = halide.variable "t3" : i32
        %82 = halide.add %80, %81 : i32
        %83 = halide.variable "func.s0.x.rebased" : i32
        %84 = halide.variable "t4" : i32
        %85 = halide.add %83, %84 : i32
        %true = arith.constant true
        halide.store "func"[%85 : i32] = %82 if %true : i32
      }
```

Note that an auxiliary `halide.let` is created for the function argument so that it can be properly
referenced by `halide.variable` ops.

Afterward, we can use the `halide-opt` driver to run the whole conversion pipeline:

```bash
halide-opt input.mlir --convert-halide-to-arith --convert-halide-to-math \
    --convert-halide-to-scf --convert-halide-to-memref --convert-halide-to-func \
    --canonicalize --cse
```

You will then get the below MLIR, which is defined in the standard dialects.

```mlir
func.func @func(%arg0: memref<?x?xi32> {halide.name = "func"}) {
  // Halide assertion logic is omitted here but they are lowered to libc calls.
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xi32>
  %dim_1 = memref.dim %arg0, %c1 : memref<?x?xi32>
  scf.for %arg1 = %c0 to %dim_1 step %c1 {
    %27 = arith.index_cast %arg1 : index to i32
    scf.for %arg2 = %c0 to %dim step %c1 {
      %28 = arith.index_cast %arg2 : index to i32
      %29 = arith.addi %28, %27 : i32
      %30 = arith.addi %28, %27 : i32
      %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<?x?xi32> into memref<?xi32>
      %31 = arith.index_cast %30 : i32 to index
      memref.store %29, %collapse_shape[%31] : memref<?xi32>
    }
  }
}
```

# How to compile

## Requirements
- C++17 compiler
- cmake
- LLVM 19+ (tested with 19 and 20)

Halide is pinned to the `21.0` release, and the release binaries are fetched during cmake
configuration.

## Compile and run the tests

```bash
mkdir -p build
cd build
cmake -GNinja ../
ninja
ninja check-halide
```

Omit `-GNinja` if you prefer to use `make`.
