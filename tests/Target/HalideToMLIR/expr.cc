// RUN: cc %s -o %t -I %halide_include -I %project_source_dir -L %lib_dir -L %halide_lib_dir -L %llvm_lib_dir -lHalideToMLIRDriver -lHalide -lMLIR -lLLVM -Wl,-rpath=%lib_dir -Wl,-rpath=%halide_lib_dir

#include "Halide.h"
#include "tests/lib/Target/Halide/HalideToMLIRDriver.hh"

int main() {
    Halide::Func func("func");
    Halide::Var x("x"), y("y");
    func(x, y) = x + y;

    convertAndPrintHalideToMLIR(func);

    return 0;
}
