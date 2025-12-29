// RUN: cc %s -o %t -I %halide_include -I %project_source_dir -L %lib_dir -lHalideToMLIRDriver %halide_lib -L %llvm_lib_dir -lMLIR -lLLVM

#include "Halide.h"
#include "tests/lib/Target/Halide/HalideToMLIRDriver.hh"

int main() {
    Halide::Func func("func");
    Halide::Var x("x"), y("y");
    func(x, y) = x + y;

    convertAndPrintHalideToMLIR(func);

    return 0;
}
