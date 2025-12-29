add_library(HalideToMLIRDriver SHARED HalideToMLIRDriver.cc)

# use whole archive so that we only need to link one
target_link_libraries(HalideToMLIRDriver PRIVATE MLIRHalideToMLIR Halide::Halide)
