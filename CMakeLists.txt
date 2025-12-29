cmake_minimum_required(VERSION 3.15)
project(halide_to_mlir)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0116)
    # Introduced in cmake 3.20
    # https://cmake.org/cmake/help/latest/policy/CMP0116.html
    cmake_policy(SET CMP0116 NEW)
endif()

find_package(LLVM 19.1 REQUIRED CONFIG)
set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
find_package(MLIR 19.1 REQUIRED CONFIG)

include(${LLVM_DIR}/AddLLVM.cmake)
include(${LLVM_DIR}/TableGen.cmake)
include(${MLIR_DIR}/AddMLIR.cmake)
include(extern/Halide.cmake)

# include globally
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(include)
add_subdirectory(lib)
