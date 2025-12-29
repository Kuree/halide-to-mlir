add_subdirectory(lib)

# compute halide path
get_target_property(HALIDE_LIB_PATH Halide::Halide IMPORTED_LOCATION_RELEASE)
get_filename_component(HALIDE_LIB_PATH ${HALIDE_LIB_PATH} DIRECTORY)
get_filename_component(HALIDE_LIB_PATH ${HALIDE_LIB_PATH} DIRECTORY)
set(HALIDE_INCLUDE_PATH "${HALIDE_LIB_PATH}/include")
set(HALIDE_LIB_PATH "${HALIDE_LIB_PATH}/lib")

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.cfg.py
)

set (MLIR_TEST_DEPENDS
        FileCheck count not
        HalideToMLIRDriver
)

add_lit_testsuite(check-halide "Running the halide-to-mlir regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MLIR_TEST_DEPENDS}
)
