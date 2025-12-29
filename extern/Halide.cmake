include(FetchContent)

set(HALIDE_VERSION "21.0.0")
set(HALIDE_URL "https://github.com/halide/Halide/releases/download/v21.0.0/Halide-21.0.0-x86-64-linux-b629c80de18f1534ec71fddd8b567aa7027a0876.tar.gz")

message(STATUS "Downloading Halide release v${HALIDE_VERSION}...")

FetchContent_Declare(
        halide_binary
        URL ${HALIDE_URL}
        # Create a dummy CMakeLists.txt so FetchContent_MakeAvailable doesn't error out
        # trying to add_subdirectory() on a folder without one.
        PATCH_COMMAND "${CMAKE_COMMAND}" -E touch CMakeLists.txt
        DOWNLOAD_EXTRACT_TIMESTAMP ON
)

FetchContent_MakeAvailable(halide_binary)

list(APPEND CMAKE_PREFIX_PATH "${halide_binary_SOURCE_DIR}/lib/cmake")
find_package(Halide REQUIRED CONFIG)
