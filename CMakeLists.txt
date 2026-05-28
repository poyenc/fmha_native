cmake_minimum_required(VERSION 3.21)
project(fmha_native LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(GPU_TARGET "gfx942" CACHE STRING "GPU target architecture")
set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGET})

find_package(hip REQUIRED)

# Google Test
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

# Runner library (cpu_ref, gpu_ref, buffers)
add_library(fmha_runner STATIC
    src/runner/buffers.cpp
    src/runner/cpu_ref.cpp
    src/runner/gpu_ref.cpp)
target_include_directories(fmha_runner PUBLIC src)
target_include_directories(fmha_runner PUBLIC ${hip_INCLUDE_DIRS})
target_link_libraries(fmha_runner PUBLIC hip::host)
target_compile_options(fmha_runner PRIVATE -fopenmp)
target_link_options(fmha_runner PRIVATE -fopenmp)
set_source_files_properties(
    src/runner/gpu_ref.cpp
    PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET}")
set_source_files_properties(
    src/runner/buffers.cpp
    PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET}")

# Kernel library
add_library(fmha_kernel STATIC src/kernel/fmha_fwd_d64_kernel.cpp)
target_include_directories(fmha_kernel PUBLIC src)
target_link_libraries(fmha_kernel PUBLIC hip::device)
target_compile_options(fmha_kernel PRIVATE
    --offload-arch=${GPU_TARGET} --save-temps)
set_source_files_properties(
    src/kernel/fmha_fwd_d64_kernel.cpp
    PROPERTIES LANGUAGE HIP)

# Test: GPU ref vs CPU ref
add_executable(test_fmha_gpu_ref tests/test_fmha_gpu_ref.cpp)
target_link_libraries(test_fmha_gpu_ref PRIVATE fmha_runner GTest::gtest_main)
set_source_files_properties(tests/test_fmha_gpu_ref.cpp PROPERTIES LANGUAGE HIP)

# Test: kernel vs GPU ref
add_executable(test_fmha_fwd_d64 tests/test_fmha_fwd_d64.cpp)
target_link_libraries(test_fmha_fwd_d64 PRIVATE fmha_kernel fmha_runner GTest::gtest_main)
set_source_files_properties(tests/test_fmha_fwd_d64.cpp PROPERTIES LANGUAGE HIP)

# Benchmark
add_executable(bench_fmha_fwd src/bench_fmha_fwd.cpp)
target_link_libraries(bench_fmha_fwd PRIVATE fmha_kernel fmha_runner)
set_source_files_properties(src/bench_fmha_fwd.cpp PROPERTIES LANGUAGE HIP)
