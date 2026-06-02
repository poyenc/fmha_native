cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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
add_library(fmha_kernel STATIC src/fused/kernel.cpp)
target_include_directories(fmha_kernel PUBLIC src)
target_link_libraries(fmha_kernel PUBLIC hip::device)
target_compile_options(fmha_kernel PRIVATE
    --offload-arch=${GPU_TARGET} --save-temps
    "SHELL:-mllvm -amdgpu-early-inline-all=true"
    "SHELL:-mllvm -amdgpu-function-calls=false"
    "SHELL:-mllvm --lsr-drop-solution=1"
    "SHELL:-mllvm -enable-post-misched=1"
    -fbracket-depth=1024
    -fno-offload-uniform-block
    -fgpu-flush-denormals-to-zero)
set_source_files_properties(
    src/fused/kernel.cpp
    PROPERTIES LANGUAGE HIP)

# Test: GPU ref vs CPU ref (incl. Stage-1 split-K oracle trust:
# cpu_ref_combine is compiled in here because it lives in components_ref and is
# not part of fmha_runner).
add_executable(test_fmha_gpu_ref tests/test_fmha_gpu_ref.cpp src/components_ref/cpu_ref_combine.cpp)
target_link_libraries(test_fmha_gpu_ref PRIVATE fmha_runner GTest::gtest_main)
target_link_options(test_fmha_gpu_ref PRIVATE -fopenmp)
set_source_files_properties(tests/test_fmha_gpu_ref.cpp PROPERTIES LANGUAGE HIP)

# Test: kernel vs GPU ref
add_executable(test_fmha_fwd_d64 tests/test_fmha_fwd_d64.cpp)
target_link_libraries(test_fmha_fwd_d64 PRIVATE fmha_kernel fmha_runner GTest::gtest_main)
target_link_options(test_fmha_fwd_d64 PRIVATE -fopenmp)
set_source_files_properties(tests/test_fmha_fwd_d64.cpp PROPERTIES LANGUAGE HIP)

# Test: K LDS staging (Phase 1 Kernel 1)
add_executable(test_k_lds tests/test_k_lds.cpp src/components_ref/cpu_ref_k_lds.cpp)
target_link_libraries(test_k_lds PRIVATE fmha_runner GTest::gtest)
set_source_files_properties(tests/test_k_lds.cpp PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET} --save-temps")

# Test: QK GEMM (Phase 1 Kernel 2)
add_executable(test_qk_gemm tests/test_qk_gemm.cpp src/components_ref/cpu_ref_qk_gemm.cpp)
target_link_libraries(test_qk_gemm PRIVATE fmha_runner GTest::gtest)
set_source_files_properties(tests/test_qk_gemm.cpp PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET} --save-temps")

# Test: Row Max (Phase 1 Kernel 3)
add_executable(test_row_max tests/test_row_max.cpp src/components_ref/cpu_ref_row_max.cpp src/components_ref/cpu_ref_qk_gemm.cpp)
target_link_libraries(test_row_max PRIVATE fmha_runner GTest::gtest)
set_source_files_properties(tests/test_row_max.cpp PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET} --save-temps")

# Test: Softmax (Phase 1 Kernel 4)
add_executable(test_softmax tests/test_softmax.cpp src/components_ref/cpu_ref_softmax.cpp src/components_ref/cpu_ref_qk_gemm.cpp)
target_link_libraries(test_softmax PRIVATE fmha_runner GTest::gtest)
set_source_files_properties(tests/test_softmax.cpp PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET} --save-temps")

# Test: V LDS staging (Phase 1 Kernel 5)
add_executable(test_v_lds tests/test_v_lds.cpp src/components_ref/cpu_ref_v_lds.cpp)
target_link_libraries(test_v_lds PRIVATE fmha_runner GTest::gtest)
set_source_files_properties(tests/test_v_lds.cpp PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET} --save-temps")

# Test: PV GEMM (Phase 1 Kernel 6)
add_executable(test_pv_gemm tests/test_pv_gemm.cpp src/components_ref/cpu_ref_pv_gemm.cpp src/components_ref/cpu_ref_qk_gemm.cpp src/components_ref/cpu_ref_softmax.cpp)
target_link_libraries(test_pv_gemm PRIVATE fmha_runner GTest::gtest)
set_source_files_properties(tests/test_pv_gemm.cpp PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET} --save-temps")

# Test: Epilog (Phase 1 Kernel 7)
add_executable(test_epilog tests/test_epilog.cpp src/components_ref/cpu_ref_epilog.cpp src/components_ref/cpu_ref_qk_gemm.cpp src/components_ref/cpu_ref_softmax.cpp src/components_ref/cpu_ref_pv_gemm.cpp)
target_link_libraries(test_epilog PRIVATE fmha_runner GTest::gtest)
set_source_files_properties(tests/test_epilog.cpp PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET} --save-temps")

# Test: split-K combine (Stage-2 RED until Task 4 lands the kernel).
# Source A uses cpu_ref_combine (lives in components_ref, NOT in fmha_runner, so
# it is compiled in explicitly — same pattern as test_epilog's refs).  Source B
# uses cpu_ref_split, which IS part of fmha_runner, so no extra source for it.
# NOTE: this target is deliberately UN-LINKABLE until Task 4 defines
# fmha_fwd_d64_bf16_combine.  run-gates.sh builds named targets only (it does NOT
# build "all"), so this dangling target does not break the gate.
add_executable(test_combine tests/test_combine.cpp
    src/components_ref/cpu_ref_combine.cpp)
target_link_libraries(test_combine PRIVATE fmha_kernel fmha_runner GTest::gtest_main)
target_link_options(test_combine PRIVATE -fopenmp)
# Both sources compile as HIP: test_combine.cpp launches a kernel, and
# cpu_ref_combine.cpp must also be HIP here because linking fmha_kernel pulls in
# hip::device's INTERFACE --offload-arch flag, which the plain C++ compiler
# rejects.  (test_fmha_gpu_ref compiles the same ref as C++ because it does NOT
# link fmha_kernel.)
set_source_files_properties(
    tests/test_combine.cpp
    src/components_ref/cpu_ref_combine.cpp
    PROPERTIES LANGUAGE HIP)
# (No gtest_discover_tests: this repo's tests are run by path via run-gates.sh,
#  and GoogleTest's CTest integration module is not include()d here.)

# Test: split-K PRODUCER (Stage-2 RED until Task 7 lands the split globals).
# Both oracles it uses — cpu_ref_split AND gpu_ref_split — live in src/runner/
# (cpu_ref.cpp / gpu_ref.cpp), which IS fmha_runner, so linking fmha_runner
# suffices: NO extra components_ref source is compiled (do not double-compile).
# NOTE: this target is deliberately UN-LINKABLE until Task 7 defines
# fmha_fwd_d64_bf16_msk0_split / _msk1_split.  run-gates.sh builds named targets
# only (it does NOT build "all"), so this dangling target does not break the
# gate.  No gtest_discover_tests (this repo runs tests by path, like test_combine).
add_executable(test_split tests/test_split.cpp)
target_link_libraries(test_split PRIVATE fmha_kernel fmha_runner GTest::gtest_main)
target_link_options(test_split PRIVATE -fopenmp)
# test_split.cpp launches a kernel and links fmha_kernel (which pulls in
# hip::device's INTERFACE --offload-arch flag), so it must compile as HIP.
set_source_files_properties(tests/test_split.cpp PROPERTIES LANGUAGE HIP)

# Benchmark
add_executable(bench_fmha_fwd src/bench_fmha_fwd.cpp)
target_link_libraries(bench_fmha_fwd PRIVATE fmha_kernel fmha_runner)
set_source_files_properties(src/bench_fmha_fwd.cpp PROPERTIES LANGUAGE HIP)
