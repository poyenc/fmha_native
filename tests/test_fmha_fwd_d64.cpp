#include <gtest/gtest.h>
#include "runner/gpu_ref.hpp"
#include "runner/cpu_ref.hpp"
#include "test_configs.hpp"
#include "test_params.hpp"

class FmhaFwdD64Test : public ::testing::TestWithParam<TestCase> {};

TEST_P(FmhaFwdD64Test, MatchesGpuRef) {
    GTEST_SKIP() << "Kernel not implemented yet";
}

INSTANTIATE_TEST_SUITE_P(Full, FmhaFwdD64Test,
    ::testing::ValuesIn(kAllFull),
    [](const auto& info) { return info.param.name; });
