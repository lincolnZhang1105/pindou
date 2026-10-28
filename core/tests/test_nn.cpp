// ONNX Runtime 封装的冒烟测试，以及「初始化前关闭遥测」的约束（AGENTS 规则 5）。
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "common/common.h"
#include "nn/model.h"

TEST(Nn, AffineModelRuns) {
  if (!bc::nn::available()) GTEST_SKIP() << "构建时没有带 ONNX Runtime";
  bc::nn::Model m(std::string(BC_TEST_DATA_DIR) + "/affine.onnx");
  const auto out = m.run({0.f, 1.f, 2.f, -1.f, 0.5f, 3.f}, {2, 3});
  ASSERT_EQ(out.shape, (std::vector<int64_t>{2, 3}));
  const float want[] = {1.f, 3.f, 5.f, -1.f, 2.f, 7.f};
  for (int i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(out.data[i], want[i]);
}

TEST(Nn, TelemetryDisabledBeforeInit) {
  if (!bc::nn::available()) GTEST_SKIP();
  bc::nn::Model m(std::string(BC_TEST_DATA_DIR) + "/affine.onnx");
  const char* v = std::getenv("ORT_DISABLE_TELEMETRY");
  ASSERT_NE(v, nullptr);
  EXPECT_STREQ(v, "1");
}

TEST(Nn, MissingModelIsInvalidInput) {
  if (!bc::nn::available()) GTEST_SKIP();
  try {
    bc::nn::Model m("does/not/exist.onnx");
    FAIL() << "应该抛异常";
  } catch (const bc::Error& e) {
    EXPECT_EQ(e.code(), "invalid_input");
  }
}
