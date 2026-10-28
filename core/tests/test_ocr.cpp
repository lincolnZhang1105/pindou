// OCR 预处理与 Python 对拍（T0.11：C++ 与 PyTorch 输出误差 < 1e-3），以及 CTC 打分的基本性质。
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "nn/ocr.h"

#include <algorithm>

namespace {

nlohmann::json golden() {
  std::ifstream f(std::string(BC_TEST_DATA_DIR) + "/ocr_golden.json");
  std::stringstream ss;
  ss << f.rdbuf();
  return nlohmann::json::parse(ss.str());
}

}  // namespace

TEST(Ocr, InkCropMatchesPython) {
  const auto g = golden();
  const int h = g["image_shape"][0], w = g["image_shape"][1];
  cv::Mat img(h, w, CV_8UC1);
  const auto& px = g["image"];
  for (int i = 0; i < h * w; ++i) img.data[i] = static_cast<uint8_t>(px[i].get<int>());
  for (const auto& it : g["items"]) {
    const auto b = it["box"];
    const cv::Mat c = bc::nn::ink_crop(img, cv::Rect(b[0], b[1], b[2], b[3]));
    ASSERT_EQ(c.rows, 32);
    ASSERT_EQ(c.cols, it["w"].get<int>()) << "框 " << b.dump();
    const auto& want = it["crop"];
    double maxd = 0;
    for (int i = 0; i < c.rows * c.cols; ++i) maxd = std::max(maxd, std::fabs(c.at<float>(i / c.cols, i % c.cols) - want[i].get<double>()));
    EXPECT_LT(maxd, 1e-3) << "框 " << b.dump();
  }
}

TEST(Ocr, CtcLogprobOfCertainPath) {
  // 构造确定的输出：T=4，C=3（blank, 'A', 'B'），每步概率 1 的路径是 A A blank B → 文字 "AB"
  bc::nn::OcrText t;
  t.T = 4;
  t.C = 3;
  const double lo = std::log(1e-9);
  const int path[4] = {1, 1, 0, 2};
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) t.lp.push_back(static_cast<float>(k == path[i] ? 0.0 : lo));
  }
  EXPECT_NEAR(bc::nn::ctc_logprob(t, "AB", "AB"), 0.0, 1e-6);
  EXPECT_LT(bc::nn::ctc_logprob(t, "AB", "BA"), -10.0);
  EXPECT_LT(bc::nn::ctc_logprob(t, "AB", "A"), -10.0);
  EXPECT_EQ(bc::nn::ctc_logprob(t, "AB", "AC"), -std::numeric_limits<double>::infinity());
}

// T0.11 DoD：C++（ONNX Runtime）与 PyTorch 的模型输出误差 < 1e-3（golden 由 training/ocr/gen_golden.py 用 PyTorch 生成）
TEST(Ocr, ModelOutputMatchesPyTorch) {
  const auto g = golden();
  const std::string name = g.value("model", "");
  const std::string onnx = std::string(BC_REPO_ROOT) + "/core/models/" + name + ".onnx";
  if (name.empty() || !bc::nn::available() || !std::ifstream(onnx).good()) GTEST_SKIP() << "没有模型输出 golden 或模型文件";
  auto m = bc::nn::OcrModel::load(std::string(BC_REPO_ROOT) + "/core/models", name);
  ASSERT_TRUE(m);
  const int h = g["image_shape"][0], w = g["image_shape"][1];
  cv::Mat img(h, w, CV_8UC1);
  for (int i = 0; i < h * w; ++i) img.data[i] = static_cast<uint8_t>(g["image"][i].get<int>());
  for (const auto& it : g["items"]) {
    if (!it.contains("logp")) continue;
    const auto b = it["box"];
    const auto t = bc::nn::read_text(*m, img, cv::Rect(b[0], b[1], b[2], b[3]));
    const auto& want = it["logp"];
    ASSERT_EQ(t.lp.size(), want.size()) << b.dump();
    double maxd = 0;
    for (size_t k = 0; k < want.size(); ++k) maxd = std::max(maxd, std::fabs(t.lp[k] - want[k].get<double>()));
    EXPECT_LT(maxd, 1e-3) << "框 " << b.dump();
  }
}
