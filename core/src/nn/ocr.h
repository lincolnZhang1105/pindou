// 文字识别（T0.11 / T0.12）：图例色号、颗数、坐标数字。模型是 training/ocr 训练的 CRNN + CTC。
// 预处理 ink_crop 必须和 training/ocr/common.py 逐步一致（有单元测试对拍）。
#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "nn/model.h"

namespace bc::nn {

constexpr int kOcrInputH = 32;

// 文字框 → 模型输入（CV_32F，高 32）。见 training/ocr/common.py 的 ink_crop。
cv::Mat ink_crop(const cv::Mat& gray, const cv::Rect& box);

struct OcrModel {
  std::unique_ptr<Model> model;
  std::string charset;  // 下标 k ≥ 1 对应 charset[k-1]，0 是 CTC blank
  std::string name;

  // 从 models_dir 加载 <name>.onnx 和 <name>.json；文件不存在时返回空指针（只走颜色路径）
  static std::unique_ptr<OcrModel> load(const std::string& models_dir, const std::string& name);
};

struct OcrText {
  std::string text;   // 贪心解码结果
  double logp = 0;    // 贪心路径对应文字的 CTC 对数概率
  int T = 0, C = 0;
  std::vector<float> lp;  // [T, C] 对数概率（封闭词表打分用）
};

OcrText read_text(const OcrModel& m, const cv::Mat& gray, const cv::Rect& box);

// CTC 前向算法：log p(label | lp)。label 是字符串，字符不在字符集里时返回 -inf。
double ctc_logprob(const OcrText& t, const std::string& charset, const std::string& label);

}  // namespace bc::nn
