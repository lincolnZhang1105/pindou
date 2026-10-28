// 引擎入口：C ABI 之下的 C++ 接口。结果直接组装成 03 §3.2 的 JSON。
#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/common.h"
#include "nn/ocr.h"
#include "palette/palette.h"

namespace bc {

using ojson = nlohmann::ordered_json;

struct Context {
  PaletteSet palettes;
  std::string models_dir;
  std::unique_ptr<nn::OcrModel> ocr;  // 没有模型文件时为空，只走颜色路径
};

struct RecognizeRequest {
  std::vector<cv::Mat> images;            // RGBA，不拷贝
  std::optional<std::string> palette_hint;
  std::optional<nlohmann::json> pages;    // bc_recognize_with_grid 时为 pages 数组（03 §3.3）
};

ojson recognize(const Context& ctx, const RecognizeRequest& req, const Progress& progress);
ojson generate(const Context& ctx, const cv::Mat& img, const nlohmann::json& params, const Progress& progress);

// 调试开关（只给 CLI 和测试用，不在 C ABI 里）：打开后结果 JSON 里附带 debug 字段
void set_debug(bool on);

}  // namespace bc
