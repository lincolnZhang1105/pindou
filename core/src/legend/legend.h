// 图例解析（04 §5.1 ③）。v0 只做「色块」：在网格以外找出纯色色块，取它们的颜色作为封闭色板。
// 色号和颗数的文字识别在 v1 由 CRNN 完成（T0.11 / T0.12），v0 先记录文字所在的区域。
#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <vector>

#include "color/color.h"

namespace bc {

struct LegendItem {
  cv::Rect swatch;             // 色块外接框
  color::RGB8 rgb{};           // 色块颜色（去掉色块里的文字后取中位数）
  color::Lab lab;
  int area = 0;                // 色块的纯色像素数
  bool text_only = false;      // 没有可见色块、只有文字的条目（白底上的白色色块）；swatch 此时是文字框
  std::optional<cv::Rect> code_box;   // 色号文字框（通常在色块里面）
  std::optional<cv::Rect> count_box;  // 颗数文字框（横排图例在色块下方，表格式在右侧）
  std::vector<cv::Rect> near_words;   // 分给这个色块的全部文字框（OCR 校验角色时的备选）
  std::optional<std::string> code;    // v1：OCR 读出的色号
  std::optional<int> count;           // v1：OCR 读出的颗数
};

struct LegendResult {
  std::vector<LegendItem> items;
  std::string layout;  // "row" / "table" / "none"
  std::vector<cv::Rect> debug_words;
  std::vector<cv::Rect> debug_regions;  // 调试：通过形状过滤的候选色块  // 调试：图例区找到的文字块
  int bands = 0;       // 色块在纵向上分成几段（间隔超过 3 个色块高）。一张图纸的图例是连成一块的，多段说明是拼图（T7）
};

// exclude：网格及坐标所在区域（含坐标栏），不在这里找色块。page_bg：页面底色。
LegendResult detect_legend(const cv::Mat& rgb, const cv::Rect& exclude, const color::RGB8& page_bg);

}  // namespace bc
