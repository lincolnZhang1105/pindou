// 格内文字证据（04 §5.1 ④）：颜色拿不准的格子，用 OCR 读格内的色号，和颜色证据的对数概率相加。
// 只读「难格子」：全读太慢（104×104 有上万格），而颜色明确的格子读字也不会改变结果。
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "classify/assign.h"
#include "classify/cells.h"
#include "nn/ocr.h"

namespace bc {

struct CellRef {
  const cv::Mat* gray = nullptr;  // 这一格所在页的灰度图
  CellRect rect{};
};

struct TextEvidenceParams {
  double min_pitch = 12.0;   // 格子小于这个边长时字读不出来，不读
  double gap_de = 5.0;       // 最近两个图例色的 ΔE 差小于它，算难格子
  float conf_thr = 0.95f;    // 或者颜色置信度低于它
  int max_cells = 600;       // 每张图最多读多少格
  double text_weight = 1.0;  // 文字对数概率的权重
};

// class_codes[k]：第 k 类的色号（规范化后）；空字符串表示不知道（不参与文字比较）。
// 返回被文字证据改判的格子数。
int apply_text_evidence(Assignment& a, const std::vector<CellSample>& samples, const std::vector<CellRef>& refs,
                        const std::vector<std::string>& class_codes, const nn::OcrModel& ocr,
                        const AssignParams& p = {}, const TextEvidenceParams& tp = {});

}  // namespace bc
