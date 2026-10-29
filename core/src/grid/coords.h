// 坐标数字（04 §5.1 ②d）：读出网格四周的行列号，用来确定网格的确切范围。
// 需要 OCR 模型；没有模型时不调用。
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "nn/ocr.h"

namespace bc {

struct CoordFit {
  bool ok = false;
  int first = 0, last = 0;  // 坐标指出的网格范围（lattice 的格子下标，闭区间）
  int support = 0;          // 支持这个拟合的坐标数字个数
  bool dense = false;       // 坐标是逐格标的（不是每 5 格一个）
  int k = 0;                // lattice 格子下标 = 坐标数字 + k（下标 i 处的坐标是 i - k；坐标可能从 0 或 1 开始）
};

struct CoordResult {
  CoordFit cols, rows;
  std::vector<int> label_rows;  // 列坐标数字（上下两侧）的中心落在哪些 lattice 行里：这些行是坐标行，不是图纸
  std::vector<int> label_cols;  // 行坐标数字（左右两侧）的中心落在哪些 lattice 列里
  std::vector<int> cols_with_word, rows_with_word;  // 坐标带里有字（不管读没读出来）的列 / 行
  std::vector<std::string> debug;
};

// lattice_x / lattice_y：网格检测追踪到的全部等距线；c0..c1、r0..r1：初步确定的网格范围（格子下标，闭区间）
CoordResult read_coords(const nn::OcrModel& ocr, const cv::Mat& gray, int page_bg_gray,
                        const std::vector<double>& lattice_x, const std::vector<double>& lattice_y, int c0, int c1,
                        int r0, int r1);

}  // namespace bc
