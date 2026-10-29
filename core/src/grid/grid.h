// 网格检测（04 §5.1 ②）。v0：投影 + 自相关估计格距 → 逐条线局部追踪（容忍累积漂移）
// → 按线段强度找出「四边都有线」的格子连通块 → 去掉坐标栏 → 得到图纸网格。
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "common/common.h"

namespace bc {

struct GridResult {
  bool found = false;
  std::string fail_reason;

  std::vector<double> xs, ys;  // 图纸网格线（cols+1 / rows+1 条），原图像素坐标
  double pitch_x = 0, pitch_y = 0;

  // 诊断信息
  std::vector<double> lattice_x, lattice_y;  // 追踪到的全部等距线
  double seg_threshold = 0;
  double period9 = 0, period3 = 0;  // 调试：两种腐蚀长度各自估出的格距
  bool header_stripped[4] = {false, false, false, false};  // 上、下、左、右坐标栏
  // 最外一圈像坐标栏（底色一致、几乎每格有字），只是字形对比分不开：三位数坐标在小格子里和同一色号连成一排很像。
  // 交给有 OCR 的引擎判断是不是一串连续的数字（上、下、左、右）
  bool header_candidate[4] = {false, false, false, false};
  double frame_fill = 0;                                   // 网格范围内「四边有线」格子的比例
  std::vector<cv::Rect> components;                        // 有线格子的连通块（像素坐标，按面积降序）
  std::vector<std::string> border_map;    // 调试：每个 lattice 格子有几条边有线（0–4），每行一个字符串
  std::vector<std::string> header_debug;  // 调试：坐标栏判断的统计（上、下、左、右，每轮 4 条）
  std::vector<std::string> comb_debug;    // 调试：各候选格距的梳子检验结果

  int cols() const { return xs.empty() ? 0 : static_cast<int>(xs.size()) - 1; }
  int rows() const { return ys.empty() ? 0 : static_cast<int>(ys.size()) - 1; }
};

GridResult detect_grid(const cv::Mat& rgb, const cv::Mat& gray, const Progress& progress);

// 以下为内部步骤，暴露出来便于单元测试
cv::Mat ridge_map(const cv::Mat& gray, bool vertical_lines);
// 逐通道求线响应再取最大：线和格子亮度相近、只是色相不同时（如 3_0 的浅灰线压在肤色格上），灰度图里看不出线
cv::Mat ridge_map_rgb(const cv::Mat& rgb, bool vertical_lines);
// 返回格距（亚像素）；找不到时返回 0。score 输出归一化自相关峰值
double estimate_period(const std::vector<double>& profile_a, const std::vector<double>& profile_b, int min_p,
                       int max_p, double* score = nullptr);
std::vector<double> track_lines(const std::vector<double>& profile, double period);

}  // namespace bc
