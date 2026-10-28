// 图像基础操作。所有处理都在原分辨率上进行（AGENTS 规则 4：保存图、截图全程不缩小）。
#pragma once

#include <opencv2/core.hpp>

#include <array>

#include "color/color.h"

namespace bc {

// RGBA → RGB（透明区域按白底合成）。返回 CV_8UC3，通道顺序 RGB。
cv::Mat rgba_to_rgb(const cv::Mat& rgba);

// RGB → 8 位亮度（Rec.601 权重，整数运算，结果确定）
cv::Mat rgb_to_gray(const cv::Mat& rgb);

// 页面底色：图像四条边上最常见的颜色（量化到 8 级后取众数，再求该桶的均值）
color::RGB8 estimate_page_background(const cv::Mat& rgb);

// 区域内像素的逐通道中位数
color::RGB8 median_rgb(const cv::Mat& rgb, const cv::Rect& r);

inline color::RGB8 px(const cv::Mat& rgb, int x, int y) {
  const auto* p = rgb.ptr<uint8_t>(y) + 3 * x;
  return {p[0], p[1], p[2]};
}

inline color::Lab lab_of(const color::RGB8& c) { return color::srgb8_to_lab(c); }

}  // namespace bc
