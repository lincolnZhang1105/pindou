// 单格采样：底色、有没有文字、文字极性。网格检测（判断坐标栏）和逐格分类都会用到。
#pragma once

#include <opencv2/core.hpp>

#include "color/color.h"

namespace bc {

struct CellRect {
  double x0, x1, y0, y1;  // 网格线坐标（像素，可为小数）
};

struct CellSample {
  color::RGB8 bg{};       // 底色：格子内部去掉中心区域后的逐通道中位数（避开居中的文字）
  color::Lab bg_lab;
  float text_frac = 0;    // 中心区域里和底色亮度差超过阈值的像素比例
  int text_px = 0;
  bool has_text = false;
  int text_polarity = 0;  // -1 深色字，+1 浅色字，0 无字
  float bg_spread = 0;    // 底色环上的亮度离散度（MAD），衡量是否是纯色格
  int text_contrast = 0;  // 文字像素和底色的亮度差（中位数）
};

struct SampleParams {
  double margin_frac = 0.16;   // 去掉网格线的边距（相对格宽）
  double center_frac = 0.62;   // 文字区域：内部中心这一比例的方框
  int text_delta = 48;         // 判定为文字像素的亮度差
  float min_text_frac = 0.035f;
  int min_text_px = 3;
};

CellSample sample_cell(const cv::Mat& rgb, const cv::Mat& gray, const CellRect& r, const SampleParams& p = {});

// 文字掩码，缩放到 size×size 的浮点图（0/1），用于比较不同格子里的字形是否相同
cv::Mat text_glyph(const cv::Mat& gray, const CellRect& r, const CellSample& s, int size = 16,
                   const SampleParams& p = {});

// 两个字形的相似度（归一化相关，-1..1）
double glyph_similarity(const cv::Mat& a, const cv::Mat& b);

}  // namespace bc
