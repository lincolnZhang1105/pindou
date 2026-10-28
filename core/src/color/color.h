// 颜色空间与色差（T0.2）。
// - sRGB ↔ CIE Lab：IEC 61966-2-1 的 sRGB 矩阵，D65 白点（CIE 1931 2°，xy = 0.3127, 0.3290），
//   与 colour-science 的 sRGB_to_XYZ + XYZ_to_Lab 默认参数一致，便于对拍。
// - OKLab：Björn Ottosson 的参考实现（公有领域）。
// - CIEDE2000：按 Sharma, Wu, Dalal (2005) 论文公式自行实现（kL = kC = kH = 1）。
// 全部用 double 计算，不开 fast-math，保证各平台结果一致。
#pragma once

#include <array>
#include <cstdint>

namespace bc::color {

struct Lab {
  double L = 0, a = 0, b = 0;
};

struct OKLab {
  double L = 0, a = 0, b = 0;
};

using RGB8 = std::array<uint8_t, 3>;

// 单个通道：sRGB 编码值（0–1）↔ 线性光
double srgb_to_linear(double c);
double linear_to_srgb(double c);

// 8 位 sRGB → Lab（D65）；以及 0–1 浮点 sRGB → Lab
Lab srgb8_to_lab(RGB8 rgb);
Lab srgb_to_lab(double r, double g, double b);
// Lab → 0–1 浮点 sRGB（未截断，可能越界）
std::array<double, 3> lab_to_srgb(const Lab& lab);
RGB8 lab_to_srgb8(const Lab& lab);  // 截断并四舍五入

OKLab srgb8_to_oklab(RGB8 rgb);
OKLab linear_srgb_to_oklab(double r, double g, double b);
std::array<double, 3> oklab_to_linear_srgb(const OKLab& c);

// CIEDE2000 色差
double delta_e_2000(const Lab& x, const Lab& y);
// OKLab 欧氏距离（粗筛用）
double delta_e_ok(const OKLab& x, const OKLab& y);

// 预计算 8 位 sRGB → 线性光的查表（像素级循环用）
const std::array<double, 256>& srgb8_linear_lut();

}  // namespace bc::color
