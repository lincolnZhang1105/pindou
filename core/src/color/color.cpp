#include "color/color.h"

#include <algorithm>
#include <cmath>

namespace bc::color {
namespace {

constexpr double kPi = 3.14159265358979323846;

// IEC 61966-2-1 sRGB → XYZ（与 colour-science RGB_COLOURSPACE_sRGB 相同）
constexpr double kM[3][3] = {
    {0.4124, 0.3576, 0.1805},
    {0.2126, 0.7152, 0.0722},
    {0.0193, 0.1192, 0.9505},
};

// D65 白点 XYZ，由 xy = (0.3127, 0.3290) 换算
constexpr double kXn = 0.3127 / 0.3290;
constexpr double kYn = 1.0;
constexpr double kZn = (1.0 - 0.3127 - 0.3290) / 0.3290;

// CIE 常数：epsilon = 216/24389，kappa = 24389/27
constexpr double kEps = 216.0 / 24389.0;
constexpr double kKappa = 24389.0 / 27.0;

double lab_f(double t) { return t > kEps ? std::cbrt(t) : (kKappa * t + 16.0) / 116.0; }

double lab_f_inv(double f) {
  const double f3 = f * f * f;
  return f3 > kEps ? f3 : (116.0 * f - 16.0) / kKappa;
}

Lab linear_to_lab(double r, double g, double b) {
  const double X = kM[0][0] * r + kM[0][1] * g + kM[0][2] * b;
  const double Y = kM[1][0] * r + kM[1][1] * g + kM[1][2] * b;
  const double Z = kM[2][0] * r + kM[2][1] * g + kM[2][2] * b;
  const double fx = lab_f(X / kXn), fy = lab_f(Y / kYn), fz = lab_f(Z / kZn);
  return {116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)};
}

// 3×3 矩阵求逆（只在初始化时算一次）
struct Inv3 {
  double m[3][3];
  Inv3() {
    const auto& a = kM;
    const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                       a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                       a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    m[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
    m[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
    m[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
    m[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
    m[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
    m[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
    m[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
    m[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
    m[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  }
};

const Inv3& inv_m() {
  static const Inv3 inv;
  return inv;
}

double deg(double rad) { return rad * 180.0 / kPi; }
double rad(double deg) { return deg * kPi / 180.0; }

}  // namespace

double srgb_to_linear(double c) {
  return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

double linear_to_srgb(double c) {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}

const std::array<double, 256>& srgb8_linear_lut() {
  static const std::array<double, 256> lut = [] {
    std::array<double, 256> t{};
    for (int i = 0; i < 256; ++i) t[i] = srgb_to_linear(i / 255.0);
    return t;
  }();
  return lut;
}

Lab srgb_to_lab(double r, double g, double b) {
  return linear_to_lab(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
}

Lab srgb8_to_lab(RGB8 rgb) {
  const auto& lut = srgb8_linear_lut();
  return linear_to_lab(lut[rgb[0]], lut[rgb[1]], lut[rgb[2]]);
}

std::array<double, 3> lab_to_srgb(const Lab& lab) {
  const double fy = (lab.L + 16.0) / 116.0;
  const double fx = fy + lab.a / 500.0;
  const double fz = fy - lab.b / 200.0;
  const double X = lab_f_inv(fx) * kXn, Y = lab_f_inv(fy) * kYn, Z = lab_f_inv(fz) * kZn;
  const auto& m = inv_m().m;
  const double r = m[0][0] * X + m[0][1] * Y + m[0][2] * Z;
  const double g = m[1][0] * X + m[1][1] * Y + m[1][2] * Z;
  const double b = m[2][0] * X + m[2][1] * Y + m[2][2] * Z;
  auto enc = [](double v) { return v < 0 ? -linear_to_srgb(-v) : linear_to_srgb(v); };
  return {enc(r), enc(g), enc(b)};
}

RGB8 lab_to_srgb8(const Lab& lab) {
  const auto c = lab_to_srgb(lab);
  RGB8 out{};
  for (int i = 0; i < 3; ++i) {
    out[i] = static_cast<uint8_t>(std::lround(std::clamp(c[i], 0.0, 1.0) * 255.0));
  }
  return out;
}

OKLab linear_srgb_to_oklab(double r, double g, double b) {
  const double l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const double m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const double s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
  const double l_ = std::cbrt(l), m_ = std::cbrt(m), s_ = std::cbrt(s);
  return {0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
          1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
          0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_};
}

std::array<double, 3> oklab_to_linear_srgb(const OKLab& c) {
  const double l_ = c.L + 0.3963377774 * c.a + 0.2158037573 * c.b;
  const double m_ = c.L - 0.1055613458 * c.a - 0.0638541728 * c.b;
  const double s_ = c.L - 0.0894841775 * c.a - 1.2914855480 * c.b;
  const double l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
  return {+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
          -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
          -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s};
}

OKLab srgb8_to_oklab(RGB8 rgb) {
  const auto& lut = srgb8_linear_lut();
  return linear_srgb_to_oklab(lut[rgb[0]], lut[rgb[1]], lut[rgb[2]]);
}

double delta_e_ok(const OKLab& x, const OKLab& y) {
  const double dL = x.L - y.L, da = x.a - y.a, db = x.b - y.b;
  return std::sqrt(dL * dL + da * da + db * db);
}

double delta_e_2000(const Lab& x, const Lab& y) {
  constexpr double k25_7 = 6103515625.0;  // 25^7
  const double C1 = std::hypot(x.a, x.b);
  const double C2 = std::hypot(y.a, y.b);
  const double Cbar = 0.5 * (C1 + C2);
  const double Cbar7 = std::pow(Cbar, 7.0);
  const double G = 0.5 * (1.0 - std::sqrt(Cbar7 / (Cbar7 + k25_7)));
  const double a1p = (1.0 + G) * x.a;
  const double a2p = (1.0 + G) * y.a;
  const double C1p = std::hypot(a1p, x.b);
  const double C2p = std::hypot(a2p, y.b);

  auto hue = [](double b, double ap) {
    if (b == 0.0 && ap == 0.0) return 0.0;
    double h = deg(std::atan2(b, ap));
    return h < 0.0 ? h + 360.0 : h;
  };
  const double h1p = hue(x.b, a1p);
  const double h2p = hue(y.b, a2p);

  const double dLp = y.L - x.L;
  const double dCp = C2p - C1p;
  const double CpProd = C1p * C2p;
  double dhp = 0.0;
  if (CpProd != 0.0) {
    dhp = h2p - h1p;
    if (dhp > 180.0) {
      dhp -= 360.0;
    } else if (dhp < -180.0) {
      dhp += 360.0;
    }
  }
  const double dHp = 2.0 * std::sqrt(CpProd) * std::sin(rad(dhp) / 2.0);

  const double Lbarp = 0.5 * (x.L + y.L);
  const double Cbarp = 0.5 * (C1p + C2p);
  double hbarp = h1p + h2p;
  if (CpProd != 0.0) {
    if (std::fabs(h1p - h2p) <= 180.0) {
      hbarp = 0.5 * (h1p + h2p);
    } else if (h1p + h2p < 360.0) {
      hbarp = 0.5 * (h1p + h2p + 360.0);
    } else {
      hbarp = 0.5 * (h1p + h2p - 360.0);
    }
  }

  const double T = 1.0 - 0.17 * std::cos(rad(hbarp - 30.0)) + 0.24 * std::cos(rad(2.0 * hbarp)) +
                   0.32 * std::cos(rad(3.0 * hbarp + 6.0)) - 0.20 * std::cos(rad(4.0 * hbarp - 63.0));
  const double dTheta = 30.0 * std::exp(-((hbarp - 275.0) / 25.0) * ((hbarp - 275.0) / 25.0));
  const double Cbarp7 = std::pow(Cbarp, 7.0);
  const double RC = 2.0 * std::sqrt(Cbarp7 / (Cbarp7 + k25_7));
  const double Lm50sq = (Lbarp - 50.0) * (Lbarp - 50.0);
  const double SL = 1.0 + 0.015 * Lm50sq / std::sqrt(20.0 + Lm50sq);
  const double SC = 1.0 + 0.045 * Cbarp;
  const double SH = 1.0 + 0.015 * Cbarp * T;
  const double RT = -std::sin(rad(2.0 * dTheta)) * RC;

  const double tL = dLp / SL, tC = dCp / SC, tH = dHp / SH;
  return std::sqrt(tL * tL + tC * tC + tH * tH + RT * tC * tH);
}

}  // namespace bc::color
