// T0.2 颜色库测试：Sharma CIEDE2000 34 组、colour-science 1 万组对拍、OKLab 与往返一致性。
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "color/color.h"

using namespace bc::color;

namespace {

std::string data_path(const char* name) { return std::string(BC_TEST_DATA_DIR) + "/" + name; }

}  // namespace

TEST(Ciede2000, SharmaTestData34Pairs) {
  std::ifstream f(data_path("ciede2000_sharma.txt"));
  ASSERT_TRUE(f.good()) << "缺少 Sharma 测试数据";
  int n = 0;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    Lab x, y;
    double expected;
    if (!(ss >> x.L >> x.a >> x.b >> y.L >> y.a >> y.b >> expected)) continue;
    ++n;
    const double d = delta_e_2000(x, y);
    EXPECT_LT(std::fabs(d - expected), 1e-4) << "第 " << n << " 组：得到 " << d << "，期望 " << expected;
    // 对称性
    EXPECT_NEAR(delta_e_2000(y, x), d, 1e-12);
  }
  EXPECT_EQ(n, 34);
}

TEST(Ciede2000, IdenticalColorsIsZero) {
  const Lab c{53.2, 80.1, 67.2};
  EXPECT_DOUBLE_EQ(delta_e_2000(c, c), 0.0);
}

// colour-science 0.4.x：sRGB_to_XYZ + XYZ_to_Lab（D65）+ delta_E(method="CIE 2000")
TEST(ColourScience, Golden10kPairs) {
  std::ifstream f(data_path("colour_golden.csv"));
  ASSERT_TRUE(f.good()) << "缺少 colour_golden.csv，运行 python -m bench.tools.gen_colour_golden 生成";
  std::string line;
  std::getline(f, line);  // 表头
  int n = 0;
  double max_lab_err = 0, max_de_err = 0;
  while (std::getline(f, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream ss(line);
    int r1, g1, b1, r2, g2, b2;
    Lab e1, e2;
    double de;
    ASSERT_TRUE(ss >> r1 >> g1 >> b1 >> r2 >> g2 >> b2 >> e1.L >> e1.a >> e1.b >> e2.L >> e2.a >> e2.b >> de);
    const Lab l1 = srgb8_to_lab({uint8_t(r1), uint8_t(g1), uint8_t(b1)});
    const Lab l2 = srgb8_to_lab({uint8_t(r2), uint8_t(g2), uint8_t(b2)});
    max_lab_err = std::max({max_lab_err, std::fabs(l1.L - e1.L), std::fabs(l1.a - e1.a), std::fabs(l1.b - e1.b),
                            std::fabs(l2.L - e2.L), std::fabs(l2.a - e2.a), std::fabs(l2.b - e2.b)});
    max_de_err = std::max(max_de_err, std::fabs(delta_e_2000(l1, l2) - de));
    ++n;
  }
  EXPECT_EQ(n, 10000);
  EXPECT_LT(max_lab_err, 1e-6);
  EXPECT_LT(max_de_err, 1e-6);
}

TEST(Lab, WhiteAndBlack) {
  const Lab w = srgb8_to_lab({255, 255, 255});
  EXPECT_NEAR(w.L, 100.0, 1e-6);
  const Lab k = srgb8_to_lab({0, 0, 0});
  EXPECT_NEAR(k.L, 0.0, 1e-9);
  EXPECT_NEAR(k.a, 0.0, 1e-9);
  EXPECT_NEAR(k.b, 0.0, 1e-9);
}

TEST(Lab, RoundTripAll8BitGreysAndRandom) {
  for (int v = 0; v < 256; ++v) {
    const RGB8 c{uint8_t(v), uint8_t(v), uint8_t(v)};
    EXPECT_EQ(lab_to_srgb8(srgb8_to_lab(c)), c);
  }
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> d(0, 255);
  for (int i = 0; i < 20000; ++i) {
    const RGB8 c{uint8_t(d(rng)), uint8_t(d(rng)), uint8_t(d(rng))};
    EXPECT_EQ(lab_to_srgb8(srgb8_to_lab(c)), c);
  }
}

TEST(OKLab, ReferenceValues) {
  // Ottosson 文中：白 → (1, 0, 0)；黑 → (0, 0, 0)
  const OKLab w = srgb8_to_oklab({255, 255, 255});
  EXPECT_NEAR(w.L, 1.0, 1e-4);
  EXPECT_NEAR(w.a, 0.0, 1e-4);
  EXPECT_NEAR(w.b, 0.0, 1e-4);
  const OKLab k = srgb8_to_oklab({0, 0, 0});
  EXPECT_NEAR(k.L, 0.0, 1e-12);
}

TEST(OKLab, RoundTrip) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> d(0.0, 1.0);
  for (int i = 0; i < 10000; ++i) {
    const double r = d(rng), g = d(rng), b = d(rng);
    const auto back = oklab_to_linear_srgb(linear_srgb_to_oklab(r, g, b));
    EXPECT_NEAR(back[0], r, 1e-6);
    EXPECT_NEAR(back[1], g, 1e-6);
    EXPECT_NEAR(back[2], b, 1e-6);
  }
}
