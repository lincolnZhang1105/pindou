// 逐格分类与图例颗数约束的单元测试
#include <gtest/gtest.h>

#include "classify/assign.h"

namespace {

bc::CellSample cell(bc::color::RGB8 rgb, bool text = true) {
  bc::CellSample s;
  s.bg = rgb;
  s.bg_lab = bc::color::srgb8_to_lab(rgb);
  s.has_text = text;
  s.text_px = text ? 10 : 0;
  s.text_frac = text ? 0.2f : 0.0f;
  return s;
}

bc::LegendItem item(bc::color::RGB8 rgb) {
  bc::LegendItem it;
  it.rgb = rgb;
  it.lab = bc::color::srgb8_to_lab(rgb);
  return it;
}

}  // namespace

TEST(Assign, NearestLegendColorAndEmpty) {
  bc::LegendResult lg;
  lg.items = {item({200, 30, 30}), item({30, 30, 200})};
  std::vector<bc::CellSample> s = {cell({201, 31, 29}), cell({29, 31, 201}), cell({255, 255, 255}, false)};
  const auto a = bc::assign_cells(s, lg, {255, 255, 255});
  EXPECT_EQ(a.cls[0], 0);
  EXPECT_EQ(a.cls[1], 1);
  EXPECT_EQ(a.cls[2], -1);
}

TEST(Assign, WhiteBeadWithTextIsNotEmpty) {
  bc::LegendResult lg;
  lg.items = {item({226, 224, 236})};  // 浅紫（D16），没有白色色块
  std::vector<bc::CellSample> s = {cell({254, 254, 254}, true), cell({255, 255, 255}, false), cell({226, 224, 236})};
  const auto a = bc::assign_cells(s, lg, {255, 255, 255});
  EXPECT_EQ(a.cls[1], -1);
  EXPECT_EQ(a.cls[2], 0);
  ASSERT_GE(a.cls[0], 1);  // 单独成一类，而不是归到浅紫
  EXPECT_TRUE(a.classes[a.cls[0]].page_colored);
}

TEST(Assign, EnforceCountsMovesClosestCell) {
  // 两个相近的红：A (200,30,30)，B (190,40,40)。图例说 A 2 颗、B 2 颗，颜色分类给了 A 3 颗、B 1 颗。
  bc::LegendResult lg;
  lg.items = {item({200, 30, 30}), item({190, 40, 40})};
  std::vector<bc::CellSample> s = {cell({200, 30, 30}), cell({199, 31, 31}), cell({196, 34, 34}), cell({190, 40, 40})};
  auto a = bc::assign_cells(s, lg, {255, 255, 255});
  ASSERT_EQ(a.cls, (std::vector<int>{0, 0, 0, 1}));
  const int moved = bc::enforce_counts(a, s, {2, 2});
  EXPECT_EQ(moved, 1);
  EXPECT_EQ(a.cls[2], 1);  // 最接近 B 的那一格被挪过去
  EXPECT_LE(a.conf[2], 0.5f);
  EXPECT_EQ(a.classes[0].cells, 2);
  EXPECT_EQ(a.classes[1].cells, 2);
}

TEST(Assign, EnforceCountsSkipsLargeMismatch) {
  bc::LegendResult lg;
  lg.items = {item({200, 30, 30}), item({30, 30, 200})};
  std::vector<bc::CellSample> s(100, cell({200, 30, 30}));
  auto a = bc::assign_cells(s, lg, {255, 255, 255});
  // 图例说蓝色 50 颗：差得太多（多半是图例读错了），不动
  EXPECT_EQ(bc::enforce_counts(a, s, {50, 50}), 0);
}
