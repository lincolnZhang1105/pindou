// 逐格分类（04 §5.1 ④ 的颜色证据部分，v0）：格子底色和图例色块比 ΔE2000，
// 白豆和空格靠「格内有没有文字」区分。没有对上图例的格子按颜色聚类成新类。
#pragma once

#include <vector>

#include "classify/cells.h"
#include "color/color.h"
#include "legend/legend.h"

namespace bc {

struct ColorClass {
  color::RGB8 rgb{};
  color::Lab lab;
  int legend_index = -1;  // -1 表示图例里没有、从格子里聚类出来的颜色
  bool page_colored = false;  // 和页面底色同色、靠格内文字认出来的豆子（通常是白豆，图例色块看不见）
  int cells = 0;
};

struct Assignment {
  std::vector<int> cls;     // 每格：-1 空格，否则是 classes 的下标
  std::vector<float> conf;  // 每格置信度
  std::vector<ColorClass> classes;
  bool has_empty_ref = false;
  color::RGB8 empty_rgb{};
};

struct AssignParams {
  double empty_de = 5.0;     // 和空格底色的 ΔE 小于它，视为同色
  double match_de = 12.0;    // 和最近图例色块的 ΔE 大于它，视为图例里没有的颜色
  double cluster_de = 4.0;   // 未匹配格子聚类的 ΔE
  double sigma = 2.5;        // 概率换算的温度（ΔE 单位）
};

Assignment assign_cells(const std::vector<CellSample>& samples, const LegendResult& legend,
                        const color::RGB8& page_bg, const AssignParams& p = {});

// 图例颗数约束（04 §5.1 ⑤）：expected[k] 是第 k 类的图例颗数（-1 表示不知道）。
// 颗数对不上、且差得不多（≤ 3% 豆数）时，把「多出来的类」里离「缺的类」最近的格子挪过去，
// 按似然代价从小到大挪，挪过的格子置信度降到 0.5（交给用户复核）。返回挪动的格子数。
int enforce_counts(Assignment& a, const std::vector<CellSample>& samples, const std::vector<int>& expected,
                   const AssignParams& p = {});

// 按置信度从低到高挑不确定格，受标记率上限约束（04 §0.1 第 3 条：≤ 豆数的 2%，且每张图 ≤ 30 格）
std::vector<int> pick_uncertain(const Assignment& a, double tau, int max_per_image = 30, double max_ratio = 0.02);

}  // namespace bc
