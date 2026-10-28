#include "classify/assign.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace bc {

Assignment assign_cells(const std::vector<CellSample>& samples, const LegendResult& legend,
                        const color::RGB8& page_bg, const AssignParams& p) {
  Assignment a;
  const int n = static_cast<int>(samples.size());
  a.cls.assign(n, -1);
  a.conf.assign(n, 0.0f);

  for (size_t i = 0; i < legend.items.size(); ++i) {
    ColorClass c;
    c.rgb = legend.items[i].rgb;
    c.lab = legend.items[i].lab;
    c.legend_index = static_cast<int>(i);
    a.classes.push_back(c);
  }

  // 空格底色：没有文字、且和页面底色相同的格子存在时，以页面底色为空格色
  const color::Lab bg_lab = color::srgb8_to_lab(page_bg);
  int empty_like = 0;
  for (const auto& s : samples) {
    if (!s.has_text && color::delta_e_2000(s.bg_lab, bg_lab) < p.empty_de) ++empty_like;
  }
  if (empty_like > 0) {
    a.has_empty_ref = true;
    a.empty_rgb = page_bg;
  }
  // 图例里有没有和空格同色的色块（白豆）
  bool legend_has_empty_color = false;
  for (const auto& c : a.classes) {
    if (a.has_empty_ref && color::delta_e_2000(c.lab, bg_lab) < p.empty_de) legend_has_empty_color = true;
  }

  const double two_s2 = 2.0 * p.sigma * p.sigma;
  std::vector<int> unmatched;
  std::vector<uint8_t> is_bg_bead(n, 0);
  for (int i = 0; i < n; ++i) {
    const auto& s = samples[i];
    const bool like_empty = a.has_empty_ref && color::delta_e_2000(s.bg_lab, bg_lab) < p.empty_de;
    if (like_empty && !s.has_text) {
      a.cls[i] = -1;
      // 图例里有白豆、而格子又太小读不出字时，空格判断没那么可靠
      a.conf[i] = legend_has_empty_color ? 0.9f : 0.99f;
      continue;
    }
    int k1 = -1;
    double d1 = 1e9, z = 0;
    std::vector<double> d(a.classes.size());
    for (size_t k = 0; k < a.classes.size(); ++k) {
      d[k] = color::delta_e_2000(s.bg_lab, a.classes[k].lab);
      if (d[k] < d1) {
        d1 = d[k];
        k1 = static_cast<int>(k);
      }
    }
    // 有字、底色就是页面底色、而且比任何图例色都更接近页面底色：这是和背景同色的豆子（通常是白豆），
    // 图例里的白色色块往往看不见（白底上画白块），不能把它硬归到最近的浅色色号上
    const bool bg_bead = s.has_text && a.has_empty_ref && color::delta_e_2000(s.bg_lab, bg_lab) < std::min(d1, p.empty_de);
    if (k1 < 0 || d1 > p.match_de || bg_bead) {
      unmatched.push_back(i);
      is_bg_bead[i] = bg_bead;
      continue;
    }
    for (double dk : d) z += std::exp(-(dk * dk - d1 * d1) / two_s2);
    a.cls[i] = k1;
    a.conf[i] = static_cast<float>(1.0 / z);
  }

  // 图例里没有的颜色：贪心聚类（按格子顺序，结果确定）
  const int first_new = static_cast<int>(a.classes.size());
  std::vector<std::vector<color::Lab>> members;
  for (int i : unmatched) {
    const auto& s = samples[i];
    int hit = -1;
    double best = p.cluster_de;
    for (int k = first_new; k < static_cast<int>(a.classes.size()); ++k) {
      const double dk = color::delta_e_2000(s.bg_lab, a.classes[k].lab);
      if (dk < best) {
        best = dk;
        hit = k;
      }
    }
    if (hit < 0) {
      ColorClass c;
      c.rgb = s.bg;
      c.lab = s.bg_lab;
      a.classes.push_back(c);
      members.emplace_back();
      hit = static_cast<int>(a.classes.size()) - 1;
    }
    members[hit - first_new].push_back(s.bg_lab);
    a.cls[i] = hit;
    if (is_bg_bead[i]) {
      // 白豆：是不是豆子完全取决于格内有没有字，字越清楚越可信（勉强过阈值的交给用户复核）
      a.classes[hit].page_colored = true;
      const double clarity = std::min(1.0, s.text_frac / (3.0 * SampleParams{}.min_text_frac));
      a.conf[i] = static_cast<float>(0.6 + 0.39 * clarity);
    } else {
      a.conf[i] = 0.5f;  // 图例里没有这个颜色，交给用户复核
    }
  }
  // 新类的颜色取成员的 Lab 均值
  for (size_t m = 0; m < members.size(); ++m) {
    color::Lab mean;
    for (const auto& l : members[m]) {
      mean.L += l.L;
      mean.a += l.a;
      mean.b += l.b;
    }
    const double k = static_cast<double>(members[m].size());
    mean = {mean.L / k, mean.a / k, mean.b / k};
    auto& c = a.classes[first_new + m];
    c.lab = mean;
    c.rgb = color::lab_to_srgb8(mean);
  }
  for (int i = 0; i < n; ++i) {
    if (a.cls[i] >= 0) ++a.classes[a.cls[i]].cells;
  }
  return a;
}

int enforce_counts(Assignment& a, const std::vector<CellSample>& samples, const std::vector<int>& expected,
                   const AssignParams& p) {
  const int K = static_cast<int>(a.classes.size());
  if (static_cast<int>(expected.size()) != K) return 0;
  std::vector<int> actual(K, 0);
  int beads = 0;
  for (int c : a.cls) {
    if (c >= 0) {
      ++actual[c];
      ++beads;
    }
  }
  int gap = 0;
  for (int k = 0; k < K; ++k) {
    if (expected[k] >= 0) gap += std::abs(expected[k] - actual[k]);
  }
  if (gap == 0 || gap > std::max(4, beads * 3 / 100)) return 0;
  std::vector<int> surplus(K, 0), deficit(K, 0);
  for (int k = 0; k < K; ++k) {
    if (expected[k] < 0) continue;
    surplus[k] = std::max(0, actual[k] - expected[k]);
    deficit[k] = std::max(0, expected[k] - actual[k]);
  }
  struct Move {
    double cost;
    int cell, from, to;
  };
  std::vector<Move> moves;
  const double two_s2 = 2.0 * p.sigma * p.sigma;
  const double min_frac = SampleParams{}.min_text_frac;
  for (int i = 0; i < static_cast<int>(a.cls.size()); ++i) {
    const int from = a.cls[i];
    for (int to = 0; to < K; ++to) {
      if (!deficit[to] || to == from) continue;
      if (from >= 0) {
        if (!surplus[from]) continue;
        const double da = color::delta_e_2000(samples[i].bg_lab, a.classes[from].lab);
        const double db = color::delta_e_2000(samples[i].bg_lab, a.classes[to].lab);
        if (db > p.match_de) continue;
        moves.push_back({(db * db - da * da) / two_s2, i, from, to});
      } else if (a.classes[to].page_colored || color::delta_e_2000(samples[i].bg_lab, a.classes[to].lab) < p.empty_de) {
        // 空格 → 和背景同色的豆子（白豆的字太淡没检测出来）：字越接近阈值越可能
        if (samples[i].text_px == 0) continue;
        moves.push_back({3.0 * (1.0 - std::min(1.0, samples[i].text_frac / min_frac)), i, from, to});
      }
    }
  }
  std::sort(moves.begin(), moves.end(), [](const Move& x, const Move& y) {
    return x.cost != y.cost ? x.cost < y.cost : x.cell < y.cell;
  });
  std::vector<uint8_t> moved(a.cls.size(), 0);
  int n = 0;
  for (const auto& m : moves) {
    if (m.cost > 8.0) break;
    if (moved[m.cell] || !deficit[m.to] || (m.from >= 0 && !surplus[m.from])) continue;
    a.cls[m.cell] = m.to;
    a.conf[m.cell] = std::min(a.conf[m.cell], 0.5f);
    moved[m.cell] = 1;
    --deficit[m.to];
    if (m.from >= 0) {
      --surplus[m.from];
      --a.classes[m.from].cells;
    }
    ++a.classes[m.to].cells;
    ++n;
  }
  return n;
}

std::vector<int> pick_uncertain(const Assignment& a, double tau, int max_per_image, double max_ratio) {
  int beads = 0;
  for (int c : a.cls) beads += c >= 0;
  const int cap = std::min(max_per_image, static_cast<int>(std::floor(max_ratio * beads)));
  std::vector<int> idx;
  for (int i = 0; i < static_cast<int>(a.conf.size()); ++i) {
    if (a.conf[i] < tau) idx.push_back(i);
  }
  std::stable_sort(idx.begin(), idx.end(), [&](int x, int y) { return a.conf[x] < a.conf[y]; });
  if (static_cast<int>(idx.size()) > cap) idx.resize(std::max(0, cap));
  std::sort(idx.begin(), idx.end());
  return idx;
}

}  // namespace bc
