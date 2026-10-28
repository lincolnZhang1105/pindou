#include "grid/coords.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>

#include "image/text_blobs.h"

namespace bc {
namespace {

std::optional<int> parse_int(const std::string& s) {
  if (s.empty() || s.size() > 4) return std::nullopt;
  for (char c : s) {
    if (c < '0' || c > '9') return std::nullopt;
  }
  return std::stoi(s);
}

struct Hit {
  int index;   // 对齐到的 lattice 格子下标
  int number;  // 读出的坐标数字
  int cross;   // 数字中心所在的另一方向的 lattice 下标（列坐标 → 行号；行坐标 → 列号），-1 表示在网格外
};

// 一个方向（列或行）的拟合：index = number + k，k 取票数最多的值；数字只在 1..999 之间
CoordFit fit(const std::vector<Hit>& hits, std::vector<int>* inlier_cross = nullptr) {
  CoordFit f;
  std::map<int, int> votes;
  for (const auto& h : hits) votes[h.index - h.number]++;
  int best_k = 0, best_n = 0;
  for (const auto& [k, n] : votes) {
    if (n > best_n) {
      best_n = n;
      best_k = k;
    }
  }
  if (best_n < 3) return f;
  int lo = 1 << 30, hi = -(1 << 30);
  std::vector<int> nums;
  std::vector<int> cross;
  for (const auto& h : hits) {
    if (h.index - h.number != best_k) continue;
    lo = std::min(lo, h.number);
    hi = std::max(hi, h.number);
    nums.push_back(h.number);
    if (h.cross >= 0) cross.push_back(h.cross);
  }
  std::sort(nums.begin(), nums.end());
  nums.erase(std::unique(nums.begin(), nums.end()), nums.end());
  f.ok = true;
  f.k = best_k;
  f.support = static_cast<int>(nums.size());
  f.first = lo + best_k;
  f.last = hi + best_k;
  f.dense = nums.size() >= 0.6 * (hi - lo + 1) && lo <= 2;
  // 坐标从 1 开始；如果最小只读到 2，也认为第一格是 1
  if (f.dense && lo == 2) f.first = 1 + best_k;
  // 只有连续、逐格标注的坐标序列才拿来判断「坐标行」（零星读到的数字可能是格子里的色号）
  if (inlier_cross && f.dense) *inlier_cross = cross;
  return f;
}

}  // namespace

CoordResult read_coords(const nn::OcrModel& ocr, const cv::Mat& gray, int page_bg_gray,
                        const std::vector<double>& lx, const std::vector<double>& ly, int c0, int c1, int r0, int r1) {
  CoordResult res;
  if (lx.size() < 3 || ly.size() < 3) return res;
  const double px = (lx.back() - lx.front()) / (lx.size() - 1);
  const double py = (ly.back() - ly.front()) / (ly.size() - 1);
  auto col_of = [&](double cx) -> int {
    for (size_t i = 0; i + 1 < lx.size(); ++i) {
      const double c = 0.5 * (lx[i] + lx[i + 1]);
      if (std::fabs(cx - c) <= 0.35 * px) return static_cast<int>(i);
    }
    return -1;
  };
  auto row_of = [&](double cy) -> int {
    for (size_t j = 0; j + 1 < ly.size(); ++j) {
      const double c = 0.5 * (ly[j] + ly[j + 1]);
      if (std::fabs(cy - c) <= 0.35 * py) return static_cast<int>(j);
    }
    return -1;
  };
  const double gx0 = lx[c0], gx1 = lx[c1 + 1], gy0 = ly[r0], gy1 = ly[r1 + 1];
  // 四条带：网格外 2.5 格，网格内 1.3 格（把可能是坐标行的最外一行也包进来）
  struct Band {
    cv::Rect r;
    bool cols;
  };
  const int ix0 = static_cast<int>(gx0 - px), ix1 = static_cast<int>(gx1 + px);
  const int iy0 = static_cast<int>(gy0 - py), iy1 = static_cast<int>(gy1 + py);
  const Band bands[4] = {
      {cv::Rect(cv::Point(ix0, static_cast<int>(gy0 - 2.5 * py)), cv::Point(ix1, static_cast<int>(gy0 + 1.3 * py))), true},
      {cv::Rect(cv::Point(ix0, static_cast<int>(gy1 - 1.3 * py)), cv::Point(ix1, static_cast<int>(gy1 + 2.5 * py))), true},
      {cv::Rect(cv::Point(static_cast<int>(gx0 - 2.5 * px), iy0), cv::Point(static_cast<int>(gx0 + 1.3 * px), iy1)), false},
      {cv::Rect(cv::Point(static_cast<int>(gx1 - 1.3 * px), iy0), cv::Point(static_cast<int>(gx1 + 2.5 * px), iy1)), false},
  };
  std::vector<Hit> col_hits, row_hits;
  const int line_len = std::max(6, static_cast<int>(std::lround(0.9 * std::min(px, py))));
  const cv::Rect grid_box(cv::Point(static_cast<int>(gx0), static_cast<int>(gy0)), cv::Point(static_cast<int>(gx1), static_cast<int>(gy1)));
  for (const auto& b : bands) {
    // 底色：只看这条带在网格外的部分（分页图的页边常常整圈是被切开的格子，整页估出来的底色不对）
    const cv::Rect band = b.r & cv::Rect(0, 0, gray.cols, gray.rows);
    if (band.area() <= 0) continue;
    std::vector<uint8_t> outside;
    for (int y = band.y; y < band.y + band.height; ++y) {
      const uint8_t* g = gray.ptr<uint8_t>(y);
      for (int x = band.x; x < band.x + band.width; ++x) {
        if (!grid_box.contains(cv::Point(x, y))) outside.push_back(g[x]);
      }
    }
    int bg = page_bg_gray;
    if (outside.size() >= 20) {
      std::nth_element(outside.begin(), outside.begin() + outside.size() / 2, outside.end());
      bg = outside[outside.size() / 2];
    }
    // 合并间距取 0.3 倍字高：同一个数的几位数字之间几乎没有空隙，相邻两个坐标之间总有一点空；
    // 坐标数字常是浅灰小字，亮度差阈值取 40
    const auto words = find_words(gray, band, bg, 40, {}, line_len, 0.3);
    for (const auto& w : words) {
      if (w.height > 0.9 * (b.cols ? py : py) || w.width > 3.5 * px || w.height < 4) continue;
      const double cx = w.x + w.width / 2.0, cy = w.y + w.height / 2.0;
      if (b.cols) {
        const int i = col_of(cx);
        if (i >= 0) res.cols_with_word.push_back(i);
      } else {
        const int j = row_of(cy);
        if (j >= 0) res.rows_with_word.push_back(j);
      }
      const auto t = nn::read_text(ocr, gray, w);
      if (res.debug.size() < 80) {
        res.debug.push_back((b.cols ? "c " : "r ") + std::to_string(w.x) + "," + std::to_string(w.y) + "," +
                            std::to_string(w.width) + "x" + std::to_string(w.height) + "=" + t.text);
      }
      const auto n = parse_int(t.text);
      if (!n || *n < 1 || *n > 999) continue;
      // 数字中心所在的 lattice 行 / 列（不要求对齐到中心）
      auto cell_of = [](const std::vector<double>& l, double v) {
        for (size_t i = 0; i + 1 < l.size(); ++i) {
          if (v >= l[i] && v < l[i + 1]) return static_cast<int>(i);
        }
        return -1;
      };
      if (b.cols) {
        const int i = col_of(cx);
        if (i >= 0) col_hits.push_back({i, *n, cell_of(ly, cy)});
      } else {
        const int j = row_of(cy);
        if (j >= 0) row_hits.push_back({j, *n, cell_of(lx, cx)});
      }
    }
  }
  res.cols = fit(col_hits, &res.label_rows);
  res.rows = fit(row_hits, &res.label_cols);
  res.debug.push_back("col_hits=" + std::to_string(col_hits.size()) + " row_hits=" + std::to_string(row_hits.size()));
  return res;
}

}  // namespace bc
