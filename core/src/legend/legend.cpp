#include "legend/legend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "image/image_ops.h"
#include "image/text_blobs.h"

namespace bc {
namespace {

struct Region {
  cv::Rect box;
  int area = 0;
  std::array<long long, 3> sum{0, 0, 0};
  std::vector<uint8_t> r, g, b;  // 用于中位数
};

int cheb(const uint8_t* a, const color::RGB8& b) {
  return std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]), std::abs(a[2] - b[2])});
}

uint8_t med(std::vector<uint8_t>& v) {
  auto m = v.begin() + v.size() / 2;
  std::nth_element(v.begin(), m, v.end());
  return *m;
}

int gray_of(const color::RGB8& c) { return (299 * c[0] + 587 * c[1] + 114 * c[2] + 500) / 1000; }

// 给每个色块配上文字框。主流图例有四种版式（bench/synth 也按这四种生成）：
//   表格（色号在色块里，颗数在右边的格子里）、色号在色块里（颗数在下方）、
//   色号在色块下方（颗数再往下）、色号写在色块右边（颗数再往右）。
// 先看大多数色块里有没有字，决定是「色号在里面」还是「色号在旁边」；再把图例区里的词分给最近的色块。
// 这里只按几何位置分角色，OCR 读出内容后由 engine 再校验（near_words 留作备选）。
void locate_text(const cv::Mat& rgb, const color::RGB8& page_bg, LegendResult& out) {
  if (out.items.empty()) return;
  cv::Mat gray = rgb_to_gray(rgb);
  const int bg = gray_of(page_bg);
  cv::Rect band = out.items.front().swatch;
  int h = 0;
  for (const auto& it : out.items) {
    band |= it.swatch;
    h = std::max(h, it.swatch.height);
  }
  const bool table = out.layout == "table";
  band = cv::Rect(band.x - h, band.y - h / 2, band.width + 8 * h, band.height + 3 * h);
  // 色块外接框外侧还有一圈抗锯齿的边缘像素，扩 2 像素一起排除，免得被当成文字
  std::vector<cv::Rect> swatches;
  for (const auto& it : out.items) swatches.push_back(cv::Rect(it.swatch.x - 2, it.swatch.y - 2, it.swatch.width + 4, it.swatch.height + 4));
  const auto words = find_words(gray, band, bg, 60, swatches, std::max(8, h * 4 / 5));
  out.debug_words = words;

  // 1. 色块里的字
  int inside = 0;
  for (auto& it : out.items) {
    const cv::Rect inner(it.swatch.x + 2, it.swatch.y + 2, std::max(1, it.swatch.width - 4), std::max(1, it.swatch.height - 4));
    const auto in = find_words(gray, inner, gray_of(it.rgb), 50, {});
    if (!in.empty()) {
      cv::Rect u = in.front();
      for (const auto& w : in) u |= w;
      it.code_box = u;
      ++inside;
    }
  }
  const bool code_inside = inside * 5 >= static_cast<int>(out.items.size()) * 3;

  // 2. 把词分给最近的色块：同一行右侧，或者正下方
  const bool allow_right = !code_inside || table;
  const bool allow_below = !code_inside || !table;
  std::vector<int> owner(words.size(), -1);
  std::vector<double> owner_d(words.size(), 1e18);
  std::vector<int> owner_dir(words.size(), 0);  // 1 右侧，2 下方
  for (size_t k = 0; k < words.size(); ++k) {
    const cv::Rect& w = words[k];
    const double wcx = w.x + w.width / 2.0, wcy = w.y + w.height / 2.0;
    for (size_t i = 0; i < out.items.size(); ++i) {
      const cv::Rect& s = out.items[i].swatch;
      const double scx = s.x + s.width / 2.0, scy = s.y + s.height / 2.0;
      const double sh = s.height;
      if (allow_right && std::fabs(wcy - scy) <= 0.6 * sh && w.x >= s.x + s.width - 2 && w.x - (s.x + s.width) <= 6 * sh) {
        const double d = w.x - (s.x + s.width);
        if (d < owner_d[k]) {
          owner_d[k] = d;
          owner[k] = static_cast<int>(i);
          owner_dir[k] = 1;
        }
      }
      if (allow_below && w.y >= s.y + s.height - 2 && w.y - (s.y + s.height) <= 2.2 * sh &&
          std::fabs(wcx - scx) <= 0.75 * s.width) {
        const double d = (w.y - (s.y + s.height)) + 0.5 * std::fabs(wcx - scx);
        if (d < owner_d[k]) {
          owner_d[k] = d;
          owner[k] = static_cast<int>(i);
          owner_dir[k] = 2;
        }
      }
    }
  }
  // 每个色块的候选词：右侧的按 x，下方的按 y
  for (size_t i = 0; i < out.items.size(); ++i) {
    std::vector<size_t> mine;
    for (size_t k = 0; k < words.size(); ++k) {
      if (owner[k] == static_cast<int>(i)) mine.push_back(k);
    }
    std::sort(mine.begin(), mine.end(), [&](size_t a, size_t b) {
      if (owner_dir[a] != owner_dir[b]) return owner_dir[a] < owner_dir[b];
      return owner_dir[a] == 1 ? words[a].x < words[b].x : words[a].y < words[b].y;
    });
    auto& it = out.items[i];
    for (size_t k : mine) it.near_words.push_back(words[k]);
    if (code_inside) {
      if (!it.near_words.empty()) it.count_box = it.near_words.front();
    } else {
      if (!it.near_words.empty()) it.code_box = it.near_words[0];
      if (it.near_words.size() >= 2) it.count_box = it.near_words[1];
    }
  }
  if (table || !code_inside) return;

  // 3. 色号在色块里的横排图例：和色块同一行、没分给任何色块的词，是看不见色块的条目（白底上的白色色块），
  //    颗数在它下方
  std::vector<int> row_cys;
  for (const auto& it : out.items) {
    const int cy = it.swatch.y + it.swatch.height / 2;
    bool seen = false;
    for (int r : row_cys) seen |= std::abs(r - cy) <= h / 2;
    if (!seen) row_cys.push_back(cy);
  }
  std::vector<uint8_t> used(words.size(), 0);
  for (size_t k = 0; k < words.size(); ++k) used[k] = owner[k] >= 0;
  std::vector<LegendItem> extra;
  for (size_t k = 0; k < words.size(); ++k) {
    if (used[k]) continue;
    const cv::Rect& w = words[k];
    const int wcy = w.y + w.height / 2;
    int row_cy = -1;
    for (int r : row_cys) {
      if (std::abs(wcy - r) <= h / 3) row_cy = r;
    }
    if (row_cy < 0 || w.height < h / 6) continue;
    LegendItem it;
    it.text_only = true;
    it.swatch = cv::Rect(w.x + w.width / 2 - h / 2, row_cy - h / 2, h, h);
    it.code_box = w;
    it.rgb = page_bg;
    it.lab = color::srgb8_to_lab(page_bg);
    for (size_t q = 0; q < words.size(); ++q) {
      if (used[q] || q == k) continue;
      const cv::Rect& v = words[q];
      if (std::abs(v.x + v.width / 2 - (w.x + w.width / 2)) <= h / 2 && v.y > w.y + w.height &&
          v.y <= it.swatch.y + 2.5 * h) {
        it.count_box = v;
        it.near_words.push_back(v);
        used[q] = 1;
        break;
      }
    }
    used[k] = 1;
    extra.push_back(it);
  }
  out.items.insert(out.items.end(), extra.begin(), extra.end());
  // 阅读顺序：先按行，再从左到右
  std::sort(out.items.begin(), out.items.end(), [h](const LegendItem& a, const LegendItem& b) {
    if (std::abs(a.swatch.y - b.swatch.y) > h / 2) return a.swatch.y < b.swatch.y;
    return a.swatch.x < b.swatch.x;
  });
}

}  // namespace

LegendResult detect_legend(const cv::Mat& rgb, const cv::Rect& exclude, const color::RGB8& page_bg) {
  LegendResult out;
  out.layout = "none";
  const int W = rgb.cols, H = rgb.rows;
  cv::Mat seen(rgb.size(), CV_8U, cv::Scalar(0));
  seen(exclude & cv::Rect(0, 0, W, H)).setTo(1);

  std::vector<Region> regions, pieces;
  std::vector<cv::Point> stack;
  const int tol = 14;
  const int img_area = W * H;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (seen.at<uint8_t>(y, x)) continue;
      const uint8_t* p0 = rgb.ptr<uint8_t>(y) + 3 * x;
      if (cheb(p0, page_bg) <= tol) {  // 页面底色不算色块
        seen.at<uint8_t>(y, x) = 1;
        continue;
      }
      const color::RGB8 seed{p0[0], p0[1], p0[2]};
      Region reg;
      int x0 = x, x1 = x, y0 = y, y1 = y;
      stack.clear();
      stack.push_back({x, y});
      seen.at<uint8_t>(y, x) = 1;
      while (!stack.empty()) {
        const cv::Point q = stack.back();
        stack.pop_back();
        const uint8_t* pq = rgb.ptr<uint8_t>(q.y) + 3 * q.x;
        ++reg.area;
        reg.r.push_back(pq[0]);
        reg.g.push_back(pq[1]);
        reg.b.push_back(pq[2]);
        x0 = std::min(x0, q.x);
        x1 = std::max(x1, q.x);
        y0 = std::min(y0, q.y);
        y1 = std::max(y1, q.y);
        const cv::Point nb[4] = {{q.x - 1, q.y}, {q.x + 1, q.y}, {q.x, q.y - 1}, {q.x, q.y + 1}};
        for (const auto& n : nb) {
          if (n.x < 0 || n.y < 0 || n.x >= W || n.y >= H || seen.at<uint8_t>(n.y, n.x)) continue;
          if (cheb(rgb.ptr<uint8_t>(n.y) + 3 * n.x, seed) <= tol) {
            seen.at<uint8_t>(n.y, n.x) = 1;
            stack.push_back(n);
          }
        }
      }
      reg.box = cv::Rect(cv::Point(x0, y0), cv::Point(x1 + 1, y1 + 1));
      if (reg.area >= 12) pieces.push_back(std::move(reg));
    }
  }

  // 平铺水印会斜着穿过色块，把一个色块切成几片：颜色相同、外接框相邻的碎片先合并
  std::vector<color::Lab> piece_lab(pieces.size());
  for (size_t i = 0; i < pieces.size(); ++i) {
    auto r = pieces[i].r, g = pieces[i].g, b = pieces[i].b;
    piece_lab[i] = color::srgb8_to_lab({med(r), med(g), med(b)});
  }
  std::vector<int> parent(pieces.size());
  for (size_t i = 0; i < parent.size(); ++i) parent[i] = static_cast<int>(i);
  std::function<int(int)> find = [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
  for (size_t i = 0; i < pieces.size(); ++i) {
    const cv::Rect a(pieces[i].box.x - 3, pieces[i].box.y - 3, pieces[i].box.width + 6, pieces[i].box.height + 6);
    for (size_t j = i + 1; j < pieces.size(); ++j) {
      if ((a & pieces[j].box).area() > 0 && color::delta_e_2000(piece_lab[i], piece_lab[j]) < 5.0) {
        parent[find(static_cast<int>(j))] = find(static_cast<int>(i));
      }
    }
  }
  std::vector<Region> merged(pieces.size());
  std::vector<uint8_t> used(pieces.size(), 0);
  for (size_t i = 0; i < pieces.size(); ++i) {
    const int root = find(static_cast<int>(i));
    Region& m = merged[root];
    m.box = used[root] ? (m.box | pieces[i].box) : pieces[i].box;
    used[root] = 1;
    m.area += pieces[i].area;
    m.r.insert(m.r.end(), pieces[i].r.begin(), pieces[i].r.end());
    m.g.insert(m.g.end(), pieces[i].g.begin(), pieces[i].g.end());
    m.b.insert(m.b.end(), pieces[i].b.begin(), pieces[i].b.end());
  }
  for (size_t i = 0; i < merged.size(); ++i) {
    if (!used[i]) continue;
    Region& reg = merged[i];
    const double fill = double(reg.area) / reg.box.area();
    const double aspect = double(reg.box.width) / reg.box.height;
    if (reg.area >= 80 && reg.box.width >= 10 && reg.box.height >= 10 && fill >= 0.55 && aspect >= 1.0 / 8 &&
        aspect <= 8 && reg.area < img_area / 10) {
      regions.push_back(std::move(reg));
    }
  }
  for (const auto& r : regions) out.debug_regions.push_back(r.box);
  if (regions.size() < 1) return out;

  // 色块成组出现：高度相近、成行或成列对齐。取成员最多的一组。
  std::vector<int> best_group;
  for (size_t i = 0; i < regions.size(); ++i) {
    std::vector<int> grp;
    const auto& a = regions[i].box;
    for (size_t j = 0; j < regions.size(); ++j) {
      const auto& b = regions[j].box;
      const bool similar = std::abs(a.height - b.height) <= 0.25 * a.height &&
                           std::abs(a.width - b.width) <= 0.35 * std::max(a.width, b.width);
      const bool row = std::abs((a.y + a.height / 2) - (b.y + b.height / 2)) <= a.height / 2;
      const bool col = std::abs(a.x - b.x) <= a.width / 4 || std::abs((a.x + a.width) - (b.x + b.width)) <= a.width / 4;
      if (similar && (row || col)) grp.push_back(static_cast<int>(j));
    }
    if (grp.size() > best_group.size()) best_group = grp;
  }
  // 把同组成员的同类（尺寸相近）也并进来：表格式图例可能分成两列
  std::vector<int> members;
  if (!best_group.empty()) {
    // 尺寸参照取组内中位数，不取第一个成员（第一个色块可能被网格排除区切掉一截）
    std::vector<int> hs, ws;
    for (int i : best_group) {
      hs.push_back(regions[i].box.height);
      ws.push_back(regions[i].box.width);
    }
    std::nth_element(hs.begin(), hs.begin() + hs.size() / 2, hs.end());
    std::nth_element(ws.begin(), ws.begin() + ws.size() / 2, ws.end());
    const int rh = hs[hs.size() / 2], rw = ws[ws.size() / 2];
    for (size_t j = 0; j < regions.size(); ++j) {
      const auto& b = regions[j].box;
      if (std::abs(rh - b.height) <= 0.25 * rh && std::abs(rw - b.width) <= 0.35 * std::max(rw, b.width)) {
        members.push_back(static_cast<int>(j));
      }
    }
  }
  if (members.size() < 2) return out;

  out.layout = (regions[members.front()].box.width > 2.5 * regions[members.front()].box.height) ? "table" : "row";

  // 阅读顺序：先上后下，再从左到右（表格式两列时先左列再右列）
  std::sort(members.begin(), members.end(), [&](int a, int b) {
    const auto& A = regions[a].box;
    const auto& B = regions[b].box;
    if (out.layout == "table" && std::abs(A.x - B.x) > A.width / 2) return A.x < B.x;
    if (std::abs(A.y - B.y) > A.height / 2) return A.y < B.y;
    return A.x < B.x;
  });
  {
    std::vector<std::pair<int, int>> spans;
    for (int idx : members) spans.push_back({regions[idx].box.y, regions[idx].box.y + regions[idx].box.height});
    std::sort(spans.begin(), spans.end());
    const int h = regions[members.front()].box.height;
    out.bands = 1;
    int end = spans.front().second;
    for (const auto& sp : spans) {
      if (sp.first > end + 3 * h) ++out.bands;
      end = std::max(end, sp.second);
    }
  }
  for (int idx : members) {
    auto& r = regions[idx];
    LegendItem it;
    it.swatch = r.box;
    it.area = r.area;
    it.rgb = {med(r.r), med(r.g), med(r.b)};
    it.lab = color::srgb8_to_lab(it.rgb);
    out.items.push_back(it);
  }
  locate_text(rgb, page_bg, out);
  return out;
}

}  // namespace bc
