#include "grid/grid.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

#include "classify/cells.h"
#include "image/image_ops.h"

namespace bc {
namespace {

double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  auto mid = v.begin() + v.size() / 2;
  std::nth_element(v.begin(), mid, v.end());
  return *mid;
}


std::vector<double> to_vec(const cv::Mat& m) {
  std::vector<double> v(m.total());
  for (size_t i = 0; i < v.size(); ++i) v[i] = m.at<double>(static_cast<int>(i));
  return v;
}

// 线响应投影：截幅后沿线方向做长度为 L 的腐蚀，去掉文字笔画这类短线段，再求和
std::vector<double> line_profile(const cv::Mat& ridge, bool vertical_lines, int L) {
  cv::Mat clipped, closed, er;
  // 截到 48：深色粗线（对比度可达 200）不能在投影里盖过很淡的细线（对比度 20–40）
  cv::min(ridge, 48, clipped);
  // 交叉点上线响应为 0（上下、左右都是线），先沿线方向做长度 3 的闭运算把 1–2 像素的缺口补上（再长会把相邻小格里的文字笔画连起来），
  // 这样后面才能用比格距还长的腐蚀去掉文字笔画，而不会把线截断
  const auto along = [&](int len) {
    return cv::getStructuringElement(cv::MORPH_RECT, vertical_lines ? cv::Size(1, len) : cv::Size(len, 1));
  };
  cv::morphologyEx(clipped, closed, cv::MORPH_CLOSE, along(3));
  cv::erode(closed, er, along(L));
  cv::Mat prof;
  cv::reduce(er, prof, vertical_lines ? 0 : 1, cv::REDUCE_SUM, CV_64F);
  return to_vec(prof);
}

std::vector<double> autocorr(const std::vector<double>& p, int max_lag, int hp_window) {
  const int n = static_cast<int>(p.size());
  if (n < 8 || max_lag >= n / 2) max_lag = n / 2 - 1;
  if (max_lag < 2) return {};
  // 高通：减去滑动平均，去掉大面积亮度变化
  std::vector<double> prefix(n + 1, 0.0);
  for (int i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + p[i];
  std::vector<double> q(n);
  const int hw = std::max(1, hp_window / 2);
  for (int i = 0; i < n; ++i) {
    const int a = std::max(0, i - hw), b = std::min(n, i + hw + 1);
    q[i] = p[i] - (prefix[b] - prefix[a]) / (b - a);
  }
  double var = 0;
  for (double v : q) var += v * v;
  if (var <= 0) return {};
  std::vector<double> A(max_lag + 1, 0.0);
  for (int l = 0; l <= max_lag; ++l) {
    double s = 0;
    for (int i = 0; i + l < n; ++i) s += q[i] * q[i + l];
    A[l] = s / var * double(n) / double(n - l);
  }
  return A;
}

bool is_local_max(const std::vector<double>& A, int l) {
  return l > 0 && l + 1 < static_cast<int>(A.size()) && A[l] >= A[l - 1] && A[l] >= A[l + 1];
}

double parabolic(const std::vector<double>& s, int i) {
  if (i <= 0 || i + 1 >= static_cast<int>(s.size())) return i;
  const double a = s[i - 1], b = s[i], c = s[i + 1];
  const double den = a - 2 * b + c;
  if (std::fabs(den) < 1e-12) return i;
  return i + std::clamp(0.5 * (a - c) / den, -0.5, 0.5);
}

struct Segments {
  int nx = 0, ny = 0;
  std::vector<double> v;  // nx × (ny-1)：第 i 条竖线在第 j 行格子旁的线段
  std::vector<double> h;  // (nx-1) × ny：第 j 条横线在第 i 列格子旁的线段
  double& V(int i, int j) { return v[i * (ny - 1) + j]; }
  double& Hs(int i, int j) { return h[j * (nx - 1) + i]; }
};

// 线段强度：沿线段取 ±halfw 带内的最大线响应，再取中位数（被文字或水印打断一小段不影响）
double segment_strength(const cv::Mat& ridge, bool vertical, double line_pos, double a0, double a1, int halfw) {
  const double len = a1 - a0;
  const int s0 = static_cast<int>(std::ceil(a0 + 0.2 * len)), s1 = static_cast<int>(std::floor(a1 - 0.2 * len));
  const int lp = static_cast<int>(std::lround(line_pos));
  std::vector<double> vals;
  for (int s = s0; s <= s1; ++s) {
    int best = 0;
    for (int d = -halfw; d <= halfw; ++d) {
      const int x = vertical ? lp + d : s, y = vertical ? s : lp + d;
      if (x < 0 || y < 0 || x >= ridge.cols || y >= ridge.rows) continue;
      best = std::max<int>(best, ridge.at<uint8_t>(y, x));
    }
    vals.push_back(best);
  }
  return median_of(vals);
}

// 判断网格最外一圈是不是坐标栏：底色一致、大多有字、而且相邻格的字形各不相同（数字 1、2、3……）。
// 图纸本身的一圈不可能满足：底色相同就是同一个色号，字形也会相同。
// band_like：底色一致、几乎每格有字这两条满足（不管字形对比的结果）
bool is_header_line(const cv::Mat& rgb, const cv::Mat& gray, const std::vector<CellRect>& cells,
                    std::string* why = nullptr, bool* band_like = nullptr) {
  if (band_like) *band_like = false;
  if (cells.size() < 3) return false;
  std::vector<CellSample> ss;
  ss.reserve(cells.size());
  for (const auto& c : cells) ss.push_back(sample_cell(rgb, gray, c));
  std::vector<double> L, A, B;
  for (const auto& s : ss) {
    L.push_back(s.bg_lab.L);
    A.push_back(s.bg_lab.a);
    B.push_back(s.bg_lab.b);
  }
  const color::Lab med{median_of(L), median_of(A), median_of(B)};
  int uniform = 0, texted = 0;
  for (const auto& s : ss) {
    if (color::delta_e_2000(s.bg_lab, med) < 6.0) ++uniform;
    if (s.has_text) ++texted;
  }
  const double n = static_cast<double>(ss.size());
  if (why) *why = "n=" + std::to_string(ss.size()) + " uniform=" + std::to_string(uniform) + " texted=" + std::to_string(texted);
  if (uniform < 0.85 * n || texted < 0.6 * n) return false;
  if (band_like) *band_like = true;
  // 同一个色号连成一圈时，相邻格的字形几乎一模一样；坐标数字相邻两个总是不同（10 和 11 虽然共用一个「1」，
  // 整体也不相同）。所以看「几乎相同」的相邻对占多少，而不是看平均相似度。
  int pairs = 0, same = 0;
  std::vector<double> sims;
  for (size_t i = 0; i + 1 < cells.size(); ++i) {
    if (!ss[i].has_text || !ss[i + 1].has_text) continue;
    ++pairs;
    const double sim = glyph_similarity(text_glyph(gray, cells[i], ss[i]), text_glyph(gray, cells[i + 1], ss[i + 1]));
    sims.push_back(sim);
    same += sim > 0.9;
  }
  if (why) {
    std::sort(sims.begin(), sims.end());
    auto q = [&](double f) { return sims.empty() ? 0.0 : sims[static_cast<size_t>(f * (sims.size() - 1))]; };
    *why += " pairs=" + std::to_string(pairs) + " same=" + std::to_string(same) + " sim_q25/50/75=" +
            std::to_string(q(0.25)).substr(0, 4) + "/" + std::to_string(q(0.5)).substr(0, 4) + "/" + std::to_string(q(0.75)).substr(0, 4);
  }
  return pairs > 0 && same < 0.3 * pairs;
}

}  // namespace

cv::Mat ridge_map(const cv::Mat& gray, bool vertical_lines) {
  CV_Assert(gray.type() == CV_8UC1);
  const int W = gray.cols, H = gray.rows;
  cv::Mat r = cv::Mat::zeros(gray.size(), CV_8U);
  if (vertical_lines) {
    for (int y = 0; y < H; ++y) {
      const uint8_t* g = gray.ptr<uint8_t>(y);
      uint8_t* o = r.ptr<uint8_t>(y);
      for (int x = 1; x < W - 1; ++x) {
        int best = 0;
        for (int k = 1; k <= 3 && x - k >= 0 && x + k < W; ++k) {
          const int a = g[x - k], b = g[x], c = g[x + k];
          best = std::max({best, std::min(a, c) - b, b - std::max(a, c)});
        }
        o[x] = static_cast<uint8_t>(best);
      }
    }
  } else {
    for (int y = 1; y < H - 1; ++y) {
      uint8_t* o = r.ptr<uint8_t>(y);
      const uint8_t* g = gray.ptr<uint8_t>(y);
      for (int x = 0; x < W; ++x) {
        int best = 0;
        for (int k = 1; k <= 3 && y - k >= 0 && y + k < H; ++k) {
          const int a = gray.ptr<uint8_t>(y - k)[x], b = g[x], c = gray.ptr<uint8_t>(y + k)[x];
          best = std::max({best, std::min(a, c) - b, b - std::max(a, c)});
        }
        o[x] = static_cast<uint8_t>(best);
      }
    }
  }
  return r;
}

cv::Mat ridge_map_rgb(const cv::Mat& rgb, bool vertical_lines) {
  CV_Assert(rgb.type() == CV_8UC3);
  cv::Mat ch[3];
  cv::split(rgb, ch);
  cv::Mat r = ridge_map(ch[0], vertical_lines);
  cv::max(r, ridge_map(ch[1], vertical_lines), r);
  cv::max(r, ridge_map(ch[2], vertical_lines), r);
  return r;
}

double estimate_period(const std::vector<double>& pa, const std::vector<double>& pb, int min_p, int max_p,
                       double* score) {
  const auto Aa = autocorr(pa, max_p + 2, 2 * max_p + 1);
  const auto Ab = autocorr(pb, max_p + 2, 2 * max_p + 1);
  std::vector<double> A;
  if (!Aa.empty() && !Ab.empty()) {
    A.resize(std::min(Aa.size(), Ab.size()));
    for (size_t i = 0; i < A.size(); ++i) A[i] = 0.5 * (Aa[i] + Ab[i]);
  } else {
    A = !Aa.empty() ? Aa : Ab;
  }
  if (score) *score = 0;
  const int top = static_cast<int>(A.size()) - 2;
  if (top <= min_p) return 0.0;
  int best = -1;
  for (int l = min_p; l <= std::min(max_p, top); ++l) {
    if (is_local_max(A, l) && (best < 0 || A[l] > A[best])) best = l;
  }
  if (best < 0 || A[best] < 0.15) return 0.0;
  // 粗线（每 5 / 10 格）会让 5p、10p 处的峰更高：检查约数，取满足条件的最小周期
  int chosen = best;
  for (int m = 10; m >= 2; --m) {
    const double c = double(best) / m;
    if (c < min_p) continue;
    int lc = -1;
    for (int l = static_cast<int>(std::floor(c)) - 1; l <= static_cast<int>(std::ceil(c)) + 1; ++l) {
      if (l >= min_p && is_local_max(A, l) && (lc < 0 || A[l] > A[lc])) lc = l;
    }
    if (lc > 0 && A[lc] >= 0.25 && A[lc] >= 0.3 * A[best]) {
      chosen = lc;
      break;
    }
  }
  if (score) *score = A[chosen];
  return parabolic(A, chosen);
}

std::vector<double> track_lines(const std::vector<double>& prof, double p) {
  const int n = static_cast<int>(prof.size());
  if (n < 5 || p < 2) return {};
  std::vector<double> s(n, 0.0);
  for (int i = 1; i + 1 < n; ++i) s[i] = 0.25 * prof[i - 1] + 0.5 * prof[i] + 0.25 * prof[i + 1];
  const int start = static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin());
  if (s[start] <= 0) return {};

  struct Line {
    double pos;
    bool real;
  };
  // 参考强度：起点两侧各 6 个周期内的峰值。起点往往是最强的粗线（每 5 / 10 格一条），
  // 只拿它当参考会把旁边的细线全判成「没有线」。
  std::vector<double> seed{s[start]};
  for (int k = -6; k <= 6; ++k) {
    if (k == 0) continue;
    const double c = start + k * p;
    const int lo = static_cast<int>(std::floor(c - 0.22 * p)), hi = static_cast<int>(std::ceil(c + 0.22 * p));
    if (lo < 0 || hi >= n) continue;
    seed.push_back(*std::max_element(s.begin() + lo, s.begin() + hi + 1));
  }
  auto walk = [&](int dir) {
    std::vector<Line> out;
    std::vector<double> recent = seed;
    double pos = parabolic(s, start), cur_p = p;
    int misses = 0;
    while (true) {
      const double pred = pos + dir * cur_p * (misses + 1);
      // 预测位置出了图片才停。分页图在接缝一侧、贴边裁的截图，最后一条线离图片边缘只有一两个像素，
      // 搜索窗口会越界：把窗口裁到图片里继续找，被裁的那一端改成要求峰值是局部极大
      if (pred < 0 || pred > n - 1) break;
      const int lo0 = static_cast<int>(std::floor(pred - 0.22 * p)), hi0 = static_cast<int>(std::ceil(pred + 0.22 * p));
      const int lo = std::max(lo0, 1), hi = std::min(hi0, n - 2);
      if (hi - lo < 2) break;
      int b = lo;
      for (int i = lo; i <= hi; ++i) {
        if (s[i] > s[b]) b = i;
      }
      const bool peak_in_window = (b > lo || (lo > lo0 && s[b] >= s[b - 1])) && (b < hi || (hi < hi0 && s[b] >= s[b + 1]));
      std::vector<double> last(recent.end() - std::min<size_t>(recent.size(), 8), recent.end());
      const double ref = median_of(last);
      // 强线：≥ 参考强度的 30%；弱线（大片空格里很淡的线）：≥ 10% 且峰值正好落在预测位置附近
      const bool strong = s[b] >= 0.3 * ref;
      const bool weak = s[b] >= 0.1 * ref && std::fabs(b - pred) <= 0.15 * p;
      if ((strong || weak) && s[b] > 0 && peak_in_window) {
        const double np = parabolic(s, b);
        const double step = std::fabs(np - pos) / (misses + 1);
        cur_p = std::clamp(0.8 * cur_p + 0.2 * step, 0.97 * p, 1.03 * p);
        for (int k = 1; k <= misses; ++k) out.push_back({pos + (np - pos) * k / (misses + 1), false});
        out.push_back({np, true});
        pos = np;
        if (strong) recent.push_back(s[b]);  // 弱线不拉低参考强度
        misses = 0;
      } else {
        // 缺线不停：按格距一直外推到图片边缘（大片同色区域里线可能读不出来），
        // 末尾没有证据的虚线最后去掉；网格的实际范围交给后面的二维判断
        ++misses;
      }
    }
    while (!out.empty() && !out.back().real) out.pop_back();
    return out;
  };
  std::vector<double> lines;
  for (const auto& l : walk(-1)) lines.push_back(l.pos);
  lines.push_back(parabolic(s, start));
  for (const auto& l : walk(+1)) lines.push_back(l.pos);
  std::sort(lines.begin(), lines.end());
  return lines;
}

namespace {

struct Comb {
  double completeness = 0;  // 活动范围内有线的梳齿比例
  double parity = 0;        // 奇数齿和偶数齿平均强度之比（较小 / 较大）
};

// 以 q 为周期在投影上放一把「梳子」：相位取让齿上总强度最大的那个，再看齿上是不是都有线
Comb comb_check(const std::vector<double>& prof, double q) {
  Comb c;
  const int n = static_cast<int>(prof.size());
  if (q < 2 || n < 3 * q) return c;
  auto at = [&](double x) {
    const int i = static_cast<int>(std::lround(x));
    double m = 0;
    for (int d = -1; d <= 1; ++d) {
      if (i + d >= 0 && i + d < n) m = std::max(m, prof[i + d]);
    }
    return m;
  };
  double best_phase = 0, best_sum = -1;
  for (double ph = 0; ph < q; ph += 0.25) {
    double sum = 0;
    for (double x = ph; x < n; x += q) sum += at(x);
    if (sum > best_sum) {
      best_sum = sum;
      best_phase = ph;
    }
  }
  std::vector<double> v;
  for (double x = best_phase; x < n; x += q) v.push_back(at(x));
  std::vector<double> sorted = v;
  std::sort(sorted.begin(), sorted.end());
  const double ref = sorted[sorted.size() * 3 / 4];
  if (ref <= 0) return c;
  // 活动范围：第一个到最后一个强齿之间
  int a = 0, b = static_cast<int>(v.size()) - 1;
  while (a < b && v[a] < 0.25 * ref) ++a;
  while (b > a && v[b] < 0.25 * ref) --b;
  int on = 0, cnt = 0;
  double even = 0, odd = 0;
  int ne = 0, no = 0;
  for (int k = a; k <= b; ++k) {
    ++cnt;
    on += v[k] >= 0.25 * ref;
    if ((k - a) % 2 == 0) {
      even += v[k];
      ++ne;
    } else {
      odd += v[k];
      ++no;
    }
  }
  if (cnt < 4) return c;
  c.completeness = double(on) / cnt;
  even /= std::max(1, ne);
  odd /= std::max(1, no);
  c.parity = std::max(even, odd) > 0 ? std::min(even, odd) / std::max(even, odd) : 0;
  return c;
}

}  // namespace

GridResult detect_grid(const cv::Mat& rgb, const cv::Mat& gray, const Progress& progress) {
  GridResult g;
  const cv::Mat rv = ridge_map_rgb(rgb, true);
  const cv::Mat rh = ridge_map_rgb(rgb, false);
  progress.report(0.3f);

  // 投影前沿线方向腐蚀，去掉文字笔画；但腐蚀长度必须小于格距，否则每个交叉点（线响应为 0）都会把线截断，
  // 只剩下更粗、更少交叉的粗线（每 5 / 10 格一条）。所以长度 9 和 3 各估一次：
  // 长度 9 的结果是长度 3 结果的整数倍（≥ 3 倍）时，说明细线被腐蚀掉了，取长度 3 的；
  // 只差 2 倍时保留长度 9 的——格内居中的文字会在半格处造出假周期。
  const int min_p = 5;  // 208 格铺满 1040 宽也有 5 像素；更小的「格距」多半是模糊或压缩造成的假峰
  const int max_p = std::max(min_p + 2, std::min(gray.cols, gray.rows) / 6);
  // 腐蚀长度 21：比文字笔画长（大格子里的字也不过 20 像素高），交叉点的缺口已经补上，所以可以比格距长
  const int L = std::min(21, std::max(5, std::min(gray.cols, gray.rows) / 20));
  double score = 0;
  std::vector<double> prof_x = line_profile(rv, true, L), prof_y = line_profile(rh, false, L);
  double p = estimate_period(prof_x, prof_y, min_p, max_p, &score);
  g.period9 = p;
  // 粗线每 N 格一条时，自相关最强的可能是 N 倍格距：从小到大检验 p/m 的「梳子」，
  // 齿上几乎都有线（≥ 0.8）且奇偶齿强度相当（≥ 0.45）的最小周期就是格距
  if (p > 0) {
    std::vector<double> cands;
    for (int m = 10; m >= 1; --m) {
      if (p / m >= min_p) cands.push_back(p / m);
    }
    for (double q : cands) {
      const auto cx = comb_check(prof_x, q);
      const auto cy = comb_check(prof_y, q);
      g.comb_debug.push_back(std::to_string(q).substr(0, 6) + ": x " + std::to_string(cx.completeness).substr(0, 4) + "/" +
                             std::to_string(cx.parity).substr(0, 4) + " y " + std::to_string(cy.completeness).substr(0, 4) +
                             "/" + std::to_string(cy.parity).substr(0, 4));
      // 两个方向平均完整度 ≥ 0.8、单个方向不低于 0.6：大片色块里某个方向的线可能整段读不出来
      const double comp = 0.5 * (cx.completeness + cy.completeness);
      if (comp >= 0.8 && std::min(cx.completeness, cy.completeness) >= 0.6 && cx.parity >= 0.45 && cy.parity >= 0.45) {
        p = q;
        break;
      }
    }
  }
  g.period3 = p;
  if (p <= 0) {
    g.fail_reason = "没有找到周期性的网格线";
    return g;
  }
  g.lattice_x = track_lines(prof_x, p);
  g.lattice_y = track_lines(prof_y, p);
  // 网格外框紧贴有色格子时，外框线往往只是「页面底色 → 有色格」的台阶边缘，不是线响应（两侧要都比线深或都比线浅），
  // 追踪不到。在两端各往外预测一格：那里有强台阶边缘、而且这一条格子大多不是页面底色，就补上这条线。
  if (g.lattice_x.size() >= 3 && g.lattice_y.size() >= 3) {
    const color::RGB8 bgc = estimate_page_background(rgb);
    const color::Lab bgl = color::srgb8_to_lab(bgc);
    auto extend = [&](std::vector<double>& L, const std::vector<double>& other, bool vertical) {
      // 沿 other 的范围求台阶边缘投影（灰度的中心差分绝对值）
      const int a0 = std::max(0, static_cast<int>(other.front())), a1 = std::min(vertical ? gray.rows : gray.cols, static_cast<int>(other.back()));
      if (a1 - a0 < 4) return;
      const int n = vertical ? gray.cols : gray.rows;
      std::vector<double> edge(n, 0.0);
      for (int t = a0; t < a1; ++t) {
        for (int q = 1; q + 1 < n; ++q) {
          const int v0 = vertical ? gray.at<uint8_t>(t, q - 1) : gray.at<uint8_t>(q - 1, t);
          const int v1 = vertical ? gray.at<uint8_t>(t, q + 1) : gray.at<uint8_t>(q + 1, t);
          edge[q] += std::abs(v1 - v0);
        }
      }
      std::vector<double> at_lines;
      for (double l : L) {
        const int q = static_cast<int>(std::lround(l));
        double m = 0;
        for (int d = -1; d <= 1; ++d) {
          if (q + d > 0 && q + d < n) m = std::max(m, edge[q + d]);
        }
        at_lines.push_back(m);
      }
      std::nth_element(at_lines.begin(), at_lines.begin() + at_lines.size() / 2, at_lines.end());
      const double ref = at_lines[at_lines.size() / 2];
      const double step = (L.back() - L.front()) / (L.size() - 1);
      for (int dir : {-1, +1}) {
        const double end = dir < 0 ? L.front() : L.back();
        const double pred = end + dir * step;
        if (pred < 0 || pred > n - 1) continue;
        // 和 track_lines 一样，窗口越界时裁到图片里（台阶边缘投影在 [1, n-2] 上有定义）
        const int lo = std::max(1, static_cast<int>(std::floor(pred - 0.25 * step)));
        const int hi = std::min(n - 2, static_cast<int>(std::ceil(pred + 0.25 * step)));
        if (hi < lo) continue;
        int best = lo;
        for (int q = lo; q <= hi; ++q) {
          if (edge[q] > edge[best]) best = q;
        }
        if (edge[best] < 0.5 * ref) continue;
        // 网格线压在图片边缘上（分页图的接缝、贴边裁的截图）：线只剩靠里的一半，线响应读不出来，只有台阶边缘。
        // 这时这一条格子是完整的一格，空白也算；网格外留了白边时预测位置上没有台阶边缘，走不到这里
        const bool at_border = dir < 0 ? pred - 0.35 * step <= 0 : pred + 0.35 * step >= n - 1;
        if (at_border) {
          if (dir < 0) {
            L.insert(L.begin(), best);
          } else {
            L.push_back(best);
          }
          continue;
        }
        // 新加的这一条格子：大多数不是页面底色
        int colored = 0, total = 0;
        for (size_t k = 0; k + 1 < other.size(); ++k) {
          const double c0 = std::min(end, double(best)), c1 = std::max(end, double(best));
          const CellRect cr = vertical ? CellRect{c0, c1, other[k], other[k + 1]} : CellRect{other[k], other[k + 1], c0, c1};
          const CellSample cs = sample_cell(rgb, gray, cr);
          ++total;
          colored += color::delta_e_2000(cs.bg_lab, bgl) > 10.0;
        }
        if (total == 0 || colored * 2 < total) continue;
        if (dir < 0) {
          L.insert(L.begin(), best);
        } else {
          L.push_back(best);
        }
      }
    };
    extend(g.lattice_x, g.lattice_y, true);
    extend(g.lattice_y, g.lattice_x, false);
  }
  progress.report(0.5f);
  if (g.lattice_x.size() < 3 || g.lattice_y.size() < 3) {
    g.fail_reason = "追踪到的网格线太少";
    return g;
  }

  // 线段强度
  Segments seg;
  seg.nx = static_cast<int>(g.lattice_x.size());
  seg.ny = static_cast<int>(g.lattice_y.size());
  seg.v.assign(seg.nx * (seg.ny - 1), 0.0);
  seg.h.assign((seg.nx - 1) * seg.ny, 0.0);
  const int halfw = std::max(1, static_cast<int>(std::lround(0.1 * p)));
  for (int i = 0; i < seg.nx; ++i) {
    for (int j = 0; j + 1 < seg.ny; ++j) {
      seg.V(i, j) = segment_strength(rv, true, g.lattice_x[i], g.lattice_y[j], g.lattice_y[j + 1], halfw);
    }
  }
  for (int j = 0; j < seg.ny; ++j) {
    for (int i = 0; i + 1 < seg.nx; ++i) {
      seg.Hs(i, j) = segment_strength(rh, false, g.lattice_y[j], g.lattice_x[i], g.lattice_x[i + 1], halfw);
    }
  }
  // 线段是否「有线」：
  //  - 两侧都是页面底色（空白格）的线段，强度直接反映线本身的对比度。若这些强度明显分成两档
  //    （比如 0_2 图例区背景里的浅灰辅助网格 vs 图纸的黑线），弱的一档视为装饰性辅助网格，不算线；
  //  - 其他线段（至少一侧是有色格）只要求超过一个很低的绝对阈值：深色格之间的线对比度本来就低。
  const color::RGB8 page_bg = estimate_page_background(rgb);
  const color::Lab page_lab = color::srgb8_to_lab(page_bg);
  const int ncx = seg.nx - 1, ncy = seg.ny - 1;
  std::vector<uint8_t> blank(static_cast<size_t>(ncx) * ncy, 0);
  for (int j = 0; j < ncy; ++j) {
    for (int i = 0; i < ncx; ++i) {
      const CellSample cs =
          sample_cell(rgb, gray, {g.lattice_x[i], g.lattice_x[i + 1], g.lattice_y[j], g.lattice_y[j + 1]});
      blank[j * ncx + i] = !cs.has_text && color::delta_e_2000(cs.bg_lab, page_lab) < 5.0;
    }
  }
  auto is_blank = [&](int i, int j) { return i >= 0 && j >= 0 && i < ncx && j < ncy && blank[j * ncx + i]; };
  // 竖线段 (i, j) 两侧是格子 (i-1, j) 和 (i, j)；横线段 (i, j) 两侧是 (i, j-1) 和 (i, j)
  std::vector<double> bb;
  for (int i = 0; i < seg.nx; ++i) {
    for (int j = 0; j < ncy; ++j) {
      if (is_blank(i - 1, j) && is_blank(i, j)) bb.push_back(seg.V(i, j));
    }
  }
  for (int j = 0; j < seg.ny; ++j) {
    for (int i = 0; i < ncx; ++i) {
      if (is_blank(i, j - 1) && is_blank(i, j)) bb.push_back(seg.Hs(i, j));
    }
  }
  const double t_low = 10.0;
  double t_blank = t_low;
  if (bb.size() >= 20) {
    // Otsu（在 log 强度上），两档均值相差 3 倍以上才认为存在辅助网格
    std::vector<double> lv;
    for (double v : bb) lv.push_back(std::log1p(v));
    std::sort(lv.begin(), lv.end());
    const int n = static_cast<int>(lv.size());
    std::vector<double> pre(n + 1, 0.0);
    for (int k = 0; k < n; ++k) pre[k + 1] = pre[k] + lv[k];
    double best_var = -1, best_t = 0;
    int best_k = 0;
    for (int k = 1; k < n; ++k) {
      if (lv[k] == lv[k - 1]) continue;
      const double w0 = double(k) / n, w1 = 1.0 - w0;
      const double m0 = pre[k] / k, m1 = (pre[n] - pre[k]) / (n - k);
      const double var = w0 * w1 * (m0 - m1) * (m0 - m1);
      if (var > best_var) {
        best_var = var;
        best_t = 0.5 * (lv[k - 1] + lv[k]);
        best_k = k;
      }
    }
    if (best_k > 0) {
      const double m0 = std::expm1(pre[best_k] / best_k), m1 = std::expm1((pre[n] - pre[best_k]) / (n - best_k));
      const bool both = best_k >= std::max(10, n / 20) && (n - best_k) >= std::max(10, n / 20);
      if (both && m1 >= 3.0 * std::max(1.0, m0)) t_blank = std::max(t_low, std::expm1(best_t));
    }
  }
  // 两档强度也可能是「细线很淡、每 5 / 10 格一条深色粗线」：这时强的线是隔几条出现一次的，
  // 而 0_2 那种辅助网格的情况是主网格里每一条线都强（强线彼此相邻）。强线大多不相邻时取消辅助网格判定。
  if (t_blank > t_low) {
    auto strong_lines = [&](bool vertical) {
      const int nl = vertical ? seg.nx : seg.ny;
      const int ns = vertical ? ncy : ncx;
      std::vector<uint8_t> strong(nl, 0);
      for (int l = 0; l < nl; ++l) {
        int nb = 0, ns_strong = 0;
        for (int k = 0; k < ns; ++k) {
          const bool bbk = vertical ? (is_blank(l - 1, k) && is_blank(l, k)) : (is_blank(k, l - 1) && is_blank(k, l));
          if (!bbk) continue;
          ++nb;
          ns_strong += (vertical ? seg.V(l, k) : seg.Hs(k, l)) >= t_blank;
        }
        strong[l] = nb >= 3 && ns_strong * 2 >= nb;
      }
      return strong;
    };
    int n_strong = 0, n_adjacent = 0;
    for (bool vertical : {true, false}) {
      const auto st = strong_lines(vertical);
      for (size_t l = 0; l < st.size(); ++l) {
        if (!st[l]) continue;
        ++n_strong;
        if ((l > 0 && st[l - 1]) || (l + 1 < st.size() && st[l + 1])) ++n_adjacent;
      }
    }
    if (n_strong == 0 || n_adjacent * 2 < n_strong) t_blank = t_low;
  }
  g.seg_threshold = t_blank;
  auto present_v = [&](int i, int j) {
    const double v = seg.V(i, j);
    if (v < t_low) return false;
    return !(is_blank(i - 1, j) && is_blank(i, j)) || v >= t_blank;
  };
  auto present_h = [&](int i, int j) {
    const double v = seg.Hs(i, j);
    if (v < t_low) return false;
    return !(is_blank(i, j - 1) && is_blank(i, j)) || v >= t_blank;
  };
  progress.report(0.7f);

  // 四边都有线的格子 → 4 连通块
  const int cx = ncx, cy = ncy;
  std::vector<uint8_t> framed(cx * cy, 0);
  for (int j = 0; j < cy; ++j) {
    for (int i = 0; i < cx; ++i) {
      // 至少 3 条边有线：大片同色区域里偶尔有线读不出来；而图例区、网格外的空白行最多只有 2 条边
      framed[j * cx + i] = (present_v(i, j) + present_v(i + 1, j) + present_h(i, j) + present_h(i, j + 1)) >= 3;
    }
  }
  for (int j = 0; j < cy; ++j) {
    std::string row;
    for (int i = 0; i < cx; ++i) {
      row.push_back(char('0' + present_v(i, j) + present_v(i + 1, j) + present_h(i, j) + present_h(i, j + 1)));
    }
    g.border_map.push_back(row);
  }
  struct Comp {
    int n = 0, i0, i1, j0, j1;  // 格子下标闭区间
  };
  std::vector<Comp> comps;
  std::vector<int> label(cx * cy, -1);
  for (int j = 0; j < cy; ++j) {
    for (int i = 0; i < cx; ++i) {
      if (!framed[j * cx + i] || label[j * cx + i] >= 0) continue;
      Comp c{0, i, i, j, j};
      std::queue<std::pair<int, int>> q;
      q.push({i, j});
      label[j * cx + i] = static_cast<int>(comps.size());
      while (!q.empty()) {
        auto [a, b] = q.front();
        q.pop();
        ++c.n;
        c.i0 = std::min(c.i0, a);
        c.i1 = std::max(c.i1, a);
        c.j0 = std::min(c.j0, b);
        c.j1 = std::max(c.j1, b);
        const int nb[4][2] = {{a - 1, b}, {a + 1, b}, {a, b - 1}, {a, b + 1}};
        for (const auto& d : nb) {
          if (d[0] < 0 || d[1] < 0 || d[0] >= cx || d[1] >= cy) continue;
          const int k = d[1] * cx + d[0];
          if (framed[k] && label[k] < 0) {
            label[k] = static_cast<int>(comps.size());
            q.push({d[0], d[1]});
          }
        }
      }
      comps.push_back(c);
    }
  }
  if (comps.empty()) {
    g.fail_reason = "没有找到至少三边有线的格子";
    return g;
  }
  std::sort(comps.begin(), comps.end(), [](const Comp& a, const Comp& b) { return a.n > b.n; });
  // 主块 + 紧挨着它、且和它一样宽（或一样高）的碎块：同一张图纸被一片读不出线的区域隔开的部分。
  // 图例表格这类窄块不会被并进来。
  Comp m = comps.front();
  for (bool merged = true; merged;) {
    merged = false;
    for (size_t k = 1; k < comps.size(); ++k) {
      Comp& c = comps[k];
      if (c.n == 0 || c.n < 4) continue;
      const int w = m.i1 - m.i0 + 1, h = m.j1 - m.j0 + 1;
      const int ov_i = std::min(m.i1, c.i1) - std::max(m.i0, c.i0) + 1;
      const int ov_j = std::min(m.j1, c.j1) - std::max(m.j0, c.j0) + 1;
      const bool near_v = c.j0 <= m.j1 + 2 && c.j1 >= m.j0 - 2;
      const bool near_h = c.i0 <= m.i1 + 2 && c.i1 >= m.i0 - 2;
      const bool stack_v = near_v && ov_i >= 0.6 * w;  // 上下相邻且同宽
      const bool stack_h = near_h && ov_j >= 0.6 * h;  // 左右相邻且同高
      const bool inside = ov_i == c.i1 - c.i0 + 1 && ov_j == c.j1 - c.j0 + 1;
      if (stack_v || stack_h || inside) {
        m.i0 = std::min(m.i0, c.i0);
        m.i1 = std::max(m.i1, c.i1);
        m.j0 = std::min(m.j0, c.j0);
        m.j1 = std::max(m.j1, c.j1);
        m.n += c.n;
        c.n = 0;
        merged = true;
      }
    }
  }
  comps.front() = m;
  std::stable_sort(comps.begin(), comps.end(), [](const Comp& a, const Comp& b) { return a.n > b.n; });
  for (const auto& c : comps) {
    if (c.n < 4) break;
    g.components.push_back(cv::Rect(cv::Point(int(g.lattice_x[c.i0]), int(g.lattice_y[c.j0])),
                                    cv::Point(int(g.lattice_x[c.i1 + 1]), int(g.lattice_y[c.j1 + 1]))));
  }
  int i0 = m.i0, i1 = m.i1 + 1, j0 = m.j0, j1 = m.j1 + 1;  // 线下标
  {
    int fill = 0;
    for (int j = m.j0; j <= m.j1; ++j) {
      for (int i = m.i0; i <= m.i1; ++i) fill += framed[j * cx + i];
    }
    g.frame_fill = double(fill) / double((m.i1 - m.i0 + 1) * (m.j1 - m.j0 + 1));
  }
  if (i1 - i0 < 2 || j1 - j0 < 2) {
    g.fail_reason = "网格太小";
    return g;
  }

  // 坐标栏：上、下、左、右各判断一次（不含四角）
  auto cell = [&](int i, int j) {
    return CellRect{g.lattice_x[i], g.lattice_x[i + 1], g.lattice_y[j], g.lattice_y[j + 1]};
  };
  auto row_cells = [&](int j) {
    std::vector<CellRect> v;
    for (int i = i0 + 1; i < i1 - 1; ++i) v.push_back(cell(i, j));
    return v;
  };
  auto col_cells = [&](int i) {
    std::vector<CellRect> v;
    for (int j = j0 + 1; j < j1 - 1; ++j) v.push_back(cell(i, j));
    return v;
  };
  // 一行 / 一列里四边（至少三边）框住的格子占多少
  auto row_fill = [&](int j) {
    int f = 0;
    for (int i = i0; i < i1; ++i) f += framed[j * cx + i];
    return double(f) / std::max(1, i1 - i0);
  };
  auto col_fill = [&](int i) {
    int f = 0;
    for (int j = j0; j < j1; ++j) f += framed[j * cx + i];
    return double(f) / std::max(1, j1 - j0);
  };
  // 主块外沿多连进来的一条「缝」：坐标栏或网格最后一行之外、到图例框 / 装饰边框之间那一行，格线断断续续，
  // 个别格子凑够三条边就连进了主块。特征是框住的格子不到 15%、而紧挨着的里面一行几乎全框住（≥ 80%）。
  // 图纸自己的空白边行格线很淡时框住的也少，但那是一片逐渐变淡的区域，里面一行通常也框不全（合成集 eval60 #34）
  for (int pass = 0; pass < 2; ++pass) {
    bool any = false;
    if (j1 - j0 >= 4 && row_fill(j0) < 0.15 && row_fill(j0 + 1) >= 0.8) ++j0, any = true;
    if (j1 - j0 >= 4 && row_fill(j1 - 1) < 0.15 && row_fill(j1 - 2) >= 0.8) --j1, any = true;
    if (i1 - i0 >= 4 && col_fill(i0) < 0.15 && col_fill(i0 + 1) >= 0.8) ++i0, any = true;
    if (i1 - i0 >= 4 && col_fill(i1 - 1) < 0.15 && col_fill(i1 - 2) >= 0.8) --i1, any = true;
    if (!any) break;
    g.header_debug.push_back("trimmed sparse edge");
  }
  // 每边最多剥两圈（有的样式在格子外还有一行带框的坐标）。
  // 坐标栏外面有时还夹着一条只有零星几格框住的缝（比如坐标栏和下方图例框之间：格线断断续续，个别格子凑够三条边
  // 就连进了主块）：最外一行不是坐标栏、框住的格子不到一半、而紧挨着的里面一行是坐标栏时，两行一起剥掉。
  // 只有里面是坐标栏才剥：空白区格线很淡时，图纸自己的边行也可能框住的格子不到一半
  for (int pass = 0; pass < 2; ++pass) {
    if (i1 - i0 < 3 || j1 - j0 < 3) break;
    std::string w[4];
    auto side = [&](int k, bool row, int inward, std::string* why) {
      if (is_header_line(rgb, gray, row ? row_cells(k) : col_cells(k), why)) return 1;
      const int lim = row ? j1 - j0 : i1 - i0;
      if (lim < 4 || (row ? row_fill(k) : col_fill(k)) >= 0.5) return 0;
      std::string inner;
      if (!is_header_line(rgb, gray, row ? row_cells(k + inward) : col_cells(k + inward), &inner)) return 0;
      if (why) *why += " | 外面是缝，里面一行：" + inner;
      return 2;
    };
    const int top = side(j0, true, +1, &w[0]);
    const int bottom = side(j1 - 1, true, -1, &w[1]);
    const int left = side(i0, false, +1, &w[2]);
    const int right = side(i1 - 1, false, -1, &w[3]);
    for (const auto& x : w) g.header_debug.push_back(x);
    g.header_stripped[0] |= top > 0;
    g.header_stripped[1] |= bottom > 0;
    g.header_stripped[2] |= left > 0;
    g.header_stripped[3] |= right > 0;
    j0 += top;
    j1 -= bottom;
    i0 += left;
    i1 -= right;
    if (!(top || bottom || left || right)) break;
  }
  if (i1 - i0 < 1 || j1 - j0 < 1) {
    g.fail_reason = "去掉坐标栏后没有格子";
    return g;
  }
  if (i1 - i0 >= 3 && j1 - j0 >= 3) {
    is_header_line(rgb, gray, row_cells(j0), nullptr, &g.header_candidate[0]);
    is_header_line(rgb, gray, row_cells(j1 - 1), nullptr, &g.header_candidate[1]);
    is_header_line(rgb, gray, col_cells(i0), nullptr, &g.header_candidate[2]);
    is_header_line(rgb, gray, col_cells(i1 - 1), nullptr, &g.header_candidate[3]);
  }

  g.xs.assign(g.lattice_x.begin() + i0, g.lattice_x.begin() + i1 + 1);
  g.ys.assign(g.lattice_y.begin() + j0, g.lattice_y.begin() + j1 + 1);
  g.pitch_x = (g.xs.back() - g.xs.front()) / g.cols();
  g.pitch_y = (g.ys.back() - g.ys.front()) / g.rows();
  g.found = true;
  progress.report(1.0f);
  return g;
}

}  // namespace bc
