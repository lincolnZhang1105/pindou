// 识别管线编排（v0，T0.10）：layout → grid → legend → classify（只用颜色）→ solver。
#include "engine/engine.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <tuple>

#include "classify/assign.h"
#include "classify/cells.h"
#include "classify/text_evidence.h"
#include "grid/coords.h"
#include "grid/grid.h"
#include "image/image_ops.h"
#include "legend/legend.h"

#ifndef BC_VERSION_STRING
#define BC_VERSION_STRING "0.0.0"
#endif

namespace bc {
namespace {

bool g_debug = false;

int gray_of(const color::RGB8& c) { return (299 * c[0] + 587 * c[1] + 114 * c[2] + 500) / 1000; }

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

double round_to(double v, double q) { return std::round(v / q) * q; }

ojson lines_json(const std::vector<double>& v) {
  ojson a = ojson::array();
  for (double x : v) a.push_back(round_to(x, 0.01));
  return a;
}

// 一页（一张图）上的网格与采样结果
struct PageResult {
  int image_index = 0;
  std::vector<double> xs, ys;
  std::string source = "auto";
  GridResult grid;  // source == auto 时有效
  cv::Rect exclude;
  std::vector<CellSample> samples;
  LegendResult legend;
  color::RGB8 page_bg{};
  int col_offset = 0, row_offset = 0;
  // 坐标数字给出的第一列 / 第一行的编号；-1 表示不知道。编号可能从 0 开始也可能从 1 开始，只用来算页与页的相对位置
  int abs_col0 = -1, abs_row0 = -1;
  bool legend_only = false;         // 多图输入里只有图例、没有网格的那张
  int cols() const { return std::max(0, static_cast<int>(xs.size()) - 1); }
  int rows() const { return std::max(0, static_cast<int>(ys.size()) - 1); }
};

cv::Rect lines_bbox(const std::vector<double>& xs, const std::vector<double>& ys) {
  return cv::Rect(cv::Point(static_cast<int>(std::floor(xs.front())), static_cast<int>(std::floor(ys.front()))),
                  cv::Point(static_cast<int>(std::ceil(xs.back())), static_cast<int>(std::ceil(ys.back()))));
}

std::vector<double> json_lines(const nlohmann::json& j, const char* key) {
  require(j.contains(key) && j[key].is_array(), BC_ERR_INVALID_INPUT, std::string("pages 缺少 ") + key);
  std::vector<double> v;
  for (const auto& x : j[key]) v.push_back(x.get<double>());
  for (size_t i = 1; i < v.size(); ++i) {
    require(v[i] > v[i - 1], BC_ERR_INVALID_INPUT, std::string(key) + " 必须严格递增");
  }
  return v;
}

// 非空格的 8 连通块（判断多图合集 T7 用）
std::vector<cv::Rect> object_components(const std::vector<int>& cls, int cols, int rows, int min_cells) {
  std::vector<int> lab(cls.size(), -1);
  std::vector<cv::Rect> out;
  for (int j = 0; j < rows; ++j) {
    for (int i = 0; i < cols; ++i) {
      const int k0 = j * cols + i;
      if (cls[k0] < 0 || lab[k0] >= 0) continue;
      int n = 0, i0 = i, i1 = i, j0 = j, j1 = j;
      std::queue<int> q;
      q.push(k0);
      lab[k0] = 1;
      while (!q.empty()) {
        const int k = q.front();
        q.pop();
        ++n;
        const int a = k % cols, b = k / cols;
        i0 = std::min(i0, a);
        i1 = std::max(i1, a);
        j0 = std::min(j0, b);
        j1 = std::max(j1, b);
        for (int db = -1; db <= 1; ++db) {
          for (int da = -1; da <= 1; ++da) {
            const int x = a + da, y = b + db;
            if (x < 0 || y < 0 || x >= cols || y >= rows) continue;
            const int kk = y * cols + x;
            if (cls[kk] >= 0 && lab[kk] < 0) {
              lab[kk] = 1;
              q.push(kk);
            }
          }
        }
      }
      if (n >= min_cells) out.push_back(cv::Rect(i0, j0, i1 - i0 + 1, j1 - j0 + 1));
    }
  }
  return out;
}

// 所有色卡里的色号（规范化后 → 原写法），封闭词表打分用
std::map<std::string, std::string> code_vocab(const PaletteSet& ps) {
  std::map<std::string, std::string> v;
  for (const auto& p : ps.palettes()) {
    for (const auto& c : p.colors) v.emplace(normalize_code(c.code), c.code);
  }
  return v;
}

// 读图例色号：贪心解码的结果在词表里就用它；否则在词表里按 CTC 概率找最可能的色号（长度差 ≤ 1），
// 比贪心结果的概率低得太多（e^-6 倍以下）就不采用，保留贪心结果
std::optional<std::string> read_code(const nn::OcrModel& m, const cv::Mat& gray, const cv::Rect& box,
                                     const std::map<std::string, std::string>& vocab) {
  const nn::OcrText t = nn::read_text(m, gray, box);
  const std::string greedy = normalize_code(t.text);
  if (greedy.empty()) return std::nullopt;
  if (vocab.empty() || vocab.count(greedy)) return greedy;
  double best = -1e18;
  std::string best_code;
  for (const auto& [norm, orig] : vocab) {
    if (std::abs(static_cast<int>(norm.size()) - static_cast<int>(greedy.size())) > 1) continue;
    const double lp = nn::ctc_logprob(t, m.charset, norm);
    if (lp > best) {
      best = lp;
      best_code = norm;
    }
  }
  if (!best_code.empty() && best >= t.logp - 6.0) return best_code;
  return greedy;
}

// 读颗数：x534、×534、534、534颗（模型把中文量词读成 #）、(534) 都接受，其余格式不认
std::optional<int> parse_count(std::string s) {
  while (!s.empty() && s.front() == 'x') s.erase(s.begin());
  while (!s.empty() && s.back() == '#') s.pop_back();
  if (s.size() >= 2 && s.front() == '(' && s.back() == ')') s = s.substr(1, s.size() - 2);
  if (s.empty() || s.size() > 6) return std::nullopt;
  for (char ch : s) {
    if (ch < '0' || ch > '9') return std::nullopt;
  }
  return std::stoi(s);
}

std::optional<int> read_count(const nn::OcrModel& m, const cv::Mat& gray, const cv::Rect& box) {
  return parse_count(nn::read_text(m, gray, box).text);
}

// 一排格子里写的是不是一串连续的数字（坐标栏）。每格贪心读一次，读成整数的格子投票定起始编号
// （递增：编号 − 序号；递减：编号 + 序号），再用 CTC 检验每格是不是「起始编号 ± 序号」：
// 和贪心结果的概率差在 e^3 倍以内就算通过。坐标栏的字常常很小（格距 12 像素时三位数只有 4 像素高），
// OCR 只能读对一半左右，所以要求：至少 4 格投出同一个起始编号，且 45% 以上的格子通过检验。
// 同一色号连成一排时每格读到的数字相同，投不出一致的起始编号（纯数字色号的品牌也一样）
bool is_number_run(const nn::OcrModel& m, const cv::Mat& gray, const std::vector<cv::Rect>& boxes, std::string* why) {
  if (boxes.size() < 5) return false;
  std::vector<nn::OcrText> reads;
  std::map<int, int> up, down;
  for (size_t k = 0; k < boxes.size(); ++k) {
    reads.push_back(nn::read_text(m, gray, boxes[k]));
    const std::string& s = reads.back().text;
    if (s.empty() || s.size() > 3 || !std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; })) continue;
    const int v = std::stoi(s), kk = static_cast<int>(k);
    ++up[v - kk];
    ++down[v + kk];
  }
  int best = 0, base = 0, dir = 0;
  for (const auto& [b, c] : up) {
    if (c > best) best = c, base = b, dir = +1;
  }
  for (const auto& [b, c] : down) {
    if (c > best) best = c, base = b, dir = -1;
  }
  if (why) *why = "votes=" + std::to_string(best) + "/" + std::to_string(boxes.size());
  if (best < 4) return false;
  int pass = 0;
  for (size_t k = 0; k < boxes.size(); ++k) {
    const int v = base + dir * static_cast<int>(k);
    if (v < 0) continue;
    pass += nn::ctc_logprob(reads[k], m.charset, std::to_string(v)) >= reads[k].logp - 3.0;
  }
  if (why) *why += " pass=" + std::to_string(pass);
  return pass >= 0.45 * boxes.size();
}

}  // namespace

void set_debug(bool on) { g_debug = on; }

ojson recognize(const Context& ctx, const RecognizeRequest& req, const Progress& progress) {
  const auto t_all = Clock::now();
  ojson timing;
  ojson warnings = ojson::array();
  ojson debug;
  progress.report(0.0f);

  const int n_img = static_cast<int>(req.images.size());
  std::vector<cv::Mat> rgbs, grays;
  for (const auto& m : req.images) {
    rgbs.push_back(rgba_to_rgb(m));
    grays.push_back(rgb_to_gray(rgbs.back()));
  }

  // ---------------------------------------------------------------- layout + grid
  auto t = Clock::now();
  std::vector<PageResult> pages;
  if (req.pages) {
    for (const auto& pj : *req.pages) {
      PageResult pr;
      pr.image_index = pj.value("image_index", 0);
      require(pr.image_index >= 0 && pr.image_index < n_img, BC_ERR_INVALID_INPUT, "pages.image_index 越界");
      pr.xs = json_lines(pj, "x_lines");
      pr.ys = json_lines(pj, "y_lines");
      require(pr.xs.size() >= 2 && pr.ys.size() >= 2, BC_ERR_INVALID_INPUT, "pages 的网格线太少");
      pr.col_offset = pj.value("col_offset", 0);
      pr.row_offset = pj.value("row_offset", 0);
      pr.source = "manual";
      const cv::Rect bb = lines_bbox(pr.xs, pr.ys);
      const int pad = static_cast<int>(std::ceil(1.2 * (pr.xs.back() - pr.xs.front()) / pr.cols()));
      pr.exclude = cv::Rect(bb.x - pad, bb.y - pad, bb.width + 2 * pad, bb.height + 2 * pad);
      pages.push_back(std::move(pr));
    }
  } else {
    for (int i = 0; i < n_img; ++i) {
      const Progress sub = progress.sub(0.05f + 0.4f * i / n_img, 0.05f + 0.4f * (i + 1) / n_img);
      PageResult pr;
      pr.image_index = i;
      pr.grid = detect_grid(rgbs[i], grays[i], sub);
      if (g_debug) {
        debug["grid"][i] = {{"lattice_x", lines_json(pr.grid.lattice_x)},
                            {"lattice_y", lines_json(pr.grid.lattice_y)},
                            {"seg_threshold", pr.grid.seg_threshold},
                            {"period9", pr.grid.period9},
                            {"period3", pr.grid.period3},
                            {"frame_fill", pr.grid.frame_fill},
                            {"header_stripped", {pr.grid.header_stripped[0], pr.grid.header_stripped[1],
                                                 pr.grid.header_stripped[2], pr.grid.header_stripped[3]}},
                            {"fail_reason", pr.grid.fail_reason},
                            {"border_map", pr.grid.border_map},
                            {"header_debug", pr.grid.header_debug},
                            {"comb_debug", pr.grid.comb_debug}};
      }
      if (!pr.grid.found) {
        // 多图输入时，没有网格的那张可能只是图例页（分页图常把图例单独放一张）
        if (n_img == 1) fail(BC_ERR_NO_GRID_FOUND, "没有找到网格：" + pr.grid.fail_reason);
        pr.legend_only = true;
        pr.exclude = cv::Rect();
        pages.push_back(std::move(pr));
        continue;
      }
      // 多个大小相近的网格块：多图合集（T7），需要用户框选
      const auto& comps = pr.grid.components;
      if (g_debug) {
        ojson cj = ojson::array();
        for (const auto& c : comps) cj.push_back({c.x, c.y, c.width, c.height});
        debug["grid"][i]["components"] = cj;
      }
      // 多图输入（分页图）时每一张都是用户给的一页，不做拼图判断
      if (n_img == 1 && comps.size() >= 2 && comps[1].area() >= 0.4 * comps[0].area()) {
        if (!g_debug) fail(BC_ERR_NEEDS_USER_CROP, "图里有多个独立的网格，请框选其中一个图案");
        warnings.push_back("debug_multi_component");
      }
      pr.xs = pr.grid.xs;
      pr.ys = pr.grid.ys;
      // 坐标数字（有 OCR 时）：去掉坐标所在的行 / 列；坐标逐格标注时，以坐标指出的范围为准
      if (ctx.ocr) {
        // 0) 网格检测标出的「疑似坐标栏」（外圈底色一致、几乎每格有字，只是字形对比分不开）：逐格读一遍，
        //    是一串连续的数字就剥掉，并记为已剥掉的坐标栏（下面按坐标定范围时不再往这一侧扩）
        for (int side = 0; side < 4; ++side) {
          if (!pr.grid.header_candidate[side] || pr.xs.size() < 5 || pr.ys.size() < 5) continue;
          const bool row = side < 2;
          const auto& along = row ? pr.xs : pr.ys;
          const auto& across = row ? pr.ys : pr.xs;
          const double a0 = side % 2 == 0 ? across[0] : across[across.size() - 2];
          const double a1 = side % 2 == 0 ? across[1] : across.back();
          std::vector<cv::Rect> boxes;
          for (size_t k = 1; k + 2 < along.size(); ++k) {  // 不含两个角
            const double b0 = along[k], b1 = along[k + 1];
            const double mx = 0.1 * (row ? b1 - b0 : a1 - a0), my = 0.1 * (row ? a1 - a0 : b1 - b0);
            const double x0 = row ? b0 : a0, x1 = row ? b1 : a1, y0 = row ? a0 : b0, y1 = row ? a1 : b1;
            boxes.push_back(cv::Rect(cv::Point(static_cast<int>(x0 + mx), static_cast<int>(y0 + my)),
                                     cv::Point(static_cast<int>(x1 - mx), static_cast<int>(y1 - my))) &
                            cv::Rect(0, 0, grays[i].cols, grays[i].rows));
          }
          std::string why;
          const bool run = is_number_run(*ctx.ocr, grays[i], boxes, &why);
          if (g_debug) debug["grid"][i]["number_band"][side] = why + (run ? " → 剥掉" : "");
          if (!run) continue;
          auto& v = row ? pr.ys : pr.xs;
          if (side % 2 == 0) {
            v.erase(v.begin());
          } else {
            v.pop_back();
          }
          pr.grid.header_stripped[side] = true;
        }
        const auto& lx = pr.grid.lattice_x;
        const auto& ly = pr.grid.lattice_y;
        int c0 = static_cast<int>(std::find(lx.begin(), lx.end(), pr.xs.front()) - lx.begin());
        int r0 = static_cast<int>(std::find(ly.begin(), ly.end(), pr.ys.front()) - ly.begin());
        int c1 = c0 + static_cast<int>(pr.xs.size()) - 2, r1 = r0 + static_cast<int>(pr.ys.size()) - 2;
        if (c1 + 1 < static_cast<int>(lx.size()) && r1 + 1 < static_cast<int>(ly.size())) {
          const auto cr = read_coords(*ctx.ocr, grays[i], gray_of(estimate_page_background(rgbs[i])), lx, ly, c0, c1, r0, r1);
          auto majority = [](const std::vector<int>& v, int idx) {
            return std::count(v.begin(), v.end(), idx) >= 3;
          };
          // 1) 网格最外一行 / 列里落了一排坐标数字：那是坐标行
          while (r1 > r0 + 1 && majority(cr.label_rows, r1)) --r1;
          while (r1 > r0 + 1 && majority(cr.label_rows, r0)) ++r0;
          while (c1 > c0 + 1 && majority(cr.label_cols, c1)) --c1;
          while (c1 > c0 + 1 && majority(cr.label_cols, c0)) ++c0;
          // 2) 坐标逐格标注、读到的够多：范围以坐标为准（只在和现有范围大体一致时采用，防止读错）
          // 坐标说网格更大：扩展；坐标说网格更小：只有被去掉的那一列 / 行在坐标带里根本没有字时才缩
          // （有字但没读出来，多半是 OCR 漏读了最后一个坐标）。
          // 网格检测已经把那一侧当坐标栏剥掉了的，不再往那一侧扩：坐标栏拐角处的数字（比如下方列坐标的第一个「30」
          // 正好落在左侧行坐标那一列）会被当成多一行的行坐标
          auto adopt = [](const CoordFit& f, int& a, int& b, int n_lines, const std::vector<int>& has_word, bool lock_a,
                          bool lock_b) {
            if (!f.ok || !f.dense || f.support < 5) return;
            const int na = std::max(0, f.first), nb = std::min(n_lines - 2, f.last);
            const int overlap = std::min(b, nb) - std::max(a, na) + 1;
            if (nb <= na || overlap < 0.8 * (b - a + 1)) return;
            auto worded = [&](int idx) { return std::find(has_word.begin(), has_word.end(), idx) != has_word.end(); };
            if (na < a) {
              if (!lock_a) a = na;
            } else {
              while (a < na && !worded(a)) ++a;
            }
            if (nb > b) {
              if (!lock_b) b = nb;
            } else {
              while (b > nb && !worded(b)) --b;
            }
          };
          const auto& hs = pr.grid.header_stripped;  // 上、下、左、右
          adopt(cr.cols, c0, c1, static_cast<int>(lx.size()), cr.cols_with_word, hs[2], hs[3]);
          adopt(cr.rows, r0, r1, static_cast<int>(ly.size()), cr.rows_with_word, hs[0], hs[1]);
          // 分页图拼接用：这一页第一列 / 第一行的坐标编号
          if (cr.cols.ok && c0 - cr.cols.k >= 0) pr.abs_col0 = c0 - cr.cols.k;
          if (cr.rows.ok && r0 - cr.rows.k >= 0) pr.abs_row0 = r0 - cr.rows.k;
          pr.xs.assign(lx.begin() + c0, lx.begin() + c1 + 2);
          pr.ys.assign(ly.begin() + r0, ly.begin() + r1 + 2);
          if (g_debug) {
            debug["grid"][i]["coords"] = {{"cols", {cr.cols.ok, cr.cols.first, cr.cols.last, cr.cols.support, cr.cols.dense}},
                                          {"rows", {cr.rows.ok, cr.rows.first, cr.rows.last, cr.rows.support, cr.rows.dense}},
                                          {"label_rows", cr.label_rows}, {"label_cols", cr.label_cols},
                                          {"words", cr.debug}};
          }
        }
      }
      // 网格只占图片很小一块、而网格外还有比网格本身更多的内容：多半是封面拼图（T7）
      {
        const cv::Mat& rgb = rgbs[i];
        const cv::Rect gb = lines_bbox(pr.xs, pr.ys) & cv::Rect(0, 0, rgb.cols, rgb.rows);
        const double grid_frac = double(gb.area()) / (double(rgb.cols) * rgb.rows);
        if (grid_frac < 0.15 && n_img == 1) {
          const color::RGB8 bg = estimate_page_background(rgb);
          const double pitch = std::max(pr.grid.pitch_x, pr.grid.pitch_y);
          const int pad = static_cast<int>(std::ceil(3 * pitch));
          const cv::Rect keep(gb.x - pad, gb.y - pad, gb.width + 2 * pad, gb.height + 2 * pad + static_cast<int>(gb.height * 0.6));
          long long busy = 0;
          for (int y = 0; y < rgb.rows; y += 2) {
            const uint8_t* rp = rgb.ptr<uint8_t>(y);
            for (int x = 0; x < rgb.cols; x += 2) {
              if (keep.contains(cv::Point(x, y))) continue;
              const uint8_t* q = rp + 3 * x;
              if (std::max({std::abs(q[0] - bg[0]), std::abs(q[1] - bg[1]), std::abs(q[2] - bg[2])}) > 24) busy += 4;
            }
          }
          if (g_debug) debug["grid"][i]["collage_check"] = {{"grid_frac", grid_frac}, {"busy_px", busy}, {"grid_px", gb.area()}};
          if (busy > gb.area()) fail(BC_ERR_NEEDS_USER_CROP, "网格只占图片的一小部分，图里像是有多个图案，请框选其中一个");
        }
      }
      const double pitch = std::max(pr.grid.pitch_x, pr.grid.pitch_y);
      const cv::Rect bb = comps.empty() ? lines_bbox(pr.xs, pr.ys) : comps.front();
      // 排除区外扩 0.6 格：盖住紧贴网格的坐标数字，又不切到下方的图例
      const int pad = static_cast<int>(std::ceil(0.6 * pitch));
      pr.exclude = cv::Rect(bb.x - pad, bb.y - pad, bb.width + 2 * pad, bb.height + 2 * pad);
      pages.push_back(std::move(pr));
    }
  }
  timing["grid"] = std::lround(ms_since(t));

  // 分页图里单独的图例页：色块排得整齐时也会被认成一个小网格。同一套分页图来自同一次导出，各页格距基本一致；
  // 格数很少（≤ 100）、格距又和其他网格页的中位数差 1.6 倍以上的，按图例页处理（板边角的小页格距和别的页一样，不受影响）
  if (!req.pages && n_img > 1) {
    auto pitch_of = [](const PageResult& p) { return std::max(p.grid.pitch_x, p.grid.pitch_y); };
    std::vector<size_t> flip;
    for (size_t a = 0; a < pages.size(); ++a) {
      if (pages[a].legend_only || pages[a].cols() * pages[a].rows() > 100) continue;
      std::vector<double> others;
      for (size_t b = 0; b < pages.size(); ++b) {
        if (b != a && !pages[b].legend_only) others.push_back(pitch_of(pages[b]));
      }
      if (others.empty()) continue;
      std::nth_element(others.begin(), others.begin() + others.size() / 2, others.end());
      const double r = pitch_of(pages[a]) / others[others.size() / 2];
      if (r > 1.6 || r < 1 / 1.6) flip.push_back(a);
    }
    for (size_t a : flip) {
      pages[a].legend_only = true;
      pages[a].xs.clear();
      pages[a].ys.clear();
      pages[a].exclude = cv::Rect();
      pages[a].abs_col0 = pages[a].abs_row0 = -1;
      if (g_debug) debug["grid"][pages[a].image_index]["reclassified_legend_page"] = true;
    }
  }
  // 多页（B9）：网格页排在前面，图例页放后面
  std::stable_partition(pages.begin(), pages.end(), [](const PageResult& p) { return !p.legend_only; });
  require(!pages.empty() && !pages.front().legend_only, BC_ERR_NO_GRID_FOUND, "所有图片都没有找到网格");
  const int n_grid = static_cast<int>(std::count_if(pages.begin(), pages.end(), [](const PageResult& p) { return !p.legend_only; }));
  if (!req.pages && n_grid > 1) {
    // 1) 每页都读出了坐标：按坐标编号直接定位（相邻页重复的边界行列会落在同一位置）
    bool by_coords = true;
    int min_c = 1 << 30, min_r = 1 << 30;
    for (int k = 0; k < n_grid; ++k) {
      by_coords &= pages[k].abs_col0 >= 0 && pages[k].abs_row0 >= 0;
      min_c = std::min(min_c, pages[k].abs_col0);
      min_r = std::min(min_r, pages[k].abs_row0);
    }
    if (by_coords) {
      // 两页落在同一个位置说明坐标读错了，退回按版式猜
      for (int a = 0; a < n_grid && by_coords; ++a) {
        for (int b = a + 1; b < n_grid; ++b) {
          if (pages[a].abs_col0 == pages[b].abs_col0 && pages[a].abs_row0 == pages[b].abs_row0) by_coords = false;
        }
      }
    }
    if (by_coords) {
      for (int k = 0; k < n_grid; ++k) {
        pages[k].col_offset = pages[k].abs_col0 - min_c;
        pages[k].row_offset = pages[k].abs_row0 - min_r;
      }
    } else {
      // 2) 按版式猜：枚举每行放几张（per_row），先纯按累加摆放，挑最自洽的一种：
      //    a. 读出了坐标的页：「坐标编号 − 摆放位置」应该是同一个常数（列、行分别算，允许 ±2），不一致的页越少越好
      //       （分页图常常只在整张图纸的外框上标坐标：最上一排页有列号、最左一列页有行号）；
      //    b. 同一张大图切出来的页，同一行的行数相同、同一列的列数相同：每对相同 +1、不同 −1，分越高越好；
      //    c. 打平时按经验顺序（prior）：列坐标回到最小值处断行、按板切、平方数排成 s×s、
      //       行数都相同排成一行、列数都相同排成一列、⌈√n⌉ 张一行；其余按离 √n 的远近
      int per_row = n_grid;
      const int s = static_cast<int>(std::lround(std::sqrt(double(n_grid))));
      const bool same_rows = std::all_of(pages.begin(), pages.begin() + n_grid, [&](const PageResult& p) { return p.rows() == pages[0].rows(); });
      const bool same_cols = std::all_of(pages.begin(), pages.begin() + n_grid, [&](const PageResult& p) { return p.cols() == pages[0].cols(); });
      // 按板切的分页：一行里除了最后一张都一样宽，最后一张是剩下的边角、明显更窄
      int maxc = 0, board_break = 0;
      for (int k = 0; k < n_grid; ++k) maxc = std::max(maxc, pages[k].cols());
      for (int k = 0; k + 1 < n_grid; ++k) {
        if (pages[k].cols() < maxc - 1) {
          board_break = k + 1;
          break;
        }
      }
      // 部分页读出了列坐标：列坐标「回到最小值」的那一页是新一行的开头（比如 1、44、87、1、44、87 → 每行 3 张）
      int coord_break = 0;
      {
        int min_known = 1 << 30;
        for (int k = 0; k < n_grid; ++k) {
          const int a = pages[k].abs_col0;
          if (a < 0) continue;
          if (k > 0 && min_known < (1 << 30) && a <= min_known) {
            coord_break = k;
            break;
          }
          min_known = std::min(min_known, a);
        }
      }
      int prior = 0;
      if (coord_break > 0) {
        prior = coord_break;
      } else if (board_break > 1) {
        prior = board_break;
      } else if (s * s == n_grid) {
        prior = s;
      } else if (same_rows) {
        prior = n_grid;
      } else if (same_cols) {
        prior = 1;
      } else {
        prior = static_cast<int>(std::ceil(std::sqrt(double(n_grid))));
      }
      // 纯按累加摆放
      auto stack = [&](int pr_n, std::vector<int>& co, std::vector<int>& ro) {
        co.assign(n_grid, 0);
        ro.assign(n_grid, 0);
        int row_off = 0;
        for (int start = 0; start < n_grid; start += pr_n) {
          int col_off = 0, band = 0;
          for (int k = start; k < std::min(n_grid, start + pr_n); ++k) {
            co[k] = col_off;
            ro[k] = row_off;
            col_off += pages[k].cols();
            band = std::max(band, pages[k].rows());
          }
          row_off += band;
        }
      };
      auto abs_of = [&](int k, bool col) { return col ? pages[k].abs_col0 : pages[k].abs_row0; };
      // 「坐标编号 − 累加位置」应是同一个常数。某一页尺寸差一两格会让后面的累加位置漂移，所以允许 ±2：
      // 取覆盖页数最多的常数（打平取先出现的）。返回 (常数, 不在误差内的页数)；没有页读出坐标时是 (0, 0)
      auto coord_fit = [&](bool col, const std::vector<int>& off) {
        std::vector<int> d;
        for (int k = 0; k < n_grid; ++k) {
          if (abs_of(k, col) >= 0) d.push_back(abs_of(k, col) - off[k]);
        }
        int base = 0, best = 0;
        for (int x : d) {
          const int c = static_cast<int>(std::count_if(d.begin(), d.end(), [&](int y) { return std::abs(y - x) <= 2; }));
          if (c > best) {
            best = c;
            base = x;
          }
        }
        return std::make_pair(base, static_cast<int>(d.size()) - best);
      };
      // 页面形状：同一行的页行数相同、同一列的页列数相同，每对相同 +1、不同 −1。
      // 只数「不同」的话，一整行排开这种比较次数很少的排法会占便宜
      auto shape_score = [&](int pr_n) {
        int score = 0;
        for (int start = 0; start < n_grid; start += pr_n) {
          for (int k = start + 1; k < std::min(n_grid, start + pr_n); ++k) score += pages[k].rows() == pages[start].rows() ? 1 : -1;
        }
        for (int j = 0; j < pr_n; ++j) {
          for (int k = j + pr_n; k < n_grid; k += pr_n) score += pages[k].cols() == pages[j].cols() ? 1 : -1;
        }
        return score;
      };
      std::vector<int> cands{prior};
      for (int c = 1; c <= n_grid; ++c) {
        if (c != prior) cands.push_back(c);
      }
      const double sq = std::sqrt(double(n_grid));
      std::stable_sort(cands.begin() + 1, cands.end(), [&](int a, int b) { return std::fabs(a - sq) < std::fabs(b - sq); });
      std::tuple<int, int, int> best_key{1 << 30, 1 << 30, 1 << 30};
      for (int rank = 0; rank < static_cast<int>(cands.size()); ++rank) {
        std::vector<int> co, ro;
        stack(cands[rank], co, ro);
        const std::tuple<int, int, int> key{coord_fit(true, co).second + coord_fit(false, ro).second,
                                            -shape_score(cands[rank]), rank};
        if (key < best_key) {
          best_key = key;
          per_row = cands[rank];
        }
      }
      if (g_debug) debug["b9_layout"] = {{"per_row", per_row}, {"prior", prior}, {"coord_bad", std::get<0>(best_key)},
                                         {"shape_score", -std::get<1>(best_key)}};
      // 摆放：读出了坐标的页直接按自己的坐标放（编号 − 基准），不受别的页尺寸误差影响；其余按累加补位。
      // 基准取第一行（行坐标）/ 第一列（列坐标）里读出了坐标的页：它们的累加位置一定是 0。这些页都没读出来时，
      // 用和累加位置最吻合的常数
      std::vector<int> co, ro;
      stack(per_row, co, ro);
      auto most_common = [](const std::map<int, int>& m) {
        int v = 0, best = 0;
        for (const auto& [x, c] : m) {
          if (c > best) {
            best = c;
            v = x;
          }
        }
        return v;
      };
      auto base_of = [&](bool col) {
        std::map<int, int> anchors;
        for (int k = 0; k < n_grid; ++k) {
          const bool first = col ? k % per_row == 0 : k < per_row;
          if (first && abs_of(k, col) >= 0) ++anchors[abs_of(k, col)];
        }
        return anchors.empty() ? coord_fit(col, col ? co : ro).first : most_common(anchors);
      };
      const int base_c = base_of(true), base_r = base_of(false);
      auto placed = [&](int k, bool col) { return abs_of(k, col) - (col ? base_c : base_r); };
      int row_off = 0;
      for (int start = 0; start < n_grid; start += per_row) {
        const int end = std::min(n_grid, start + per_row);
        // 这一行的起始行：行里读出了行坐标的页给出的位置（取最常见的），没有就接着上一行往下排
        std::map<int, int> rows_known;
        for (int k = start; k < end; ++k) {
          if (abs_of(k, false) >= 0 && placed(k, false) >= 0) ++rows_known[placed(k, false)];
        }
        if (!rows_known.empty()) row_off = most_common(rows_known);
        int col_off = 0, band_end = row_off;
        for (int k = start; k < end; ++k) {
          if (abs_of(k, true) >= 0 && placed(k, true) >= 0) col_off = placed(k, true);
          pages[k].col_offset = col_off;
          pages[k].row_offset = abs_of(k, false) >= 0 && placed(k, false) >= 0 ? placed(k, false) : row_off;
          col_off += pages[k].cols();
          band_end = std::max(band_end, pages[k].row_offset + pages[k].rows());
        }
        row_off = band_end;
      }
      warnings.push_back("pages_order_unverified");
    }
  }
  if (g_debug && n_grid > 1) {
    for (int k = 0; k < n_grid; ++k) {
      debug["b9"].push_back({{"image_index", pages[k].image_index}, {"abs_col0", pages[k].abs_col0},
                             {"abs_row0", pages[k].abs_row0}, {"cols", pages[k].cols()}, {"rows", pages[k].rows()},
                             {"col_offset", pages[k].col_offset}, {"row_offset", pages[k].row_offset}});
    }
  }
  int cols = 0, rows = 0;
  for (const auto& p : pages) {
    cols = std::max(cols, p.col_offset + p.cols());
    rows = std::max(rows, p.row_offset + p.rows());
  }
  require(cols > 0 && rows > 0, BC_ERR_NO_GRID_FOUND, "网格为空");
  require(static_cast<long long>(cols) * rows <= 1000LL * 1000LL, BC_ERR_INVALID_INPUT, "网格过大");
  progress.report(0.45f);

  // ---------------------------------------------------------------- legend
  t = Clock::now();
  for (auto& p : pages) {
    p.page_bg = estimate_page_background(rgbs[p.image_index]);
    p.legend = detect_legend(rgbs[p.image_index], p.exclude, p.page_bg);
  }
  timing["legend"] = std::lround(ms_since(t));
  // 图例文字：色号和颗数（有 OCR 模型时）
  if (ctx.ocr) {
    t = Clock::now();
    const auto vocab = code_vocab(ctx.palettes);
    for (auto& p : pages) {
      const cv::Mat& gray = grays[p.image_index];
      for (auto& it : p.legend.items) {
        if (it.code_box) it.code = read_code(*ctx.ocr, gray, *it.code_box, vocab);
        if (it.count_box) it.count = read_count(*ctx.ocr, gray, *it.count_box);
        // 几何位置给的角色读不通时，用内容校验：在分给这个色块的其他词里找合法的色号 / 颗数
        auto same = [](const cv::Rect& a, const std::optional<cv::Rect>& b) { return b && a == *b; };
        if ((!it.code || (!vocab.empty() && !vocab.count(*it.code))) && !it.near_words.empty()) {
          for (const auto& w : it.near_words) {
            if (same(w, it.count_box)) continue;
            const auto c = read_code(*ctx.ocr, gray, w, vocab);
            if (c && (vocab.empty() || vocab.count(*c))) {
              it.code = c;
              it.code_box = w;
              break;
            }
          }
        }
        if (!it.count && !it.near_words.empty()) {
          for (const auto& w : it.near_words) {
            if (same(w, it.code_box)) continue;
            const auto n = read_count(*ctx.ocr, gray, w);
            if (n) {
              it.count = n;
              it.count_box = w;
              break;
            }
          }
        }
      }
    }
    timing["ocr"] = std::lround(ms_since(t));
  }
  if (!req.pages && n_img == 1) {
    for (const auto& p : pages) {
      if (p.legend.bands >= 2) fail(BC_ERR_NEEDS_USER_CROP, "图里有多组图例，像是多个图案的拼图，请框选其中一个");
    }
  }
  progress.report(0.55f);

  // ---------------------------------------------------------------- classify（逐格采样）
  t = Clock::now();
  double min_pitch = 1e9;
  std::vector<CellSample> samples(static_cast<size_t>(cols) * rows);
  std::vector<CellRef> refs(samples.size());
  std::vector<uint8_t> covered(samples.size(), 0);
  for (auto& p : pages) {
    const cv::Mat& rgb = rgbs[p.image_index];
    const cv::Mat& gray = grays[p.image_index];
    for (int r = 0; r < p.rows(); ++r) {
      progress.sub(0.55f, 0.85f).report(float(r) / p.rows());
      for (int c = 0; c < p.cols(); ++c) {
        const CellRect cr{p.xs[c], p.xs[c + 1], p.ys[r], p.ys[r + 1]};
        min_pitch = std::min({min_pitch, cr.x1 - cr.x0, cr.y1 - cr.y0});
        const size_t k = static_cast<size_t>(p.row_offset + r) * cols + (p.col_offset + c);
        samples[k] = sample_cell(rgb, gray, cr);
        refs[k] = {&gray, cr};
        covered[k] = 1;
      }
    }
  }
  if (min_pitch < 14) warnings.push_back("cell_too_small_for_text");

  // 多页时把各页图例合并（同色只留一个）
  LegendResult legend;
  // 有单独的图例页（只有图例、没有网格）时，只用图例页的图例：网格页边缘被切开的半格很像色块
  const bool has_legend_page = std::any_of(pages.begin(), pages.end(),
                                           [](const PageResult& p) { return p.legend_only && p.legend.items.size() >= 2; });
  for (const auto& p : pages) {
    if (has_legend_page && !p.legend_only) continue;
    for (const auto& it : p.legend.items) {
      bool dup = false;
      for (const auto& e : legend.items) dup |= color::delta_e_2000(e.lab, it.lab) < 3.0;
      if (!dup) legend.items.push_back(it);
    }
  }
  legend.layout = pages.front().legend.layout;
  Assignment asg = assign_cells(samples, legend, pages.front().page_bg);
  for (size_t k = 0; k < covered.size(); ++k) {
    if (!covered[k]) {
      asg.cls[k] = -1;
      asg.conf[k] = 0.0f;
    }
  }
  timing["classify"] = std::lround(ms_since(t));
  progress.report(0.85f);

  // ---------------------------------------------------------------- 多图合集判断：无图例且有多个分散的图案
  // 只针对没有图例的大网格（像 0_0 那种把很多小图摆在一张网格上的封面）。
  // 带色号或带图例的图纸里有几个分开的小图案是正常的，不能要求用户框选。
  if (!req.pages && n_img == 1 && legend.items.empty()) {
    int beads = 0, texted = 0;
    for (size_t k = 0; k < asg.cls.size(); ++k) {
      if (asg.cls[k] < 0) continue;
      ++beads;
      texted += samples[k].has_text;
    }
    const auto objs = object_components(asg.cls, cols, rows, std::max(6, beads / 40));
    // 「格内没有色号」或「没有图例却有 40 种以上颜色」（单张图纸通常 10–40 色）
    const bool cover_like = texted * 5 < beads || asg.classes.size() > 40;
    if (objs.size() >= 4 && cover_like && cols * rows >= 900) {
      fail(BC_ERR_NEEDS_USER_CROP, "图里有多个独立的图案，请框选其中一个");
    }
  }

  // ---------------------------------------------------------------- solver：色号映射、置信度、不确定格
  t = Clock::now();
  // 品牌（B8）：图例读出了色号时，按「哪套色卡包含的色号最多」判断（并列时优先提示的色卡，再优先色数少、
  // 更具体的那套）；读不出色号时，有提示用提示，否则按颜色最接近的色卡猜（置信度封顶 0.5）
  const Palette* pal = nullptr;
  double pal_conf = 0.0;
  std::vector<std::string> legend_codes;
  for (const auto& it : legend.items) {
    if (it.code) legend_codes.push_back(*it.code);
  }
  if (!legend_codes.empty() && !ctx.palettes.empty()) {
    int best_hits = 0;
    for (const auto& cand : ctx.palettes.palettes()) {
      std::set<std::string> codes;
      for (const auto& c : cand.colors) codes.insert(normalize_code(c.code));
      int hits = 0;
      for (const auto& c : legend_codes) hits += codes.count(c) > 0;
      if (hits == 0) continue;
      const bool cand_hinted = req.palette_hint && cand.id == *req.palette_hint;
      const bool cur_hinted = pal && req.palette_hint && pal->id == *req.palette_hint;
      bool better = hits > best_hits;
      if (!better && hits == best_hits && pal) {
        better = cand_hinted || (!cur_hinted && cand.colors.size() < pal->colors.size());
      }
      if (better) {
        best_hits = hits;
        pal = &cand;
      }
    }
    if (pal) pal_conf = double(best_hits) / legend_codes.size();
  }
  if (!pal && req.palette_hint) {
    pal = ctx.palettes.find(*req.palette_hint);
    if (!pal) warnings.push_back("palette_hint_not_found");
    pal_conf = pal ? 1.0 : 0.0;
  }
  if (!pal && !asg.classes.empty()) {
    double best = 1e9;
    for (const auto& cand : ctx.palettes.palettes()) {
      if (cand.colors.empty()) continue;
      double sum = 0;
      for (const auto& c : asg.classes) {
        double m = 1e9;
        for (const auto& pc : cand.colors) m = std::min(m, color::delta_e_2000(c.lab, pc.lab));
        sum += m;
      }
      const double mean = sum / asg.classes.size();
      if (mean < best) {
        best = mean;
        pal = &cand;
      }
    }
    if (pal) pal_conf = std::clamp(1.0 - best / 20.0, 0.0, 1.0) * 0.5;  // 只靠颜色猜，置信度封顶 0.5
  }

  ojson codebook = ojson::array();
  std::vector<std::string> used_codes;
  int unknown_seq = 0;
  for (const auto& c : asg.classes) {
    std::string code;
    bool from_ocr = false;
    if (c.legend_index >= 0 && legend.items[c.legend_index].code) {
      // 图例读出的色号：换成色卡里的原写法（A01 / A1 以色卡为准）
      code = *legend.items[c.legend_index].code;
      from_ocr = true;
      if (pal) {
        for (const auto& pc : pal->colors) {
          if (normalize_code(pc.code) == code) {
            code = pc.code;
            break;
          }
        }
      }
    } else if (pal && !pal->colors.empty()) {
      const PaletteColor* best = nullptr;
      double bd = 1e9;
      for (const auto& pc : pal->colors) {
        const double d = color::delta_e_2000(c.lab, pc.lab);
        if (d < bd) {
          bd = d;
          best = &pc;
        }
      }
      code = best->code;
    }
    const bool dup = std::find(used_codes.begin(), used_codes.end(), code) != used_codes.end();
    if (code.empty() || (!from_ocr && dup)) {
      code = "?" + std::to_string(++unknown_seq);  // 色号待定，交给用户确认
    } else if (from_ocr && dup) {
      warnings.push_back("duplicate_legend_code");
    }
    used_codes.push_back(code);
    ojson e;
    e["palette_id"] = pal ? pal->id : "";
    e["palette_version"] = pal ? pal->version : "";
    e["code"] = code;
    e["legend_rgb"] = {c.rgb[0], c.rgb[1], c.rgb[2]};
    if (c.legend_index < 0) e["in_legend"] = false;
    if (c.page_colored) e["page_colored"] = true;
    codebook.push_back(e);
  }

  // 格内文字证据：颜色拿不准的格子读格内色号（有 OCR、图例读出了色号时）
  if (ctx.ocr) {
    auto tt = Clock::now();
    std::vector<std::string> class_codes(asg.classes.size());
    for (size_t k = 0; k < asg.classes.size(); ++k) {
      const int li = asg.classes[k].legend_index;
      if (li >= 0 && legend.items[li].code) class_codes[k] = *legend.items[li].code;
    }
    const int changed = apply_text_evidence(asg, samples, refs, class_codes, *ctx.ocr);
    if (changed > 0) warnings.push_back("text_evidence_changed:" + std::to_string(changed));
    timing["cell_text"] = std::lround(ms_since(tt));
  }
  // 图例颗数约束：读出了颗数的图例类，颗数对不上时调整低置信度的格子
  {
    std::vector<int> expected(asg.classes.size(), -1);
    for (size_t k = 0; k < asg.classes.size(); ++k) {
      const int li = asg.classes[k].legend_index;
      if (li >= 0 && legend.items[li].count) expected[k] = *legend.items[li].count;
    }
    const int moved = enforce_counts(asg, samples, expected);
    if (moved > 0) warnings.push_back("legend_count_adjusted:" + std::to_string(moved));
  }
  const auto uncertain = pick_uncertain(asg, 0.85);
  ojson cells = ojson::array(), conf = ojson::array(), unc = ojson::array();
  for (size_t k = 0; k < asg.cls.size(); ++k) {
    cells.push_back(asg.cls[k] + 1);  // 0 = 空格，k ≥ 1 指向 codebook[k-1]
    conf.push_back(round_to(asg.conf[k], 0.001));
  }
  for (int k : uncertain) unc.push_back(k);
  timing["solve"] = std::lround(ms_since(t));

  // ---------------------------------------------------------------- 输出
  ojson out;
  out["engine_version"] = BC_VERSION_STRING;
  out["palette_guess"] = {{"palette_id", pal ? pal->id : ""}, {"confidence", round_to(pal_conf, 0.01)}};
  out["grid"] = {{"cols", cols}, {"rows", rows}};
  ojson pages_j = ojson::array();
  for (const auto& p : pages) {
    if (p.legend_only) continue;
    const cv::Rect bb = lines_bbox(p.xs, p.ys);
    pages_j.push_back({{"image_index", p.image_index},
                       {"bbox", {bb.x, bb.y, bb.x + bb.width, bb.y + bb.height}},
                       {"x_lines", lines_json(p.xs)},
                       {"y_lines", lines_json(p.ys)},
                       {"col_offset", p.col_offset},
                       {"row_offset", p.row_offset},
                       {"cols", p.cols()},
                       {"rows", p.rows()},
                       {"source", p.source}});
  }
  out["pages"] = pages_j;
  out["codebook"] = codebook;
  out["cells"] = cells;
  out["confidence"] = conf;
  out["uncertain"] = unc;
  // 图例与核对（B5）：读出了颗数的图例条目和格子数逐一比对；图例里没有、格子里却出现的颜色按期望 0 计
  {
    ojson lj = ojson::array(), diff = ojson::array();
    bool all_read = !legend.items.empty();
    for (size_t k = 0; k < asg.classes.size(); ++k) {
      const auto& c = asg.classes[k];
      const std::string code = codebook[k]["code"];
      if (c.legend_index >= 0) {
        const auto& it = legend.items[c.legend_index];
        if (!it.code || !it.count) {
          all_read = false;
          continue;
        }
        lj.push_back({{"palette_id", codebook[k]["palette_id"]}, {"code", code}, {"count", *it.count}});
        if (*it.count != c.cells) diff.push_back({{"code", code}, {"expected", *it.count}, {"actual", c.cells}});
      } else if (c.cells > 0) {
        diff.push_back({{"code", code}, {"expected", 0}, {"actual", c.cells}});
      }
    }
    out["legend"] = lj;
    if (all_read) {
      out["legend_check"] = {{"ok", diff.empty()}, {"diff", diff}};
    } else {
      out["legend_check"] = {{"ok", false}, {"diff", diff}, {"unavailable", true}};
    }
  }
  out["warnings"] = warnings;
  timing["total"] = std::lround(ms_since(t_all));
  out["timing_ms"] = timing;
  if (g_debug) {
    ojson lg = ojson::array();
    for (const auto& it : legend.items) {
      ojson e = {{"swatch", {it.swatch.x, it.swatch.y, it.swatch.width, it.swatch.height}},
                 {"rgb", {it.rgb[0], it.rgb[1], it.rgb[2]}}};
      if (it.text_only) e["text_only"] = true;
      if (it.code) e["code"] = *it.code;
      if (it.count) e["count"] = *it.count;
      if (it.code_box) e["code_box"] = {it.code_box->x, it.code_box->y, it.code_box->width, it.code_box->height};
      if (it.count_box) e["count_box"] = {it.count_box->x, it.count_box->y, it.count_box->width, it.count_box->height};
      lg.push_back(e);
    }
    debug["legend"] = lg;
    ojson dw = ojson::array();
    for (const auto& r : pages.front().legend.debug_words) dw.push_back({r.x, r.y, r.width, r.height});
    debug["legend_words"] = dw;
    ojson dr = ojson::array();
    for (const auto& r : pages.front().legend.debug_regions) dr.push_back({r.x, r.y, r.width, r.height});
    debug["legend_regions"] = dr;
    debug["exclude"] = {pages.front().exclude.x, pages.front().exclude.y, pages.front().exclude.width, pages.front().exclude.height};
    debug["legend_layout"] = legend.layout;
    debug["page_bg"] = {pages.front().page_bg[0], pages.front().page_bg[1], pages.front().page_bg[2]};
    ojson txt = ojson::array();
    for (const auto& s : samples) txt.push_back(s.has_text ? 1 : 0);
    debug["has_text"] = txt;
    out["debug"] = debug;
  }
  progress.report(1.0f);
  return out;
}

ojson generate(const Context& ctx, const cv::Mat& img, const nlohmann::json& params, const Progress& progress) {
  progress.report(0.0f);
  fail(BC_ERR_INTERNAL, "照片转图纸尚未实现（T1.8）");
}

}  // namespace bc
