// 网格检测与逐格分类的合成图测试：在测试里现画图纸，所以不依赖任何外部数据，CI 上都能跑。
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "beadcore/beadcore.h"
#include "grid/grid.h"
#ifdef BC_HAVE_IMAGEIO
#include "image_io.h"
#endif

namespace {

struct Spec {
  int cols = 20, rows = 15;
  double pitch = 18.0;          // 可以是小数，模拟重采样带来的非整数格距
  int margin = 40;
  bool bold5 = true;            // 每 5 格一条粗线
  bool header_ring = false;     // 0_2 那种青色坐标栏
  bool text = true;             // 格内写色号（用一个深色小块代替文字）
  bool legend = true;
  cv::Scalar line{200, 200, 200};
  unsigned seed = 1;
};

struct Truth {
  std::vector<int> cells;  // -1 空格，否则颜色下标
  std::vector<cv::Scalar> palette;
};

const std::vector<cv::Scalar> kColors = {
    {236, 197, 172}, {0, 0, 0}, {171, 35, 47}, {249, 238, 114}, {98, 114, 52}, {15, 82, 189}, {255, 255, 255},
};

// 画一张图纸（RGB）。颜色 6 是白豆：和背景同色，只靠格内的「字」和空格区分
cv::Mat render(const Spec& s, Truth& t) {
  std::mt19937 rng(s.seed);
  const int ring = s.header_ring ? 1 : 0;
  const int gc = s.cols + 2 * ring, gr = s.rows + 2 * ring;
  const int W = static_cast<int>(std::ceil(s.margin * 2 + gc * s.pitch));
  const int legend_h = s.legend ? 120 : 0;
  const int H = static_cast<int>(std::ceil(s.margin * 2 + gr * s.pitch)) + legend_h;
  cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
  auto X = [&](int i) { return static_cast<int>(std::floor(s.margin + i * s.pitch)); };
  auto Y = [&](int j) { return static_cast<int>(std::floor(s.margin + j * s.pitch)); };

  // 图案：一个椭圆形的「主体」，内部随机分块上色，外面是空格；主体里留一些白豆
  t.palette = kColors;
  t.cells.assign(static_cast<size_t>(s.cols) * s.rows, -1);
  std::uniform_int_distribution<int> pick(0, 5), coin(0, 9);
  for (int r = 0; r < s.rows; ++r) {
    for (int c = 0; c < s.cols; ++c) {
      const double dx = (c + 0.5 - s.cols / 2.0) / (s.cols * 0.45), dy = (r + 0.5 - s.rows / 2.0) / (s.rows * 0.45);
      if (dx * dx + dy * dy > 1.0) continue;
      t.cells[r * s.cols + c] = coin(rng) == 0 ? 6 : pick(rng);
    }
  }
  for (int r = 0; r < gr; ++r) {
    for (int c = 0; c < gc; ++c) {
      const bool hdr = s.header_ring && (r == 0 || c == 0 || r == gr - 1 || c == gc - 1);
      const bool corner = s.header_ring && (r == 0 || r == gr - 1) && (c == 0 || c == gc - 1);
      cv::Rect cell(cv::Point(X(c), Y(r)), cv::Point(X(c + 1), Y(r + 1)));
      if (hdr) {
        if (corner) continue;
        cv::rectangle(img, cell, cv::Scalar(170, 235, 225), cv::FILLED);
        const int num = (r == 0 || r == gr - 1) ? c : r;
        cv::putText(img, std::to_string(num), cv::Point(cell.x + 3, cell.y + cell.height - 6), cv::FONT_HERSHEY_SIMPLEX,
                    s.pitch / 50.0, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        continue;
      }
      const int k = t.cells[(r - ring) * s.cols + (c - ring)];
      if (k < 0) continue;
      cv::rectangle(img, cell, kColors[k], cv::FILLED);
      if (s.text) {
        const cv::Scalar ink = (k == 1 || k == 2 || k == 4 || k == 5) ? cv::Scalar(255, 255, 255) : cv::Scalar(30, 30, 30);
        const int cx = (cell.x + cell.br().x) / 2, cy = (cell.y + cell.br().y) / 2, w = static_cast<int>(s.pitch * 0.18);
        cv::rectangle(img, cv::Rect(cx - w, cy - w / 2, 2 * w, w + 1), ink, cv::FILLED);
      }
    }
  }
  const cv::Scalar line = s.header_ring ? cv::Scalar(0, 0, 0) : s.line;
  for (int i = 0; i <= gc; ++i) {
    const bool bold = s.bold5 && (i - ring) % 5 == 0;
    cv::line(img, {X(i), Y(0)}, {X(i), Y(gr)}, bold ? cv::Scalar(180, 160, 230) : line, bold ? 2 : 1);
  }
  for (int j = 0; j <= gr; ++j) {
    const bool bold = s.bold5 && (j - ring) % 5 == 0;
    cv::line(img, {X(0), Y(j)}, {X(gc), Y(j)}, bold ? cv::Scalar(180, 160, 230) : line, bold ? 2 : 1);
  }
  if (s.legend) {
    for (int k = 0; k < 6; ++k) {
      const cv::Rect sw(s.margin + k * 60, Y(gr) + 30, 48, 48);
      cv::rectangle(img, sw, kColors[k], cv::FILLED);
    }
  }
  return img;
}

nlohmann::json run(const cv::Mat& rgb) {
  cv::Mat rgba;
  cv::cvtColor(rgb, rgba, cv::COLOR_RGB2RGBA);
#ifdef BC_HAVE_IMAGEIO
  // 调试：设置 BC_TEST_DUMP=目录 时，把合成图存下来
  if (const char* dir = std::getenv("BC_TEST_DUMP")) {
    static int n = 0;
    const std::string name = std::string(dir) + "/synth_" + std::to_string(n++) + ".png";
    bc::cli::save_png(name, rgba.cols, rgba.rows, 4, rgba.data, static_cast<int>(rgba.step));
  }
#endif
  bc_ctx* ctx = bc_create(nullptr, R"({"palettes":[]})", nullptr);
  bc_image img{rgba.data, rgba.cols, rgba.rows, static_cast<int>(rgba.step)};
  char* out = bc_recognize(ctx, &img, 1, nullptr, nullptr, nullptr);
  auto j = nlohmann::json::parse(out);
  bc_free(out);
  bc_destroy(ctx);
  return j;
}

// 预测的类别和真值颜色之间按多数映射，返回分错的格子数（空格判断错也算）
int cell_errors(const nlohmann::json& j, const Truth& t) {
  const auto& cells = j["cells"];
  std::map<int, std::map<int, int>> votes;
  for (size_t i = 0; i < t.cells.size(); ++i) votes[cells[i].get<int>()][t.cells[i]]++;
  std::map<int, int> to_truth;
  for (const auto& [p, m] : votes) {
    int best = -2, n = -1;
    for (const auto& [tt, cnt] : m) {
      if (cnt > n) {
        n = cnt;
        best = tt;
      }
    }
    to_truth[p] = best;
  }
  int err = 0;
  for (size_t i = 0; i < t.cells.size(); ++i) {
    const int p = cells[i].get<int>();
    const int mapped = p == 0 ? -1 : to_truth[p];
    if (mapped != t.cells[i]) ++err;
  }
  return err;
}

void expect_exact(const Spec& s) {
  Truth t;
  const cv::Mat img = render(s, t);
  const auto j = run(img);
  ASSERT_FALSE(j.contains("error")) << j.dump();
  EXPECT_EQ(j["grid"]["cols"].get<int>(), s.cols);
  EXPECT_EQ(j["grid"]["rows"].get<int>(), s.rows);
  if (j["grid"]["cols"].get<int>() == s.cols && j["grid"]["rows"].get<int>() == s.rows) {
    EXPECT_EQ(cell_errors(j, t), 0);
  }
}

}  // namespace

TEST(Grid, ThinLinesBold5WithLegend) { expect_exact(Spec{}); }

TEST(Grid, NonIntegerPitchDrift) {
  Spec s;
  s.cols = 41;
  s.rows = 37;
  s.pitch = 17.3;
  expect_exact(s);
}

TEST(Grid, HeaderRingIsStripped) {
  Spec s;
  s.cols = 12;
  s.rows = 10;
  s.pitch = 44;
  s.header_ring = true;
  s.bold5 = false;
  expect_exact(s);
}

// 坐标栏下面还夹着一条「缝」（到图例框之间：一条浅色横线，个别竖线伸下来，零星几格凑够三条边连进了网格）。
// 判断坐标栏时看的最外一行要先去掉这条缝，否则底部坐标栏会留在网格里
TEST(Grid, HeaderRingWithSparseGapBelow) {
  Spec s;
  s.cols = 12;
  s.rows = 10;
  s.pitch = 44;
  s.header_ring = true;
  s.bold5 = false;
  s.legend = false;
  s.margin = 60;
  Truth t;
  cv::Mat img = render(s, t);
  auto X = [&](int i) { return static_cast<int>(std::floor(s.margin + i * s.pitch)); };
  auto Y = [&](int j) { return static_cast<int>(std::floor(s.margin + j * s.pitch)); };
  const int gr = s.rows + 2, gc = s.cols + 2;  // 含坐标栏
  const int y_gap = Y(gr) + static_cast<int>(s.pitch * 0.8);
  cv::line(img, {X(0), y_gap}, {X(gc), y_gap}, cv::Scalar(150, 150, 150), 1);
  for (int i : {3, 4, 9}) cv::line(img, {X(i), Y(gr)}, {X(i), y_gap}, cv::Scalar(0, 0, 0), 1);
  const auto j = run(img);
  ASSERT_FALSE(j.contains("error")) << j.dump();
  EXPECT_EQ(j["grid"]["cols"].get<int>(), s.cols);
  EXPECT_EQ(j["grid"]["rows"].get<int>(), s.rows);
}

TEST(Grid, SmallCellsWithoutText) {
  Spec s;
  s.cols = 64;
  s.rows = 48;
  s.pitch = 9;
  s.text = false;
  s.seed = 7;
  Truth t;
  const auto j = run(render(s, t));
  ASSERT_FALSE(j.contains("error")) << j.dump();
  EXPECT_EQ(j["grid"]["cols"].get<int>(), s.cols);
  EXPECT_EQ(j["grid"]["rows"].get<int>(), s.rows);
}

TEST(Grid, BlankImageHasNoGrid) {
  const cv::Mat img(400, 300, CV_8UC3, cv::Scalar(255, 255, 255));
  const auto j = run(img);
  ASSERT_TRUE(j.contains("error"));
  EXPECT_EQ(j["error"]["code"].get<std::string>(), "no_grid_found");
}

TEST(Grid, PeriodFromSpikeTrain) {
  // 每 13 像素一条细线、每 65 像素一条强线：应该取 13 而不是 65
  std::vector<double> prof(1000, 0.0);
  for (int i = 20; i < 980; i += 13) prof[i] = ((i - 20) % 65 == 0) ? 10.0 : 3.0;
  const double p = bc::estimate_period(prof, {}, 4, 160);
  EXPECT_NEAR(p, 13.0, 0.2);
}

namespace {

// 把整张图纸（RGB）按列切点 cut_cs、行切点 cut_rs 切成若干页（按行优先排），页边多带 pad 像素的相邻半格
std::vector<cv::Mat> cut_pages(const cv::Mat& full, const Spec& s, const std::vector<int>& cut_cs,
                               const std::vector<int>& cut_rs, int pad) {
  auto X = [&](int i) { return static_cast<int>(std::floor(s.margin + i * s.pitch)); };
  auto Y = [&](int j) { return static_cast<int>(std::floor(s.margin + j * s.pitch)); };
  std::vector<int> xs{0}, ys{0};
  for (int c : cut_cs) xs.push_back(X(c));
  for (int r : cut_rs) ys.push_back(Y(r));
  xs.push_back(full.cols);
  ys.push_back(full.rows);
  std::vector<cv::Mat> pages;
  for (size_t pr = 0; pr + 1 < ys.size(); ++pr) {
    for (size_t pc = 0; pc + 1 < xs.size(); ++pc) {
      const int x0 = pc == 0 ? 0 : xs[pc] - pad, x1 = pc + 2 == xs.size() ? full.cols : xs[pc + 1] + pad + 1;
      const int y0 = pr == 0 ? 0 : ys[pr] - pad, y1 = pr + 2 == ys.size() ? full.rows : ys[pr + 1] + pad + 1;
      pages.push_back(full(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone());
    }
  }
  return pages;
}

nlohmann::json run_pages(const std::vector<cv::Mat>& rgb_pages) {
  std::vector<cv::Mat> rgba(rgb_pages.size());
  std::vector<bc_image> imgs;
  for (size_t k = 0; k < rgb_pages.size(); ++k) {
    cv::cvtColor(rgb_pages[k], rgba[k], cv::COLOR_RGB2RGBA);
    imgs.push_back({rgba[k].data, rgba[k].cols, rgba[k].rows, static_cast<int>(rgba[k].step)});
  }
  bc_ctx* ctx = bc_create(nullptr, R"({"palettes":[]})", nullptr);
  char* out = bc_recognize(ctx, imgs.data(), static_cast<int>(imgs.size()), nullptr, nullptr, nullptr);
  auto j = nlohmann::json::parse(out);
  bc_free(out);
  bc_destroy(ctx);
  return j;
}

void expect_pages_exact(const std::vector<cv::Mat>& pages, const Spec& s, const Truth& t) {
  const auto j = run_pages(pages);
  ASSERT_FALSE(j.contains("error")) << j.dump();
  EXPECT_EQ(j["grid"]["cols"].get<int>(), s.cols);
  EXPECT_EQ(j["grid"]["rows"].get<int>(), s.rows);
  if (j["grid"]["cols"].get<int>() == s.cols && j["grid"]["rows"].get<int>() == s.rows) {
    EXPECT_EQ(cell_errors(j, t), 0);
  }
}

Spec paged_spec(int cols, int rows) {
  Spec s;
  s.cols = cols;
  s.rows = rows;
  s.pitch = 18;
  s.legend = false;
  return s;
}

}  // namespace

// 分页图（B9）：一张图纸按 2×2 切成四页（页边带着被切开的半格），没有坐标时按九宫格版式拼回去
TEST(Grid, PagedFourPiecesFallbackLayout) {
  const Spec s = paged_spec(24, 20);
  Truth t;
  const cv::Mat full = render(s, t);
  expect_pages_exact(cut_pages(full, s, {12}, {10}, 7), s, t);
}

// 接缝一侧没有留白：最后一条网格线离图片边缘只有两个像素（线响应只剩一半），每页都不能少一行或一列
TEST(Grid, PagedSeamLineAtImageEdge) {
  const Spec s = paged_spec(24, 20);
  Truth t;
  const cv::Mat full = render(s, t);
  expect_pages_exact(cut_pages(full, s, {12}, {10}, 2), s, t);
}

// 2 列 × 3 行，各列宽度只差一格、各行高度不同、没有坐标：靠「同一行行数相同、同一列列数相同」排对
//（以前按 ⌈√6⌉ = 3 张一行排）
TEST(Grid, PagedSixPiecesLayoutFromPageShapes) {
  const Spec s = paged_spec(23, 31);
  Truth t;
  const cv::Mat full = render(s, t);
  expect_pages_exact(cut_pages(full, s, {12}, {10, 21}, 7), s, t);
}

// 图例单独成一页、而且是带格线的表格（每行「色块 | 色号 | 颗数」）：会被检测成一个格距很大的小网格，
// 不能当成网格页参与拼接
TEST(Grid, PagedLegendOnlyImageIsNotAGrid) {
  const Spec s = paged_spec(24, 20);
  Truth t;
  const cv::Mat full = render(s, t);
  auto pages = cut_pages(full, s, {12}, {10}, 7);
  const int cell = 50, tc = 6, tr = 5, m = 30;
  cv::Mat legend(2 * m + tr * cell, 2 * m + tc * cell, CV_8UC3, cv::Scalar(255, 255, 255));
  for (int r = 0; r < tr; ++r) {
    for (int c = 0; c < tc; ++c) {
      const cv::Rect box(m + c * cell, m + r * cell, cell, cell);
      const int item = r * 2 + c / 3;
      if (c % 3 == 0) {
        cv::rectangle(legend, box, kColors[item % 6], cv::FILLED);
      } else {
        const std::string txt = c % 3 == 1 ? "A" + std::to_string(item + 1) : "x" + std::to_string(10 + item);
        cv::putText(legend, txt, cv::Point(box.x + 6, box.y + 32), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(30, 30, 30), 1,
                    cv::LINE_AA);
      }
    }
  }
  for (int c = 0; c <= tc; ++c) cv::line(legend, {m + c * cell, m}, {m + c * cell, m + tr * cell}, cv::Scalar(120, 120, 120), 1);
  for (int r = 0; r <= tr; ++r) cv::line(legend, {m, m + r * cell}, {m + tc * cell, m + r * cell}, cv::Scalar(120, 120, 120), 1);
  pages.push_back(legend);
  expect_pages_exact(pages, s, t);
}
