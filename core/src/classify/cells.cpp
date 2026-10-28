#include "classify/cells.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "image/image_ops.h"

namespace bc {
namespace {

struct IRect {
  int x0, y0, x1, y1;  // 半开区间 [x0, x1)
  int w() const { return x1 - x0; }
  int h() const { return y1 - y0; }
};

IRect inner_rect(const cv::Mat& img, const CellRect& r, double margin_frac) {
  const double w = r.x1 - r.x0, h = r.y1 - r.y0;
  const double mx = std::max(1.0, w * margin_frac), my = std::max(1.0, h * margin_frac);
  IRect o{static_cast<int>(std::ceil(r.x0 + mx)), static_cast<int>(std::ceil(r.y0 + my)),
          static_cast<int>(std::floor(r.x1 - mx)), static_cast<int>(std::floor(r.y1 - my))};
  o.x0 = std::clamp(o.x0, 0, img.cols);
  o.x1 = std::clamp(o.x1, o.x0, img.cols);
  o.y0 = std::clamp(o.y0, 0, img.rows);
  o.y1 = std::clamp(o.y1, o.y0, img.rows);
  if (o.w() <= 0 || o.h() <= 0) {
    // 格子太小：退化为中心像素
    const int cx = std::clamp(static_cast<int>((r.x0 + r.x1) / 2), 0, img.cols - 1);
    const int cy = std::clamp(static_cast<int>((r.y0 + r.y1) / 2), 0, img.rows - 1);
    o = {cx, cy, cx + 1, cy + 1};
  }
  return o;
}

IRect center_rect(const IRect& in, double frac) {
  const int cw = std::max(1, static_cast<int>(std::lround(in.w() * frac)));
  const int ch = std::max(1, static_cast<int>(std::lround(in.h() * frac)));
  const int x0 = in.x0 + (in.w() - cw) / 2, y0 = in.y0 + (in.h() - ch) / 2;
  return {x0, y0, x0 + cw, y0 + ch};
}

uint8_t median_u8(std::vector<uint8_t>& v) {
  auto mid = v.begin() + v.size() / 2;
  std::nth_element(v.begin(), mid, v.end());
  return *mid;
}

}  // namespace

CellSample sample_cell(const cv::Mat& rgb, const cv::Mat& gray, const CellRect& r, const SampleParams& p) {
  CellSample s;
  const IRect in = inner_rect(rgb, r, p.margin_frac);
  const IRect c = center_rect(in, p.center_frac);

  // 底色：内部减去中心区域（文字通常居中）。环太窄时用整个内部。
  std::array<std::vector<uint8_t>, 3> ch;
  std::vector<uint8_t> ring_gray;
  const bool use_ring = (in.w() - c.w()) >= 2 && (in.h() - c.h()) >= 2;
  for (int y = in.y0; y < in.y1; ++y) {
    const uint8_t* rp = rgb.ptr<uint8_t>(y);
    const uint8_t* gp = gray.ptr<uint8_t>(y);
    for (int x = in.x0; x < in.x1; ++x) {
      const bool inside_center = x >= c.x0 && x < c.x1 && y >= c.y0 && y < c.y1;
      if (use_ring && inside_center) continue;
      ch[0].push_back(rp[3 * x]);
      ch[1].push_back(rp[3 * x + 1]);
      ch[2].push_back(rp[3 * x + 2]);
      ring_gray.push_back(gp[x]);
    }
  }
  s.bg = {median_u8(ch[0]), median_u8(ch[1]), median_u8(ch[2])};
  s.bg_lab = color::srgb8_to_lab(s.bg);
  const int bg_gray = (299 * s.bg[0] + 587 * s.bg[1] + 114 * s.bg[2] + 500) / 1000;
  {
    std::vector<uint8_t> dev;
    dev.reserve(ring_gray.size());
    for (uint8_t g : ring_gray) dev.push_back(static_cast<uint8_t>(std::min(255, std::abs(int(g) - bg_gray))));
    s.bg_spread = median_u8(dev);
  }

  // 文字：中心区域里和底色亮度差大的像素
  int dark = 0, light = 0, total = 0;
  std::vector<uint8_t> contrast;
  for (int y = c.y0; y < c.y1; ++y) {
    const uint8_t* gp = gray.ptr<uint8_t>(y);
    for (int x = c.x0; x < c.x1; ++x) {
      ++total;
      const int d = int(gp[x]) - bg_gray;
      if (std::abs(d) >= p.text_delta) {
        (d < 0 ? dark : light)++;
        contrast.push_back(static_cast<uint8_t>(std::min(255, std::abs(d))));
      }
    }
  }
  s.text_px = dark + light;
  s.text_frac = total > 0 ? float(s.text_px) / float(total) : 0.0f;
  s.has_text = s.text_px >= p.min_text_px && s.text_frac >= p.min_text_frac;
  if (s.has_text) {
    s.text_polarity = dark >= light ? -1 : +1;
    s.text_contrast = median_u8(contrast);
  }
  return s;
}

cv::Mat text_glyph(const cv::Mat& gray, const CellRect& r, const CellSample& s, int size, const SampleParams& p) {
  const IRect in = inner_rect(gray, r, p.margin_frac);
  const int bg_gray = (299 * s.bg[0] + 587 * s.bg[1] + 114 * s.bg[2] + 500) / 1000;
  cv::Mat m(in.h(), in.w(), CV_32F);
  for (int y = in.y0; y < in.y1; ++y) {
    const uint8_t* gp = gray.ptr<uint8_t>(y);
    float* mp = m.ptr<float>(y - in.y0);
    for (int x = in.x0; x < in.x1; ++x) mp[x - in.x0] = std::abs(int(gp[x]) - bg_gray) >= p.text_delta ? 1.0f : 0.0f;
  }
  cv::Mat out;
  cv::resize(m, out, cv::Size(size, size), 0, 0, cv::INTER_AREA);
  return out;
}

double glyph_similarity(const cv::Mat& a, const cv::Mat& b) {
  cv::Scalar ma, sa, mb, sb;
  cv::meanStdDev(a, ma, sa);
  cv::meanStdDev(b, mb, sb);
  if (sa[0] < 1e-6 || sb[0] < 1e-6) return (sa[0] < 1e-6 && sb[0] < 1e-6) ? 1.0 : 0.0;
  cv::Mat da = a - ma[0], db = b - mb[0];
  return da.dot(db) / (sa[0] * sb[0] * a.total());
}

}  // namespace bc
