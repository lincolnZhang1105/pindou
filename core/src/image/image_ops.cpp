#include "image/image_ops.h"

#include <algorithm>
#include <map>
#include <vector>

#include "common/common.h"

namespace bc {

cv::Mat rgba_to_rgb(const cv::Mat& rgba) {
  CV_Assert(rgba.type() == CV_8UC4);
  cv::Mat out(rgba.size(), CV_8UC3);
  for (int y = 0; y < rgba.rows; ++y) {
    const uint8_t* s = rgba.ptr<uint8_t>(y);
    uint8_t* d = out.ptr<uint8_t>(y);
    for (int x = 0; x < rgba.cols; ++x, s += 4, d += 3) {
      const int a = s[3];
      if (a == 255) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
      } else {
        // 白底合成：c*a + 255*(255-a)，整数四舍五入
        for (int k = 0; k < 3; ++k) d[k] = static_cast<uint8_t>((s[k] * a + 255 * (255 - a) + 127) / 255);
      }
    }
  }
  return out;
}

cv::Mat rgb_to_gray(const cv::Mat& rgb) {
  CV_Assert(rgb.type() == CV_8UC3);
  cv::Mat g(rgb.size(), CV_8UC1);
  for (int y = 0; y < rgb.rows; ++y) {
    const uint8_t* s = rgb.ptr<uint8_t>(y);
    uint8_t* d = g.ptr<uint8_t>(y);
    for (int x = 0; x < rgb.cols; ++x, s += 3) d[x] = static_cast<uint8_t>((299 * s[0] + 587 * s[1] + 114 * s[2] + 500) / 1000);
  }
  return g;
}

color::RGB8 estimate_page_background(const cv::Mat& rgb) {
  std::map<int, std::array<long long, 4>> buckets;  // key → (sum r, sum g, sum b, n)
  auto add = [&](int x, int y) {
    const auto c = px(rgb, x, y);
    const int key = ((c[0] >> 5) << 6) | ((c[1] >> 5) << 3) | (c[2] >> 5);
    auto& b = buckets[key];
    b[0] += c[0];
    b[1] += c[1];
    b[2] += c[2];
    b[3] += 1;
  };
  const int w = rgb.cols, h = rgb.rows;
  const int band = std::max(1, std::min(w, h) / 100);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < band; ++x) {
      add(x, y);
      add(w - 1 - x, y);
    }
  }
  for (int x = 0; x < w; ++x) {
    for (int y = 0; y < band; ++y) {
      add(x, y);
      add(x, h - 1 - y);
    }
  }
  const auto best = std::max_element(buckets.begin(), buckets.end(),
                                     [](const auto& a, const auto& b) { return a.second[3] < b.second[3]; });
  const auto& s = best->second;
  return {static_cast<uint8_t>(s[0] / s[3]), static_cast<uint8_t>(s[1] / s[3]), static_cast<uint8_t>(s[2] / s[3])};
}

color::RGB8 median_rgb(const cv::Mat& rgb, const cv::Rect& r) {
  const cv::Rect rr = r & cv::Rect(0, 0, rgb.cols, rgb.rows);
  if (rr.area() <= 0) return {255, 255, 255};
  std::array<std::vector<uint8_t>, 3> ch;
  for (auto& v : ch) v.reserve(rr.area());
  for (int y = rr.y; y < rr.y + rr.height; ++y) {
    const uint8_t* s = rgb.ptr<uint8_t>(y) + 3 * rr.x;
    for (int x = 0; x < rr.width; ++x, s += 3) {
      ch[0].push_back(s[0]);
      ch[1].push_back(s[1]);
      ch[2].push_back(s[2]);
    }
  }
  color::RGB8 out{};
  for (int k = 0; k < 3; ++k) {
    auto mid = ch[k].begin() + ch[k].size() / 2;
    std::nth_element(ch[k].begin(), mid, ch[k].end());
    out[k] = *mid;
  }
  return out;
}

}  // namespace bc
