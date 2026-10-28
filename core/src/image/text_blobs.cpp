#include "image/text_blobs.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace bc {

// 文字块：和背景亮度差大的像素的连通块，再按行合并成「词」
// line_len > 0 时先去掉长度超过它的横竖直线（表格式图例的格线会把文字连成一整块）
std::vector<cv::Rect> find_words(const cv::Mat& gray, const cv::Rect& area, int bg_gray, int delta,
                                 const std::vector<cv::Rect>& skip, int line_len, double merge_gap) {
  const cv::Rect a = area & cv::Rect(0, 0, gray.cols, gray.rows);
  std::vector<cv::Rect> words;
  if (a.area() <= 0) return words;
  cv::Mat mask(a.size(), CV_8U, cv::Scalar(0));
  for (int y = 0; y < a.height; ++y) {
    const uint8_t* g = gray.ptr<uint8_t>(a.y + y) + a.x;
    uint8_t* m = mask.ptr<uint8_t>(y);
    for (int x = 0; x < a.width; ++x) m[x] = std::abs(int(g[x]) - bg_gray) >= delta ? 255 : 0;
  }
  for (const auto& s : skip) {
    const cv::Rect r = (s - a.tl()) & cv::Rect(0, 0, a.width, a.height);
    if (r.area() > 0) mask(r).setTo(0);
  }
  if (line_len > 0) {
    cv::Mat hl, vl;
    cv::morphologyEx(mask, hl, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(line_len, 1)));
    cv::morphologyEx(mask, vl, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, line_len)));
    cv::Mat lines = hl | vl;
    cv::dilate(lines, lines, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    mask.setTo(0, lines);
  }
  cv::Mat labels, stats, cents;
  const int n = cv::connectedComponentsWithStats(mask, labels, stats, cents, 8, CV_32S);
  std::vector<cv::Rect> boxes;
  for (int i = 1; i < n; ++i) {
    if (stats.at<int>(i, cv::CC_STAT_AREA) < 3) continue;
    boxes.push_back(cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT) + a.x, stats.at<int>(i, cv::CC_STAT_TOP) + a.y,
                             stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT)));
  }
  std::sort(boxes.begin(), boxes.end(), [](const cv::Rect& p, const cv::Rect& q) { return p.x < q.x; });
  // 同一行、间距小于字高 0.6 倍的块合并成一个词
  std::vector<uint8_t> used(boxes.size(), 0);
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (used[i]) continue;
    cv::Rect w = boxes[i];
    used[i] = 1;
    for (bool grew = true; grew;) {
      grew = false;
      for (size_t j = 0; j < boxes.size(); ++j) {
        if (used[j]) continue;
        const cv::Rect& b = boxes[j];
        const int ov = std::min(w.y + w.height, b.y + b.height) - std::max(w.y, b.y);
        const int gap = std::max(b.x - (w.x + w.width), w.x - (b.x + b.width));
        const int h = std::max(w.height, b.height);
        if (ov >= 0.4 * std::min(w.height, b.height) && gap <= merge_gap * h) {
          w |= b;
          used[j] = 1;
          grew = true;
        }
      }
    }
    if (w.height >= 4 && w.width >= 2) words.push_back(w);
  }
  return words;
}

}  // namespace bc
