// 文字块检测：和背景亮度差大的像素的连通块，按行合并成「词」。图例和坐标数字共用。
#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace bc {

// area：搜索范围；bg_gray：背景亮度；delta：判定为文字的亮度差；skip：不搜索的区域；
// line_len > 0 时先去掉长度超过它的横竖直线（表格线、网格线会把文字连成一整块）
// merge_gap：同一行两块之间的间距小于「字高 × merge_gap」就合并成一个词（坐标数字挨得近，要用更小的值）
std::vector<cv::Rect> find_words(const cv::Mat& gray, const cv::Rect& area, int bg_gray, int delta,
                                 const std::vector<cv::Rect>& skip, int line_len = 0, double merge_gap = 0.6);

}  // namespace bc
