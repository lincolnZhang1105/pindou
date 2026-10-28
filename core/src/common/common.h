// beadcore 内部公共设施：错误、进度与取消、图片视图。
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "beadcore/beadcore.h"

namespace bc {

// 内部统一用异常传递错误，在 C ABI 边界转换成 {"error": {...}}（03 §3.1）
class Error : public std::runtime_error {
 public:
  Error(std::string code, const std::string& message) : std::runtime_error(message), code_(std::move(code)) {}
  const std::string& code() const { return code_; }

 private:
  std::string code_;
};

[[noreturn]] inline void fail(const char* code, const std::string& message) { throw Error(code, message); }

inline void require(bool cond, const char* code, const std::string& message) {
  if (!cond) fail(code, message);
}

// 进度回调的分段包装：子步骤把自己的 0–1 映射到父区间 [lo, hi]。
// 回调返回非 0 时抛出 cancelled，由 C ABI 边界转换成错误 JSON。
class Progress {
 public:
  Progress() = default;
  Progress(bc_progress_cb cb, void* user) : cb_(cb), user_(user) {}

  void report(float p) const {
    if (!cb_) return;
    last_ = std::min(1.0f, std::max(0.0f, p));
    const float v = lo_ + (hi_ - lo_) * last_;
    if (cb_(v, user_) != 0) fail(BC_ERR_CANCELLED, "用户取消");
  }
  // 只检查是否取消，不推进进度
  void check() const { report(last_); }

  Progress sub(float a, float b) const {
    Progress s = *this;
    s.lo_ = lo_ + (hi_ - lo_) * a;
    s.hi_ = lo_ + (hi_ - lo_) * b;
    s.last_ = 0.0f;
    return s;
  }

 private:
  bc_progress_cb cb_ = nullptr;
  void* user_ = nullptr;
  float lo_ = 0.0f, hi_ = 1.0f;
  mutable float last_ = 0.0f;
};

// 把调用方传进来的 RGBA 包成 cv::Mat（不拷贝）。通道顺序保持 RGBA。
inline cv::Mat wrap_rgba(const bc_image& img) {
  require(img.rgba != nullptr, BC_ERR_INVALID_INPUT, "图片数据为空");
  require(img.width > 0 && img.height > 0, BC_ERR_INVALID_INPUT, "图片尺寸无效");
  require(img.stride >= img.width * 4, BC_ERR_INVALID_INPUT, "stride 小于 width*4");
  return cv::Mat(img.height, img.width, CV_8UC4, const_cast<uint8_t*>(img.rgba), static_cast<size_t>(img.stride));
}

}  // namespace bc
