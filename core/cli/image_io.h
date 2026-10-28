// 桌面 CLI 专用的图片解码。App 里的解码由平台层负责（03 §3.1），这里只是为了让
// benchmark 能直接读文件。行为要和平台层一致：按原分辨率解码、按 EXIF 方向校正、不缩放。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bc::cli {

struct DecodedImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgba;  // 紧密排列，stride = width * 4
  int exif_orientation = 1;   // 1 表示无需旋转；已经应用到像素上
  std::string format;         // png / jpeg / webp / ...
};

// 成功返回 true；失败时 err 写入原因
bool load_image(const std::string& path, DecodedImage& out, std::string& err);

// 调试输出 PNG（RGBA 或 RGB）
bool save_png(const std::string& path, int width, int height, int channels, const uint8_t* data, int stride);

}  // namespace bc::cli
