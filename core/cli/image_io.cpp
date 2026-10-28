#include "image_io.h"

#include <cstring>
#include <fstream>
#include <iterator>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "webp/decode.h"

namespace bc::cli {
namespace {

bool read_file(const std::string& path, std::vector<uint8_t>& buf) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  buf.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

uint16_t rd16(const uint8_t* p, bool le) { return le ? uint16_t(p[0] | (p[1] << 8)) : uint16_t((p[0] << 8) | p[1]); }
uint32_t rd32(const uint8_t* p, bool le) {
  return le ? uint32_t(p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t(p[3]) << 24))
            : uint32_t((uint32_t(p[0]) << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
}

// 从 TIFF 结构（Exif 载荷）里读 Orientation（0x0112）
int orientation_from_tiff(const uint8_t* t, size_t n) {
  if (n < 8) return 1;
  bool le;
  if (t[0] == 'I' && t[1] == 'I') {
    le = true;
  } else if (t[0] == 'M' && t[1] == 'M') {
    le = false;
  } else {
    return 1;
  }
  const uint32_t ifd = rd32(t + 4, le);
  if (ifd + 2 > n) return 1;
  const uint16_t count = rd16(t + ifd, le);
  for (uint16_t i = 0; i < count; ++i) {
    const size_t e = ifd + 2 + size_t(i) * 12;
    if (e + 12 > n) break;
    if (rd16(t + e, le) == 0x0112) {
      const int v = rd16(t + e + 8, le);
      return (v >= 1 && v <= 8) ? v : 1;
    }
  }
  return 1;
}

int jpeg_orientation(const std::vector<uint8_t>& b) {
  if (b.size() < 4 || b[0] != 0xFF || b[1] != 0xD8) return 1;
  size_t i = 2;
  while (i + 4 <= b.size()) {
    if (b[i] != 0xFF) return 1;
    const uint8_t marker = b[i + 1];
    if (marker == 0xDA || marker == 0xD9) break;  // 图像数据开始
    const size_t len = (size_t(b[i + 2]) << 8) | b[i + 3];
    if (marker == 0xE1 && len >= 8 && i + 2 + len <= b.size() && std::memcmp(&b[i + 4], "Exif\0\0", 6) == 0) {
      return orientation_from_tiff(&b[i + 10], len - 8);
    }
    i += 2 + len;
  }
  return 1;
}

// 按 EXIF Orientation 把像素转成正向（和 iOS ImageIO / Android ExifInterface 的语义一致）
void apply_orientation(DecodedImage& img, int o) {
  if (o <= 1 || o > 8) return;
  const int w = img.width, h = img.height;
  const bool swap = o >= 5;
  const int nw = swap ? h : w, nh = swap ? w : h;
  std::vector<uint8_t> out(size_t(nw) * nh * 4);
  for (int y = 0; y < nh; ++y) {
    for (int x = 0; x < nw; ++x) {
      int sx = 0, sy = 0;
      switch (o) {
        case 2: sx = w - 1 - x; sy = y; break;
        case 3: sx = w - 1 - x; sy = h - 1 - y; break;
        case 4: sx = x; sy = h - 1 - y; break;
        case 5: sx = y; sy = x; break;
        case 6: sx = y; sy = h - 1 - x; break;
        case 7: sx = w - 1 - y; sy = h - 1 - x; break;
        case 8: sx = w - 1 - y; sy = x; break;
        default: break;
      }
      std::memcpy(&out[(size_t(y) * nw + x) * 4], &img.rgba[(size_t(sy) * w + sx) * 4], 4);
    }
  }
  img.rgba.swap(out);
  img.width = nw;
  img.height = nh;
}

bool is_webp(const std::vector<uint8_t>& b) {
  return b.size() >= 12 && std::memcmp(b.data(), "RIFF", 4) == 0 && std::memcmp(b.data() + 8, "WEBP", 4) == 0;
}

bool is_heif(const std::vector<uint8_t>& b) {
  if (b.size() < 12 || std::memcmp(b.data() + 4, "ftyp", 4) != 0) return false;
  const std::string brand(reinterpret_cast<const char*>(b.data() + 8), 4);
  return brand == "heic" || brand == "heix" || brand == "mif1" || brand == "msf1" || brand == "heim" ||
         brand == "heis" || brand == "avif";
}

}  // namespace

bool load_image(const std::string& path, DecodedImage& out, std::string& err) {
  std::vector<uint8_t> buf;
  if (!read_file(path, buf)) {
    err = "无法读取文件：" + path;
    return false;
  }
  if (is_heif(buf)) {
    err = "HEIC/HEIF/AVIF 需要先在 bench 里转换成 PNG（CLI 不引入 libheif，见 05 §4.1）：" + path;
    return false;
  }
  if (is_webp(buf)) {
    int w = 0, h = 0;
    uint8_t* px = WebPDecodeRGBA(buf.data(), buf.size(), &w, &h);
    if (!px) {
      err = "WebP 解码失败：" + path;
      return false;
    }
    out.width = w;
    out.height = h;
    out.rgba.assign(px, px + size_t(w) * h * 4);
    WebPFree(px);
    out.format = "webp";
    return true;
  }
  int w = 0, h = 0, n = 0;
  uint8_t* px = stbi_load_from_memory(buf.data(), static_cast<int>(buf.size()), &w, &h, &n, 4);
  if (!px) {
    err = std::string("解码失败（") + stbi_failure_reason() + "）：" + path;
    return false;
  }
  out.width = w;
  out.height = h;
  out.rgba.assign(px, px + size_t(w) * h * 4);
  stbi_image_free(px);
  out.format = (buf.size() > 2 && buf[0] == 0xFF && buf[1] == 0xD8) ? "jpeg" : (buf.size() > 4 && buf[1] == 'P' && buf[2] == 'N' && buf[3] == 'G') ? "png" : "other";
  if (out.format == "jpeg") {
    out.exif_orientation = jpeg_orientation(buf);
    apply_orientation(out, out.exif_orientation);
  }
  return true;
}

bool save_png(const std::string& path, int width, int height, int channels, const uint8_t* data, int stride) {
  return stbi_write_png(path.c_str(), width, height, channels, data, stride) != 0;
}

}  // namespace bc::cli
