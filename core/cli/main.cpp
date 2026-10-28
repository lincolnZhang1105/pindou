// beadcore-cli：benchmark 和真机测速用的命令行工具（03 §2、T0.14）。
// 输出一律是 JSON（stdout）；错误同样是 {"error": {...}}，退出码非 0。
//
//   beadcore-cli version
//   beadcore-cli recognize [选项] IMG [IMG...]
//   beadcore-cli recognize-grid --pages PAGES.json [选项] IMG [IMG...]
//   beadcore-cli generate --params PARAMS.json [选项] IMG
//   beadcore-cli decode IMG                 # 只解码，输出尺寸和格式（核对分辨率用）
//
// 选项：
//   --palettes FILE   palettes_json（默认 $BC_PALETTES，其次 <repo>/palettes/build/palettes.json）
//   --models DIR      模型目录（默认不加载）
//   --hint ID         色卡提示，例如 MARD-291
//   --pretty          格式化输出
//   --repeat N        重复识别 N 次，输出里附带每次耗时（测速用）
//   --debug           结果里附带 debug 字段（网格候选线、图例色块等）
//   --render PREFIX   另存两张调试图：PREFIX.overlay.png（网格线叠在原图上）、PREFIX.pattern.png（识别结果重绘）
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "beadcore/beadcore.h"
#include "engine/engine.h"
#include "image_io.h"

#ifndef BC_REPO_ROOT
#define BC_REPO_ROOT "."
#endif

namespace {

using ojson = nlohmann::ordered_json;

int emit_error(const std::string& code, const std::string& message) {
  ojson j;
  j["error"]["code"] = code;
  j["error"]["message"] = message;
  std::cout << j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << std::endl;
  return 1;
}

bool read_text(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return f.good();
}

struct Args {
  std::string command;
  std::string palettes, models, hint, pages, params;
  bool pretty = false;
  bool debug = false;
  std::string render;
  int repeat = 1;
  std::vector<std::string> images;
};

bool parse_args(int argc, char** argv, Args& a, std::string& err) {
  if (argc < 2) {
    err = "缺少子命令（version / recognize / recognize-grid / generate / decode）";
    return false;
  }
  a.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) {
        err = s + " 缺少参数";
        return false;
      }
      dst = argv[++i];
      return true;
    };
    if (s == "--palettes") {
      if (!next(a.palettes)) return false;
    } else if (s == "--models") {
      if (!next(a.models)) return false;
    } else if (s == "--hint") {
      if (!next(a.hint)) return false;
    } else if (s == "--pages") {
      if (!next(a.pages)) return false;
    } else if (s == "--params") {
      if (!next(a.params)) return false;
    } else if (s == "--pretty") {
      a.pretty = true;
    } else if (s == "--debug") {
      a.debug = true;
    } else if (s == "--render") {
      if (!next(a.render)) return false;
    } else if (s == "--repeat") {
      std::string v;
      if (!next(v)) return false;
      a.repeat = std::max(1, std::atoi(v.c_str()));
    } else if (!s.empty() && s[0] == '-') {
      err = "未知选项：" + s;
      return false;
    } else {
      a.images.push_back(s);
    }
  }
  return true;
}

std::string default_palettes_path() {
  if (const char* env = std::getenv("BC_PALETTES")) return env;
  return std::string(BC_REPO_ROOT) + "/palettes/build/palettes.json";
}

void put(std::vector<uint8_t>& img, int w, int h, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
  if (x < 0 || y < 0 || x >= w || y >= h) return;
  uint8_t* p = &img[(size_t(y) * w + x) * 4];
  p[0] = r;
  p[1] = g;
  p[2] = b;
  p[3] = 255;
}

// 调试图：原图叠网格线（红）和不确定格（黄框）；另画一张识别结果的重绘图
void render_debug(const std::string& prefix, const std::vector<bc::cli::DecodedImage>& decoded, const ojson& j) {
  const int cols = j["grid"]["cols"], rows = j["grid"]["rows"];
  std::vector<int> unc;
  for (const auto& u : j["uncertain"]) unc.push_back(u.get<int>());
  for (const auto& p : j["pages"]) {
    const int idx = p["image_index"];
    auto img = decoded[idx].rgba;
    const int w = decoded[idx].width, h = decoded[idx].height;
    std::vector<double> xs = p["x_lines"].get<std::vector<double>>(), ys = p["y_lines"].get<std::vector<double>>();
    for (double x : xs) {
      for (int y = int(ys.front()); y <= int(ys.back()); ++y) put(img, w, h, int(std::lround(x)), y, 255, 0, 0);
    }
    for (double y : ys) {
      for (int x = int(xs.front()); x <= int(xs.back()); ++x) put(img, w, h, x, int(std::lround(y)), 255, 0, 0);
    }
    const int co = p["col_offset"], ro = p["row_offset"];
    for (int k : unc) {
      const int c = k % cols - co, r = k / cols - ro;
      if (c < 0 || r < 0 || c >= int(xs.size()) - 1 || r >= int(ys.size()) - 1) continue;
      for (int t = 1; t <= 2; ++t) {
        for (int x = int(xs[c]) + t; x <= int(xs[c + 1]) - t; ++x) {
          put(img, w, h, x, int(ys[r]) + t, 255, 200, 0);
          put(img, w, h, x, int(ys[r + 1]) - t, 255, 200, 0);
        }
        for (int y = int(ys[r]) + t; y <= int(ys[r + 1]) - t; ++y) {
          put(img, w, h, int(xs[c]) + t, y, 255, 200, 0);
          put(img, w, h, int(xs[c + 1]) - t, y, 255, 200, 0);
        }
      }
    }
    // 调试模式下画出图例：色块（蓝）、色号框（绿）、颗数框（品红）
    if (j.contains("debug") && j["debug"].contains("legend") && idx == 0) {
      auto box = [&](const ojson& b, uint8_t R, uint8_t G, uint8_t B) {
        const int x0 = b[0], y0 = b[1], x1 = x0 + b[2].get<int>(), y1 = y0 + b[3].get<int>();
        for (int x = x0; x <= x1; ++x) {
          put(img, w, h, x, y0, R, G, B);
          put(img, w, h, x, y1, R, G, B);
        }
        for (int y = y0; y <= y1; ++y) {
          put(img, w, h, x0, y, R, G, B);
          put(img, w, h, x1, y, R, G, B);
        }
      };
      for (const auto& it : j["debug"]["legend"]) {
        box(it["swatch"], 0, 90, 255);
        if (it.contains("code_box")) box(it["code_box"], 0, 200, 0);
        if (it.contains("count_box")) box(it["count_box"], 230, 0, 200);
      }
    }
    bc::cli::save_png(prefix + ".overlay" + std::to_string(idx) + ".png", w, h, 4, img.data(), w * 4);
  }
  const int cs = 12;
  const int W = cols * cs, H = rows * cs;
  std::vector<uint8_t> pat(size_t(W) * H * 4, 255);
  const auto& cells = j["cells"];
  const auto& book = j["codebook"];
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int k = cells[r * cols + c];
      uint8_t R = 255, G = 255, B = 255;
      if (k > 0) {
        const auto& rgb = book[k - 1]["legend_rgb"];
        R = rgb[0];
        G = rgb[1];
        B = rgb[2];
      }
      for (int y = 0; y < cs; ++y) {
        for (int x = 0; x < cs; ++x) {
          const bool edge = x == 0 || y == 0;
          put(pat, W, H, c * cs + x, r * cs + y, edge ? 200 : R, edge ? 200 : G, edge ? 200 : B);
        }
      }
      if (k == 0) {
        put(pat, W, H, c * cs + cs / 2, r * cs + cs / 2, 230, 230, 230);
      }
    }
  }
  bc::cli::save_png(prefix + ".pattern.png", W, H, 4, pat.data(), W * 4);
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  std::string err;
  if (!parse_args(argc, argv, a, err)) return emit_error(BC_ERR_INVALID_INPUT, err);

  if (a.command == "version") {
    ojson j;
    j["engine_version"] = bc_version();
    std::cout << j.dump() << std::endl;
    return 0;
  }

  if (a.images.empty()) return emit_error(BC_ERR_INVALID_INPUT, "没有输入图片");

  std::vector<bc::cli::DecodedImage> decoded(a.images.size());
  for (size_t i = 0; i < a.images.size(); ++i) {
    if (!bc::cli::load_image(a.images[i], decoded[i], err)) return emit_error(BC_ERR_INVALID_INPUT, err);
  }

  if (a.command == "decode") {
    ojson arr = ojson::array();
    for (size_t i = 0; i < decoded.size(); ++i) {
      arr.push_back({{"file", a.images[i]},
                     {"width", decoded[i].width},
                     {"height", decoded[i].height},
                     {"format", decoded[i].format},
                     {"exif_orientation", decoded[i].exif_orientation}});
    }
    std::cout << arr.dump(a.pretty ? 2 : -1) << std::endl;
    return 0;
  }

  if (a.debug) bc::set_debug(true);

  // 色卡
  std::string palettes_json = R"({"palettes":[]})";
  const std::string pal_path = a.palettes.empty() ? default_palettes_path() : a.palettes;
  if (!a.palettes.empty() || file_exists(pal_path)) {
    if (!read_text(pal_path, palettes_json)) return emit_error(BC_ERR_INVALID_INPUT, "无法读取色卡文件：" + pal_path);
  }
  char* cerr = nullptr;
  bc_ctx* ctx = bc_create(a.models.empty() ? nullptr : a.models.c_str(), palettes_json.c_str(), &cerr);
  if (!ctx) {
    std::cout << cerr << std::endl;
    bc_free(cerr);
    return 1;
  }

  std::vector<bc_image> imgs;
  for (auto& d : decoded) imgs.push_back({d.rgba.data(), d.width, d.height, d.width * 4});
  const char* hint = a.hint.empty() ? nullptr : a.hint.c_str();

  std::string text;
  if (a.command == "recognize-grid" && !read_text(a.pages, text)) {
    bc_destroy(ctx);
    return emit_error(BC_ERR_INVALID_INPUT, "无法读取 pages 文件：" + a.pages);
  }
  if (a.command == "generate" && !read_text(a.params, text)) {
    bc_destroy(ctx);
    return emit_error(BC_ERR_INVALID_INPUT, "无法读取 params 文件：" + a.params);
  }

  std::vector<double> times_ms;
  char* out = nullptr;
  for (int r = 0; r < a.repeat; ++r) {
    if (out) bc_free(out);
    const auto t0 = std::chrono::steady_clock::now();
    if (a.command == "recognize") {
      out = bc_recognize(ctx, imgs.data(), static_cast<int>(imgs.size()), hint, nullptr, nullptr);
    } else if (a.command == "recognize-grid") {
      out = bc_recognize_with_grid(ctx, imgs.data(), static_cast<int>(imgs.size()), text.c_str(), hint, nullptr,
                                   nullptr);
    } else if (a.command == "generate") {
      out = bc_generate(ctx, imgs.data(), text.c_str(), nullptr, nullptr);
    } else {
      bc_destroy(ctx);
      return emit_error(BC_ERR_INVALID_INPUT, "未知子命令：" + a.command);
    }
    times_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  bc_destroy(ctx);

  ojson j = ojson::parse(out, nullptr, false);
  bc_free(out);
  if (j.is_discarded()) return emit_error(BC_ERR_INTERNAL, "引擎输出不是合法 JSON");
  if (a.repeat > 1 || std::getenv("BC_CLI_WALL_TIME")) j["cli_wall_ms"] = times_ms;
  if (!a.render.empty() && !j.contains("error")) render_debug(a.render, decoded, j);
  std::cout << j.dump(a.pretty ? 2 : -1, ' ', false, nlohmann::json::error_handler_t::replace) << std::endl;
  return j.contains("error") ? 1 : 0;
}
