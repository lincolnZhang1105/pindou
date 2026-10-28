// 色卡数据（palettes_json，03 §3.5）。识别时只用来判断品牌、校验色号和无图例时兜底；
// 格子颜色优先和图纸自带图例的色块比对（04 §5.1）。
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "color/color.h"

namespace bc {

struct PaletteColor {
  std::string code;
  std::string name;
  std::string series;
  color::RGB8 rgb{};
  color::Lab lab;
  bool transparent = false;
};

struct Palette {
  std::string id;
  std::string version;
  std::string source;
  std::string license;
  std::vector<PaletteColor> colors;
  std::unordered_map<std::string, int> by_code;

  const PaletteColor* find(const std::string& code) const {
    auto it = by_code.find(code);
    return it == by_code.end() ? nullptr : &colors[it->second];
  }
};

class PaletteSet {
 public:
  // 解析 palettes_json；格式错误时抛出 invalid_input
  static PaletteSet parse(const std::string& json_text);

  const std::vector<Palette>& palettes() const { return palettes_; }
  const Palette* find(const std::string& id) const;
  bool empty() const { return palettes_.empty(); }

 private:
  std::vector<Palette> palettes_;
};

// 色号规范化：去掉空白、转大写、去掉数字部分的前导 0（A01 → A1），用于跨来源比对
std::string normalize_code(const std::string& code);

}  // namespace bc
