#include "palette/palette.h"

#include <cctype>

#include <nlohmann/json.hpp>

#include "common/common.h"

namespace bc {

std::string normalize_code(const std::string& code) {
  std::string s;
  for (char c : code) {
    if (!std::isspace(static_cast<unsigned char>(c))) s.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  // 字母前缀 + 数字：去掉数字的前导 0
  size_t i = 0;
  while (i < s.size() && std::isalpha(static_cast<unsigned char>(s[i]))) ++i;
  if (i > 0 && i < s.size()) {
    size_t j = i;
    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
    if (j == s.size()) {
      size_t k = i;
      while (k + 1 < s.size() && s[k] == '0') ++k;
      s = s.substr(0, i) + s.substr(k);
    }
  }
  return s;
}

PaletteSet PaletteSet::parse(const std::string& json_text) {
  nlohmann::json root;
  try {
    root = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    fail(BC_ERR_INVALID_INPUT, std::string("palettes_json 不是合法 JSON：") + e.what());
  }
  require(root.is_object() && root.contains("palettes") && root["palettes"].is_array(), BC_ERR_INVALID_INPUT,
          "palettes_json 缺少 palettes 数组");

  PaletteSet set;
  for (const auto& p : root["palettes"]) {
    Palette pal;
    require(p.contains("id") && p["id"].is_string(), BC_ERR_INVALID_INPUT, "色卡缺少 id");
    pal.id = p["id"].get<std::string>();
    pal.version = p.value("version", "");
    pal.source = p.value("source", "");
    pal.license = p.value("license", "");
    require(p.contains("colors") && p["colors"].is_array(), BC_ERR_INVALID_INPUT, "色卡 " + pal.id + " 缺少 colors");
    for (const auto& c : p["colors"]) {
      PaletteColor pc;
      require(c.contains("code") && c["code"].is_string(), BC_ERR_INVALID_INPUT, "色卡 " + pal.id + " 有颜色缺少 code");
      pc.code = c["code"].get<std::string>();
      pc.name = c.value("name", "");
      pc.series = c.value("series", "");
      pc.transparent = c.value("transparent", false);
      require(c.contains("rgb") && c["rgb"].is_array() && c["rgb"].size() == 3, BC_ERR_INVALID_INPUT,
              "色号 " + pc.code + " 的 rgb 格式错误");
      for (int i = 0; i < 3; ++i) {
        const int v = c["rgb"][i].get<int>();
        require(v >= 0 && v <= 255, BC_ERR_INVALID_INPUT, "色号 " + pc.code + " 的 rgb 越界");
        pc.rgb[i] = static_cast<uint8_t>(v);
      }
      // Lab 一律由 rgb 现算，保证和识别时用的公式一致
      pc.lab = color::srgb8_to_lab(pc.rgb);
      pal.by_code.emplace(pc.code, static_cast<int>(pal.colors.size()));
      pal.colors.push_back(std::move(pc));
    }
    set.palettes_.push_back(std::move(pal));
  }
  return set;
}

const Palette* PaletteSet::find(const std::string& id) const {
  for (const auto& p : palettes_) {
    if (p.id == id) return &p;
  }
  return nullptr;
}

}  // namespace bc
