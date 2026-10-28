// C ABI 约定测试（03 §3.1）：不返回 NULL、错误 JSON 格式、取消、色卡解析。
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "beadcore/beadcore.h"
#include "palette/palette.h"

namespace {

const char* kTinyPalettes = R"({"palettes":[{"id":"TEST-3","version":"x","source":"test","license":"MIT",
  "colors":[{"code":"A1","name":"","rgb":[255,0,0],"lab":[0,0,0],"series":"A"},
            {"code":"A2","name":"","rgb":[0,255,0],"lab":[0,0,0],"series":"A"},
            {"code":"B01","name":"","rgb":[0,0,255],"lab":[0,0,0],"series":"B"}]}]})";

nlohmann::json take(char* s) {
  EXPECT_NE(s, nullptr);
  auto j = nlohmann::json::parse(s);
  bc_free(s);
  return j;
}

std::string err_code(const nlohmann::json& j) {
  return j.contains("error") ? j["error"]["code"].get<std::string>() : std::string();
}

}  // namespace

TEST(Api, VersionIsNonEmpty) { EXPECT_GT(std::string(bc_version()).size(), 0u); }

TEST(Api, CreateWithBadJsonReturnsNullAndError) {
  char* err = nullptr;
  bc_ctx* ctx = bc_create(nullptr, "{not json", &err);
  EXPECT_EQ(ctx, nullptr);
  ASSERT_NE(err, nullptr);
  EXPECT_EQ(err_code(take(err)), "invalid_input");

  ctx = bc_create(nullptr, nullptr, nullptr);  // err_json 可以传 NULL
  EXPECT_EQ(ctx, nullptr);
}

TEST(Api, NullInputsNeverReturnNull) {
  EXPECT_EQ(err_code(take(bc_recognize(nullptr, nullptr, 0, nullptr, nullptr, nullptr))), "invalid_input");
  bc_ctx* ctx = bc_create(nullptr, kTinyPalettes, nullptr);
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(err_code(take(bc_recognize(ctx, nullptr, 0, nullptr, nullptr, nullptr))), "invalid_input");
  bc_image bad{nullptr, 10, 10, 40};
  EXPECT_EQ(err_code(take(bc_recognize(ctx, &bad, 1, nullptr, nullptr, nullptr))), "invalid_input");
  std::vector<uint8_t> px(10 * 10 * 4, 255);
  bc_image small_stride{px.data(), 10, 10, 8};
  EXPECT_EQ(err_code(take(bc_recognize(ctx, &small_stride, 1, nullptr, nullptr, nullptr))), "invalid_input");
  bc_image ok{px.data(), 10, 10, 40};
  EXPECT_EQ(err_code(take(bc_recognize_with_grid(ctx, &ok, 1, "[oops", nullptr, nullptr, nullptr))), "invalid_input");
  EXPECT_EQ(err_code(take(bc_generate(ctx, &ok, "{", nullptr, nullptr))), "invalid_input");
  bc_destroy(ctx);
}

TEST(Api, ProgressCallbackCanCancel) {
  bc_ctx* ctx = bc_create(nullptr, kTinyPalettes, nullptr);
  ASSERT_NE(ctx, nullptr);
  std::vector<uint8_t> px(64 * 64 * 4, 255);
  bc_image img{px.data(), 64, 64, 64 * 4};
  auto cancel_now = [](float, void*) -> int { return 1; };
  EXPECT_EQ(err_code(take(bc_recognize(ctx, &img, 1, nullptr, cancel_now, nullptr))), "cancelled");
  bc_destroy(ctx);
}

TEST(Api, FreeNullIsNoop) { bc_free(nullptr); }

TEST(Palette, ParseAndLookup) {
  const auto set = bc::PaletteSet::parse(kTinyPalettes);
  ASSERT_EQ(set.palettes().size(), 1u);
  const auto* p = set.find("TEST-3");
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p->colors.size(), 3u);
  ASSERT_NE(p->find("A2"), nullptr);
  // Lab 由 rgb 现算，忽略输入里的 lab 字段
  EXPECT_GT(p->find("A2")->lab.L, 80.0);
  EXPECT_EQ(set.find("NOPE"), nullptr);
}

TEST(Palette, NormalizeCode) {
  EXPECT_EQ(bc::normalize_code("A01"), "A1");
  EXPECT_EQ(bc::normalize_code(" h 07 "), "H7");
  EXPECT_EQ(bc::normalize_code("ZG1"), "ZG1");
  EXPECT_EQ(bc::normalize_code("A10"), "A10");
  EXPECT_EQ(bc::normalize_code("A0"), "A0");
  EXPECT_EQ(bc::normalize_code("UNKNOWN-01"), "UNKNOWN-01");
}
