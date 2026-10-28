// 真实样例冒烟测试（03 §3.6）。data/samples/ 是画师的图纸，不进 git，没有这些文件时跳过。
// 断言：0_2 是 16×16、3_0 是 58×53；其余样例不崩溃，或者正确返回 needs_user_crop。
// 另外把已经确认过的结果也锁住（尺寸、每个颜色的颗数和图例一致），防止识别回退。
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "beadcore/beadcore.h"

#ifdef BC_HAVE_IMAGEIO
#include "image_io.h"

namespace {

std::string sample(const std::string& name) { return std::string(BC_REPO_ROOT) + "/data/samples/" + name; }

bool exists(const std::string& p) { return std::ifstream(p).good(); }

std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// with_models：用 core/models 下的 OCR 模型，并加载 bench 的色卡（含 MARD）；两者缺一时跳过
nlohmann::json recognize_file(const std::string& path, bool with_models = false) {
  bc::cli::DecodedImage img;
  std::string err;
  EXPECT_TRUE(bc::cli::load_image(path, img, err)) << err;
  const std::string models = std::string(BC_REPO_ROOT) + "/core/models";
  const std::string pal_path = std::string(BC_REPO_ROOT) + "/bench/palettes_unverified/build/palettes_bench.json";
  const std::string palettes = with_models ? read_file(pal_path) : R"({"palettes":[]})";
  bc_ctx* ctx = bc_create(with_models ? models.c_str() : nullptr, palettes.c_str(), nullptr);
  EXPECT_NE(ctx, nullptr);
  bc_image bi{img.rgba.data(), img.width, img.height, img.width * 4};
  char* out = bc_recognize(ctx, &bi, 1, nullptr, nullptr, nullptr);
  auto j = nlohmann::json::parse(out);
  bc_free(out);
  bc_destroy(ctx);
  return j;
}

// 每个 codebook 条目对应的格子数，按数量从大到小排序（v0 还读不出色号，只比颗数的分布）
std::vector<int> sorted_counts(const nlohmann::json& j) {
  std::map<int, int> cnt;
  for (const auto& c : j["cells"]) {
    if (c.get<int>() > 0) cnt[c.get<int>()]++;
  }
  std::vector<int> v;
  for (const auto& [k, n] : cnt) v.push_back(n);
  std::sort(v.rbegin(), v.rend());
  return v;
}

struct Expect {
  const char* file;
  int cols, rows;
  std::vector<int> legend_counts;  // 图里图例印的颗数，从大到小
};

}  // namespace

class SampleSmoke : public ::testing::TestWithParam<Expect> {};

TEST_P(SampleSmoke, GridAndLegendCounts) {
  const auto& e = GetParam();
  if (!exists(sample(e.file))) GTEST_SKIP() << "没有 data/samples/" << e.file;
  const auto j = recognize_file(sample(e.file));
  ASSERT_FALSE(j.contains("error")) << j.dump();
  EXPECT_EQ(j["grid"]["cols"].get<int>(), e.cols);
  EXPECT_EQ(j["grid"]["rows"].get<int>(), e.rows);
  EXPECT_EQ(sorted_counts(j), e.legend_counts);
}

INSTANTIATE_TEST_SUITE_P(
    RealPatterns, SampleSmoke,
    ::testing::Values(
        // 0_2：D4 71、D1 51、C7 44、C8 13、G6 5、D5 4、D20 2、A10 2
        Expect{"0_2.png", 16, 16, {71, 51, 44, 13, 5, 4, 2, 2}},
        // 0_3：F11 61、G10 52、F10 37、G8 13、A5 5、A14 3、F8 2
        Expect{"0_3.png", 16, 16, {61, 52, 37, 13, 5, 3, 2}},
        // 3_0：H7 718、G3 534、H2 264、A1 102、G6 100、G13 70、B13 57、A15 18、F1 11、F9 5、F14 4、A11 2
        Expect{"3_0.png", 58, 53, {718, 534, 264, 102, 100, 70, 57, 18, 11, 5, 4, 2}},
        // 3_2：H7 551、H2 424、G3 377、F8 219、A4 101、D16 54、F11 49、H4 46、F9 32、F1 30、B11 15
        Expect{"3_2.png", 58, 52, {551, 424, 377, 219, 101, 54, 49, 46, 32, 30, 15}}),
    [](const ::testing::TestParamInfo<Expect>& info) {
      std::string n = info.param.file;
      return "s" + n.substr(0, n.find('.'));
    });

// 有 OCR 模型时：图例色号、颗数全部读对，和格子核对一致，品牌判成 MARD
TEST(SampleSmoke, LegendCodesWithOcr) {
  if (!exists(std::string(BC_REPO_ROOT) + "/core/models/ocr_v1.onnx") ||
      !exists(std::string(BC_REPO_ROOT) + "/bench/palettes_unverified/build/palettes_bench.json")) {
    GTEST_SKIP() << "没有 OCR 模型或 bench 色卡";
  }
  const std::map<std::string, std::vector<std::string>> want = {
      {"0_2.png", {"D4", "D20", "D5", "C7", "D1", "G6", "C8", "A10"}},
      {"0_3.png", {"F11", "F8", "A14", "F10", "G10", "A5", "G8"}},
      {"3_0.png", {"A1", "A11", "A15", "B13", "F1", "F14", "F9", "G13", "G3", "G6", "H2", "H7"}},
      {"3_2.png", {"A4", "B11", "D16", "F1", "F11", "F8", "F9", "G3", "H2", "H4", "H7"}},
  };
  for (const auto& [file, codes] : want) {
    if (!exists(sample(file))) continue;
    const auto j = recognize_file(sample(file), true);
    ASSERT_FALSE(j.contains("error")) << file;
    std::vector<std::string> got;
    for (const auto& e : j["codebook"]) got.push_back(e["code"]);
    EXPECT_EQ(got, codes) << file;
    EXPECT_TRUE(j["legend_check"]["ok"].get<bool>()) << file << " " << j["legend_check"].dump();
    EXPECT_EQ(j["palette_guess"]["palette_id"].get<std::string>().substr(0, 4), "MARD") << file;
  }
}

// 多图合集（T7）和无网格像素画：不崩溃，返回 needs_user_crop 或 no_grid_found
TEST(SampleSmoke, CollagesNeedCropOrNoGrid) {
  for (const char* f : {"0_0.png", "0_1.png", "3_1.png"}) {
    if (!exists(sample(f))) continue;
    const auto j = recognize_file(sample(f));
    ASSERT_TRUE(j.contains("error")) << f;
    const std::string code = j["error"]["code"];
    EXPECT_TRUE(code == "needs_user_crop" || code == "no_grid_found") << f << ": " << code;
  }
}

#endif  // BC_HAVE_IMAGEIO
