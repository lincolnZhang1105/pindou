// C ABI 实现（03 §3.1）。这里是异常和 C 世界的边界：任何异常都转换成错误 JSON，绝不返回 NULL。
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cstring>
#include <new>

#include <nlohmann/json.hpp>

#include "beadcore/beadcore.h"
#include "engine/engine.h"

#ifndef BC_VERSION_STRING
#define BC_VERSION_STRING "0.0.0"
#endif

struct bc_ctx {
  bc::Context ctx;
};

namespace {

char* dup_string(const std::string& s) {
  char* p = static_cast<char*>(std::malloc(s.size() + 1));
  if (!p) return nullptr;
  std::memcpy(p, s.c_str(), s.size() + 1);
  return p;
}

// 兜底的 oom 字符串：连错误 JSON 都分配不出来时，返回这块静态区（仍然不返回 NULL），bc_free 会跳过它
const char kOomJson[] = R"({"error":{"code":"oom","message":"out of memory"}})";

char* oom_json() {
  char* p = dup_string(kOomJson);
  return p ? p : const_cast<char*>(kOomJson);
}

char* error_json(const std::string& code, const std::string& message) {
  try {
    nlohmann::ordered_json j;
    j["error"]["code"] = code;
    j["error"]["message"] = message;
    return dup_string(j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
  } catch (...) {
    return oom_json();
  }
}

template <typename F>
char* guarded(F&& f) {
  try {
    const std::string out = f();
    char* p = dup_string(out);
    return p ? p : oom_json();
  } catch (const bc::Error& e) {
    return error_json(e.code(), e.what());
  } catch (const std::bad_alloc&) {
    return oom_json();
  } catch (const cv::Exception& e) {
    if (e.code == cv::Error::StsNoMem) return oom_json();
    return error_json(BC_ERR_INTERNAL, std::string("OpenCV: ") + e.what());
  } catch (const std::exception& e) {
    return error_json(BC_ERR_INTERNAL, e.what());
  } catch (...) {
    return error_json(BC_ERR_INTERNAL, "unknown exception");
  }
}

std::vector<cv::Mat> wrap_images(const bc_image* imgs, int n_imgs) {
  bc::require(imgs != nullptr && n_imgs > 0, BC_ERR_INVALID_INPUT, "没有传入图片");
  std::vector<cv::Mat> out;
  out.reserve(n_imgs);
  for (int i = 0; i < n_imgs; ++i) out.push_back(bc::wrap_rgba(imgs[i]));
  return out;
}

std::string dump(const bc::ojson& j) { return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace); }

}  // namespace

extern "C" {

bc_ctx* bc_create(const char* models_dir, const char* palettes_json, char** err_json) {
  if (err_json) *err_json = nullptr;
  bc_ctx* ctx = nullptr;
  char* err = guarded([&]() -> std::string {
    bc::require(palettes_json != nullptr, BC_ERR_INVALID_INPUT, "palettes_json 为空");
    auto c = std::make_unique<bc_ctx>();
    c->ctx.palettes = bc::PaletteSet::parse(palettes_json);
    if (models_dir) {
      c->ctx.models_dir = models_dir;
      // models.json 指定当前启用的模型（{"ocr": "ocr_v2"}）；没有时用 ocr_v1
      std::string ocr_name = "ocr_v1";
      std::ifstream mf(std::string(models_dir) + "/models.json", std::ios::binary);
      if (mf.good()) {
        std::stringstream ss;
        ss << mf.rdbuf();
        const auto mj = nlohmann::json::parse(ss.str(), nullptr, false);
        if (mj.is_object() && mj.contains("ocr") && mj["ocr"].is_string()) ocr_name = mj["ocr"].get<std::string>();
      }
      c->ctx.ocr = bc::nn::OcrModel::load(models_dir, ocr_name);
    }
    ctx = c.release();
    return "{}";
  });
  if (ctx) {
    bc_free(err);
    return ctx;
  }
  if (err_json) {
    *err_json = err;
  } else {
    bc_free(err);
  }
  return nullptr;
}

void bc_destroy(bc_ctx* ctx) { delete ctx; }

char* bc_recognize(bc_ctx* ctx, const bc_image* imgs, int n_imgs, const char* palette_hint,
                   bc_progress_cb progress_cb, void* user) {
  return guarded([&]() -> std::string {
    bc::require(ctx != nullptr, BC_ERR_INVALID_INPUT, "ctx 为空");
    bc::RecognizeRequest req;
    req.images = wrap_images(imgs, n_imgs);
    if (palette_hint && *palette_hint) req.palette_hint = std::string(palette_hint);
    return dump(bc::recognize(ctx->ctx, req, bc::Progress(progress_cb, user)));
  });
}

char* bc_recognize_with_grid(bc_ctx* ctx, const bc_image* imgs, int n_imgs, const char* pages_json,
                             const char* palette_hint, bc_progress_cb progress_cb, void* user) {
  return guarded([&]() -> std::string {
    bc::require(ctx != nullptr, BC_ERR_INVALID_INPUT, "ctx 为空");
    bc::require(pages_json != nullptr, BC_ERR_INVALID_INPUT, "pages_json 为空");
    bc::RecognizeRequest req;
    req.images = wrap_images(imgs, n_imgs);
    if (palette_hint && *palette_hint) req.palette_hint = std::string(palette_hint);
    try {
      req.pages = nlohmann::json::parse(pages_json);
    } catch (const nlohmann::json::exception& e) {
      bc::fail(BC_ERR_INVALID_INPUT, std::string("pages_json 不是合法 JSON：") + e.what());
    }
    bc::require(req.pages->is_array() && !req.pages->empty(), BC_ERR_INVALID_INPUT, "pages_json 必须是非空数组");
    return dump(bc::recognize(ctx->ctx, req, bc::Progress(progress_cb, user)));
  });
}

char* bc_generate(bc_ctx* ctx, const bc_image* img, const char* params_json, bc_progress_cb progress_cb,
                  void* user) {
  return guarded([&]() -> std::string {
    bc::require(ctx != nullptr, BC_ERR_INVALID_INPUT, "ctx 为空");
    bc::require(img != nullptr, BC_ERR_INVALID_INPUT, "img 为空");
    bc::require(params_json != nullptr, BC_ERR_INVALID_INPUT, "params_json 为空");
    nlohmann::json params;
    try {
      params = nlohmann::json::parse(params_json);
    } catch (const nlohmann::json::exception& e) {
      bc::fail(BC_ERR_INVALID_INPUT, std::string("params_json 不是合法 JSON：") + e.what());
    }
    return dump(bc::generate(ctx->ctx, bc::wrap_rgba(*img), params, bc::Progress(progress_cb, user)));
  });
}

void bc_free(char* s) {
  if (s == nullptr || s == kOomJson) return;
  std::free(s);
}

const char* bc_version(void) { return BC_VERSION_STRING; }

}  // extern "C"
