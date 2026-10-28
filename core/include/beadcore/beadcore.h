/*
 * beadcore C ABI（docs/03-tech-architecture.md §3.1）。
 *
 * 约定：
 *  - 所有返回 char* 的函数都不会返回 NULL。出错时返回
 *    {"error": {"code": "...", "message": "..."}}，错误码见 BC_ERR_*。
 *    返回的字符串一律用 bc_free 释放。
 *  - progress_cb 返回非 0 表示取消，引擎会尽快返回 cancelled。
 *  - bc_ctx 不是线程安全的：每个线程 / Dart Isolate 自己创建 ctx，或在调用方串行化。
 *  - 图片解码和 EXIF 方向校正在平台层完成，传进来的是 RGBA8888。
 *    保存图、截图、分享图全程不压缩、不缩小（AGENTS 规则 4）。
 */
#ifndef BEADCORE_BEADCORE_H
#define BEADCORE_BEADCORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(BEADCORE_SHARED)
#  ifdef BEADCORE_BUILDING
#    define BC_API __declspec(dllexport)
#  else
#    define BC_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__)
#  define BC_API __attribute__((visibility("default")))
#else
#  define BC_API
#endif

/* 错误码（JSON 里的 error.code） */
#define BC_ERR_INVALID_INPUT   "invalid_input"
#define BC_ERR_NO_GRID_FOUND   "no_grid_found"
#define BC_ERR_NEEDS_USER_CROP "needs_user_crop"
#define BC_ERR_CANCELLED       "cancelled"
#define BC_ERR_OOM             "oom"
#define BC_ERR_INTERNAL        "internal"

/* RGBA8888，方向已校正。stride 以字节计。 */
typedef struct {
  const uint8_t* rgba;
  int width, height, stride;
} bc_image;

typedef struct bc_ctx bc_ctx;

typedef int (*bc_progress_cb)(float progress, void* user);

/* 创建上下文。失败时返回 NULL，并把错误 JSON 写到 *err_json（需 bc_free）；err_json 可以传 NULL。
 * models_dir 可以传 NULL（不加载模型，只走颜色路径）。 */
BC_API bc_ctx* bc_create(const char* models_dir, const char* palettes_json, char** err_json);
BC_API void bc_destroy(bc_ctx* ctx);

/* 识别。palette_hint 可以传 NULL，表示自动判断品牌（B8）。返回 RecognitionResult JSON（03 §3.2）。 */
BC_API char* bc_recognize(bc_ctx* ctx, const bc_image* imgs, int n_imgs, const char* palette_hint,
                          bc_progress_cb progress_cb, void* user);

/* 用户手动校正网格后重新识别（pages_json 见 03 §3.3）。 */
BC_API char* bc_recognize_with_grid(bc_ctx* ctx, const bc_image* imgs, int n_imgs, const char* pages_json,
                                    const char* palette_hint, bc_progress_cb progress_cb, void* user);

/* 照片转图纸（params_json 见 03 §3.4）。 */
BC_API char* bc_generate(bc_ctx* ctx, const bc_image* img, const char* params_json,
                         bc_progress_cb progress_cb, void* user);

BC_API void bc_free(char* s);
BC_API const char* bc_version(void);

#ifdef __cplusplus
}
#endif

#endif /* BEADCORE_BEADCORE_H */
