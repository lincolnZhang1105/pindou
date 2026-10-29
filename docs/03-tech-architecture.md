# 03 技术架构

> 读者：负责工程实现的 AI 或工程师。
> 本文写的是**已经做出的决定**和背后的理由。如果要推翻某个决定，先写清理由，再更新本文。
> 许可证以 `05` 为准，合规以 `06` 为准；功能编号（A1、B3……）见 `02`；任务编号（T0.1……）见 `07`。

---

## 1. 总体决策

| 决策 | 选择 | 理由 | 不选的方案 |
|---|---|---|---|
| App 框架 | **Flutter（Dart）** | 一套代码覆盖 iOS、Android、iPad。Impeller 加 `drawRawAtlas` 足够画大网格。`dart:ffi` 可以直接调 C++ | 原生双端开发（成本翻倍）；Capacitor/WebView（大网格性能有风险）；React Native（C++ 互操作和画布能力较弱） |
| 识别与生成核心 | **C++17 `beadcore`**，基于 OpenCV 4.x | 同一份代码跑 iOS、Android 和桌面 CLI，**benchmark 的结果就等于线上效果** | Dart 实现（太慢）；各平台分别实现（结果不一致） |
| OpenCV 编译范围 | 只编 `core`、`imgproc` 两个模块，**关闭 `videoio`、`highgui`、`imgcodecs`**（不引入 FFmpeg，图片解码交给平台层）；同时关闭 IPP（Intel 单独许可、只有 x86，关掉保证各平台结果一致） | 减小体积，避开 LGPL 依赖 | |
| 桌面 CLI 的图片解码 | `beadcore-cli` 不经过平台层，自带解码：stb_image（PNG/JPEG，MIT）+ libwebp（WebP，BSD-3），并按 JPEG 的 EXIF 方向校正。HEIC 由 `bench.collect convert` 先无损转 PNG。**这两个库只链接进 CLI，不进 App** | 本表要求 core 不含解码，但 benchmark 必须直接读文件；解码放在 CLI 里，core 库本身仍然不含任何解码 | 打开 OpenCV imgcodecs（会把 core 和解码绑在一起） |
| 端侧推理 | **ONNX Runtime（MIT）**，按算子裁剪做定制构建，**必须加 `--no_telemetry` 编译**；ncnn（BSD-3）作为体积备选 | PyTorch 导出 ONNX 后可以直接用。⚠️ ORT 官方构建（含 Android / iOS 预编译包）默认开启微软 1DS 遥测：会经 HTTPS 上报、生成设备标识，自己从源码编译默认也开（2026-10-28 核实其 `Privacy.md`）。这和「同意前不联网、只用白名单 SDK」（AGENTS 规则 5）冲突，所以 App 只能用关闭遥测的定制构建；core 另外在初始化 ORT 前设置 `ORT_DISABLE_TELEMETRY=1` 并调用 `DisableTelemetryEvents` 作为双保险（`core/src/nn/model.cpp`，有单元测试） | TFLite；Core ML 和 NNAPI 分平台实现 |
| 文字识别（色号、坐标、图例） | **自训小模型**：单格色号用 CNN 分类器，图例和坐标用 CRNN + CTC；字符集由候选色卡的色号表生成。**实现现状（2026-10-28）**：图例色号、颗数、坐标数字共用一个 CRNN + CTC（`training/ocr`，约 40 万参数、1.6 MB），读色号时在全部色卡的色号表里用 CTC 概率做封闭词表打分，所以新色号不需要重新训练分类头；格内色号的分类器还没做，格子目前只靠颜色和「有没有字」 | 识别对象是封闭的小词表，自训模型更小也更准 | **识别主路径不使用任何平台 OCR**（Vision、ML Kit），否则 benchmark 和线上结果会不一致。PP-OCRv5/v6 只在训练和冷启动阶段用于离线标注，**不打包进 App** |
| BLE | **flutter_reactive_ble（BSD-3）** | | **禁止使用 flutter_blue_plus**（付费许可，还会上报遥测，见 05 第 4.1 节） |
| 本地存储 | SQLite（`drift`）；格子数据存成 BLOB | 本地优先，方便导出 | Hive / Isar |
| 状态管理 | Riverpod | | |
| 后端 | **MVP 阶段就要有最小后端**（Go 或 Node + PostgreSQL），负责账号、订单、验签、权益、埋点接收和崩溃上报接收。P1 再加云同步。部署在境内云服务器上，域名做 ICP 备案（06 第 2 节） | 支付验签必须在服务端做；没有账号就无法恢复购买（02 J2a） | 海外 BaaS（RevenueCat 等不覆盖国内安卓支付，国内能否稳定访问也没有核实） |
| 崩溃上报 | `sentry_flutter`（MIT），上报到**自建的 GlitchTip**（开源、兼容 Sentry 协议） | 数据留在自己的服务器上；SDK 清单里只需登记一项 | Bugly 等第三方 SaaS |
| 埋点 | 自建：事件批量上报到自己的后端 | 不引入第三方统计 SDK | 友盟等 |
| 本地通知 | `flutter_local_notifications`（BSD-3，引入时在 05 登记） | 到期提醒、续费提醒；**不引入推送 SDK** | 厂商推送 SDK |

**允许使用的 SDK 白名单**（首发版本，同时也是 06 第 7.1 节要求的 SDK 清单的基础）：微信 OpenSDK、支付宝 SDK、sentry_flutter（指向自建 GlitchTip）。**在白名单之外新增任何第三方 SDK，都要先改本表和 06。**

---

## 2. 仓库结构（monorepo）

```
pindou/
├── app/                          # Flutter
│   ├── lib/
│   │   ├── core/                 # 数据库、FFI 绑定、色卡加载、路由、主题、隐私同意闸门
│   │   ├── features/
│   │   │   ├── import/           # A：导入、系统分享接收、框选（B10）
│   │   │   ├── recognition/      # B：识别、复核、手动校正网格、分页合成（B9）
│   │   │   ├── library/          # C：图纸库（含临时图纸 C5）
│   │   │   ├── editor/           # D：编辑、快照
│   │   │   ├── assist/           # E：跟拼（对性能最敏感）
│   │   │   ├── inventory/        # F：库存
│   │   │   ├── generate/         # G：照片转图纸
│   │   │   ├── board/            # H：智能板（首发之后）
│   │   │   ├── export/           # I：导出、备份导入导出
│   │   │   ├── account/          # J：账号、注销、云同步
│   │   │   ├── paywall/          # 会员、权益、自动续费管理、「我们的承诺」
│   │   │   └── settings/         # K：设置、关于、开源许可、客服
│   │   └── main.dart
│   ├── free_limits.json          # 免费额度的唯一来源（产品承诺 1）
│   └── test/ integration_test/
├── core/                         # C++ beadcore
│   ├── include/beadcore/*.h      # C ABI
│   ├── src/{color,layout,grid,legend,classify,solver,stitch,quant,nn}/
│   ├── src/board/                # 智能板协议编解码（首发之后），对外接口为 bc_board_*
│   ├── models/                   # .onnx / .ort，总大小 < 3 MB
│   ├── cli/                      # beadcore-cli（benchmark 用；另出一个 Android arm64 版本用于真机测速，见 T0.14）
│   ├── tests/
│   └── CMakeLists.txt
├── bench/                        # Python
│   ├── collect/                  # 只生成 manifest（图片靠人工保存，不写抓取脚本）
│   ├── label/                    # 真值标注工具
│   ├── synth/                    # 合成数据
│   ├── competitors/              # 竞品结果的录入和解析（adb 自动化）
│   ├── eval/                     # 评测器，config.yaml 里固化 θ、τ、标记率上限等参数
│   ├── engine/                   # 用 beadcore-cli 批量跑识别，产出 preds/<run>/
│   ├── palettes_unverified/      # 来源存疑、只供 bench 内部用的色卡（MARD 等，见 palettes/SOURCES.md）
│   ├── common/                   # 共用的数据格式、颜色公式（与 core 一致）
│   └── datasets/                 # 只放 manifest；图片放在私有存储里（见 3.6）
├── training/                     # PyTorch 训练代码 → 导出 ONNX
├── palettes/                     # 色卡数据（来源和许可证在 palettes/SOURCES.md 登记，见 05 第 3 节）
├── backend/                      # 账号、订单、验签、权益、埋点、GlitchTip；P1 再加同步
└── docs/
```

**会被打包进 App 或部署到服务端的目录**：`app/`、`core/`、`palettes/`、`core/models/`、`backend/`。这些目录**禁止**放入 GPL、AGPL、CC-BY-NC 许可的内容，也禁止放入来源存疑的数据（AGENTS 规则 1）。

---

## 3. 识别引擎（beadcore）

算法见 `04` 第 5 节；指标和评测协议见 `04` 第 0、3、4 节。

### 3.1 C ABI

```c
typedef struct { const uint8_t* rgba; int width, height, stride; } bc_image;  // RGBA8888，方向已校正
typedef struct bc_ctx bc_ctx;

bc_ctx* bc_create(const char* models_dir, const char* palettes_json, char** err_json);
void    bc_destroy(bc_ctx*);

// 识别。palette_hint 可以传 NULL，表示自动判断（B8）
char* bc_recognize(bc_ctx*, const bc_image* imgs, int n_imgs, const char* palette_hint,
                   int (*progress_cb)(float progress, void* user), void* user);
// 用户手动校正网格后重新识别（支持多页；pages_json 的格式见 3.3）
char* bc_recognize_with_grid(bc_ctx*, const bc_image* imgs, int n_imgs, const char* pages_json,
                             const char* palette_hint, int (*progress_cb)(float, void*), void* user);
// 照片转图纸（params 和返回值的格式见 3.4）
char* bc_generate(bc_ctx*, const bc_image* img, const char* params_json,
                  int (*progress_cb)(float, void*), void* user);
void  bc_free(char*);
const char* bc_version(void);
```

**约定：**
- **所有返回 `char*` 的函数都不会返回 NULL。** 出错时返回 `{"error": {"code": "...", "message": "..."}}`。错误码包括：`invalid_input`、`no_grid_found`、`needs_user_crop`、`cancelled`、`oom`、`internal`。
- **`progress_cb` 返回非 0，表示取消。** 引擎会尽快返回 `cancelled`。
- **线程**：`bc_ctx` **不是线程安全的**。每个 Dart Isolate 创建自己的 ctx，或者在 Dart 侧把调用串行化。
- **解码放在平台层**：
  - iOS 用 ImageIO 解码，Android 用 `ImageDecoder`（API 28+）或 `BitmapFactory`；
  - EXIF 方向在平台层先校正好，再以 RGBA 传给 core；
  - HEIC 支持范围见 02 A1。
- **原图分辨率**：
  - 保存图、截图、分享图，从导入到识别全程**不压缩、不缩小**。
  - 拍照导入允许在透视校正后，按每格 ≥ 24 px 重采样（02 A1）。
  - 内存峰值 ≤ 200 MB；超大图分块处理。
- **模型文件位置**：
  - Android 的 assets 不是普通目录，首次启动时把模型解压到 `getFilesDir()/models/<version>/`，再把这个路径传给 `models_dir`。
  - iOS 直接用 bundle 里的路径。

### 3.2 RecognitionResult

```json
{
  "engine_version": "0.3.1",
  "palette_guess": {"palette_id": "MARD-221", "confidence": 0.97},
  "grid": {"cols": 58, "rows": 53},
  "pages": [{"image_index": 0, "bbox": [0,0,0,0], "x_lines": [], "y_lines": [],
             "col_offset": 0, "row_offset": 0, "cols": 58, "rows": 53, "source": "auto|coords|manual"}],
  "codebook": [{"palette_id": "MARD-221", "palette_version": "<commit>", "code": "G3", "legend_rgb": [228,196,170]}],
  "cells": [0, 1, 0],
  "confidence": [0.99, 0.62, 0.98],
  "uncertain": [1234, 2051],
  "legend": [{"palette_id": "MARD-221", "code": "G3", "count": 534}],
  "legend_check": {"ok": false, "diff": [{"code": "H2", "expected": 264, "actual": 262}]},
  "warnings": ["cell_too_small_for_text", "watermark_detected"],
  "timing_ms": {"layout": 40, "grid": 120, "legend": 60, "classify": 900, "solve": 80}
}
```

- `cells`：0 表示空格；k ≥ 1 时指向 `codebook[k-1]`。这和 02 里 `Pattern.cells` 的语义完全一致。
- codebook 条目的附加字段（v0 起）：`in_legend: false` 表示图例里没有这个颜色、是从格子里聚类出来的；`page_colored: true` 表示和页面底色同色、靠格内文字认出来的豆子（通常是白豆：白底上的白色图例色块看不见）。色号读不出来时 `code` 写成 `?1`、`?2`……，交给用户确认。
- v0 还不能读出图例颗数：`legend` 为空数组，`legend_check` 为 `{"ok": false, "diff": [], "unavailable": true}`。v1（T0.12）起按本节格式输出。
- **多图输入（分页图，B9）**：
  - 每张图分别检测网格；没有网格的那张当作图例页。有图例页时只用图例页的图例，因为网格页边缘被切开的半格很像色块。
  - 表格式或排得很整齐的图例页也可能被检测成一个小网格。同一套分页图的格距基本一致，所以格数 ≤ 100、格距又和其他网格页的中位数相差 1.6 倍以上的页，也按图例页处理。板边角的小页格距和别的页相同，不受影响。
  - 接缝一侧通常没有留白，最后一条网格线离图片边缘只有一两个像素。网格检测在图片边缘附近照样追踪和补线，不能因为搜索窗口越界就停下。
  - 坐标编号可能从 0 开始，也可能从 1 开始，只用来算页与页的相对位置。
  - 每页都读出了坐标编号时，按编号直接定位（`pages[].col_offset / row_offset`）。
  - 否则按版式猜，并在 `warnings` 里写 `pages_order_unverified`，提示 UI 让用户确认或拖动排序。猜法是枚举每行放几张，先按累加摆放，再选最自洽的一种：
    1. 读出了坐标的页（分页图常常只在整张图纸的外框上标坐标），「坐标编号 − 摆放位置」应是同一个常数，允许 ±2 格的累加误差，不一致的页越少越好；
    2. 同一行的页行数相同、同一列的页列数相同，每对相同 +1、不同 −1，总分越高越好；
    3. 前两条打平时按经验顺序：列坐标回到最小值处断行、按板切（行尾那张更窄）、平方数张按九宫格、行数都相同排成一行、列数都相同排成一列、⌈√n⌉ 张一行，其余按离 √n 的远近。
  - 版式定下后，读出坐标的页按自己的坐标放；基准取第一行、第一列里读出坐标的页，它们不受其他页尺寸误差的影响。其余的页按累加补位。
  - 多图输入不做拼图（T7）判断。
- `needs_user_crop` 的触发条件（v0）：网格外还有比网格本身更多的内容且网格占图面 < 15%；图例色块在纵向上分成多段；或者没有图例而图案分成 ≥ 4 个分散的块。
- **用 codebook 而不是直接用色卡索引的原因**：
  - 上游色卡调整色号顺序时，已经保存的图纸不会悄悄变色；
  - 一张图纸可以混用多个品牌的色号（04 第 5.3 节）。

### 3.3 pages_json（手动校正网格时的输入）

格式就是 3.2 里的 `pages` 数组，`source` 字段写 `"manual"`。

### 3.4 bc_generate 的参数

```json
// params
{"palette_id": "MARD-221", "cols": 104, "rows": 0, "max_colors": 24,
 "remove_background": false, "merge_threshold_de": 3.0, "dither": "none"}
```

- `rows` 填 0，表示按原图比例自动计算。
- 返回值使用 3.2 的子集，只有 `grid`、`codebook`、`cells`，再加上 `stats: {"colors_used": 22}`。

### 3.5 palettes_json、备份 JSON、CSV

**palettes_json**（由 `palettes/` 在构建时生成）：

```json
{"palettes": [{"id": "MARD-221", "version": "<上游 commit>", "source": "maxcleme/beadcolors",
               "license": "MIT", "colors": [{"code": "A1", "name": "", "rgb": [0,0,0], "lab": [0,0,0], "series": "A"}]}]}
```

- 各品牌的主数据源和核实状态，见 05 第 3 节和 `palettes/SOURCES.md`。

**备份 JSON**（功能 I3、I4）：

```json
{"format": "pindou-backup", "version": 1, "exported_at": "...",
 "patterns": [/* Pattern 对象，cells 用 base64 编码 */],
 "progress": [], "inventory": [], "inventory_txns": [], "settings": {}}
```

- 不包含权益和账号信息。
- 导入时按 id 合并，冲突时让用户选「覆盖」还是「保留两份」。

**用量 CSV**：

```
palette_id,code,name,count,in_stock
```

### 3.6 质量门禁和数据存放

- **数据存放**：benchmark 的图片和真值放在**私有对象存储**（或私有 NAS）里，不进任何 git 仓库。
- **CI 访问**：只有 self-hosted runner 能读这些数据，凭据放在 CI secrets 里。
- **门禁**：每次修改识别路径都要跑 dev 集，以下**确定性指标不允许变差**：
  - 网格正确率
  - 零错误率
  - Clean 集每千颗错误数
  - 不确定标记召回（在标记率上限之内统计）

  耗时只报告，不设门禁。

  > 注意（2026-10-28 实现评测器时发现）：「不确定标记召回不允许变差」可能挡住一次真实的准确率提升——错误变少以后，剩下的错误往往更难标出，召回反而下降。遇到这种情况，以前三项指标都没变差、且每千颗错误数下降为准，由 PO 审批后用 `bench.eval gate --update --allow-regression` 更新基线，并在提交说明里写明原因。
- **`core/tests` 必须包含**：
  - Sharma CIEDE2000 的 34 组测试数据；
  - 和 colour-science 做对拍；
  - 合成数据的 golden 测试；
  - **冒烟测试**：断言 `data/samples/0_2.png` 是 16×16、`3_0.png` 是 58×53；其余样例只断言不崩溃，或者正确返回 `needs_user_crop`。样例图片不进 git，没有时测试自动跳过（`core/tests/test_samples.cpp`，另外锁住了 0_3、3_2 的尺寸和各颜色颗数）。
  - **合成图测试**：测试里现画图纸（细线 + 每 5 格粗线、非整数格距、青色坐标栏、9 px 小格），断言网格尺寸精确、逐格零错误（`core/tests/test_grid.cpp`）。CI 上不依赖任何私有数据。

---

## 4. 跟拼模式的渲染（assist）

目标：208×208 的图纸在中端机上保持 60fps。

- **LOD（分级渲染）**：
  - 缩小时，把整张图渲染成「每格 1 像素」的位图，用 `FilterQuality.none` 放大显示；
  - 放大到每格 ≥ 12 逻辑像素时，只画可见区域，文字用字形图集（glyph atlas）配合 `drawRawAtlas` 批量绘制，**不要逐格调用 TextPainter**。
- **高亮**：每个色号对应的格子预先算成 bitmask；切换高亮只重建遮罩层。
- **进度**：用 Bitset 表示，去抖 300 ms 后增量写入数据库。
- **屏幕常亮**：用 `wakelock_plus`，许可证在 05 第 6 节登记。

---

## 5. 智能板（board，首发之后上线）

**原则：只接入以下三类协议。**
1. 开放协议（比如 BeadLink）；
2. 拿到书面授权的协议；
3. 经过法务书面确认、本身**没有鉴权**的第三方逆向协议，并且实现代码只能用宽松许可证。

**我方不对任何带鉴权或加密的协议做抓包和逆向。** 第 3 类协议的实机验证，只限于读取广播名和 GATT 服务表。分级细则见附录 A.4。

```dart
// 底层用 flutter_reactive_ble。协议编解码在 core/src/board（bc_board_* 接口），或者用 Dart 实现
abstract class BoardAdapter {
  String get vendor;
  bool matches(DiscoveredBoard d);                 // 按「设备名前缀 + service UUID」组合识别
  Future<void> connect(DiscoveredBoard d);
  Future<void> disconnect();
  Future<int> negotiateMtu(int desired);
  Future<BoardInfo> info();                        // BoardInfo { cols, rows, firmware, capabilities }
  Future<void> showCells(List<CellLight> cells);   // CellLight { x, y, rgb, brightness }
  Future<void> clear();
  Stream<BoardEvent> events();                     // BoardEvent: keyPrev | keyNext | disconnected | error
}
// 图纸大于板子时：BoardViewport { colOffset, rowOffset }，由 UI 选择要显示的区域
```

- **权限**：蓝牙权限在 H1 上线时才声明（02 H 组的注意事项）。
- **授权约束**：连板的代码路径**不读取 Entitlement**，以保证产品承诺第 5 条（连板永远免费）。

---

## 6. 账号、付费与权益（account / paywall）

**商品形态**见 02 第 4 节，**合规要求**见 06 第 4、5 节（paywall 必须逐条满足 06 第 5 节的第 1–7 条）。

- **账号**（02 J2a）：
  - 用户购买时触发登录，支持手机号或 Sign in with Apple；
  - 后端签发 token；
  - 支持在 App 内注销（15 个工作日内完成）。
- **iOS**：StoreKit 2，通过 `in_app_purchase` 接入。
  - 商品有：自动续期订阅（月 / 年）、非续期订阅（单次的月卡 / 年卡，到期时间由服务端计算）、非消耗型（永久）、消耗型（积分，不过期）。
  - 提供「恢复购买」。
  - 后端接收 App Store Server Notifications V2，处理退款和撤销。
  - 加入小型企业计划。
- **Android**：
  - 微信 App 支付和支付宝 App 支付。插件（比如 fluwx）在引入前要在 05 登记。
  - **首发只做单次购买**。
  - 订单由后端生成，后端负责签名和验签；**客户端回传的支付结果不能作为发放权益的依据**。
- **提醒**：
  - iOS 自动续费：在续费前 5 日提醒，每次扣款前也要提醒。实现方式是本地通知，加上后端通过 **APNs** 补发（APNs 是系统服务，不是第三方 SDK）。
  - Android 单次卡：到期前 5 日发本地通知，提醒用户手动续费。
  - 通知权限在用户第一次购买成功后再申请；用户拒绝的话，在「我的自动续费」页里显示提醒的说明。
- **权益**：
  - 服务端是唯一的权威来源，客户端本地缓存一份。离线宽限期 7 天（02 第 4 节）。
  - **会员过期后的行为按 02 第 4 节「会员过期后」一列执行**，单元测试要覆盖每一行。
- **防止重复扣费**：用户购买永久会员前，如果检测到有生效中的订阅，就提示他去 `showManageSubscriptions` 取消。

---

## 7. 隐私与安全（细节见 06 第 7 节）

- 识别只在端侧进行，图片不上传。只有 AI 功能（P2）在用户逐次同意后，才会上传当次那一张图片。
- **隐私同意闸门**：
  - `app/lib/core` 里的闸门在用户同意之前，**拦住所有 SDK 的初始化和所有网络请求**。
  - 用户拒绝时进入「基础模式」（02 K3）。
  - 集成测试要验证：首次启动、同意之前，**没有任何网络请求**。
- **权限**：

  | 权限 | 做法 |
  |---|---|
  | 相册读取 | 不申请（用系统选择器） |
  | 相册写入 | iOS 只声明 `NSPhotoLibraryAddUsageDescription`；Android 用 SAF 或 MediaStore（API 29+），不申请存储权限 |
  | 相机 | 用户点击拍照时才申请 |
  | 通知 | 第一次购买成功后才申请 |
  | 蓝牙 | H1 上线时才声明，细节见 06 第 7 节 |

- **云同步**（J2b）：只同步格子数据、进度、库存和元数据，**不上传原始图片**；服务端加密存储。

---

## 8. 构建与发布

- CMake 在 CI 里产出：iOS xcframework、Android AAR（arm64-v8a、armeabi-v7a）、桌面 CLI、Android arm64 CLI（用于真机测速）。
- ONNX Runtime 按模型裁剪，arm64 的 .so ≤ 5 MB；模型总大小 < 3 MB，每个模型都带版本号。
- **ORT 的 App 构建必须带 `--no_telemetry`**，发布前用抓包确认冷启动和识别过程中没有任何对微软域名的请求（纳入 T1.1「同意之前没有任何网络请求」的集成测试）。桌面 benchmark 用的是官方预编译包（`core/deps.json`），只在开发机和 CI 上运行，不分发。
- App 内放「开源许可」页，按 05 第 5 节履行各项许可义务。

---

## 附录 A：智能板协议调研（2026-10-27）

> 「已核实」指读过一手来源；「用户说」指小红书、豆瓣上的二手讨论。**本附录不构成法律意见。**

### A.1 市场格局

| 体系 | 品牌 / 官方 App | 尺寸 | 协议情况 |
|---|---|---|---|
| **I-ledshow**（深圳；已核实，包名都是 `com.led.*`） | PIXDOU（俗称「熊猫」）、拼个豆吧（牛板）、FusePop；光豆社可能也是（未核实） | 52 / 64 / 78 / 104 | **没有公开协议**。唯一的信息来源是 GPL 项目 x1a0q1sx/pindou-studio 的研究文档，没有在实机上验证过：service `0xA950`，控制 `0xA951`，数据 `0xA952`，帧格式 `[0x54][cmd][len16][payload][cks16BE]`，图案以 GIF 下发。App 里有厂商白名单 `supportsLighting` |
| **Heaton / PixDotDot**（已核实，「拼个豆豆」的包名是 `com.heaton.pindoudou`） | 拼个豆豆 App、像素糖果 HPD 系列 | 32 / 52 / 78 / 104 | 有开源实现 oxyroid/pixdot-protocol（Apache-2.0）。设备名以 `PDD_` 开头，MTU 512，整图下发时带 CRC-32 和 ACK，**没有鉴权**，service UUID 没有公开 |
| **松鼠** | 官方 App 据说是拼个豆豆 | 52 / 78 / 104 等 | **归属未确定**。有两条相反的线索：①用户说松鼠和像素糖果几乎一模一样，官方 App 是 Heaton 的拼个豆豆，指向 Heaton 系；②颂拼豆能连松鼠，而有研究说颂拼豆用的是 PIXDOU（I-ledshow）协议。见 A.6 |
| **宝石角** | 豆好拼 | 52 / 104 | **协议封闭，而且会主动封堵**（用户说：连过官方 App 后会被升级固件）。已核实，它和拼个豆吧（牛板）有官方合作 |
| 小品牌 | PixelPad、粒方、豆艺社（评论疑似营销）、拼豆达人等 | — | 协议都没查到 |

价格（用户说）：松鼠 104 格约 ¥180–200；PIXDOU 约 ¥270–330；宝石角 104 格约 ¥300–500。

### A.2 开放协议和开源实现

| 项目 | 许可 | 能连哪些现货板子 | 能不能用 |
|---|---|---|---|
| scsagentclub/BeadLink | 规范 CC-BY 4.0，代码 Apache-2.0 | 暂无（v0.1 草案） | ✅ 作为我们的开放驱动基线 |
| tumao7/pindouban（酷猫 kmpd） | 自定义：集成进收费产品需要书面授权 | 几乎没有；UUID 是 ESP32 例程里的默认值 | ❌ 拿到授权之前不能用 |
| oxyroid/pixdot-protocol | Apache-2.0 | Heaton 系 | ⚠️ 属于第 3 类协议，要等法务结论 |
| x1a0q1sx/pindou-studio | **GPL-3.0** | 实现了 kmpd 和 PIXDOU 两种协议，还带虚拟板 | ❌ 代码不能复制 |
| christophersmith2737-commits/Pin_assistant | Apache-2.0 | 某款 104 格板子（fa00 家族） | 仅参考 |

### A.3 竞品连板情况

| 竞品 | 收费 |
|---|---|
| 颂拼豆 | 免费 |
| 拼豆大王 | 用户说免费；已核实支持 FusePop |
| AI 豆仓 | 授权费 ¥48；用户说它破解了宝石角 |
| 我嘞个豆 | 有积分门槛 |
| 拼豆钥匙 | 第三方转接器 |

结论：连板免费只是追平竞品，不构成优势。

### A.4 接入分级

| 级别 | 做法 | 对象 |
|---|---|---|
| ✅ 直接做 | 实现开放协议 | BeadLink |
| ✅ 优先谈合作 | 和方案商签书面协议，拿到协议文档和白名单资格 | 按顺序：① I-ledshow ② Heaton ③ 酷猫 kmpd ④ 小厂 ⑤ 宝石角（只能走合作） |
| ⚠️ 法务确认后才能做 | 第 3 类：逆向得到、但没有鉴权的协议，只能用宽松许可的实现 | Heaton 系（oxyroid/pixdot-protocol） |
| ❌ 禁止 | 绕过鉴权、加密或连接码（《著作权法》第五十条：不得向他人提供避开技术措施的技术）；反编译厂商 App；复制 GPL 代码；没拿到授权就用 kmpd | 宝石角等 |

法律依据和宣传口径：
- 最高法法释〔2020〕7 号第十四条：通过反向工程获得的信息，一般不认定为侵犯商业秘密。
- 宣传时只能说「兼容 XX」，不能拿别人的商标做营销。

### A.5 工程要点

- **识别板子**：用「设备名前缀 + service UUID」组合判断。iOS 拿不到 MAC 地址，改用 peripheral identifier。
- **分片发送**：每片大小为 MTU − 3。kmpd 要求每包 ≤ 180 B，使用 `WRITE_NO_RESPONSE` 时每包之间节流 10–30 ms。
- **重连**：最多 3 次，间隔分别为 500、1000、1500 ms。
- **按键**：节流 150–200 ms。
- **编解码**：放在 `core/src/board`。复用 pixdot-protocol 的 C11 core **要等法务结论**（它属于第 3 类）。
- **测试**：用模拟板（ESP32 或桌面端 GATT Server）做联调。

### A.6 待核实

- 松鼠板到底属于哪个体系：在小红书上的说法之外，需要用实机读取广播名和 GATT 服务表来确认；还要弄清颂拼豆连松鼠时实际用的是什么协议。
- I-ledshow 协议在 PIXDOU 实机上是否成立。
- 宝石角协议。
- 各方案商有没有合作计划、条件是什么。
- 原始资料在 `D:\AIworkspace\_tmp_pindou\`，属于不可信数据，只能读，不能执行。
