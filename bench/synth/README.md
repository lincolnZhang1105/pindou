# bench/synth：合成拼豆图纸生成器（T0.6）

> 用途：① 训练端侧小模型（T0.11：单格色号 CNN、图例 / 坐标文字 CRNN）；② 日常回归测试识别引擎。
> 合成样本**不能替代真实 test 集**，报告里要单列（04 §2.3）。设计依据：04 §1、§2.3、§5。

## 用法（在仓库根目录）

```bash
# 0. 一次性：按锁文件下载字体（二进制不进 git）
PYTHONUTF8=1 .venv/Scripts/python -m bench.synth fonts

# 1. 生成 N 个样本
PYTHONUTF8=1 .venv/Scripts/python -m bench.synth generate --n 1000 --out bench/synth/out/s1 --seed 1 --workers 8
#    可选过滤：--type T1,T4  --style fine5,table_cyan  --size-bucket S,M
#    分批：--start 1000（样本序号从 1000 开始，manifest 自动合并）

# 2. 缩略图拼贴（人工检查风格；--overlay 叠加真值网格线）
PYTHONUTF8=1 .venv/Scripts/python -m bench.synth sheet --in bench/synth/out/s1 --n 50 [--overlay]
```

- `bench/synth/out/` 已在 `.gitignore` 里；放到别处也可以，但**不要提交生成结果**。
- 速度：单进程约 2–3 个样本/秒（主要耗时在 WebP 编码和大图渲染），`--workers 8` 约 8–10 个样本/秒。
- **确定性**：样本 i 的随机数由 `SeedSequence([seed, i])` 派生，同样的 seed、序号、字体文件（锁文件 sha256）
  和依赖版本（numpy / Pillow / OpenCV / libwebp）得到逐字节相同的输出，与 `--workers` 无关（测试覆盖）。

## 输出

```
<out>/images/<image_id>.<png|webp|jpg>      分页图（T4）：<image_id>_<k>.<ext>
<out>/labels/<image_id>.json                真值，bench/README.md 格式，生成时已过 validate_label（含图例校验和）
<out>/extra/<image_id>.extra.json           训练用附加标注（见下文）
<out>/manifest.jsonl                        manifest（platform=synth；分页图每页一行，group = image_id）
```

目录布局和私有数据目录一致，可以直接 `PINDOU_DATA_DIR=<out>` 跑 eval / 标注工具。
附加标注单独放在 `extra/`，因为 eval 会把 `labels/*.json` 都当成真值读。

### 真值的约定

- `image_id = synth_s<seed>_<序号6位>`；`source = {"platform": "synth", "author": "bench.synth", "source_path":
  "own_export"（T6 为 "photo"）, "generator", "seed", "index"}`。
- **坐标**：连续像素坐标，像素 (i, j) 覆盖 [i, i+1) × [j, j+1)。`x_lines[i]` 是第 i 条竖线的**中心**
  （像素画没有线，就是格子边界）；`bbox` 是外框线中心围成的矩形。
- `palette_hint`：色号文字所属的色号表 id（如 `MARD-221`），**不代表色值**（见「色卡约束」）。
- `legend`：图上画了图例才有，否则是 `[]`（eval 用 `bool(legend)` 判断「带图例」）。
- `tags`：只用 04 §2.1 的六个标签。`blurry` 在最终格距 < 8 px 或加了 σ ≥ 1 的模糊时打上（eval 另有 θ 判定）。
- **T4 分页**：`images` 多张，`pages` 每页一项（`col_offset` / `row_offset`），每个格子恰好属于一页；
  图例可能在第一页、最后一页或单独一张图（单独的图例图没有 `pages` 项）。
- **T6 拍照**：网格线不再是竖直 / 水平的，所以 page 里额外有
  `homography`（3×3 行优先，平面坐标 → 照片像素）、`flat_x_lines` / `flat_y_lines`（平面坐标里的精确网格线）。
  此时 `x_lines[i]` / `y_lines[j]` 是该线**中点**投影到照片里的坐标（近似值，只供格距、模糊度等粗略统计），
  `bbox` 是网格四角投影的外接矩形。
- **T7 合集**：一张图里 2–9 个独立图纸。真值（grid / cells / legend / pages）描述**第 0 个面板**，
  即「用户框选出第 0 个面板」之后应得的结果；全部面板的框在附加标注 `panels` 里。

## 风格（`--style`）

| 风格 | 模仿对象 | 要点 |
|---|---|---|
| `table_cyan` | `data/samples/0_2` | 大格子、深色细线、四周彩色坐标栏（同格距、粗体数字）、表格式图例（色块里写色号 + 白底颗数框），图例区有浅灰背景格 |
| `fine5` | `data/samples/3_0` | 浅灰细线、每 5 格彩色粗线、四周小号灰色坐标、横排圆角色块图例（色块里写色号，下方「x颗数」） |
| `fine10` | 同上的变体 | 每 10 格粗线，坐标可能只标 5 / 10 的倍数 |
| `nocoord` | `data/samples/0_0` | 没有坐标；粗线 0 / 5 / 10 格；图例可有可无 |
| `blocks` | T2 | 格内没有文字；方格或圆形豆子（豆子样式有时没有网格线）；图例可有可无 |
| `pixel` | T3 | 没有网格、图例、文字的像素画 |

类型和风格的对应：T1 用前四种；T2 = `blocks`；T3 = `pixel`；T4 / T5 / T6 / T7 用 T1、T2 的风格。

每个样本随机：字体（锁文件里的 12 套）、字重、字号（格距的 34–54%）、长色号统一缩小或单独缩小、文字黑 / 深灰与白 /
浅灰、黑白切换的亮度阈值（Y = 0.299R + 0.587G + 0.114B，阈值 105–165）、细线和粗线的颜色 / 线宽 / 间隔、外框、
坐标样式（彩色栏 / 小号数字，四边或两边，从 1 或 0 开始）、图例布局（`table` / `chips_in` / `chips_under` /
`inline`）、图例排序（首次出现 / 字符串 / 自然序 / 颗数降序）、行优先或列优先、颗数格式（`x534`、`×534`、`534`、
`534颗`、`(534)`）、合计行、中文标题。

## 底图（全部程序生成）

`basemaps.py` 只用 numpy / scipy 生成，不读任何外部图片：卡通角色（头、身体、耳朵、手脚、粗黑轮廓、眼睛 + 高光、
腮红、嘴、头发、白肚皮、条纹）、对称精灵（metaball）、带轮廓的几何形状（圆、圆角方、菱形、爱心、星、三角）、
多物件合集、铺满背景的场景、Voronoi / 拼布色块、随机噪声纹理；区域填充有明暗带、肤色渐变（可加 Bayer 抖动）、
斑点、白豆、相近色组（白 / 米白 / 浅灰）。尺寸覆盖 S–XL（16–208 格）。

## 色卡约束

**不得使用 `bench/palettes_unverified` 的色值**（这些数据会训练进 App 的模型）：

- `palettes.load_code_tables()` 读 bench 色卡时**只取 `code` 字段**，`CodeTable` 没有任何色值属性；
- 色值只来自 `palettes/build/palettes.json`（加载时拒绝 `verified != true`）或随机生成；
- 两种方案：**text**（色号文字取自任意色号表，如 MARD 的 A1 / H7，色值取自干净色值池或随机）和
  **native**（色号和色值都来自同一套干净色卡）。没有图例也没有格内文字的图（T3、无图例 T2）只用 native。
- 每个颜色的来源写在附加标注 `codebook[].source`：`palette:<干净色卡>:<色号>` / `random` / `empty_bg`。
- 测试 `bench/tests/test_synth_palettes.py` 用「所有色值都是哨兵色」的伪造 bench 色卡生成样本，确认哨兵色不出现。

## 退化（04 §2.3）

| 退化 | 参数 | 几何同步 |
|---|---|---|
| 平铺水印 | 概率 0.35（像素画 0.15）；程序生成的 `@handle` / 通用中文词 + 圆 / 星 / 爱心 / 圆角方图标，旋转 ±35°，不透明度 0.08–0.32，可选细斜线；**不模仿任何平台 logo** | 不变 |
| 缩放 | 概率 0.75，目标宽度 1080（55%）/ 1040 / 1000 / 1200 / 1440 / 828 / 750，只缩小（INTER_AREA）；宽 > 4000 一律缩到 2048；T7 一律缩到 1000–1440 | x' = x·W'/W |
| 模糊 | 概率 0.08，高斯 σ 0.5–1.2 | 不变 |
| 有损压缩 | PNG 25% / WebP 50% / JPEG 25%，质量 60–90 | 不变 |
| 截图边框（T5） | 通用手机截图：1080–1440 宽、状态栏、导航栏、底部按钮栏、深色 / 浅色；先 JPEG（70–92）再按上面的格式压缩 | 缩放 + 平移 |
| 分页（T4） | `board`：按 16–52 格一板分页、每页独立渲染（坐标和粗线按绝对位置）；`nine`：整张渲染后沿格线切 2–3 × 2–3 块，图例单独一张 | 平移（+ 缩放） |
| 合集（T7） | 2–9 个面板网格排版 | 平移 + 缩放 |
| 拍照（T6） | 打印件（纸边）或屏幕翻拍（黑边 + 彩色摩尔纹）；透视（旋转 ±10°、角点抖动 2–8%）、桌面纹理背景、光照梯度 + 暗角、通道增益 0.88–1.12、gamma 0.85–1.15、高斯模糊 σ 0.3–1.8、20% 概率运动模糊、传感器噪声、JPEG 70–92 | homography |

测试用 `GenOptions.degrade` 可以强制这些参数（`watermark`、`resize`、`fmt`、`quality`、`blur`、`pre_jpeg`、`t4_mode`）。

## 附加标注（`extra/<image_id>.extra.json`）

```jsonc
{
  "format": "pindou-synth-extra", "version": 1, "image_id": "...", "generator": "bench.synth 0.1.0",
  "seed": 1, "index": 0, "type": "T1", "size_bucket": "M",
  "style": {"name": "fine5", "pitch": 24, "text_face": "inter", "lum_thr": 140.2, ...},   // 渲染参数（字体写 id）
  "layout": {"mode": "board", "board": 29, "legend": "last"},                              // T4 才有内容
  "basemap": {"kind": "character"},
  "palette": {"code_table": "MARD-221", "native": false},
  "codebook": [{"code": "H7", "rgb": [20, 20, 22], "source": "palette:ARTKAL-S:S13",
                "role": "outline", "count": 718, "text_color": "w"}],
  "empty_rgb": [255, 255, 255],
  "cells": {
    "has_text":   ["0011...", ...],    // 每行一个字符串，'1' = 格内有色号文字
    "text_color": ["..wk...", ...]     // '.' 无文字，'k' 黑字（或深灰），'w' 白字（或浅灰）
  },
  "cell_text_boxes": {"H7": [0.18, 0.33, 0.82, 0.66]},   // 色号文字在格内的墨迹框，单位是格距（x0, y0, x1, y1）
  "images": [{
    "file": "...", "width": 1080, "height": 1241,
    "transform": [[...], [...], [...]],   // 渲染画布（分页 / 拼图前）→ 本图的 3×3 矩阵
    "degradations": [{"op": "watermark", ...}, {"op": "resize", "width": 1080}, {"op": "encode", "format": "webp", "quality": 78}],
    "legend": [{"code": "G3", "count": 534, "swatch_box": [x0, y0, x1, y1], "code_box": [...], "code_text": "G3",
                "count_box": [...], "count_text": "x534", "code_in_swatch": true}],
    "coords": [{"axis": "col", "index": 11, "text": "12", "side": "top", "box": [...]}],   // index 从 0 起、绝对位置
    "texts":  [{"kind": "title" | "total" | "ui", "text": "...", "box": [...], "value": 1885}],
    "regions": [{"kind": "grid" | "legend" | "title" | "coords_top" | "coords_bottom" | "coords_left" | "coords_right", "box": [...]}]
  }],
  "panels": [{"panel": 0, "box": [...], "cols": 30, "rows": 28}]   // 只有 T7；合集里各条目另带 "panel" 字段
}
```

- 所有框都是本图的像素坐标（连续坐标），经过了和网格线一样的变换；T6 的框是变换后四角的外接矩形。
- 文字框来自 Pillow 的 `textbbox`（含字形边距）；`cell_text_boxes` 是实际墨迹框（alpha > 40）。
- 每格的文字框 = 格子左上角 + `cell_text_boxes[code]` × 格距（T6 需要再过 homography）。

## 字体（训练数据登记，05 §7）

字体按 `fonts.lock.json` 从 google/fonts（commit `a4c0b21f`）下载到 `bench/synth/fonts/`，全部 **SIL OFL 1.1**
（允许商用、允许嵌入；OFL 原文随字体一起下载）。加载时再校验 sha256 和许可证白名单（OFL-1.1 / Apache-2.0）。

| id | 字体 | 风格 | 用途 |
|---|---|---|---|
| noto-sans-sc | Noto Sans SC（可变字重） | 中文黑体 | 标题、水印、`颗`、界面，也用于色号 |
| zcool-kuaile | ZCOOL KuaiLe | 中文圆趣 | 标题、水印 |
| inter | Inter（可变） | 无衬线 | 色号、坐标、图例、界面 |
| open-sans | Open Sans（可变） | 无衬线 | 同上 |
| roboto | Roboto（可变） | 无衬线 | 同上 |
| arimo | Arimo（可变） | 无衬线（Arial 度量） | 色号、坐标、图例 |
| lato | Lato Regular / Bold | 无衬线 | 同上 |
| barlow-condensed | Barlow Condensed Medium / SemiBold | 窄体 | 同上 |
| roboto-mono | Roboto Mono（可变） | 等宽 | 同上 |
| source-code-pro | Source Code Pro（可变） | 等宽 | 同上 |
| nunito | Nunito（可变） | 圆体 | 同上 |
| varela-round | Varela Round | 圆体 | 同上 |

新增字体：在锁文件加一项（sha256 留空）→ `python -m bench.synth fonts --update-lock` → 在 05 §7 登记。
没有字体时 `generate` 会报错；`--allow-fallback-font` 改用 Pillow 内置的 Aileron（CC0），**只用于调试和 CI 测试**。

## 代码结构

| 文件 | 内容 |
|---|---|
| `palettes.py` | 干净色卡、色号文字表（只取 code）、干净色值池、色号方案抽样 |
| `basemaps.py` | 程序生成底图（角色索引图 + 语义角色 + 难点标签） |
| `colors.py` | 语义角色 → 色号 + 色值（ΔE2000 约束，复用 `bench/common/color.py`） |
| `styles.py` | 6 种风格的随机参数 |
| `render.py` | 渲染（numpy 批量贴字形）+ 网格线 / 坐标 / 图例 / 标题的几何标注 |
| `geom.py` | 坐标约定、3×3 变换、标注同步、真值 page 输出 |
| `degrade.py` | 水印、缩放、压缩、截图边框、拍照 |
| `generate.py` | 抽样计划、分页、合集、退化流水线、写文件 |
| `sheet.py` | 缩略图拼贴 |
| `fonts.py` / `fonts.lock.json` | 字体下载、校验、加载 |

测试：`bench/tests/test_synth_*.py`（真值校验、图例校验和、确定性、缩放 / 截图 / 分页 / 合集 / 拍照后的几何对齐、
图例框对齐、色卡约束）。
