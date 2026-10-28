# bench：识别准确率 benchmark（阶段 0）

> 设计依据：`docs/04-recognition-benchmark.md`。任务编号见 `docs/07-roadmap-and-tasks.md` 阶段 0。
> **纪律**：test 集只在两个时间点使用（W2 baseline、W6 最终评测），**禁止用 test 集调参**（AGENTS 规则 3）。

## 运行约定

- Python ≥ 3.11。依赖见 `bench/requirements.txt`。在**仓库根目录**运行：`python -m bench.<模块> ...`。
- 本地开发用仓库根目录的 `.venv`：`python -m venv .venv && .venv/Scripts/pip install -r bench/requirements.txt`（Linux/macOS 用 `.venv/bin/pip`）。
- 测试：`python -m pytest bench/tests`。

## 目录

| 目录 | 内容 | 任务 |
|---|---|---|
| `collect/` | 生成和维护 manifest、划分 dev/test、锁定 test 集。**只处理人工保存下来的图片，不写任何抓取代码** | T0.3 |
| `label/` | 真值标注工具（本地网页） | T0.4 |
| `synth/` | 合成数据生成器 | T0.6 |
| `eval/` | 评测器、质量门禁、`config.yaml`（θ、τ、标记率上限等参数） | T0.7 |
| `competitors/` | 竞品结果的录入和 adb 自动化；`competitors/pindo/`：Pindo baseline（外部 Node 进程，见其 README） | T0.8、T0.9 |
| `engine/` | 用 beadcore-cli 批量跑识别，写 `preds/<run>/` | T0.8、T0.10、T0.12 |
| `datasets/` | **只放 manifest 和划分清单**，不放图片 | T0.3、T0.5 |
| `palettes_unverified/` | 来源存疑、只供 bench 内部使用的色卡（见 `palettes/SOURCES.md`） | T0.2 |
| `common/` | 共享代码：颜色公式（与 core 一致）、数据格式读写 | — |
| `tools/` | 零散工具（如 colour-science 对拍数据生成） | — |

## 私有数据（不进任何 git 仓库）

benchmark 的图片、真值和预测结果放在私有存储里（03 §3.6），本地用环境变量 `PINDOU_DATA_DIR` 指向它：

```
$PINDOU_DATA_DIR/
├── inbox/                      # 刚从手机导出、还没登记的图片
├── images/<image_id>.<ext>     # 登记后的原图（文件名就是 image_id，扩展名保持原格式，不转码、不缩放）
├── labels/<image_id>.json      # 真值，格式见下文
├── preds/<run>/<image_id>.json # 某次运行的预测结果（RecognitionResult，或 {"error": ...}）
├── competitors/<app>/...       # 竞品导出的结果
└── reports/<run>/              # 评测报告
```

CI 里只有 self-hosted runner 能挂载这份数据，凭据放在 CI secrets（03 §3.6）。

## 数据格式

### manifest（`bench/datasets/manifest.jsonl`，每行一张图）

```json
{"image_id": "xhs_6aa72c3e_0", "file": "xhs_6aa72c3e_0.png", "sha256": "...", "width": 1080, "height": 1241,
 "format": "png", "platform": "xhs", "note_id": "6aa72c3e000000002901a806", "author": "...", "url": "...",
 "saved_at": "2026-10-28T10:00:00+08:00", "source_path": "app_save", "type": "T1", "size_bucket": "M",
 "palette_hint": "MARD-221", "tags": ["watermark"], "group": null, "notes": ""}
```

- `source_path`：`app_save`（在 App 里「保存图片」）、`own_export`（自己用主流 App 导出，真值已知）、`photo`（拍照）、`web`（Web 端版本，**只用于 T0.3 的分辨率对比**，不参与划分、统计和标注）。
- `group`：分页图（T4）的同一套图共享一个 group id，页码按 `image_id` 末尾的序号。
- 可选字段：`pair_of`（`web` 版本对应的 `app_save` 图的 image_id）、`grid`（登记时人工预估的 `{"cols", "rows"}`）、`exif_orientation`。
- JPEG 的 `width` / `height` 是文件里存储的尺寸（没按 EXIF 旋转）；真值坐标一律以方向校正后的图为准，和 core 一致。
- 合成数据（`bench/synth`）的 manifest 用 `platform: "synth"` 区分，`source_path` 是 `own_export`（T6 是 `photo`）。
- `type`、`size_bucket`、`tags` 在采集时由人工预估填写，标注完成后以真值为准。

### 真值（`labels/<image_id>.json`）——04 §2.2

```json
{
  "image_id": "xhs_6aa72c3e_0",
  "source": {"platform": "xhs", "note_id": "...", "author": "...", "saved_at": "...", "source_path": "app_save"},
  "type": "T1", "size_bucket": "M", "palette_hint": "MARD-221",
  "images": [{"file": "xhs_6aa72c3e_0.png", "sha256": "..."}],
  "grid": {"cols": 58, "rows": 53},
  "pages": [{"image_index": 0, "bbox": [x0, y0, x1, y1], "x_lines": [...], "y_lines": [...],
             "col_offset": 0, "row_offset": 0, "cols": 58, "rows": 53}],
  "legend": [{"code": "G3", "count": 534}],
  "cells": [["H7", null, "G3"]],
  "tags": ["watermark", "white_vs_empty"]
}
```

- 分页图（T4）一整套是一个「标注单元」，真值文件名是 `labels/<group>.json`；草稿在 `labels/drafts/`。
- 标注工具另外写入 `needs_review`、`annotator`、`labeled_at`、`reviewed_by`、`reviewed_at`（XL 和分页图要第二人复核）。图纸本身没有图例时 `legend` 写 `[]`。
- 合成 T6（拍照）的真值在 page 里额外带 `homography`、`flat_x_lines`、`flat_y_lines`（拍照后网格线不再横平竖直，`x_lines` / `y_lines` 只是近似）。**这三个字段只出现在合成数据里**，core 的 pages_json 不接受它们；真实拍照样本的标注格式在拍照导入转正（B7）时再定。
- `cells[row][col]`：色号字符串；`null` 表示空格；混用其他品牌时写 `palette_id:code`（如 `ZG:ZG1`）。
- `x_lines` 有 `cols + 1` 个值，`y_lines` 有 `rows + 1` 个值，单位是原图像素（可以是小数）。
- 分页图：`images` 有多张，`pages` 每页一项，用 `col_offset` / `row_offset` 表示该页在整张图纸里的位置。
- 多分页图（T4）有多张 `images`；多图合集（T7）不进入自动识别统计。

### 预测结果（`preds/<run>/<image_id>.json`）

就是 beadcore 的 RecognitionResult（`docs/03-tech-architecture.md` §3.2），失败时是 `{"error": {"code": ..., "message": ...}}`。
`cells` 是一维数组、按行优先排列，`0` 表示空格，`k ≥ 1` 指向 `codebook[k-1]`。

### 划分与 test 集锁定

- `bench/datasets/dev.txt`、`bench/datasets/test.txt`：每行一个 `image_id`（`#` 开头为注释）。
- `bench/datasets/test.lock`（T0.5b 完成后生成，之后不得修改）：

  ```json
  {"created_at": "...", "ids_sha256": "<sha256(\"\\n\".join(sorted(test_ids)))>",
   "entries": [{"image_id": "...", "sha256": "<图片 sha256>"}]}
  ```
- `bench/datasets/test_usage.jsonl`：test 集的使用记录，每次使用追加一行 `{"at", "reason", "run", "git_rev"}`。
  `reason` 只允许 `baseline_w2`、`final_w6`，**每个 reason 只能用一次**（04 §6 纪律）。

### 合成数据

`python -m bench.synth generate --n N --out DIR --seed S` 生成的目录和 `$PINDOU_DATA_DIR` 布局相同（`images/`、`labels/`、`manifest.jsonl`），另有 `extra/`（训练用的附加标注：图例和坐标的文字框、格内文字等，格式见 `bench/synth/README.md`）。
评测合成集：`python -m bench.engine run --ids ... --manifest DIR/manifest.jsonl` 后
`python -m bench.eval run --split synth --manifest DIR/manifest.jsonl --labels DIR/labels --images DIR/images --preds DIR/preds/<run>`。
**合成集只用于开发和训练，不能代替真实 dev / test 集**（04 §2.3）。

## 色卡

- 正式色卡（可以进 App）：`python -m palettes.build_palettes` → `palettes/build/palettes.json`
- bench 用的合并色卡（额外包含 MARD、COCO 等来源存疑的色卡）：`python -m bench.palettes_unverified.import_unverified` → `bench/palettes_unverified/build/palettes_bench.json`
- **合成训练数据不得使用 `palettes_unverified` 的色值**（只可以用其中的色号文字）。

## 版权

`data/samples/` 和 benchmark 图片的版权属于画师：只放私有存储，不进公开仓库，不打包进 App，**不用于训练**（AGENTS 规则 8）。
