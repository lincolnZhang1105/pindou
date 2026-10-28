# bench/collect：样本登记工具（T0.3）

只负责把**人工保存下来的图片**登记进 manifest、划分 dev/test、锁定 test 集。**不抓取、不爬虫、不自动下载**（AGENTS 规则 5）。
数据格式见 `bench/README.md`，覆盖要求见 `docs/04-recognition-benchmark.md` §2.1。

所有命令都在仓库根目录运行（Windows 下加 `PYTHONUTF8=1`）：

```bash
export PINDOU_DATA_DIR=/path/to/private/pindou-bench   # 私有存储，不在仓库里
PYTHONUTF8=1 .venv/Scripts/python -m bench.collect <子命令> -h
```

## 1. 采集图片

1. 在小红书 / 抖音 App 里打开图纸笔记，长按图片 →「保存图片」到手机相册。**不要截图**（截图另算 T5）。
2. 把图片原样导出到电脑（数据线、AirDrop、网盘原图下载都行，**不要用微信发送**，会被压缩），放进 `$PINDOU_DATA_DIR/inbox/`。
3. 记下笔记 id（链接里 `explore/` 后面那串）、作者、链接。

## 2. 登记（add）

一篇笔记的几张图一起登记，元数据从命令行给：

```bash
python -m bench.collect add inbox/IMG_0001.jpg inbox/IMG_0002.jpg \
  --platform xhs --note-id 6aa72c3e000000002901a806 --author 某画师 \
  --url https://www.xiaohongshu.com/explore/6aa72c3e000000002901a806 --saved-at 2026-10-28T10:00:00+08:00 \
  --source-path app_save --type T1 --grid 58x53 --palette-hint MARD-221 --tags watermark,white_vs_empty --move
```

- 不给路径时登记 `inbox/` 下的全部图片（按文件名自然排序）。`--dry-run` 只校验不写入；`--move` 登记后把源文件移到 `inbox/_registered/`。
- `image_id` 默认是 `<platform>_<note_id 前 8 位>_<序号>`（如 `xhs_6aa72c3e_0`），序号接着已有的往下排；`--id` 可手动指定（只能单张）。
- 图片**原字节复制**到 `images/<image_id>.<扩展名>`，扩展名按真实格式（文件名是 .jpg、内容是 WebP 时登记为 .webp，只改名不转码），并记录 sha256、宽高（只读文件头）。
- **sha256 重复会拒绝**；一批里只要有一张出错，整批都不登记。
- `--grid 列x行` 会自动推算 `size_bucket`（S ≤ 40 < M ≤ 80 < L ≤ 130 < XL）；不知道行列数时用 `--size-bucket` 估一个。T7 可以不填。
- `--saved-at mtime` 表示取文件修改时间（手机导出不一定保留保存时间，能看到真实时间就手填）。

| 字段 | 取值 |
|---|---|
| `--platform` | `xhs` / `douyin` / `own`（自己用 App 导出）/ `photo`（拍照）/ `other` |
| `--source-path` | `app_save`（App 里保存）/ `own_export`（自己导出，真值已知）/ `photo`（拍照）/ `web`（Web 端版本，只做分辨率对比） |
| `--type` | `T1` 带色号 / `T2` 纯色块 / `T3` 像素画 / `T4` 分页 / `T5` 截图 / `T6` 拍照 / `T7` 多图合集 |
| `--tags` | `similar_colors`、`single_pixel_detail`、`gradient_skin`、`white_vs_empty`、`watermark`、`blurry` |
| `--palette-hint` | `MARD-221`、`MARD-291`、`COCO-291`、`MANMAN-278`、`PANPAN-289`、`MIXIAOWO-290` 等 |

**分页图（T4）**：同一套的几页一起登记，`--type T4`。`group` 默认是 `<platform>_<note_id 前 8 位>`，页码按 image_id 末尾的序号，
所以**要按页码顺序给出文件**（或在 CSV 里按顺序排）。分页图一律计入 XL 档。真值文件是 `labels/<group>.json`。

**批量登记**：写一个 sidecar CSV（UTF-8，第一行是列名，`file` 必填，其余列名同上面的参数名，空格子用命令行参数的值）：

```csv
file,note_id,author,url,type,grid,tags,palette_hint
IMG_0001.jpg,6aa72c3e000000002901a806,某画师,https://...,T1,58x53,watermark;white_vs_empty,MARD-221
IMG_0002.jpg,682a8972000000002202b28f,另一位,https://...,T4,,watermark,MARD-291
```

```bash
python -m bench.collect add --csv inbox/batch.csv --platform xhs --source-path app_save --move
```

`file` 相对 CSV 所在目录；标签用 `;` 分隔。

## 3. 分辨率对比（resolution，T0.3 子项）

同一张图在电脑浏览器（Web 端）里右键「图片另存为」，登记成 App 版本的配对：

```bash
python -m bench.collect add inbox/web_6aa72c3e_0.webp --pair-of xhs_6aa72c3e_0     # 默认 id：xhs_6aa72c3e_0_web
python -m bench.collect resolution --out resolution.md
```

输出 Markdown 表：两边的尺寸、格式、文件大小、宽度比、估算的每格像素（有真值时按 x_lines 精确算，否则用「整图宽 / 列数」作上限）。
Web 版本不进入划分、不计入覆盖统计、不需要标注。结论由主 agent 写进 04 §1.1。

## 4. 核对和统计

```bash
python -m bench.collect verify          # 图片都在、sha256 一致、文件头尺寸一致、划分和 test.lock 一致
python -m bench.collect stats           # 对照 04 §2.1 的覆盖要求：尺寸档、类型、Hard、品牌、难点标签，末尾列出警告
```

统计按「图纸单元」计：普通图纸 1 张算 1 个，分页图一组算 1 套。Hard = L/XL 档、T4、带 watermark 或 blurry 标签。

## 5. 划分和锁定

```bash
python -m bench.collect split           # 按（size_bucket, type）分层，dev:test = 1:2；T7 不参与；同组分页图不拆开
python -m bench.collect lock-test       # T0.5b：test 集全部标完后锁定，写 bench/datasets/test.lock
```

- `split` 是**增量**的：已经在 dev.txt / test.txt 里的保持不动，只给新登记的分配（否则调过参的 dev 图可能被换进 test）。
  种子固定，同样的输入结果相同。`--reshuffle` 会丢弃已有划分，慎用。
- `test.lock` 存在后 `split` 和 `lock-test` 都会拒绝执行。`lock-test` 默认要求 test 集每个单元都有通过校验的真值，
  确实要提前锁定时加 `--allow-unlabeled`。
- 评测器用 `from bench.collect import check_lock` 校验锁定（返回错误列表，空表示通过）。

## 6. HEIC（convert）

iPhone 拍照（T6）多是 HEIC。登记时照常登记（保留原图，宽高从文件头解析，不需要额外依赖），然后：

```bash
python -m bench.collect convert         # 无损转成 $PINDOU_DATA_DIR/cache/decoded/<image_id>.png
```

转出的 PNG 已按 HEIF 旋转和 EXIF 方向校正，beadcore-cli 和标注工具都用它，**真值坐标也以它为准**。
解码依赖可选的 `pillow-heif`（`.venv/Scripts/pip install pillow-heif`）：源码 BSD-3-Clause，但**二进制 wheel 捆绑了
x265（GPL-2.0）、libheif / libde265（LGPL-3.0），整体按 GPLv2 发布**。只能在 bench 本地使用，不得进入 `app/`、`core/`、`palettes/`、`backend/`。

## 约定（给 eval / label 的接口）

- `bench/common/dataset.py`：私有数据目录下的路径（`images_dir`、`labels_dir`、`decoded_path`、`readable_image_path` …）、
  标注单元（`unit_key`、`build_units`）、`is_hard`。
- `readable_image_path(rec)`：喂给 beadcore-cli 的图片路径（HEIC 用转出的 PNG）。
