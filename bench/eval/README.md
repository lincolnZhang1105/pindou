# bench/eval：识别 benchmark 评测器（T0.7）

> 设计依据：`docs/04-recognition-benchmark.md` §0（go/no-go）、§3（指标）、§4（竞品），`docs/03-tech-architecture.md` §3.2（RecognitionResult）、§3.6（质量门禁）。
> 数据格式与私有数据目录见 `bench/README.md`。参数见同目录的 `config.yaml`。
> **纪律**：test 集只在 W2 baseline 和 W6 最终评测各用一次，**禁止用 test 集调参**（AGENTS 规则 3）。评测器在代码里强制执行这条纪律（见第 6 节）。

在仓库根目录运行：`PYTHONUTF8=1 .venv/Scripts/python -m bench.eval <子命令> ...`（Linux/macOS 用 `.venv/bin/python`）。

## 1. 命令

| 子命令 | 作用 |
|---|---|
| `run --preds DIR --split dev\|test [--test-reason R] [--labels DIR] [--images DIR] [--manifest F] [--out DIR] [--name RUN] [--allow-missing-labels]` | 评测一个预测结果目录，输出 Markdown 报告、两份 CSV 和 `results.json` |
| `compare --a A --b B --split dev\|test [--out F]` | 两次运行在同一批图上的配对 bootstrap 比较 |
| `gate --preds DIR [--update [--allow-regression]] [--baseline F]` | dev 集质量门禁（CI 用） |
| `gonogo --run RUN --tier-b DIR --tier-b-entry ID [--tier-a CSV...] [--allow-partial-tier-b] [--device-timing-ms X --device-name S]` | 在 test 集上给出 GO / NO-GO |
| `calibrate-theta [--out F]` | 只用 dev 集给出 θ 的建议值和依据 |

所有子命令都接受 `--config`（默认 `bench/eval/config.yaml`）、`--labels`（默认 `$PINDOU_DATA_DIR/labels`）、`--images`（默认 `$PINDOU_DATA_DIR/images`）、`--manifest`（默认 `bench/datasets/manifest.jsonl`）。

- `--preds`：目录路径，或 `$PINDOU_DATA_DIR/preds/` 下的 run 名。run 名默认取目录名。
- `run` 的输出目录默认是 `$PINDOU_DATA_DIR/reports/<run>/<split>/`。
- `compare`：dev 集上 A、B 是预测结果目录（现场评测）；test 集上 A、B 是已经用 `run` 评测并登记过的 run 名（只读 `results.json`，不再消耗 test 集）。
- `--allow-missing-labels` 只用于 dev（标注未完成时跳过没有真值的图），test 集和 gate 不允许。

**退出码**：0 = 成功 / GO / 门禁通过；1 = NO-GO / 门禁退化 / 无法标定 θ；2 = 参数或数据错误（含 test 集纪律拒绝）；3 = 门禁基线缺失或已失效。

**典型流程**

```bash
# W1：dev 集
python -m bench.eval calibrate-theta                      # 标定 θ，把建议值写进 config.yaml 并设 theta_frozen: true
python -m bench.eval run --preds pindo --split dev
python -m bench.eval gate --preds beadcore-main --update  # 在主干版本上建立门禁基线（提交 baselines/dev.json）
python -m bench.eval gate --preds beadcore-pr123          # CI：退化时退出码非 0
python -m bench.eval compare --a beadcore-pr123 --b beadcore-main --split dev
# W2：test 集 baseline（只能一次）
python -m bench.eval run --preds pindo --split test --test-reason baseline_w2
# W6：冻结版本的最终评测（只能一次），然后判定
python -m bench.eval run --preds beadcore-v1 --split test --test-reason final_w6
python -m bench.eval gonogo --run beadcore-v1 --tier-b $PINDOU_DATA_DIR/competitors/<app>/tierb --tier-b-entry <app>/android/image
```

## 2. 输入约定

- **评测单元** = 一个真值文件（约定见 `bench/common/dataset.py`）：普通图纸的 key 是 image_id；分页图（T4）同一 group 的各页合成一个单元，key 是 group id，按整张图纸比较。split 文件里逐页写 image_id；评测器按「`labels/<id>.json` → manifest 的 `group` → 扫描所有真值的 `images[].file`」找到所属真值。同一单元的分页出现在另一个 split 里时报错（数据泄漏）。
- **预测结果文件名**：`preds/<run>/<单元 key>.json`。为了兼容，也接受 group id 或某一页的 image_id 命名，但同一单元出现多个候选文件时报错（有歧义）。目录里没有对应任何单元的文件会在报告里警告（多半是命名不一致）。
- 真值必须通过 `validate_label`（含图例校验和），否则拒绝评测。
- `type`、`tags` 以真值为准；`size_bucket` 由真值网格用 `size_bucket_of` 计算（分页图一律 XL），和真值里填的不一致时警告。

## 3. 指标定义

记号：对一张图，`n_beads` = 非空真值格数（总豆数），`n_empty` = 空真值格数。所有指标都写成「Σ分子 / Σ分母」，逐图的分子分母见 `summary.csv` 和 `per_image.csv`。

### 3.1 逐图状态与失败计分（04 §0.2）

| status | 含义 | 算失败 |
|---|---|---|
| `ok` | 网格列数、行数都等于真值 | 否 |
| `grid_wrong` | 网格尺寸错（含行列颠倒） | 是 |
| `error` | 预测结果是 `{"error": {...}}`（含 `needs_user_crop`） | 是 |
| `missing` | 没有预测结果文件（视为程序崩溃） | 是 |
| `invalid` | JSON 读不出来、缺 grid/codebook/cells、cells 长度不对、codebook 索引越界（视为程序崩溃） | 是 |

**失败的图**：错误格数 = `n_beads`（所有非空真值格按错误计）；零错误记「否」；非空格准确率记 0；用量清单按「没有输出」计（误差 = `n_beads`）；不确定标记命中数为 0。竞品同样处理。

### 3.2 逐格比较（只对 ok 的图）

| 情况 | 计为 |
|---|---|
| 真值非空，预测是另一个色号 | 色号错误 `wrong_code` |
| 真值非空，预测为空 | 豆子→空格 `bead_to_empty` |
| 真值为空，预测非空 | 空格→豆子 `empty_to_bead` |

错误格数 `errors` = 三者之和。注意：ok 的图的错误格数可以超过 `n_beads`（空格→豆子也算错），失败图则固定为 `n_beads`。

### 3.3 汇总指标

| key | 指标 | 定义（Σ 对口径内的图求和） | 边界情况 |
|---|---|---|---|
| `grid_accuracy` | 网格正确率 | Σ[status = ok] / 图数 | 失败图都算网格不正确 |
| `per_mille` | **每千颗错误数（微平均，主指标）** | 1000 × Σerrors / Σn_beads | Σn_beads = 0 时无定义 |
| `nonempty_acc_macro` | 非空格准确率（宏平均） | 对每张图算「非空格中预测正确的比例」，再对图取平均 | `n_beads = 0` 的图不参与；失败图记 0 |
| `zero_error_rate` | **零错误率** | Σ[status = ok 且 errors = 0] / 图数 | 全空的图纸预测也全空算零错误 |
| `failure_rate` | 失败率 | Σ[失败] / 图数 | |
| `bom_error` | 用量清单误差（微平均） | Σ_图 Σ_色号 \|识别颗数 − 真值颗数\| / Σn_beads | 识别颗数取自预测结果的 **cells**（App 的用量清单就是从 cells 统计的），不是 `legend`；失败图按「没有输出」计 |
| `empty_to_bead_rate` | 空格→豆子 | Σempty_to_bead / Σn_empty | 只统计 ok 的图 |
| `bead_to_empty_rate` | 豆子→空格 | Σbead_to_empty / Σn_beads | 只统计 ok 的图 |
| `legend_accuracy` | 图例解析完全正确率 | Σ[legend_ok] / 真值带图例的图数 | 见 3.4 |
| `flag_recall` | 不确定标记召回（上限内） | Σflag_hits / Σerrors | 失败图的错误格计入分母、命中为 0；Σerrors = 0 时无定义（门禁按 1 处理） |
| `flag_recall_ok_only` | 同上，只算 ok 的图 | | 辅助指标，便于区分「标记质量」和「失败」 |
| `flag_precision` | 不确定标记精确率 | Σflag_hits / Σflag_kept | 只统计 ok 的图 |
| `flag_rate` | 标记率（截断后） | Σflag_kept / Σn_beads | 只统计 ok 的图；按构造 ≤ 2% |
| `flag_rate_raw` | 标记率（引擎原始） | Σflag_raw / Σn_beads | 用来发现引擎没守住上限 |

### 3.4 图例解析完全正确

只统计真值带 `legend` 的图。把真值图例和预测结果的 `legend` 各自变成「比较键 → 颗数」（同一色号出现多行时颗数相加），要求**色号集合相同**（预测里多出一个色号、哪怕颗数是 0，也算错）且每个色号颗数相同。预测没有 `legend`、格式不对、error/missing/invalid 都算不正确。图例与网格无关：`grid_wrong` 的图照样评图例。

### 3.5 不确定标记与上限截断（04 §0.1 第 3 条）

- 被标记的格子 = 预测结果的 `uncertain`（行优先编号）；去重，丢掉越界或非整数的编号（报告里警告）。没有 `uncertain` 字段但有 `confidence` 时，用 `confidence < τ` 推出；两者都没有则没有标记。
- **每张图的上限** `cap = min(floor(2% × n_beads), 30)`（用精确分数计算，没有浮点误差）。逐图施加 2% 上限，整个数据集的标记率自然 ≤ 2%。豆数少于 50 的图上限为 0。
- **截断**：标记数超过 `cap` 时，按 `confidence` 升序排序（相同时按格子编号升序；没有可用的 `confidence` 时只按格子编号），只保留前 `cap` 个，其余视为未标记。之后再计算召回、精确率、标记率。
- 这样「把所有格子都标成不确定」最多只能拿到 `cap` 个标记，无法刷高召回。

### 3.6 色号比较

- 色号用 `normalize_code` 规范化（`A01` = `A1`）。
- **品牌** = palette_id 第一个 `-` 之前的部分，大写（`MARD-221`、`MARD-291` → `MARD`；`HAMA-MIDI` → `HAMA`）。
- **主品牌** = 真值 `palette_hint` 的品牌。主品牌的色号比较键不带前缀，其他品牌写成 `品牌:色号`：
  - 真值里没有前缀的色号属于主品牌；`palette_id:code` 的前缀换成它的品牌（和主品牌相同就去掉）。
  - 预测结果 codebook / legend 的 `palette_id` 同样换成品牌：缺省或等于主品牌时不带前缀。
  - 因此品牌认错（MARD 图纸整张认成 COCO）时每格都算色号错误。
- 真值缺 `palette_hint`：两边都去掉品牌前缀，只比较色号（报告里警告）。

## 4. 口径、Hard / Clean、blurry

| 口径 | 包含 | 用于 |
|---|---|---|
| 主统计池 | type ∉ {T6, T7} | 第 1、2、4 条；主指标表；尺寸档、标签分层；门禁的网格正确率、零错误率 |
| 第 3、6 条口径 | type ≠ T7 | 不确定标记、图例 |
| Hard 集 | 主统计池中满足任一条：尺寸档 L/XL；`watermark` 标签；`blurry`（规则判定）；type = T4 | 第 1 条 |
| Clean 集 | 主统计池中不是 Hard 的 | 第 2 条；门禁 |
| 类型分层 | 全部图纸 | T6、T7 在这里单独报告（T7 另报 `needs_user_crop` 的比例） |

**blurry 规则**（满足任一即为 blurry；人工的 `blurry` 标签不参与 Hard 判定，和规则不一致时在报告里警告）：

1. **最小格边长 < 8 px**：每页、每个方向取平均格距 `(最后一条网格线 − 第一条) / 格数`，再取所有页、两个方向的最小值。用平均格距而不是逐格最小值，是为了不受标注时个别网格线偏移的影响。
2. **拉普拉斯方差 < θ**：原图按 EXIF 方向校正（与 core 的输入一致），带透明通道的先叠白底，转灰度（ITU-R 601 luma，uint8），裁出真值网格线围成的矩形，`cv2.Laplacian(ksize=1)` 后取方差。分页图取各页的最小值。HEIC 等格式读 `bench.collect convert` 转出的 PNG。

θ 在 dev 集上用 `calibrate-theta` 标定：以 dev 集人工 `blurry` 标签为依据，只看格边长 ≥ 8 px 的主统计池图纸，候选阈值取相邻方差的中点，选 Youden J（TPR − FPR）最大的；并列时取最小 θ（Clean 更大、第 2 条更严，偏保守）。标定后把 `theta_frozen` 改为 true；**θ 未冻结时评测器拒绝使用 test 集**。

## 5. 统计方法

- **bootstrap 置信区间**：按图纸有放回重采样 n 张（n = 口径内图数），重复 10000 次（`config.yaml`），对每轮算 Σ分子/Σ分母，取 2.5% / 97.5% 百分位。某轮分母为 0 时该轮丢弃。随机数用 `numpy.random.RandomState(seed)`（这个接口的随机流跨 numpy 版本不变），同样的输入总得到同样的区间。
- **配对 bootstrap**（`compare` 和 go/no-go 第 1 条）：只在两边都有结果的同一批图上，每轮对两组用同一批重采样的图，统计量 Δ = 1000 × Σ(err_A − err_B) / Σn_beads。`compare` 报 Δ = A − B（**Δ < 0 表示 A 更好**），CI 整个 < 0 为「A 显著更好」，整个 > 0 为「B 显著更好」，否则「不显著」。两次运行的真值（sha256）或 Hard/口径划分不同时拒绝配对。

## 6. test 集纪律

`run --split test` 依次检查，任何一步不通过都以退出码 2 拒绝，且**不写任何东西**：

1. `--test-reason` 必须是 `baseline_w2` 或 `final_w6`，且 `test_usage.jsonl` 里没有用过这个 reason；
2. `config.yaml` 的 θ 已冻结；
3. `bench/datasets/test.lock` 存在、格式正确；`test.txt` 的 id 集合与 `ids_sha256`、`entries` 一致；manifest 里登记的 sha256 与 lock 一致；每张图片文件的 sha256 与 lock 一致（评测器自带的最小实现，与 `bench.collect` 的 `check_lock()` 按同一格式校验）；
4. 真值里登记的图片 sha256（非空时）与 lock 一致；输出目录里还没有 test 结果（不覆盖）。

全部计算在内存里完成后，先往 `test_usage.jsonl` 追加一行，再写出报告：`{"at", "reason", "run", "git_rev"}`（README 规定的四个字段），另加 `split`、`tool`、`preds_sha256`、`results_sha256`、`config_sha256`、`n_units`。计算出错不会白白消耗 reason；结果也不会在登记之前被看到。

`compare --split test` 和 `gonogo` 只读已登记的 `results.json`，并核对它的 sha256 与使用记录一致（文件被改过就拒绝），不追加新记录。`gonogo` 只接受 `final_w6` 的结果，并重新校验 test.lock 和真值（真值在最终评测之后被改过就拒绝判定）。

## 7. dev 集质量门禁（03 §3.6）

四个确定性指标与 `baselines/dev.json` 比，**都不允许变差**（用分子分母的精确分数比较）：

| 指标 | 口径 | 方向 |
|---|---|---|
| 网格正确率 | 主统计池 | 越大越好 |
| 零错误率 | 主统计池 | 越大越好 |
| Clean 集每千颗错误数 | Clean | 越小越好 |
| 不确定标记召回（上限内） | 不含 T7 | 越大越好（没有错误格时按 1） |

耗时（预测结果 `timing_ms` 各项之和的中位数）只报告。基线里记录数据集指纹（图纸集合、真值 sha256、Hard/口径划分、评测参数指纹）；指纹变了就退出码 3，需要用主干版本的预测结果 `gate --update` 重建。`--update` 遇到退化时拒绝，除非加 `--allow-regression`。

## 8. go / no-go（04 §0）

第 1、2、4 条同时达标为 **GO**，否则 **NO-GO**；第 3、5、6 条只报告。门槛比较用精确分数。

1. **最强竞品**：Tier A 里用量清单误差（微平均，网格错或处理失败按「没有输出」计）最小的竞品条目。只用主统计池里**所有条目都覆盖到**的图比较（某个条目缺图时警告）。并列时按「不惩罚网格错误的原始误差」、再按名称取。若不惩罚网格错误时最强竞品会变，报告里给出敏感性警告。我方的同口径数字列在表里作参考。
2. **第 1 条**：在「Hard ∩ 主统计池 ∩ 最强竞品有 Tier B 结果」的图上配对：我方每千颗错误数 ≤ 0.5 × 竞品，**且** Δ = 竞品 − 我方的配对 bootstrap 95% CI 下界 > 0。`--tier-b-entry` 必须就是最强竞品。Tier B 目录缺某张 Hard 图的文件时默认拒绝判定（竞品处理失败要记成 error JSON；缺文件表示没测，不能当成竞品失败，否则会偏向 GO）；`--allow-partial-tier-b` 时只在有结果的图上比较并警告。比值达标但不显著时，报告提示按 04 §7 第 5 条扩充 Hard 图纸到 ≥ 80 张后再判定，而不是降低门槛。
3. **第 2 条**：Clean 集每千颗错误数 ≤ 1.0。**第 4 条**：主统计池网格正确率 ≥ 98%。口径为空时判为未达标（无法证明达标）。
4. **第 3 条**：上限内召回 ≥ 80%（口径不含 T7），同时报截断后 / 原始标记率和被截断的图数。**第 5 条**：`--device-timing-ms`（T0.14 实测）≤ 3000 ms，没提供时写「未提供」。**第 6 条**：图例完全正确率 ≥ 95%。

输出：`<我方 test 报告目录>/gonogo/gonogo.md`、`gonogo.json`。

## 9. 竞品数据格式

### 9.1 Tier A CSV（每个竞品条目一份或多份，默认读 `$PINDOU_DATA_DIR/competitors/*/tier_a*.csv`）

UTF-8（可带 BOM），长表，**一行一个色号**：

```
app,app_version,platform,import_path,tested_at,image_id,status,cols,rows,code,count
拼豆大王,3.2.1,android,image,2026-11-20,xhs_6aa72c3e_0,ok,58,53,G3,534
拼豆大王,3.2.1,android,image,2026-11-20,xhs_6aa72c3e_0,ok,58,53,H2,264
拼豆大王,3.2.1,android,image,2026-11-20,xhs_7b0d11aa_0,error,,,,
```

- **竞品条目** = `app/platform/import_path`（如 `拼豆大王/android/image`、`拼豆大王/android/link`），iOS 和 Android、「导入图片」和「链接导入」分开记录（04 §4.2）。同一条目内 `app_version`、`tested_at` 必须一致。
- `image_id` 写评测单元 key（分页图写 group id，按整张图纸记录）；不属于 test 集的 id 报错；T6、T7 的行允许但不参与排名。
- `status`：`ok` 或 `error`（竞品无法处理）。`error` 行的 cols/rows/code/count 留空。
- `ok` 行：`cols`、`rows` 是竞品识别出的网格尺寸；`code`、`count` 是用量清单的一项。一张图没有豆子时写一行、code 和 count 都留空。同一张图里同一色号（规范化后）不能出现两次。
- `code` 的写法同真值：主品牌直接写色号；其他品牌写 `palette_id:code`。竞品 App 用的品牌必须与图纸一致（测试时在 App 里选对品牌）。

### 9.2 Tier B

最强竞品导出的 PNG 图纸经我们自己的解析器读取后，存成 RecognitionResult（与我方预测结果同一格式），一个目录、按评测单元 key 命名；竞品处理失败写 `{"error": {...}}`。用 `--tier-b DIR --tier-b-entry <app/platform/import_path>` 传入。

## 10. 输出文件

`run` 在输出目录写：

- `report.md`：主指标（含 95% CI）、与门槛对照、Hard/Clean、尺寸档/类型/标签分层、不确定标记、失败列表、错误最多的图、耗时。
- `per_image.csv`（UTF-8 带 BOM）：每张图一行。列：`image_id, type, size_bucket, tags（; 分隔）, in_main, in_flag_scope, hard, hard_reasons, clean, blurry, min_cell_px, lap_var, gt_cols, gt_rows, n_beads, n_empty, status, error_code, pred_cols, pred_rows, errors, per_mille, wrong_code, bead_to_empty, empty_to_bead, correct_beads, nonempty_acc, zero_error, bom_abs, bom_abs_raw, bom_error, has_legend, legend_ok, flag_source（uncertain/tau/none）, flag_raw, flag_cap, flag_kept, flag_hits, timing_ms, detail, warnings`。布尔值写 1/0，无值留空。
- `summary.csv`（UTF-8 带 BOM）：长表，列 `group_kind, group, metric, value, ci_low, ci_high, num, den, n_images`。`group_kind` ∈ `scope`（`main` / `flag_legend` / `hard` / `clean` / `all`）、`size`（S/M/L/XL，主统计池）、`type`（T1–T7，全部图纸）、`tag`（各标签与「（无标签）」，主统计池）；`metric` 是 3.3 节的 key。`per_mille` 的 value 已乘 1000，其余为比例。
- `results.json`：运行元数据（run、split、git、预测结果指纹、参数指纹、数据集指纹、test_reason、警告）和逐图结果，供 `compare` / `gonogo` 复用。

## 11. 已知限制

- 拉普拉斯方差同时受内容（网格线、格内文字密度）和分辨率影响，θ 只对与 dev 集同类的图纸有意义。
- 逐图 2% 上限对豆数 < 50 的小图等于不允许标记；这是按 04 字面执行的结果。
- 召回在分母里计入失败图的全部豆子，一张失败的大图会显著拉低召回；`flag_recall_ok_only` 单独给出不含失败图的值。
