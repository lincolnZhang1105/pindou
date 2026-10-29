# 阶段 0 进展与交接（截至 2026-10-29）

> 写给产品负责人（PO）。任务编号见 `07`，指标定义见 `04`。本文随进展更新。

## 1. 一句话现状

- **工具链已经齐了**：采集、标注、评测、合成数据、识别引擎、benchmark runner，都能端到端跑通。
- **真实数据还是空的**：阶段 0 的 go/no-go 结论要等 PO 采集、标注 dev 和 test 集，并测完竞品。
- 现有的 4 张真实图纸样例，引擎能**全部识别正确**：网格尺寸、每个色号、每个颗数都对，「与图例完全一致」。
- 在难度更大的合成集上还有明显差距，见第 3 节。合成集上的调优已暂停，等真实 dev 集（第 5 节）。
- **现在卡在第 4 节的 PO 事项上**，最关键的是第 2 项：采集和标注。

## 2. 任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| T0.1 仓库骨架和 CI | ✅ 本地完成 | Windows（MSVC）、Linux（WSL）编译和测试通过；`.github/workflows/ci.yml` 已写好。**macOS 和 CI 本身还没跑过**：仓库还没有远端。self-hosted runner 和私有存储等 PO 提供（外部依赖 E） |
| T0.2 色卡 | ✅ | 颜色库通过 Sharma 34 组测试和 colour-science 1 万组对拍（误差 < 1e-6）。12 套国际色卡已核实来源后引入。**MARD 和国内品牌的色值来源有问题**，见第 4 节 |
| T0.3 采集工具 | ✅ 工具完成 | `bench/collect`。人工采集 ≥ 150 张（负责人 LBL）还没开始。「App 保存图 vs Web 图」分辨率结论要等有配对样本 |
| T0.4 标注工具 | ✅ 工具完成 | `bench/label`（本地网页）。在 3_0 上用「按图例颜色自动填充」，12 个色号颗数全对。**「58×53 十分钟标完」还没有真人计时** |
| T0.5a/b 标注 dev 40、test 80 | ⏳ 等 LBL | |
| T0.6 合成数据 | ✅ | `bench/synth`：6 种渲染风格、T1–T7 全覆盖、可复现 |
| T0.7 评测器 | ✅ | `bench/eval`：04 §3 的全部指标、配对 bootstrap、dev 门禁、test 集使用纪律，80 个用例。对 04 歧义定义的 12 条解释已写进 04 §3.1 |
| T0.8 Pindo baseline | ✅ 工具完成，⏳ 等 dev 集 | `python -m bench.competitors.pindo`：Pindo（GPL）源码按锁文件下载到本地、不进仓库，打包后作为 Node 进程运行，和 Chrome 里的结果逐格一致。合成集上网格正确 0/39（按真值裁到网格区域后 5/39）：它把坐标栏和图例也算进网格。合成集评不了它的颜色路径，**正式 baseline 报告要等 T0.5b 的 dev / test 集**。见 `bench/competitors/pindo/README.md` |
| T0.9 竞品 Tier A | ⏳ 等 PO | 要买竞品会员（约 ¥200–300）；Android 自动化脚本由 AI 写 |
| T0.10 beadcore v0 | ✅ | 网格检测、图例色块、颜色分类、求解 |
| T0.11 模型 | ✅ 基本完成 | 一个 CRNN + CTC（40 万参数、1.6 MB）同时读图例色号、颗数、坐标和格内色号，只用合成数据训练（第三版 `ocr_v3`）。合成验证集：色号 99.8%、颗数 99.3%、坐标 87%、格内色号 91%。C++（ONNX Runtime）与 PyTorch 输出误差 < 1e-3（有单测）。和 07 的原设计不同：格内色号没有单独做 CNN 分类器，而是复用同一个模型在图例色号里做封闭词表打分，见 03 §1 |
| T0.12 beadcore v1 | 🔄 进行中 | 已有：OCR 读图例色号和颗数、按色号集合判断品牌（B8）、图例核对（B5）、颗数约束求解、坐标数字定范围、格内文字证据（只读颜色拿不准的格子）、拼图识别（`needs_user_crop`）、看不见色块的白豆条目、分页拼接（B9：按坐标定位，读不出时按版式猜）。还缺：不确定格的置信度标定（要等真实 dev 集）。合成分页图拼对的从 9/30 提到 24/30（2026-10-29：修了接缝处少一行一列、坐标从 0 编号、图例页被当成网格、没坐标时的版式推断、表格式坐标栏被当成多一行一列），剩下的主要是个别页的网格检测本身出错 |
| T0.13 go/no-go 报告 | ⏳ | 依赖 T0.5b、T0.9、T0.12 |
| T0.14 真机测速 | ⏳ 等设备 | 工具已就绪：`tools/build-android.sh` 交叉编译 arm64 版 CLI（OpenCV 关闭 ARM 加速库以保证结果和桌面一致），`tools/make_bench_image.py` 生成 104×104 标准测速图，`tools/android-bench.sh` 用 adb 推送并测速、结果写入 `build/android-bench.md`。手机插上、打开 USB 调试即可测。桌面端同一张图约 130 ms |

## 3. 当前数字（只作开发参考，不是 go/no-go 结论）

| 数据 | 网格正确 | 色号和颗数 | 说明 |
|---|---|---|---|
| 真实样例（`data/samples` 中 4 张带图例的图纸） | 4 / 4 | 4 / 4 全对，图例核对一致 | 只有 4 张，说明不了统计上的问题 |
| 合成 T1 集（60 张，含大量模糊、水印、XL 大图） | 50 / 60 | 图例完全正确 37.5%（OCR 第三版） | 剩下的：格距 ≤ 8 px 的表格式坐标栏（坐标数字只有三四个像素高，读不出来）、个别图行方向网格线整体偏半格，以及合成图例的文字识别 |
| 合成 T4 分页集（30 套） | 24 / 30（修复前 9 / 30） | — | 合成分页样式很极端（圆形豆子、4 px 高的坐标、每 5 格标一个）。剩下 6 套：个别页格距检测成一半或漏掉大段线（格距 6–11 px），以及网格外不带框的坐标数字行被算进网格 |

合成集比真实图纸更难，也更「怪」。正式结论只看真实 dev 和 test 集。

## 4. 需要 PO 处理或拍板的事项

1. **私有存储和 CI**：私有对象存储（或 NAS）、一台 self-hosted runner、一个 git 远端仓库。有了以后，把仓库变量 `PINDOU_DEV_GATE` 设为 true，就会打开 dev 集质量门禁。
2. **采集和标注**：按 `bench/collect/README.md`、`bench/label/README.md` 操作。目标是 150 张采集、120 张标注；标注前请先用 1–2 张试一下速度。
3. **竞品会员和测试**（T0.9）。
4. **一台中端安卓机**（骁龙 7 系），用于 T0.14 测速。
5. **法务**：
   - 色卡数据：原定的 MARD 主数据源 maxcleme/beadcolors，和 Zippland（AGPL）的数据有 290/291 处相同；COCO、漫漫、盼盼、咪小窝也都没有来源可确认的色值（`palettes/SOURCES.md`）；
   - 合成训练数据用到了这些品牌的**色号文字**（不含色值），请一并确认。
6. **建议提前自建色卡**：买实物色卡，用色度计测色。识别靠图例，不受影响；但照片转图纸和平替推荐需要可用的色值。
7. **ONNX Runtime 的遥测**：官方构建默认会把数据上报给微软。App 必须用 `--no_telemetry` 编译的版本（已写进 03、05、06）。这是工程约束，不需要拍板，但上线检查时要核对。
8. **合成 T6（拍照）的真值格式**：给 page 加了 `homography` 等字段，暂定只在合成数据里用，见 `bench/README.md`。

## 5. AI 接下来要做的

**2026-10-29 决定：合成集上的调优先暂停。**
- 合成集上剩下的失败合成痕迹越来越重：格距 ≤ 8 px 的坐标栏、个别图行方向网格整体偏半格、圆形豆子的分页样式。继续针对它们调阈值，容易把引擎调得只适合合成集。
- 等真实 dev 集标好，按真实图纸的失败分布再定先修什么。

目前 AI 这边没有不受阻塞的大块工作，都在等第 4 节的事项。**dev 集标好后按这个顺序做**（命令细节见 `bench/eval/README.md`）：

1. **跑我方引擎并出报告**：
   - `python -m bench.engine run --split dev --run beadcore-v1`
   - `python -m bench.eval run --preds beadcore-v1 --split dev`
2. **标定**：
   - θ：`python -m bench.eval calibrate-theta`；
   - 不确定格的置信度：在标记率上限内召回 ≥ 80%，T0.12 的剩余部分。
3. **建门禁基线**：`python -m bench.eval gate --preds beadcore-v1 --update`。之后每次改识别都要过 dev 门禁（AGENTS 规则 3）。
4. **Pindo baseline（T0.8）**：
   - `python -m bench.competitors.pindo run --split dev --run pindo-4ae3613 --white-as-empty`
   - 再用 `bench.eval run` 出报告。
5. **按 dev 集的失败分布排修复优先级**。上面暂停的几类问题，在那时一起评估。
6. **W2**：test 集上只跑一次 Pindo baseline（`--test-reason baseline_w2`）。我方引擎的 test 集评测只在阶段冻结时做一次（`final_w6`）。

## 6. 真机测速怎么跑（T0.14）

1. 编译（Linux 或 WSL）：`tools/build-android.sh`。第一次运行会下载 Android NDK r27c，产物是 `~/pindou-build/android-arm64/beadcore-cli-stripped`。
2. 手机打开「开发者选项 → USB 调试」，用数据线连上电脑，确认 `adb devices` 能看到设备。
3. 在 Git Bash 里运行（WSL 用户名按本机调整）：

```bash
tools/android-bench.sh //wsl.localhost/Ubuntu-24.04/home/dministrator/pindou-build/android-arm64/beadcore-cli-stripped //wsl.localhost/Ubuntu-24.04/home/dministrator/pindou-deps/android-arm64-v8a/onnxruntime/jni/arm64-v8a/libonnxruntime.so
```

结果追加到 `build/android-bench.md`。门槛是中端机识别 104×104 ≤ 3 秒（04 §0.1 第 5 条）。
