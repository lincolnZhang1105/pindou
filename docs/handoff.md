# 给接手 AI 的交接说明（截至 2026-10-29）

> 写给下一个接手工程实现的 AI。本文说清三件事：项目在做什么、做到了哪里、还有哪些问题。
> 给产品负责人（PO）看的进展和待办在 [phase0-status.md](phase0-status.md)。两份文档冲突时，以 phase0-status 为准，并顺手把本文改对。

## 0. 接手时先读什么

1. **本文**：花 10 分钟了解全局和工作环境的坑。
2. [AGENTS.md](../AGENTS.md)：项目规则，**必须遵守**，包括许可证红线、test 集纪律、隐私和版权。
3. [phase0-status.md](phase0-status.md)：任务状态、当前数字、等 PO 处理的事项、dev 集到位后的执行顺序。
4. 按需再看：
   - [03](03-tech-architecture.md)：架构、C ABI、JSON 格式；B9 分页拼接和坐标栏的规则；
   - [04](04-recognition-benchmark.md)：指标定义，§3.1 是评测器对歧义定义的解释；
   - [05](05-open-source-references.md)：依赖登记和许可证；
   - [07](07-roadmap-and-tasks.md)：任务编号和 DoD。

## 1. 项目在做什么

- **产品**：「豆伴」，面向中国拼豆玩家的工具 App。核心功能是识别小红书等平台上保存下来的拼豆图纸，产出网格、每格色号和用量清单，再做高亮跟拼、库存管理等。产品定义见 02，商业取舍见 README。
- **阶段 0（当前）**：先证明识别做得比最强的竞品好，再决定要不要入场。
  - 产出是 C++ 识别核心 `beadcore` 和一整套 benchmark 工具。
  - 阶段 0 结束时有 **go/no-go 关口**（04 §0）。**no-go 就停下，不写 App**。
- **正式 App 还没开始写**：阶段 1 必须等 go 之后才能开始。
- **例外**：PO 2026-10-31 决定先做一个安卓原型（`app/`），只装在 PO 自己的手机上试识别效果，见第 4.1 节和 07「例外：安卓原型」。

## 2. 做到了哪里

工具链已经齐了：采集、标注、合成数据、识别引擎、评测器、竞品 baseline、真机测速，都能端到端跑通。**但真实数据还是空的**，所以还没有任何 go/no-go 结论。

| 任务 | 状态 |
|---|---|
| T0.1 骨架和 CI | Windows / Linux 本地编译和测试通过；CI 写好但没跑过（没有远端仓库） |
| T0.2 色卡、T0.6 合成数据、T0.7 评测器、T0.10 v0 引擎 | 完成 |
| T0.3 采集工具、T0.4 标注工具 | 工具完成，**真人采集和标注都还没开始** |
| T0.5 标注 dev / test 集 | 等 PO（负责人 LBL） |
| T0.8 Pindo baseline | 工具完成，等 dev 集 |
| T0.9 竞品 Tier A | 等 PO 买会员；Android 自动化脚本由 AI 写，但需要装好竞品的设备 |
| T0.11 OCR 模型 | 基本完成（`ocr_v3`） |
| T0.12 v1 引擎 | 进行中，剩下的要等 dev 集：不确定格的置信度标定，以及按真实失败分布修问题 |
| T0.13 go/no-go 报告 | 等 T0.5b、T0.9、T0.12 |
| T0.14 真机测速 | 工具完成，等中端安卓机 |

**当前数字**（只作开发参考，详见 phase0-status §3）：
- 4 张真实样例：全部识别正确。
- 合成 T1 集 eval60：网格正确 50/60，图例完全正确 37.5%。
- 合成分页集 evalT4：拼对 24/30。

**2026-10-29 与 PO 约定：合成集上的调优暂停**，等真实 dev 集标好，再按真实失败分布排优先级。**不要再主动针对合成集调阈值**，剩下的合成失败合成痕迹很重，继续调容易过拟合。

## 3. 仓库地图

| 路径 | 内容 |
|---|---|
| `core/` | C++17 识别核心，对外只有 C ABI（`include/beadcore/beadcore.h`）。说明见 `core/README.md` |
| `core/src/grid` | 网格检测：线响应投影 → 格距 → 逐线追踪 → 「四边有线」的格子连通块 → 去缝 / 剥坐标栏；坐标数字读取 |
| `core/src/engine` | 管线编排：多图拼接（B9）、图例、OCR、分类、颗数约束求解、输出 JSON |
| `core/src/legend`、`classify`、`nn`、`color`、`palette` | 图例检测、逐格分类、ONNX Runtime 封装和 OCR、色差、色卡解析 |
| `core/models/` | OCR 的 ONNX 模型；`models.json` 指定启用 `ocr_v3` |
| `core/tests/` | GoogleTest，41 个：颜色对拍、C ABI、合成图网格、OCR 对拍、真实样例冒烟（样例不在时跳过） |
| `bench/` | Python benchmark 工具，说明见 `bench/README.md`。子模块：`collect` 登记，`label` 标注网页，`synth` 合成数据，`eval` 评测器，`engine` 批量跑 CLI，`competitors/pindo` Pindo baseline，`tools` 诊断工具，`palettes_unverified` bench 专用色卡 |
| `training/ocr/` | OCR 训练（CRNN + CTC，只用合成数据）；`training/data`、`training/runs` 不进 git |
| `palettes/` | 正式色卡，可进 App：12 套已核实来源的国际色卡，**不含 MARD 等国内品牌**。来源登记在 `palettes/SOURCES.md` |
| `tools/` | Windows 编译、WSL 调用、Android 交叉编译和测速 |
| `data/` | `data/xhs` 是需求调研数据，`data/samples` 是 7 张画师图纸样例。两者的图片和正文都不进 git |
| `docs/` | 方案文档 01–07、`phase0-status.md`（给 PO）、本文 |

**不在仓库里、但本机上有的东西**（都在 gitignore 里，能重新生成或下载）：
- `.deps/`：OpenCV、ORT 等依赖，用 `core/scripts/fetch_deps.py` 下载；
- `build/`：编译产物和调试输出；
- `bench/synth/out/`：合成评测集和各次运行的预测结果；
- `training/data`、`training/runs`：OCR 训练数据和 checkpoint；
- `bench/competitors/pindo/.work`、`node_modules`：Pindo 源码和打包产物（GPL，**绝不能进 git**）。

## 4. 怎么编译、怎么跑

本机是 Windows 11 + Git Bash，另有 WSL Ubuntu-24.04。

```bash
tools/build-win.sh
```

上面这条是 Windows 编译（MSVC + Ninja），产物在 `build/win/`。Windows 上的单测：

```bash
build/win/beadcore_tests.exe --gtest_brief=1
```

Linux（WSL）编译和单测。cmake 和 ninja 在 `~/.venvs/pdtools`，编译目录是 `~/pindou-build/linux`：

```bash
bash tools/wsl-run.sh 'export PATH=$HOME/.venvs/pdtools/bin:$PATH; cd ~/pindou-build/linux && cmake --build . --parallel && ./beadcore_tests --gtest_brief=1'
```

bench 测试（120 个）：

```bash
PYTHONUTF8=1 .venv/Scripts/python -m pytest bench/tests -q
```

合成评测集不在 git 里，没有的话按固定 seed 重新生成：

```bash
PYTHONUTF8=1 .venv/Scripts/python -m bench.synth generate --n 60 --out bench/synth/out/eval60 --seed 7 --workers 8 --type T1
```

```bash
PYTHONUTF8=1 .venv/Scripts/python -m bench.synth generate --n 30 --out bench/synth/out/evalT4 --seed 8 --workers 4 --type T4
```

在合成集上跑一次识别（以 eval60 为例）。`--ids` 要列出 manifest 里的全部 image_id：

```bash
D=bench/synth/out/eval60; export PINDOU_DATA_DIR=$D PYTHONUTF8=1; IDS=$(.venv/Scripts/python -c "import json;print(' '.join(json.loads(l)['image_id'] for l in open('$D/manifest.jsonl',encoding='utf-8')))"); .venv/Scripts/python -m bench.engine run --ids $IDS --run myrun --manifest $D/manifest.jsonl --overwrite --jobs 6
```

然后评测这次运行：

```bash
PYTHONUTF8=1 PINDOU_DATA_DIR=bench/synth/out/eval60 .venv/Scripts/python -m bench.eval run --split synth --manifest bench/synth/out/eval60/manifest.jsonl --labels bench/synth/out/eval60/labels --images bench/synth/out/eval60/images --preds bench/synth/out/eval60/preds/myrun
```

**改识别路径后的回归检查**（dev 集到位前的替代办法）：两个合成集都要跑，再逐张对比改前和改后。要求「由对变错」为空，网格都对的图错误格数也不能变多：

```bash
PYTHONUTF8=1 .venv/Scripts/python -m bench.tools.synth_compare --data bench/synth/out/eval60 bench/synth/out/evalT4 --a hdr5 --b myrun
```

`hdr5` 是用 `7ad086e` 的识别代码跑出的结果（之后的提交没改识别），可以直接拿来当对照。诊断工具：

- `bench.tools.page_check`：逐页列出网格少了或多了哪一行、哪一列，在网格哪一侧，拼接偏移对不对；加 `--all` 也检查单页图；
- `bench.tools.partition_check`：只看格子分组对不对，不看色号。

**给人试用**（没有 App，这是桌面上的试用入口）：把图纸图片拖到 `tools/try.cmd` 上，或者运行 `python tools/try_recognize.py 图片... [--hint MARD-221]`。它会跑识别，把结果整理成网页 `build/try/<时间>/index.html` 并打开，内容有网格尺寸、判断的品牌、按识别结果重画的图纸、用量清单（和图例颗数逐项对照）、网格叠加图和警告。

单张图调试用 CLI 的 `--debug`，会输出每页的 lattice、border_map、坐标栏判断依据、坐标读取、B9 版式选择等：

```bash
build/win/beadcore-cli.exe recognize IMG... --debug --models core/models --palettes bench/palettes_unverified/build/palettes_bench.json --hint MARD-221
```

### 4.1 安卓原型（`app/`）

- **编译和安装**：
  1. 在 WSL 里编识别库：`tools/build-android.sh`；
  2. 在 Git Bash 里打包：`tools/build-app.sh`，产物是 `app/build/app/outputs/flutter-apk/app-release.apk`，只含 arm64；
  3. 装机：`adb install -r ...`；
  4. 测 MARD 等国内品牌图纸前：`tools/app-push-palettes.sh`。
- **在电脑上测**：
  - 本机有 Android 16 的 x86_64 模拟器（`Medium_Phone_API_36.1`），可以用 `-no-window` 无窗口启动；
  - **arm64 识别库在模拟器的 ARM 转译下一加载 ONNX Runtime 就崩**（读 CPU 信息时空指针），这只是模拟器的问题；
  - 所以要先 `ABI=x86_64 tools/build-android.sh`，再 `APP_ABIS="arm64-v8a x86_64" tools/build-app.sh` 打一个双架构的测试包；
  - 用 `adb shell input tap` 和 `screencap` 就能走完导入、识别、高亮的全流程。
- **Flutter SDK**：在仓库外的 `D:\AIworkspace\_tools\flutter`（3.47.9），已关闭匿名统计。
- **Gradle 第一次构建**：
  - 要装 NDK 28.2.13676358。新版 cmdline-tools 把 `sdkmanager` 换成了 Android CLI，自动安装会失败，需要手动下载 `android-ndk-r28c-windows.zip`，校验 sha1 后解压到 `Sdk/ndk/28.2.13676358`；
  - 工程在 D 盘、pub 缓存在 C 盘，Kotlin 增量编译会报「Storage ... is already registered」，已在 `app/android/gradle.properties` 里关掉。
- **色卡**：安装包里只有 12 套国际色卡，识别 MARD 图纸时品牌会显示「未确定」，色号写法也可能被改。推送测试色卡后才正常。

## 5. 环境的坑（都踩过）

- **Python**：
  - 用仓库里的 `.venv/Scripts/python`（系统 Python 没有 numpy），而且**一律加 `PYTHONUTF8=1`**，否则 Windows 下按 GBK 读写会出错；
  - Git Bash 的 `/tmp` 对 Windows Python 不可见，调试产物放 `build/dbg/`。
- **MSVC**：
  - 没有全局的 cmake 或编译器，全靠 `tools/msvc.cmd`。这个文件**必须保持 ASCII + CRLF**，否则 cmd.exe 按 GBK 解析会坏；
  - 编译报错是 GBK 编码，`build-win.sh` 已经做了转码。
- **WSL**：一定通过 `tools/wsl-run.sh` 调用，它用了 `--exec` 和 `MSYS_NO_PATHCONV=1`。直接用 `wsl -- bash -lc` 会二次展开 `$PATH`，还会改写 `/mnt/...` 路径。WSL 用户名是 `dministrator`，没有免密 sudo。
- **网络**：
  - git 和 npm 都配了代理 `127.0.0.1:7890`，**这个代理已经失效**；
  - 访问 GitHub 用 curl（直连可用）；
  - npm 加 `--noproxy=* --proxy=null --https-proxy=null`（`bench.competitors.pindo setup --no-proxy` 会自动加）；
  - **不要改用户的全局配置**。Python 的 urllib 直连可用。
- **编译时文件被占用**：`beadcore-cli.exe` 正在被评测调用时重新编译会链接失败，等评测跑完再编译。
- **跑得慢的命令放后台**：两个合成集各跑一遍识别加评测，通常要 2–10 分钟。
- **两个 adb**：PATH 里的是 `C:\soft\platform-tools\adb.exe`，另一个在 Android SDK 里。Flutter 会提示冲突，统一用 SDK 里的那个。
- **ONNX Runtime 的 `Ort::Env` 故意不析构**：退出时 ORT 自己的静态对象可能已先析构，在 Android 的 bionic 上会因为锁了一把已销毁的互斥锁而中止进程（`core/src/nn/model.cpp`）。

## 6. 工作纪律（AGENTS 之外，和 PO 约定过的）

- **代码只在本地提交，永远不要 `git push`**，也不要添加 remote。仓库目前没有远端。到了里程碑就在本地 main 上提交，提交信息用中文，结尾带 `Co-Authored-By` 行。
- **test 集纪律**：
  - 每个 reason 只能用一次：`baseline_w2` 留给 Pindo baseline，我方引擎只在阶段冻结时用 `final_w6` 跑一次；
  - **任何调参都只用 dev 集和合成集**。
- **识别改动必须做回归检查**（见第 4 节）。dev 集到位后改用 dev 门禁：`bench.eval gate`。
- **版权**：
  - `data/samples` 的 7 张是 10-27 从网页抓取、转成 PNG 的，**只能用于内部调试和试走流程，不能登记进正式 benchmark**；
  - 正式 benchmark 的图片必须是在手机 App 里「保存图片」得到的原图，原字节保存；
  - **不写任何抓取脚本**，也不替 PO 上网下载图纸。
- **试走采集或标注流程**：用环境变量 `PINDOU_MANIFEST` 和 `PINDOU_DATA_DIR` 把 manifest 和数据都指到仓库外，免得混进 `bench/datasets/manifest.jsonl`。本机已经备好 `D:\AIworkspace\pindou-trial`：0_2、3_0 两张已登记，没有标注。
- **红线项目**：只能作为外部进程运行，不复制代码。Pindo 就是这样做的：源码在 gitignore 的 `.work/`，仓库里只有自己写的调用代码。
- **文档和实现冲突时以文档为准**。认为文档有错，就先改文档、写明理由，再改代码。
- **交互上**：用户希望用中文、像同事一样讨论，有不同意见直接说并给出理由。PO 该做的事（采集、买会员、真机）不要替 PO 做，列清楚交给 PO。

## 7. 和原方案不一样的地方（都已写进对应文档）

| 事项 | 原方案 | 实际 | 记在哪里 |
|---|---|---|---|
| 格内色号识别 | 单独做一个 CNN 分类器 | 复用同一个 CRNN，在图例色号里做封闭词表打分 | 03 §1、phase0-status T0.11 |
| ONNX Runtime | 官方包 | 官方包默认开启遥测，App 必须用 `--no_telemetry` 编译；core 初始化前也会关掉遥测 | 03 §1、05、06 §7.1 |
| Pindo baseline | 用无头浏览器驱动 | 用 esbuild 打包识别模块，作为 Node 进程运行；和 Chrome 逐格一致 | 04 §4.1、`bench/competitors/pindo/README.md` |
| MARD 等国内品牌色卡 | 用 maxcleme/beadcolors | 来源存疑（和 AGPL 项目 290/291 处相同），只放在 bench 专用的 `palettes_unverified`，不进 `palettes/` | 05 §3、`palettes/SOURCES.md`、phase0-status §4 |
| 04 里有歧义的指标定义 | — | 评测器按 12 条解释实现 | 04 §3.1 |
| 分页拼接（B9） | 按坐标定位 | 坐标可以从 0 编号；读不全坐标时枚举版式打分；单独的图例页识别 | 03「多图输入」 |
| 坐标栏 | 剥掉外圈 | 加了去缝和 OCR 确认连续数字；按坐标定范围时不往已剥掉的一侧扩 | 03「表格式坐标栏」 |
| 合成 T6（拍照）真值 | — | page 多了 `homography` 等字段，只在合成数据里用 | `bench/README.md` |

## 8. 还有哪些问题

### 8.1 最大的不确定性：真实准确率

- 合成集上离 go 门槛还远：网格正确 83%（门槛 98%），图例完全正确 37.5%（门槛 95%）。4 张真实样例全对，但样本太少，说明不了问题。
- OCR 只用合成数据训练过（合成验证集上：色号 99.8%、坐标 87%、格内色号 91%），真实图纸的字体和压缩可能不一样。
- **能不能 go，只有标好真实 dev / test 集、测完竞品才知道。这是整个项目最大的风险。**

### 8.2 暂停、等 dev 集再评估的识别问题

都是合成集上剩下的失败，按约定暂停：

- **格距 ≤ 8 px 的表格式坐标栏**：坐标数字只有三四个像素高，「有字」检测和 OCR 都读不出来。例：eval60 #16、#32、#51。
- **行方向网格线整体偏半格**：例：eval60 #28。
- **圆形豆子的分页样式**：格距 6–11 px 时，格距被检测成一半，或漏掉大段线。例：evalT4 #6、#16、#27、#29。
- **网格外不带框的坐标数字行被算进网格**：例：evalT4 #16、#20、#22。
- **合成图例的文字识别**：这是图例完全正确率只有 37.5% 的主要原因。
- **不确定格的置信度标定**：门槛是标记率上限内召回 ≥ 80%，必须用 dev 集标定。

### 8.3 流程和基础设施

- **没有远端、没有备份**：所有提交只在 D 盘这一份，`D:\AIworkspace\pindou.zip` 是 10-27 的旧版本。PO 说暂时不用备份。
- **CI 从没跑过**，macOS 编译也没验证过：缺远端仓库、self-hosted runner 和私有存储。
- **几个关键数字还没测**：
  - 「58×53 的图 10 分钟标完」没有真人计时：PO 暂缓了试标，试用目录已备好；
  - 「App 保存图 vs 网页图」的分辨率差异，要有配对样本；
  - 中端安卓机上的识别耗时，要有设备。
- `D:\AIworkspace\pindou-data\inbox` 里有一张用户误放的 Google 截图，不是图纸，不要登记。

### 8.4 法务和数据

- MARD、COCO、漫漫、盼盼、咪小窝的色值都没有可确认的来源。合成训练数据用到了这些品牌的色号文字（不含色值），也需要法务确认。详见 phase0-status §4。
- Pindo baseline 里，Perler 两边的编号体系不同（`P01` 对 `80-15xxx`），暂按「不支持的色卡」记为失败。等 dev 集里真有 Perler 图纸再补。

## 9. 接下来做什么

- **dev 集标好后**：按 phase0-status §5 的顺序做。
  1. 跑引擎、出报告；
  2. 标定 θ 和置信度；
  3. 建门禁基线；
  4. 跑 Pindo baseline；
  5. 按真实失败分布修问题；
  6. W2 跑 test 集 baseline。
- **PO 买好竞品会员、装好设备后**：写 T0.9 的 Android adb 自动化脚本，录入 Tier A 的 CSV（格式见 `bench/eval/README.md` §9.1）。
- **有安卓机后**：按 phase0-status §6 跑 T0.14 测速。
- **有远端仓库和 runner 后**：跑通 CI，并打开 `PINDOU_DEV_GATE`。
- **在这之前**：没有不受阻塞的大块工作。不要为了「有事做」去调合成集，也不要开始写 App。
