# 05 开源参考与依赖许可

> 核实日期：2026-10-27。
> 核实方法：读各仓库默认分支 HEAD 上的 LICENSE 原文；对 Zippland/perler-beads、liangdabiao/perler-beads-ai、flutter_blue_plus 额外拉了完整 git 历史，查过许可证变更；模型许可来自 Hugging Face 模型卡的元数据和原文；SDK 条款来自官方文档。
> **本文不是法律意见。** 第 4 节「灰区」里的项目，在使用前要请法务复核。
> **每引入一个新依赖，都要在第 6 节的依赖登记表里登记。**

---

## 1. 依赖白名单（可以直接引入）

| 依赖 | 用途 | 许可 | 版本 / 日期 | 义务与注意事项 |
|---|---|---|---|---|
| [OpenCV](https://github.com/opencv/opencv) 4.x | C++ 核心：预处理、形态学、透视校正、投影 | Apache-2.0（4.5.0 起） | 4.14.0，2026-07-19 | 在 App 内附 NOTICE 和 `3rdparty/` 下各库的许可证；暂时不用 5.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime)（mobile） | **主推理引擎** | MIT | 1.31.1，2026-10-26 | ⚠️ **官方构建默认开启 1DS 遥测**（上报微软、生成设备标识），App 必须用 `--no_telemetry` 的定制构建，见 03 §1。附 `ThirdPartyNotices.txt`。可以按模型裁剪算子做定制构建，官方示例中 arm64 的 .so 从 16.3 MB 降到 4.0 MB。PaddleOCR 3.x 官方的 Android 和 iOS demo 走的就是 ORT + ONNX 这条路线 |
| [ncnn](https://github.com/Tencent/ncnn) | 备选推理引擎（如果需要更小的二进制） | BSD-3-Clause（另附 zlib、BSD-2 授权的第三方代码） | 20261020 版 | 保留版权声明 |
| [MNN](https://github.com/alibaba/MNN) | 备选推理引擎 | Apache-2.0 | 3.6.1 | |
| [PP-OCRv5_mobile_rec / PP-OCRv6 tiny·small](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) 的 ONNX 权重 | **只在训练和冷启动阶段离线使用**（辅助生成标注），**不打包进 App**；App 里用自训的 CRNN（见 03） | Apache-2.0（写在模型卡元数据里，模型仓库没有单独的 LICENSE 文件） | v5 发布于 2025-07；v6 发布于 2026-06 | 在 App 的「开源许可」页注明模型名称和许可证 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 代码 | 参考它的移动端部署 demo（`deploy/ppocr-android`、`deploy/ios_demo`） | Apache-2.0 | v3.7.0 | iOS demo 里内嵌了 Clipper（BSL-1.0），引用时要保留 |
| Apple Vision | 只用于去背景：`VNGenerateForegroundInstanceMaskRequest`，**要求 iOS 17+** | 系统 API | — | **识别主路径不使用平台 OCR**，否则 benchmark 的结果和线上会不一致（见 03 §1） |
| [flutter_reactive_ble](https://github.com/PhilipsHue/flutter_reactive_ble) | **BLE 首选** | BSD-3-Clause | 5.6.0，2026-09 | |
| [universal_ble](https://github.com/Navideck/universal_ble) | BLE 备选 | BSD-3-Clause | 3.1.2，2026-10 | |
| [in_app_purchase](https://github.com/flutter/packages/tree/main/packages/in_app_purchase) | iOS（以及 Google Play）应用内购买 | BSD-3-Clause | 3.3.1 | **不覆盖国内安卓应用商店**，国内安卓的支付方案见 06 |
| [RevenueCat purchases_flutter](https://github.com/RevenueCat/purchases-flutter) | 可选：收据校验 + 数据看板 | SDK 是 MIT；服务按量收费（月追踪收入 $2.5k 以内免费，超出后收 1%） | 10.18.1 | 在国内网络能否访问**未核实**，MVP 阶段先不用 |
| [flutter_unionad](https://github.com/gstory0404/flutter_unionad) | 穿山甲广告（P2） | Apache-2.0 | 3.0.0，支持鸿蒙 | 穿山甲**没有官方 Flutter 插件**，更推荐自己用 platform channel 包一层官方原生 SDK |
| [OKLab 参考实现](https://bottosson.github.io/posts/oklab/)（Björn Ottosson） | OKLab 色彩空间转换 | 公有领域（作者另外给出 MIT 授权） | — | 可以直接使用 |

> Flutter 的常规插件（drift、riverpod、wakelock_plus、pdf 等）在引入时，按第 6 节登记许可证即可，这里不逐一列出。

---

## 2. 算法参考（读思路，自己用 C++ 重新实现）

| 项目 | 许可 | 可以复用的程度 | 参考什么 |
|---|---|---|---|
| [zwhy149/bead-grid-studio](https://github.com/zwhy149/bead-grid-studio)（`packages/core`） | **Apache-2.0** | ✅ 可以移植代码，但要保留 LICENSE 和 NOTICE；**不得使用**它的名称和 logo（见其 TRADEMARKS.md） | **照片转图纸流程的首选参考**：`packages/core/src/{color,palette,quantize,geometry,pattern,analysis}.js`；先用 OKLab 粗筛、再用 CIEDE2000 精排；用 Otsu + 边界连通洪水填充提取背景；对 60 格以内的小线稿，按连通分量归属做骨架保护；控制颜色数量预算；不用随机 k-means，保证输出可复现；palette 和 pattern 都有 JSON Schema。原理文档在 `docs/algorithm.md` |
| [Jett-Wu/Perler_Beads_Generator](https://github.com/Jett-Wu/Perler_Beads_Generator) | **MIT** | ✅ | `src/imageToBeads.ts`：保护特征色、去除孤立斑点（`reduceSpeckles` / `reduceTinyRegions`）、合并相似色、估计背景。它的编辑器交互可以拿来改造成 benchmark 的**真值标注工具**（任务 T0.4） |
| [Retro-Diffusion/pixel-art-fixer](https://github.com/Retro-Diffusion/pixel-art-fixer) | **MIT** | ✅ | **网格 / 像素尺寸估计的最佳参考**：`docs/HOW_IT_WORKS.md` 里融合了多个证据通道（分带自相关 + comb 打分、run-length 软 GCD、自相似），输出亚像素精度的 `step_x/step_y` 和行列数。有 Python 和 Rust 两套实现，方便移植到 C++ |
| [Hugo-Dz/spritefusion-pixel-snapper](https://github.com/Hugo-Dz/spritefusion-pixel-snapper) | MIT | ✅ | 把像素画对齐到网格，再量化到固定调色板 |
| [KennethJAllen/proper-pixel-art](https://github.com/KennethJAllen/proper-pixel-art) | MIT | ✅（只移植算法，不引入它对 FFmpeg 的依赖） | 处理不均匀网格和噪声，还原真实分辨率。可以用来应对「越往上越对不齐」的问题 |
| [xavctn/img2table](https://github.com/xavctn/img2table) | MIT | ✅ | 基于 OpenCV 检测表格线和单元格，可以借鉴到图例表格的解析 |
| [HansBug/pypindou](https://github.com/HansBug/pypindou) | Apache-2.0 | ✅ | Python 实现的量化、Floyd-Steinberg 抖动、邻域清理。**适合作为离线参照**，和我们的 C++ 实现对比结果 |
| [cornelk/beadmachine](https://github.com/cornelk/beadmachine) | MIT | ✅ | 按板子尺寸拆分、生成 HTML 图纸（项目已不活跃） |
| [colour-science/colour](https://github.com/colour-science/colour) | BSD-3-Clause | ✅（只在离线环境用） | 用 `delta_E_CIE2000` 生成 golden 数据，和 C++ 实现对比结果，不打包进 App |
| Sharma 等人的 CIEDE2000 测试数据（[项目页](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/)） | MATLAB 和 Excel 实现只限个人或研究用途；数据文件本身没有声明许可 | ⚠️ | **只在单元测试里用测试数据，不随产品分发；CIEDE2000 按论文公式自己实现，不照抄 .m 或 .xls 文件**。论文：Sharma, Wu, Dalal, *Color Res. Appl.* 30(1):21–30, 2005 |

---

## 3. 色卡数据

| 数据源 | 许可 | 覆盖范围 | 怎么用 |
|---|---|---|---|
| [maxcleme/beadcolors](https://github.com/maxcleme/beadcolors) | MIT | 15 套国际色卡（Hama、Perler、Artkal A/C/M/R/S、Nabbi 等），包含 MARD。zwhy149/bead-grid-studio 用的 MARD 221 也来自这里 | **国际品牌的主数据源**（12 套已引入 `palettes/`）。⚠️ **其中的 MARD 不能用**（T0.2 核实，2026-10-27）：MARD 由 PR #66 从第三方站 pixel-beads.com 抓取，初版 291 个色值里 290 个和 Zippland（AGPL）的 `colorSystemMapping.json` 完全相同；PR #68 改过 156 个，现版本仍有 134 个相同。按本节和第 4 节的标准属于「来源存疑」，只放在 `bench/palettes_unverified/` 供 benchmark 内部使用，详见 `palettes/SOURCES.md` |
| [HansBug/pindou-color-data](https://github.com/HansBug/pindou-color-data) | MIT（文本措辞改成了 "Data"，所以 GitHub 识别为 NOASSERTION） | 11 套国内色卡：MARD 221（两个来源）、MARD 291、优肯 C197 / M221 / 418、COCO、漫漫、盼盼、咪小窝、优肯 174 | ⚠️ **它的上游有一部分是从第三方网站和工具前端交叉整理来的，其中包括 Zippland（AGPL）的 `colorSystemMapping.json`。** 所以必须**逐套核实来源**（任务 T0.2）：只引入能确认「不是从 Zippland 派生」的那几套；来源说不清的，交给法务判断（06 §13）。MARD 两个来源的差异，只用来交叉校验 |

**策略：**
1. **识别不依赖官方色值**：识别时拿格子颜色去和图纸自带图例里渲染出来的色块比对（见 04 第 5 节）。所以色卡数据只用在三个地方：判断品牌、照片转图纸、推荐平替色。
2. 在 App 内注明数据出处；品牌名（MARD、Perler、Hama、Artkal 等）只用来说明「兼容哪个品牌」，不能当作我们自己的品牌来用。
3. **自建色卡要提前到上线前**（原计划 P3）：买实物豆子，在标准光源下拍照或用色度计测量，建立自有色卡数据，摆脱对第三方整理数据的权属依赖。T0.2 核实后，MARD、COCO、漫漫、盼盼、咪小窝**都没有来源可确认的色值**（MARD 见上表；国内其他品牌的上游 get-colors-from-beans 来源不明，漫漫部分派生自 Zippland）。在法务结论或自测数据出来之前，App 里这些品牌只能靠图纸自带的图例识别，照片转图纸（G1）和平替推荐（D2/F5）用不了它们的色值。

---

## 4. 红线与灰区

### 4.1 红线：不能复制任何代码或数据

| 项目 | 原因 |
|---|---|
| LunarXuan/Pindo | GPL-3.0-only。**只能在 `bench/` 里作为外部进程跑 baseline**，其色卡 JSON 的权属也不清楚 |
| Zippland/perler-beads | 当前许可是 AGPL-3.0，作者还在 LICENSE 前附了一段书面声明，明确反对把代码包装成闭源商业产品。这条红线也包括它的 `colorSystemMapping.json` 和 `色号对应表.csv`。2026-03-29 之前的旧版本虽然是 Apache-2.0，法律上可以用，但作者明确反对，**不要用** |
| liangdabiao/perler-beads-ai | 许可是 Apache-2.0（fork 自 Zippland 改许可之前的版本），法律上可以用，但同样有舆论风险，**只读它的思路**。另外它的 AI 优化走的是云端火山引擎，不是端侧方案 |
| tumao7/pindouban（酷猫的智能板协议） | 自定义许可：集成进收费产品必须事先拿到授权，条款可以单方修改，还涉及「酷猫拼豆」商标。**没有书面授权不能接入** |
| DanZai233/PixelBead | CC BY-NC，不能商用 |
| Danamir/pixelart-cleaner | GPL-3.0 |
| libimagequant / imagequant | GPL-3.0-or-later，闭源使用要买商业许可。unfake.js 间接依赖了它，**不要整包引入 unfake.js** |
| **flutter_blue_plus** 2.x（2025-09-21 之后的版本） | 改成了自定义的 FBP License：营利组织在开发和测试阶段使用就要买商业许可，构建时还会上报遥测（包名、App 名、版本）。最后一个 BSD 版本是 1.35.6，但锁定在旧版本就再也拿不到更新。**改用 flutter_reactive_ble** |
| BRIA RMBG-1.4 / RMBG-2.0 | 前者是 source-available 但仅限非商用，后者是 CC BY-NC 4.0；商用都要另签协议 |
| rembg 的默认模型 `bria-rmbg` | 就是 RMBG-2.0。rembg 代码本身是 MIT，但**不能沿用它的默认模型** |
| x1a0q1sx/pindou-studio | GPL-3.0（智能板协议研究，见 03 附录 A.2）。注意：GitHub 上还有两个同名仓库，AngKernel/pindou-studio 是 AGPL，Aswellle/Pindou-Studio 是 MIT。**全文引用开源项目时一律写 `owner/repo`** |
| HEN3DRIK/BeadSorter、Chipdelmal/PixArt-Beads、a31521424/pixel-to-beads、desdic/beads、terremoth/beads-designer | GPL-3.0 |
| AngKernel/pindou-studio、lulu0119/BeadLoom | AGPL-3.0 |
| wuZHeBoy/bead-pattern | 许可证无法识别（NOASSERTION），按红线处理 |
| libheif | LGPL-3.0。HEIC 交给平台层解码，**不要引入 libheif** |
| OpenCV 的 `videoio` 模块 | 会带进 FFmpeg（LGPL）。编译时关闭 videoio、highgui、imgcodecs（03 §1） |

**兜底规则：** `data/github/repos.csv` 里许可证是 GPL、AGPL、NOASSERTION 或空的仓库，除非本节另有说明，**一律视为红线**。

### 4.2 灰区：用之前先请法务评估

| 项目 | 风险 |
|---|---|
| U-2-Net / u2netp 权重（4.7 MB，适合端侧） | 代码是 Apache-2.0，但**权重没有单独声明许可** |
| IS-Net（isnet-general-use）、BiRefNet 部分版本 | 权重标的是 MIT 或 Apache，但训练用的 DIS5K 数据集条款明确禁止商用 |
| SkyTNT/anime-seg 权重（动漫角色抠图） | 权重许可未核实 |
| MODNet | Apache-2.0，可以商用，**但只适合人像** |
| Google ML Kit | 专有但免费。bundled 版不依赖 GMS（只有第三方实测，没有官方声明）；**会向 Google 上报使用指标，必须写进隐私政策**；Subject Segmentation 只能通过 Google Play services 下发模型，在没有 GMS 的国行安卓机上基本用不了，而且还是 Beta。**识别主路径不用它**；它也不在 03 §1 的 SDK 白名单里 |

### 4.3 去背景（G2）的选型结论

| 平台 | 方案 |
|---|---|
| iOS 17 及以上 | Apple Vision（系统 API，没有许可证问题） |
| iOS 17 以下、国行安卓 | 先让法务评估 u2netp，人像场景用 MODNet；**长期方案是用有商用授权的数据自己训练** |
| 有 GMS 的安卓 | 可以用 ML Kit Subject Segmentation，但它是 Beta，没有 SLA 保障，不能作为主方案 |

这个功能是 P1，**不阻塞 MVP**。

---

## 5. 通用义务

- 每个 App 都要有「开源许可」页面：Apache-2.0 组件要附上 LICENSE 和 NOTICE，改过的文件要标注；MIT 和 BSD 组件要保留版权声明；ONNX Runtime 要附 `ThirdPartyNotices.txt`；OpenCV 要附 `3rdparty` 下各库的许可证。
- 用到的模型（PP-OCR 等）在许可页里写明名称和许可证。
- 色卡数据在许可页里注明出处。

---

## 6. 依赖登记表（由实现方持续维护）

| 依赖 | 版本 | 许可 | 用在哪个模块 | 引入日期 | 义务是否已履行 |
|---|---|---|---|---|---|
| OpenCV | 4.14.0 | Apache-2.0 | core（只编 core、imgproc；关闭 videoio、highgui、imgcodecs、IPP） | 2026-10-27 | 待做：App 的「开源许可」页附 NOTICE 和 `3rdparty/` 许可证（阶段 1） |
| GoogleTest | 1.18.0 | BSD-3-Clause | core/tests（不打包） | 2026-10-27 | 不分发，无需履行 |
| nlohmann/json | 3.12.0 | MIT | core | 2026-10-27 | 待做：许可页保留版权声明（阶段 1） |
| stb_image / stb_image_write（nothings/stb @003501f6） | — | MIT 或 Public Domain（取 MIT） | **只在** core/cli（桌面和真机测速 CLI 解码图片），不进 App | 2026-10-27 | CLI 不分发 |
| libwebp | 1.6.0 | BSD-3-Clause | **只在** core/cli（解码 WebP），不进 App | 2026-10-27 | CLI 不分发 |
| ONNX Runtime 官方预编译包（win-x64 / linux-x64 / osx-arm64） | 1.31.1 | MIT | core/src/nn（只用于桌面 benchmark 和 CI，不分发；App 用 `--no_telemetry` 定制构建，阶段 1） | 2026-10-28 | 不分发；App 构建时附 `ThirdPartyNotices.txt` |
| maxcleme/beadcolors 色卡数据 @f97ff428 | — | MIT | palettes（12 套国际色卡，不含 MARD） | 2026-10-27 | 待做：许可页注明数据出处（05 §5） |
| bench 离线依赖：numpy、pillow、pyyaml、scipy、opencv-python-headless、colour-science、pytest | 见 `bench/requirements.txt` | BSD / MIT / HPND / Apache-2.0 | bench/、training/（不打包） | 2026-10-27 | 不分发 |
| wakelock_plus、flutter_local_notifications、drift、riverpod、pdf、sentry_flutter、fluwx（微信支付）、支付宝插件 | 待定 | **引入时读 LICENSE 原文核实** | app | | |
| zwhy149/bead-grid-studio（移植进 `core/src/quant`） | 待定 | Apache-2.0 | core | | 保留 LICENSE 和 NOTICE、标注改动、不使用其名称和 logo |

---

## 7. 训练数据登记表（由实现方维护）

训练数据会影响打包进 App 的模型权重，所以和代码依赖一样要登记许可证。

| 数据来源 | 许可 | 用途（合成底图 / 字体 / 真实样本） | 能否用于训练商用模型 | 登记日期 |
|---|---|---|---|---|
| 程序生成的底图（`bench/synth/basemaps.py`：角色、精灵、几何形状、渐变、噪声等） | 自有 | 合成底图 | ✅ | 2026-10-28 |
| 色值：`palettes/build/palettes.json`（maxcleme/beadcolors，MIT）或随机颜色 | MIT / 自有 | 合成渲染的格子和图例颜色 | ✅ | 2026-10-28 |
| 色号**文字**：MARD、COCO、盼盼、咪小窝的色号表（只用作要渲染的字符串，不用其色值） | 事实性数据，来源见 `palettes/SOURCES.md` | 合成渲染的格内文字、图例文字；OCR 训练的文字内容 | ⚠️ 风险低，法务一并确认（06 §13 第 12 条） | 2026-10-28 |
| 字体（google/fonts @a4c0b21f）：Noto Sans SC、ZCOOL KuaiLe、Inter、Open Sans、Roboto、Arimo、Lato、Barlow Condensed、Roboto Mono、Source Code Pro、Nunito、Varela Round | **SIL OFL 1.1**（允许商用和嵌入），URL 与 sha256 见 `bench/synth/fonts.lock.json` | 合成渲染、OCR 训练 | ✅ | 2026-10-28 |
| OCR 训练样本（`training/ocr/gen.py` 现场渲染 + `training/ocr/synth_pool.py` 从整页合成图切出） | 由以上各项生成 | `core/models/ocr_*.onnx` 的训练数据 | ✅ | 2026-10-28 |
| 小红书真实样本、`data/samples/` | 版权属于画师 | **只用于评测，不用于训练** | ❌ | |
