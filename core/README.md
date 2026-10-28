# beadcore：识别与生成核心

C++17，对外只有 `include/beadcore/beadcore.h` 的 C ABI（`docs/03-tech-architecture.md` §3）。
同一份代码跑 App（iOS / Android）和桌面 CLI，所以 **benchmark 的结果就等于线上效果**。

## 目录

| 目录 | 内容 |
|---|---|
| `src/api` | C ABI 实现：异常 → 错误 JSON，绝不返回 NULL |
| `src/engine` | 识别管线编排，组装 RecognitionResult（03 §3.2） |
| `src/grid` | 网格检测（线响应投影 → 格距 → 逐线追踪 → 四边有线的格子连通块 → 坐标栏）、坐标数字读取 |
| `src/legend` | 图例色块、色号和颗数文字框（表格 / 色号在色块里 / 色号在下方 / 色号在右边四种版式） |
| `src/classify` | 逐格采样、颜色分类、白豆和空格、格内文字证据、图例颗数约束 |
| `src/nn` | ONNX Runtime 封装（初始化前关闭遥测）、OCR 预处理和 CTC 解码 |
| `src/color` | sRGB ↔ Lab、OKLab、CIEDE2000 |
| `src/palette` | palettes_json 解析 |
| `cli` | `beadcore-cli`：benchmark 和真机测速用；图片解码只在这里（stb_image + libwebp） |
| `models` | ONNX 模型和元数据；`models.json` 指定当前启用的 OCR 模型 |
| `tests` | GoogleTest：颜色库对拍、C ABI 约定、合成图网格测试、OCR 对拍、真实样例冒烟（样例不在仓库里时跳过） |

## 编译

依赖由脚本下载、校验 sha256（`deps.json`），放在仓库根目录的 `.deps/`：

```bash
python core/scripts/fetch_deps.py --build-opencv
```

Linux / macOS：

```bash
cmake -S core -B build/linux -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build/linux --parallel && ctest --test-dir build/linux --output-on-failure
```

Windows（Git Bash；需要 Visual Studio 2022 或 Build Tools 的 C++ 工具链）：

```bash
tools/build-win.sh
```

产物在 `build/win/`：`beadcore-cli.exe`、`beadcore_tests.exe`（`onnxruntime.dll` 会自动拷过去）。

## CLI

```bash
build/win/beadcore-cli.exe recognize IMG [IMG...] --palettes bench/palettes_unverified/build/palettes_bench.json --models core/models
```

- `--hint MARD-291`：色卡提示；`--pretty`：格式化输出；`--repeat N`：测速。
- `--debug --render PREFIX`：输出 debug 字段，另存叠加了网格线和图例框的调试图。
- 输出总是 JSON；出错时是 `{"error": {"code", "message"}}`，退出码非 0。

批量跑 benchmark 用 `python -m bench.engine run`（见 `bench/README.md`）。

## 注意

- App 构建里的 ONNX Runtime 必须用 `--no_telemetry` 编译（官方包默认开启遥测，见 03 §1）。
- 识别路径的任何改动都要过 dev 集门禁（AGENTS 规则 3）；dev 集没准备好之前，至少跑 `core/tests` 和合成集（`bench/synth`）。
