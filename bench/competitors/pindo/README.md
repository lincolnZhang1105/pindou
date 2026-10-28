# Pindo baseline（T0.8）

把 [LunarXuan/Pindo](https://github.com/LunarXuan/Pindo) 的「识别图纸」功能作为**外部 Node 进程**运行，产出和 beadcore 同格式的预测结果，交给 `bench.eval` 评测。

## 许可证边界（AGENTS 规则 1，docs/05 §4）

Pindo 是 **GPL-3.0**，属于红线项目：只能作为外部进程运行，不能复制进我们的代码。

| 东西 | 在哪里 | 进 git 吗 |
|---|---|---|
| Pindo 源码（按 `pindo.lock.json` 固定的提交下载，校验 sha256） | `.work/src-<commit>/` | 否（gitignore），不修改、不分发 |
| 打包产物（我们的胶水代码和 Pindo 代码合成一个文件） | `.work/bundle-<commit>.mjs`、`-browser.js` | 否，只在本地运行 |
| npm 依赖：esbuild 0.28.2（MIT）、@napi-rs/canvas 1.0.13（MIT） | `node_modules/` | 否 |
| 我们自己写的胶水代码：`entry.mjs`、`browser_entry.mjs`、`job.mjs`、`build.mjs`、`*.py` | 本目录 | 是 |

胶水代码只调用 Pindo 导出的 `recognizePatternFromImage`、`matchColor`、`loadPalette`，不包含它的实现。

## 用法

需要 Node.js ≥ 20。

```bash
python -m bench.competitors.pindo setup --no-proxy
```

`setup` 会下载 Pindo 源码、`npm ci`、打出 Node 版和浏览器版两个包。`--no-proxy` 让 npm 绕过 npm 配置里的代理；本机配置的 127.0.0.1:7890 代理已失效，所以要加这个参数。

```bash
python -m bench.competitors.pindo run --split dev --run pindo-4ae3613 --white-as-empty
```

结果写入 `$PINDOU_DATA_DIR/preds/pindo-4ae3613/`；加了 `--white-as-empty` 时，还会另写一份 `pindo-4ae3613-wae/`。之后照常运行 `python -m bench.eval run --split dev --preds pindo-4ae3613`。跑 test 集要加 `--allow-test`，而且只在 W2 baseline 和 W6 最终评测时各跑一次（04 §6）。

```bash
python -m bench.competitors.pindo check-browser IMG... [--brand mard]
```

`check-browser` 把同样的图片在 Node 和无头 Chrome 里各跑一遍并逐格比对，用来确认 Node 替身层没有引入偏差。

## 跑法的口径

- **参数照 Pindo 网页的默认状态**：品牌按 `palette_hint` 换算（没有 hint 时用默认的 MARD）；整套色卡，不过滤色系；`preferAutoGrid = true`。
- **自动检测不到网格时**，Pindo 会按用户填的宽高均分。`--size` 决定这个宽高从哪来：
  - `manifest`（默认）：登记时人工预估的 `grid`，没有就退回网页默认的 35×35；
  - `default`：一律 35×35；
  - `label`：真值网格。
- **Pindo 不支持的情况直接记为失败**：
  - 多页图纸（B9）→ `unsupported`；
  - 色号写法对不上的色卡 → `unsupported_palette`。PERLER 两边编号体系不同（Pindo 写 `P01`，我们写 `80-15xxx`），按颜色名只能对上 61/70 个，暂不支持。
- **空格**：Pindo 的识别没有空格概念，空白格会被认成白豆。
  - `-wae` 变体把「纯白对应的色号」（由 Pindo 自己的 `matchColor` 算出）当成空格；
  - 代价是真正的白豆会被算成空格。两份结果都报告。
- **图例**：Pindo 不读图例，`legend` 为空数组，所以「图例解析完全正确」一律为否。用量清单误差由评测器从 cells 计算。
- **图片**：原图直接交给 Pindo。带 EXIF 方向的 JPEG 先无损转正成 PNG，和浏览器的行为一致；不缩放、不有损重编码。
- **`--crop label`（颜色路径上限对照）**：
  - 先按真值网格外框、每边向外留 0.25 格裁图，再交给 Pindo，相当于用户手动裁得非常准；
  - 结果**用了真值**，只能作对照，报告里要注明。

## 已验证（2026-10-28）

- **Node 替身层 = 真浏览器**：合成集和测速图共 8 张（webp / jpg / png），在 Node 和 Chrome 156 里网格尺寸、网格来源、每格色号**完全一致**。
- **合成集 eval60**（60 张 T1，其中 21 张的色卡 Pindo 不支持）：
  - 原样运行：网格正确 0/39；
  - `--crop label`：网格正确 5/39。Pindo 总是报告「自动检测成功」，从不退回用户填的宽高；坐标栏、标题、图例和格内文字都会干扰它的网格检测。
  - **合成集评不了 Pindo 的颜色路径**：合成图纸的格子颜色不是真实品牌色值（bench/README「色卡」：合成数据只能借用存疑色卡的色号文字），只按颜色匹配的方法注定全错。颜色路径要等真实 dev 集。
- **本地两张真实样例**（data/samples，只在本地看）：
  - 网格同样不对（16×16 → 43×72，58×53 → 63×72，把图例区域也算了进去）；
  - 图例里的色号基本都出现在 Pindo 的结果里，颜色路径本身是通的。

预期：在真实 dev 集上，Pindo 原样运行的网格正确率会接近 0，作为 baseline 偏弱。这和 04 §4.1 的判断一致：它只针对不带色号的图纸。
