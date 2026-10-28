# 色卡数据来源登记（T0.2）

> 核实日期：2026-10-27。核实人：负责实现的 AI。**本文不是法律意见**，标 ⚠️ 的项需要法务结论（06 §13 第 12 条）。
> 规则（AGENTS 规则 1、05 §3）：`palettes/` 会打包进 App，只收录**来源可确认、许可允许商用、且确认不是派生自 Zippland/perler-beads（AGPL）** 的色卡。

## 1. 结论一览

| 色卡 id | 品牌 | 色数 | 上游 | derived_from_zippland | 放在哪里 |
|---|---|---|---|---|---|
| HAMA-MIDI / HAMA-MINI / HAMA-MAXI | Hama | 92 / 78 / 25 | maxcleme/beadcolors `raw/*.csv` @ `f97ff428` | **false** | `palettes/data/` ✅ |
| PERLER / PERLER-MINI | Perler | 103 / 41 | 同上 | **false** | `palettes/data/` ✅ |
| NABBI | Nabbi | 30 | 同上 | **false** | `palettes/data/` ✅ |
| ARTKAL-A / C / M / R / S | Artkal（优肯） | 145 / 174 / 220 / 89 / 199 | 同上 | **false** | `palettes/data/` ✅ |
| YANT | Yant | 119 | 同上 | **false** | `palettes/data/` ✅ |
| MARD-291 | MARD | 291 | beadcolors `raw/mard.csv`（数据抓自第三方站 pixel-beads.com） | ⚠️ **partial** | 只放 `bench/palettes_unverified/` |
| MARD-221 | MARD | 221 | MARD-291 的子集；221 色号表取自 HansBug `mard-221-github` | ⚠️ **partial** | 只放 `bench/palettes_unverified/` |
| COCO-291 | COCO | 291 | HansBug/pindou-color-data `coco-291` ← get-colors-from-beans | ⚠️ **unknown** | 只放 `bench/palettes_unverified/` |
| PANPAN-289 | 盼盼 | 285（去掉 4 个色号不明的） | HansBug `panpan-289` ← get-colors-from-beans | ⚠️ **unknown** | 只放 `bench/palettes_unverified/` |
| MIXIAOWO-290 | 咪小窝 | 286（同上） | HansBug `mixiaowo-290` ← get-colors-from-beans | ⚠️ **unknown** | 只放 `bench/palettes_unverified/` |
| （不引入）MANMAN-278 | 漫漫 | 278 | HansBug `manman-278` | ❌ **true**（5 个错码按 Zippland 的 `colorSystemMapping.json` 交叉校正） | 不引入，任何目录都不放 |
| （不引入）MARD-221 alfonse 版 | MARD | 221 | HansBug `mard-221-alfonse-doudou` ← alfonse.online、doudougongfang.com（工具站自称官方色卡） | ⚠️ unknown | 不引入（同品牌已有 MARD-221） |
| （不引入）youken-public-174 | 优肯旧表 | 174 | HansBug ← get-colors-from-beans | ⚠️ unknown | 不引入（已有 ARTKAL-*） |
| （可引入，暂未引入）artkal-c-197-official / artkal-m-221-official / 418 合并版 | 优肯 / Artkal | 197 / 221 / 418 | HansBug，数据取自 Artkal 官方 RGB PDF（Shopify CDN），MH1 一项取自 bitbead 工具站 | **false** | 暂不引入：和 beadcolors 的 ARTKAL-C/M 重复。将来要用官方口径时可以引入 |

`palettes/` 目前的产物：12 套色卡、1315 色（`python -m palettes.build_palettes` → `palettes/build/palettes.json`）。

## 2. 关键发现：beadcolors 的 MARD 数据不能当作「干净的 MIT 数据」

05 §3 原先把 maxcleme/beadcolors 定为「MARD 和国际品牌的主数据源」。逐条核对后发现，**MARD 这一套不成立**：

1. **来源**：MARD 由 PR #66 加入（2026-03-08，Asher Billings），说明写的是「Data from this page: https://www.pixel-beads.com/mard-bead-color-chart」，也就是从第三方工具站抓取的数据。beadcolors 仓库的 MIT 许可管不到这个上游。
2. **和 Zippland 的重合**：把色号规范化（`A01` → `A1`）之后，PR #66 的 291 个色值里有 **290 个**和 Zippland/perler-beads（AGPL）`src/app/colorSystemMapping.json` 的 MARD 色值**逐字节相同**，唯一的差别是 R11 的蓝色通道差 1。作为对照，其他公开来源（HansBug 收录的 mard-221-github、mard-291-github、alfonse 版）和 Zippland 只有 1%–6% 相同。这样高的重合率，说明 PR #66 的数据要么派生自 Zippland，要么和 Zippland 来自同一个上游。
3. **之后的修改**：PR #68（2026-08-11）按 pixel-beads.com 的「2026 修订版」改了 156 个色值。当前版本仍有 **134/291** 和 Zippland 相同。
4. 核对脚本只在临时目录读取 Zippland 的文件做比对，没有把它的任何内容放进本仓库（红线项目可以阅读，见 AGENTS 规则 1）。

**处理**：
- MARD 色值标为 `derived_from_zippland: partial`，不进 `palettes/`。只放在 `bench/palettes_unverified/`，供 benchmark 内部判断品牌、给无图例的图纸兜底。
- **合成训练数据不使用这些色值**（05 §7）：合成器只用到 MARD 的色号文字，颜色取自 `palettes/` 里的干净色卡或随机生成。
- 识别本身不依赖官方色值（04 §5.1：格子颜色和图纸自带图例的色块比对），所以对 go/no-go 的影响仅限于无图例的图纸（T3 像素画）。
- **对上线的影响**：MARD 是国内最主流的体系，App 的品牌判断（B8）、照片转图纸（G1）、平替推荐（D2/F5）都需要它。在法务给出结论之前，MARD 在 App 里没有可用的色值来源。建议的出路见第 4 节。

## 3. 国内其他品牌

- COCO、盼盼、咪小窝、MARD 的 HansBug 版本，上游都是 `git.xiongxiao.me/abearxiong/get-colors-from-beans` 的 `get-colors.json`。这个仓库本身没有说明数据从哪来，所以都标 `unknown`。和 Zippland 的色值重合率只有 3% 左右，**不像是**直接复制 Zippland，但不能排除。
- 漫漫（manman-278）：HansBug 用 Zippland 的映射表修正了 5 个错码，属于部分派生，标 `true`，**不引入**。
- 盼盼和咪小窝各有 4 个上游色号为「-」的颜色，导入时直接去掉，不猜测。

## 4. 需要产品负责人和法务处理的事项

1. **法务**：色号和色值这类事实性数据是否受保护；`partial` / `unknown` 的色卡能否进 App（06 §13 第 12 条）。
2. **建议提前启动「自建色卡」**（05 §3 策略 3，原计划在 P3）：购买 MARD 221/291、COCO、漫漫、盼盼、咪小窝的实物色卡或豆子，在标准光源下用色度计（或标准光源 + 色卡校准拍照）测色，得到自有数据。色号表本身是商品编号，风险低；色值由我们自己测，就不再依赖任何第三方整理的数据。预计每个品牌一个工作日左右，外加色度计的采购成本。
3. 在法务结论和自测数据出来之前，App 侧对没有干净色值的品牌：识别照常可用（靠图例），照片转图纸和平替推荐只提供 `palettes/` 里已有的品牌。

## 5. 更新流程

- 改 `palettes/sources.json` → `python -m palettes.import_sources` → `python -m palettes.build_palettes`。
- 新增任何一套色卡，都要先在本文第 1 节登记核实结论；`import_sources.py` 会拒绝 `derived_from_zippland` 不是 `false` 的条目。
