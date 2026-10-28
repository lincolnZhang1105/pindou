# core/tests/data

只用于单元测试，**不随产品分发**。

| 文件 | 来源 | 许可 / 说明 |
|---|---|---|
| `ciede2000_sharma.txt` | Sharma, Wu, Dalal, *Color Res. Appl.* 30(1):21–30, 2005 的配套测试数据：<https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/dataNprograms/ciede2000testdata.txt>（2026-10-27 下载） | 数据文件本身没有声明许可。按 05 §2：只在单元测试里用，不随产品分发；CIEDE2000 按论文公式自己实现，没有参考 .m / .xls。每行 `L1 a1 b1 L2 a2 b2 ΔE00`，共 34 组 |
| `colour_golden.csv` | `bench/tools/gen_colour_golden.py` 用 colour-science（BSD-3）离线生成 | 自有。1 万组固定种子的随机 sRGB 颜色对，记录 Lab（D65）和 ΔE00，用于和 C++ 实现对拍 |
