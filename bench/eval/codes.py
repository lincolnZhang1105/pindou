"""色号比较键：把真值、预测结果、竞品清单里的色号统一成可比较的字符串。

规则（写进 bench/eval/README.md「色号比较」一节）：
- 色号部分一律用 `normalize_code` 规范化（A01 与 A1 相同）。
- 品牌 = palette_id 第一个 `-` 之前的部分，转大写（MARD-221、MARD-291 → MARD；COCO-291 → COCO；HAMA-MIDI → HAMA）。
- 「主品牌」= 真值 `palette_hint` 的品牌。主品牌的色号比较键不带前缀；其他品牌写成 `品牌:色号`。
  - 真值 cells 里没有前缀的色号属于主品牌；写成 `palette_id:code` 的，前缀换成它的品牌（与主品牌相同则去掉前缀）。
  - 预测结果 codebook 的 `palette_id` 同样换成品牌：等于主品牌或缺省时不带前缀，否则带 `品牌:` 前缀。
  - 所以预测成了错误的品牌（比如 MARD 图纸整张认成 COCO），每一格都算色号错误。
- 真值缺 `palette_hint` 时无法判断主品牌：两边都去掉品牌前缀，只比较色号（报告里会给出警告）。
"""
from __future__ import annotations

from bench.common.formats import code_key, normalize_code


def brand_of(palette_id: str | None) -> str | None:
    if palette_id is None:
        return None
    s = str(palette_id).strip().upper()
    if not s:
        return None
    return s.split("-", 1)[0]


class CodeMapper:
    def __init__(self, main_palette: str | None):
        self.main_brand = brand_of(main_palette)
        self._cache: dict[str, str] = {}

    def gt_key(self, key: str | None) -> str | None:
        """输入是 `code_key()` 的结果（可能带 `palette_id:` 前缀），输出比较键。"""
        if key is None:
            return None
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if ":" in key:
            pal, c = key.split(":", 1)
            b = brand_of(pal)
            out = c if (self.main_brand is None or b is None or b == self.main_brand) else f"{b}:{c}"
        else:
            out = key
        self._cache[key] = out
        return out

    def raw_key(self, code: str | None) -> str | None:
        """真值写法的原始色号（`A01`、`COCO-291:A1`）→ 比较键。图例、竞品清单也用这个。"""
        if code is None:
            return None
        return self.gt_key(code_key(str(code)))

    def pred_key(self, palette_id: str | None, code: str) -> str:
        """预测结果 codebook / legend 里的一项 → 比较键。"""
        code = str(code)
        if ":" in code:  # 预测结果自己写了品牌前缀，按真值写法处理
            return self.raw_key(code)  # type: ignore[return-value]
        c = normalize_code(code)
        b = brand_of(palette_id)
        if b is None or self.main_brand is None or b == self.main_brand:
            return c
        return f"{b}:{c}"
