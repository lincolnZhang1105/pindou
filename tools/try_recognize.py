"""试用识别：给一张（或一套分页的几张）图纸，跑 beadcore-cli，把结果整理成一个网页并打开。

    python tools/try_recognize.py 图纸.png [第二页.png ...] [--hint MARD-221] [--no-open]

也可以把图片直接拖到 tools/try.cmd 上。网页写在 build/try/<时间>/index.html，内容：
重画的图纸、网格线叠在原图上的样子、用量清单（图例颗数和格子实际颗数对照）、警告。

这是桌面上的试用入口，不是 App：App 要等阶段 0 判定 go 之后才开始做（AGENTS.md）。
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import time
import webbrowser
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

WARNINGS = {
    "cell_too_small_for_text": "格子太小，格内文字读不出来，只按颜色识别",
    "duplicate_legend_code": "图例里有重复的色号",
    "legend_count_adjusted": "按图例颗数调整了部分格子",
    "pages_order_unverified": "分页图的拼接顺序是按版式猜的，请核对",
    "palette_hint_not_found": "没有找到指定的色卡",
    "text_evidence_changed": "按格内色号文字改判了部分格子",
}
ERRORS = {
    "no_grid_found": "没有找到网格",
    "needs_user_crop": "图里像是有多个图案，需要先框选其中一个",
    "invalid_input": "输入有问题",
    "internal": "识别程序内部出错",
}


def find_cli() -> Path:
    env = os.environ.get("BC_CLI")
    for p in ([Path(env)] if env else []) + [REPO / "build" / "win" / "beadcore-cli.exe", REPO / "build" / "linux" / "beadcore-cli",
                                              REPO / "build" / "macos" / "beadcore-cli"]:
        if p.exists():
            return p
    raise SystemExit("找不到 beadcore-cli，先编译：tools/build-win.sh")


def find_palettes() -> Path:
    for p in (REPO / "bench" / "palettes_unverified" / "build" / "palettes_bench.json", REPO / "palettes" / "build" / "palettes.json"):
        if p.exists():
            return p
    raise SystemExit("找不到色卡文件，先运行 python -m bench.palettes_unverified.import_unverified")


def warning_text(w: str) -> str:
    key, _, n = w.partition(":")
    text = WARNINGS.get(key, w)
    return f"{text}（{n} 格）" if n else text


def page(images: list[Path], res: dict, out: Path, wall_ms: float) -> str:
    e = html.escape
    head = f"<h1>识别结果</h1><p class=meta>{' · '.join(e(p.name) for p in images)}</p>"
    if "error" in res:
        err = res["error"]
        msg = ERRORS.get(err.get("code"), err.get("code"))
        return head + f"<p class=bad>{e(msg)}</p><p class=meta>{e(err.get('message', ''))}</p>"
    grid, book, cells = res["grid"], res["codebook"], res["cells"]
    counts = Counter(c for c in cells if c)
    guess = res.get("palette_guess") or {}
    check = res.get("legend_check") or {}
    legend = {it["code"]: it["count"] for it in res.get("legend", [])}
    if check.get("unavailable") or not legend:
        status = "<span class=warn>没有读到图例，无法核对</span>"
    elif check.get("ok"):
        status = "<span class=good>与图例完全一致</span>"
    else:
        status = f"<span class=bad>有 {len(check.get('diff', []))} 个色号的颗数和图例对不上</span>"
    rows = []
    for k, ent in sorted(enumerate(book, 1), key=lambda kv: -counts.get(kv[0], 0)):
        rgb = ent.get("legend_rgb") or [200, 200, 200]
        n, want = counts.get(k, 0), legend.get(ent["code"])
        mark = "" if want is None or want == n else " class=diff"
        rows.append(f"<tr{mark}><td><span class=sw style='background:rgb({rgb[0]},{rgb[1]},{rgb[2]})'></span>{e(ent['code'])}</td>"
                    f"<td>{n}</td><td>{'' if want is None else want}</td></tr>")
    total = sum(counts.values())
    warns = "".join(f"<li>{e(warning_text(w))}</li>" for w in res.get("warnings", []))
    unc = len(res.get("uncertain", []))
    overlays = "".join(f"<figure><img src='r.overlay{i}.png'><figcaption>第 {i + 1} 张：网格线叠在原图上</figcaption></figure>"
                       for i in range(len(images)) if (out / f"r.overlay{i}.png").exists())
    return head + f"""
<div class=cards>
  <div><b>{grid['cols']} × {grid['rows']}</b><span>网格（列 × 行）</span></div>
  <div><b>{total}</b><span>豆子总数</span></div>
  <div><b>{len(book)}</b><span>色号</span></div>
  <div><b>{e(str(guess.get('palette_id', '—')))}</b><span>判断的品牌</span></div>
  <div><b>{(res.get('timing_ms') or {}).get('total', '—')} ms</b><span>识别耗时（含加载模型、读图共 {round(wall_ms)} ms）</span></div>
</div>
<p>图例核对：{status}　不确定的格子：{unc}</p>
{f'<ul class=warns>{warns}</ul>' if warns else ''}
<div class=row>
  <figure><img src='r.pattern.png'><figcaption>按识别结果重画的图纸</figcaption></figure>
  <table><tr><th>色号</th><th>识别颗数</th><th>图例颗数</th></tr>{''.join(rows)}</table>
</div>
{overlays}"""


STYLE = """body{font:14px/1.6 system-ui,'Microsoft YaHei',sans-serif;margin:24px;color:#222;max-width:1200px}
h1{margin:0 0 4px}.meta{color:#777;margin:0 0 16px}.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px}
.cards div{border:1px solid #ddd;border-radius:8px;padding:8px 14px;min-width:110px}.cards b{display:block;font-size:20px}
.cards span{color:#777;font-size:12px}.good{color:#1a7f37;font-weight:600}.bad{color:#c62828;font-weight:600}
.warn{color:#b26a00;font-weight:600}.warns{color:#b26a00}.row{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
figure{margin:12px 0}img{max-width:100%;border:1px solid #ddd}figcaption{color:#777;font-size:12px}
table{border-collapse:collapse}td,th{border-bottom:1px solid #eee;padding:3px 12px;text-align:right}
td:first-child,th:first-child{text-align:left}tr.diff td{color:#c62828;font-weight:600}
.sw{display:inline-block;width:14px;height:14px;border:1px solid #999;margin-right:6px;vertical-align:-2px}"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path, help="图纸图片；分页图按页码顺序给出几张")
    ap.add_argument("--hint", help="色卡提示，比如 MARD-221（不给时自动判断）")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    a = ap.parse_args(argv)
    images = [p.resolve() for p in a.images]
    for p in images:
        if not p.exists():
            raise SystemExit(f"找不到图片：{p}")
    out = REPO / "build" / "try" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    args = [str(find_cli()), "recognize", "--models", str(REPO / "core" / "models"), "--palettes", str(find_palettes()),
            "--render", str(out / "r")] + (["--hint", a.hint] if a.hint else []) + [str(p) for p in images]
    t0 = time.perf_counter()
    cp = subprocess.run(args, capture_output=True, env=dict(os.environ, ORT_DISABLE_TELEMETRY="1"))
    wall_ms = (time.perf_counter() - t0) * 1000
    try:
        res = json.loads(cp.stdout.decode("utf-8"))
    except json.JSONDecodeError:
        res = {"error": {"code": "internal", "message": cp.stderr.decode("utf-8", "replace")[-500:]}}
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    index = out / "index.html"
    index.write_text(f"<!doctype html><meta charset=utf-8><title>识别结果</title><style>{STYLE}</style>"
                     + page(images, res, out, wall_ms), encoding="utf-8")
    print(f"结果：{index}")
    if not a.no_open:
        webbrowser.open(index.as_uri())
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    sys.exit(main())
