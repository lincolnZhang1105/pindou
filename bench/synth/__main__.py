"""命令行入口：python -m bench.synth <子命令>

    generate --n N --out DIR --seed S [--style ...] [--size-bucket ...] [--type ...] [--workers K] [--start I]
    sheet    --in DIR [--n 50] [--out FILE] [--overlay]
    fonts    [--update-lock]

详见 bench/synth/README.md。
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

from bench.common.formats import read_jsonl, write_jsonl


def _csv(s: str | None) -> tuple[str, ...] | None:
    if not s:
        return None
    return tuple(x.strip() for x in s.split(",") if x.strip())


def cmd_generate(a) -> int:
    from bench.synth.generate import TYPES, GenOptions, Generator, _init, _work
    from bench.synth.styles import STYLE_NAMES
    types, styles, buckets = _csv(a.type), _csv(a.style), _csv(a.size_bucket)
    for v, ok, name in ((types, TYPES, "--type"), (styles, STYLE_NAMES, "--style"),
                        (buckets, ("S", "M", "L", "XL"), "--size-bucket")):
        bad = [x for x in (v or ()) if x not in ok]
        if bad:
            print(f"{name} 不认识：{bad}（可选 {', '.join(ok)}）", file=sys.stderr)
            return 2
    opts = GenOptions(types=types, styles=styles, buckets=buckets, allow_fallback_font=a.allow_fallback_font,
                      max_side=a.max_side)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.seed, i, str(out)) for i in range(a.start, a.start + a.n)]
    t0 = time.time()
    rows: list[dict] = []
    if a.workers <= 1:
        gen = Generator(opts)
        for k, (s, i, o) in enumerate(jobs):
            rows.extend(gen.generate(s, i, Path(o)))
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(jobs)}  {(k + 1) / (time.time() - t0):.1f} 张/秒", flush=True)
    else:
        with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(opts,)) as pool:
            for k, r in enumerate(pool.imap(_work, jobs, chunksize=2)):
                rows.extend(r)
                if (k + 1) % 50 == 0:
                    print(f"  {k + 1}/{len(jobs)}  {(k + 1) / (time.time() - t0):.1f} 张/秒", flush=True)
    # manifest：与已有记录合并（按 image_id 覆盖），按 image_id 排序
    mpath = out / "manifest.jsonl"
    old = {r["image_id"]: r for r in read_jsonl(mpath)}
    for r in rows:
        old[r["image_id"]] = r
    write_jsonl(mpath, [old[k] for k in sorted(old)])
    dt = time.time() - t0
    print(f"生成 {a.n} 个样本（{len(rows)} 张图片）→ {out}，用时 {dt:.1f}s（{a.n / max(dt, 1e-9):.1f} 样本/秒）")
    return 0


def cmd_sheet(a) -> int:
    from bench.synth.sheet import make_sheet
    p = make_sheet(Path(a.inp), n=a.n, out=Path(a.out) if a.out else None, overlay=a.overlay, offset=a.offset,
                   ncols=a.cols)
    print(f"拼贴图 → {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.synth", description="合成拼豆图纸数据生成器（T0.6）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="生成 N 个样本（图片 + 真值 + 附加标注）")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--start", type=int, default=0, help="起始样本序号（分批生成时用）")
    g.add_argument("--style", help="逗号分隔：table_cyan,fine5,fine10,nocoord,blocks,pixel")
    g.add_argument("--size-bucket", help="逗号分隔：S,M,L,XL")
    g.add_argument("--type", help="逗号分隔：T1..T7")
    g.add_argument("--workers", type=int, default=1, help="并行进程数（输出与进程数无关）")
    g.add_argument("--max-side", type=int, default=None, help="限制图纸边长（调试用）")
    g.add_argument("--allow-fallback-font", action="store_true", help="没下载字体时用 Pillow 内置字体（只用于调试）")
    g.set_defaults(fn=cmd_generate)
    s = sub.add_parser("sheet", help="缩略图拼贴，人工检查风格")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--cols", type=int, default=10)
    s.add_argument("--out")
    s.add_argument("--overlay", action="store_true", help="叠加真值网格线（红色）")
    s.set_defaults(fn=cmd_sheet)
    f = sub.add_parser("fonts", help="按锁文件下载字体")
    f.add_argument("--update-lock", action="store_true")
    f.set_defaults(fn=lambda a: __import__("bench.synth.fonts", fromlist=["main"]).main(
        ["--update-lock"] if a.update_lock else []))
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
