"""T0.8 baseline：把 LunarXuan/Pindo 的识别模块作为外部 Node 进程运行，产出评测器要的预测结果目录。

    python -m bench.competitors.pindo setup [--no-proxy]
    python -m bench.competitors.pindo run --split dev --run pindo-4ae3613 [--white-as-empty] [--size manifest]
    python -m bench.competitors.pindo run --ids ... --run debug --manifest DIR/manifest.jsonl
    python -m bench.competitors.pindo check-browser IMG... [--brand mard]   # 和真浏览器逐格对照

许可证（AGENTS 规则 1，docs/05 §4）：Pindo 是 GPL-3.0。
- 源码按 pindo.lock.json 固定的提交下载到 .work/（gitignore），不进仓库、不修改、不分发；
- 仓库里只有我们自己写的胶水代码（entry.mjs、build.mjs、本目录的 Python），不复制 Pindo 的代码；
- 打包产物 .work/bundle-*.mjs 含 Pindo 代码，只在本地运行。

约定和 bench.engine 相同：输出 `$PINDOU_DATA_DIR/preds/<run>/<单元 key>.json` 和 `_run.json`；
跑 test 集要加 --allow-test，test 集的使用登记由 `bench.eval run --test-reason` 负责。
Pindo 不支持的情况直接记为失败：多页图纸（B9）→ `unsupported`，色卡不在 convert.BRANDS 里 → `unsupported_palette`。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from bench.common import dataset as D
from bench.common import formats as F
from bench.competitors.pindo import convert as C
from bench.engine.__main__ import git_rev

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
LOCK = HERE / "pindo.lock.json"
NPM_NO_PROXY = ["--noproxy=*", "--proxy=null", "--https-proxy=null"]
# --crop label 每边向外多留的格数。2026-10-28 在合成 eval60 上试过 -0.1 / 0 / 0.02 / 0.25 / 0.5 / 1 格，
# Pindo 网格正确分别是 3 / 1 / 3 / 5 / 4 / 1（共 39 张），取最好的 0.25（对我们是更保守的比较）
CROP_PAD = 0.25
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
        ".gif": "image/gif", ".bmp": "image/bmp"}


def load_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def short(lock: dict) -> str:
    return lock["commit"][:7]


def bundle_path(lock: dict) -> Path:
    return WORK / f"bundle-{short(lock)}.mjs"


def browser_bundle_path(lock: dict) -> Path:
    return WORK / f"bundle-{short(lock)}-browser.js"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- setup

def cmd_setup(a: argparse.Namespace) -> int:
    lock = load_lock()
    WORK.mkdir(exist_ok=True)
    tgz = WORK / f"pindo-{short(lock)}.tar.gz"
    if not tgz.exists():
        print(f"[pindo] 下载 {lock['repo']}@{short(lock)}（{lock['license']}）")
        req = urllib.request.Request(lock["url"], headers={"User-Agent": "pindou-bench"})
        part = tgz.with_suffix(".part")
        with urllib.request.urlopen(req, timeout=120) as r, part.open("wb") as f:
            shutil.copyfileobj(r, f)
        part.replace(tgz)
    got = sha256_file(tgz)
    if got != lock["sha256"]:
        print(f"[pindo] 压缩包 sha256 不符：{got} ≠ 锁文件 {lock['sha256']}。删掉 {tgz} 重试；"
              "仍不符说明 GitHub 重新生成了压缩包，人工核对内容后再更新 pindo.lock.json", file=sys.stderr)
        return 1
    src = WORK / f"src-{short(lock)}"
    if not src.exists():
        staging = WORK / f"src-{short(lock)}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        with tarfile.open(tgz) as t:
            t.extractall(staging, filter="data")
        (top,) = list(staging.iterdir())  # codeload 的压缩包只有一个顶层目录
        top.rename(src)
        staging.rmdir()
    node, npm = shutil.which("node"), shutil.which("npm")
    if node is None or npm is None:
        print("[pindo] 找不到 node / npm（需要 Node.js ≥ 20）", file=sys.stderr)
        return 2
    if not (HERE / "node_modules" / "@napi-rs" / "canvas").exists():
        args = [npm, "ci", "--no-audit", "--no-fund"] + (NPM_NO_PROXY if a.no_proxy else [])
        print(f"[pindo] 安装 npm 依赖：{' '.join(args[1:])}")
        subprocess.run(args, cwd=HERE, check=True)
    out = bundle_path(lock)
    subprocess.run([node, str(HERE / "build.mjs"), str(src), str(out)], cwd=HERE, check=True)
    subprocess.run([node, str(HERE / "build.mjs"), str(src), str(browser_bundle_path(lock)), "--browser"], cwd=HERE,
                   check=True)
    print(f"[pindo] 就绪：{out}")
    return 0


# --------------------------------------------------------------------------- run

def prepare_image(path: Path, tmp: Path) -> Path:
    """浏览器画到 canvas 上时会按 EXIF 方向旋转；napi-rs 的解码器不保证这一点，所以有方向标记的 JPEG
    先转正成无损 PNG（像素不缩放、不重新有损编码）。其余图片原样交给 Pindo。"""
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return path
    with Image.open(path) as im:
        if im.getexif().get(0x0112, 1) in (None, 1):
            return path
        out = tmp / (path.stem + ".oriented.png")
        ImageOps.exif_transpose(im).save(out)
        return out


def crop_to_grid(path: Path, label: dict, tmp: Path) -> Path:
    """--crop label：按真值网格的外框、每边向外留 CROP_PAD 格裁图，存成无损 PNG。相当于用户手动裁得非常准，
    只用来衡量 Pindo 颜色路径的上限，报告里要注明用了真值。"""
    page = label["pages"][0]
    xs, ys = sorted(page["x_lines"]), sorted(page["y_lines"])
    px, py = (xs[-1] - xs[0]) / page["cols"] * CROP_PAD, (ys[-1] - ys[0]) / page["rows"] * CROP_PAD
    x0, x1 = math.floor(xs[0] - px), math.ceil(xs[-1] + px)
    y0, y1 = math.floor(ys[0] - py), math.ceil(ys[-1] + py)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        out = tmp / (path.stem + ".grid.png")
        im.crop((max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1))).save(out)
    return out


def run_node(node: str, bundle: Path, jobs: list[dict], timeout: float) -> dict[str, dict]:
    """跑一个 Node 进程，返回 {key: 输出行}。进程崩溃或超时时，已经写出的行照样收下。"""
    if not jobs:
        return {}
    payload = json.dumps({"jobs": jobs}, ensure_ascii=False).encode("utf-8")
    try:
        cp = subprocess.run([node, str(bundle)], input=payload, capture_output=True, timeout=timeout)
        stdout, note = cp.stdout, (f"退出码 {cp.returncode}：{cp.stderr.decode('utf-8', 'replace')[-300:]}"
                                   if cp.returncode else "")
    except subprocess.TimeoutExpired as e:
        stdout, note = e.stdout or b"", f"超时（>{timeout:.0f}s）"
    got: dict[str, dict] = {}
    for line in stdout.decode("utf-8", "replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "key" in row:
            got[row["key"]] = row
    for j in jobs:
        if j["key"] not in got and note:
            got[j["key"]] = {"key": j["key"], "ok": False, "error": f"Node 进程异常结束（{note}）"}
    return got


def cmd_run(a: argparse.Namespace) -> int:
    lock = load_lock()
    bundle = bundle_path(lock)
    node = shutil.which("node")
    if node is None or not bundle.exists():
        print(f"找不到 node 或 {bundle}，先运行 python -m bench.competitors.pindo setup", file=sys.stderr)
        return 2
    if not D.valid_id(a.run):
        print(f"run 名不合法：{a.run}", file=sys.stderr)
        return 2
    root = F.data_dir()
    manifest = F.load_manifest(Path(a.manifest)) if a.manifest else F.load_manifest()
    if a.ids:
        ids = a.ids
    else:
        if a.split == "test" and not a.allow_test:
            print("跑 test 集要加 --allow-test（test 集只在 W2 baseline 和 W6 最终评测时使用）", file=sys.stderr)
            return 2
        ids = F.read_split(a.split)
        if not ids:
            print(f"bench/datasets/{a.split}.txt 为空或不存在", file=sys.stderr)
            return 2
    missing = [i for i in ids if i not in manifest]
    if missing:
        print(f"manifest 里没有这些 image_id：{missing[:10]}", file=sys.stderr)
        return 2
    units = D.build_units(manifest, ids)
    runs = [a.run] + ([f"{a.run}-wae"] if a.white_as_empty else [])
    out_dirs = [D.preds_dir(root) / r for r in runs]
    for d in out_dirs:
        if d.exists() and any(d.glob("*.json")) and not a.overwrite:
            print(f"{d} 已有结果，加 --overwrite 覆盖", file=sys.stderr)
            return 2

    preds: dict[str, list[dict]] = {}   # key → 每个 run 一份
    info: dict[str, dict] = {}          # key → 诊断信息（写进 _run.json）
    jobs: list[dict] = []
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        for key, pages in units.items():
            attrs = D.unit_attrs(pages)
            hint = attrs.get("palette_hint")
            if attrs["paged"]:
                preds[key] = [C.error("unsupported", "Pindo 不支持多页图纸（B9）")] * len(runs)
                continue
            brand = C.pindo_brand(hint)
            if brand is None:
                preds[key] = [C.error("unsupported_palette", f"Pindo 没有和 {hint} 色号写法一致的色卡")] * len(runs)
                continue
            p = D.readable_image_path(pages[0], root)
            if not p.exists():
                preds[key] = [C.error("invalid_input", f"缺少图片 {p}")] * len(runs)
                continue
            p = prepare_image(p, Path(td))
            label = None
            if a.crop == "label" or a.size == "label":
                lp = D.label_path(key, root)
                if not lp.exists():
                    preds[key] = [C.error("invalid_input", f"--crop/--size label 需要真值 {lp}")] * len(runs)
                    continue
                label = F.load_json(lp)
            if a.crop == "label":
                p = crop_to_grid(p, label, Path(td))
            (w, h), size_src = C.size_hint(pages, a.size, label)
            info[key] = {"palette_hint": hint, "brand": brand, "size_hint": [w, h], "size_source": size_src}
            jobs.append({"key": key, "path": str(p), "mime": MIME.get(p.suffix.lower(), "application/octet-stream"),
                         "brand": brand, "width": w, "height": h})

        n = max(1, min(a.jobs, len(jobs)))
        chunks = [jobs[i::n] for i in range(n)]
        with ThreadPoolExecutor(max_workers=n) as ex:
            parts = list(ex.map(lambda ch: run_node(node, bundle, ch, a.timeout * len(ch) + 30), chunks))
        got = {k: v for part in parts for k, v in part.items()}
        # 进程中途崩溃时，后面没跑到的图逐张重跑一次，免得一张坏图连累整批
        for j in jobs:
            if j["key"] not in got:
                got.update(run_node(node, bundle, [j], a.timeout + 30))
    for j in jobs:
        out = got.get(j["key"], {"key": j["key"], "ok": False, "error": "没有输出"})
        hint = info[j["key"]]["palette_hint"]
        preds[j["key"]] = [C.to_prediction(out, hint, wae) for wae in (False, True)[:len(runs)]]
        info[j["key"]].update({"grid_source": out.get("grid_source"), "pindo_ms": out.get("ms")})

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    for key, per_run in preds.items():
        for d, pred in zip(out_dirs, per_run):
            F.save_json(d / f"{key}.json", pred, indent=None)
    failed = {k: v[0]["error"]["code"] for k, v in preds.items() if F.is_error(v[0])}
    node_ver = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    for r, d in zip(runs, out_dirs):
        meta = {
            "run": r,
            "split": a.split if not a.ids else None,
            "git_rev": git_rev(),
            "engine": "pindo",
            "pindo": {"repo": lock["repo"], "commit": lock["commit"], "license": lock["license"],
                      "bundle_sha256": sha256_file(bundle), "node": node_ver},
            "options": {"size": a.size, "crop": a.crop, "white_as_empty": r.endswith("-wae"), "prefer_auto_grid": True,
                        "palette_filter": "none"},
            "units": len(preds),
            "failed": failed,
            "grid_source": dict(Counter(v.get("grid_source") for v in info.values() if v.get("grid_source"))),
            "unit_info": info,
            "wall_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }
        F.save_json(d / "_run.json", meta)
    print(f"{len(preds)} 个单元 → {', '.join(str(d) for d in out_dirs)}；失败 {len(failed)} 个")
    for k, c in sorted(failed.items()):
        print(f"  {k}: {c}")
    return 0


# --------------------------------------------------------------------------- check-browser

CHROME_CANDIDATES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                     r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                     "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                     "google-chrome", "chromium", "chromium-browser"]


def find_chrome(explicit: str | None) -> str | None:
    for c in ([explicit] if explicit else []) + [os.environ.get("CHROME", "")] + CHROME_CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return c if Path(c).exists() else shutil.which(c)
    return None


def run_browser(chrome: str, bundle: Path, jobs: list[dict], timeout: float) -> dict[str, dict]:
    """在无头 Chrome 里跑浏览器版 bundle。图片以 base64 嵌进页面，不需要放开 file:// 的跨源限制。"""
    page_jobs = [dict(j, b64=base64.b64encode(Path(j["path"]).read_bytes()).decode("ascii")) for j in jobs]
    js = bundle.read_text(encoding="utf-8").replace("</script", "<\\/script")
    data = json.dumps(page_jobs).replace("</", "<\\/")
    page = WORK / "check" / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text('<!doctype html><html><head><meta charset="utf-8"><title>running</title></head><body>'
                    f'<pre id="out"></pre><script id="jobs" type="application/json">{data}</script>'
                    f"<script>{js}</script></body></html>", encoding="utf-8")
    args = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--disable-background-networking", "--disable-extensions", "--disable-component-update",
            f"--user-data-dir={WORK / 'check' / 'profile'}", f"--virtual-time-budget={int(timeout * 1000)}",
            "--dump-dom", page.resolve().as_uri()]
    cp = subprocess.run(args, capture_output=True, timeout=timeout + 60)
    m = re.search(r'<pre id="out">(.*?)</pre>', cp.stdout.decode("utf-8", "replace"), re.S)
    if not m or not m.group(1).strip():
        raise RuntimeError(f"浏览器没有输出结果（退出码 {cp.returncode}）：{cp.stderr.decode('utf-8', 'replace')[-300:]}")
    rows = json.loads(html.unescape(m.group(1)))
    if isinstance(rows, dict):
        raise RuntimeError(f"浏览器端出错：{rows}")
    return {r["key"]: r for r in rows}


def cmd_check_browser(a: argparse.Namespace) -> int:
    lock = load_lock()
    node, chrome = shutil.which("node"), find_chrome(a.chrome)
    bundle, bbundle = bundle_path(lock), browser_bundle_path(lock)
    if node is None or not bundle.exists() or not bbundle.exists():
        print("先运行 python -m bench.competitors.pindo setup", file=sys.stderr)
        return 2
    if chrome is None:
        print("找不到 Chrome / Edge，用 --chrome 或环境变量 CHROME 指定", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as td:
        jobs = []
        for i, p in enumerate(Path(x) for x in a.images):
            q = prepare_image(p, Path(td))
            jobs.append({"key": p.name, "path": str(q), "mime": MIME.get(q.suffix.lower(), "application/octet-stream"),
                         "brand": a.brand, "width": a.width, "height": a.height})
        n_out = run_node(node, bundle, jobs, a.timeout * len(jobs) + 30)
        b_out = run_browser(chrome, bbundle, jobs, a.timeout * len(jobs))
    same = 0
    print(f"浏览器：{chrome}\n| 图片 | Node（网格 / 来源） | 浏览器（网格 / 来源） | 色号不同的格子 |\n|---|---|---|---|")
    for j in jobs:
        n, b = n_out.get(j["key"], {}), b_out.get(j["key"], {})
        desc = lambda o: f"{o.get('cols')}×{o.get('rows')} / {o.get('grid_source')}" if o.get("ok") \
            else f"失败：{o.get('error')}"  # noqa: E731
        if n.get("ok") and b.get("ok") and (n["cols"], n["rows"]) == (b["cols"], b["rows"]):
            diff = sum(x != y for x, y in zip(n["codes"], b["codes"]))
        else:
            diff = None
        ok = diff == 0 and n.get("grid_source") == b.get("grid_source")
        same += ok
        print(f"| {j['key']} | {desc(n)} | {desc(b)} | {'—' if diff is None else diff} |")
    print(f"\n{same}/{len(jobs)} 张完全一致")
    return 0 if same == len(jobs) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.competitors.pindo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="下载固定版本的 Pindo 源码、安装 npm 依赖、打包")
    s.add_argument("--no-proxy", action="store_true", help="npm 安装时绕过 npm 配置里的代理（本机代理失效时用）")
    r = sub.add_parser("run", help="批量识别，写 preds/<run>/")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=["dev", "test"])
    g.add_argument("--ids", nargs="+")
    r.add_argument("--run", required=True)
    r.add_argument("--manifest", help="默认 bench/datasets/manifest.jsonl；合成集传 DIR/manifest.jsonl")
    r.add_argument("--size", choices=["manifest", "default", "label"], default="manifest",
                   help="自动检测不到网格时给 Pindo 的宽高：manifest 里人工预估的 grid（默认）、Pindo 网页默认的 35×35，"
                        "或真值网格（只用于颜色路径上限对照）")
    r.add_argument("--crop", choices=["none", "label"], default="none",
                   help="label：先按真值网格外框裁图再交给 Pindo（颜色路径上限对照，结论里要注明用了真值）")
    r.add_argument("--white-as-empty", action="store_true", help="另写一份 <run>-wae：纯白对应的色号当成空格")
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--timeout", type=float, default=120, help="每张图的超时（秒）")
    r.add_argument("--allow-test", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("check-browser", help="同样的图片在 Node 和无头 Chrome 里各跑一遍，逐格对照（验证替身层）")
    c.add_argument("images", nargs="+")
    c.add_argument("--brand", default=C.DEFAULT_BRAND)
    c.add_argument("--width", type=int, default=C.DEFAULT_SIZE[0])
    c.add_argument("--height", type=int, default=C.DEFAULT_SIZE[1])
    c.add_argument("--chrome", help="Chrome / Edge 可执行文件（默认自动查找，或环境变量 CHROME）")
    c.add_argument("--timeout", type=float, default=60, help="每张图的超时（秒）")
    a = ap.parse_args(argv)
    return {"setup": cmd_setup, "run": cmd_run, "check-browser": cmd_check_browser}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
